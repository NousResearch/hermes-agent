export const LUNAR_CITY_PERF_BRIDGE_VERSION = 1
export const LUNAR_CITY_PERF_FLAG = 'HERMES_LUNAR_CITY_PERF_ACCEPTANCE'
export const LUNAR_CITY_PERF_NONCE = 'HERMES_LUNAR_CITY_PERF_NONCE'
export const LUNAR_CITY_PROCESS_METRICS_SOURCE = 'electron.app.getAppMetrics' as const
export const LUNAR_CITY_GPU_MEMORY_SOURCE = 'chromium-memory-infra-v1' as const

const EXACT_SHA = /^[0-9a-f]{40}$/iu
const CANONICAL_TIMESTAMP = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$/u
const SAFE_NONCE = /^[a-z0-9._-]{16,256}$/iu

export interface LunarCityPerfBuildStamp {
  builtAt: string
  commit: string
  dirty: false
  schemaVersion: 1
  source: 'ci' | 'local'
}

export interface LunarCityPerfLaunch {
  buildStamp: LunarCityPerfBuildStamp
  launchNonce: string
}

function isCleanBuildStamp(value: unknown): value is LunarCityPerfBuildStamp {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return false
  }

  const stamp = value as Partial<LunarCityPerfBuildStamp>

  return (
    stamp.schemaVersion === 1 &&
    stamp.dirty === false &&
    (stamp.source === 'local' || stamp.source === 'ci') &&
    typeof stamp.commit === 'string' &&
    EXACT_SHA.test(stamp.commit) &&
    !/^0{40}$/u.test(stamp.commit) &&
    typeof stamp.builtAt === 'string' &&
    CANONICAL_TIMESTAMP.test(stamp.builtAt)
  )
}

export function resolveLunarCityPerfLaunch(input: {
  buildStamp: unknown
  env: Readonly<Record<string, string | undefined>>
  isPackaged: boolean
}): LunarCityPerfLaunch | undefined {
  const nonce = input.env[LUNAR_CITY_PERF_NONCE]

  if (
    !input.isPackaged ||
    input.env[LUNAR_CITY_PERF_FLAG] !== '1' ||
    typeof nonce !== 'string' ||
    !SAFE_NONCE.test(nonce) ||
    !isCleanBuildStamp(input.buildStamp)
  ) {
    return undefined
  }

  return { buildStamp: input.buildStamp, launchNonce: nonce }
}

export function buildLunarCityPerfHandshake(
  input: LunarCityPerfLaunch & {
    mainPid: number
    rendererPid: number
    rendererStartedAtMs: number
  }
) {
  return {
    bridgeVersion: LUNAR_CITY_PERF_BRIDGE_VERSION,
    buildSha: input.buildStamp.commit,
    buildStamp: input.buildStamp,
    launchNonce: input.launchNonce,
    mainPid: input.mainPid,
    packaged: true,
    processMetricsSource: LUNAR_CITY_PROCESS_METRICS_SOURCE,
    rendererIdentity: { pid: input.rendererPid, startedAtMs: input.rendererStartedAtMs },
    supportedPhases: ['baseline-shell', 'mounted-city'] as const
  }
}

interface NativeProcessMetric {
  cpu?: { percentCPUUsage?: unknown }
  memory?: { workingSetSize?: unknown }
  pid?: unknown
  type?: unknown
  [key: string]: unknown
}

export function sanitizeAppMetrics(rows: readonly unknown[]) {
  return rows.flatMap(value => {
    const row = value as NativeProcessMetric

    if (
      !Number.isInteger(row.pid) ||
      (row.pid as number) <= 0 ||
      typeof row.type !== 'string' ||
      typeof row.cpu?.percentCPUUsage !== 'number' ||
      !Number.isFinite(row.cpu.percentCPUUsage) ||
      row.cpu.percentCPUUsage < 0 ||
      typeof row.memory?.workingSetSize !== 'number' ||
      !Number.isFinite(row.memory.workingSetSize) ||
      row.memory.workingSetSize < 0
    ) {
      return []
    }

    return [
      {
        cpu: { percentCPUUsage: row.cpu.percentCPUUsage },
        memory: { workingSetSize: row.memory.workingSetSize },
        pid: row.pid as number,
        type: row.type
      }
    ]
  })
}

interface TraceEvent {
  args?: {
    dumps?: {
      allocators?: Record<string, { attrs?: { effective_size?: { units?: unknown; value?: unknown } } }>
    }
  }
  cat?: unknown
  id?: unknown
  ph?: unknown
  pid?: unknown
  ts?: unknown
}

function parseHexBytes(value: unknown): number | undefined {
  if (typeof value !== 'string' || !/^[0-9a-f]+$/iu.test(value)) {
    return undefined
  }

  const bytes = Number.parseInt(value, 16)

  return Number.isSafeInteger(bytes) && bytes >= 0 ? bytes : undefined
}

/**
 * Parse only explicitly GPU-attributed Chromium memory-infra allocator dumps.
 * RSS, manifest estimates, and process totals are intentionally not accepted.
 */
export function parseChromiumMemoryInfraGpuAllocation(trace: { traceEvents?: readonly TraceEvent[] }): {
  gpuMemoryMiB: number | null
  gpuMemorySource: typeof LUNAR_CITY_GPU_MEMORY_SOURCE | 'unavailable'
} {
  const unavailable = { gpuMemoryMiB: null, gpuMemorySource: 'unavailable' as const }
  const groups = new Map<string, { allocators: Map<string, unknown>; lastTimestamp: number }>()
  const processIds = new Set<number>()
  let previousTimestamp = Number.NEGATIVE_INFINITY

  for (const event of trace.traceEvents ?? []) {
    if (
      event.ph !== 'v' ||
      typeof event.cat !== 'string' ||
      !event.cat.split(',').includes('disabled-by-default-memory-infra')
    ) {
      continue
    }

    const allocators = event.args?.dumps?.allocators

    if (!allocators) {
      continue
    }

    const gpuAllocators = Object.entries(allocators).filter(([name]) => name === 'gpu' || name.startsWith('gpu/'))

    if (gpuAllocators.length === 0) {
      continue
    }

    if (
      (typeof event.id !== 'string' && typeof event.id !== 'number') ||
      !Number.isInteger(event.pid) ||
      (event.pid as number) <= 0 ||
      typeof event.ts !== 'number' ||
      !Number.isFinite(event.ts) ||
      event.ts < previousTimestamp
    ) {
      return unavailable
    }

    previousTimestamp = event.ts
    processIds.add(event.pid as number)

    if (processIds.size > 1) {
      return unavailable
    }

    const key = `${String(event.id)}:${event.pid}`
    const group = groups.get(key) ?? { allocators: new Map<string, unknown>(), lastTimestamp: event.ts }

    for (const [name, allocator] of gpuAllocators) {
      const existing = group.allocators.get(name) as
        { attrs?: { effective_size?: { units?: unknown; value?: unknown } } } | undefined

      const next = allocator.attrs?.effective_size

      if (existing) {
        const current = existing.attrs?.effective_size

        if (current?.units !== next?.units || current?.value !== next?.value) {
          return unavailable
        }

        continue
      }

      group.allocators.set(name, allocator)
    }

    group.lastTimestamp = event.ts
    groups.set(key, group)
  }

  if (groups.size === 0) {
    return unavailable
  }

  const newestTimestamp = Math.max(...[...groups.values()].map(group => group.lastTimestamp))
  const newestGroups = [...groups.values()].filter(group => group.lastTimestamp === newestTimestamp)

  if (newestGroups.length !== 1) {
    return unavailable
  }

  const selectedNames: string[] = []
  let bytes = 0

  for (const [name, rawAllocator] of [...newestGroups[0]!.allocators.entries()].sort(
    ([left], [right]) => left.split('/').length - right.split('/').length || left.localeCompare(right)
  )) {
    const allocator = rawAllocator as { attrs?: { effective_size?: { units?: unknown; value?: unknown } } }
    const size = allocator.attrs?.effective_size
    const parsed = size?.units === 'bytes' ? parseHexBytes(size.value) : undefined

    if (parsed === undefined) {
      return unavailable
    }

    if (selectedNames.some(parent => name.startsWith(`${parent}/`))) {
      continue
    }

    selectedNames.push(name)
    bytes += parsed
  }

  return selectedNames.length === 0
    ? unavailable
    : { gpuMemoryMiB: bytes / 1024 / 1024, gpuMemorySource: LUNAR_CITY_GPU_MEMORY_SOURCE }
}

export interface ChromiumMemoryInfraCaptureConfig {
  category: 'disabled-by-default-memory-infra'
  dumpMode: 'detailed'
  periodicIntervalMs: 250
}

export function createChromiumMemoryInfraGpuProbe(
  capture: (config: ChromiumMemoryInfraCaptureConfig) => Promise<{ traceEvents?: readonly TraceEvent[] }>
) {
  return async (): Promise<GpuSnapshot> => {
    try {
      const trace = await capture({
        category: 'disabled-by-default-memory-infra',
        dumpMode: 'detailed',
        periodicIntervalMs: 250
      })

      return parseChromiumMemoryInfraGpuAllocation(trace)
    } catch {
      return { gpuMemoryMiB: null, gpuMemorySource: 'unavailable' }
    }
  }
}

interface PerfSender {
  getOSProcessId(): number
  id: number
  isDestroyed(): boolean
  send(channel: string, payload: unknown): void
}

interface PerfEvent {
  frameId: number
  sender: PerfSender
}

type GpuSnapshot = ReturnType<typeof parseChromiumMemoryInfraGpuAllocation>

export interface LunarCityPerfMainControllerOptions {
  appMetrics: () => readonly unknown[]
  gpuSnapshot: () => Promise<GpuSnapshot>
  environmentSnapshot?: (sender: PerfSender) => Promise<Record<string, unknown>>
  launch: LunarCityPerfLaunch
  mainPid: number
  now: () => number
  ownsSender: (sender: PerfSender) => boolean
  requestTimeoutMs?: number
  scenarioWindowAction?: (
    sender: PerfSender,
    action: 'window-hidden' | 'window-minimized' | 'window-visible-cycle'
  ) => Promise<Record<string, unknown>>
}

interface PendingRequest {
  action: string
  identity: RendererRequestIdentity
  reject: (reason: Error) => void
  resolve: (value: unknown) => void
  timer: ReturnType<typeof setTimeout>
}

interface RendererLifetime {
  generation: number
  pid: number
  startedAtMs: number
}

interface RendererRequestIdentity {
  bridgeVersion: typeof LUNAR_CITY_PERF_BRIDGE_VERSION
  buildSha: string
  frameId: number
  launchNonce: string
  mainPid: number
  rendererGeneration: number
  rendererPid: number
  rendererStartedAtMs: number
  senderId: number
}

interface RendererSession {
  active: boolean
  frameId: number
  handshake: ReturnType<typeof buildLunarCityPerfHandshake>
  identity: RendererRequestIdentity
  responderRegistered: boolean
  senderId: number
}

function sameHandshake(left: unknown, right: ReturnType<typeof buildLunarCityPerfHandshake>): boolean {
  if (!left || typeof left !== 'object' || Array.isArray(left)) {
    return false
  }

  const value = left as Record<string, unknown>

  return (
    value.bridgeVersion === right.bridgeVersion &&
    value.buildSha === right.buildSha &&
    value.launchNonce === right.launchNonce &&
    value.mainPid === right.mainPid &&
    value.packaged === true &&
    value.processMetricsSource === right.processMetricsSource &&
    (value.rendererIdentity as Record<string, unknown> | undefined)?.pid === right.rendererIdentity.pid &&
    (value.rendererIdentity as Record<string, unknown> | undefined)?.startedAtMs === right.rendererIdentity.startedAtMs
  )
}

function sameIdentity(left: unknown, right: RendererRequestIdentity): boolean {
  if (!left || typeof left !== 'object' || Array.isArray(left)) {
    return false
  }

  const value = left as Record<string, unknown>

  return Object.entries(right).every(([key, expected]) => value[key] === expected)
}

/**
 * Main-process authority for the acceptance bridge. Every operation is bound
 * to the one launched BrowserWindow sender and its renderer lifetime.
 */
export function createLunarCityPerfMainController(options: LunarCityPerfMainControllerOptions) {
  const pending = new Map<string, PendingRequest>()
  const rendererLifetimes = new Map<number, RendererLifetime>()
  const requestTimeoutMs = options.requestTimeoutMs ?? 5_000
  let requestSequence = 0
  let rendererGeneration = 0
  let session: RendererSession | undefined
  let disposed = false

  const lifetimeFor = (sender: PerfSender) => {
    if (!options.ownsSender(sender) || sender.isDestroyed()) {
      return undefined
    }

    const pid = sender.getOSProcessId()

    if (!Number.isInteger(pid) || pid <= 0) {
      return undefined
    }

    const existing = rendererLifetimes.get(sender.id)

    if (existing?.pid === pid) {
      return existing
    }

    const created = { generation: ++rendererGeneration, pid, startedAtMs: options.now() }
    rendererLifetimes.set(sender.id, created)

    return created
  }

  const bootstrap = (event: PerfEvent) => {
    if (disposed) {
      return undefined
    }

    const renderer = lifetimeFor(event.sender)

    return renderer && Number.isInteger(event.frameId) && event.frameId >= 0
      ? buildLunarCityPerfHandshake({
          ...options.launch,
          mainPid: options.mainPid,
          rendererPid: renderer.pid,
          rendererStartedAtMs: renderer.startedAtMs
        })
      : undefined
  }

  const sessionFor = (event: PerfEvent, requireActive = true): RendererSession | undefined => {
    const current = session
    const lifetime = lifetimeFor(event.sender)

    if (
      !current ||
      (requireActive && !current.active) ||
      !lifetime ||
      current.senderId !== event.sender.id ||
      current.frameId !== event.frameId ||
      current.identity.rendererPid !== lifetime.pid ||
      current.identity.rendererStartedAtMs !== lifetime.startedAtMs ||
      current.identity.rendererGeneration !== lifetime.generation
    ) {
      return undefined
    }

    return current
  }

  const registerResponder = (event: PerfEvent, handshake: unknown): boolean => {
    if (disposed || session) {
      return false
    }

    const lifetime = lifetimeFor(event.sender)
    const exact = bootstrap(event)

    if (!lifetime || !exact || !sameHandshake(handshake, exact)) {
      return false
    }

    session = {
      active: false,
      frameId: event.frameId,
      handshake: exact,
      identity: {
        bridgeVersion: LUNAR_CITY_PERF_BRIDGE_VERSION,
        buildSha: options.launch.buildStamp.commit,
        frameId: event.frameId,
        launchNonce: options.launch.launchNonce,
        mainPid: options.mainPid,
        rendererGeneration: lifetime.generation,
        rendererPid: lifetime.pid,
        rendererStartedAtMs: lifetime.startedAtMs,
        senderId: event.sender.id
      },
      responderRegistered: true,
      senderId: event.sender.id
    }

    return true
  }

  const activate = (event: PerfEvent, handshake: unknown): boolean => {
    const current = sessionFor(event, false)

    if (!current?.responderRegistered || current.active || !sameHandshake(handshake, current.handshake)) {
      return false
    }

    current.active = true

    return true
  }

  const requestRenderer = (event: PerfEvent, action: string, payload?: unknown): Promise<unknown> => {
    if (disposed || typeof action !== 'string' || action.length === 0) {
      return Promise.resolve(undefined)
    }

    const current = sessionFor(event)

    if (!current) {
      return Promise.resolve(undefined)
    }

    const requestId = [
      'lcperf-v1',
      current.identity.buildSha,
      current.identity.launchNonce,
      current.identity.mainPid,
      current.identity.senderId,
      current.identity.frameId,
      current.identity.rendererPid,
      current.identity.rendererGeneration,
      current.identity.rendererStartedAtMs,
      ++requestSequence
    ].join(':')

    const bindScenarioResult = (result: unknown) => {
      if (action !== 'scenario-action' || !result || typeof result !== 'object' || Array.isArray(result)) {
        return result
      }

      return {
        ...result,
        bridgeBinding: {
          action,
          identity: { ...current.identity },
          payload: structuredClone(payload),
          requestId
        }
      }
    }

    if (
      action === 'scenario-action' &&
      payload &&
      typeof payload === 'object' &&
      !Array.isArray(payload) &&
      ((payload as { action?: unknown }).action === 'window-hidden' ||
        (payload as { action?: unknown }).action === 'window-minimized' ||
        (payload as { action?: unknown }).action === 'window-visible-cycle')
    ) {
      const windowAction = (payload as { action: 'window-hidden' | 'window-minimized' | 'window-visible-cycle' }).action

      return options.scenarioWindowAction
        ? options.scenarioWindowAction(event.sender, windowAction).then(bindScenarioResult)
        : Promise.reject(new Error(`Lunar City performance window action is unavailable: ${windowAction}`))
    }

    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        pending.delete(requestId)
        reject(new Error(`Lunar City performance renderer request timed out: ${action}`))
      }, requestTimeoutMs)

      pending.set(requestId, { action, identity: current.identity, reject, resolve, timer })
      event.sender.send('hermes:lunar-city-perf:request', {
        action,
        identity: current.identity,
        payload,
        requestId
      })
    }).then(async result => {
      if (action !== 'snapshot' || !result || typeof result !== 'object' || Array.isArray(result)) {
        return bindScenarioResult(result)
      }

      const metrics = result as Record<string, unknown>

      if (
        metrics.rendererPid !== current.identity.rendererPid ||
        metrics.rendererStartedAtMs !== current.identity.rendererStartedAtMs ||
        metrics.rendererGeneration !== current.identity.rendererGeneration
      ) {
        throw new Error('Lunar City performance renderer lifetime changed')
      }

      const gpu = await options.gpuSnapshot()
      const environment = await options.environmentSnapshot?.(event.sender)

      return {
        ...metrics,
        ...gpu,
        ...(environment
          ? {
              environment: {
                ...(metrics.environment && typeof metrics.environment === 'object' ? metrics.environment : {}),
                ...environment
              }
            }
          : {})
      }
    })
  }

  const resolveRendererResponse = (event: PerfEvent, response: unknown): boolean => {
    if (disposed || !response || typeof response !== 'object' || Array.isArray(response)) {
      return false
    }

    const envelope = response as { action?: unknown; identity?: unknown; requestId?: unknown; value?: unknown }

    if (typeof envelope.requestId !== 'string' || typeof envelope.action !== 'string') {
      return false
    }

    const request = pending.get(envelope.requestId)
    const current = sessionFor(event)

    if (
      !request ||
      !current ||
      request.action !== envelope.action ||
      !sameIdentity(envelope.identity, request.identity) ||
      !sameIdentity(current.identity, request.identity)
    ) {
      return false
    }

    clearTimeout(request.timer)
    pending.delete(envelope.requestId)
    request.resolve(envelope.value)

    return true
  }

  return {
    activate,
    bootstrap,
    dispose(): void {
      if (disposed) {
        return
      }

      disposed = true

      for (const request of pending.values()) {
        clearTimeout(request.timer)
        request.reject(new Error('Lunar City performance bridge disposed'))
      }

      pending.clear()
      rendererLifetimes.clear()
    },
    invalidateRenderer(sender: PerfSender, reason: string): void {
      if (!session || session.senderId !== sender.id) {
        return
      }

      session = undefined
      rendererLifetimes.delete(sender.id)

      for (const request of pending.values()) {
        clearTimeout(request.timer)
        request.reject(new Error(`Lunar City performance renderer invalidated: ${reason}`))
      }

      pending.clear()
    },
    processMetrics(event: PerfEvent) {
      return !disposed && sessionFor(event) ? sanitizeAppMetrics(options.appMetrics()) : undefined
    },
    registerResponder,
    requestRenderer,
    resolveRendererResponse
  }
}
