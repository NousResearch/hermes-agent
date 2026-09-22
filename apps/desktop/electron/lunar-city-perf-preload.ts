interface IpcRendererLike {
  invoke(channel: string, ...values: unknown[]): Promise<unknown>
  on(channel: string, listener: (...values: unknown[]) => void): void
  removeListener(channel: string, listener: (...values: unknown[]) => void): void
  send(channel: string, ...values: unknown[]): void
  sendSync(channel: string, ...values: unknown[]): unknown
}

export interface LunarCityPerfRuntimePort {
  close?(): void
  onmessage: ((event: { data: unknown }) => void) | null
  postMessage(value: unknown): void
  start?(): void
}

export interface LunarCityPerfHandshake {
  bridgeVersion: 1
  buildSha: string
  buildStamp: {
    builtAt: string
    commit: string
    dirty: false
    schemaVersion: 1
    source: 'ci' | 'local'
  }
  launchNonce: string
  mainPid: number
  packaged: true
  processMetricsSource: 'electron.app.getAppMetrics'
  rendererIdentity: { pid: number; startedAtMs: number }
  supportedPhases: readonly ['baseline-shell', 'mounted-city']
}

export type LunarCityPerfScenarioAction =
  | 'context-loss-restore'
  | 'dispose'
  | 'focus'
  | 'interior'
  | 'leader-dialogue'
  | 'orbit'
  | 'quality'
  | 'window-hidden'
  | 'window-minimized'
  | 'window-visible-cycle'
  | 'zoom'

const SCENARIO_ACTIONS = new Set<LunarCityPerfScenarioAction>([
  'context-loss-restore',
  'dispose',
  'focus',
  'interior',
  'leader-dialogue',
  'orbit',
  'quality',
  'window-hidden',
  'window-minimized',
  'window-visible-cycle',
  'zoom'
])

function isHandshake(value: unknown): value is LunarCityPerfHandshake {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return false
  }

  const candidate = value as Partial<LunarCityPerfHandshake>

  return (
    candidate.bridgeVersion === 1 &&
    candidate.packaged === true &&
    typeof candidate.launchNonce === 'string' &&
    typeof candidate.buildSha === 'string' &&
    typeof candidate.mainPid === 'number' &&
    candidate.processMetricsSource === 'electron.app.getAppMetrics' &&
    typeof candidate.rendererIdentity?.pid === 'number' &&
    typeof candidate.rendererIdentity?.startedAtMs === 'number'
  )
}

const copyHandshake = (handshake: LunarCityPerfHandshake): LunarCityPerfHandshake => structuredClone(handshake)

export function createLunarCityPerfPreload(ipcRenderer: IpcRendererLike, runtimePort: LunarCityPerfRuntimePort) {
  const handshake = ipcRenderer.sendSync('hermes:lunar-city-perf:bootstrap')

  if (!isHandshake(handshake)) {
    return undefined
  }

  if (ipcRenderer.sendSync('hermes:lunar-city-perf:register-responder', handshake) !== true) {
    runtimePort.close?.()

    return undefined
  }

  let activated = false
  let runtimeReady = false
  let resolveReady: (() => void) | undefined

  const ready = new Promise<void>(resolve => {
    resolveReady = resolve
  })

  const pending = new Map<string, { action: string; identity: Record<string, unknown>; requestId: string }>()

  const listener = async (_event: unknown, request: unknown): Promise<void> => {
    if (!activated || !runtimeReady || !request || typeof request !== 'object' || Array.isArray(request)) {
      return
    }

    const value = request as {
      action?: unknown
      identity?: Record<string, unknown>
      payload?: unknown
      requestId?: unknown
    }

    const identity = value.identity

    if (
      typeof value.action !== 'string' ||
      typeof value.requestId !== 'string' ||
      !identity ||
      identity.bridgeVersion !== handshake.bridgeVersion ||
      identity.buildSha !== handshake.buildSha ||
      identity.launchNonce !== handshake.launchNonce ||
      identity.mainPid !== handshake.mainPid ||
      identity.rendererPid !== handshake.rendererIdentity.pid ||
      identity.rendererStartedAtMs !== handshake.rendererIdentity.startedAtMs ||
      !Number.isInteger(identity.rendererGeneration) ||
      !Number.isInteger(identity.frameId) ||
      !Number.isInteger(identity.senderId)
    ) {
      return
    }

    if (pending.has(value.requestId)) {
      return
    }

    pending.set(value.requestId, { action: value.action, identity, requestId: value.requestId })
    runtimePort.postMessage({
      action: value.action,
      payload: value.payload,
      requestId: value.requestId,
      type: 'request'
    })
  }

  runtimePort.onmessage = event => {
    const message = event.data

    if (!message || typeof message !== 'object' || Array.isArray(message)) {
      return
    }

    const value = message as { requestId?: unknown; type?: unknown; value?: unknown }

    if (value.type === 'ready' && !runtimeReady) {
      runtimeReady = true
      resolveReady?.()
      resolveReady = undefined

      return
    }

    if (value.type !== 'response' || typeof value.requestId !== 'string') {
      return
    }

    const request = pending.get(value.requestId)

    if (!request) {
      return
    }

    pending.delete(value.requestId)
    let responseValue = value.value

    try {
      if (request.action === 'snapshot') {
        if (!responseValue || typeof responseValue !== 'object' || Array.isArray(responseValue)) {
          throw new Error('Lunar City performance snapshot response is malformed')
        }

        const metrics = responseValue as Record<string, unknown>

        if (
          (metrics.rendererPid !== undefined && metrics.rendererPid !== request.identity.rendererPid) ||
          (metrics.rendererStartedAtMs !== undefined &&
            metrics.rendererStartedAtMs !== request.identity.rendererStartedAtMs) ||
          (metrics.rendererGeneration !== undefined &&
            metrics.rendererGeneration !== request.identity.rendererGeneration)
        ) {
          throw new Error('Lunar City performance snapshot identity mismatch')
        }

        responseValue = {
          ...metrics,
          rendererGeneration: request.identity.rendererGeneration,
          rendererPid: request.identity.rendererPid,
          rendererStartedAtMs: request.identity.rendererStartedAtMs
        }
      }

      ipcRenderer.send('hermes:lunar-city-perf:response', {
        action: request.action,
        identity: request.identity,
        requestId: request.requestId,
        value: responseValue
      })
    } catch (error) {
      ipcRenderer.send('hermes:lunar-city-perf:response', {
        action: request.action,
        identity: request.identity,
        requestId: request.requestId,
        value: { error: error instanceof Error ? error.message : String(error) }
      })
    }
  }

  runtimePort.start?.()

  ipcRenderer.on('hermes:lunar-city-perf:request', listener)

  const requireHandshake = (): void => {
    if (!activated) {
      throw new Error('Lunar City performance handshake is required')
    }
  }

  const requestRenderer = (action: string, payload?: unknown) => {
    requireHandshake()

    return ipcRenderer.invoke('hermes:lunar-city-perf:renderer-request', action, payload)
  }

  return {
    ready,
    surface: {
      handshake(expected: { bridgeVersion?: unknown; launchNonce?: unknown }): LunarCityPerfHandshake | null {
        if (
          activated ||
          expected?.bridgeVersion !== 1 ||
          expected.launchNonce !== handshake.launchNonce ||
          ipcRenderer.sendSync('hermes:lunar-city-perf:activate', handshake) !== true
        ) {
          return null
        }

        activated = true

        return copyHandshake(handshake)
      },
      mountCity: () => requestRenderer('mount-city'),
      prepareBaselineShell: () => requestRenderer('prepare-baseline-shell'),
      processMetrics: () => {
        requireHandshake()

        return ipcRenderer.invoke('hermes:lunar-city-perf:process-metrics')
      },
      runAction(action: LunarCityPerfScenarioAction, payload?: unknown) {
        if (!SCENARIO_ACTIONS.has(action)) {
          return Promise.reject(new Error(`Unsupported Lunar City performance action: ${String(action)}`))
        }

        return requestRenderer('scenario-action', { action, payload })
      },
      snapshot: () => requestRenderer('snapshot')
    }
  }
}
