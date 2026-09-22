import { leaderModelIdForOwner, leaderOwnerForProfile } from './leader-runtime'
import type { CameraControlState, CameraIntent, EntityKey, LeaderId, LunarCitySnapshot, QualityTier } from './model'
import { lunarCityPerfRuntimeEndpoint } from './perf-runtime-channel'

export interface LunarCityWorldPerfMetrics {
  activeAnimations: number
  drawCalls: number
  entities: number
  frameMs: number
  frameTimestampsMs: readonly number[]
  listeners: number
  rafs: number
  renderFrames: number
  targetFps: 0 | 15 | 30
  textures: number
  timers: number
  visibleTriangles: number
  worldUpdateMs: number
  worldUpdateTimestampsMs: readonly number[]
}

export interface LunarCityPerfRouteRegistration {
  canvas: HTMLCanvasElement
  getCameraPose(): { alpha: number; beta: number; radius: number }
  getCameraState(): CameraControlState
  getCitySnapshot(): LunarCitySnapshot
  getDialogueState(): string
  getInteriorState(): boolean
  getQuality(): { internalRenderScale: number; qualityTier: QualityTier }
  getWorldGeneration(): number
  getWorldMetrics(): LunarCityWorldPerfMetrics
  performLeaderDialogue(leaderId: LeaderId): Promise<{ opened: number; received: number; sent: number }>
  routeMountKey?: string
  setInterior(value: boolean): void
  setQuality(tier: QualityTier): void
  worldAction(intent: CameraIntent): void
}

interface LunarCityPerfEndpoint {
  onRequest(callback: (action: string, payload: unknown) => Promise<unknown>): () => void
}

interface ActionCounters {
  cameraActions: { overview: number; focus: number; orbit: number; zoom: number; indoor: number }
  dialogueActions: { opened: number; messagesSent: number; responsesReceived: number }
  lifecycleActions: { contextLosses: number; recoveries: number; disposals: number }
  qualityActions: { transitions: number }
}

interface RouteState {
  disposed: boolean
  generation: number
  mountId: string
  registration: LunarCityPerfRouteRegistration
  startedAtMs: number
}

const ACTIVE_ANIMATIONS = new Set(['acknowledging', 'listening', 'move', 'talking', 'thinking', 'triage', 'work'])
const QUALITY_TIERS = new Set<QualityTier>(['efficient', 'balanced', 'detailed'])

function finite(value: unknown, fallback = 0): number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 ? value : fallback
}

function signedFinite(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined
}

function lodFor(entity: LunarCitySnapshot['entities'] extends ReadonlyMap<EntityKey, infer E> ? E : never) {
  const placement = entity.presentation?.placement

  if (placement?.overflow) {
    return 'far' as const
  }

  if ((placement?.lodHint ?? 0) <= 0) {
    return 'near' as const
  }

  if ((placement?.lodHint ?? 0) === 1) {
    return 'mid' as const
  }

  return 'far' as const
}

function population(snapshot: LunarCitySnapshot) {
  const lodMix = { far: 0, mid: 0, near: 0 }
  const populationSourceMix: Record<string, number> = {}
  let active = 0
  let activeLeaderAnimations = 0
  let activeWorkerAnimations = 0

  for (const entity of snapshot.entities.values()) {
    lodMix[lodFor(entity)] += 1
    populationSourceMix[entity.identity.connectionId] = (populationSourceMix[entity.identity.connectionId] ?? 0) + 1

    if (ACTIVE_ANIMATIONS.has(entity.animation)) {
      active += 1

      if (entity.identity.kind === 'profile') {
        activeLeaderAnimations += 1
      } else {
        activeWorkerAnimations += 1
      }
    }
  }

  return {
    activeLeaderAnimations,
    activeWorkerAnimations,
    population: {
      active,
      lodMix,
      observed: snapshot.entities.size,
      source: 'lunar-city-snapshot-v1'
    },
    populationSourceMix
  }
}

function emptyWorldMetrics(): LunarCityWorldPerfMetrics {
  return {
    activeAnimations: 0,
    drawCalls: 0,
    entities: 0,
    frameMs: 0,
    frameTimestampsMs: [],
    listeners: 0,
    rafs: 0,
    renderFrames: 0,
    targetFps: 0,
    textures: 0,
    timers: 0,
    visibleTriangles: 0,
    worldUpdateMs: 0,
    worldUpdateTimestampsMs: []
  }
}

export function createLunarCityPerfRuntime(
  endpoint: LunarCityPerfEndpoint | undefined,
  options: { actionTimeoutMs?: number; navigate?: (path: '/' | '/lunar-city') => void; now?: () => number } = {}
) {
  if (!endpoint) {
    return { enabled: false as const, registerRoute: undefined }
  }

  const now = options.now ?? (() => performance.timeOrigin + performance.now())
  const actionTimeoutMs = options.actionTimeoutMs ?? 5_000

  const navigate =
    options.navigate ??
    ((path: '/' | '/lunar-city') => {
      window.location.hash = `#${path}`
    })

  const counters: ActionCounters = {
    cameraActions: { overview: 0, focus: 0, orbit: 0, zoom: 0, indoor: 0 },
    dialogueActions: { opened: 0, messagesSent: 0, responsesReceived: 0 },
    lifecycleActions: { contextLosses: 0, recoveries: 0, disposals: 0 },
    qualityActions: { transitions: 0 }
  }

  let generation = 0
  let lifecycleState: 'contextLost' | 'disposed' | 'mounted' | 'recovered' = 'disposed'
  let route: RouteState | undefined
  let lastMount = { generation: 1, id: 'lunar-city-scene:unmounted', startedAtMs: now() }
  let runtimeDisposed = false

  const waitFor = async (predicate: () => boolean, label: string): Promise<void> => {
    const deadline = Date.now() + actionTimeoutMs

    while (!predicate()) {
      if (Date.now() >= deadline) {
        throw new Error(`Timed out observing Lunar City ${label}`)
      }

      await new Promise(resolve => setTimeout(resolve, 10))
    }
  }

  const disposeRoute = (state: RouteState): void => {
    if (state.disposed) {
      return
    }

    state.disposed = true
    counters.lifecycleActions.disposals += 1
    lifecycleState = 'disposed'

    if (route === state) {
      route = undefined
    }
  }

  const snapshot = () => {
    const activeRoute = route && !route.disposed ? route : undefined
    const world = activeRoute?.registration.getWorldMetrics() ?? emptyWorldMetrics()
    const citySnapshot = activeRoute?.registration.getCitySnapshot()

    const exact = activeRoute
      ? population(citySnapshot!)
      : {
          activeLeaderAnimations: 0,
          activeWorkerAnimations: 0,
          population: { active: 0, lodMix: {}, observed: 0, source: 'route-unmounted' },
          populationSourceMix: {}
        }

    const workerTarget = citySnapshot
      ? [...citySnapshot.entities.values()].find(entity => entity.identity.kind !== 'profile')
      : undefined

    const leaderTarget = citySnapshot
      ? [...citySnapshot.entities.values()].map(leaderOwnerForProfile).find(owner => owner !== undefined)
      : undefined

    const quality = activeRoute?.registration.getQuality() ?? {
      internalRenderScale: 1,
      qualityTier: 'balanced' as const
    }

    const camera = activeRoute?.registration.getCameraState()

    const cameraState = activeRoute?.registration.getInteriorState()
      ? 'indoor'
      : camera?.focusedEntityKey
        ? 'worker-focus'
        : counters.cameraActions.orbit > 0 && counters.cameraActions.zoom > 0
          ? 'orbit-zoom'
          : 'overview'

    const qualityTier = `${quality.qualityTier.charAt(0).toUpperCase()}${quality.qualityTier.slice(1)}`

    return {
      ...world,
      ...exact,
      activeAnimations: finite(world.activeAnimations),
      cameraActions: structuredClone(counters.cameraActions),
      cameraState,
      dialogueActions: structuredClone(counters.dialogueActions),
      dialogueState: activeRoute?.registration.getDialogueState() ?? 'idle',
      environment: {
        electronMode: 'packaged',
        gpuEnabled: true
      },
      internalRenderScale: quality.internalRenderScale,
      lifecycleActions: structuredClone(counters.lifecycleActions),
      lifecycleState,
      qualityTier,
      qualityActions: structuredClone(counters.qualityActions),
      scenarioTargets: {
        ...(leaderTarget ? { leaderId: leaderModelIdForOwner(leaderTarget) } : {}),
        ...(workerTarget ? { workerEntityKey: workerTarget.key } : {})
      },
      sceneMount: structuredClone(lastMount),
      worldGeneration: activeRoute?.registration.getWorldGeneration() ?? 0
    }
  }

  const scenarioAction = async (value: unknown) => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      throw new Error('Unsupported scenario action')
    }

    const request = value as { action?: unknown; payload?: unknown }
    const activeRoute = route && !route.disposed ? route : undefined

    if (!activeRoute || typeof request.action !== 'string') {
      throw new Error('Unsupported scenario action')
    }

    const payload =
      request.payload && typeof request.payload === 'object' ? (request.payload as Record<string, unknown>) : {}

    const registration = activeRoute.registration

    switch (request.action) {
      case 'quality': {
        const tier = payload.tier

        if (!QUALITY_TIERS.has(tier as QualityTier)) {
          throw new Error('Unsupported quality tier')
        }

        const before = registration.getQuality()

        if (before.qualityTier === tier) {
          throw new Error(`Quality tier is already ${tier}; a nonzero transition is required`)
        }

        registration.setQuality(tier as QualityTier)
        await waitFor(() => registration.getQuality().qualityTier === tier, 'quality transition')
        const after = registration.getQuality()

        if (after.internalRenderScale === before.internalRenderScale) {
          throw new Error('Quality transition did not change the internal render scale')
        }

        counters.qualityActions.transitions += 1

        return {
          action: request.action,
          from: { internalRenderScale: before.internalRenderScale, tier: before.qualityTier },
          proof: counters.qualityActions.transitions,
          to: { internalRenderScale: after.internalRenderScale, tier: after.qualityTier }
        }
      }

      case 'orbit': {
        const deltaAlpha = signedFinite(payload.deltaAlpha)
        const deltaBeta = signedFinite(payload.deltaBeta)

        if (deltaAlpha === undefined || deltaBeta === undefined || (deltaAlpha === 0 && deltaBeta === 0)) {
          throw new Error('Orbit requires a nonzero finite camera delta')
        }

        const before = registration.getCameraPose()
        registration.worldAction({ deltaAlpha, deltaBeta, kind: 'orbit' })
        await waitFor(() => {
          const after = registration.getCameraPose()

          return after.alpha !== before.alpha || after.beta !== before.beta
        }, 'orbit camera delta')
        counters.cameraActions.orbit += 1

        return { action: request.action, proof: counters.cameraActions.orbit }
      }

      case 'zoom': {
        const delta = signedFinite(payload.delta)

        if (delta === undefined || delta === 0) {
          throw new Error('Zoom requires a nonzero finite camera delta')
        }

        const before = registration.getCameraPose().radius
        registration.worldAction({ delta, kind: 'zoom' })
        await waitFor(() => registration.getCameraPose().radius !== before, 'zoom camera delta')
        counters.cameraActions.zoom += 1

        return { action: request.action, proof: counters.cameraActions.zoom }
      }

      case 'focus': {
        if (typeof payload.entityKey !== 'string') {
          throw new Error('Exact focus entityKey is required')
        }

        const entity = registration.getCitySnapshot().entities.get(payload.entityKey as EntityKey)

        if (!entity || entity.identity.kind === 'profile') {
          throw new Error('Focus requires an existing exact worker entityKey')
        }

        registration.worldAction({ entityKey: payload.entityKey as EntityKey, follow: true, kind: 'focus' })
        await waitFor(
          () => registration.getCameraState().focusedEntityKey === payload.entityKey,
          'exact worker focus transition'
        )
        counters.cameraActions.focus += 1

        return { action: request.action, entityKey: payload.entityKey, proof: counters.cameraActions.focus }
      }

      case 'interior': {
        if (registration.getInteriorState()) {
          throw new Error('Lunar City is already in an interior')
        }

        registration.setInterior(true)
        await waitFor(() => registration.getInteriorState(), 'interior transition')
        counters.cameraActions.indoor += 1

        return { action: request.action, proof: counters.cameraActions.indoor }
      }

      case 'leader-dialogue': {
        const leaderId = payload.leaderId

        if (
          ![
            'owl',
            'fox',
            'badger',
            'otter',
            'bird',
            'stag',
            'elephant',
            'cat',
            'capybara',
            'lion',
            'beaver',
            'monkey'
          ].includes(String(leaderId))
        ) {
          throw new Error('Exact leaderId is required')
        }

        const observed = await registration.performLeaderDialogue(leaderId as LeaderId)

        if (observed.opened <= 0 || observed.sent <= 0 || observed.received <= 0) {
          throw new Error('Leader dialogue did not complete opened, sent, and received events')
        }

        counters.dialogueActions.opened += observed.opened
        counters.dialogueActions.messagesSent += observed.sent
        counters.dialogueActions.responsesReceived += observed.received

        return { action: request.action, leaderId, proof: counters.dialogueActions.responsesReceived }
      }

      case 'context-loss-restore': {
        const gl = registration.canvas.getContext('webgl2') ?? registration.canvas.getContext('webgl')
        const extension = gl?.getExtension('WEBGL_lose_context')

        if (!extension) {
          throw new Error('WEBGL_lose_context is unavailable')
        }

        const beforeGeneration = registration.getWorldGeneration()
        let lost = false
        let restored = false
        let restoreAttempted = false

        const onLost = (): void => {
          lost = true
        }

        const onRestored = (): void => {
          restored = true
        }

        registration.canvas.addEventListener('webglcontextlost', onLost, { once: true })
        registration.canvas.addEventListener('webglcontextrestored', onRestored, { once: true })

        try {
          extension.loseContext()
          await waitFor(() => lost, 'WEBGL context loss event')
          counters.lifecycleActions.contextLosses += 1
          lifecycleState = 'contextLost'
          restoreAttempted = true
          extension.restoreContext()
          await waitFor(() => restored, 'WEBGL context restore event')
          await waitFor(
            () => (route?.registration.getWorldGeneration() ?? 0) > beforeGeneration,
            'restored world generation'
          )
          counters.lifecycleActions.recoveries += 1
          lifecycleState = 'recovered'

          return {
            action: request.action,
            lifecycleActions: structuredClone(counters.lifecycleActions),
            lifecycleTrace: ['contextLost', 'recovered'],
            proof: counters.lifecycleActions.recoveries
          }
        } finally {
          if (!restoreAttempted) {
            try {
              extension.restoreContext()
            } catch {
              // Best-effort bounded recovery after a loss-event timeout.
            }
          }

          registration.canvas.removeEventListener('webglcontextlost', onLost)
          registration.canvas.removeEventListener('webglcontextrestored', onRestored)
        }
      }

      case 'dispose': {
        const disposingRoute = activeRoute

        navigate('/')
        await waitFor(() => disposingRoute.disposed && route === undefined, 'route unmount and Babylon world disposal')

        return { action: request.action, proof: counters.lifecycleActions.disposals }
      }

      default:
        throw new Error(`Unsupported Lunar City performance action: ${request.action}`)
    }
  }

  const release = endpoint.onRequest(async (action, payload) => {
    if (runtimeDisposed) {
      throw new Error('Lunar City performance runtime disposed')
    }

    if (action === 'snapshot') {
      return snapshot()
    }

    if (action === 'prepare-baseline-shell') {
      navigate('/')
      counters.cameraActions.overview += 1

      return { phase: 'baseline-shell', proof: counters.cameraActions.overview }
    }

    if (action === 'mount-city') {
      navigate('/lunar-city')

      return { phase: 'mounted-city' }
    }

    if (action === 'scenario-action') {
      return scenarioAction(payload)
    }

    throw new Error(`Unsupported Lunar City performance request: ${action}`)
  })

  return {
    dispose(): void {
      if (runtimeDisposed) {
        return
      }

      runtimeDisposed = true

      if (route) {
        disposeRoute(route)
      }

      release()
    },
    enabled: true as const,
    registerRoute(registration: LunarCityPerfRouteRegistration) {
      generation += 1

      const sameRouteMount = registration.routeMountKey && lastMount.id.startsWith(`${registration.routeMountKey}:`)
      const mountGeneration = sameRouteMount ? lastMount.generation : generation
      const mountStartedAt = sameRouteMount ? lastMount.startedAtMs : now()

      const state: RouteState = {
        disposed: false,
        generation: mountGeneration,
        mountId: sameRouteMount
          ? lastMount.id
          : `${registration.routeMountKey ?? 'lunar-city-scene'}:${mountGeneration}:${Math.round(mountStartedAt)}`,
        registration,
        startedAtMs: mountStartedAt
      }

      route = state
      lifecycleState = 'mounted'
      lastMount = { generation: state.generation, id: state.mountId, startedAtMs: state.startedAtMs }

      return {
        dispose(): void {
          disposeRoute(state)
        }
      }
    }
  }
}

export const lunarCityPerfRuntime = createLunarCityPerfRuntime(
  typeof window === 'undefined' || window.__LUNAR_CITY_PERF_AUTHORIZED__ !== true
    ? undefined
    : lunarCityPerfRuntimeEndpoint
)
