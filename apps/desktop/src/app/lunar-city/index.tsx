import './lunar-city.css'

import { LOCAL_CONNECTION_ID } from '@hermes/shared'
import { useStore } from '@nanostores/react'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

import { Button } from '@/components/ui/button'
import { Moon, Network } from '@/lib/icons'
import { $fleetRoster } from '@/store/fleet-roster'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $connection } from '@/store/session'

import { createRegisteredKanbanCitySource } from './adapters/kanban'
import { startLunarCityReconciler } from './adapters/reconciler'
import { LunarCityCommandRuntime } from './command-runtime'
import { AlertsPanel } from './components/alerts-panel'
import { CameraControls } from './components/camera-controls'
import { DelegationWorld } from './components/delegation-world'
import type { InspectorSessionTarget } from './components/entity-inspector'
import { entityFriendlyLabel, EntityList } from './components/entity-list'
import { InteriorCutawayControls } from './components/interior-cutaway-controls'
import { LeaderDialogueRuntime } from './components/leader-dialogue-runtime'
import { ProjectCompoundsPanel } from './components/project-compounds-panel'
import { QualityControl, type RendererStatus } from './components/quality-control'
import { reducedMotionPresentation } from './components/reduced-motion'
import { SourceHealthPanel } from './components/source-health'
import { LunarCityStatusBadges } from './components/status-badges'
import { WorkerLife } from './components/worker-life'
import {
  leaderModelFocusKeyForOwner,
  leaderModelIdForOwner,
  leaderOwnerForProfile,
  profileLeaders
} from './leader-runtime'
import { type LeaderOwner, leaderOwnerKey, type LeaderSession, resolveLeaderSession } from './leader-sessions'
import { loadWorldManifest } from './manifest'
import type {
  CameraControlState,
  CameraIntent,
  EntityKey,
  LeaderId,
  LunarCityIntent,
  LunarCityWorldHandle,
  LunarEntity,
  QualityTier,
  WorldManifestV2
} from './model'
import { lunarCityPerfRuntime } from './perf-runtime'
import { $lunarCitySnapshot } from './store'
import { createLunarCityWorld } from './world/create-world'
import { createProjectCompoundController } from './world/project-compounds'

const LUNAR_CITY_MANIFEST_URL = './lunar-city/v2/world-manifest.v2.json'
// A renderer must never leave the route in an unbounded STARTING state. This
// is deliberately generous for a cold Electron asset load while still giving
// browser previews and low-power machines a deterministic degraded surface.
const RENDERER_START_TIMEOUT_MS = 30_000

/**
 * Hosts can provide a reviewed asset-overlay manifest without publishing
 * their meshes in Hermes.  Relative URLs (and same-origin absolute URLs) are
 * accepted; cross-origin URLs stay on the built-in asset-neutral manifest so
 * a query string cannot silently turn the city route into an asset fetcher.
 */
export function lunarCityManifestUrl(locationLike?: Location): string {
  if (!locationLike) {
    return LUNAR_CITY_MANIFEST_URL
  }

  const candidate = new URLSearchParams(locationLike.search).get('lunarCityManifest')?.trim()

  if (!candidate) {
    return LUNAR_CITY_MANIFEST_URL
  }

  try {
    const resolved = new URL(candidate, locationLike.href)

    if (resolved.origin !== locationLike.origin && /^[a-z][a-z\d+.-]*:/i.test(candidate)) {
      return LUNAR_CITY_MANIFEST_URL
    }

    return candidate
  } catch {
    return LUNAR_CITY_MANIFEST_URL
  }
}

export function disposeLunarCityRuntime(
  stopReconciler: (() => void) | undefined,
  stopSnapshot: (() => void) | undefined,
  world: LunarCityWorldHandle | undefined
): void {
  stopReconciler?.()
  stopSnapshot?.()
  world?.destroy()
}

export interface LunarCityProps {
  onOpenEntitySession?: (target: InspectorSessionTarget) => void
  onOpenFullChat?: (storedId: string, owner: LeaderOwner) => Promise<void> | void
  onOpenMemoryGraph: () => void
}

function sameProfileLeaderList(left: readonly LunarEntity[], right: readonly LunarEntity[]): boolean {
  return left.length === right.length && left.every((entity, index) => entity.key === right[index]?.key)
}

export function lunarCityHudStatus(
  rendererStatus: RendererStatus | 'loading',
  snapshot: ReturnType<typeof $lunarCitySnapshot.get>
): 'DEGRADED' | 'EMPTY' | 'LIVE' | 'RENDERER UNAVAILABLE' | 'STALE' | 'STARTING' | 'UNAVAILABLE' {
  if (rendererStatus === 'unavailable') {
    return 'RENDERER UNAVAILABLE'
  }

  if (rendererStatus === 'loading') {
    return 'STARTING'
  }

  if (rendererStatus === 'degraded') {
    return 'DEGRADED'
  }

  if (snapshot.sources.length === 0 || snapshot.entities.size === 0) {
    return 'EMPTY'
  }

  if (snapshot.sources.some(source => source.error || source.authority === 'unknown')) {
    return 'UNAVAILABLE'
  }

  if (snapshot.sources.some(source => source.authority === 'partial' || source.authority === 'stale')) {
    return 'STALE'
  }

  return 'LIVE'
}

function LunarCityHudStatus({ rendererStatus }: { rendererStatus: RendererStatus | 'loading' }) {
  const snapshot = useStore($lunarCitySnapshot)

  return <>{lunarCityHudStatus(rendererStatus, snapshot)}</>
}

interface LunarCityOperationsProps {
  manifest: WorldManifestV2 | undefined
  worldRef: { current: LunarCityWorldHandle | undefined }
  getCameraOrder(): readonly EntityKey[]
  onQualityChange(tier: QualityTier): void
  onSelect(entity: LunarEntity): void
  qualityTier: QualityTier
  reducedMotion: boolean
  rendererStatus: RendererStatus
  selectedEntityKey?: EntityKey
}

/**
 * The text-first operational surface owns its narrow store subscription so
 * high-frequency world publications never re-render the route or canvas.
 */
function LunarCityOperations({
  manifest,
  worldRef,
  getCameraOrder,
  onQualityChange,
  onSelect,
  qualityTier,
  reducedMotion,
  rendererStatus,
  selectedEntityKey
}: LunarCityOperationsProps) {
  const snapshot = useStore($lunarCitySnapshot)
  const motion = reducedMotionPresentation(reducedMotion)
  const projects = useMemo(() => createProjectCompoundController(manifest?.projectSlots ?? []), [manifest])
  const projectReport = useMemo(() => projects.update(snapshot), [projects, snapshot])

  return (
    <aside aria-label="Lunar City accessible operations" className="lunar-city-accessible-operations">
      {rendererStatus === 'unavailable' ? (
        <p className="lunar-city-renderer-fallback" role="status">
          3D world renderer unavailable. Live Hermes operations remain available below.
        </p>
      ) : null}
      <p className="lunar-city-motion-status">
        {motion.snapToDestination
          ? 'Reduced motion: destinations snap into place; camera easing and looping clips are stopped.'
          : 'Motion: camera easing and worker travel are enabled.'}
      </p>
      <QualityControl onTierChange={onQualityChange} rendererStatus={rendererStatus} tier={qualityTier} />
      <SourceHealthPanel sources={snapshot.sources} />
      <AlertsPanel onSelectEntity={onSelect} snapshot={snapshot} />
      <ProjectCompoundsPanel onSelectEntity={onSelect} report={projectReport} snapshot={snapshot} />
      <WorkerLife onSelect={onSelect} selectedEntityKey={selectedEntityKey} snapshot={snapshot} worldRef={worldRef} />
      <EntityList
        cameraOrder={getCameraOrder()}
        onSelect={onSelect}
        selectedEntityKey={selectedEntityKey}
        snapshot={snapshot}
      />
    </aside>
  )
}

export function LunarCity({ onOpenEntitySession, onOpenFullChat, onOpenMemoryGraph }: LunarCityProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const activeGatewayProfile = useStore($activeGatewayProfile)
  const connection = useStore($connection)
  const [inside, setInside] = useState(false)
  const [cameraState, setCameraState] = useState<CameraControlState>({ focusedEntityKey: undefined, following: false })
  const [selectedEntityKey, setSelectedEntityKey] = useState<EntityKey | undefined>(undefined)
  const [focusedEntityLabel, setFocusedEntityLabel] = useState<string | undefined>(undefined)
  const [qualityTier, setQualityTier] = useState<QualityTier>('efficient')
  const [rendererStatus, setRendererStatus] = useState<RendererStatus | 'loading'>('loading')
  const [operationsReady, setOperationsReady] = useState(false)

  const [reducedMotion, setReducedMotion] = useState(
    () =>
      typeof globalThis.matchMedia === 'function' && globalThis.matchMedia('(prefers-reduced-motion: reduce)').matches
  )

  const [worldManifest, setWorldManifest] = useState<WorldManifestV2 | undefined>(undefined)

  const [profileLeaderEntities, setProfileLeaderEntities] = useState(() =>
    profileLeaders($lunarCitySnapshot.get().entities)
  )

  const [selectedLeaderOwner, setSelectedLeaderOwner] = useState<LeaderOwner | undefined>(undefined)
  const [leaderDisambiguationId, setLeaderDisambiguationId] = useState<string | undefined>(undefined)
  const [leaderSession, setLeaderSession] = useState<LeaderSession | undefined>(undefined)
  const [leaderSessionError, setLeaderSessionError] = useState<string | undefined>(undefined)
  const worldHandleRef = useRef<LunarCityWorldHandle | undefined>(undefined)
  const qualityTierRef = useRef(qualityTier)
  const profileLeaderEntitiesRef = useRef(profileLeaderEntities)
  const selectedLeaderOwnerRef = useRef(selectedLeaderOwner)
  const leaderSessionRef = useRef(leaderSession)

  const leaderPerfScenarioRef = useRef<
    ((text: string) => Promise<{ opened: number; received: number; sent: number }>) | undefined
  >(undefined)

  const insideRef = useRef(inside)
  const leaderModelIdsRef = useRef<readonly LeaderId[] | undefined>(undefined)
  leaderModelIdsRef.current = worldManifest?.characterAssets.leaders.map(asset => asset.id)
  const openLeaderRef = useRef<(entity: LunarEntity) => void>(() => undefined)
  const stopReconcilerRef = useRef<(() => void) | undefined>(undefined)
  selectedLeaderOwnerRef.current = selectedLeaderOwner
  leaderSessionRef.current = leaderSession
  insideRef.current = inside

  const kanbanScope = useMemo(() => {
    const connectionId =
      connection?.connectionId?.trim() || (connection?.mode === 'remote' ? undefined : LOCAL_CONNECTION_ID)

    return connectionId
      ? {
          connectionId,
          profile: normalizeProfileKey(activeGatewayProfile)
        }
      : undefined
  }, [activeGatewayProfile, connection?.connectionId, connection?.mode])

  const dispatchCamera = (intent: CameraIntent): void => {
    const world = worldHandleRef.current

    world?.dispatchCamera(intent)
    const state = world?.getCameraState() ?? { focusedEntityKey: undefined, following: false }
    setCameraState(state)

    if (!state.focusedEntityKey) {
      setFocusedEntityLabel(undefined)
    } else {
      const entity = $lunarCitySnapshot.get().entities.get(state.focusedEntityKey)
      setFocusedEntityLabel(entity ? entityFriendlyLabel(entity) : 'selected entity')
    }
  }

  const selectEntity = (entity: LunarEntity): void => {
    setSelectedEntityKey(entity.key)
    dispatchCamera({ kind: 'focus', entityKey: entity.key, follow: false })
    setFocusedEntityLabel(entityFriendlyLabel(entity))
  }

  const changeQuality = (tier: QualityTier): void => {
    qualityTierRef.current = tier
    setQualityTier(tier)
    worldHandleRef.current?.setQuality(tier)
  }

  useEffect(() => {
    if (typeof globalThis.matchMedia !== 'function') {
      return
    }

    const query = globalThis.matchMedia('(prefers-reduced-motion: reduce)')

    const applyPreference = (matches: boolean): void => {
      setReducedMotion(matches)
      worldHandleRef.current?.setReducedMotion(matches)
    }

    const onChange = (event: MediaQueryListEvent): void => applyPreference(event.matches)
    applyPreference(query.matches)
    query.addEventListener?.('change', onChange)

    return () => query.removeEventListener?.('change', onChange)
  }, [])

  const selectedLeaderKey = selectedLeaderOwner ? leaderOwnerKey(selectedLeaderOwner) : undefined

  const selectedLeaderModel = selectedLeaderOwner
    ? leaderModelIdForOwner(selectedLeaderOwner, leaderModelIdsRef.current)
    : undefined

  const ambiguousLeaderModels = useMemo(() => {
    const counts = new Map<string, number>()

    for (const entity of profileLeaderEntities) {
      const owner = leaderOwnerForProfile(entity)

      if (owner) {
        const modelId = leaderModelIdForOwner(owner, leaderModelIdsRef.current)
        counts.set(modelId, (counts.get(modelId) ?? 0) + 1)
      }
    }

    return [...counts].filter((entry): entry is [string, number] => entry[1] > 1)
  }, [profileLeaderEntities])

  const onLeaderStateChange = useCallback(
    (state: Parameters<LunarCityWorldHandle['setLeaderAnimation']>[1]): void => {
      if (selectedLeaderModel) {
        worldHandleRef.current?.setLeaderAnimation(selectedLeaderModel, state)
      }
    },
    [selectedLeaderModel]
  )

  const closeLeaderDialogue = (): void => {
    if (leaderSessionError) {
      onLeaderStateChange('idle')
    }

    setSelectedLeaderOwner(undefined)
    setLeaderSession(undefined)
    setLeaderSessionError(undefined)
  }

  const openLeader = (entity: LunarEntity): void => {
    const owner = leaderOwnerForProfile(entity)

    if (!owner) {
      return
    }

    // The source profile is the selected identity. The animal model is only a
    // deterministic visual assignment and never a route or mutation key.
    setSelectedLeaderOwner(owner)
    setLeaderSession(undefined)
    setLeaderSessionError(undefined)
    setLeaderDisambiguationId(undefined)
    setSelectedEntityKey(entity.key)
    dispatchCamera({
      kind: 'focus',
      entityKey: leaderModelFocusKeyForOwner(owner, leaderModelIdsRef.current),
      follow: false
    })
    setFocusedEntityLabel(`${owner.profile} leader`)
  }

  openLeaderRef.current = openLeader

  const selectPhysicalLeaderModel = (leaderId: string): void => {
    const candidates = profileLeaderEntitiesRef.current.filter(candidate => {
      const owner = leaderOwnerForProfile(candidate)

      return owner && leaderModelIdForOwner(owner, leaderModelIdsRef.current) === leaderId
    })

    if (candidates.length === 1) {
      openLeaderRef.current(candidates[0]!)
    } else if (candidates.length > 1) {
      setLeaderDisambiguationId(leaderId)
      setFocusedEntityLabel(`${candidates.length} exact ${leaderId} profiles`)
    }
  }

  // Profile entities change far less often than workers. Keep this compact
  // semantic list separate from the imperative world snapshot listener so
  // worker/frame publications do not re-render the React route.
  useEffect(
    () =>
      $lunarCitySnapshot.listen(snapshot => {
        const next = profileLeaders(snapshot.entities)
        const current = profileLeaderEntitiesRef.current

        if (!sameProfileLeaderList(current, next)) {
          profileLeaderEntitiesRef.current = next
          setProfileLeaderEntities(next)
        }
      }),
    []
  )

  useEffect(() => {
    if (!selectedLeaderKey) {
      return
    }

    if (!profileLeaderEntities.some(entity => leaderOwnerKey(leaderOwnerForProfile(entity)!) === selectedLeaderKey)) {
      setSelectedLeaderOwner(undefined)
      setLeaderSession(undefined)
      setLeaderSessionError(undefined)
    }
  }, [profileLeaderEntities, selectedLeaderKey])

  useEffect(() => {
    if (!selectedLeaderOwner || !selectedLeaderKey) {
      return
    }

    let current = true
    const owner = selectedLeaderOwner

    setLeaderSession(undefined)
    setLeaderSessionError(undefined)
    void resolveLeaderSession(owner).then(
      resolved => {
        if (current && leaderOwnerKey(owner) === selectedLeaderKey) {
          setLeaderSession(resolved)
        }
      },
      error => {
        if (current && leaderOwnerKey(owner) === selectedLeaderKey) {
          setLeaderSessionError(error instanceof Error ? error.message : String(error))
        }
      }
    )

    return () => {
      current = false
    }
  }, [selectedLeaderKey, selectedLeaderOwner])

  useEffect(() => {
    if (leaderSessionError) {
      onLeaderStateChange('unavailable')
    }
  }, [leaderSessionError, onLeaderStateChange])

  const voiceAvailable = Boolean(
    selectedLeaderOwner &&
    kanbanScope &&
    selectedLeaderOwner.connectionId === kanbanScope.connectionId &&
    selectedLeaderOwner.profile === kanbanScope.profile
  )

  const openSelectedLeaderFullChat = async (storedId: string, owner: LeaderOwner): Promise<void> => {
    if (!onOpenFullChat) {
      throw new Error('Open Full Chat is available only from the dedicated Lunar City route')
    }

    await onOpenFullChat(storedId, owner)
  }

  // The reconciler is imperative route lifetime state, so the world cleanup can stop it before disposal.
  // eslint-disable-next-line no-restricted-syntax
  useEffect(() => {
    if (!worldManifest) {
      return
    }

    const kanbanSource = createRegisteredKanbanCitySource({ manifest: worldManifest, roster: $fleetRoster })

    const stop = startLunarCityReconciler({ optionalSources: [kanbanSource] })
    let stopped = false

    const stopOnce = (): void => {
      if (stopped) {
        return
      }

      stopped = true

      if (stopReconcilerRef.current === stopOnce) {
        stopReconcilerRef.current = undefined
      }

      stop()
    }

    stopReconcilerRef.current = stopOnce

    return stopOnce
  }, [worldManifest])

  // The Babylon handle is imperative runtime state, not a mirror of reactive route data.
  // eslint-disable-next-line no-restricted-syntax
  useEffect(() => {
    const canvas = canvasRef.current

    if (!canvas) {
      return
    }

    canvas.dataset.worldStatus = 'loading'

    const abortController = new AbortController()
    let disposed = false
    let generation = 0
    let manifest: WorldManifestV2 | undefined
    let restorationArmedGeneration: number | undefined
    let restorationAttempted = false
    let world: LunarCityWorldHandle | undefined
    let perfRoute: { dispose(): void } | undefined
    let stopSnapshot: (() => void) | undefined
    let rendererStartTimeout: number | undefined

    const manifestUrl = lunarCityManifestUrl(
      typeof globalThis.location !== 'undefined' ? globalThis.location : undefined
    )

    const handleWorldIntent = (intent: LunarCityIntent): void => {
      if (intent.kind === 'camera-state' && intent.state) {
        setCameraState(intent.state)

        const focused = intent.state.focusedEntityKey
          ? $lunarCitySnapshot.get().entities.get(intent.state.focusedEntityKey)
          : undefined

        setFocusedEntityLabel(
          intent.state.focusedEntityKey ? (focused ? entityFriendlyLabel(focused) : 'selected entity') : undefined
        )
      } else if (intent.kind === 'select-focus') {
        const entity = $lunarCitySnapshot.get().entities.get(intent.entityKey)

        if (entity) {
          setSelectedEntityKey(intent.entityKey)
          setFocusedEntityLabel(entityFriendlyLabel(entity))

          if (entity.identity.kind === 'profile') {
            openLeaderRef.current(entity)
          }

          return
        }

        const leaderId = manifest?.characterAssets.leaders.find(
          leader => `lunar-city:leader:${leader.id}` === intent.entityKey
        )?.id

        if (leaderId) {
          selectPhysicalLeaderModel(leaderId)
        } else if (manifest?.models.some(model => `lunar-city:model:${encodeURIComponent(model.id)}` === intent.entityKey)) {
          world?.dispatchCamera({ kind: 'focus', entityKey: intent.entityKey, follow: false })
        }
      } else if (intent.kind === 'clear-selection') {
        setSelectedEntityKey(undefined)
        setFocusedEntityLabel(undefined)
      }
    }

    const retireWorld = (): void => {
      stopSnapshot?.()
      stopSnapshot = undefined
      const retired = world
      world = undefined
      perfRoute?.dispose()
      perfRoute = undefined

      if (worldHandleRef.current === retired) {
        worldHandleRef.current = undefined
      }

      retired?.destroy()
    }

    const createWorldGeneration = async (): Promise<void> => {
      const ownGeneration = ++generation

      try {
        manifest ??= await loadWorldManifest(manifestUrl, abortController.signal)

        if (disposed || ownGeneration !== generation) {
          return
        }

        setWorldManifest(manifest)

        const created = await createLunarCityWorld(canvas, manifest, handleWorldIntent, undefined, manifestUrl)

        if (disposed || ownGeneration !== generation) {
          created.destroy()
        } else {
          world = created
          worldHandleRef.current = created
          created.setQuality(qualityTierRef.current)
          created.setReducedMotion(
            typeof globalThis.matchMedia === 'function' &&
              globalThis.matchMedia('(prefers-reduced-motion: reduce)').matches
          )
          created.applySnapshot($lunarCitySnapshot.get())
          perfRoute = lunarCityPerfRuntime.registerRoute?.({
            canvas,
            getCameraPose: () => {
              const metrics = created.getPerfSnapshot?.() as
                { cameraAlpha?: number; cameraBeta?: number; cameraRadius?: number } | undefined

              return {
                alpha: metrics?.cameraAlpha ?? 0,
                beta: metrics?.cameraBeta ?? 0,
                radius: metrics?.cameraRadius ?? 0
              }
            },
            getCameraState: () => created.getCameraState(),
            getCitySnapshot: () => $lunarCitySnapshot.get(),
            getDialogueState: () => (selectedLeaderOwnerRef.current ? 'active' : 'idle'),
            getInteriorState: () => insideRef.current,
            getQuality: () => {
              const metrics = created.getPerfSnapshot?.() as
                { internalRenderScale?: number; qualityTier?: QualityTier } | undefined

              return {
                internalRenderScale: metrics?.internalRenderScale ?? 1,
                qualityTier: metrics?.qualityTier ?? qualityTierRef.current
              }
            },
            getWorldGeneration: () => ownGeneration,
            getWorldMetrics: () =>
              created.getPerfSnapshot?.() ?? {
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
              },
            performLeaderDialogue: async leaderId => {
              const entity = profileLeaderEntitiesRef.current.find(candidate => {
                const owner = leaderOwnerForProfile(candidate)

                return owner ? leaderModelIdForOwner(owner, leaderModelIdsRef.current) === leaderId : false
              })

              if (!entity) {
                throw new Error(`Leader ${leaderId} is unavailable`)
              }

              openLeaderRef.current(entity)
              const deadline = Date.now() + 2_000

              while (!leaderSessionRef.current) {
                if (Date.now() >= deadline) {
                  throw new Error(`Leader ${leaderId} dialogue did not open`)
                }

                await new Promise(resolve => setTimeout(resolve, 10))
              }

              while (!leaderPerfScenarioRef.current) {
                if (Date.now() >= deadline) {
                  throw new Error(`Leader ${leaderId} dialogue runtime did not become ready`)
                }

                await new Promise(resolve => setTimeout(resolve, 10))
              }

              return leaderPerfScenarioRef.current('Lunar City packaged fake-backend voice acceptance turn.')
            },
            routeMountKey: 'lunar-city-route',
            setInterior: value => setInside(value),
            setQuality: changeQuality,
            worldAction: intent => created.dispatchCamera(intent)
          })
          stopSnapshot = $lunarCitySnapshot.listen(snapshot => {
            if (!disposed && ownGeneration === generation && worldHandleRef.current === created) {
              created.applySnapshot(snapshot)
            }
          })

          if (rendererStartTimeout !== undefined) {
            window.clearTimeout(rendererStartTimeout)
            rendererStartTimeout = undefined
          }

          canvas.dataset.worldStatus = 'ready'
          setRendererStatus('ready')
          setOperationsReady(true)
        }
      } catch (error) {
        if (
          !disposed &&
          ownGeneration === generation &&
          !(error instanceof DOMException && error.name === 'AbortError')
        ) {
          console.error('[lunar-city] renderer startup failed', error)
          canvas.dataset.worldStatus = 'unavailable'
          setRendererStatus('unavailable')
          setOperationsReady(true)
        }
      }
    }

    const onContextLost = (event: Event): void => {
      event.preventDefault()

      if (disposed) {
        return
      }

      if (restorationArmedGeneration !== undefined) {
        return
      }

      canvas.dataset.worldStatus = 'restoring'
      setRendererStatus('degraded')
      generation += 1
      retireWorld()

      if (restorationAttempted) {
        canvas.dataset.worldStatus = 'unavailable'
        setRendererStatus('unavailable')

        return
      }

      restorationArmedGeneration = generation
    }

    const onContextRestored = (): void => {
      const armedGeneration = restorationArmedGeneration

      if (disposed || armedGeneration === undefined || armedGeneration !== generation || restorationAttempted) {
        return
      }

      restorationArmedGeneration = undefined
      restorationAttempted = true
      void createWorldGeneration()
    }

    canvas.addEventListener('webglcontextlost', onContextLost)
    canvas.addEventListener('webglcontextrestored', onContextRestored)
    rendererStartTimeout = window.setTimeout(() => {
      if (disposed || generation === 0 || worldHandleRef.current || canvas.dataset.worldStatus !== 'loading') {
        return
      }

      // Invalidate the pending generation so a late Babylon import cannot
      // attach a half-created world after we have surfaced the fallback.
      generation += 1
      abortController.abort()
      canvas.dataset.worldStatus = 'unavailable'
      setRendererStatus('unavailable')
      setOperationsReady(true)
    }, RENDERER_START_TIMEOUT_MS)
    void createWorldGeneration()

    return () => {
      disposed = true
      generation += 1
      abortController.abort()
      canvas.removeEventListener('webglcontextlost', onContextLost)
      canvas.removeEventListener('webglcontextrestored', onContextRestored)
      disposeLunarCityRuntime(stopReconcilerRef.current, stopSnapshot, world)
      perfRoute?.dispose()
      perfRoute = undefined

      if (rendererStartTimeout !== undefined) {
        window.clearTimeout(rendererStartTimeout)
        rendererStartTimeout = undefined
      }

      stopSnapshot = undefined
      worldHandleRef.current = undefined
      world = undefined
    }
  }, [])

  return (
    <div className="lunar-city relative flex h-full min-h-0 flex-1 flex-col overflow-hidden bg-(--ui-bg-editor)">
      <header className="lunar-city-hud pointer-events-none absolute inset-x-0 top-0 z-50 flex items-center justify-between gap-4 px-4 py-3 sm:px-6">
        <div className="pointer-events-auto flex min-w-0 items-center gap-3">
          <div className="flex size-9 shrink-0 items-center justify-center rounded-full bg-(--ui-accent)/12 text-(--ui-accent)">
            <Moon aria-hidden="true" size={18} />
          </div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <h2 className="truncate text-sm font-semibold text-foreground">Lunar City</h2>
              <span className="rounded-full bg-(--ui-green)/12 px-2 py-0.5 text-[0.62rem] font-medium text-(--ui-green)">
                <LunarCityHudStatus rendererStatus={rendererStatus} />
              </span>
            </div>
            <p className="truncate text-xs text-muted-foreground">
              Exact Hermes profiles, sessions, workers, and groups
            </p>
          </div>
        </div>
        <div className="pointer-events-auto flex shrink-0 items-center gap-1.5">
          <Button aria-label="Open memory graph" onClick={onOpenMemoryGraph} size="sm" variant="secondary">
            <Network aria-hidden="true" />
            <span className="hidden sm:inline">Memory graph</span>
          </Button>
        </div>
      </header>

      <div
        className="lunar-city-viewport relative min-h-0 flex-1 overflow-hidden"
        data-camera="angled-simcity"
        data-testid="lunar-city-viewport"
      >
        <div className="lunar-city-world absolute inset-[-4%]">
          <canvas
            aria-label="Interactive 3D Lunar City"
            className="lunar-city-terrain absolute inset-0 size-full object-cover object-center"
            ref={canvasRef}
          />
          <div aria-hidden="true" className="lunar-city-atmosphere absolute inset-0" />
        </div>

        <DelegationWorld enabled={rendererStatus === 'ready' && qualityTier !== 'efficient'} onSelect={selectEntity} reducedMotion={reducedMotion} worldRef={worldHandleRef} />
        <LunarCityStatusBadges
          onSelect={selectEntity}
          qualityTier={qualityTier}
          rendererStatus={rendererStatus}
          worldRef={worldHandleRef}
        />

        <div className="pointer-events-none absolute inset-0 z-20">
          {selectedEntityKey ? (
            <LunarCityCommandRuntime onOpenSession={onOpenEntitySession} selectedEntityKey={selectedEntityKey} />
          ) : null}

          {operationsReady ? (
            <LunarCityOperations
        getCameraOrder={() => {
                const world = worldHandleRef.current

                return typeof world?.getEntityCameraOrder === 'function' ? world.getEntityCameraOrder() : []
              }}
        manifest={worldManifest}
              onQualityChange={changeQuality}
              onSelect={selectEntity}
              qualityTier={qualityTier}
              reducedMotion={reducedMotion}
              rendererStatus={rendererStatus === 'loading' ? 'degraded' : rendererStatus}
              selectedEntityKey={selectedEntityKey}
              worldRef={worldHandleRef}
            />
          ) : null}

          <div className="pointer-events-auto absolute left-3 top-20 flex items-center gap-1.5 sm:left-5 sm:top-24">
            <CameraControls dispatch={dispatchCamera} focusedEntityLabel={focusedEntityLabel} state={cameraState} />
            <InteriorCutawayControls focusedEntityKey={cameraState.focusedEntityKey} world={worldHandleRef.current} />
          </div>

          {profileLeaderEntities.length > 0 ? (
            <section
              aria-label={leaderDisambiguationId ? `Choose exact ${leaderDisambiguationId} profile` : 'Profile leaders'}
              className="pointer-events-auto absolute left-3 top-34 z-30 flex max-w-[min(20rem,calc(100%-1.5rem))] flex-wrap gap-1.5 sm:left-5 sm:top-40"
            >
              {!leaderDisambiguationId
                ? ambiguousLeaderModels.map(([leaderId, count]) => (
                    <Button
                      aria-label={`Select ${leaderId} leader model with ${count} exact profiles`}
                      key={`model:${leaderId}`}
                      onClick={() => selectPhysicalLeaderModel(leaderId)}
                      size="xs"
                      variant="secondary"
                    >
                      Choose {leaderId} leader ({count})
                    </Button>
                  ))
                : null}
              {profileLeaderEntities
                .filter(entity => {
                  const owner = leaderOwnerForProfile(entity)!

                  return (
                    !leaderDisambiguationId ||
                    leaderModelIdForOwner(owner, leaderModelIdsRef.current) === leaderDisambiguationId
                  )
                })
                .map(entity => {
                  const owner = leaderOwnerForProfile(entity)!
                  const selected = leaderOwnerKey(owner) === selectedLeaderKey

                  return (
                    <Button
                      aria-label={`Talk to ${owner.profile} leader on ${owner.connectionId}`}
                      key={entity.key}
                      onClick={() => openLeader(entity)}
                      size="xs"
                      variant={selected ? 'default' : 'secondary'}
                    >
                      Talk to {owner.profile} · {owner.connectionId}
                    </Button>
                  )
                })}
            </section>
          ) : null}

          {selectedLeaderOwner && leaderSession ? (
            <div className="pointer-events-auto absolute right-3 top-20 z-40 w-[min(27rem,calc(100%-1.5rem))] sm:right-5 sm:top-24">
              <LeaderDialogueRuntime
                clips={
                  selectedLeaderModel ? worldHandleRef.current?.leaderStateClips?.get?.(selectedLeaderModel) : undefined
                }
                key={`${selectedLeaderKey}::${leaderSession.storedId}::${leaderSession.runtimeId}`}
                leaderLabel={`${selectedLeaderOwner.profile} leader`}
                onClose={closeLeaderDialogue}
                onLeaderStateChange={onLeaderStateChange}
                onOpenFullChat={openSelectedLeaderFullChat}
                onPerfScenarioReady={run => {
                  leaderPerfScenarioRef.current = run
                }}
                owner={selectedLeaderOwner}
                session={leaderSession}
                voiceAvailable={voiceAvailable}
              />
            </div>
          ) : null}

          {selectedLeaderOwner && leaderSessionError ? (
            <section
              aria-label={`${selectedLeaderOwner.profile} leader conversation error`}
              className="pointer-events-auto absolute right-3 top-20 z-40 w-[min(27rem,calc(100%-1.5rem))] rounded-xl border border-(--ui-red)/35 bg-background/95 p-4 text-sm shadow-xl backdrop-blur-xl sm:right-5 sm:top-24"
              role="status"
            >
              <p className="font-medium text-foreground">Unable to open {selectedLeaderOwner.profile} leader</p>
              <p className="mt-1 text-xs text-muted-foreground">{leaderSessionError}</p>
              <Button className="mt-3" onClick={closeLeaderDialogue} size="xs" variant="secondary">
                Close
              </Button>
            </section>
          ) : null}
        </div>
      </div>
    </div>
  )
}
