import { atom, computed } from 'nanostores'

import { $workspaceMode, $workspaceNewSessionTarget } from '@/components/pane-shell/workspace-scope'
import { persistBoolean, storedBoolean } from '@/lib/storage'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'
import { $busy } from '@/store/session'

/**
 * Petdex mascot state for the desktop floating pet.
 *
 * The spritesheet payload comes from the gateway `pet.info` RPC (shared with
 * the TUI). The animation *state* is derived here from the same activity
 * signals the chat already tracks, mirroring the priority order documented in
 * `agent/pet/state.py` so the Python and TS surfaces never drift.
 */

export type PetState = 'idle' | 'wave' | 'run' | 'failed' | 'review' | 'jump' | 'waiting'

export interface PetInfo {
  enabled: boolean
  slug?: string
  displayName?: string
  mime?: string
  spritesheetBase64?: string
  // Stable sheet revision (`mtime_ns:size`) from the gateway; lets the desktop
  // skip full sprite payload refreshes when the active pet hasn't changed.
  spritesheetRevision?: string
  frameW?: number
  frameH?: number
  framesPerState?: number
  // Real (padding-trimmed) frame count per state row, from the engine. Lets the
  // canvas step only frames that exist instead of a fixed framesPerState, which
  // would animate into the transparent padding of ragged sheets (blank flash).
  framesByState?: Record<string, number>
  // Concrete Codex row counts (e.g. running-right may have 8 frames even though
  // the Hermes "run" activity state uses the in-place running row).
  framesByRow?: Record<string, number>
  loopMs?: number
  scale?: number
  stateRows?: string[]
}

export interface PetInfoMeta {
  enabled: boolean
  slug?: string
  displayName?: string
  scale?: number
  spritesheetRevision?: string
}

export function hasPetSpriteForMeta(info: PetInfo, meta: PetInfoMeta): boolean {
  return (
    meta.enabled &&
    info.enabled &&
    Boolean(info.spritesheetBase64) &&
    info.slug === meta.slug &&
    Boolean(info.spritesheetRevision) &&
    info.spritesheetRevision === meta.spritesheetRevision
  )
}

export function mergePetInfoMeta(info: PetInfo, meta: PetInfoMeta): PetInfo {
  if (!meta.enabled) {
    return info.enabled ? { enabled: false } : info
  }

  // Fast-path: nothing changed — return the same reference so callers can
  // skip the store update (nanostores fires on .set() regardless of deep
  // equality; returning `info` avoids a redundant re-render on every poll).
  if (
    info.enabled &&
    info.slug === meta.slug &&
    info.displayName === meta.displayName &&
    info.scale === meta.scale &&
    info.spritesheetRevision === meta.spritesheetRevision
  ) {
    return info
  }

  return {
    ...info,
    enabled: true,
    slug: meta.slug,
    displayName: meta.displayName,
    scale: meta.scale,
    spritesheetRevision: meta.spritesheetRevision
  }
}

export interface PetActivity {
  busy?: boolean
  awaitingInput?: boolean
  toolRunning?: boolean
  reasoning?: boolean
  error?: boolean
  justCompleted?: boolean
  celebrate?: boolean
}

/**
 * Resolve the animation state from coarse activity signals.
 *
 * Priority (highest first) mirrors `agent.pet.state.derive_pet_state`:
 * error → celebrate → justCompleted → awaitingInput → toolRunning → reasoning →
 * busy → idle. `awaitingInput` (a clarify/approval blocking on the user) outranks
 * the in-flight signals because the turn is paused on you, not working.
 */
export function derivePetState(activity: PetActivity): PetState {
  if (activity.error) {
    return 'failed'
  }

  if (activity.celebrate) {
    return 'jump'
  }

  if (activity.justCompleted) {
    return 'wave'
  }

  if (activity.awaitingInput) {
    return 'waiting'
  }

  if (activity.toolRunning) {
    return 'run'
  }

  if (activity.reasoning) {
    return 'review'
  }

  if (activity.busy) {
    return 'run'
  }

  return 'idle'
}

export const $petInfo = atom<PetInfo>({ enabled: false })
export const $petActivity = atom<PetActivity>({})

/** Pet installed + enabled with a loaded spritesheet (ready to show/react). */
export const $petActive = computed($petInfo, info => info.enabled && Boolean(info.spritesheetBase64))

/**
 * Exact backend owner whose pet should be rendered. Pets are per-profile: both
 * `display.pet.*` and installed sprites live under that profile's HERMES_HOME.
 * Bot Mode keeps the Sessions gateway ambient, so the workspace route — not the
 * active gateway — is authoritative while a bot chat is selected.
 */
export interface PetOwner {
  connectionId?: string
  profile: string
  targetProfile: string
}

export const $petOwner = computed(
  [$activeGatewayProfile, $workspaceMode, $workspaceNewSessionTarget],
  (activeGatewayProfile, workspaceMode, workspaceTarget): PetOwner => {
    const activeProfile = normalizeProfileKey(activeGatewayProfile)

    if (workspaceMode === 'bots' && workspaceTarget?.kind === 'route') {
      const route = workspaceTarget.route
      const connectionId = String(route?.connectionId ?? '').trim()
      const routeProfile = String(route?.profile ?? '').trim()

      if (connectionId && routeProfile) {
        const profile = normalizeProfileKey(routeProfile)

        return {
          connectionId,
          profile,
          targetProfile: normalizeProfileKey(route.targetProfile || profile)
        }
      }
    }

    return { profile: activeProfile, targetProfile: activeProfile }
  }
)

export function petOwner(): PetOwner {
  return $petOwner.get()
}

/** Whether a previously rendered owner actually changed. The missing previous
 * owner is initial hydration, not a switch — callers must preserve warm state. */
export function petOwnerChanged(previous: PetOwner | undefined, next: PetOwner): boolean {
  return Boolean(
    previous &&
      (previous.connectionId !== next.connectionId ||
        previous.profile !== next.profile ||
        previous.targetProfile !== next.targetProfile)
  )
}

/** Ambient gateway state/events apply only when no exact Bot owner route exists. */
export function petOwnerUsesAmbientGateway(owner: PetOwner): boolean {
  return !owner.connectionId
}

/** Profile whose config + pet store should answer pet RPCs. */
export function petProfile(): string {
  return petOwner().targetProfile
}

/** Route one pet RPC through the exact Bot owner when there is one. The
 *  target profile remains explicit in params for shared-remote backends. */
export function requestPetForOwner<T>(
  owner: PetOwner,
  method: string,
  params: Record<string, unknown>,
  requestAmbient: (method: string, params: Record<string, unknown>) => Promise<T>,
  requestRouted: (
    connectionId: string,
    profile: string,
    method: string,
    params: Record<string, unknown>
  ) => Promise<T>
): Promise<T> {
  const routedParams = { ...params, profile: owner.targetProfile }

  return owner.connectionId
    ? requestRouted(owner.connectionId, owner.profile, method, routedParams)
    : requestAmbient(method, routedParams)
}

/**
 * Pet-local "you have a new message" flag, surfaced as the overlay's mail icon.
 * Deliberately not real unread tracking: it flips on when a turn finishes while
 * the app isn't focused, and off when the user opens the app via the mail icon
 * (or returns to the window). No persistence — it's a glance hint, not state.
 */
export const $petUnread = atom(false)
export const markPetUnread = () => $petUnread.set(true)
export const clearPetUnread = () => $petUnread.set(false)

/** Steady activity flags (toolRunning / reasoning) set + cleared by the stream. */
export const setPetActivity = (next: Partial<PetActivity>) => $petActivity.set({ ...$petActivity.get(), ...next })

let flashTimer: ReturnType<typeof setTimeout> | undefined

/** Fire a transient reaction beat (error / celebrate / justCompleted) that
 *  decays back to the steady state after `ms`.
 *
 *  Each beat first clears its siblings so a stale one can't win the priority
 *  race: without this, a completion beat (`celebrate`) would merge on top of a
 *  lingering `error`, and `derivePetState` checks `error` first — so a clean
 *  finish would render the sad/failed pose. */
export const flashPetActivity = (next: Partial<PetActivity>, ms = 1600) => {
  setPetActivity({ celebrate: false, error: false, justCompleted: false, ...next })
  clearTimeout(flashTimer)
  flashTimer = setTimeout(() => setPetActivity({ celebrate: false, error: false, justCompleted: false }), ms)
}

export const setPetInfo = (info: PetInfo) => $petInfo.set(info)

/**
 * Resolve the live activity state from the dedicated activity atom, falling back
 * to the always-present `$busy` chat signal so the pet reacts out of the box.
 *
 * `awaitingInput` (a clarify/approval blocking on the user) is an explicit flag
 * on `$petActivity` — set by the controller from `$attentionSessionIds` and
 * mirrored to the pop-out overlay through the same atom, so both surfaces agree
 * without the overlay needing the session list.
 */
function deriveLivePetState(activity: PetActivity, busy: boolean): PetState {
  const live = activity.busy ?? busy

  return derivePetState({
    busy: live,
    awaitingInput: activity.awaitingInput,
    // Steady flags only count mid-turn — ignore stale ones once at rest so an
    // interrupted turn can't pin the pet on `run`/`review`.
    toolRunning: live && activity.toolRunning,
    reasoning: live && activity.reasoning,
    error: activity.error,
    justCompleted: activity.justCompleted,
    celebrate: activity.celebrate
  })
}

/**
 * Opt-in: let the floating mascot wander around the window on its own while
 * idle. Pure desktop-client behavior (no agent/config dependency), so it lives
 * in localStorage like the pet's drag position — per-device, not per-profile.
 */
const ROAM_KEY = 'hermes.desktop.pet-roam.v1'
export const $petRoam = atom<boolean>(storedBoolean(ROAM_KEY, false))

export const setPetRoam = (on: boolean) => {
  $petRoam.set(on)
  persistBoolean(ROAM_KEY, on)
}

/**
 * The pose the roam loop is currently driving: `run` while walking a surface,
 * `jump` while hopping/falling between surfaces, or `null` at rest. Surfaced
 * through `$petState` (below) so the canvas animates the wander without any prop
 * change or re-render — it already subscribes to `$petState`.
 */
export const $petMotion = atom<PetState | null>(null)

/**
 * Horizontal travel direction while roaming: -1 left, 1 right, 0 not walking.
 * The floating pet maps this to the directional run row + mirror, keeping the
 * wander loop free of sprite-row knowledge.
 */
export const $petRoamDir = atom<-1 | 0 | 1>(0)

/**
 * Whether the agent-driven state is at rest (plain `idle`). The roam loop gates
 * on this — never on `$petState` itself, which would feed back on its own
 * `$petMotion`-driven pose and stall the wander.
 */
export const $petAtRest = computed(
  [$petActivity, $busy],
  (activity, busy): boolean => deriveLivePetState(activity, busy) === 'idle'
)

/**
 * The live pet state. Activity always wins; only when the agent is at rest does
 * a roam pose (walking → `run`, hopping → `jump`) show through, so the wander
 * reads as deliberate movement.
 */
export const $petState = computed([$petActivity, $busy, $petMotion], (activity, busy, motion): PetState => {
  const base = deriveLivePetState(activity, busy)

  return base === 'idle' && motion ? motion : base
})
