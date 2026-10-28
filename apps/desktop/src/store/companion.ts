import { atom, computed } from 'nanostores'

import { PRIMARY_SESSION_VIEW } from '@/app/chat/session-view'
import { persistBoolean, storedBoolean } from '@/lib/storage'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'

/**
 * Companiondex mascot state for the desktop floating companion.
 *
 * The spritesheet payload comes from the gateway `companion.info` RPC (shared with
 * the TUI). The animation *state* is derived here from the same activity
 * signals the chat already tracks, mirroring the priority order documented in
 * `agent/companion/state.py` so the Python and TS surfaces never drift.
 */

export type CompanionState = 'idle' | 'wave' | 'run' | 'failed' | 'review' | 'jump' | 'waiting'

export interface CompanionInfo {
  enabled: boolean
  slug?: string
  displayName?: string
  mime?: string
  spritesheetBase64?: string
  // Stable sheet revision (`mtime_ns:size`) from the gateway; lets the desktop
  // skip full sprite payload refreshes when the active companion hasn't changed.
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

export interface CompanionInfoMeta {
  enabled: boolean
  slug?: string
  displayName?: string
  scale?: number
  spritesheetRevision?: string
}

export function hasCompanionSpriteForMeta(info: CompanionInfo, meta: CompanionInfoMeta): boolean {
  return (
    meta.enabled &&
    info.enabled &&
    Boolean(info.spritesheetBase64) &&
    info.slug === meta.slug &&
    Boolean(info.spritesheetRevision) &&
    info.spritesheetRevision === meta.spritesheetRevision
  )
}

export function mergeCompanionInfoMeta(info: CompanionInfo, meta: CompanionInfoMeta): CompanionInfo {
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

export interface CompanionActivity {
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
 * Priority (highest first) mirrors `agent.companion.state.derive_companion_state`:
 * error → celebrate → justCompleted → awaitingInput → toolRunning → reasoning →
 * busy → idle. `awaitingInput` (a clarify/approval blocking on the user) outranks
 * the in-flight signals because the turn is paused on you, not working.
 */
export function deriveCompanionState(activity: CompanionActivity): CompanionState {
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

export const $companionInfo = atom<CompanionInfo>({ enabled: false })
export const $companionActivity = atom<CompanionActivity>({})

/** Companion installed + enabled with a loaded spritesheet (ready to show/react). */
export const $companionActive = computed($companionInfo, info => info.enabled && Boolean(info.spritesheetBase64))

/**
 * Profile the companion RPCs should resolve against. Companions are per-profile — the active
 * companion (`display.companion.*`) and the installed sprites live under each profile's
 * HERMES_HOME — so every companion RPC carries this. The gateway no-ops it for the
 * launch profile (own-profile backends already resolve it) and rebinds for any
 * other profile, which is what makes per-profile companions work in app-global remote
 * mode (one backend serving every profile).
 */
export function companionProfile(): string {
  return normalizeProfileKey($activeGatewayProfile.get())
}

/**
 * Companion-local "you have a new message" flag, surfaced as the overlay's mail icon.
 * Deliberately not real unread tracking: it flips on when a turn finishes while
 * the app isn't focused, and off when the user opens the app via the mail icon
 * (or returns to the window). No persistence — it's a glance hint, not state.
 */
export const $companionUnread = atom(false)
export const markCompanionUnread = () => $companionUnread.set(true)
export const clearCompanionUnread = () => $companionUnread.set(false)

/** Steady activity flags (toolRunning / reasoning) set + cleared by the stream. */
export const setCompanionActivity = (next: Partial<CompanionActivity>) => $companionActivity.set({ ...$companionActivity.get(), ...next })

let flashTimer: ReturnType<typeof setTimeout> | undefined

/** Fire a transient reaction beat (error / celebrate / justCompleted) that
 *  decays back to the steady state after `ms`.
 *
 *  Each beat first clears its siblings so a stale one can't win the priority
 *  race: without this, a completion beat (`celebrate`) would merge on top of a
 *  lingering `error`, and `deriveCompanionState` checks `error` first — so a clean
 *  finish would render the sad/failed pose. */
export const flashCompanionActivity = (next: Partial<CompanionActivity>, ms = 1600) => {
  setCompanionActivity({ celebrate: false, error: false, justCompleted: false, ...next })
  clearTimeout(flashTimer)
  flashTimer = setTimeout(() => setCompanionActivity({ celebrate: false, error: false, justCompleted: false }), ms)
}

export const setCompanionInfo = (info: CompanionInfo) => $companionInfo.set(info)

/**
 * Resolve the live activity state from the dedicated activity atom, falling back
 * to the primary session view's turn-busy so the companion reacts out of the box.
 *
 * `awaitingInput` (a clarify/approval blocking on the user) is an explicit flag
 * on `$companionActivity` — set by the controller from `$attentionSessionIds` and
 * mirrored to the pop-out overlay through the same atom, so both surfaces agree
 * without the overlay needing the session list.
 */
function deriveLiveCompanionState(activity: CompanionActivity, busy: boolean): CompanionState {
  const live = activity.busy ?? busy

  return deriveCompanionState({
    busy: live,
    awaitingInput: activity.awaitingInput,
    // Steady flags only count mid-turn — ignore stale ones once at rest so an
    // interrupted turn can't pin the companion on `run`/`review`.
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
 * in localStorage like the companion's drag position — per-device, not per-profile.
 */
const ROAM_KEY = 'hermes.desktop.companion-roam.v1'
export const $companionRoam = atom<boolean>(storedBoolean(ROAM_KEY, false))

export const setCompanionRoam = (on: boolean) => {
  $companionRoam.set(on)
  persistBoolean(ROAM_KEY, on)
}

/**
 * The pose the roam loop is currently driving: `run` while walking a surface,
 * `jump` while hopping/falling between surfaces, or `null` at rest. Surfaced
 * through `$companionState` (below) so the canvas animates the wander without any prop
 * change or re-render — it already subscribes to `$companionState`.
 */
export const $companionMotion = atom<CompanionState | null>(null)

/**
 * Horizontal travel direction while roaming: -1 left, 1 right, 0 not walking.
 * The floating companion maps this to the directional run row + mirror, keeping the
 * wander loop free of sprite-row knowledge.
 */
export const $companionRoamDir = atom<-1 | 0 | 1>(0)

/**
 * Whether the agent-driven state is at rest (plain `idle`). The roam loop gates
 * on this — never on `$companionState` itself, which would feed back on its own
 * `$companionMotion`-driven pose and stall the wander.
 */
// Turn-busy comes from the active runtime's own slice (`$primaryBusy` in
// session-view.tsx explains why the global `$busy` mirror can't be trusted:
// #84434 / #84438). The pop-out overlay push reads the same atom so both
// surfaces derive the pose from one signal.
export const $companionAtRest = computed(
  [$companionActivity, PRIMARY_SESSION_VIEW.$busy],
  (activity, busy): boolean => deriveLiveCompanionState(activity, busy) === 'idle'
)

/**
 * The live companion state. Activity always wins; only when the agent is at rest does
 * a roam pose (walking → `run`, hopping → `jump`) show through, so the wander
 * reads as deliberate movement.
 */
export const $companionState = computed(
  [$companionActivity, PRIMARY_SESSION_VIEW.$busy, $companionMotion],
  (activity, busy, motion): CompanionState => {
    const base = deriveLiveCompanionState(activity, busy)

    return base === 'idle' && motion ? motion : base
  }
)
