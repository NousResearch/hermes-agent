import { atom, computed } from 'nanostores'

import { PRIMARY_SESSION_VIEW } from '@/app/chat/session-view'
import { roamHopRow, roamWalkRow } from '@/components/pet/pet-rows'
import { persistBoolean, storedBoolean } from '@/lib/storage'
import { $activeGatewayProfile, normalizeProfileKey } from '@/store/profile'

/**
 * Petdex mascot state for the desktop floating pet.
 *
 * The spritesheet payload comes from the gateway `pet.info` RPC (shared with
 * the TUI). The animation *state* is derived here from the same activity
 * signals the chat already tracks, mirroring the priority order documented in
 * `agent/pet/state.py` so the Python and TS surfaces never drift.
 */

// The user handling the in-window pet: a click (`poke`, or `wake` on a sleeping
// pet), carrying it (`held`), dropping from height (`fall`), touchdown (`land`).
const HANDLING_STATES = ['poke', 'wake', 'held', 'fall', 'land'] as const
export type PetHandlingState = (typeof HANDLING_STATES)[number]

// Plugin code is untyped at runtime: `react()` only accepts these names.
const PET_STATES = ['idle', 'wave', 'run', 'failed', 'review', 'jump', 'waiting', 'sleep', ...HANDLING_STATES] as const
export type PetState = (typeof PET_STATES)[number]

/** What a plugin hears through `ctx.pet.onInteract`. */
export type PetInteraction = 'poke' | 'pickup' | 'drop' | 'land'

/** Context passed with each `ctx.pet.onInteract` event. */
export interface PetInteractInfo {
  /** The pet was asleep when touched (any interaction wakes it). */
  asleep: boolean
}

/** Per-pet behaviors from `pet.json` (mirrors `agent.pet.behaviors.PetSpec.payload`); absent when it declares none. */
export interface PetBehaviors {
  /** State → the rows it plays; several rotate, each held `holdSeconds[state]` (default 30). */
  rows?: Partial<Record<PetState, string[]>>
  sleepAfterSeconds?: number
  holdSeconds?: Partial<Record<PetState, number>>
}

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
  behaviors?: PetBehaviors
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
  /** At rest past the pet's declared sleep delay (only pets whose pet.json opts in). */
  sleepy?: boolean
}

/**
 * Resolve the animation state from coarse activity signals.
 *
 * Priority (highest first) mirrors `agent.pet.state.derive_pet_state`:
 * error → celebrate → justCompleted → awaitingInput → toolRunning → reasoning →
 * busy → sleepy → idle. `awaitingInput` (a clarify/approval blocking on the user) outranks
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

  return activity.sleepy ? 'sleep' : 'idle'
}

export const $petInfo = atom<PetInfo>({ enabled: false })
export const $petActivity = atom<PetActivity>({})

/** Pet installed + enabled with a loaded spritesheet (ready to show/react). */
export const $petActive = computed($petInfo, info => info.enabled && Boolean(info.spritesheetBase64))

/**
 * Profile the pet RPCs should resolve against. Pets are per-profile — the active
 * pet (`display.pet.*`) and the installed sprites live under each profile's
 * HERMES_HOME — so every pet RPC carries this. The gateway no-ops it for the
 * launch profile (own-profile backends already resolve it) and rebinds for any
 * other profile, which is what makes per-profile pets work in app-global remote
 * mode (one backend serving every profile).
 */
export function petProfile(): string {
  return normalizeProfileKey($activeGatewayProfile.get())
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
 * to the primary session view's turn-busy so the pet reacts out of the box.
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
    celebrate: activity.celebrate,
    sleepy: activity.sleepy
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
// Turn-busy comes from the active runtime's own slice (`$primaryBusy` in
// session-view.tsx explains why the global `$busy` mirror can't be trusted:
// #84434 / #84438). The pop-out overlay push reads the same atom so both
// surfaces derive the pose from one signal.
export const $petAtRest = computed(
  [$petActivity, PRIMARY_SESSION_VIEW.$busy],
  (activity, busy): boolean => deriveLivePetState(activity, busy) === 'idle'
)

/** The live state with the nap ignored: what `startPetNap` arms on (notifies only when it changes). */
export const $petAwakeState = computed([$petActivity, PRIMARY_SESSION_VIEW.$busy], (activity, busy) =>
  deriveLivePetState({ ...activity, sleepy: false }, busy)
)

/** The handling phase the user has the floating pet in, or null. Purely visual: it never reaches `$petState`. */
export const $petHandling = atom<PetHandlingState | null>(null)

let handlingTimer: ReturnType<typeof setTimeout> | undefined

/** Play a handling state (`ms` makes it a one-shot), or clear it with null. */
export const setPetHandling = (state: PetHandlingState | null, ms?: number) => {
  clearTimeout(handlingTimer)
  $petHandling.set(state)

  if (state && ms) {
    handlingTimer = setTimeout(() => $petHandling.set(null), ms)
  }
}

type PetInteractListener = (kind: PetInteraction, info: PetInteractInfo) => void

const interactListeners = new Set<PetInteractListener>()

/** Hear the user handle the pet. Returns an unsubscribe. */
export function onPetInteract(listener: PetInteractListener): () => void {
  interactListeners.add(listener)

  return () => void interactListeners.delete(listener)
}

/** The pet's pet.json maps `state` to a row its sheet has (the gateway drops the rest). */
export const petMaps = (state: PetState, info = $petInfo.get()): boolean =>
  Boolean(info.behaviors?.rows?.[state]?.length)

const INTERACT_STATES: Record<PetInteraction, PetHandlingState | null> = {
  poke: 'poke',
  pickup: 'held',
  drop: null,
  land: 'land'
}

/**
 * The user touched the floating pet: record the phase's handling state (`state`
 * overrides the table, e.g. `fall` for a drop that falls; a poke on a sleeping
 * pet that maps `wake` plays that) and tell plugins whether it was asleep. A
 * throwing plugin listener can't break the drag.
 */
export function petInteract(kind: PetInteraction, state = INTERACT_STATES[kind]): void {
  const asleep = $petState.get() === 'sleep'
  const oneShot = kind === 'poke' || kind === 'land'

  setPetHandling(
    kind === 'poke' && asleep && petMaps('wake') ? 'wake' : state,
    oneShot ? ($petInfo.get().loopMs ?? 1100) : undefined
  )

  for (const listener of [...interactListeners]) {
    try {
      listener(kind, { asleep })
    } catch (error) {
      console.error('[plugins] pet interaction listener failed', error)
    }
  }
}

/** A plugin's brief reaction (`ctx.pet.react`): the latest call wins. */
export interface PetPluginReaction {
  owner: string
  state: PetState
}

export const $petPluginReaction = atom<PetPluginReaction | null>(null)
const PET_REACT_MAX_MS = 10_000

let reactionTimer: ReturnType<typeof setTimeout> | undefined

const endReaction = (matches: (r: PetPluginReaction) => boolean) => {
  const current = $petPluginReaction.get()

  if (current && matches(current)) {
    clearTimeout(reactionTimer)
    $petPluginReaction.set(null)
  }
}

/** Play `state`'s row for `ms` (default one animation loop, capped). Returns a cancel. */
export function reactPet(owner: string, state: PetState, ms?: number): () => void {
  if (!(PET_STATES as readonly string[]).includes(state)) {
    return () => {}
  }

  const reaction = { owner, state }
  const end = () => endReaction(r => r === reaction)

  clearTimeout(reactionTimer)
  $petPluginReaction.set(reaction)
  reactionTimer = setTimeout(end, Math.min(PET_REACT_MAX_MS, ms ?? $petInfo.get().loopMs ?? 1100))

  return end
}

/** Drop `owner`'s reaction if it's the one playing (plugin unload). */
export const clearPetReaction = (owner: string) => endReaction(r => r.owner === owner)

/**
 * The live pet state. Activity always wins; only when the agent is at rest does
 * a roam pose (walking → `run`, hopping → `jump`) show through, so the wander
 * reads as deliberate movement.
 */
export const $petState = computed(
  [$petActivity, PRIMARY_SESSION_VIEW.$busy, $petMotion],
  (activity, busy, motion): PetState => {
    const base = deriveLivePetState(activity, busy)

    return base === 'idle' && motion ? motion : base
  }
)

// Failed / waiting-on-you need the user, and a sleeping pet stays asleep until
// something real wakes it: a plugin reaction never shows over them.
const REACTION_BLOCKED: ReadonlySet<PetState> = new Set<PetState>(['failed', 'waiting', 'sleep'])

const $petShownReaction = computed([$petState, $petPluginReaction], (state, reaction) =>
  reaction && !REACTION_BLOCKED.has(state) ? reaction.state : null
)

/** The user is handling the pet or a reaction plays: the wander holds still, so it only moves on a walk / hop row. */
export const $petPosing = computed(
  [$petHandling, $petShownReaction],
  (handling, reaction) => handling !== null || reaction !== null
)

/** What the canvas draws: a concrete `row` (else `state`'s), and the travel facing (`undefined` at rest). */
export interface PetSpriteTarget {
  mirror?: boolean
  row?: string
  state: PetState
}

/**
 * The one resolved sprite target, highest first: a handling state the pet maps
 * (the user is holding it), a plugin reaction, the roam's walk / hop row, the
 * live state. A pose while travelling faces the travel direction (the art faces
 * left, so rightward mirrors); a directional walk row brings its own mirror.
 */
export const $petSpriteTarget = computed(
  [$petState, $petHandling, $petShownReaction, $petInfo, $petRoamDir, $petMotion],
  (state, handling, reaction, info, dir, motion): PetSpriteTarget => {
    const pose = (handling && petMaps(handling, info) ? handling : null) ?? reaction

    if (pose) {
      return { mirror: dir ? dir > 0 : undefined, state: pose }
    }

    const walk = roamWalkRow(dir, info.stateRows)

    return { mirror: dir ? walk.mirror : undefined, row: walk.row ?? roamHopRow(motion, info.stateRows), state }
  }
)
