import { atom } from 'nanostores'

/**
 * `display.busy_input_mode` — what a submission typed while the agent is
 * running should do. One config key shared with the CLI, the TUI and the
 * messaging gateway; the desktop used to ignore it entirely and hard-wire
 * every busy Enter to an active-turn redirect.
 *
 * The three modes, spelled the same way everywhere:
 *
 * - `interrupt` — redirect the live model request (stop-and-correct). The
 *   gateway decides atomically whether the agent is generating, executing a
 *   tool, or too old to redirect; a redirect it cannot perform falls back to
 *   a queue, so the words still reach the model.
 * - `steer` — inject the text into the model's next tool result WITHOUT
 *   cancelling the current turn. No new user turn, no role alternation.
 * - `queue` — hold the text and run it as the next turn.
 *
 * `interrupt` is the framework default (hermes_cli/config_defaults.py), so an
 * absent or malformed value keeps the historical desktop behaviour instead of
 * silently changing what Enter does for everyone else.
 */

export type BusyInputMode = 'interrupt' | 'queue' | 'steer'
export type BusyComposerAction = 'redirect' | 'queue' | 'steer' | 'stop'

const BUSY_INPUT_MODES: ReadonlySet<BusyInputMode> = new Set<BusyInputMode>(['interrupt', 'queue', 'steer'])

/** Coerce a config.yaml value into a mode; unknown values fall back to `interrupt`. */
export const normalizeBusyInputMode = (value: unknown): BusyInputMode => {
  if (typeof value !== 'string') {
    return 'interrupt'
  }

  const mode = value.trim().toLowerCase() as BusyInputMode

  return BUSY_INPUT_MODES.has(mode) ? mode : 'interrupt'
}

/**
 * The action the composer should take for a busy Enter, so the visible
 * affordance (Send vs Queue button, its tooltip and the keyboard-shortcuts
 * page) and the submit path resolve from the SAME decision.
 *
 * `redirect` and `steer` both require a live, text-only, non-slash payload:
 * attachments cannot ride either RPC (no tool-result image carriage) and
 * slash commands execute locally instead of reaching the model. A turn parked
 * on an approval/sudo/secret prompt also rules both out — the tool batch is
 * blocked on the user, so neither can reach the model and the text queues
 * behind it (steering there would sit undelivered until the prompt times out).
 */
export function resolveBusyComposerAction({
  busy,
  canCorrect,
  compacting,
  hasPayload,
  blockingPrompt,
  mode
}: {
  busy: boolean
  /** Both corrections are gated by the same payload/state preconditions. */
  canCorrect: boolean
  compacting: boolean
  hasPayload: boolean
  blockingPrompt: boolean
  mode: BusyInputMode
}): BusyComposerAction {
  if (!busy) {
    return 'stop'
  }

  // A compaction owns the turn: neither correction is safe, so the payload
  // waits for the next turn and an empty composer stops.
  if (compacting) {
    return hasPayload ? 'queue' : 'stop'
  }

  if (mode === 'queue') {
    return hasPayload ? 'queue' : 'stop'
  }

  if (blockingPrompt) {
    return hasPayload ? 'queue' : 'stop'
  }

  if (!canCorrect) {
    return hasPayload ? 'queue' : 'stop'
  }

  return mode === 'steer' ? 'steer' : 'redirect'
}

/**
 * Whether the configured mode has a carrier to ride. `interrupt` redirects the
 * live turn through `onSteer`; `steer` injects through `onSteerHidden`. Asking
 * for the wrong one would let the UI promise an action the submit path cannot
 * deliver (a steer-mode composer with only `onSteer` wired would read as
 * steer-able and then silently queue on Enter).
 */
export function busyModeHasCarrier(mode: BusyInputMode, carriers: {
  onSteer: boolean
  onSteerHidden: boolean
}): boolean {
  return mode === 'steer' ? carriers.onSteerHidden : carriers.onSteer
}

/**
 * The resolved mode as a plain atom, mirroring the other display.* knobs the
 * composer reads (`store/display-timestamps.ts`, `store/reasoning-disclosure.ts`).
 * Reading it from a store rather than from the config query keeps ChatBar free
 * of a QueryClient dependency — the composer mounts inside panes and test
 * harnesses that only provide the app's nanostore context.
 */
export const $busyInputMode = atom<BusyInputMode>('interrupt')

/** Publish the config value into the atom. Called by the config refresh path. */
export const setBusyInputModeFromConfig = (value: unknown): void => {
  $busyInputMode.set(normalizeBusyInputMode(value))
}
