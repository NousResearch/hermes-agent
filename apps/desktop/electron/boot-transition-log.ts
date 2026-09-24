/**
 * boot-transition-log.ts
 *
 * Pure formatter for the main-process boot-progress transition log (#96743,
 * item 3). Kept dependency-free so the unit test does not boot Electron.
 *
 * The renderer's boot overlay is driven entirely by `updateBootProgress()`
 * in main.ts (push over `hermes:boot-progress` + pull via
 * `hermes:boot-progress:get`). Before #96743 the only boot logging was a
 * free-form `[boot] <message>` line, gated on `update.message` being set —
 * silent for pure phase transitions like `backend.remote → backend.ready`,
 * which is exactly the "9-minute silent gap" the issue reports. This helper
 * produces a stable one-line format so desktop.log shows every main→renderer
 * state transition, even when the update carries no message.
 */

export interface BootTransitionSlice {
  error?: null | string
  phase: string
  running?: boolean
}

/**
 * Returns the log line to persist, or null when nothing user-visible changed
 * (pure progress-percentage ticks stay quiet to avoid log spam).
 *
 * `options.delivered === false` marks the case where the state changed but the
 * renderer could not be told (window gone or destroyed) — the transition still
 * happened in main, and the log must say so, since those are exactly the
 * transitions that produce "main says ready, UI stuck" confusion.
 */
export function formatBootTransitionLog(
  prev: BootTransitionSlice,
  next: BootTransitionSlice,
  options: { delivered?: boolean } = {}
): null | string {
  const phaseChanged = prev.phase !== next.phase
  const errorChanged = (prev.error ?? null) !== (next.error ?? null)

  if (!phaseChanged && !errorChanged) {
    return null
  }

  const base = `[boot] transition ${prev.phase} -> ${next.phase}${next.running === false ? ' (settled)' : ''}`
  const withError = next.error ? `${base} error=${JSON.stringify(next.error)}` : base

  return options.delivered === false ? `${withError} not-delivered` : withError
}

/**
 * Sends the boot-progress payload to the renderer, swallowing the throw that
 * occurs when the window dies between the caller's isDestroyed() checks and
 * the send itself. Returns true only when the send actually went out.
 *
 * The whole point of the transition log is to record what happened when the
 * renderer *didn't* see it — so a destroyed-mid-send window must report
 * not-delivered, never skip the log line (#96743).
 */
export function trySendBootProgress(sender: { send(channel: string, payload: unknown): void }, channel: string, payload: unknown): boolean {
  try {
    sender.send(channel, payload)

    return true
  } catch {
    return false
  }
}
