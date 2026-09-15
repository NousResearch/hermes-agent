/**
 * How the composer decides to commit a draft, and the double-tap window that
 * `double-enter` mode measures against.
 *
 * Main owns the persisted value (a small JSON file under userData, hand-editable
 * like `data-url-read-max.json`); the renderer mirrors it in Settings →
 * Keyboards and clamps optimistically before sending. Both ends have to agree on
 * the default and the bounds, so they live here rather than as two constants
 * with a "keep these in sync" comment between them.
 *
 * `enter` is the historical binding, so it stays the default — an upgrade must
 * never change what Enter does to someone who never opens Settings.
 */

export const COMPOSER_SEND_MODES = ['enter', 'double-enter', 'pause', 'mod-enter'] as const

/** - `enter` — Enter sends; Shift+Enter breaks the line.
 *  - `double-enter` — Enter breaks the line; tapping it twice in a row sends.
 *  - `pause` — Enter breaks the line while you're mid-flow, and sends once you
 *    have stopped typing (held briefly, see `sendGraceMs`, so a wrong guess is
 *    cancellable instead of destructive).
 *  - `mod-enter` — Enter only ever breaks the line; ⌘/Ctrl+Enter sends.
 *
 *  `hold` is deliberately NOT a mode: it is the `sendOnHold` flag below, so it
 *  rides ALONGSIDE whichever mode is chosen. "Hold to send" is an extra way out
 *  of the composer, not a different opinion about what a tap means — a user who
 *  likes double-tap should not have to give it up to get the long press. */
export type ComposerSendMode = (typeof COMPOSER_SEND_MODES)[number]

export const COMPOSER_SEND_DEFAULT_MODE: ComposerSendMode = 'enter'

/** A second plain Enter inside this window sends (`double-enter`, and the
 *  mid-flow half of `pause`). 400ms is the usual double-click allowance: long
 *  enough for a deliberate double-tap, short enough to be over before someone
 *  who just broke a line types again. */
export const DOUBLE_ENTER_DEFAULT_MS = 400

/** Below this, a fast double-tap starts reading as two separate presses; above
 *  it, breaking two lines in a row starts sending. */
export const DOUBLE_ENTER_MIN_MS = 120
export const DOUBLE_ENTER_MAX_MS = 1500

/** `sendOnHold` only: how long the key has to stay down before the press becomes a
 *  send. The TIMER is the threshold, not the operating system's key repeat —
 *  otherwise it could not be set to a precise value and the gesture would stop
 *  firing entirely for anyone with key repeat switched off. Comfortably under
 *  the OS default delay so a hold reads as deliberate without feeling long. */
export const HOLD_DEFAULT_MS = 350
export const HOLD_MIN_MS = 150
export const HOLD_MAX_MS = 1500

/** `pause` only: how long the composer must go without typing before a bare
 *  Enter means "I'm done" rather than "new line". Comfortably above the gap
 *  between two keystrokes of a sentence, below a glance away and back. */
export const TYPING_IDLE_DEFAULT_MS = 1000
export const TYPING_IDLE_MIN_MS = 400
export const TYPING_IDLE_MAX_MS = 5000

/**
 * Which sends wait for the grace window so Esc can take them back.
 *
 * Per-situation rather than a three-way scope, because the situations are not
 * interchangeable: a held key is a DELIBERATE act and should not pay a delay
 * someone only wanted on the send the app guessed. One entry per way a draft
 * can be committed:
 *
 * - `enter` — the default mode's bare Enter.
 * - `doubleTap` — the second of two fast presses.
 * - `pause` — the single Enter after a typing pause: the one send Hermes works
 *   out on the user's behalf, and the default.
 * - `hold` — the long press.
 *
 * A modifier chord is never in this list and never will be: you cannot hit
 * ⌘Enter by accident, and there is nothing to take back.
 */
export const SEND_GRACE_REASONS = ['enter', 'doubleTap', 'pause', 'hold'] as const
export type SendGraceReason = (typeof SEND_GRACE_REASONS)[number]

export const SEND_GRACE_DEFAULT_REASONS: readonly SendGraceReason[] = ['pause']

/** How long a held send waits before it goes. */
export const SEND_GRACE_DEFAULT_MS = 900
export const SEND_GRACE_MIN_MS = 0
export const SEND_GRACE_MAX_MS = 5000

export interface ComposerSendPrefs {
  mode: ComposerSendMode
  doubleEnterMs: number
  /** Independent of `mode`: holding Enter down sends, wherever a bare Enter
   *  does not already send on the press. Composes with every mode but `enter`,
   *  where the press has already committed by the time a hold could register. */
  sendOnHold: boolean
  holdMs: number
  typingIdleMs: number
  /** Which sends wait for the grace window (see `SEND_GRACE_REASONS`). */
  sendGraceFor: readonly SendGraceReason[]
  sendGraceMs: number
}

export function isComposerSendMode(value: unknown): value is ComposerSendMode {
  return COMPOSER_SEND_MODES.includes(value as ComposerSendMode)
}

/** Keep only the known situations, in canonical order, so a hand-edited file
 *  cannot introduce a duplicate or an unknown entry. */
function normalizeGraceReasons(record: Record<string, unknown>): readonly SendGraceReason[] {
  const stored = record.sendGraceFor

  if (Array.isArray(stored)) {
    return SEND_GRACE_REASONS.filter(reason => stored.includes(reason))
  }

  // Legacy three-way scope, from before this became per-situation. `inferred`
  // was the scope that covered the guessed send only — a held key was never
  // inferred, so it does not come along.
  if (record.sendGrace === 'all') {
    return [...SEND_GRACE_REASONS]
  }

  if (record.sendGrace === 'off') {
    return []
  }

  return SEND_GRACE_DEFAULT_REASONS
}

export function clampDoubleEnterMs(value: unknown): number {
  const parsed = Number(value)

  if (!Number.isFinite(parsed)) {
    return DOUBLE_ENTER_DEFAULT_MS
  }

  return Math.min(DOUBLE_ENTER_MAX_MS, Math.max(DOUBLE_ENTER_MIN_MS, Math.round(parsed)))
}

export function clampHoldMs(value: unknown): number {
  const parsed = Number(value)

  if (!Number.isFinite(parsed)) {
    return HOLD_DEFAULT_MS
  }

  return Math.min(HOLD_MAX_MS, Math.max(HOLD_MIN_MS, Math.round(parsed)))
}

export function clampTypingIdleMs(value: unknown): number {
  const parsed = Number(value)

  if (!Number.isFinite(parsed)) {
    return TYPING_IDLE_DEFAULT_MS
  }

  return Math.min(TYPING_IDLE_MAX_MS, Math.max(TYPING_IDLE_MIN_MS, Math.round(parsed)))
}

export function clampSendGraceMs(value: unknown): number {
  const parsed = Number(value)

  if (!Number.isFinite(parsed)) {
    return SEND_GRACE_DEFAULT_MS
  }

  return Math.min(SEND_GRACE_MAX_MS, Math.max(SEND_GRACE_MIN_MS, Math.round(parsed)))
}

/** Coerce anything read off disk (or off the IPC bridge) into valid prefs. An
 *  unparseable mode falls back to the default; an out-of-range window clamps. */
export function normalizeComposerSendPrefs(value: unknown): ComposerSendPrefs {
  const record = (value ?? {}) as Omit<Partial<ComposerSendPrefs>, 'mode'> & { mode?: unknown }
  // `mode: 'hold'` predates the flag — it was briefly a mode of its own before
  // becoming something that rides alongside one. Preserve its behaviour exactly
  // rather than falling back to the default: a bare Enter only ever breaks the
  // line, and the long press sends.
  const legacyHold = record.mode === 'hold'

  return {
    mode: legacyHold ? 'mod-enter' : isComposerSendMode(record.mode) ? record.mode : COMPOSER_SEND_DEFAULT_MODE,
    doubleEnterMs: clampDoubleEnterMs(record.doubleEnterMs ?? DOUBLE_ENTER_DEFAULT_MS),
    holdMs: clampHoldMs(record.holdMs ?? HOLD_DEFAULT_MS),
    sendOnHold: legacyHold || record.sendOnHold === true,
    typingIdleMs: clampTypingIdleMs(record.typingIdleMs ?? TYPING_IDLE_DEFAULT_MS),
    sendGraceFor: normalizeGraceReasons(record as Record<string, unknown>),
    sendGraceMs: clampSendGraceMs(record.sendGraceMs ?? SEND_GRACE_DEFAULT_MS)
  }
}
