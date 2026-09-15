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

export const COMPOSER_SEND_MODES = ['enter', 'double-enter', 'pause', 'hold', 'mod-enter'] as const

/** - `enter` — Enter sends; Shift+Enter breaks the line.
 *  - `double-enter` — Enter breaks the line; tapping it twice in a row sends.
 *  - `pause` — Enter breaks the line while you're mid-flow, and sends once you
 *    have stopped typing (held briefly, see `sendGraceMs`, so a wrong guess is
 *    cancellable instead of destructive).
 *  - `hold` — Enter breaks the line; holding the key down sends. The press
 *    becomes a send once the operating system's key repeat starts, so the
 *    threshold is the user's own repeat setting rather than a number we invent,
 *    and a stray tap can never send.
 *  - `mod-enter` — Enter only ever breaks the line; ⌘/Ctrl+Enter sends. */
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

/** `hold` only: how long the key has to stay down before the press becomes a
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
 * Which sends get held before they go, so Esc can take them back.
 *
 * - `off` — nothing is held; every send fires on its keystroke.
 * - `inferred` — only the sends the app decided on the user's behalf (the
 *   single Enter after a typing pause). An explicit gesture never waits.
 * - `all` — every send started by a bare Enter, including the default mode's,
 *   so undo-send works for people who never change their binding. Modifier
 *   chords are never held: you cannot hit ⌘Enter or Shift+Enter by accident,
 *   and a deliberate correction should not be delayed.
 */
export const SEND_GRACE_SCOPES = ['off', 'inferred', 'all'] as const
export type SendGraceScope = (typeof SEND_GRACE_SCOPES)[number]

export const SEND_GRACE_DEFAULT_SCOPE: SendGraceScope = 'inferred'

/** How long a held send waits before it goes. */
export const SEND_GRACE_DEFAULT_MS = 900
export const SEND_GRACE_MIN_MS = 0
export const SEND_GRACE_MAX_MS = 5000

export interface ComposerSendPrefs {
  mode: ComposerSendMode
  doubleEnterMs: number
  holdMs: number
  typingIdleMs: number
  sendGrace: SendGraceScope
  sendGraceMs: number
}

export function isSendGraceScope(value: unknown): value is SendGraceScope {
  return SEND_GRACE_SCOPES.includes(value as SendGraceScope)
}

export function isComposerSendMode(value: unknown): value is ComposerSendMode {
  return COMPOSER_SEND_MODES.includes(value as ComposerSendMode)
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
  const record = (value ?? {}) as Partial<ComposerSendPrefs>

  return {
    mode: isComposerSendMode(record.mode) ? record.mode : COMPOSER_SEND_DEFAULT_MODE,
    doubleEnterMs: clampDoubleEnterMs(record.doubleEnterMs ?? DOUBLE_ENTER_DEFAULT_MS),
    holdMs: clampHoldMs(record.holdMs ?? HOLD_DEFAULT_MS),
    typingIdleMs: clampTypingIdleMs(record.typingIdleMs ?? TYPING_IDLE_DEFAULT_MS),
    sendGrace: isSendGraceScope(record.sendGrace) ? record.sendGrace : SEND_GRACE_DEFAULT_SCOPE,
    sendGraceMs: clampSendGraceMs(record.sendGraceMs ?? SEND_GRACE_DEFAULT_MS)
  }
}
