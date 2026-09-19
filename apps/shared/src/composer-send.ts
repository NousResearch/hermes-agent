/**
 * How the composer decides to commit a draft, and the windows the gestures
 * measure against.
 *
 * Main owns the persisted value (a small JSON file under userData, hand-editable
 * like `data-url-read-max.json`); the renderer mirrors it in Settings →
 * Keyboards and clamps optimistically before sending. Both ends have to agree on
 * the default and the bounds, so they live here rather than as two constants
 * with a "keep these in sync" comment between them.
 *
 * The shape is three KINDS of setting, not one enum:
 *
 *   1. `enterSends` — the GATE. True means a bare Enter commits on the press,
 *      which is the historical binding; false means a lone press never sends,
 *      whatever else is configured. The panel presents this as "keep a bare
 *      Enter from sending", since the feature exists to stop accidental sends,
 *      and inverts the flag in that one place rather than storing the negative.
 *   2. `enterNewline` — what a press does when it is NOT sending. Separate from
 *      the gate on purpose: a line break and a send are different outcomes, and
 *      someone who wants the key to do nothing at all is asking for neither.
 *      Only reachable while the gate is closed.
 *   3. the gesture flags — ADDITIONAL ways to commit. These overlap freely: a
 *      user can double tap mid-flow, hold when they mean it, and let the pause
 *      rule catch the single Enter after they stop typing. As an enum they could
 *      only ever pick one, which forbade combinations nothing had a reason to
 *      forbid.
 *
 * Gestures are unreachable while `enterSends` is true — the press has already
 * committed by the time any of them could register — so readers filter them
 * through `activeSendGestures` instead of trusting the flags alone. They are
 * kept in the file rather than forced off, so switching back restores what the
 * user chose.
 *
 * `enterSends: true` is the historical binding and stays the default: an upgrade
 * must never change what Enter does to someone who never opens Settings.
 */

/** Ways to commit a draft other than the bare press. Each has its own window. */
export const COMPOSER_SEND_GESTURES = ['doubleTap', 'pause', 'hold'] as const
export type ComposerSendGesture = (typeof COMPOSER_SEND_GESTURES)[number]

/** A second plain Enter inside this window sends. 400ms is the usual
 *  double-click allowance: long enough for a deliberate double-tap, short enough
 *  to be over before someone who just broke a line types again. */
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

/** `idle` only: how long the composer waits, with nobody touching it, before
 *  sending on its own. LONGER than the pause window on purpose — those are
 *  different questions. `typingIdleMs` asks "when does a press count as
 *  finished"; this asks "when do I give up waiting for you", and giving up on
 *  someone deserves more rope. */
export const IDLE_SEND_DEFAULT_MS = 2500
export const IDLE_SEND_MIN_MS = 1000
export const IDLE_SEND_MAX_MS = 15000

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
 * - `enter` — the bare Enter, when it sends directly.
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
  /** Whether a bare Enter commits the draft: the historical binding. False is
   *  the gate, and means a lone press never sends whatever else is configured. */
  enterSends: boolean
  /** What a press does when it is not sending: break the line, or nothing at
   *  all. Only meaningful while `enterSends` is false. */
  enterNewline: boolean
  /** Commits on the second of two fast presses. */
  sendOnDoubleTap: boolean
  /** Commits on a single Enter, once typing has stopped for `typingIdleMs`. */
  sendOnPause: boolean
  /** Commits when the key is held for `holdMs`. */
  sendOnHold: boolean
  /** Commits on its own, with no press at all, once typing has stopped for
   *  `idleSendMs`. The only gesture that acts without being asked, so it is
   *  always delayed through the grace window — see `SEND_GRACE_REASONS`. */
  sendOnIdle: boolean
  idleSendMs: number
  doubleEnterMs: number
  holdMs: number
  typingIdleMs: number
  /** Which sends wait for the grace window (see `SEND_GRACE_REASONS`). */
  sendGraceFor: readonly SendGraceReason[]
  sendGraceMs: number
}

const GESTURE_FLAG: Record<ComposerSendGesture, keyof ComposerSendPrefs> = {
  doubleTap: 'sendOnDoubleTap',
  hold: 'sendOnHold',
  pause: 'sendOnPause'
}

/** The gestures that can actually fire for these prefs. Empty while the bare
 *  Enter commits on the press, because nothing else ever gets a turn. */
export function activeSendGestures(prefs: ComposerSendPrefs): readonly ComposerSendGesture[] {
  if (prefs.enterSends) {
    return []
  }

  return COMPOSER_SEND_GESTURES.filter(gesture => prefs[GESTURE_FLAG[gesture]] === true)
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

/**
 * The `mode` enum this replaced, expanded into the equivalent flags. Every value
 * that ever shipped is here, including `hold`, which was briefly a mode of its
 * own before becoming a flag.
 *
 * Landing an old value on the DEFAULT would be a silent behaviour change — a
 * user who had moved sending off Enter would find Enter sending again, which is
 * the exact thing they opened Settings to prevent — so unknown values fall
 * through to the default rather than guessing.
 */
function fromLegacyMode(mode: unknown): Omit<ComposerSendPrefs, 'doubleEnterMs' | 'enterNewline' | 'holdMs' | 'idleSendMs' | 'sendGraceFor' | 'sendGraceMs' | 'sendOnIdle' | 'typingIdleMs'> | null {
  switch (mode) {
    case 'enter':
      return { enterSends: true, sendOnDoubleTap: false, sendOnHold: false, sendOnPause: false }

    case 'double-enter':
      return { enterSends: false, sendOnDoubleTap: true, sendOnHold: false, sendOnPause: false }

    case 'pause':
      return { enterSends: false, sendOnDoubleTap: false, sendOnHold: false, sendOnPause: true }

    case 'mod-enter':
      return { enterSends: false, sendOnDoubleTap: false, sendOnHold: false, sendOnPause: false }

    case 'hold':
      return { enterSends: false, sendOnDoubleTap: false, sendOnHold: true, sendOnPause: false }

    default:
      return null
  }
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

export function clampIdleSendMs(value: unknown): number {
  const parsed = Number(value)

  if (!Number.isFinite(parsed)) {
    return IDLE_SEND_DEFAULT_MS
  }

  return Math.min(IDLE_SEND_MAX_MS, Math.max(IDLE_SEND_MIN_MS, Math.round(parsed)))
}

export function clampSendGraceMs(value: unknown): number {
  const parsed = Number(value)

  if (!Number.isFinite(parsed)) {
    return SEND_GRACE_DEFAULT_MS
  }

  return Math.min(SEND_GRACE_MAX_MS, Math.max(SEND_GRACE_MIN_MS, Math.round(parsed)))
}

/** Coerce anything read off disk (or off the IPC bridge) into valid prefs. A
 *  legacy `mode` is translated; an out-of-range window clamps. */
export function normalizeComposerSendPrefs(value: unknown): ComposerSendPrefs {
  const record = (value ?? {}) as Record<string, unknown> & Partial<ComposerSendPrefs>
  const legacy = record.enterSends === undefined ? fromLegacyMode(record.mode) : null

  return {
    enterSends: legacy ? legacy.enterSends : record.enterSends !== false,
    // Defaults ON: a file written before this key existed keeps breaking the
    // line as it always did, and closing the gate gives the least surprising
    // result. Turning it off is the deliberate "the key does nothing" choice.
    enterNewline: record.enterNewline !== false,
    sendOnDoubleTap: legacy ? legacy.sendOnDoubleTap : record.sendOnDoubleTap === true,
    sendOnHold: legacy ? legacy.sendOnHold : record.sendOnHold === true,
    sendOnPause: legacy ? legacy.sendOnPause : record.sendOnPause === true,
    // Never migrated from a legacy value: the auto-send did not exist before, so
    // defaulting it on for an old file would start sending messages nobody asked
    // it to send.
    sendOnIdle: record.sendOnIdle === true,
    idleSendMs: clampIdleSendMs(record.idleSendMs ?? IDLE_SEND_DEFAULT_MS),
    doubleEnterMs: clampDoubleEnterMs(record.doubleEnterMs ?? DOUBLE_ENTER_DEFAULT_MS),
    holdMs: clampHoldMs(record.holdMs ?? HOLD_DEFAULT_MS),
    typingIdleMs: clampTypingIdleMs(record.typingIdleMs ?? TYPING_IDLE_DEFAULT_MS),
    sendGraceFor: normalizeGraceReasons(record),
    sendGraceMs: clampSendGraceMs(record.sendGraceMs ?? SEND_GRACE_DEFAULT_MS)
  }
}
