/**
 * Quick Entry — the global-hotkey mini composer.
 *
 * A small frameless always-on-top window that a global shortcut summons from
 * anywhere so the user can fire a prompt at Hermes without raising the whole
 * app. The window carries NO gateway connection of its own: it forwards the
 * text to the primary renderer, which sends it through the SAME prompt-submit
 * path the normal composer uses (see app/contrib/hooks/use-quick-entry-bridge).
 *
 * Everything Electron-free lives here so the parts that actually break a user —
 * accelerator validation, "disabled means never register", and surfacing a
 * shortcut another app already owns — are unit-testable without booting
 * Electron. main.ts owns the BrowserWindow, the file I/O, and the real
 * `globalShortcut`.
 */

// Default matches the muscle memory of the apps this ports from (Claude
// Desktop's quick entry / ChatGPT's Quick Chat sit on a Cmd+Shift chord).
const DEFAULT_QUICK_ENTRY_SHORTCUT = 'CommandOrControl+Shift+Space'

// Compact capture surface: wide enough for a sentence, short enough to read as
// a HUD rather than a second app window. Height covers the composer row plus
// the session-target picker row; the renderer never grows the OS window in v1.
const QUICK_ENTRY_WINDOW_WIDTH = 640
const QUICK_ENTRY_WINDOW_HEIGHT = 168

// Spotlight-ish placement, as fractions (0..1) of the target display's work
// area: `x` is where the window's HORIZONTAL CENTRE sits (0.5 = centered),
// `y` is where its TOP EDGE sits. This is only the SHIPPED default — the live
// value comes from the user's saved settings (quick-entry.json), so a chosen
// placement never has to be hard-coded here.
const DEFAULT_QUICK_ENTRY_POSITION: QuickEntryPosition = { x: 0.5, y: 0.22 }

export interface QuickEntrySubmitRelayResult {
  ok: boolean
  [key: string]: unknown
}

export interface QuickEntrySubmitRelay {
  begin: (forward: (correlationId: string) => void) => Promise<QuickEntrySubmitRelayResult>
  acknowledge: (correlationId: string, result: QuickEntrySubmitRelayResult) => void
  pendingCount: () => number
  /** Correlations whose outcome is UNKNOWN: the relay timed out but the backend
   *  may still have accepted the prompt. Kept so a late ack reconciles instead
   *  of being silently dropped. */
  reconcilableCount: () => number
}

export function createQuickEntrySubmitRelay(options: {
  onLateResult?: (correlationId: string, result: QuickEntrySubmitRelayResult) => void
  onSuccess: () => void
  timeoutMs?: number
}): QuickEntrySubmitRelay {
  const pending = new Map<string, { resolve: (result: QuickEntrySubmitRelayResult) => void; timer: NodeJS.Timeout }>()

  // Timed-out correlations. Delivery is UNCONFIRMED, so a late ack must
  // reconcile; a correlation is never resolved twice.
  const reconcilable = new Set<string>()
  let sequence = 0

  return {
    acknowledge(correlationId, result) {
      const request = pending.get(correlationId)

      if (!request) {
        // A late ack for a timed-out submit: the outcome is now known. Do NOT
        // hide the window here — the user may already be typing again. The
        // renderer reconciles the unknown outcome itself.
        if (reconcilable.delete(correlationId)) {
          options.onLateResult?.(correlationId, result)
        }

        return
      }

      clearTimeout(request.timer)
      pending.delete(correlationId)
      request.resolve(result)

      if (result.ok === true) {
        options.onSuccess()
      }
    },
    begin(forward) {
      const correlationId = ['qe', Date.now(), ++sequence].join('-')

      return new Promise(resolve => {
        const timer = setTimeout(() => {
          pending.delete(correlationId)
          // UNKNOWN, not failed: the prompt may already be accepted, so this
          // must never invite a retry. Keep the correlation for late acks.
          reconcilable.add(correlationId)
          resolve({
            code: 'timeout',
            message: 'Hermes has not confirmed the prompt yet — it may still be delivered.',
            ok: false,
            retryable: false
          })
        }, options.timeoutMs ?? 15_000)

        pending.set(correlationId, { resolve, timer })
        forward(correlationId)
      })
    },
    pendingCount() {
      return pending.size
    },
    reconcilableCount() {
      return reconcilable.size
    }
  }
}

// Electron accelerator vocabulary (electronjs.org/docs/latest/api/accelerator).
// Kept as data so validation and the settings UI agree on one list.
const ACCELERATOR_MODIFIERS = new Set([
  'alt',
  'altgr',
  'cmd',
  'cmdorctrl',
  'command',
  'commandorcontrol',
  'control',
  'ctrl',
  'meta',
  'option',
  'shift',
  'super'
])

const ACCELERATOR_KEYS = new Set([
  'backspace',
  'delete',
  'down',
  'end',
  'enter',
  'escape',
  'home',
  'insert',
  'left',
  'medianexttrack',
  'mediaplaypause',
  'mediaprevioustrack',
  'mediastop',
  'pagedown',
  'pageup',
  'plus',
  'printscreen',
  'return',
  'right',
  'space',
  'tab',
  'up',
  'volumedown',
  'volumemute',
  'volumeup'
])

// Single printable characters Electron accepts verbatim, plus 0-9 / A-Z below.
const ACCELERATOR_PUNCTUATION = new Set([
  '!',
  '"',
  '#',
  '$',
  '%',
  '&',
  "'",
  '(',
  ')',
  '*',
  '+',
  ',',
  '-',
  '.',
  '/',
  ':',
  ';',
  '<',
  '=',
  '>',
  '?',
  '@',
  '[',
  '\\',
  ']',
  '^',
  '_',
  '`',
  '{',
  '|',
  '}',
  '~'
])

/** Why a shortcut string was rejected. The renderer maps these to copy. */
export type QuickEntryShortcutError =
  'empty' | 'invalid-key' | 'invalid-modifier' | 'no-key' | 'no-modifier' | 'reserved'

export type QuickEntryShortcutParse = { ok: false; reason: QuickEntryShortcutError } | { accelerator: string; ok: true }

function isAcceleratorKey(token: string): boolean {
  if (ACCELERATOR_KEYS.has(token)) {
    return true
  }

  if (/^f([1-9]|1[0-9]|2[0-4])$/.test(token)) {
    return true
  }

  if (/^num(?:[0-9]|lock|dec|add|sub|mult|div)$/.test(token)) {
    return true
  }

  return token.length === 1 && (/^[a-z0-9]$/.test(token) || ACCELERATOR_PUNCTUATION.has(token))
}

/**
 * Validate + normalize a user-typed accelerator.
 *
 * Rules beyond Electron's own grammar, both deliberate:
 * - At least one modifier. A bare global key steals that key from EVERY app.
 * - `Escape` can't be the key: inside the window Escape means "hide", so
 *   binding it globally would make the shortcut un-toggleable.
 */
export function parseQuickEntryShortcut(raw: unknown): QuickEntryShortcutParse {
  if (typeof raw !== 'string' || !raw.trim()) {
    return { ok: false, reason: 'empty' }
  }

  const parts = raw
    .split('+')
    .map(part => part.trim())
    .filter(Boolean)

  if (parts.length === 0) {
    return { ok: false, reason: 'empty' }
  }

  const modifiers: string[] = []
  let key: null | string = null

  for (const part of parts) {
    const lower = part.toLowerCase()

    if (ACCELERATOR_MODIFIERS.has(lower)) {
      if (key) {
        // A modifier after the key ("A+Shift") is not a valid accelerator.
        return { ok: false, reason: 'invalid-modifier' }
      }

      modifiers.push(lower)

      continue
    }

    if (key) {
      // Two non-modifier keys ("Shift+A+B").
      return { ok: false, reason: 'invalid-key' }
    }

    if (!isAcceleratorKey(lower)) {
      return { ok: false, reason: 'invalid-key' }
    }

    key = lower
  }

  if (!key) {
    return { ok: false, reason: 'no-key' }
  }

  if (modifiers.length === 0) {
    return { ok: false, reason: 'no-modifier' }
  }

  if (key === 'escape') {
    return { ok: false, reason: 'reserved' }
  }

  // Canonical casing so a saved shortcut round-trips identically no matter how
  // the user typed it, and duplicate modifiers collapse.
  const seen = new Set<string>()

  const normalizedModifiers = modifiers
    .map(modifier => CANONICAL_MODIFIER[modifier] ?? modifier)
    .filter(modifier => (seen.has(modifier) ? false : (seen.add(modifier), true)))
    // Stable display order (Electron itself is order-insensitive).
    .sort((left, right) => MODIFIER_ORDER.indexOf(left) - MODIFIER_ORDER.indexOf(right))

  return { accelerator: [...normalizedModifiers, canonicalKey(key)].join('+'), ok: true }
}

const CANONICAL_MODIFIER: Record<string, string> = {
  alt: 'Alt',
  altgr: 'AltGr',
  cmd: 'Command',
  cmdorctrl: 'CommandOrControl',
  command: 'Command',
  commandorcontrol: 'CommandOrControl',
  control: 'Control',
  ctrl: 'Control',
  meta: 'Super',
  option: 'Option',
  shift: 'Shift',
  super: 'Super'
}

const MODIFIER_ORDER = ['CommandOrControl', 'Command', 'Control', 'Super', 'Alt', 'Option', 'AltGr', 'Shift']

const CANONICAL_KEY: Record<string, string> = {
  backspace: 'Backspace',
  delete: 'Delete',
  down: 'Down',
  end: 'End',
  enter: 'Enter',
  escape: 'Escape',
  home: 'Home',
  insert: 'Insert',
  medianexttrack: 'MediaNextTrack',
  mediaplaypause: 'MediaPlayPause',
  mediaprevioustrack: 'MediaPreviousTrack',
  mediastop: 'MediaStop',
  pagedown: 'PageDown',
  pageup: 'PageUp',
  plus: 'Plus',
  printscreen: 'PrintScreen',
  return: 'Return',
  right: 'Right',
  space: 'Space',
  tab: 'Tab',
  up: 'Up',
  volumedown: 'VolumeDown',
  volumemute: 'VolumeMute',
  volumeup: 'VolumeUp',
  left: 'Left'
}

function canonicalKey(key: string): string {
  if (CANONICAL_KEY[key]) {
    return CANONICAL_KEY[key]
  }

  if (/^f([1-9]|1[0-9]|2[0-4])$/.test(key)) {
    return key.toUpperCase()
  }

  if (key.length === 1 && /^[a-z]$/.test(key)) {
    return key.toUpperCase()
  }

  return key
}

/**
 * Where the window sits on a display's work area — both axes are FRACTIONS of
 * that work area (0..1), not pixels, so one setting is correct on every
 * monitor. `x` places the window's horizontal centre, `y` its top edge.
 *
 * Fractions rather than absolute pixels on purpose: summoning re-anchors the
 * window to whichever display the CURSOR is on, so a stored pixel pair would
 * land in the wrong corner (or off-screen) the moment a portrait second
 * display becomes the target. A fraction stays valid wherever it is applied.
 */
export interface QuickEntryPosition {
  x: number
  y: number
}

/** The persisted shape of `quick-entry.json` (main-process owned). */
export interface QuickEntrySettings {
  enabled: boolean
  position: QuickEntryPosition
  shortcut: string
}

/** One axis: a real, finite number clamped into 0..1. Anything else is junk. */
function sanitizePositionAxis(value: unknown, fallback: number): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return fallback
  }

  return Math.min(1, Math.max(0, value))
}

/**
 * Position → usable position. A missing `position` (legacy `{enabled, shortcut}`
 * files), a non-object, or a junk axis falls back to the shipped default per
 * axis; a present axis is clamped rather than discarded.
 */
function sanitizeQuickEntryPosition(raw: unknown): QuickEntryPosition {
  const record = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {}

  return {
    x: sanitizePositionAxis(record.x, DEFAULT_QUICK_ENTRY_POSITION.x),
    y: sanitizePositionAxis(record.y, DEFAULT_QUICK_ENTRY_POSITION.y)
  }
}

/**
 * Raw persisted JSON → usable settings. A malformed/absent file, or a shortcut
 * that no longer validates (hand-edited, or from a future build), falls back to
 * the default shortcut rather than leaving the feature un-summonable.
 */
export function sanitizeQuickEntrySettings(raw: unknown): QuickEntrySettings {
  const record = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {}
  const parsed = parseQuickEntryShortcut(record.shortcut)

  return {
    // Default ON: the feature is inert until the shortcut is pressed.
    enabled: record.enabled === undefined ? true : record.enabled === true,
    position: sanitizeQuickEntryPosition(record.position),
    shortcut: parsed.ok ? parsed.accelerator : DEFAULT_QUICK_ENTRY_SHORTCUT
  }
}

/** The slice of Electron's `globalShortcut` we use (injected for testing). */
export interface GlobalShortcutLike {
  isRegistered(accelerator: string): boolean
  register(accelerator: string, callback: () => void): boolean
  unregister(accelerator: string): void
}

/**
 * What Settings shows. `registered` is the ground truth (we asked the OS);
 * `error` distinguishes "you turned it off" from "another app owns that chord",
 * which is the failure this feature must never swallow.
 */
export interface QuickEntryRegistration {
  error: null | QuickEntryRegistrationError
  registered: boolean
  shortcut: string
}

export type QuickEntryRegistrationError = 'invalid' | 'taken'

export interface QuickEntryShortcutController {
  /** Registration state as of the last apply. */
  current(): QuickEntryRegistration
  /** Release the shortcut (quit / feature off). Idempotent. */
  dispose(): void
  /**
   * Re-register to match `settings`. Returns the resulting state.
   *
   * A Pick of the persisted shape, not the whole thing: `position` is
   * placement, which main applies to the window itself — the shortcut
   * controller never reads it.
   */
  apply(settings: Pick<QuickEntrySettings, 'enabled' | 'shortcut'>): QuickEntryRegistration
}

/**
 * Owns the one live global accelerator. Single resolver so every caller — boot,
 * the settings write, quit — gets the same answer and we can never leak two
 * registrations for one feature.
 *
 * Disabled settings never touch `register()` at all: a user who turned Quick
 * Entry off must not have their chord silently held hostage.
 */
export function createQuickEntryShortcut(
  globalShortcut: GlobalShortcutLike,
  onTrigger: () => void
): QuickEntryShortcutController {
  let active: null | string = null
  let state: QuickEntryRegistration = { error: null, registered: false, shortcut: DEFAULT_QUICK_ENTRY_SHORTCUT }

  const release = () => {
    if (active) {
      try {
        globalShortcut.unregister(active)
      } catch {
        // Best effort — a dead accelerator must not block a re-register.
      }

      active = null
    }
  }

  return {
    apply(settings) {
      const parsed = parseQuickEntryShortcut(settings.shortcut)
      const shortcut = parsed.ok ? parsed.accelerator : settings.shortcut

      release()

      if (!settings.enabled) {
        state = { error: null, registered: false, shortcut }

        return state
      }

      if (!parsed.ok) {
        state = { error: 'invalid', registered: false, shortcut }

        return state
      }

      // `isRegistered` catches the common conflict before we ask, and
      // `register()` returning false catches the rest (another process owns it
      // OS-wide). Both land in the same surfaced 'taken' state.
      let ok = false

      try {
        ok = globalShortcut.isRegistered(parsed.accelerator)
          ? false
          : globalShortcut.register(parsed.accelerator, onTrigger)
      } catch {
        ok = false
      }

      active = ok ? parsed.accelerator : null
      state = { error: ok ? null : 'taken', registered: ok, shortcut: parsed.accelerator }

      return state
    },
    current() {
      return state
    },
    dispose() {
      release()
      state = { ...state, error: null, registered: false }
    }
  }
}

/**
 * Where the quick window opens on a given display work area: the window's
 * horizontal CENTRE and its TOP EDGE land on `position`'s fractions of that
 * work area, then both axes clamp so the window stays fully inside it on small
 * or odd displays. Omitted `position` means the shipped placement, so callers
 * and tests that never think about position keep behaving exactly as before.
 * Pure and Electron-free — unit-tested without booting Electron.
 */
export function quickEntryWindowBounds(
  workArea?: { height: number; width: number; x: number; y: number },
  position: QuickEntryPosition = DEFAULT_QUICK_ENTRY_POSITION
): {
  height: number
  width: number
  x: number
  y: number
} {
  const width = Math.min(QUICK_ENTRY_WINDOW_WIDTH, workArea?.width ?? QUICK_ENTRY_WINDOW_WIDTH)
  const height = Math.min(QUICK_ENTRY_WINDOW_HEIGHT, workArea?.height ?? QUICK_ENTRY_WINDOW_HEIGHT)

  if (!workArea) {
    return { height, width, x: 0, y: 0 }
  }

  const { x: xFraction, y: yFraction } = sanitizeQuickEntryPosition(position)
  const centeredX = workArea.x + workArea.width * xFraction - width / 2
  const topY = workArea.y + workArea.height * yFraction
  const maxX = workArea.x + workArea.width - width
  const maxY = workArea.y + workArea.height - height

  // Round first, clamp second: the clamp must win so no edge of the window can
  // land outside the work area even when its origin is fractional.
  const x = Math.min(Math.max(Math.round(centeredX), workArea.x), maxX)
  const y = Math.min(Math.max(Math.round(topY), workArea.y), maxY)

  return { height, width, x, y }
}

/**
 * Inverse of {@link quickEntryWindowBounds}: where the window actually sits,
 * expressed as fractions of the work area it is on (its horizontal centre and
 * its top edge), clamped to 0..1. This is how a drag the user performed gets
 * persisted — main measures the live bounds and stores them through the same
 * settings file the numeric fields write.
 *
 * The round trip is stable: for a window already placed by
 * `quickEntryWindowBounds`, feeding these fractions back reproduces the same
 * bounds (the only rounding is `Math.round` of a value that is already an
 * integer ±floating-point noise), so a programmatic reposition re-persisting
 * itself does not drift a pixel at a time.
 */
export function quickEntryPositionFromBounds(
  bounds: { height: number; width: number; x: number; y: number },
  workArea: { height: number; width: number; x: number; y: number }
): QuickEntryPosition {
  // A degenerate work area cannot locate anything — fall back to the defaults
  // rather than dividing by zero into NaN/Infinity.
  const axis = (value: number, origin: number, extent: number, fallback: number) =>
    extent > 0 && Number.isFinite(value) ? Math.min(1, Math.max(0, (value - origin) / extent)) : fallback

  return {
    x: axis(bounds.x + bounds.width / 2, workArea.x, workArea.width, DEFAULT_QUICK_ENTRY_POSITION.x),
    y: axis(bounds.y, workArea.y, workArea.height, DEFAULT_QUICK_ENTRY_POSITION.y)
  }
}

export {
  DEFAULT_QUICK_ENTRY_POSITION,
  DEFAULT_QUICK_ENTRY_SHORTCUT,
  QUICK_ENTRY_WINDOW_HEIGHT,
  QUICK_ENTRY_WINDOW_WIDTH
}
