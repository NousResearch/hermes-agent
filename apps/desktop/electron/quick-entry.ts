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
const QUICK_ENTRY_WINDOW_WIDTH = 760
const QUICK_ENTRY_WINDOW_HEIGHT = 420
// The agent launcher is a dropdown, not a second HUD panel. Keep the native
// window tight so its transparent hit area and shadow never read as a large
// floating box after a cold restart.
const QUICK_ENTRY_AGENT_WINDOW_WIDTH = 224
const QUICK_ENTRY_AGENT_WINDOW_HEIGHT = 238

const QUICK_ENTRY_FALLBACK_PROFILES = new Set(['default'])

export type QuickEntryMode = 'agents' | 'composer'

export interface QuickEntryAnchorRect {
  height: number
  viewportHeight?: number
  viewportWidth?: number
  width: number
  x: number
  y: number
}

export function quickEntryScreenAnchorRect(
  contentBounds: { height?: number; width?: number; x: number; y: number } | undefined,
  anchorRect: QuickEntryAnchorRect | undefined
): QuickEntryAnchorRect | undefined {
  if (
    !contentBounds ||
    !anchorRect ||
    !Number.isFinite(anchorRect.x) ||
    !Number.isFinite(anchorRect.y) ||
    !Number.isFinite(anchorRect.width) ||
    !Number.isFinite(anchorRect.height) ||
    anchorRect.width <= 0 ||
    anchorRect.height <= 0
  ) {
    return undefined
  }

  const hasViewportScale =
    Number.isFinite(contentBounds.width) &&
    Number.isFinite(contentBounds.height) &&
    Number.isFinite(anchorRect.viewportWidth) &&
    Number.isFinite(anchorRect.viewportHeight) &&
    (contentBounds.width ?? 0) > 0 &&
    (contentBounds.height ?? 0) > 0 &&
    (anchorRect.viewportWidth ?? 0) > 0 &&
    (anchorRect.viewportHeight ?? 0) > 0

  const scaleX = hasViewportScale ? (contentBounds.width ?? 1) / (anchorRect.viewportWidth ?? 1) : 1
  const scaleY = hasViewportScale ? (contentBounds.height ?? 1) / (anchorRect.viewportHeight ?? 1) : 1

  return {
    height: anchorRect.height * scaleY,
    width: anchorRect.width * scaleX,
    x: contentBounds.x + anchorRect.x * scaleX,
    y: contentBounds.y + anchorRect.y * scaleY
  }
}

export function isQuickEntryFallbackProfile(profile: string): boolean {
  return QUICK_ENTRY_FALLBACK_PROFILES.has(profile.trim().toLowerCase())
}

/**
 * Fallback agents exist only for the brief pre-roster loading state. Once Main
 * has received any roster array, that live result is authoritative—even when
 * empty or when a shipped fallback profile is explicitly unreachable.
 */
export function isQuickEntryAgentOffered(
  profile: string,
  agents: Array<{ profile?: unknown; reachable?: unknown }> | undefined
): boolean {
  if (!Array.isArray(agents)) {
    return isQuickEntryFallbackProfile(profile)
  }

  return agents.some(agent => agent?.profile === profile && agent?.reachable === true)
}

export function quickEntryRejectedLaunchResult(
  payload: unknown,
  error: string
): { error: string; ok: false; profile: string; requestId: string } | undefined {
  if (!payload || typeof payload !== 'object') {
    return undefined
  }

  const candidate = payload as { action?: unknown; groupId?: unknown; profile?: unknown; requestId?: unknown }
  const requestId = typeof candidate.requestId === 'string' ? candidate.requestId.trim() : ''

  if (!requestId) {
    return undefined
  }

  const profile =
    candidate.action === 'open-agent' && typeof candidate.profile === 'string'
      ? candidate.profile.trim()
      : candidate.action === 'open-group' && typeof candidate.groupId === 'string'
        ? `group:${candidate.groupId.trim()}`
        : ''

  if (!profile) {
    return undefined
  }

  return { error: error.trim().slice(0, 240), ok: false, profile, requestId }
}

// Spotlight-ish fallback placement when a cursor point is unavailable.
const QUICK_ENTRY_TOP_FRACTION = 0.22
// Keep the capture surface close enough to read as pointer-adjacent without
// covering the control directly under the cursor.
const QUICK_ENTRY_POINTER_GAP = 18
// The pet picker is a visually attached menu, so it sits closer than the
// general-purpose quick composer without covering the mascot itself.
const QUICK_ENTRY_AGENT_GAP = 10

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

/** The persisted shape of `quick-entry.json` (main-process owned). */
export interface QuickEntrySettings {
  enabled: boolean
  shortcut: string
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
  /** Re-register to match `settings`. Returns the resulting state. */
  apply(settings: QuickEntrySettings): QuickEntryRegistration
}

interface QuickEntryShowHost {
  isDestroyed: () => boolean
  webContents: unknown
}

/** Only an explicit click in a declared, live host may summon Quick Entry. */
export function canShowQuickEntryFrom(
  sender: unknown,
  hosts: Array<null | QuickEntryShowHost | undefined>
): boolean {
  return hosts.some(host => Boolean(host && !host.isDestroyed() && host.webContents === sender))
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
 * Where the quick window opens on a given display work area.
 *
 * The compact pet chooser opens ABOVE the summon point, horizontally centred on
 * it, so it reads as attached to the pet it was launched from instead of as a
 * popup that happened to land near the mouse. It drops below only when there is
 * genuinely no room above; when neither side fits it takes the roomier one, so
 * the unavoidable overlap on a very short work area is as small as the display
 * allows. The composer keeps its original lower-right adjacent placement.
 *
 * Both modes clamp to the active display's work area. Without a pointer the
 * original Spotlight placement is used.
 */
export function quickEntryWindowBounds(
  workArea?: { height: number; width: number; x: number; y: number },
  cursor?: { x: number; y: number },
  mode: QuickEntryMode = 'composer',
  anchorRect?: QuickEntryAnchorRect
): {
  height: number
  width: number
  x: number
  y: number
} {
  const requestedWidth = mode === 'agents' ? QUICK_ENTRY_AGENT_WINDOW_WIDTH : QUICK_ENTRY_WINDOW_WIDTH
  const requestedHeight = mode === 'agents' ? QUICK_ENTRY_AGENT_WINDOW_HEIGHT : QUICK_ENTRY_WINDOW_HEIGHT
  const width = Math.min(requestedWidth, workArea?.width ?? requestedWidth)
  let height = Math.min(requestedHeight, workArea?.height ?? requestedHeight)

  if (!workArea) {
    return { height, width, x: 0, y: 0 }
  }

  if (mode === 'agents' && anchorRect) {
    const minX = workArea.x
    const maxX = workArea.x + workArea.width - width
    const minY = workArea.y
    const workBottom = workArea.y + workArea.height
    const anchorBottom = anchorRect.y + anchorRect.height
    const roomAbove = Math.max(0, anchorRect.y - QUICK_ENTRY_AGENT_GAP - minY)
    const roomBelow = Math.max(0, workBottom - anchorBottom - QUICK_ENTRY_AGENT_GAP)
    const above = requestedHeight <= roomAbove || (requestedHeight > roomBelow && roomAbove >= roomBelow)
    const availableHeight = above ? roomAbove : roomBelow

    if (availableHeight > 0) {
      height = Math.min(height, availableHeight)
    }

    const maxY = workBottom - height
    const centeredX = anchorRect.x + anchorRect.width / 2 - width / 2
    const aboveY = anchorRect.y - QUICK_ENTRY_AGENT_GAP - height
    const belowY = anchorBottom + QUICK_ENTRY_AGENT_GAP

    return {
      height,
      width,
      x: Math.round(Math.min(Math.max(minX, centeredX), maxX)),
      y: Math.round(Math.min(Math.max(minY, above ? aboveY : belowY), maxY))
    }
  }

  if (cursor && Number.isFinite(cursor.x) && Number.isFinite(cursor.y)) {
    const minX = workArea.x
    const maxX = workArea.x + workArea.width - width
    const minY = workArea.y
    const maxY = workArea.y + workArea.height - height

    if (mode === 'agents') {
      const centeredX = cursor.x - width / 2
      const aboveY = cursor.y - height - QUICK_ENTRY_AGENT_GAP
      const belowY = cursor.y + QUICK_ENTRY_AGENT_GAP
      const roomAbove = cursor.y - QUICK_ENTRY_AGENT_GAP - minY
      const roomBelow = maxY + height - (cursor.y + QUICK_ENTRY_AGENT_GAP)
      const above = aboveY >= minY || roomAbove >= roomBelow

      return {
        height,
        width,
        x: Math.round(Math.min(Math.max(minX, centeredX), maxX)),
        y: Math.round(Math.min(Math.max(minY, above ? aboveY : belowY), maxY))
      }
    }

    const preferredX = cursor.x + QUICK_ENTRY_POINTER_GAP
    const preferredY = cursor.y + QUICK_ENTRY_POINTER_GAP
    const flippedX = cursor.x - width - QUICK_ENTRY_POINTER_GAP
    const flippedY = cursor.y - height - QUICK_ENTRY_POINTER_GAP

    return {
      height,
      width,
      x: Math.round(Math.min(Math.max(minX, preferredX <= maxX ? preferredX : flippedX), maxX)),
      y: Math.round(Math.min(Math.max(minY, preferredY <= maxY ? preferredY : flippedY), maxY))
    }
  }

  const x = Math.round(workArea.x + (workArea.width - width) / 2)
  const maxY = workArea.y + workArea.height - height
  const y = Math.round(Math.min(Math.max(workArea.y, workArea.y + workArea.height * QUICK_ENTRY_TOP_FRACTION), maxY))

  return { height, width, x, y }
}

export {
  DEFAULT_QUICK_ENTRY_SHORTCUT,
  QUICK_ENTRY_AGENT_WINDOW_HEIGHT,
  QUICK_ENTRY_AGENT_WINDOW_WIDTH,
  QUICK_ENTRY_POINTER_GAP,
  QUICK_ENTRY_TOP_FRACTION,
  QUICK_ENTRY_WINDOW_HEIGHT,
  QUICK_ENTRY_WINDOW_WIDTH
}
