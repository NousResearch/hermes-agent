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

import { execFile } from 'node:child_process'
import { promisify } from 'node:util'

// Default matches the muscle memory of the apps this ports from (Claude
// Desktop's quick entry / ChatGPT's Quick Chat sit on a Cmd+Shift chord).
const DEFAULT_QUICK_ENTRY_SHORTCUT = 'CommandOrControl+Shift+Space'

// Compact capture surface: wide enough for a sentence, short enough to read as
// a HUD rather than a second app window. Height covers the composer row plus
// the session-target picker row; the renderer never grows the OS window in v1.
const QUICK_ENTRY_WINDOW_WIDTH = 640
const QUICK_ENTRY_WINDOW_HEIGHT = 168

// Spotlight-ish placement: horizontally centered on the active display, a
// comfortable fraction down from the top rather than dead center.
const QUICK_ENTRY_TOP_FRACTION = 0.22

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
 * `error` distinguishes "you turned it off" from "another app owns that chord"
 * from "this desktop session cannot host global shortcuts at all", which are
 * the failures this feature must never swallow.
 */
export interface QuickEntryRegistration {
  error: null | QuickEntryRegistrationError
  registered: boolean
  shortcut: string
  /** Probe reason when `error` is `'unavailable'`; absent otherwise. */
  detail?: string
}

export type QuickEntryRegistrationError = 'invalid' | 'taken' | 'unavailable'

/**
 * Result of asking the session whether the GlobalShortcuts portal is actually
 * serviceable. `detail` carries a short human-readable reason for logs.
 */
export interface GlobalShortcutsPortalProbeResult {
  available: boolean
  detail?: string
}

/** Async probe shape (injected for testing). */
export type GlobalShortcutsPortalProbe = () => Promise<GlobalShortcutsPortalProbeResult>

const PORTAL_DBUS_TIMEOUT_MS = 1_500

/**
 * Ask the session bus whether `org.freedesktop.portal.GlobalShortcuts` — the
 * interface Electron's Linux/Wayland globalShortcut path goes through — is
 * reachable. `busctl --user` is the primary tool with `dbus-send` as the
 * fallback; a successful DBus Peer.Ping against the portal's bus name proves
 * the portal process is up and owning that name. Anything else (tools missing,
 * portal name unowned, session bus dead, timeout) means registration attempts
 * will fail for reasons that are NOT another application holding the chord.
 */
export const probeGlobalShortcutsPortalViaCli: GlobalShortcutsPortalProbe = async () => {
  const attempts: Array<{ args: string[]; command: string }> = [
    {
      args: [
        '--user',
        'call',
        'org.freedesktop.portal.Desktop',
        '/org/freedesktop/portal/desktop',
        'org.freedesktop.DBus.Peer',
        'Ping'
      ],
      command: 'busctl'
    },
    {
      args: [
        '--session',
        '--dest=org.freedesktop.portal.Desktop',
        '--type=method_call',
        '--print-reply',
        '/org/freedesktop/portal/desktop',
        'org.freedesktop.DBus.Peer.Ping'
      ],
      command: 'dbus-send'
    }
  ]

  const failures: string[] = []

  for (const attempt of attempts) {
    try {
      await promisify(execFile)(attempt.command, attempt.args, { timeout: PORTAL_DBUS_TIMEOUT_MS })

      return { available: true }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error)

      failures.push(`${attempt.command}: ${message.split('\n')[0]?.slice(0, 160) || 'failed'}`)
    }
  }

  return { available: false, detail: failures.join(' | ') }
}

export interface QuickEntryShortcutController {
  /** Registration state as of the last apply. */
  current(): QuickEntryRegistration
  /** Release the shortcut (quit / feature off). Idempotent. */
  dispose(): void
  /** Re-register to match `settings`. Returns the resulting state. */
  apply(settings: QuickEntrySettings): Promise<QuickEntryRegistration>
}

/**
 * Platforms whose globalShortcut registration can fail because the desktop
 * session's shortcut service is missing or unreachable rather than because
 * another application owns the chord: Linux/Wayland routes the chord through
 * the GlobalShortcuts portal, and remote/secure Windows sessions can refuse
 * RegisterHotKey without a real conflict. On those, a failed registration
 * must be probed before blaming a conflict. macOS registration goes through
 * Carbon APIs with no such session service, so it keeps the plain 'taken'
 * report.
 */
function sessionCanRefuseShortcutsWithoutConflict(): boolean {
  try {
    return process.platform === 'linux' || process.platform === 'win32'
  } catch {
    return false
  }
}

/**
 * Owns the one live global accelerator. Single resolver so every caller — boot,
 * the settings write, quit — gets the same answer and we can never leak two
 * registrations for one feature.
 *
 * Disabled settings never touch `register()` at all: a user who turned Quick
 * Entry off must not have their chord silently held hostage.
 *
 * When a registration attempt fails, a Linux/Wayland session is probed for the
 * GlobalShortcuts portal (the path Electron's globalShortcut goes through
 * there) before blaming another application: on KDE/GNOME Wayland a dead or
 * unreachable portal fails every chord, and "already taken" sends users
 * chasing conflicts that do not exist (#95132). The probe is injectable so
 * tests stay hermetic.
 */
export function createQuickEntryShortcut(
  globalShortcut: GlobalShortcutLike,
  onTrigger: () => void,
  probePortal: GlobalShortcutsPortalProbe = probeGlobalShortcutsPortalViaCli
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

  const tryRegister = (accelerator: string): boolean => {
    try {
      return globalShortcut.register(accelerator, onTrigger)
    } catch {
      return false
    }
  }

  return {
    async apply(settings) {
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

      if (!parsed.accelerator) {
        state = { error: 'invalid', registered: false, shortcut }

        return state
      }

      // A chord the OS reports as already held is DIRECT evidence of a
      // conflict — no probe needed, report 'taken'.
      if (globalShortcut.isRegistered(parsed.accelerator)) {
        active = null
        state = { error: 'taken', registered: false, shortcut: parsed.accelerator }

        return state
      }

      if (tryRegister(parsed.accelerator)) {
        active = parsed.accelerator
        state = { error: null, registered: true, shortcut: parsed.accelerator }

        return state
      }

      // Ambiguous failure: `register()` refused or threw without saying who
      // owns the chord. On Linux/Wayland that is usually the session's
      // GlobalShortcuts portal being unreachable — not a conflict — so probe
      // before reporting 'taken' (#95132). A probe that throws defaults to
      // the historical 'taken' report.
      let error: QuickEntryRegistrationError = 'taken'
      let detail: undefined | string

      if (sessionCanRefuseShortcutsWithoutConflict()) {
        try {
          const probe = await probePortal()

          if (!probe.available) {
            error = 'unavailable'
            detail = probe.detail
          }
        } catch {
          // Probe failures keep the historical 'taken' report.
        }
      }

      active = null

      if (detail) {
        state = { error, registered: false, shortcut: parsed.accelerator, detail }
      } else {
        state = { error, registered: false, shortcut: parsed.accelerator }
      }

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
 * Where the quick window opens on a given display work area. Centered
 * horizontally, a fraction down from the top, and clamped so it stays fully
 * inside the work area on small/odd displays.
 */
export function quickEntryWindowBounds(workArea?: { height: number; width: number; x: number; y: number }): {
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

  const x = Math.round(workArea.x + (workArea.width - width) / 2)
  const maxY = workArea.y + workArea.height - height
  const y = Math.round(Math.min(Math.max(workArea.y, workArea.y + workArea.height * QUICK_ENTRY_TOP_FRACTION), maxY))

  return { height, width, x, y }
}

export { DEFAULT_QUICK_ENTRY_SHORTCUT, QUICK_ENTRY_TOP_FRACTION, QUICK_ENTRY_WINDOW_HEIGHT, QUICK_ENTRY_WINDOW_WIDTH }
