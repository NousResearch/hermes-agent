/**
 * Tray quick access — the preferences behind the system-tray entry points.
 *
 * `minimize-to-tray.ts` owns the icon itself and the hide/restore mechanics;
 * this module owns the small, device-local preference set that decides WHAT
 * the tray icon does (open the Mini Assistant on a fresh conversation, or just
 * raise the app window) and whether Hermes comes back on its own at login.
 *
 * Everything here is pure — no `electron` import — so the defaults, the legacy
 * migration and the menu shape are unit-testable without booting Electron,
 * same split as `quick-entry.ts` / `window-state.ts`. `main.ts` owns the file
 * I/O and the real `app.setLoginItemSettings`.
 */

export interface TrayPreferences {
  /** Start Hermes when the user logs into this machine (Windows/macOS; a
   *  no-op on Linux, where the desktop itself decides autostart). */
  launchAtLogin: boolean
  /** Left-clicking the tray icon opens the Mini Assistant on a NEW
   *  conversation with the caret already in the composer. Off means the
   *  click merely raises the ordinary app window (the pre-tray-access
   *  behavior, which users have muscle memory for). */
  openNewConversationOnClick: boolean
}

export const TRAY_PREFERENCES_DEFAULTS: TrayPreferences = {
  launchAtLogin: false,
  openNewConversationOnClick: true
}

/**
 * Raw JSON → clean preferences. Anything missing, malformed or of the wrong
 * type falls back to the shipped default rather than throwing: a corrupted
 * preference file must cost the user their habit, never the tray.
 *
 * Note that "show the tray icon" is deliberately NOT in here — it is
 * minimize-to-tray's own `enabled` preference, and the icon's existence
 * follows from it. Duplicating it here would give the same switch two homes
 * that can disagree.
 */
export function sanitizeTrayPreferences(raw: unknown): TrayPreferences {
  const record = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>

  return {
    launchAtLogin: record.launchAtLogin === true,
    openNewConversationOnClick:
      record.openNewConversationOnClick === undefined
        ? TRAY_PREFERENCES_DEFAULTS.openNewConversationOnClick
        : record.openNewConversationOnClick === true
  }
}

/** The menu items the tray icon offers, as data. */
export type TrayMenuAction = 'new-conversation' | 'open-app' | 'settings' | 'quit'

/** Rendered order. A unit test asserts it, because the click and the menu are
 *  the only two ways a user reaches the app from the tray and their order is
 *  the muscle memory. */
export const TRAY_MENU_ACTIONS: readonly TrayMenuAction[] = [
  'new-conversation',
  'open-app',
  'settings',
  'quit'
]

export interface TrayMenuLabels {
  newConversation: string
  openApp: string
  quit: string
  settings: string
}

export const DEFAULT_TRAY_MENU_LABELS: TrayMenuLabels = {
  newConversation: 'New conversation',
  openApp: 'Open Hermes',
  quit: 'Quit Hermes',
  settings: 'Settings'
}

export type TrayMenuEntry =
  | { action: TrayMenuAction; label: string }
  | { action?: undefined; label?: undefined; type: 'separator' }

/**
 * The context menu, as plain data. `minimize-to-tray.ts` turns this into a
 * real Electron menu through an injected action sink, so the shape (items,
 * order, the separator before Quit) is provable here.
 *
 * Labels stay English: they are built by the main process, which has no
 * renderer locale, and Electron menu items cannot participate in the
 * renderer's language packs. The in-app settings rows that configure them are
 * translated.
 */
export function buildTrayMenuTemplate(labels: TrayMenuLabels = DEFAULT_TRAY_MENU_LABELS): TrayMenuEntry[] {
  return [
    { action: 'new-conversation', label: labels.newConversation },
    { action: 'open-app', label: labels.openApp },
    { action: 'settings', label: labels.settings },
    { type: 'separator' },
    { action: 'quit', label: labels.quit }
  ]
}
