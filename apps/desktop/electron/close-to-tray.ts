/**
 * Close to tray — closing the window hides Hermes instead of ending it.
 *
 * Closing the last window used to end the process: the backend is torn down and
 * any turn in flight dies mid-tool-call, so a user who only meant to tidy their
 * desktop loses the work. Claude Desktop and Codex keep the app alive behind a
 * tray icon for exactly that reason, and a long-running agent belongs there too.
 *
 * This module owns the preference, the close decision and the tray menu
 * contents; `main.ts` owns the Electron objects it is handed through
 * `CloseToTrayHost` (same split as `quit-guard.ts`, which owns copy while
 * `main.ts` owns the dialog). Every rule here is therefore provable without
 * booting Electron.
 */

import fs from 'node:fs'
import path from 'node:path'

export interface CloseToTrayPrefs {
  /** Closing the primary window hides it to the tray instead of quitting. */
  enabled: boolean
  /** The one-time "Hermes keeps running" balloon has been shown. */
  noticeShown: boolean
}

export const CLOSE_TO_TRAY_PREFS_FILE = 'close-to-tray.json'

/** Shipped default: on. The close button hiding the app IS the requested behaviour. */
export const DEFAULT_CLOSE_TO_TRAY_PREFS: CloseToTrayPrefs = { enabled: true, noticeShown: false }

/**
 * How long a `before-quit` still counts as "this close belongs to a real quit".
 *
 * `before-quit` is the only synchronous signal a close carries, and a quit a
 * later handler abandons (the active-work prompt's "Keep Running", a managed
 * update's re-entry) must not leave the latch stuck on — otherwise the next
 * window close would quit for real instead of hiding. Electron closes the
 * windows within milliseconds of `before-quit`, so seconds are generous.
 */
export const QUIT_LATCH_MS = 5_000

export type CloseAction = 'close' | 'hide'

/** Coerce whatever is on disk (possibly corrupt, possibly from a future build). */
export function normalizeCloseToTrayPrefs(raw: unknown): CloseToTrayPrefs {
  const record = raw && typeof raw === 'object' ? (raw as Record<string, unknown>) : {}

  return {
    enabled: typeof record.enabled === 'boolean' ? record.enabled : DEFAULT_CLOSE_TO_TRAY_PREFS.enabled,
    noticeShown: record.noticeShown === true
  }
}

/** Never throws: a missing or unreadable pref file must not stop the app booting. */
export function readCloseToTrayPrefs(filePath: string, fileSystem: typeof fs = fs): CloseToTrayPrefs {
  try {
    return normalizeCloseToTrayPrefs(JSON.parse(fileSystem.readFileSync(filePath, 'utf8')))
  } catch {
    return { ...DEFAULT_CLOSE_TO_TRAY_PREFS }
  }
}

/** Write-then-rename so a crash mid-write cannot leave a half-written prefs file. */
export function writeCloseToTrayPrefs(
  filePath: string,
  prefs: CloseToTrayPrefs,
  fileSystem: typeof fs = fs
): boolean {
  try {
    fileSystem.mkdirSync(path.dirname(filePath), { recursive: true })
    const temporary = `${filePath}.tmp`
    fileSystem.writeFileSync(temporary, `${JSON.stringify(prefs, null, 2)}\n`, { encoding: 'utf8', mode: 0o600 })
    fileSystem.renameSync(temporary, filePath)

    return true
  } catch {
    return false
  }
}

export interface CloseDecisionContext {
  /** The tray icon is actually up, so there is a way back to a hidden window. */
  trayAvailable: boolean
  /** The close belongs to a real quit (app.quit meaningful, hand-off, OS shutdown). */
  isQuitting: boolean
  /** The preference. */
  enabled: boolean
}

/**
 * Whether the window's `close` event means "hide" or "really close".
 *
 * Hiding without a tray icon strands the user with a running process and no
 * window and no way back, so `trayAvailable` outranks the preference: the
 * feature degrades to the old behaviour instead of trapping the app.
 */
export function decideCloseAction({ trayAvailable, isQuitting, enabled }: CloseDecisionContext): CloseAction {
  return enabled && trayAvailable && !isQuitting ? 'hide' : 'close'
}

export interface CloseToTrayLabels {
  noticeBody: string
  noticeTitle: string
  quit: string
  show: string
  toggle: string
  tooltip: string
}

const LABELS = {
  en: {
    noticeBody: 'Hermes is hidden in the system tray. Click the tray icon to reopen it.',
    noticeTitle: 'Hermes is still running',
    quit: 'Quit Hermes',
    show: 'Show Hermes',
    toggle: 'Hide to tray on close',
    tooltip: 'Hermes is running'
  },
  zh: {
    noticeBody: 'Hermes 已隐藏到系统托盘，点击托盘图标可以重新打开。',
    noticeTitle: 'Hermes 仍在运行',
    quit: '退出 Hermes',
    show: '显示 Hermes',
    toggle: '关闭时隐藏到系统托盘',
    tooltip: 'Hermes 正在后台运行'
  }
} satisfies Record<string, CloseToTrayLabels>

/**
 * Tray copy follows the OS locale — the main process cannot see the renderer's
 * language preference (it lives in the renderer's storage), and shipping two
 * strings beats shipping a menu in a language the user does not read.
 */
export function closeToTrayLabels(locale: string): CloseToTrayLabels {
  return String(locale || '').toLowerCase().startsWith('zh') ? LABELS.zh : LABELS.en
}

export type TrayMenuEntry =
  | { kind: 'separator' }
  | { kind: 'show'; label: string }
  | { checked: boolean; kind: 'toggle'; label: string }
  | { kind: 'quit'; label: string }

/** The tray menu, as data — `main.ts` attaches the click handlers. */
export function closeToTrayMenuTemplate(prefs: CloseToTrayPrefs, labels: CloseToTrayLabels): TrayMenuEntry[] {
  return [
    { kind: 'show', label: labels.show },
    { kind: 'separator' },
    { checked: prefs.enabled, kind: 'toggle', label: labels.toggle },
    { kind: 'separator' },
    { kind: 'quit', label: labels.quit }
  ]
}

/** The slice of Electron's `Menu.buildFromTemplate` template we use. */
export interface TrayMenuTemplateItem {
  checked?: boolean
  click?: () => void
  label?: string
  type?: 'checkbox' | 'normal' | 'separator'
}

/** The slice of Electron's `Tray` we use (injected so tests need no Electron). */
export interface TrayHandle {
  destroy: () => void
  displayBalloon?: (options: { content?: string; iconType?: string; title?: string }) => void
  on: (event: string, listener: () => void) => unknown
  setContextMenu: (menu: unknown) => void
  setToolTip: (text: string) => void
}

export interface CloseToTrayWindow {
  hide: () => void
  isDestroyed: () => boolean
}

export interface CloseToTrayHost {
  buildMenu: (template: TrayMenuTemplateItem[]) => unknown
  createTray: (iconPath: string) => TrayHandle
  fileSystem?: typeof fs
  getMainWindow: () => CloseToTrayWindow | null | undefined
  /**
   * Read lazily, on first use: `app.getLocale()` is only valid after `ready`,
   * and the tray is built on the first close, never at module scope.
   */
  getLocale: () => string
  /** A real quit is under way — see QUIT_LATCH_MS. */
  isQuitting: () => boolean
  log: (message: string) => void
  prefsPath: string
  /** `app.quit()`, not `app.exit()`: the active-work guard must still get its say. */
  requestQuit: () => void
  /** Resolved app-icon path, lazily — undefined disables the tray, and with it the feature. */
  resolveIconPath: () => string | undefined
  /** Show the primary window, recreating it when it is gone. */
  showMainWindow: () => void
}

export interface CloseToTrayController {
  /** Answer the window's `close` event: 'hide' means preventDefault and stay alive. */
  decideClose: () => CloseAction
  /** Release the tray icon (the process is going away, or the feature is being torn down). */
  destroyTray: () => void
  /** Hide the primary window behind the tray icon (call after `decideClose` → 'hide'). */
  hideMainWindow: () => void
  /** True while the tray owns the process's lifetime: `window-all-closed` must not quit. */
  keepsProcessAlive: () => boolean
}

export function installCloseToTray(host: CloseToTrayHost): CloseToTrayController {
  const fileSystem = host.fileSystem ?? fs
  let prefs = readCloseToTrayPrefs(host.prefsPath, fileSystem)
  let tray: TrayHandle | null = null
  let labels: CloseToTrayLabels | null = null

  const resolvedLabels = () => (labels ??= closeToTrayLabels(host.getLocale()))

  const persist = () => {
    if (!writeCloseToTrayPrefs(host.prefsPath, prefs, fileSystem)) {
      host.log(`[tray] could not persist ${path.basename(host.prefsPath)}`)
    }
  }

  const showMainWindowForTray = () => {
    host.showMainWindow()
  }

  const toggleCloseToTray = () => {
    prefs = { ...prefs, enabled: !prefs.enabled }
    persist()
    buildMenu()
    host.log(`[tray] close to tray ${prefs.enabled ? 'enabled' : 'disabled'}`)
  }

  const actions: Record<TrayMenuEntry['kind'], (() => void) | null> = {
    quit: () => host.requestQuit(),
    separator: null,
    show: showMainWindowForTray,
    toggle: toggleCloseToTray
  }

  function buildMenu() {
    if (!tray) {
      return
    }

    const template = closeToTrayMenuTemplate(prefs, resolvedLabels()).map<TrayMenuTemplateItem>(entry => {
      if (entry.kind === 'separator') {
        return { type: 'separator' }
      }

      return {
        checked: entry.kind === 'toggle' ? entry.checked : undefined,
        click: actions[entry.kind] ?? undefined,
        label: entry.label,
        type: entry.kind === 'toggle' ? 'checkbox' : 'normal'
      }
    })

    tray.setContextMenu(host.buildMenu(template))
  }

  /**
   * Bring the tray icon up (idempotent). Created on first use rather than at
   * launch: an icon sitting in the tray of a user who never closes the window
   * is clutter, and the absence of one is the signal that the feature is off.
   */
  function ensureTray() {
    const iconPath = host.resolveIconPath()

    if (tray || !iconPath) {
      return
    }

    try {
      tray = host.createTray(iconPath)
      tray.setToolTip(resolvedLabels().tooltip)
      tray.on('click', showMainWindowForTray)
      tray.on('double-click', showMainWindowForTray)
      buildMenu()
    } catch (error) {
      // No tray ⇒ no way back from a hidden window. Fail CLOSED: leave the
      // close button meaning "close" rather than stranding the process.
      tray = null
      host.log(`[tray] tray icon unavailable, close to tray stays off: ${error?.message || error}`)
    }
  }

  return {
    decideClose() {
      if (!prefs.enabled || host.isQuitting()) {
        return 'close'
      }

      ensureTray()

      return decideCloseAction({ enabled: prefs.enabled, isQuitting: host.isQuitting(), trayAvailable: tray !== null })
    },
    destroyTray() {
      try {
        tray?.destroy()
      } catch {
        void 0
      }

      tray = null
    },
    hideMainWindow() {
      const window = host.getMainWindow()

      if (window && !window.isDestroyed()) {
        window.hide()
      }

      if (prefs.noticeShown) {
        return
      }

      prefs = { ...prefs, noticeShown: true }
      persist()

      // Windows-only balloon: a window that vanishes into the notification area
      // needs one line saying where it went, once per install.
      try {
        tray?.displayBalloon?.({ content: resolvedLabels().noticeBody, iconType: 'info', title: resolvedLabels().noticeTitle })
      } catch {
        void 0
      }
    },
    keepsProcessAlive() {
      return tray !== null && !host.isQuitting()
    }
  }
}