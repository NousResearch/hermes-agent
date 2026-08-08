import type { App, IpcMain } from 'electron'

export interface LoginItemSetOptions {
  openAtLogin: boolean
  openAsHidden?: boolean
}

export interface LoginItemState {
  openAtLogin: boolean
  supported: boolean
}

const LOGIN_ITEM_SUPPORTED_PLATFORMS = ['darwin', 'win32']

/**
 * Launch Hermes Desktop at login, via Electron's native login-item API.
 *
 * Electron documents `app.setLoginItemSettings` for macOS and Windows only;
 * Linux autostart is NOT provided by this API (it would need a real
 * `~/.config/autostart/*.desktop` implementation). On unsupported platforms
 * the handlers stay registered but report `supported: false`, so the UI can
 * hide the toggle instead of presenting a control that silently no-ops.
 */
export function registerLoginItemHandlers(app: App, ipcMain: IpcMain): void {
  const supported = LOGIN_ITEM_SUPPORTED_PLATFORMS.includes(process.platform)

  ipcMain.handle('hermes:login-item:get', (): LoginItemState => {
    if (!supported) {
      return { openAtLogin: false, supported: false }
    }

    const settings = app.getLoginItemSettings()

    return { openAtLogin: settings.openAtLogin, supported: true }
  })

  ipcMain.handle('hermes:login-item:set', (_event, options: LoginItemSetOptions): LoginItemState => {
    if (!supported) {
      return { openAtLogin: false, supported: false }
    }

    // The IPC payload is renderer-controlled: coerce to booleans so
    // setLoginItemSettings never receives unexpected types.
    const openAtLogin = Boolean(options.openAtLogin)
    const openAsHidden = options.openAsHidden === undefined ? false : Boolean(options.openAsHidden)
    // In dev, Electron's default-app convention needs the entry script as an
    // argument — but only when it actually exists (an empty string would
    // register a broken login item).
    const args = process.defaultApp && process.argv[1] ? [process.argv[1]] : []

    app.setLoginItemSettings({
      openAtLogin,
      openAsHidden,
      // Keep development and packaged builds pointed at the executable that
      // registered the login item.
      path: process.execPath,
      args
    })

    // `setLoginItemSettings` is authoritative about what actually landed:
    // return its view of the state instead of echoing the requested value.
    const settings = app.getLoginItemSettings()

    return { openAtLogin: settings.openAtLogin, supported: true }
  })
}
