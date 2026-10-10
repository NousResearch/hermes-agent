/**
 * Launch at login — the "start Hermes when I sign in" preference.
 *
 * Electron's `app.setLoginItemSettings` is the only supported way to register
 * a login item on Windows (Run key) and macOS (Login Items), so `main.ts`
 * owns the call. What it is CALLED with is decided here, pure, so the shape
 * is unit-testable: the settings object, the executable path it points at,
 * and the fact that Linux has no such API at all (the desktop's own autostart
 * facilities own that job — GNOME/KDE/XFCE ship their own, and Electron
 * silently ignores the call there anyway).
 */

/** Windows/macOS only. Electron has no login-item API on Linux. */
export const LAUNCH_AT_LOGIN_SUPPORTED = process.platform === 'darwin' || process.platform === 'win32'

export interface LaunchAtLoginSettings {
  enabled: boolean
  /** Windows: the exact binary to run at login. Packaged builds resolve to
   *  the installed Hermes executable; a dev launch resolves to the Electron
   *  binary, which is the honest answer for "run this app at login". */
  path?: string
}

/**
 * Settings object for `app.setLoginItemSettings`.
 *
 * `path` is only meaningful on Windows — macOS registers the running bundle —
 * so it is passed through when the caller has one and omitted otherwise,
 * letting Electron's own default (the current executable) stand.
 */
export function launchAtLoginSettingsFor(enabled: boolean, executablePath?: string | null): LaunchAtLoginSettings {
  const settings: LaunchAtLoginSettings = { enabled: enabled === true }

  if (executablePath) {
    settings.path = executablePath
  }

  return settings
}
