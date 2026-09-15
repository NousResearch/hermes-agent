import path from 'node:path'

/**
 * The one supported Windows login registration for the desktop shell.
 *
 * This module deliberately does not touch the registry directly. Electron owns
 * the per-user registration and the caller supplies the exact executable path
 * for the installed desktop app.
 */

export const DESKTOP_LOGIN_STARTUP_ARGS = ['--start-minimized'] as const
// Same AUMID as package.json build.appId and the native notification identity.
export const DESKTOP_LOGIN_ITEM_NAME = 'com.nousresearch.hermes'

/** The subset of Electron's readback needed after qualifying it with path/args. */
export type DesktopLoginItemSettings = {
  openAtLogin: boolean
  launchItems?: { name: string; path: string; args: string[]; scope: string; enabled: boolean }[]
}

export type DesktopLoginStartupApp = {
  getLoginItemSettings(options?: { path?: string; args?: string[] }): DesktopLoginItemSettings
  setLoginItemSettings(settings: { openAtLogin: boolean; name: string; path: string; args: string[] }): void
}

export type DesktopLoginStartupOptions = {
  app: DesktopLoginStartupApp
  executablePath: string
  platform: string
}

export type DesktopLoginStartupStatus = {
  supported: boolean
  openAtLogin: boolean
}

export type DesktopLoginStartupSupportOptions = {
  platform: string
  isPackaged: boolean
  isTest: boolean
}

function registrationArgs(): string[] {
  return [...DESKTOP_LOGIN_STARTUP_ARGS]
}

function assertExecutablePath(executablePath: string): void {
  if (executablePath.trim().length === 0) {
    throw new Error('A packaged desktop executable path is required for Windows login startup')
  }
}

function unsupportedStatus(): DesktopLoginStartupStatus {
  return {
    supported: false,
    openAtLogin: false
  }
}

export function isDesktopLoginStartupSupported(options: DesktopLoginStartupSupportOptions): boolean {
  return options.platform === 'win32' && options.isPackaged && !options.isTest
}

function readWithRegistration(
  options: DesktopLoginStartupOptions,
  registration: { path: string; args: string[] }
): DesktopLoginStartupStatus {
  // Electron qualifies openAtLogin by the same path/args supplied here; it does
  // not return those values as top-level fields in LoginItemSettings.
  const readback = options.app.getLoginItemSettings({
    // Electron parses this option as a command line for its Run-entry lookup.
    path: `"${registration.path}"`,
    args: [...registration.args]
  })

  return {
    supported: true,
    // Run-key presence alone ignores Windows StartupApproved. Do not let an
    // enabled registration mask our disabled entry. Qualified openAtLogin
    // proves exact arguments; launchItems.args omits switches in Electron 41.
    openAtLogin: Boolean(
      readback.openAtLogin &&
      readback.launchItems?.some(
        item =>
          item.scope === 'user' &&
          item.name === DESKTOP_LOGIN_ITEM_NAME &&
          item.enabled &&
          path.win32.normalize(item.path).toLowerCase() === path.win32.normalize(registration.path).toLowerCase()
      )
    )
  }
}

/**
 * Reads the authoritative Windows login-item state for the supplied path and
 * exact launch arguments. No Electron boundary is called on other platforms.
 */
export function readDesktopLoginStartup(options: DesktopLoginStartupOptions): DesktopLoginStartupStatus {
  if (options.platform !== 'win32') {
    return unsupportedStatus()
  }

  assertExecutablePath(options.executablePath)

  return readWithRegistration(options, {
    path: options.executablePath,
    args: registrationArgs()
  })
}

/**
 * Writes the per-user Windows login item only when the caller has an explicit
 * user choice, then returns Electron's qualified readback. A setter call alone
 * is never reported as successful configuration.
 */
export function configureDesktopLoginStartup(
  options: DesktopLoginStartupOptions & { openAtLogin: boolean }
): DesktopLoginStartupStatus {
  if (typeof options.openAtLogin !== 'boolean') {
    throw new Error('Login startup requires an explicit boolean choice')
  }

  if (options.platform !== 'win32') {
    return unsupportedStatus()
  }

  assertExecutablePath(options.executablePath)

  const registration = {
    openAtLogin: options.openAtLogin,
    name: DESKTOP_LOGIN_ITEM_NAME,
    path: options.executablePath,
    args: registrationArgs()
  }

  options.app.setLoginItemSettings(registration)

  return readWithRegistration(options, registration)
}

/** Returns true only when argv contains the exact first-launch token. */
export function isStartMinimizedLaunch(argv: readonly string[]): boolean {
  return argv.some(argument => argument === '--start-minimized')
}

/** The initial hidden/minimized decision; callers must use it only for cold boot. */
export function shouldStartMinimizedFirstLaunch(options: {
  platform: string
  argv: readonly string[]
  hasColdDeepLink: boolean
}): boolean {
  return options.platform === 'win32' && !options.hasColdDeepLink && isStartMinimizedLaunch(options.argv)
}

/** Deep links own their ready-aware focus path; background login never raises a window. */
export function shouldFocusSecondInstance(options: {
  platform: string
  argv: readonly string[]
  hasDeepLink: boolean
}): boolean {
  return !options.hasDeepLink && !(options.platform === 'win32' && isStartMinimizedLaunch(options.argv))
}
