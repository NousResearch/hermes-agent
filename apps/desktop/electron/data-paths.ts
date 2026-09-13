import path from 'node:path'

import { normalizeHermesHomeRoot } from './backend-env'

export function platformDefaultHermesHome(
  home: string,
  env: NodeJS.ProcessEnv = process.env,
  platform: NodeJS.Platform = process.platform
): string {
  const suffix: string = env.HERMES_DATA_DIR_SUFFIX || ''

  if (platform === 'win32') {
    const base: string = env.LOCALAPPDATA?.trim() || path.win32.join(home, 'AppData', 'Local')

    return path.win32.join(base, 'hermes') + suffix
  }

  return path.posix.join(home, '.hermes') + suffix
}

export function resolveDesktopUserData(defaultPath: string, env: NodeJS.ProcessEnv = process.env): string {
  return env.HERMES_DESKTOP_USER_DATA_DIR
    ? path.resolve(env.HERMES_DESKTOP_USER_DATA_DIR)
    : defaultPath + (env.HERMES_DATA_DIR_SUFFIX || '')
}

interface HermesHomeOptions {
  home: string
  env?: NodeJS.ProcessEnv
  platform?: NodeJS.Platform
  directoryExists?: (directory: string) => boolean
  readWindowsHome?: () => string | null
}

export function resolveDesktopHermesHome({
  home,
  env = process.env,
  platform = process.platform,
  directoryExists = (): boolean => false,
  readWindowsHome = (): null => null
}: HermesHomeOptions): string {
  const paths: typeof path = platform === 'win32' ? path.win32 : path.posix

  if (env.HERMES_HOME) {
    return normalizeHermesHomeRoot(env.HERMES_HOME, { pathModule: paths })
  }

  // Fresh-install rehearsals must not touch the real Hermes home.
  if (env.HERMES_DESKTOP_USER_DATA_DIR) {
    return paths.join(paths.resolve(env.HERMES_DESKTOP_USER_DATA_DIR), 'hermes-home')
  }

  if (platform === 'win32' && env.HERMES_HOME === undefined) {
    // Explorer can miss setx changes. An explicit empty value opts out of that fallback.
    const registryHome: string | null = readWindowsHome()

    if (registryHome) {
      return normalizeHermesHomeRoot(registryHome, { pathModule: paths })
    }
  }

  const defaultHome: string = platformDefaultHermesHome(home, env, platform)

  // Keep the legacy migration for ordinary installs, not isolated suffix runs.
  if (platform === 'win32' && !env.HERMES_DATA_DIR_SUFFIX) {
    const legacy: string = paths.join(home, '.hermes')

    if (!directoryExists(defaultHome) && directoryExists(legacy)) {
      return legacy
    }
  }

  return defaultHome
}
