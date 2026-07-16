import os from 'node:os'
import path from 'node:path'

import { normalizeHermesHomeRoot } from './backend-env'
import { readWindowsUserEnvVar } from './windows-user-env'

interface HermesHomeOptions {
  env?: NodeJS.ProcessEnv
  userDataOverride?: string
  isWindows?: boolean
  homeDir?: string
  directoryExists?: (target: string) => boolean
  readUserEnvVar?: (name: string) => string | null
  pathModule?: typeof path.win32
  normalizeRoot?: typeof normalizeHermesHomeRoot
}

function windowsDefaultHermesHome({
  env = process.env,
  homeDir = os.homedir(),
  pathModule = path.win32
}: Pick<HermesHomeOptions, 'env' | 'homeDir' | 'pathModule'> = {}): string {
  const localAppData = String(env.LOCALAPPDATA || '').trim()
  const base = localAppData || pathModule.join(homeDir, 'AppData', 'Local')

  return pathModule.join(base, 'hermes')
}

function resolveHermesHome({
  env = process.env,
  userDataOverride = env.HERMES_DESKTOP_USER_DATA_DIR,
  isWindows = process.platform === 'win32',
  homeDir = os.homedir(),
  directoryExists = () => false,
  readUserEnvVar = readWindowsUserEnvVar,
  pathModule = isWindows ? path.win32 : path.posix,
  normalizeRoot = normalizeHermesHomeRoot
}: HermesHomeOptions = {}): string {
  if (env.HERMES_HOME) {
    return normalizeRoot(env.HERMES_HOME, { pathModule })
  }

  if (userDataOverride) {
    return pathModule.join(pathModule.resolve(userDataOverride), 'hermes-home')
  }

  if (isWindows) {
    const fromRegistry = readUserEnvVar('HERMES_HOME')

    if (fromRegistry) {
      return normalizeRoot(fromRegistry, { pathModule })
    }

    const localAppData = windowsDefaultHermesHome({ env, homeDir, pathModule })
    const legacy = pathModule.join(homeDir, '.hermes')

    if (!directoryExists(localAppData) && directoryExists(legacy)) {
      return legacy
    }

    return localAppData
  }

  return pathModule.join(homeDir, '.hermes')
}

export { resolveHermesHome, windowsDefaultHermesHome }
