/**
 * The single Hermes-home authority for the desktop main process.
 *
 * Extracted from main.ts::resolveHermesHome so there is exactly one resolver:
 * main.ts (what the app actually uses) AND the dev-CDP debug descriptor
 * (dev-cdp.ts) both consume this output, so they can never diverge.
 *
 * Resolution ladder (unchanged behavior from main.ts):
 *   1. HERMES_HOME env (normalized via normalizeHermesHomeRoot)
 *   2. HERMES_DESKTOP_USER_DATA_DIR override → <userData>/hermes-home
 *      (used by test:desktop:fresh sandboxes — a fresh-install run never
 *      touches the user's real home)
 *   3. Windows only: user-scoped registry HERMES_HOME (a GUI app launched
 *      from Explorer inherits the login-time env block, so `setx` after
 *      login is invisible in process.env — #45471)
 *   4. Windows only: %LOCALAPPDATA%\hermes, honouring a legacy ~/.hermes
 *      when no LOCALAPPDATA install exists yet (don't orphan existing state)
 *   5. ~/.hermes
 *
 * Every input is injected, so this is testable without Electron.
 */

import path from 'node:path'

import { normalizeHermesHomeRoot } from './backend-env'
import { readWindowsUserEnvVar } from './windows-user-env'

export type HermesHomeInput = {
  env: Record<string, string | undefined>
  isWindows: boolean
  appHome: string
  userDataOverride?: string
  readWindowsUserEnvVar: (name: string) => string | undefined
  directoryExists: (p: string) => boolean
  /** path module matching the platform being resolved (injected for tests). */
  pathModule?: typeof path
}

export function resolveHermesHomeFromInputs(input: HermesHomeInput): string {
  const { env, isWindows, appHome, userDataOverride } = input
  const pathModule = input.pathModule ?? path

  if (env.HERMES_HOME) {
    return normalizeHermesHomeRoot(env.HERMES_HOME, { pathModule })
  }

  if (userDataOverride) {
    // path.resolve here mirrors main.ts's `path.resolve(USER_DATA_OVERRIDE)`.
    return pathModule.join(pathModule.resolve(userDataOverride), 'hermes-home')
  }

  if (isWindows) {
    const fromRegistry = input.readWindowsUserEnvVar('HERMES_HOME')

    if (fromRegistry) {
      return normalizeHermesHomeRoot(fromRegistry, { pathModule })
    }
  }

  if (isWindows && env.LOCALAPPDATA) {
    const localappdata = pathModule.join(env.LOCALAPPDATA, 'hermes')
    const legacy = pathModule.join(appHome, '.hermes')

    if (!input.directoryExists(localappdata) && input.directoryExists(legacy)) {
      return legacy
    }

    return localappdata
  }

  return pathModule.join(appHome, '.hermes')
}


// Re-exported so main.ts keeps its existing import surface working.
export { readWindowsUserEnvVar }
