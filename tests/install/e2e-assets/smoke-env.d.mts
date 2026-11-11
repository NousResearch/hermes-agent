export interface SmokeEnvironment extends Record<string, string> {
  HOME: string
  USERPROFILE: string
  HERMES_HOME: string
  HERMES_DESKTOP_USER_DATA_DIR: string
  XDG_CONFIG_HOME: string
  XDG_DATA_HOME: string
  XDG_CACHE_HOME: string
  APPDATA: string
  LOCALAPPDATA: string
}

export function within(root: string, candidate: string): boolean
export function smokeEnvironment(inherited: NodeJS.ProcessEnv, home: string, userData: string): SmokeEnvironment
export function isolateUpdateWindowEnvironment<T extends Record<string, string>>(
  env: T & { HERMES_DESKTOP_USER_DATA_DIR: string },
): T & { HERMES_DESKTOP_USER_DATA_DIR: string }
export function isolatedElectronArgs(args: string[], userData: string): string[]
export function updateWindowEnvironment(inherited: NodeJS.ProcessEnv, root: string, origin: 'source' | 'bundled'): SmokeEnvironment
