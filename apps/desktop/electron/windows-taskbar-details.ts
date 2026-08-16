import path from 'node:path'

export const WINDOWS_APP_USER_MODEL_ID = 'com.nousresearch.hermes'
export const WINDOWS_DEV_APP_USER_MODEL_ID = `${WINDOWS_APP_USER_MODEL_ID}.Dev`

export interface WindowsTaskbarDetailsTarget {
  setAppDetails(details: {
    appId: string
    appIconIndex?: number
    appIconPath?: string
    relaunchCommand: string
    relaunchDisplayName: string
  }): void
}

export interface WindowsTaskbarDetailsOptions {
  appId: string
  iconPath?: string
  isWindows: boolean
  relaunchCommand: string
  relaunchDisplayName: string
}

export interface WindowsRelaunchCommandOptions {
  executablePath: string
  appEntryPath?: string
  isDefaultApp: boolean
}

export function resolveWindowsAppUserModelId(isDefaultApp: boolean): string {
  return isDefaultApp ? WINDOWS_DEV_APP_USER_MODEL_ID : WINDOWS_APP_USER_MODEL_ID
}

export function resolveWindowsDevRelaunchAppPath(isDefaultApp: boolean, argv: unknown): string | undefined {
  if (!isDefaultApp || !Array.isArray(argv)) {
    return undefined
  }

  const appEntryPath = argv.slice(1).find(argument => typeof argument === 'string' && !argument.startsWith('-'))

  if (typeof appEntryPath !== 'string' || appEntryPath.length === 0) {
    return undefined
  }

  // CI exercises this Windows-only helper on Linux too. node:path resolves a
  // drive-letter path relative to the POSIX cwd there, so preserve an already
  // absolute Windows path with the win32 implementation.
  return path.win32.isAbsolute(appEntryPath)
    ? path.win32.normalize(appEntryPath)
    : path.resolve(appEntryPath)
}

function quoteWindowsCommandArgument(value: string): string {
  return `"${value.replace(/(\\*)"/g, '$1$1\\"').replace(/(\\*)$/, '$1$1')}"`
}

export function buildWindowsRelaunchCommand({
  executablePath,
  appEntryPath,
  isDefaultApp
}: WindowsRelaunchCommandOptions): string {
  const command = quoteWindowsCommandArgument(executablePath)

  return isDefaultApp && appEntryPath ? `${command} ${quoteWindowsCommandArgument(appEntryPath)}` : command
}

export function configureWindowsTaskbarDetails(
  target: WindowsTaskbarDetailsTarget,
  { appId, iconPath, isWindows, relaunchCommand, relaunchDisplayName }: WindowsTaskbarDetailsOptions
): void {
  if (!isWindows) {
    return
  }

  target.setAppDetails({
    appId,
    ...(iconPath ? { appIconIndex: 0, appIconPath: iconPath } : {}),
    relaunchCommand,
    relaunchDisplayName
  })
}
