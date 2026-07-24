import fs from 'node:fs'

export interface TrayPreferences {
  /** Master switch: create/use the Windows tray. */
  enabled: boolean
  /** Hide the main window to tray on title-bar close (Windows). */
  closeToTray: boolean
  /** Start the first main window hidden when tray is available. */
  startInTray: boolean
  /** Ask the pet to pop out after the renderer reports availability. */
  popOutPetOnStartup: boolean
  /** Register Hermes with the OS login items (Windows). */
  launchAtLogin: boolean
}

export const DEFAULT_TRAY_PREFERENCES: TrayPreferences = {
  enabled: true,
  closeToTray: true,
  startInTray: false,
  popOutPetOnStartup: false,
  launchAtLogin: false
}

type ReadFs = {
  readFileSync(path: string, encoding: 'utf8'): string
}
type WriteFs = {
  writeFileSync(path: string, data: string, encoding: 'utf8'): void
  renameSync(oldPath: string, newPath: string): void
}

function asBoolean(value: unknown, fallback: boolean): boolean {
  return typeof value === 'boolean' ? value : fallback
}

export function parseTrayPreferences(value: unknown): TrayPreferences {
  if (!value || typeof value !== 'object') {
    return { ...DEFAULT_TRAY_PREFERENCES }
  }

  const record = value as Record<string, unknown>

  return {
    enabled: asBoolean(record.enabled, DEFAULT_TRAY_PREFERENCES.enabled),
    closeToTray: asBoolean(record.closeToTray, DEFAULT_TRAY_PREFERENCES.closeToTray),
    startInTray: asBoolean(record.startInTray, DEFAULT_TRAY_PREFERENCES.startInTray),
    popOutPetOnStartup: asBoolean(
      record.popOutPetOnStartup,
      DEFAULT_TRAY_PREFERENCES.popOutPetOnStartup
    ),
    launchAtLogin: asBoolean(record.launchAtLogin, DEFAULT_TRAY_PREFERENCES.launchAtLogin)
  }
}

export function loadTrayPreferences(filePath: string, fileSystem: ReadFs = fs): TrayPreferences {
  try {
    const raw = fileSystem.readFileSync(filePath, 'utf8')

    return parseTrayPreferences(JSON.parse(String(raw)))
  } catch {
    return { ...DEFAULT_TRAY_PREFERENCES }
  }
}

export function saveTrayPreferences(
  filePath: string,
  preferences: TrayPreferences,
  fileSystem: WriteFs = fs
): void {
  const temporaryPath = `${filePath}.tmp`
  const normalized = parseTrayPreferences(preferences)

  fileSystem.writeFileSync(temporaryPath, JSON.stringify(normalized, null, 2), 'utf8')
  fileSystem.renameSync(temporaryPath, filePath)
}
