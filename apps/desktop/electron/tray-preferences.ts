import fs from 'node:fs'

export interface TrayPreferences {
  startInTray: boolean
  popOutPetOnStartup: boolean
}

export const DEFAULT_TRAY_PREFERENCES: TrayPreferences = {
  startInTray: false,
  popOutPetOnStartup: false
}

type ReadFs = {
  readFileSync(path: string, encoding: 'utf8'): string
}
type WriteFs = {
  writeFileSync(path: string, data: string, encoding: 'utf8'): void
  renameSync(oldPath: string, newPath: string): void
}

export function parseTrayPreferences(value: unknown): TrayPreferences {
  if (!value || typeof value !== 'object') {
    return { ...DEFAULT_TRAY_PREFERENCES }
  }

  const record = value as Record<string, unknown>

  return {
    startInTray: typeof record.startInTray === 'boolean' ? record.startInTray : false,
    popOutPetOnStartup: typeof record.popOutPetOnStartup === 'boolean' ? record.popOutPetOnStartup : false
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

export function saveTrayPreferences(filePath: string, preferences: TrayPreferences, fileSystem: WriteFs = fs): void {
  const temporaryPath = `${filePath}.tmp`
  const normalized = parseTrayPreferences(preferences)

  fileSystem.writeFileSync(temporaryPath, JSON.stringify(normalized, null, 2), 'utf8')
  fileSystem.renameSync(temporaryPath, filePath)
}
