import fs from 'node:fs'
import path from 'node:path'

export interface PreviewUblockSettings {
  enabled: boolean
}

const DEFAULT_PREVIEW_UBLOCK_SETTINGS: PreviewUblockSettings = { enabled: false }

export function readPreviewUblockSettings(settingsPath: string): PreviewUblockSettings {
  try {
    const value: unknown = JSON.parse(fs.readFileSync(settingsPath, 'utf8'))

    if (value && typeof value === 'object' && 'enabled' in value && typeof value.enabled === 'boolean') {
      return { enabled: value.enabled }
    }
  } catch {
    // Missing or malformed local settings use the opt-in default.
  }

  return { ...DEFAULT_PREVIEW_UBLOCK_SETTINGS }
}

export function writePreviewUblockSettings(settingsPath: string, settings: PreviewUblockSettings): void {
  const directory = path.dirname(settingsPath)
  const temporaryPath = `${settingsPath}.tmp-${process.pid}`

  fs.mkdirSync(directory, { recursive: true })
  fs.writeFileSync(temporaryPath, JSON.stringify({ enabled: settings.enabled }, null, 2), 'utf8')
  fs.renameSync(temporaryPath, settingsPath)
}
