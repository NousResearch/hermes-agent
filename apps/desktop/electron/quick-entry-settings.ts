import fs from 'node:fs'
import path from 'node:path'

import { app } from 'electron'

import { type QuickEntrySettings, sanitizeQuickEntrySettings } from './quick-entry'

// Main owns this file: the preference must survive a cold launch without the
// renderer ever visiting Settings, the same authority split as keep-awake.
let logFailure: (message: string) => void = () => {}

/** Point the store at the app's logger once, at startup. */
export function initQuickEntrySettings(logger: (message: string) => void): void {
  logFailure = logger
}

function configPath(): string {
  return path.join(app.getPath('userData'), 'quick-entry.json')
}

export function readQuickEntrySettings(): QuickEntrySettings {
  try {
    return sanitizeQuickEntrySettings(JSON.parse(fs.readFileSync(configPath(), 'utf8')))
  } catch {
    // Missing / unreadable / malformed → shipped defaults (enabled, default chord).
    return sanitizeQuickEntrySettings(undefined)
  }
}

export function writeQuickEntrySettings(settings: QuickEntrySettings): void {
  try {
    fs.mkdirSync(path.dirname(configPath()), { recursive: true })
    fs.writeFileSync(configPath(), JSON.stringify(settings, null, 2), 'utf8')
  } catch (error) {
    logFailure(`[quick-entry] write failed: ${(error as Error).message}`)
  }
}
