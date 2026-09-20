import fs from 'node:fs'

import { parse } from 'yaml'

const DISABLED_BACKUP_MODES = new Set(['off', 'false', 'none', 'disabled'])

export function preUpdateBackupEnabled(config: unknown): boolean {
  if (!config || typeof config !== 'object' || Array.isArray(config)) {
    return true
  }

  const updates = (config as Record<string, unknown>).updates

  if (!updates || typeof updates !== 'object' || Array.isArray(updates)) {
    return true
  }

  const value = (updates as Record<string, unknown>).pre_update_backup

  if (value === undefined) {
    return true
  }

  if (value === false || value === null) {
    return false
  }

  return typeof value !== 'string' || !DISABLED_BACKUP_MODES.has(value.trim().toLowerCase())
}

export function readPreUpdateBackupEnabled(configPath: string): boolean {
  try {
    return preUpdateBackupEnabled(parse(fs.readFileSync(configPath, 'utf8')))
  } catch {
    // Match the Python updater: an unreadable or invalid config falls back to
    // the safe default instead of silently disabling recovery data.
    return true
  }
}