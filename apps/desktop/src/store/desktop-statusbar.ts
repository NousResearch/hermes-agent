import { atom } from 'nanostores'

import { getHermesConfigRecord, saveHermesConfig } from '@/hermes'
import type { HermesConfigRecord } from '@/types/hermes'

export type DesktopStatusbarMode = 'auto-hide' | 'off' | 'on'

export const DEFAULT_DESKTOP_STATUSBAR_MODE: DesktopStatusbarMode = 'on'

export const $desktopStatusbarMode = atom<DesktopStatusbarMode>(DEFAULT_DESKTOP_STATUSBAR_MODE)

export function normalizeDesktopStatusbarMode(value: unknown): DesktopStatusbarMode {
  return value === 'off' || value === 'auto-hide' || value === 'on' ? value : DEFAULT_DESKTOP_STATUSBAR_MODE
}

export function applyDesktopStatusbarFromConfig(
  config: { display?: { desktop_statusbar?: unknown } | null } | null | undefined
): void {
  $desktopStatusbarMode.set(normalizeDesktopStatusbarMode(config?.display?.desktop_statusbar))
}

/**
 * Persist the profile-scoped Desktop status bar preference. The atom updates
 * optimistically so the chrome responds immediately, then rolls back if the
 * whole-record config write fails.
 */
export async function persistDesktopStatusbarMode(mode: DesktopStatusbarMode): Promise<HermesConfigRecord> {
  const previous = $desktopStatusbarMode.get()

  if (previous !== mode) {
    $desktopStatusbarMode.set(mode)
  }

  try {
    const record = await getHermesConfigRecord()

    const display =
      record.display && typeof record.display === 'object' && !Array.isArray(record.display)
        ? (record.display as Record<string, unknown>)
        : {}

    const next = { ...record, display: { ...display, desktop_statusbar: mode } }

    await saveHermesConfig(next)

    return next
  } catch (error) {
    $desktopStatusbarMode.set(previous)

    throw error
  }
}
