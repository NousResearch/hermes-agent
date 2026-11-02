import { ipcMain } from 'electron'

import { type QuickEntrySettings, sanitizeQuickEntrySettings } from './quick-entry'
import { readQuickEntrySettings, writeQuickEntrySettings } from './quick-entry-settings'

type ShortcutState = { error: null | string; registered: boolean; shortcut: string }

export type QuickEntrySettingsIpcDeps = {
  /** Re-apply the preference to the OS accelerator; returns what actually resulted. */
  apply: (settings: QuickEntrySettings) => unknown
  /** Push the new default into a live capture window, if one is open. */
  notifyWindow: () => void
  /** The live registration state, for the renderer's mount read. */
  shortcutState: () => ShortcutState
}

/**
 * The renderer reads the live registration state on settings mount and writes
 * the preference back. Main is authoritative — it owns the OS accelerator — so
 * both handlers return the state that ACTUALLY resulted, including
 * `registered: false` + `error: 'taken'` when another app owns the chord.
 */
export function registerQuickEntrySettingsIpc(deps: QuickEntrySettingsIpcDeps): void {
  ipcMain.handle('hermes:quick-entry:settings:get', async () => {
    const settings = readQuickEntrySettings()
    const state = deps.shortcutState()

    // Ground truth is what the last apply produced; the shortcut we report is
    // the live one (a saved-but-rejected chord still shows what was asked for).
    return {
      enabled: settings.enabled,
      error: state.error,
      registered: state.registered,
      shortcut: settings.enabled ? state.shortcut : settings.shortcut,
      target: settings.target
    }
  })

  ipcMain.handle('hermes:quick-entry:settings:set', async (_event, patch: Partial<QuickEntrySettings> | undefined) => {
    const current = readQuickEntrySettings()

    const next = sanitizeQuickEntrySettings({
      enabled: patch?.enabled === undefined ? current.enabled : patch.enabled === true,
      shortcut: typeof patch?.shortcut === 'string' && patch.shortcut.trim() ? patch.shortcut : current.shortcut,
      target: patch?.target === undefined ? current.target : patch.target
    })

    writeQuickEntrySettings(next)

    const applied = deps.apply(next)

    // A live quick window keeps its reducer state across summons, so the new
    // default target must reach it now — the next summon is what reads it.
    deps.notifyWindow()

    return applied
  })
}
