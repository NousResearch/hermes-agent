import { describe, expect, it, vi } from 'vitest'

import {
  DEFAULT_TRAY_PREFERENCES,
  loadTrayPreferences,
  parseTrayPreferences,
  saveTrayPreferences
} from './tray-preferences'

describe('tray preferences', () => {
  it('uses safe defaults for missing, malformed, and partial values', () => {
    expect(parseTrayPreferences(null)).toEqual(DEFAULT_TRAY_PREFERENCES)
    expect(parseTrayPreferences('bad')).toEqual(DEFAULT_TRAY_PREFERENCES)
    expect(parseTrayPreferences({ startInTray: true })).toEqual({
      ...DEFAULT_TRAY_PREFERENCES,
      startInTray: true
    })
    expect(parseTrayPreferences({ enabled: false, closeToTray: false, launchAtLogin: true })).toEqual({
      enabled: false,
      closeToTray: false,
      startInTray: false,
      popOutPetOnStartup: false,
      launchAtLogin: true
    })
  })

  it('loads malformed JSON without throwing', () => {
    expect(loadTrayPreferences('prefs.json', { readFileSync: () => '{' })).toEqual(
      DEFAULT_TRAY_PREFERENCES
    )
  })

  it('writes independent values through a temporary replacement', () => {
    const writeFileSync = vi.fn()
    const renameSync = vi.fn()

    saveTrayPreferences(
      'prefs.json',
      {
        enabled: true,
        closeToTray: true,
        startInTray: false,
        popOutPetOnStartup: true,
        launchAtLogin: true
      },
      { writeFileSync, renameSync }
    )

    expect(writeFileSync).toHaveBeenCalledWith(
      'prefs.json.tmp',
      JSON.stringify(
        {
          enabled: true,
          closeToTray: true,
          startInTray: false,
          popOutPetOnStartup: true,
          launchAtLogin: true
        },
        null,
        2
      ),
      'utf8'
    )
    expect(renameSync).toHaveBeenCalledWith('prefs.json.tmp', 'prefs.json')
  })
})
