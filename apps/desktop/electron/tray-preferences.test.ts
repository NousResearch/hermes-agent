import { describe, expect, it, vi } from 'vitest'

import { DEFAULT_TRAY_PREFERENCES, loadTrayPreferences, parseTrayPreferences, saveTrayPreferences } from './tray-preferences'

describe('tray preferences', () => {
  it('uses safe defaults for missing, malformed, and partial values', () => {
    expect(parseTrayPreferences(null)).toEqual(DEFAULT_TRAY_PREFERENCES)
    expect(parseTrayPreferences('bad')).toEqual(DEFAULT_TRAY_PREFERENCES)
    expect(parseTrayPreferences({ startInTray: true })).toEqual({
      startInTray: true,
      popOutPetOnStartup: false
    })
  })

  it('loads malformed JSON without throwing', () => {
    expect(loadTrayPreferences('prefs.json', { readFileSync: () => '{' })).toEqual(DEFAULT_TRAY_PREFERENCES)
  })

  it('writes independent values through a temporary replacement', () => {
    const writeFileSync = vi.fn()
    const renameSync = vi.fn()

    saveTrayPreferences(
      'prefs.json',
      { startInTray: false, popOutPetOnStartup: true },
      { writeFileSync, renameSync }
    )

    expect(writeFileSync).toHaveBeenCalledWith(
      'prefs.json.tmp',
      JSON.stringify({ startInTray: false, popOutPetOnStartup: true }, null, 2),
      'utf8'
    )
    expect(renameSync).toHaveBeenCalledWith('prefs.json.tmp', 'prefs.json')
  })
})
