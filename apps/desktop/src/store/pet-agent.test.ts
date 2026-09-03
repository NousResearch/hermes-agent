import { beforeEach, describe, expect, it } from 'vitest'

import {
  $desktopPetAgentProfile,
  DESKTOP_PET_AGENT_PROFILE_KEY,
  listenForDesktopPetAgentProfile,
  normalizeDesktopPetAgentProfile,
  setDesktopPetAgentProfile
} from './pet-agent'

describe('desktop pet agent selection', () => {
  beforeEach(() => {
    window.localStorage.clear()
    $desktopPetAgentProfile.set('default')
  })

  it('normalizes profile keys and rejects unsafe values', () => {
    expect(normalizeDesktopPetAgentProfile(' Warren ')).toBe('warren')
    expect(normalizeDesktopPetAgentProfile('jarvis/../../')).toBe('default')
    expect(normalizeDesktopPetAgentProfile('')).toBe('default')
  })

  it('persists the selected character', () => {
    setDesktopPetAgentProfile('jarvis')

    expect($desktopPetAgentProfile.get()).toBe('jarvis')
    expect(window.localStorage.getItem(DESKTOP_PET_AGENT_PROFILE_KEY)).toBe('jarvis')
  })

  it('adopts a selection made by the Quick Entry renderer', () => {
    const dispose = listenForDesktopPetAgentProfile()

    window.dispatchEvent(
      new StorageEvent('storage', {
        key: DESKTOP_PET_AGENT_PROFILE_KEY,
        newValue: 'sabiska'
      })
    )

    expect($desktopPetAgentProfile.get()).toBe('sabiska')
    dispose()
  })
})

