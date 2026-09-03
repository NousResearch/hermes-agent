import { describe, expect, it } from 'vitest'

import { isProtectedTeamHermesExecutable } from './protected-edition'

describe('protected Team Hermes edition', () => {
  it('recognizes a versioned external Windows package', () => {
    expect(
      isProtectedTeamHermesExecutable(
        'C:\\Users\\someone\\AppData\\Local\\hermes\\desktop-builds\\team-hermes-custom-20260821-021605-229b3de87f1f\\win-unpacked\\Hermes.exe'
      )
    ).toBe(true)
  })

  it('does not protect the updater-managed official release', () => {
    expect(
      isProtectedTeamHermesExecutable(
        'C:\\Users\\someone\\AppData\\Local\\hermes\\hermes-agent\\apps\\desktop\\release\\win-unpacked\\Hermes.exe'
      )
    ).toBe(false)
  })

  it('recognizes the separately branded Team Hermes executable', () => {
    expect(
      isProtectedTeamHermesExecutable(
        'C:\\Users\\someone\\AppData\\Local\\hermes\\desktop-builds\\team-hermes-desktop-20260822-abcdef123456\\win-unpacked\\Team Hermes Desktop.exe'
      )
    ).toBe(true)
  })
})
