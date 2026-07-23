import { describe, expect, it } from 'vitest'

import { buildPosixUpdateArgs, chooseDesktopUpdateStrategy } from './update-strategy'

describe('desktop update strategy', () => {
  it('uses POSIX in-app updates on macOS/Linux even when a staged updater exists', () => {
    expect(chooseDesktopUpdateStrategy({ isWindows: false, updater: '/Users/me/.hermes/hermes-setup' })).toBe(
      'posix-in-app'
    )
    expect(chooseDesktopUpdateStrategy({ isWindows: false, updater: null })).toBe('posix-in-app')
  })

  it('keeps the staged-updater handoff on Windows when available', () => {
    expect(
      chooseDesktopUpdateStrategy({
        isWindows: true,
        updater: 'C:\\Users\\me\\AppData\\Local\\hermes\\hermes-setup.exe'
      })
    ).toBe('staged-updater')
  })

  it('surfaces the manual path on Windows without a staged updater', () => {
    expect(chooseDesktopUpdateStrategy({ isWindows: true, updater: null })).toBe('manual')
  })
})

describe('POSIX update arguments', () => {
  it('pins the configured update branch', () => {
    expect(buildPosixUpdateArgs('main')).toEqual(['update', '--yes', '--branch', 'main'])
    expect(buildPosixUpdateArgs('bb/gui')).toEqual(['update', '--yes', '--branch', 'bb/gui'])
    expect(buildPosixUpdateArgs('  release/test  ')).toEqual(['update', '--yes', '--branch', 'release/test'])
  })

  it('falls back to bare update when the configured branch is absent', () => {
    expect(buildPosixUpdateArgs('')).toEqual(['update', '--yes'])
    expect(buildPosixUpdateArgs(null)).toEqual(['update', '--yes'])
  })
})
