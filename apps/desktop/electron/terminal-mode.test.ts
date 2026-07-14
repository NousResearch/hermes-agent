import { describe, expect, it } from 'vitest'

import {
  distributionFromWslPath,
  normalizeTerminalMode,
  parseWslDistributions,
  resolveWindowsTerminalMode,
  toWslPath
} from './terminal-mode'

describe('terminal mode', () => {
  it('normalizes unknown values to smart', () => {
    expect(normalizeTerminalMode('wsl2')).toBe('wsl2')
    expect(normalizeTerminalMode('bad')).toBe('smart')
  })

  it('parses the null-padded output returned by wsl.exe', () => {
    expect(
      parseWslDistributions(
        'U\u0000b\u0000u\u0000n\u0000t\u0000u\u0000\r\u0000\n\u0000D\u0000e\u0000b\u0000i\u0000a\u0000n\u0000'
      )
    ).toEqual(['Ubuntu', 'Debian'])
  })

  it('uses WSL for a matching UNC project in smart mode', () => {
    expect(
      resolveWindowsTerminalMode({
        configuredMode: 'smart',
        cwd: '\\\\wsl.localhost\\Ubuntu\\home\\shay\\project',
        isWindows: true,
        wslDistributions: ['Ubuntu']
      })
    ).toEqual({ distribution: 'Ubuntu', mode: 'wsl2' })
  })

  it('keeps ordinary Windows projects native in smart mode', () => {
    expect(
      resolveWindowsTerminalMode({
        configuredMode: 'smart',
        cwd: 'C:\\Users\\shay\\project',
        isWindows: true,
        wslDistributions: ['Ubuntu']
      })
    ).toEqual({ distribution: null, mode: 'windows-native' })
  })

  it('forces the first installed distro in WSL2 mode', () => {
    expect(
      resolveWindowsTerminalMode({
        configuredMode: 'wsl2',
        cwd: 'C:\\Users\\shay\\project',
        isWindows: true,
        wslDistributions: ['Ubuntu', 'Debian']
      })
    ).toEqual({ distribution: 'Ubuntu', mode: 'wsl2' })
  })

  it('maps Windows and WSL UNC paths', () => {
    expect(toWslPath('C:\\Users\\shay\\project', 'Ubuntu')).toBe('/mnt/c/Users/shay/project')
    expect(toWslPath('\\\\wsl$\\Ubuntu\\home\\shay\\project', 'Ubuntu')).toBe('/home/shay/project')
    expect(distributionFromWslPath('\\\\wsl.localhost\\Ubuntu\\home\\shay')).toBe('Ubuntu')
  })
})
