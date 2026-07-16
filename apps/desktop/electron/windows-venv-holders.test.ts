import { describe, expect, it } from 'vitest'

import {
  isVenvHolderProcess,
  normalizeWinPath,
  selectVenvHolders
} from './windows-venv-holders'

const ROOT = 'C:\\Users\\si\\AppData\\Local\\hermes\\hermes-agent'

describe('normalizeWinPath', () => {
  it('lowercases and unifies separators', () => {
    expect(normalizeWinPath('C:/Users/SI/AppData/Local/hermes/hermes-agent/')).toBe(
      'c:\\users\\si\\appdata\\local\\hermes\\hermes-agent'
    )
  })
})

describe('isVenvHolderProcess', () => {
  it('matches venv python.exe by path', () => {
    expect(
      isVenvHolderProcess(
        {
          pid: 42,
          exe: ROOT + '\\venv\\Scripts\\python.exe',
          cmdline: ROOT + '\\venv\\Scripts\\python.exe -m hermes_cli.main gateway run'
        },
        ROOT
      )
    ).toBe(true)
  })

  it('matches base python trampoline that imports the venv via cmdline', () => {
    expect(
      isVenvHolderProcess(
        {
          pid: 7,
          exe: 'C:\\Users\\si\\AppData\\Local\\Programs\\Python\\Python311\\python.exe',
          cmdline:
            'python.exe -m hermes_cli.main --profile sophie gateway run ' +
            ROOT +
            '\\venv\\Lib\\site-packages'
        },
        ROOT
      )
    ).toBe(true)
  })

  it('matches tui_gateway slash_worker under install cwd', () => {
    expect(
      isVenvHolderProcess(
        {
          pid: 9,
          exe: ROOT + '\\venv\\Scripts\\python.exe',
          cmdline: 'python.exe -m tui_gateway.slash_worker --session-key x',
          cwd: ROOT
        },
        ROOT
      )
    ).toBe(true)
  })

  it('skips listed pids and unrelated processes', () => {
    expect(
      isVenvHolderProcess(
        {
          pid: 1,
          exe: ROOT + '\\venv\\Scripts\\python.exe',
          cmdline: 'x'
        },
        ROOT,
        new Set([1])
      )
    ).toBe(false)

    expect(
      isVenvHolderProcess(
        {
          pid: 2,
          exe: 'C:\\Windows\\System32\\notepad.exe',
          cmdline: 'notepad'
        },
        ROOT
      )
    ).toBe(false)
  })
})

describe('selectVenvHolders', () => {
  it('dedupes and filters', () => {
    const holders = selectVenvHolders(
      [
        {
          pid: 10,
          name: 'python.exe',
          exe: ROOT + '\\venv\\Scripts\\python.exe',
          cmdline: 'gateway'
        },
        {
          pid: 10,
          name: 'python.exe',
          exe: ROOT + '\\venv\\Scripts\\python.exe',
          cmdline: 'gateway'
        },
        { pid: 11, name: 'notepad.exe', exe: 'C:\\Windows\\notepad.exe', cmdline: 'n' }
      ],
      ROOT
    )

    expect(holders).toEqual([
      {
        pid: 10,
        name: 'python.exe',
        exe: ROOT + '\\venv\\Scripts\\python.exe',
        cmdline: 'gateway'
      }
    ])
  })
})
