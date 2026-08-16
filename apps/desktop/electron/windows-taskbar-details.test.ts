import { describe, expect, it, vi } from 'vitest'

import {
  buildWindowsRelaunchCommand,
  configureWindowsTaskbarDetails,
  resolveWindowsAppUserModelId,
  resolveWindowsDevRelaunchAppPath,
  WINDOWS_APP_USER_MODEL_ID,
  WINDOWS_DEV_APP_USER_MODEL_ID
} from './windows-taskbar-details'

describe('configureWindowsTaskbarDetails', () => {
  it('gives a Windows window the Hermes taskbar identity and relaunch details', () => {
    const setAppDetails = vi.fn()

    configureWindowsTaskbarDetails(
      { setAppDetails },
      {
        appId: WINDOWS_APP_USER_MODEL_ID,
        iconPath: 'C:\\Program Files\\Hermes\\resources\\icon.ico',
        isWindows: true,
        relaunchCommand: '"C:\\Program Files\\Hermes\\Hermes.exe"',
        relaunchDisplayName: 'Hermes'
      }
    )

    expect(setAppDetails).toHaveBeenCalledExactlyOnceWith({
      appId: WINDOWS_APP_USER_MODEL_ID,
      appIconIndex: 0,
      appIconPath: 'C:\\Program Files\\Hermes\\resources\\icon.ico',
      relaunchCommand: '"C:\\Program Files\\Hermes\\Hermes.exe"',
      relaunchDisplayName: 'Hermes'
    })
  })

  it('does nothing outside Windows', () => {
    const setAppDetails = vi.fn()

    configureWindowsTaskbarDetails(
      { setAppDetails },
      {
        appId: WINDOWS_APP_USER_MODEL_ID,
        isWindows: false,
        relaunchCommand: '"C:\\Hermes\\Hermes.exe"',
        relaunchDisplayName: 'Hermes'
      }
    )

    expect(setAppDetails).not.toHaveBeenCalled()
  })

  it('includes only a real app entry for Electron default-app launches', () => {
    const appEntryPath = 'C:\\Users\\gwmai\\git\\hermes-agent\\apps\\desktop'

    expect(resolveWindowsDevRelaunchAppPath(true, ['electron.exe', appEntryPath])).toBe(appEntryPath)
    expect(resolveWindowsDevRelaunchAppPath(true, ['electron.exe', '.'])).toBe(process.cwd())
    expect(resolveWindowsDevRelaunchAppPath(true, ['electron.exe', '--inspect=9229', '.'])).toBe(process.cwd())
    expect(resolveWindowsDevRelaunchAppPath(true, ['electron.exe', '--inspect'])).toBeUndefined()
    expect(resolveWindowsDevRelaunchAppPath(false, ['electron.exe', appEntryPath])).toBeUndefined()
  })

  it('keeps unpackaged Electron runs out of the production Hermes taskbar group', () => {
    expect(resolveWindowsAppUserModelId(false)).toBe(WINDOWS_APP_USER_MODEL_ID)
    expect(resolveWindowsAppUserModelId(true)).toBe(WINDOWS_DEV_APP_USER_MODEL_ID)
    expect(
      buildWindowsRelaunchCommand({
        executablePath: 'C:\\Program Files\\Hermes\\electron.exe',
        appEntryPath: 'C:\\Users\\gwmai\\git\\hermes-agent\\apps\\desktop',
        isDefaultApp: true
      })
    ).toBe('"C:\\Program Files\\Hermes\\electron.exe" "C:\\Users\\gwmai\\git\\hermes-agent\\apps\\desktop"')
  })
})
