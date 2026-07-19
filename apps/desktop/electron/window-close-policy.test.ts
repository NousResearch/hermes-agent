import { describe, expect, it } from 'vitest'

import { shouldHideMainWindowToTray, shouldQuitAfterAllWindowsClose, shouldShowMainWindowOnStartup } from './window-close-policy'

describe('Windows close-to-tray policy', () => {
  it('does not hide when the tray could not be created', () => {
    expect(
      shouldHideMainWindowToTray({
        isWindows: true,
        trayAvailable: false,
        isQuitting: false,
        isQuittingForHandoff: false
      })
    ).toBe(false)
  })

  it('hides the main window for a normal Windows close', () => {
    expect(
      shouldHideMainWindowToTray({
        isWindows: true,
        trayAvailable: true,
        isQuitting: false,
        isQuittingForHandoff: false
      })
    ).toBe(true)
  })

  it('allows a real quit from the tray menu', () => {
    expect(
      shouldHideMainWindowToTray({
        isWindows: true,
        trayAvailable: true,
        isQuitting: true,
        isQuittingForHandoff: false
      })
    ).toBe(false)
  })

  it('allows updater handoff to close the window', () => {
    expect(
      shouldHideMainWindowToTray({
        isWindows: true,
        trayAvailable: true,
        isQuitting: false,
        isQuittingForHandoff: true
      })
    ).toBe(false)
  })

  it('does not change non-Windows close behavior', () => {
    expect(
      shouldHideMainWindowToTray({
        isWindows: false,
        trayAvailable: false,
        isQuitting: false,
        isQuittingForHandoff: false
      })
    ).toBe(false)
  })

  it('quits Windows when no tray is available to restore the app', () => {
    expect(
      shouldQuitAfterAllWindowsClose({
        isWindows: true,
        isMac: false,
        trayAvailable: false,
        isQuitting: false,
        isQuittingForHandoff: false
      })
    ).toBe(true)
  })

  it('keeps Windows alive while the tray owns the app lifecycle', () => {
    expect(
      shouldQuitAfterAllWindowsClose({
        isWindows: true,
        isMac: false,
        trayAvailable: true,
        isQuitting: false,
        isQuittingForHandoff: false
      })
    ).toBe(false)
  })

  it('quits Windows after an explicit tray quit or updater handoff', () => {
    expect(
      shouldQuitAfterAllWindowsClose({
        isWindows: true,
        isMac: false,
        trayAvailable: true,
        isQuitting: true,
        isQuittingForHandoff: false
      })
    ).toBe(true)
    expect(
      shouldQuitAfterAllWindowsClose({
        isWindows: true,
        isMac: false,
        trayAvailable: true,
        isQuitting: false,
        isQuittingForHandoff: true
      })
    ).toBe(true)
  })

  it('starts hidden only when Windows has a usable tray and the preference is enabled', () => {
    expect(shouldShowMainWindowOnStartup({ isWindows: true, trayAvailable: true, startInTray: true })).toBe(false)
    expect(shouldShowMainWindowOnStartup({ isWindows: true, trayAvailable: false, startInTray: true })).toBe(true)
    expect(shouldShowMainWindowOnStartup({ isWindows: true, trayAvailable: true, startInTray: false })).toBe(true)
    expect(shouldShowMainWindowOnStartup({ isWindows: false, trayAvailable: true, startInTray: true })).toBe(true)
    expect(
      shouldShowMainWindowOnStartup({
        isWindows: true,
        trayAvailable: true,
        startInTray: true,
        isInitialWindow: false
      })
    ).toBe(true)
  })
})
