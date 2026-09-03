import { describe, expect, it } from 'vitest'

import {
  shouldHideMainWindowToTray,
  shouldQuitAfterAllWindowsClose,
  shouldShowMainWindowOnStartup
} from './window-close-policy'

describe('Windows close-to-tray policy', () => {
  it('does not hide when the tray could not be created', () => {
    expect(
      shouldHideMainWindowToTray({
        isWindows: true,
        trayAvailable: false,
        closeToTray: true,
        isQuitting: false,
        isQuittingForHandoff: false
      })
    ).toBe(false)
  })

  it('hides when Windows has a tray and close-to-tray is enabled', () => {
    expect(
      shouldHideMainWindowToTray({
        isWindows: true,
        trayAvailable: true,
        closeToTray: true,
        isQuitting: false,
        isQuittingForHandoff: false
      })
    ).toBe(true)
  })

  it('does not hide when close-to-tray is disabled', () => {
    expect(
      shouldHideMainWindowToTray({
        isWindows: true,
        trayAvailable: true,
        closeToTray: false,
        isQuitting: false,
        isQuittingForHandoff: false
      })
    ).toBe(false)
  })

  it('allows a real quit from the tray menu', () => {
    expect(
      shouldHideMainWindowToTray({
        isWindows: true,
        trayAvailable: true,
        closeToTray: true,
        isQuitting: true,
        isQuittingForHandoff: false
      })
    ).toBe(false)
  })

  it('allows updater/uninstall handoff to close the window for real', () => {
    expect(
      shouldHideMainWindowToTray({
        isWindows: true,
        trayAvailable: true,
        closeToTray: true,
        isQuitting: false,
        isQuittingForHandoff: true
      })
    ).toBe(false)
  })

  it('never hides on non-Windows platforms', () => {
    expect(
      shouldHideMainWindowToTray({
        isWindows: false,
        trayAvailable: true,
        closeToTray: true,
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
        closeToTray: true,
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
        closeToTray: true,
        isQuitting: false,
        isQuittingForHandoff: false
      })
    ).toBe(false)
  })

  it('quits Windows when close-to-tray is disabled even if the tray exists', () => {
    expect(
      shouldQuitAfterAllWindowsClose({
        isWindows: true,
        isMac: false,
        trayAvailable: true,
        closeToTray: false,
        isQuitting: false,
        isQuittingForHandoff: false
      })
    ).toBe(true)
  })

  it('quits Windows after an explicit tray quit or updater handoff', () => {
    expect(
      shouldQuitAfterAllWindowsClose({
        isWindows: true,
        isMac: false,
        trayAvailable: true,
        closeToTray: true,
        isQuitting: true,
        isQuittingForHandoff: false
      })
    ).toBe(true)
    expect(
      shouldQuitAfterAllWindowsClose({
        isWindows: true,
        isMac: false,
        trayAvailable: true,
        closeToTray: true,
        isQuitting: false,
        isQuittingForHandoff: true
      })
    ).toBe(true)
  })

  it('starts hidden only when Windows has a usable tray and the preference is enabled', () => {
    expect(
      shouldShowMainWindowOnStartup({ isWindows: true, trayAvailable: true, startInTray: true })
    ).toBe(false)
    expect(
      shouldShowMainWindowOnStartup({ isWindows: true, trayAvailable: false, startInTray: true })
    ).toBe(true)
    expect(
      shouldShowMainWindowOnStartup({ isWindows: true, trayAvailable: true, startInTray: false })
    ).toBe(true)
    expect(
      shouldShowMainWindowOnStartup({ isWindows: false, trayAvailable: true, startInTray: true })
    ).toBe(true)
  })

  it('always shows non-initial windows even when start-in-tray is enabled', () => {
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
