import { describe, expect, it, vi } from 'vitest'

import { createTrayController, shouldMinimizeToTray, type TrayBuildOptions } from './tray-close'

describe('shouldMinimizeToTray', () => {
  it('hides the main window on Windows when enabled and not quitting', () => {
    const event = { preventDefault: vi.fn() }

    expect(
      shouldMinimizeToTray({
        event,
        isEnabled: true,
        isMainWindow: true,
        isQuitting: false,
        isQuittingForHandoff: false,
        isWindows: true
      })
    ).toBe(true)
    expect(event.preventDefault).toHaveBeenCalledTimes(1)
  })

  it('does NOT minimize on non-Windows platforms', () => {
    const event = { preventDefault: vi.fn() }

    expect(
      shouldMinimizeToTray({
        event,
        isEnabled: true,
        isMainWindow: true,
        isQuitting: false,
        isQuittingForHandoff: false,
        isWindows: false
      })
    ).toBe(false)
    expect(event.preventDefault).not.toHaveBeenCalled()
  })

  it('does NOT minimize when the preference is off', () => {
    const event = { preventDefault: vi.fn() }

    expect(
      shouldMinimizeToTray({
        event,
        isEnabled: false,
        isMainWindow: true,
        isQuitting: false,
        isQuittingForHandoff: false,
        isWindows: true
      })
    ).toBe(false)
    expect(event.preventDefault).not.toHaveBeenCalled()
  })

  it('does NOT minimize a secondary (non-main) window', () => {
    const event = { preventDefault: vi.fn() }

    expect(
      shouldMinimizeToTray({
        event,
        isEnabled: true,
        isMainWindow: false,
        isQuitting: false,
        isQuittingForHandoff: false,
        isWindows: true
      })
    ).toBe(false)
    expect(event.preventDefault).not.toHaveBeenCalled()
  })

  it('does NOT minimize during a real quit (so the window actually closes)', () => {
    const event = { preventDefault: vi.fn() }

    expect(
      shouldMinimizeToTray({
        event,
        isEnabled: true,
        isMainWindow: true,
        isQuitting: true,
        isQuittingForHandoff: false,
        isWindows: true
      })
    ).toBe(false)
    expect(event.preventDefault).not.toHaveBeenCalled()
  })

  it('does NOT minimize during a hand-off relaunch (update/swap/uninstall)', () => {
    const event = { preventDefault: vi.fn() }

    expect(
      shouldMinimizeToTray({
        event,
        isEnabled: true,
        isMainWindow: true,
        isQuitting: false,
        isQuittingForHandoff: true,
        isWindows: true
      })
    ).toBe(false)
    expect(event.preventDefault).not.toHaveBeenCalled()
  })

  it('returns the decision even when no event is supplied (no preventDefault)', () => {
    // The event is only needed to swallow the close; the decision itself is
    // independent of it. A null event still yields a true decision here, with
    // preventDefault safely skipped.
    expect(
      shouldMinimizeToTray({
        event: null,
        isEnabled: true,
        isMainWindow: true,
        isQuitting: false,
        isQuittingForHandoff: false,
        isWindows: true
      })
    ).toBe(true)
  })

  it('REGRESSION: a plain quit must close even when every active-work latch is false', () => {
    // Bug caught in review (PR #81342): the close handler used to derive its
    // "is quitting" signal from `quitPromptOpen || quitConfirmedWithActiveWork
    // || isQuittingForHandoff`. A plain File → Quit with no active work leaves
    // all of those false, so a tray-minimized window was hidden instead of
    // quit — and because before-quit had already destroyed the tray icon, the
    // app became a hidden window with no way to bring it back. The fix reads a
    // dedicated `isQuitting` latch set in before-quit.
    //
    // This asserts the decision is driven by `isQuitting` alone, NOT by the
    // absence of active work: with all latches false but isQuitting=true, the
    // window must close (no hide).
    const event = { preventDefault: vi.fn() }

    expect(
      shouldMinimizeToTray({
        event,
        isEnabled: true,
        isMainWindow: true,
        isQuitting: true, // the only thing that should distinguish a quit
        isQuittingForHandoff: false,
        isWindows: true
      })
    ).toBe(false)
    expect(event.preventDefault).not.toHaveBeenCalled()

    // And when isQuitting is false, a plain click-the-X (no active work) hides —
    // the two cases must not be conflated.
    const hideEvent = { preventDefault: vi.fn() }
    expect(
      shouldMinimizeToTray({
        event: hideEvent,
        isEnabled: true,
        isMainWindow: true,
        isQuitting: false,
        isQuittingForHandoff: false,
        isWindows: true
      })
    ).toBe(true)
    expect(hideEvent.preventDefault).toHaveBeenCalledTimes(1)
  })
})

describe('createTrayController', () => {
  function fakeTray() {
    const handlers: Record<string, (...args: unknown[]) => void> = {}
    const menu: unknown[] = []

    return {
      _handlers: handlers,
      _menu: menu,
      destroy: vi.fn(),
      on: vi.fn((event: string, cb: (...args: unknown[]) => void) => {
        handlers[event] = cb
      }),
      setContextMenu: vi.fn((m: unknown) => {
        menu.push(m)
      }),
      setToolTip: vi.fn()
    }
  }

  function setup(platformIsWindows: boolean) {
    const tray = fakeTray()
    const makeTray = vi.fn(() => tray as never)
    const makeIcon = vi.fn((iconPath: null | string) => ({ iconPath } as never))
    const makeMenu = vi.fn((template: unknown[]) => ({ template } as never))
    const controller = createTrayController(platformIsWindows, makeTray, makeIcon, makeMenu)

    return { controller, makeTray, makeIcon, makeMenu, tray }
  }

  it('builds a tray with restore + quit items on Windows only', () => {
    const { controller, makeTray, tray } = setup(true)
    const onRestore = vi.fn()
    const onQuit = vi.fn()

    controller.build({ iconPath: '/icon.png', onRestore, onQuit })

    expect(makeTray).toHaveBeenCalledTimes(1)
    expect(controller.isActive()).toBe(true)
    expect(tray.setToolTip).toHaveBeenCalledWith('Hermes')

    // Context menu wires Open + Exit.
    const menu = tray._menu as Array<{ template: Array<{ click?: () => void; label: string; type?: string }> }>
    const items = menu[0]?.template ?? []
    const open = items.find(item => item.label === 'Open Hermes')
    const exit = items.find(item => item.label === 'Exit')

    expect(open).toBeDefined()
    expect(exit).toBeDefined()
    open?.click?.()
    exit?.click?.()
    expect(onRestore).toHaveBeenCalledTimes(1)
    expect(onQuit).toHaveBeenCalledTimes(1)

    // Left-click restores.
    tray._handlers['click']?.()
    expect(onRestore).toHaveBeenCalledTimes(2)
  })

  it('does not build a tray off Windows', () => {
    const { controller, makeTray } = setup(false)

    controller.build({ iconPath: '/icon.png', onRestore: vi.fn(), onQuit: vi.fn() })

    expect(makeTray).not.toHaveBeenCalled()
    expect(controller.isActive()).toBe(false)
  })

  it('destroy() removes the tray and isActive reflects it', () => {
    const { controller, tray } = setup(true)

    controller.build({ iconPath: '/icon.png', onRestore: vi.fn(), onQuit: vi.fn() })
    expect(controller.isActive()).toBe(true)
    controller.destroy()
    expect(controller.isActive()).toBe(false)
    expect(tray.destroy).toHaveBeenCalledTimes(1)
  })

  it('rebuild() recreates the tray (destroy before build)', () => {
    const { controller, makeTray } = setup(true)

    controller.build({ iconPath: '/a.png', onRestore: vi.fn(), onQuit: vi.fn() })
    controller.build({ iconPath: '/b.png', onRestore: vi.fn(), onQuit: vi.fn() })

    expect(makeTray).toHaveBeenCalledTimes(2)
    expect(controller.isActive()).toBe(true)
  })

  it('accepts a null icon path without throwing', () => {
    const { controller, makeTray } = setup(true)

    expect(() =>
      controller.build({ iconPath: null, onRestore: vi.fn(), onQuit: vi.fn() })
    ).not.toThrow()
    expect(makeTray).toHaveBeenCalledTimes(1)
  })

  // Ensures the TrayBuildOptions shape stays in sync with build() callers.
  const _typecheck: TrayBuildOptions = { iconPath: null, onRestore: () => {}, onQuit: () => {} }
  void _typecheck
})
