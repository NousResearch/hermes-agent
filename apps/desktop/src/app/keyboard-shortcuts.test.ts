import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { type CloseAction, isCleanShortcut, resolveCloseAction, setupCloseCascadeListener, setupShortcutListener } from './keyboard-shortcuts'

describe('resolveCloseAction', () => {
  it('closes overlay when any overlay is open', () => {
    expect(
      resolveCloseAction({
        overlayOpen: true,
        rightSidebarOpen: true,
        previewRailOpen: true,
        leftSidebarOpen: true
      })
    ).toEqual({ type: 'close-overlay' })
  })

  it('closes right sidebar when overlay is closed but sidebar is open', () => {
    expect(
      resolveCloseAction({
        overlayOpen: false,
        rightSidebarOpen: true,
        previewRailOpen: true,
        leftSidebarOpen: true
      })
    ).toEqual({ type: 'close-right-sidebar' })
  })

  it('closes preview rail when overlay and right sidebar are closed', () => {
    expect(
      resolveCloseAction({
        overlayOpen: false,
        rightSidebarOpen: false,
        previewRailOpen: true,
        leftSidebarOpen: true
      })
    ).toEqual({ type: 'close-preview-rail' })
  })

  it('closes left sidebar when only left sidebar is open', () => {
    expect(
      resolveCloseAction({
        overlayOpen: false,
        rightSidebarOpen: false,
        previewRailOpen: false,
        leftSidebarOpen: true
      })
    ).toEqual({ type: 'close-left-sidebar' })
  })

  it('closes window when nothing is open', () => {
    expect(
      resolveCloseAction({
        overlayOpen: false,
        rightSidebarOpen: false,
        previewRailOpen: false,
        leftSidebarOpen: false
      })
    ).toEqual({ type: 'close-window' })
  })

  it('skips closed layers and finds the first open one', () => {
    expect(
      resolveCloseAction({
        overlayOpen: false,
        rightSidebarOpen: false,
        previewRailOpen: true,
        leftSidebarOpen: false
      })
    ).toEqual({ type: 'close-preview-rail' })
  })
})

describe('isCleanShortcut', () => {
  function keyEvent(init: Partial<KeyboardEvent> & { key: string }): KeyboardEvent {
    return new KeyboardEvent('keydown', {
      metaKey: false,
      ctrlKey: false,
      altKey: false,
      shiftKey: false,
      ...init
    })
  }

  it('returns true for Cmd+W on macOS', () => {
    expect(isCleanShortcut(keyEvent({ key: 'w', metaKey: true }), 'w')).toBe(true)
  })

  it('returns true for Ctrl+W on Windows/Linux', () => {
    expect(isCleanShortcut(keyEvent({ key: 'w', ctrlKey: true }), 'w')).toBe(true)
  })

  it('returns false when Alt is pressed', () => {
    expect(isCleanShortcut(keyEvent({ key: 'w', metaKey: true, altKey: true }), 'w')).toBe(false)
  })

  it('returns false when Shift is pressed', () => {
    expect(isCleanShortcut(keyEvent({ key: 'w', metaKey: true, shiftKey: true }), 'w')).toBe(false)
  })

  it('returns false for wrong key', () => {
    expect(isCleanShortcut(keyEvent({ key: 'w', metaKey: true }), 'k')).toBe(false)
  })

  it('returns false when no modifier is pressed', () => {
    expect(isCleanShortcut(keyEvent({ key: 'w' }), 'w')).toBe(false)
  })
})

describe('setupCloseCascadeListener', () => {
  let actions: CloseAction[]
  let getState: () => { overlayOpen: boolean; rightSidebarOpen: boolean; previewRailOpen: boolean; leftSidebarOpen: boolean }
  let cleanup: () => void

  beforeEach(() => {
    actions = []
    getState = () => ({
      overlayOpen: false,
      rightSidebarOpen: false,
      previewRailOpen: false,
      leftSidebarOpen: false
    })
    cleanup = setupCloseCascadeListener(
      () => getState(),
      action => actions.push(action)
    )
  })

  afterEach(() => {
    cleanup()
  })

  function pressCmdW() {
    window.dispatchEvent(
      new KeyboardEvent('keydown', {
        key: 'w',
        metaKey: true,
        bubbles: true
      })
    )
  }

  function pressCtrlW() {
    window.dispatchEvent(
      new KeyboardEvent('keydown', {
        key: 'w',
        ctrlKey: true,
        bubbles: true
      })
    )
  }

  it('dispatches close-overlay when overlay is open', () => {
    getState = () => ({
      overlayOpen: true,
      rightSidebarOpen: true,
      previewRailOpen: true,
      leftSidebarOpen: true
    })
    pressCmdW()
    expect(actions).toEqual([{ type: 'close-overlay' }])
  })

  it('dispatches close-right-sidebar when only sidebar is open', () => {
    getState = () => ({
      overlayOpen: false,
      rightSidebarOpen: true,
      previewRailOpen: false,
      leftSidebarOpen: false
    })
    pressCmdW()
    expect(actions).toEqual([{ type: 'close-right-sidebar' }])
  })

  it('dispatches close-window when nothing is open', () => {
    pressCmdW()
    expect(actions).toEqual([{ type: 'close-window' }])
  })

  it('works with Ctrl+W (Windows/Linux)', () => {
    pressCtrlW()
    expect(actions).toEqual([{ type: 'close-window' }])
  })

  it('does not dispatch on other key combos', () => {
    window.dispatchEvent(
      new KeyboardEvent('keydown', {
        key: 'k',
        metaKey: true,
        bubbles: true
      })
    )
    expect(actions).toEqual([])
  })

  it('prevents default on Cmd+W', () => {
    const event = new KeyboardEvent('keydown', {
      key: 'w',
      metaKey: true,
      bubbles: true,
      cancelable: true
    })
    const spy = vi.spyOn(event, 'preventDefault')
    window.dispatchEvent(event)
    expect(spy).toHaveBeenCalled()
  })
})

describe('setupShortcutListener', () => {
  let cleanup: () => void

  afterEach(() => {
    cleanup?.()
  })

  function pressKey(key: string, modifiers: Partial<Pick<KeyboardEvent, 'metaKey' | 'ctrlKey' | 'altKey' | 'shiftKey'>> = {}) {
    window.dispatchEvent(
      new KeyboardEvent('keydown', {
        key,
        metaKey: false,
        ctrlKey: false,
        altKey: false,
        shiftKey: false,
        ...modifiers,
        bubbles: true,
        cancelable: true
      })
    )
  }

  it('calls callback when Cmd+, is pressed', () => {
    const fn = vi.fn()
    cleanup = setupShortcutListener(',', fn)

    pressKey(',', { metaKey: true })
    expect(fn).toHaveBeenCalledTimes(1)
  })

  it('calls callback when Ctrl+, is pressed (Windows/Linux)', () => {
    const fn = vi.fn()
    cleanup = setupShortcutListener(',', fn)

    pressKey(',', { ctrlKey: true })
    expect(fn).toHaveBeenCalledTimes(1)
  })

  it('does not call callback for wrong key', () => {
    const fn = vi.fn()
    cleanup = setupShortcutListener(',', fn)

    pressKey('.', { metaKey: true })
    expect(fn).not.toHaveBeenCalled()
  })

  it('does not call callback when Alt is held', () => {
    const fn = vi.fn()
    cleanup = setupShortcutListener(',', fn)

    pressKey(',', { metaKey: true, altKey: true })
    expect(fn).not.toHaveBeenCalled()
  })

  it('does not call callback when Shift is held', () => {
    const fn = vi.fn()
    cleanup = setupShortcutListener(',', fn)

    pressKey(',', { metaKey: true, shiftKey: true })
    expect(fn).not.toHaveBeenCalled()
  })

  it('prevents default when shortcut matches', () => {
    const fn = vi.fn()
    cleanup = setupShortcutListener(',', fn)

    const event = new KeyboardEvent('keydown', {
      key: ',',
      metaKey: true,
      bubbles: true,
      cancelable: true
    })
    const spy = vi.spyOn(event, 'preventDefault')
    window.dispatchEvent(event)
    expect(spy).toHaveBeenCalled()
  })

  it('cleans up listener on unsubscribe', () => {
    const fn = vi.fn()
    cleanup = setupShortcutListener(',', fn)
    cleanup()

    pressKey(',', { metaKey: true })
    expect(fn).not.toHaveBeenCalled()
  })
})

describe('Cmd+K (command center)', () => {
  it('triggers callback for Cmd+K', () => {
    const fn = vi.fn()
    const cleanup = setupShortcutListener('k', fn)

    window.dispatchEvent(
      new KeyboardEvent('keydown', {
        key: 'k',
        metaKey: true,
        bubbles: true
      })
    )
    expect(fn).toHaveBeenCalledTimes(1)
    cleanup()
  })
})

describe('Cmd+. (right panel)', () => {
  it('triggers callback for Cmd+.', () => {
    const fn = vi.fn()
    const cleanup = setupShortcutListener('.', fn)

    window.dispatchEvent(
      new KeyboardEvent('keydown', {
        key: '.',
        metaKey: true,
        bubbles: true
      })
    )
    expect(fn).toHaveBeenCalledTimes(1)
    cleanup()
  })
})

describe('cascade ordering on repeated presses', () => {
  it('closes layers one by one on successive Cmd+W presses', () => {
    // Start with everything open
    let state = { overlayOpen: true, rightSidebarOpen: true, previewRailOpen: true, leftSidebarOpen: true }

    expect(resolveCloseAction(state)).toEqual({ type: 'close-overlay' })

    state = { ...state, overlayOpen: false }
    expect(resolveCloseAction(state)).toEqual({ type: 'close-right-sidebar' })

    state = { ...state, rightSidebarOpen: false }
    expect(resolveCloseAction(state)).toEqual({ type: 'close-preview-rail' })

    state = { ...state, previewRailOpen: false }
    expect(resolveCloseAction(state)).toEqual({ type: 'close-left-sidebar' })

    state = { ...state, leftSidebarOpen: false }
    expect(resolveCloseAction(state)).toEqual({ type: 'close-window' })
  })
})

describe('shortcuts do not interfere with each other', () => {
  it('Cmd+K does not trigger Cmd+W cascade', () => {
    const closeActions: string[] = []
    const searchFn = vi.fn()

    const cleanupClose = setupCloseCascadeListener(
      () => ({ overlayOpen: false, rightSidebarOpen: false, previewRailOpen: false, leftSidebarOpen: false }),
      action => closeActions.push(action.type)
    )
    const cleanupSearch = setupShortcutListener('k', searchFn)

    window.dispatchEvent(
      new KeyboardEvent('keydown', { key: 'k', metaKey: true, bubbles: true })
    )

    expect(searchFn).toHaveBeenCalledTimes(1)
    expect(closeActions).toEqual([])

    cleanupClose()
    cleanupSearch()
  })
})
