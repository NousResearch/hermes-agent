/**
 * Unit tests for the main-process keyboard shortcut guards.
 * The bug this fixes: Ctrl+W pressed in another app (e.g. a browser) closes
 * that window; if Hermes inherits focus mid-chord the leftover keyup lands
 * on our webContents and closes a session tab without the user pressing
 * anything *in* Hermes (#105498).
 */

import assert from 'node:assert/strict'
import { EventEmitter } from 'node:events'

import type { BrowserWindow, Input } from 'electron'
import { describe, test, vi } from 'vitest'

import { installInputShortcuts } from './input-shortcuts'

interface FakeBrowserWindow {
  id: number
  webContents: FakeWebContents
  on: typeof EventEmitter.prototype.on
  off: typeof EventEmitter.prototype.off
  emit: (event: string | symbol, ...args: unknown[]) => boolean
}

interface FakeWebContents {
  calls: {
    preventDefault: number
    sendClose: number
    sendReload: number
    setZoom: Array<number>
  }
  isDestroyed: () => boolean
  getZoomLevel: () => number
  on: typeof EventEmitter.prototype.on
  off: typeof EventEmitter.prototype.off
  emit: (event: string | symbol, ...args: unknown[]) => boolean
}

function makeFakeWindow(id: number = 1): FakeBrowserWindow {
  const windowEmitter = new EventEmitter()
  const contentsEmitter = new EventEmitter()

  const calls = {
    preventDefault: 0,
    sendClose: 0,
    sendReload: 0,
    setZoom: [] as Array<number>
  }

  let zoomLevel = 0

  const webContents: FakeWebContents = {
    calls,
    isDestroyed: () => false,
    getZoomLevel: () => zoomLevel,
    on: contentsEmitter.on.bind(contentsEmitter),
    off: contentsEmitter.off.bind(contentsEmitter),
    emit: contentsEmitter.emit.bind(contentsEmitter)
  }

  return {
    id,
    webContents,
    on: windowEmitter.on.bind(windowEmitter),
    off: windowEmitter.off.bind(windowEmitter),
    emit: windowEmitter.emit.bind(windowEmitter)
  }
}

function asWin(fake: FakeBrowserWindow): BrowserWindow {
  return fake as unknown as BrowserWindow
}

function makeInput(overrides: Partial<Input> = {}): Input {
  return {
    type: 'keyDown',
    key: '',
    code: '',
    isAutoRepeat: false,
    isComposing: false,
    shift: false,
    control: false,
    alt: false,
    meta: false,
    location: 0,
    modifiers: [],
    ...overrides
  }
}

describe('installInputShortcuts', () => {
  describe('Close Tab (Ctrl/Cmd+W)', () => {
    test('fires Close Tab on keyDown of Ctrl+W (Windows)', () => {
      const win = makeFakeWindow()
      const sendClose = vi.fn()
      installInputShortcuts(asWin(win), false, { sendClosePreviewRequested: sendClose })

      const input = makeInput({ key: 'w', control: true })
      const event = { preventDefault: () => win.webContents.calls.preventDefault++ }
      win.webContents.emit('before-input-event', event, input)

      assert.equal(win.webContents.calls.preventDefault, 1)
      assert.equal(sendClose.mock.calls.length, 1)
    })

    test('fires Close Tab on keyDown of Cmd+W (macOS)', () => {
      const win = makeFakeWindow()
      const sendClose = vi.fn()
      installInputShortcuts(asWin(win), true, { sendClosePreviewRequested: sendClose })

      const input = makeInput({ key: 'w', meta: true })
      const event = { preventDefault: () => win.webContents.calls.preventDefault++ }
      win.webContents.emit('before-input-event', event, input)

      assert.equal(win.webContents.calls.preventDefault, 1)
      assert.equal(sendClose.mock.calls.length, 1)
    })

    test('does not close a tab on keyUp of Ctrl+W', () => {
      const win = makeFakeWindow()
      const sendClose = vi.fn()
      installInputShortcuts(asWin(win), false, { sendClosePreviewRequested: sendClose })

      const input = makeInput({ type: 'keyUp', key: 'w', control: true })
      const event = { preventDefault: () => win.webContents.calls.preventDefault++ }
      win.webContents.emit('before-input-event', event, input)

      assert.equal(win.webContents.calls.preventDefault, 0)
      assert.equal(sendClose.mock.calls.length, 0)
    })

    test('ignores auto-repeat of Ctrl+W', () => {
      const win = makeFakeWindow()
      const sendClose = vi.fn()
      installInputShortcuts(asWin(win), false, { sendClosePreviewRequested: sendClose })

      const input = makeInput({ key: 'w', control: true, isAutoRepeat: true })
      const event = { preventDefault: () => win.webContents.calls.preventDefault++ }
      win.webContents.emit('before-input-event', event, input)

      assert.equal(win.webContents.calls.preventDefault, 0)
      assert.equal(sendClose.mock.calls.length, 0)
    })

    test('ignores Ctrl+W within the focus-grace window (200ms)', () => {
      const win = makeFakeWindow()
      const sendClose = vi.fn()
      installInputShortcuts(asWin(win), false, { sendClosePreviewRequested: sendClose })

      // Simulate focus transfer right now
      win.emit('focus')

      const input = makeInput({ key: 'w', control: true })
      const event = { preventDefault: () => win.webContents.calls.preventDefault++ }
      win.webContents.emit('before-input-event', event, input)

      assert.equal(win.webContents.calls.preventDefault, 0)
      assert.equal(sendClose.mock.calls.length, 0)
    })

    test('fires Close Tab when focus-grace has expired (>200ms)', () => {
      const win = makeFakeWindow()
      const sendClose = vi.fn()
      installInputShortcuts(asWin(win), false, { sendClosePreviewRequested: sendClose })

      vi.useFakeTimers()

      try {
        win.emit('focus')
        vi.advanceTimersByTime(210)

        const input = makeInput({ key: 'w', control: true })
        const event = { preventDefault: () => win.webContents.calls.preventDefault++ }
        win.webContents.emit('before-input-event', event, input)

        assert.equal(win.webContents.calls.preventDefault, 1)
        assert.equal(sendClose.mock.calls.length, 1)
      } finally {
        vi.useRealTimers()
      }
    })

    test('does not fire when Shift is held (Ctrl+Shift+W is not Close Tab)', () => {
      const win = makeFakeWindow()
      const sendClose = vi.fn()
      installInputShortcuts(asWin(win), false, { sendClosePreviewRequested: sendClose })

      const input = makeInput({ key: 'w', control: true, shift: true })
      const event = { preventDefault: () => win.webContents.calls.preventDefault++ }
      win.webContents.emit('before-input-event', event, input)

      assert.equal(win.webContents.calls.preventDefault, 0)
      assert.equal(sendClose.mock.calls.length, 0)
    })
  })

  describe('Reload (Ctrl/Cmd+R)', () => {
    test('fires Reload on keyDown of Ctrl+R', () => {
      const win = makeFakeWindow()
      const sendNav = vi.fn()
      installInputShortcuts(asWin(win), false, { sendPreviewNavCommand: sendNav })

      const input = makeInput({ key: 'r', control: true })
      const event = { preventDefault: () => win.webContents.calls.preventDefault++ }
      win.webContents.emit('before-input-event', event, input)

      assert.equal(win.webContents.calls.preventDefault, 1)
      assert.equal(sendNav.mock.calls.length, 1)
      assert.deepEqual(sendNav.mock.calls[0], ['reload'])
    })

    test('does not reload on keyUp of Ctrl+R', () => {
      const win = makeFakeWindow()
      const sendNav = vi.fn()
      installInputShortcuts(asWin(win), false, { sendPreviewNavCommand: sendNav })

      const input = makeInput({ type: 'keyUp', key: 'r', control: true })
      const event = { preventDefault: () => win.webContents.calls.preventDefault++ }
      win.webContents.emit('before-input-event', event, input)

      assert.equal(win.webContents.calls.preventDefault, 0)
      assert.equal(sendNav.mock.calls.length, 0)
    })

    test('ignores Ctrl+R within the focus-grace window', () => {
      const win = makeFakeWindow()
      const sendNav = vi.fn()
      installInputShortcuts(asWin(win), false, { sendPreviewNavCommand: sendNav })

      win.emit('focus')

      const input = makeInput({ key: 'r', control: true })
      const event = { preventDefault: () => win.webContents.calls.preventDefault++ }
      win.webContents.emit('before-input-event', event, input)

      assert.equal(win.webContents.calls.preventDefault, 0)
      assert.equal(sendNav.mock.calls.length, 0)
    })
  })

  describe('Zoom shortcuts (Ctrl/Cmd + +/-/0)', () => {
    test('zooms in on Ctrl+=', () => {
      const win = makeFakeWindow()
      const setZoom = vi.fn()
      installInputShortcuts(asWin(win), false, {
        setAndPersistZoomLevel: setZoom,
        getZoomStep: () => 0.1
      })

      const input = makeInput({ key: '=', control: true })
      const event = { preventDefault: () => win.webContents.calls.preventDefault++ }
      win.webContents.emit('before-input-event', event, input)

      assert.equal(win.webContents.calls.preventDefault, 1)
      assert.equal(setZoom.mock.calls.length, 1)
      assert.equal(setZoom.mock.calls[0][1], 0.1)
    })

    test('zooms out on Ctrl+-', () => {
      const win = makeFakeWindow()
      const setZoom = vi.fn()
      installInputShortcuts(asWin(win), false, {
        setAndPersistZoomLevel: setZoom,
        getZoomStep: () => 0.1
      })

      const input = makeInput({ key: '-', control: true })
      const event = { preventDefault: () => win.webContents.calls.preventDefault++ }
      win.webContents.emit('before-input-event', event, input)

      assert.equal(win.webContents.calls.preventDefault, 1)
      assert.equal(setZoom.mock.calls.length, 1)
      assert.equal(setZoom.mock.calls[0][1], -0.1)
    })

    test('resets zoom on Ctrl+0', () => {
      const win = makeFakeWindow()
      const setZoom = vi.fn()
      installInputShortcuts(asWin(win), false, {
        setAndPersistZoomLevel: setZoom,
        getDefaultZoomLevel: () => 0
      })

      const input = makeInput({ key: '0', control: true })
      const event = { preventDefault: () => win.webContents.calls.preventDefault++ }
      win.webContents.emit('before-input-event', event, input)

      assert.equal(win.webContents.calls.preventDefault, 1)
      assert.equal(setZoom.mock.calls.length, 1)
      assert.equal(setZoom.mock.calls[0][1], 0)
    })

    test('does not zoom on keyUp of Ctrl+0', () => {
      const win = makeFakeWindow()
      const setZoom = vi.fn()
      installInputShortcuts(asWin(win), false, {
        setAndPersistZoomLevel: setZoom,
        getDefaultZoomLevel: () => 0
      })

      const input = makeInput({ type: 'keyUp', key: '0', control: true })
      const event = { preventDefault: () => win.webContents.calls.preventDefault++ }
      win.webContents.emit('before-input-event', event, input)

      assert.equal(win.webContents.calls.preventDefault, 0)
      assert.equal(setZoom.mock.calls.length, 0)
    })

    test('handles zoom-changed (wheel)', () => {
      const win = makeFakeWindow()
      const setZoom = vi.fn()
      installInputShortcuts(asWin(win), false, {
        setAndPersistZoomLevel: setZoom,
        getZoomStep: () => 0.1
      })

      const event = { preventDefault: () => {} }
      win.webContents.emit('zoom-changed', event, 'in')

      assert.equal(setZoom.mock.calls.length, 1)
      assert.equal(setZoom.mock.calls[0][1], 0.1)
    })
  })

  describe('cleanup', () => {
    test('detaches all listeners on uninstall', () => {
      const win = makeFakeWindow()
      const sendClose = vi.fn()
      const uninstall = installInputShortcuts(asWin(win), false, { sendClosePreviewRequested: sendClose })

      const input = makeInput({ key: 'w', control: true })
      const event = { preventDefault: () => win.webContents.calls.preventDefault++ }

      win.webContents.emit('before-input-event', event, input)
      assert.equal(win.webContents.calls.preventDefault, 1)

      uninstall()

      win.webContents.emit('before-input-event', event, input)
      assert.equal(win.webContents.calls.preventDefault, 1, 'listener not detached')
    })
  })
})
