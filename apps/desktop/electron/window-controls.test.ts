import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import { customWindowControlsEnabled, performWindowControl, windowControlState } from './window-controls'

class FakeWindow {
  closed = false
  focusCalls = 0
  maximized = false
  minimized = false

  close() {
    this.closed = true
  }

  focus() {
    this.focusCalls += 1
  }

  isDestroyed() {
    return false
  }

  isMaximized() {
    return this.maximized
  }

  maximize() {
    this.maximized = true
  }

  minimize() {
    this.minimized = true
  }

  unmaximize() {
    this.maximized = false
  }
}

describe('performWindowControl', () => {
  test('minimizes the sender window', () => {
    const win = new FakeWindow()

    assert.equal(performWindowControl(win, 'minimize'), true)
    assert.equal(win.minimized, true)
  })

  test('toggles maximize and restore', () => {
    const win = new FakeWindow()

    assert.equal(performWindowControl(win, 'toggle-maximize'), true)
    assert.equal(win.maximized, true)

    assert.equal(performWindowControl(win, 'toggle-maximize'), true)
    assert.equal(win.maximized, false)
  })

  test('restores keyboard focus after maximize and restore', () => {
    const win = new FakeWindow()

    performWindowControl(win, 'toggle-maximize')
    assert.equal(win.focusCalls, 1)

    performWindowControl(win, 'toggle-maximize')
    assert.equal(win.focusCalls, 2)
  })

  test('closes the sender window', () => {
    const win = new FakeWindow()

    assert.equal(performWindowControl(win, 'close'), true)
    assert.equal(win.closed, true)
  })

  test('rejects unknown actions and destroyed windows', () => {
    const win = new FakeWindow()

    assert.equal(performWindowControl(win, 'unknown'), false)
    assert.equal(performWindowControl({ ...win, isDestroyed: () => true }, 'minimize'), false)
  })
})

test('windowControlState exposes the custom-control and maximize state', () => {
  const win = new FakeWindow()
  win.maximized = true

  assert.deepEqual(windowControlState(win, true), {
    customWindowControls: true,
    isMaximized: true
  })
})

test('custom window controls use the same WSL kernel fallback as the main process', () => {
  assert.equal(
    customWindowControlsEnabled({ env: {}, kernelRelease: '6.6.87.2-microsoft-standard-WSL2', platform: 'linux' }),
    true
  )
})
