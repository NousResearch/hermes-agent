/**
 * Unit tests for the X11-backed HUD cursor reader.
 *
 * The reader exists because `screen.getCursorScreenPoint()` freezes on X11 the
 * instant the HUD window ignores the mouse (Chromium's cursor cache stops being
 * fed by motion events), which is the one-way door behind the Linux HUD bug.
 * XQueryPointer asks the X server directly, so it always knows the real
 * pointer position. These tests pin down the reader's contract: it never
 * throws, it degrades to null when the X connection is unavailable, and it
 * stops polling after close().
 */

import assert from 'node:assert/strict'

import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'

import { createX11CursorReader } from './hud-cursor-x11'

// --- mock the x11 module -----------------------------------------------------
// The tests run in a headless environment with no X server; the module under
// test is the only consumer, so a hand-rolled fake is exact.

type QueryPointerCb = (err: Error | null, reply: { rootX: number; rootY: number } | null) => void

const mocks = vi.hoisted(() => ({
  createClient: vi.fn(),
  queryPointer: vi.fn(),
  terminate: vi.fn()
}))

vi.mock('x11', () => ({
  default: {
    createClient: mocks.createClient
  }
}))

describe('createX11CursorReader', () => {
  let captureCreateCallback: (err: Error | null, display?: any) => void

  beforeEach(() => {
    vi.clearAllMocks()
    mocks.createClient.mockImplementation((cb: (err: Error | null, display?: any) => void) => {
      captureCreateCallback = cb
    })
  })

  afterEach(() => {
    vi.clearAllMocks()
  })

  function connectDisplay() {
    captureCreateCallback(null, {
      client: { QueryPointer: mocks.queryPointer, terminate: mocks.terminate },
      screen: [{ root: 0x1234 }]
    })
  }

  test('read() returns null before the X connection is established', () => {
    const reader = createX11CursorReader()
    assert.equal(reader.read(), null)
  })

  test('read() returns the latest XQueryPointer result', () => {
    mocks.queryPointer.mockImplementation((_root: number, cb: QueryPointerCb) => {
      cb(null, { rootX: 100, rootY: 200 })
    })

    const reader = createX11CursorReader()
    connectDisplay()
    reader.read()
    reader.read()

    assert.deepEqual(reader.read(), { x: 100, y: 200 })
    expect(mocks.queryPointer).toHaveBeenCalled()
  })

  test('a failed X connection degrades to null instead of throwing', () => {
    const reader = createX11CursorReader()
    captureCreateCallback(new Error('cannot open display'))
    assert.equal(reader.read(), null)
  })

  test('close() terminates the X connection and stops polling', () => {
    mocks.queryPointer.mockImplementation((_root: number, cb: QueryPointerCb) => {
      cb(null, { rootX: 1, rootY: 2 })
    })

    const reader = createX11CursorReader()
    connectDisplay()
    reader.read()
    reader.close()
    expect(mocks.terminate).toHaveBeenCalled()
    // After close, no further polling happens.
    const callsBefore = mocks.queryPointer.mock.calls.length
    reader.read()
    assert.equal(mocks.queryPointer.mock.calls.length, callsBefore)
  })
})
