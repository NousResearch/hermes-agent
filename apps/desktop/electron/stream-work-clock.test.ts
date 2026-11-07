import { EventEmitter } from 'node:events'

import type { BrowserWindow, IpcMain, WebContents } from 'electron'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'

import { registerStreamWorkClock } from './stream-work-clock'

beforeEach(() => vi.useFakeTimers())
afterEach(() => vi.useRealTimers())

function fixture() {
  const delivered: string[] = []
  const ipc = new EventEmitter()

  const sender = Object.assign(new EventEmitter(), {
    mainFrame: {
      detached: false,
      isDestroyed: () => false,
      send: (_channel: string, token: string) => delivered.push(token)
    },
    isDestroyed: () => false
  })

  const owner = { webContents: sender, isDestroyed: () => false }
  registerStreamWorkClock(ipc as unknown as IpcMain, () => owner as unknown as BrowserWindow)

  return {
    sender,
    delivered,
    request(payload: unknown, frame: object = sender.mainFrame, contents: object = sender) {
      ipc.emit('hermes:stream-work-wake', { sender: contents as WebContents, senderFrame: frame }, payload)
    }
  }
}

test('replacing a deadline does not deliver superseded work; stale cancellation cannot cancel its replacement', () => {
  const f = fixture()
  f.request({ token: 'old', delayMs: 400 })
  vi.advanceTimersByTime(100)
  f.request({ token: 'new', delayMs: 33 })
  f.request({ token: 'old', delayMs: null })
  vi.advanceTimersByTime(32)
  expect(f.delivered).toEqual([])
  vi.advanceTimersByTime(1)
  expect(f.delivered).toEqual(['new'])
  vi.advanceTimersByTime(1000)
  expect(f.delivered).toEqual(['new'])
})

test('subframes, guest contents and malformed requests cannot replace an owner deadline', () => {
  const f = fixture()
  f.request({ token: 'owner', delayMs: 400 })
  f.request({ token: 'child', delayMs: 0 }, {})
  const guest = { mainFrame: {} }
  f.request({ token: 'guest', delayMs: 0 }, guest.mainFrame, guest)

  for (const payload of [
    null,
    {},
    { token: 1, delayMs: 0 },
    { token: '', delayMs: 0 },
    { token: 'x'.repeat(81), delayMs: 0 },
    { token: 'bad', delayMs: NaN },
    { token: 'bad', delayMs: Infinity },
    { token: 'bad', delayMs: -1 },
    { token: 'bad', delayMs: 401 }
  ]) {
    f.request(payload)
  }

  vi.advanceTimersByTime(400)
  expect(f.delivered).toEqual(['owner'])
})

test.each(['render-process-gone', 'did-navigate', 'destroyed'])('%s drops the old document deadline', event => {
  const f = fixture()
  f.request({ token: 'old-document', delayMs: 400 })
  f.sender.emit(event)
  vi.advanceTimersByTime(1000)
  expect(f.delivered).toEqual([])

  if (event !== 'destroyed') {
    f.request({ token: 'new-document', delayMs: 33 })
    vi.advanceTimersByTime(33)
    expect(f.delivered).toEqual(['new-document'])
  }
})

test('cancelled work cannot fire later', () => {
  const f = fixture()
  f.request({ token: 'cancelled', delayMs: 400 })
  f.request({ token: 'cancelled', delayMs: null })
  vi.advanceTimersByTime(1000)
  expect(f.delivered).toEqual([])
})

test('a detached document cannot receive a deadline even before lifecycle cleanup arrives', () => {
  const f = fixture()
  f.request({ token: 'old-document', delayMs: 400 })
  f.sender.mainFrame.detached = true
  vi.advanceTimersByTime(400)
  expect(f.delivered).toEqual([])
})
