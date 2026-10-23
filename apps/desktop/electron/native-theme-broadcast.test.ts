import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  NATIVE_THEME_UPDATED_CHANNEL,
  broadcastNativeThemeUpdated
} from './native-theme-broadcast'

function fakeWindow(options: { destroyed?: boolean; contentsDestroyed?: boolean; invalidate?: boolean } = {}) {
  const { destroyed = false, contentsDestroyed = false, invalidate = true } = options
  const sent: Array<{ channel: string; payload: unknown }> = []
  let invalidated = 0
  const webContents: {
    isDestroyed: () => boolean
    send: (channel: string, payload: unknown) => void
    invalidate?: () => void
  } = {
    isDestroyed: () => contentsDestroyed,
    send: (channel: string, payload: unknown) => {
      sent.push({ channel, payload })
    }
  }

  if (invalidate) {
    webContents.invalidate = () => {
      invalidated += 1
    }
  }

  return {
    win: { isDestroyed: () => destroyed, webContents },
    sent,
    invalidated: () => invalidated
  }
}

test('notifies live renderers of the OS theme change and invalidates for repaint', () => {
  const first = fakeWindow()
  const second = fakeWindow()
  const notified = broadcastNativeThemeUpdated([first.win, second.win], true, { streaming: false })

  assert.equal(notified, 2)

  for (const f of [first, second]) {
    assert.deepEqual(f.sent, [{ channel: NATIVE_THEME_UPDATED_CHANNEL, payload: { dark: true } }])
    assert.equal(f.invalidated(), 1)
  }
})

test('skips destroyed windows and destroyed contents', () => {
  const deadWindow = fakeWindow({ destroyed: true })
  const deadContents = fakeWindow({ contentsDestroyed: true })
  const notified = broadcastNativeThemeUpdated([deadWindow.win, deadContents.win], false, {
    streaming: false
  })

  assert.equal(notified, 0)
  assert.equal(deadWindow.sent.length, 0)
  assert.equal(deadContents.sent.length, 0)
  assert.equal(deadWindow.invalidated(), 0)
})

test('still notifies but skips the invalidate while streaming', () => {
  const f = fakeWindow()
  const notified = broadcastNativeThemeUpdated([f.win], true, { streaming: true })

  assert.equal(notified, 1)
  assert.deepEqual(f.sent, [{ channel: NATIVE_THEME_UPDATED_CHANNEL, payload: { dark: true } }])
  assert.equal(f.invalidated(), 0)
})

test('tolerates windows without invalidate (older shells)', () => {
  const f = fakeWindow({ invalidate: false })
  const notified = broadcastNativeThemeUpdated([f.win], false, { streaming: false })

  assert.equal(notified, 1)
  assert.deepEqual(f.sent, [{ channel: NATIVE_THEME_UPDATED_CHANNEL, payload: { dark: false } }])
})
