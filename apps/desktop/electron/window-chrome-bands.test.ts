import assert from 'node:assert/strict'

import { test } from 'vitest'

import { CLASSIC_TITLEBAR_HEIGHT, PORTAL_TITLEBAR_HEIGHT, SHELL_CHOICE_KEY, windowButtonPosition } from './window-chrome'
import { createWindowChromeBands } from './window-chrome-bands'

function fakeWindow() {
  const closed: (() => void)[] = []

  return {
    positions: [] as { x: number; y: number }[],
    close: () => closed.forEach(listener => listener()),
    isDestroyed: () => false,
    once: (_event: 'closed', listener: () => void) => closed.push(listener),
    setWindowButtonPosition(position: { x: number; y: number }) {
      this.positions.push(position)
    }
  }
}

test('a shell flip moves every open shell window to its new band; the Browser stays classic', () => {
  const relaid: unknown[] = []
  const bands = createWindowChromeBands({ isMac: true, portal: true, relayout: win => relaid.push(win) })
  const chat = fakeWindow()
  const browser = fakeWindow()

  bands.track(chat, 'chat')
  bands.track(browser, 'browser')
  assert.deepEqual(chat.positions, [windowButtonPosition(PORTAL_TITLEBAR_HEIGHT)])
  assert.deepEqual(browser.positions, [windowButtonPosition(CLASSIC_TITLEBAR_HEIGHT)])
  assert.equal(bands.titlebarHeightFor(bands.kindOf(chat)), PORTAL_TITLEBAR_HEIGHT)

  bands.applyPreferences({ [SHELL_CHOICE_KEY]: 'false' })
  assert.equal(bands.titlebarHeightFor(bands.kindOf(chat)), CLASSIC_TITLEBAR_HEIGHT)
  assert.deepEqual(chat.positions.at(-1), windowButtonPosition(CLASSIC_TITLEBAR_HEIGHT))
  assert.deepEqual(relaid, [chat, browser])

  // An unrelated preference write is not a flip.
  bands.applyPreferences({ [SHELL_CHOICE_KEY]: 'false', other: '1' })
  assert.equal(relaid.length, 2)

  chat.close()
  bands.applyPreferences({})
  assert.deepEqual(relaid, [chat, browser, browser])
})

test('only macOS seats traffic lights', () => {
  const bands = createWindowChromeBands({ isMac: false, portal: true, relayout: () => undefined })
  const win = fakeWindow()

  bands.track(win, 'chat')
  assert.deepEqual(win.positions, [])
})
