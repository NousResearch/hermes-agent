import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import {
  DEFAULT_CLOSE_BEHAVIOR,
  readCloseBehaviorState,
  shouldHideToTray,
  writeCloseBehaviorState,
  type CloseBehaviorState
} from './close-behavior'

function makeStateFile(): string {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'close-behavior-test-'))

  return path.join(dir, 'close-behavior.json')
}

test('missing file falls back to the default state', () => {
  const filePath = makeStateFile()
  const state = readCloseBehaviorState(filePath)

  assert.deepEqual(state, DEFAULT_CLOSE_BEHAVIOR)
})

test('corrupt JSON falls back to the default state', () => {
  const filePath = makeStateFile()
  fs.writeFileSync(filePath, '{not json')

  assert.deepEqual(readCloseBehaviorState(filePath), DEFAULT_CLOSE_BEHAVIOR)
})

test('write then read round-trips mode and trayNotified', () => {
  const filePath = makeStateFile()
  const state: CloseBehaviorState = { mode: 'quit', trayNotified: true }

  writeCloseBehaviorState(filePath, state)

  assert.deepEqual(readCloseBehaviorState(filePath), state)
})

test('unknown mode in the file falls back to tray', () => {
  const filePath = makeStateFile()
  fs.writeFileSync(filePath, JSON.stringify({ mode: 'minimize', trayNotified: false }))

  assert.deepEqual(readCloseBehaviorState(filePath), { mode: 'tray', trayNotified: false })
})

test('trayNotified survives a restart (persisted once-per-install balloon)', () => {
  const filePath = makeStateFile()

  // First close-to-tray: not notified yet → shows the balloon.
  const first = readCloseBehaviorState(filePath)
  assert.equal(first.trayNotified, false)

  // After showing the balloon the flag is persisted.
  writeCloseBehaviorState(filePath, { ...first, trayNotified: true })

  // A fresh read (new process, same file) sees the flag.
  assert.equal(readCloseBehaviorState(filePath).trayNotified, true)
})

test('shouldHideToTray: Windows + tray mode + not quitting → hide', () => {
  assert.equal(shouldHideToTray({ isWindows: true, isQuitting: false, mode: 'tray' }), true)
})

test('shouldHideToTray: non-Windows keeps the stock close behavior', () => {
  assert.equal(shouldHideToTray({ isWindows: false, isQuitting: false, mode: 'tray' }), false)
})

test('shouldHideToTray: a real quit always closes the window', () => {
  assert.equal(shouldHideToTray({ isWindows: true, isQuitting: true, mode: 'tray' }), false)
})

test('shouldHideToTray: quit mode disables the tray path', () => {
  assert.equal(shouldHideToTray({ isWindows: true, isQuitting: false, mode: 'quit' }), false)
})
