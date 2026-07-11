import assert from 'node:assert/strict'
import test from 'node:test'

import {
  decideTrayClickAction,
  shouldHideToTray,
  shouldQuitOnLastWindowClose
} from './tray-lifecycle'

// ─── shouldHideToTray ─────────────────────────────────────────────────────────

test('shouldHideToTray — no tray icon → false', () => {
  assert.equal(shouldHideToTray({ hasTray: false, isHandoff: false }), false)
})

test('shouldHideToTray — no tray + handoff → false (must quit for handoff)', () => {
  assert.equal(shouldHideToTray({ hasTray: false, isHandoff: true }), false)
})

test('shouldHideToTray — has tray + not handoff → true', () => {
  assert.equal(shouldHideToTray({ hasTray: true, isHandoff: false }), true)
})

test('shouldHideToTray — has tray + handoff → false (must quit for handoff)', () => {
  assert.equal(shouldHideToTray({ hasTray: true, isHandoff: true }), false)
})

// ─── decideTrayClickAction ────────────────────────────────────────────────────

test('decideTrayClickAction — no window → create', () => {
  assert.equal(decideTrayClickAction({ hasWindow: false, isMac: true }), 'create')
  assert.equal(decideTrayClickAction({ hasWindow: false, isMac: false }), 'create')
})

test('decideTrayClickAction — visible window + macOS → hide', () => {
  const result = decideTrayClickAction({ hasWindow: true, isMac: true, isWindowVisible: true })
  assert.equal(result, 'hide')
})

test('decideTrayClickAction — visible window + Windows/Linux → minimize', () => {
  const result = decideTrayClickAction({ hasWindow: true, isMac: false, isWindowVisible: true })
  assert.equal(result, 'minimize')
})

test('decideTrayClickAction — hidden window → show', () => {
  const macResult = decideTrayClickAction({ hasWindow: true, isMac: true, isWindowVisible: false })
  const winResult = decideTrayClickAction({ hasWindow: true, isMac: false, isWindowVisible: false })
  assert.equal(macResult, 'show')
  assert.equal(winResult, 'show')
})

// ─── shouldQuitOnLastWindowClose ──────────────────────────────────────────────

test('shouldQuitOnLastWindowClose — handoff → quit', () => {
  // Handoff always quits regardless of platform/tray
  assert.equal(
    shouldQuitOnLastWindowClose({ isHandoff: true, hasTray: false, isMac: true }),
    'quit'
  )
  assert.equal(
    shouldQuitOnLastWindowClose({ isHandoff: true, hasTray: true, isMac: false }),
    'quit'
  )
})

test('shouldQuitOnLastWindowClose — no tray + mac → stay-alive', () => {
  assert.equal(
    shouldQuitOnLastWindowClose({ isHandoff: false, hasTray: false, isMac: true }),
    'stay-alive'
  )
})

test('shouldQuitOnLastWindowClose — no tray + non-mac → quit', () => {
  assert.equal(
    shouldQuitOnLastWindowClose({ isHandoff: false, hasTray: false, isMac: false }),
    'quit'
  )
})

test('shouldQuitOnLastWindowClose — has tray + mac → stay-alive', () => {
  assert.equal(
    shouldQuitOnLastWindowClose({ isHandoff: false, hasTray: true, isMac: true }),
    'stay-alive'
  )
})

test('shouldQuitOnLastWindowClose — has tray + non-mac → stay-alive', () => {
  assert.equal(
    shouldQuitOnLastWindowClose({ isHandoff: false, hasTray: true, isMac: false }),
    'stay-alive'
  )
})
