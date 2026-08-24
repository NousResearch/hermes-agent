/**
 * Unit tests for the HUD input policy. The split is the point: the solid
 * fallback has to reach the X11 windows where the input region never comes
 * back, and has to leave native Wayland (and macOS/Windows) on click-through.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { hudInputPolicy, hudUsesNativeDrag, hudUsesWorkspaceTransfer } from './hud-input-policy'

const X11_SESSION = { DISPLAY: ':0', XDG_SESSION_TYPE: 'x11' }
const WAYLAND_SESSION = { WAYLAND_DISPLAY: 'wayland-0', XDG_SESSION_TYPE: 'wayland' }
const XWAYLAND_SESSION = { DISPLAY: ':0', WAYLAND_DISPLAY: 'wayland-0', XDG_SESSION_TYPE: 'wayland' }

test('the click-through design is kept everywhere it works', () => {
  for (const platform of ['darwin', 'win32']) {
    assert.equal(hudInputPolicy(platform, {}, []), 'click-through')
    assert.equal(hudInputPolicy(platform, X11_SESSION, []), 'click-through')
  }
})

test('an X11 session gets the solid HUD', () => {
  assert.equal(hudInputPolicy('linux', X11_SESSION, []), 'solid')
})

test('a Wayland session keeps click-through — Electron 20+ is a native Wayland client there', () => {
  assert.equal(hudInputPolicy('linux', WAYLAND_SESSION, []), 'click-through')
  assert.equal(hudInputPolicy('linux', XWAYLAND_SESSION, []), 'click-through')
})

test('asking for a native Wayland surface keeps the click-through path', () => {
  for (const argv of [['--ozone-platform=wayland'], ['--ozone-platform-hint=wayland']]) {
    assert.equal(hudInputPolicy('linux', WAYLAND_SESSION, argv), 'click-through')
  }

  assert.equal(
    hudInputPolicy('linux', { ...WAYLAND_SESSION, ELECTRON_OZONE_PLATFORM_HINT: 'wayland' }, []),
    'click-through'
  )
})

test('an auto hint follows the session', () => {
  assert.equal(hudInputPolicy('linux', WAYLAND_SESSION, ['--ozone-platform-hint=auto']), 'click-through')
  assert.equal(hudInputPolicy('linux', X11_SESSION, ['--ozone-platform-hint=auto']), 'solid')
  assert.equal(hudInputPolicy('linux', XWAYLAND_SESSION, ['--ozone-platform-hint=auto']), 'click-through')
})

test('asking for X11 on a Wayland session gets the solid HUD', () => {
  assert.equal(hudInputPolicy('linux', WAYLAND_SESSION, ['--ozone-platform=x11']), 'solid')
  assert.equal(hudInputPolicy('linux', { ...WAYLAND_SESSION, ELECTRON_OZONE_PLATFORM_HINT: 'x11' }, []), 'solid')
})

test('the explicit switch beats the hint, and the last switch wins', () => {
  assert.equal(
    hudInputPolicy('linux', WAYLAND_SESSION, ['--ozone-platform-hint=auto', '--ozone-platform=x11']),
    'solid'
  )
  assert.equal(
    hudInputPolicy('linux', X11_SESSION, ['--ozone-platform=x11', '--ozone-platform=wayland']),
    'click-through'
  )
})

test('a backend nobody recognises follows the session, not a silent X11 default', () => {
  assert.equal(hudInputPolicy('linux', X11_SESSION, ['--ozone-platform=headless']), 'solid')
  assert.equal(hudInputPolicy('linux', WAYLAND_SESSION, ['--ozone-platform=headless']), 'click-through')
  assert.equal(hudInputPolicy('linux', {}, []), 'solid')
})

test('only a native Wayland HUD gets a compositor drag region', () => {
  assert.equal(hudUsesNativeDrag('darwin', WAYLAND_SESSION, []), false)
  assert.equal(hudUsesNativeDrag('win32', WAYLAND_SESSION, []), false)
  assert.equal(hudUsesNativeDrag('linux', X11_SESSION, []), false)
  assert.equal(hudUsesNativeDrag('linux', WAYLAND_SESSION, []), true)
  assert.equal(hudUsesNativeDrag('linux', WAYLAND_SESSION, ['--ozone-platform=x11']), false)
  assert.equal(hudUsesNativeDrag('linux', X11_SESSION, ['--ozone-platform=wayland']), true)
})

test('only a renderer-driven Linux HUD needs temporary all-workspace visibility', () => {
  assert.equal(hudUsesWorkspaceTransfer('darwin', X11_SESSION, []), false)
  assert.equal(hudUsesWorkspaceTransfer('win32', X11_SESSION, []), false)
  assert.equal(hudUsesWorkspaceTransfer('linux', X11_SESSION, []), true)
  assert.equal(hudUsesWorkspaceTransfer('linux', WAYLAND_SESSION, []), false)
  assert.equal(hudUsesWorkspaceTransfer('linux', WAYLAND_SESSION, ['--ozone-platform=x11']), true)
  assert.equal(hudUsesWorkspaceTransfer('linux', X11_SESSION, ['--ozone-platform=wayland']), false)
})
