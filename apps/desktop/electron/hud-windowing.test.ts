/**
 * Unit tests for the HUD windowing profile. Ozone parsing and the capability
 * matrix live here: one profile, every operation reads the same flags.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { hudWindowingView, linuxOzoneBackend, resolveHudWindowing } from './hud-windowing'

const X11_SESSION = { DISPLAY: ':0', XDG_SESSION_TYPE: 'x11' }
const WAYLAND_SESSION = { WAYLAND_DISPLAY: 'wayland-0', XDG_SESSION_TYPE: 'wayland' }
const XWAYLAND_SESSION = { DISPLAY: ':0', WAYLAND_DISPLAY: 'wayland-0', XDG_SESSION_TYPE: 'wayland' }

const input = (platform: string, env: NodeJS.ProcessEnv, argv: readonly string[] = []) =>
  resolveHudWindowing(platform, env, argv).input

test('macOS and Windows share the click-through + renderer-drag profile', () => {
  for (const platform of ['darwin', 'win32'] as const) {
    const windowing = resolveHudWindowing(platform, WAYLAND_SESSION, ['--ozone-platform=wayland'])

    assert.equal(windowing.backend, platform === 'darwin' ? 'cocoa' : 'win32')
    assert.equal(windowing.input, 'click-through')
    assert.equal(windowing.move, 'renderer')
    assert.equal(windowing.clientPlacement, true)
    assert.equal(windowing.controlDrag, false)
    assert.equal(windowing.workspaceTransfer, false)
    assert.equal(windowing.cursorFeed, false)
    assert.equal(windowing.ignoreMouse, true)
  }
})

test('Linux X11 is solid, renderer-drag, and can place the window', () => {
  const windowing = resolveHudWindowing('linux', X11_SESSION, [])

  assert.equal(windowing.backend, 'x11')
  assert.equal(windowing.input, 'solid')
  assert.equal(windowing.move, 'renderer')
  assert.equal(windowing.clientPlacement, true)
  assert.equal(windowing.controlDrag, true)
  assert.equal(windowing.workspaceTransfer, true)
  assert.equal(windowing.cursorFeed, false)
  assert.equal(windowing.ignoreMouse, false)
})

test('native Wayland cannot place the window and needs compositor drag + cursor feed', () => {
  const windowing = resolveHudWindowing('linux', WAYLAND_SESSION, [])

  assert.equal(windowing.backend, 'wayland')
  assert.equal(windowing.input, 'click-through')
  assert.equal(windowing.move, 'native-drag')
  assert.equal(windowing.clientPlacement, false)
  assert.equal(windowing.controlDrag, false)
  assert.equal(windowing.workspaceTransfer, false)
  assert.equal(windowing.cursorFeed, true)
  assert.equal(windowing.ignoreMouse, true)
})

test('a Wayland session with DISPLAY set is still a native Wayland client under Electron 20+', () => {
  assert.equal(input('linux', XWAYLAND_SESSION), 'click-through')
})

test('asking for a native Wayland surface keeps the click-through path', () => {
  for (const argv of [['--ozone-platform=wayland'], ['--ozone-platform-hint=wayland']]) {
    assert.equal(input('linux', WAYLAND_SESSION, argv), 'click-through')
  }

  assert.equal(input('linux', { ...WAYLAND_SESSION, ELECTRON_OZONE_PLATFORM_HINT: 'wayland' }), 'click-through')
})

test('an auto hint follows the session', () => {
  assert.equal(input('linux', WAYLAND_SESSION, ['--ozone-platform-hint=auto']), 'click-through')
  assert.equal(input('linux', X11_SESSION, ['--ozone-platform-hint=auto']), 'solid')
  assert.equal(input('linux', XWAYLAND_SESSION, ['--ozone-platform-hint=auto']), 'click-through')
})

test('asking for X11 on a Wayland session takes the solid / renderer-drag profile', () => {
  const windowing = resolveHudWindowing('linux', WAYLAND_SESSION, ['--ozone-platform=x11'])

  assert.equal(windowing.backend, 'x11')
  assert.equal(windowing.input, 'solid')
  assert.equal(windowing.move, 'renderer')
  assert.equal(windowing.clientPlacement, true)
  assert.equal(input('linux', { ...WAYLAND_SESSION, ELECTRON_OZONE_PLATFORM_HINT: 'x11' }), 'solid')
})

test('the explicit switch beats the hint, and the last switch wins', () => {
  assert.equal(input('linux', WAYLAND_SESSION, ['--ozone-platform-hint=auto', '--ozone-platform=x11']), 'solid')
  assert.equal(input('linux', X11_SESSION, ['--ozone-platform=x11', '--ozone-platform=wayland']), 'click-through')
})

test('a backend nobody recognises follows the session, not a silent X11 default', () => {
  assert.equal(input('linux', X11_SESSION, ['--ozone-platform=headless']), 'solid')
  assert.equal(input('linux', WAYLAND_SESSION, ['--ozone-platform=headless']), 'click-through')
  assert.equal(input('linux', {}, []), 'solid')
})

test('linuxOzoneBackend is the session/argv normalizer', () => {
  assert.equal(linuxOzoneBackend(X11_SESSION, []), 'x11')
  assert.equal(linuxOzoneBackend(WAYLAND_SESSION, []), 'wayland')
  assert.equal(linuxOzoneBackend(WAYLAND_SESSION, ['--ozone-platform=x11']), 'x11')
  assert.equal(linuxOzoneBackend(X11_SESSION, ['--ozone-platform-hint=auto']), 'x11')
})

test('the renderer view is a boolean slice of the profile', () => {
  const wayland = hudWindowingView(resolveHudWindowing('linux', WAYLAND_SESSION, []))
  const x11 = hudWindowingView(resolveHudWindowing('linux', X11_SESSION, []))

  assert.deepEqual(wayland, {
    clientPlacement: false,
    controlDrag: false,
    nativeDrag: true,
    workspaceTransfer: false
  })
  assert.deepEqual(x11, {
    clientPlacement: true,
    controlDrag: true,
    nativeDrag: false,
    workspaceTransfer: true
  })
})
