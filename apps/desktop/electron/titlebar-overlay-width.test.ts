import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  MACOS_TAHOE_DARWIN_MAJOR,
  macTitleBarOverlayHeight,
  nativeOverlayWidth,
  OVERLAY_FALLBACK_WIDTH,
  scaledOverlayHeight,
  titleBarOverlayOptions
} from './titlebar-overlay-width'

// This static reservation is only the pre-layout FALLBACK. Once laid out the
// renderer reads the exact width from navigator.windowControlsOverlay
// (use-window-controls-overlay-width.ts) and uses these values only when the WCO
// API is unavailable.

test('Windows reserves the overlay fallback width', () => {
  assert.equal(nativeOverlayWidth({ isWindows: true }), OVERLAY_FALLBACK_WIDTH)
})

test('WSLg custom controls reserve the same fallback width', () => {
  // The original bug: WSL fell through to 0, so the right tools sat under the
  // controls and the title overran into them.
  assert.equal(nativeOverlayWidth({ isWsl: true }), OVERLAY_FALLBACK_WIDTH)
})

test('WSLg disables the undersized native overlay in favor of renderer controls', () => {
  assert.equal(
    titleBarOverlayOptions({
      platform: 'wslg',
      titlebarHeight: 34,
      color: 'transparent',
      foreground: '#ffffff',
      dark: true
    }),
    false
  )
})

test('native Windows and Linux keep the same window-controls overlay', () => {
  const input = { titlebarHeight: 34, color: 'transparent', foreground: '#ffffff', dark: false }
  const expected = { color: 'transparent', height: 34, symbolColor: '#ffffff' }

  for (const platform of ['windows', 'linux'] as const) {
    assert.deepEqual(titleBarOverlayOptions({ platform, ...input }), expected)
  }
})

test('macOS keeps its height-only traffic-light overlay', () => {
  assert.deepEqual(
    titleBarOverlayOptions({
      platform: 'mac',
      darwinMajor: MACOS_TAHOE_DARWIN_MAJOR,
      titlebarHeight: 34,
      color: 'transparent',
      foreground: '#ffffff'
    }),
    { height: 0 }
  )
})

test('plain Linux paints the WCO too, so it reserves the fallback width', () => {
  // Regression #53185: re-enabling the overlay on plain Linux (KDE/GNOME)
  // without reserving its width left the native min/max/close buttons painting
  // on top of the app's right-edge titlebar tools.
  assert.equal(nativeOverlayWidth({ isWindows: false, isWsl: false }), OVERLAY_FALLBACK_WIDTH)
  assert.equal(nativeOverlayWidth(), OVERLAY_FALLBACK_WIDTH)
  assert.equal(nativeOverlayWidth({}), OVERLAY_FALLBACK_WIDTH)
})

test('macOS uses traffic lights, not a WCO overlay, so it reserves nothing', () => {
  assert.equal(nativeOverlayWidth({ isMac: true }), 0)
})

test('pre-Tahoe keeps the full titlebar overlay height', () => {
  assert.equal(macTitleBarOverlayHeight({ darwinMajor: MACOS_TAHOE_DARWIN_MAJOR - 1, titlebarHeight: 34 }), 34)
})

test('Tahoe (Darwin 25+) drops the overlay height to 0 to avoid electron#49183', () => {
  assert.equal(macTitleBarOverlayHeight({ darwinMajor: MACOS_TAHOE_DARWIN_MAJOR, titlebarHeight: 34 }), 0)
  assert.equal(macTitleBarOverlayHeight({ darwinMajor: MACOS_TAHOE_DARWIN_MAJOR + 1, titlebarHeight: 34 }), 0)
})

test('macTitleBarOverlayHeight tolerates missing args (unknown platform → 0)', () => {
  assert.equal(macTitleBarOverlayHeight(), 0)
})

// -- zoom-scaled overlay height (#81086) -------------------------------------

test('scaledOverlayHeight tracks the zoom factor in whole pixels', () => {
  assert.equal(scaledOverlayHeight(34, 1), 34)
  assert.equal(scaledOverlayHeight(34, 1.2), 41)
  assert.equal(scaledOverlayHeight(34, 0.9), 31)
})

test('scaledOverlayHeight clamps garbage input instead of breaking the overlay', () => {
  assert.equal(scaledOverlayHeight(34, NaN), 34)
  assert.equal(scaledOverlayHeight(34, 0), 34)
  assert.equal(scaledOverlayHeight(34, -2), 34)
  assert.equal(scaledOverlayHeight(NaN, 1.5), NaN)
})

test('scaledOverlayHeight never returns a zero-height overlay', () => {
  // Extreme zoom-out: the overlay API is integer-based and 0 would collapse it.
  assert.equal(scaledOverlayHeight(34, 0.01), 1)
})

test('titleBarOverlayOptions scales the Windows/Linux overlay height with zoom', () => {
  const base = { titlebarHeight: 34, color: 'transparent', foreground: '#ffffff', dark: false }

  assert.deepEqual(titleBarOverlayOptions({ platform: 'windows', ...base, zoomFactor: 1 }), {
    color: 'transparent',
    height: 34,
    symbolColor: '#ffffff'
  })
  assert.deepEqual(titleBarOverlayOptions({ platform: 'windows', ...base, zoomFactor: 1.2 }), {
    color: 'transparent',
    height: 41,
    symbolColor: '#ffffff'
  })
})

test('titleBarOverlayOptions keeps WSLg and macOS untouched by zoom', () => {
  // WSLg: renderer paints its own scaled controls. macOS: traffic lights,
  // positioned separately; a zoomed height would shove them out of place.
  assert.equal(
    titleBarOverlayOptions({ platform: 'wslg', titlebarHeight: 34, color: 'transparent', zoomFactor: 1.5 }),
    false
  )
  assert.deepEqual(
    titleBarOverlayOptions({
      platform: 'mac',
      darwinMajor: MACOS_TAHOE_DARWIN_MAJOR,
      titlebarHeight: 34,
      zoomFactor: 1.5
    }),
    { height: 0 }
  )
})

test('default zoom leaves the overlay byte-identical to today', () => {
  // zoomFactor defaults to 1: no caller change, no height change.
  const base = { titlebarHeight: 34, color: 'transparent', foreground: '#ffffff', dark: false }
  assert.deepEqual(titleBarOverlayOptions({ platform: 'windows', ...base }), {
    color: 'transparent',
    height: 34,
    symbolColor: '#ffffff'
  })
})
