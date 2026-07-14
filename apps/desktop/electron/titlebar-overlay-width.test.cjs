const assert = require('node:assert/strict')
const test = require('node:test')

const { OVERLAY_FALLBACK_WIDTH, nativeOverlayWidth } = require('./titlebar-overlay-width.cjs')

// This static reservation is only the pre-layout FALLBACK. Once laid out the
// renderer reads the exact width from navigator.windowControlsOverlay
// (use-window-controls-overlay-width.ts) and uses these values only when the WCO
// API is unavailable.

test('Windows reserves the overlay fallback width', () => {
  assert.equal(nativeOverlayWidth({ isWindows: true }), OVERLAY_FALLBACK_WIDTH)
})

test('WSLg paints the same WCO, so it reserves the same fallback width', () => {
  // The original bug: WSL fell through to 0, so the right tools sat under the
  // controls and the title overran into them.
  assert.equal(nativeOverlayWidth({ isWsl: true }), OVERLAY_FALLBACK_WIDTH)
})

test('plain Linux and macOS reserve nothing', () => {
  assert.equal(nativeOverlayWidth({ isWindows: false, isWsl: false }), 0)
  assert.equal(nativeOverlayWidth(), 0)
  assert.equal(nativeOverlayWidth({}), 0)
})

test('the fallback width is a sane positive pixel value', () => {
  assert.ok(Number.isInteger(OVERLAY_FALLBACK_WIDTH) && OVERLAY_FALLBACK_WIDTH > 0)
})
