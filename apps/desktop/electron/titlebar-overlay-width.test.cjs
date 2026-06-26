const assert = require('node:assert/strict')
const test = require('node:test')

const {
  NATIVE_OVERLAY_BUTTON_WIDTH,
  WSL_OVERLAY_RESERVE_WIDTH,
  nativeOverlayWidth
} = require('./titlebar-overlay-width.cjs')

// Note: this static reservation is the FALLBACK. The renderer prefers the exact
// width measured live from navigator.windowControlsOverlay (see
// use-window-controls-overlay-width.ts) and only uses these values when the WCO
// API is unavailable.

test('native Windows reserves the standard WCO footprint', () => {
  assert.equal(nativeOverlayWidth({ isWindows: true, isWsl: false }), NATIVE_OVERLAY_BUTTON_WIDTH)
})

test('WSLg reserves its own (further-inset) overlay width as a fallback', () => {
  // The original bug: WSL fell through to 0, so the right tools sat under the
  // controls and the title overran into them. WSLg's WCO cluster sits further
  // from the right edge than native Windows, hence its own reserve width.
  assert.equal(nativeOverlayWidth({ isWindows: false, isWsl: true }), WSL_OVERLAY_RESERVE_WIDTH)
})

test('plain Linux (overlay disabled) reserves nothing', () => {
  assert.equal(nativeOverlayWidth({ isWindows: false, isWsl: false }), 0)
})

test('macOS-style call (no flags) reserves nothing — traffic lights are on the left', () => {
  assert.equal(nativeOverlayWidth(), 0)
  assert.equal(nativeOverlayWidth({}), 0)
})

test('Windows takes precedence if both flags are somehow set', () => {
  assert.equal(nativeOverlayWidth({ isWindows: true, isWsl: true }), NATIVE_OVERLAY_BUTTON_WIDTH)
})

test('both reservation widths are sane positive pixel values', () => {
  assert.ok(Number.isInteger(NATIVE_OVERLAY_BUTTON_WIDTH) && NATIVE_OVERLAY_BUTTON_WIDTH > 0)
  assert.ok(Number.isInteger(WSL_OVERLAY_RESERVE_WIDTH) && WSL_OVERLAY_RESERVE_WIDTH > 0)
})
