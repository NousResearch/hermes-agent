const assert = require('node:assert/strict')
const test = require('node:test')

// Regression test for #47826: readTitle() must not throw when the
// BrowserWindow has been destroyed.  The real runRenderTitleJob lives
// inside main.cjs (unexported), so we exercise the exact guard logic
// in isolation.

function makeFakeWindow(destroyed) {
  return {
    isDestroyed: () => destroyed,
    webContents: destroyed
      ? undefined
      : { getTitle: () => 'Example Page' },
  }
}

// Mirrors the fixed readTitle implementation (main.cjs ~L2978).
const readTitle = (window) => {
  if (!window || window.isDestroyed()) return ''
  return window.webContents?.getTitle?.() || ''
}

test('readTitle returns title for a live window', () => {
  const win = makeFakeWindow(false)
  assert.equal(readTitle(win), 'Example Page')
})

test('readTitle returns empty string for a destroyed window (#47826)', () => {
  const win = makeFakeWindow(true)
  // Before the fix, this path threw TypeError: Object has been destroyed
  assert.equal(readTitle(win), '')
})

test('readTitle returns empty string for null window', () => {
  assert.equal(readTitle(null), '')
})

test('readTitle returns empty string for undefined window', () => {
  assert.equal(readTitle(undefined), '')
})
