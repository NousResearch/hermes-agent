'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')

const {
  GUEST_POPUP_WINDOW,
  IN_APP_BROWSER_PARTITION,
  classifyGuestWindowOpen
} = require('./webview-guest.cjs')

test('http(s) URLs open as a same-session popup (OAuth/sign-in flows)', () => {
  assert.deepEqual(classifyGuestWindowOpen('https://accounts.google.com/o/oauth2/v2/auth?x=1'), {
    action: 'popup'
  })
  assert.deepEqual(classifyGuestWindowOpen('http://example.com/login'), { action: 'popup' })
})

test('about:blank opens as a popup (OAuth opens blank then navigates)', () => {
  assert.deepEqual(classifyGuestWindowOpen('about:blank'), { action: 'popup' })
  assert.deepEqual(classifyGuestWindowOpen('about:blank#fragment'), { action: 'popup' })
})

test('non-web schemes are handed to the OS browser (re-validated host-side)', () => {
  assert.deepEqual(classifyGuestWindowOpen('mailto:hi@example.com'), { action: 'external' })
  assert.deepEqual(classifyGuestWindowOpen('tel:+15551234567'), { action: 'external' })
  // javascript:/data: route to external where openExternalUrl rejects them —
  // they never get a popup window.
  assert.deepEqual(classifyGuestWindowOpen('javascript:alert(1)'), { action: 'external' })
  assert.deepEqual(classifyGuestWindowOpen('data:text/html,<h1>x</h1>'), { action: 'external' })
})

test('empty / non-string URLs are denied', () => {
  assert.deepEqual(classifyGuestWindowOpen(''), { action: 'deny' })
  assert.deepEqual(classifyGuestWindowOpen('   '), { action: 'deny' })
  assert.deepEqual(classifyGuestWindowOpen(undefined), { action: 'deny' })
  assert.deepEqual(classifyGuestWindowOpen(null), { action: 'deny' })
  assert.deepEqual(classifyGuestWindowOpen(42), { action: 'deny' })
})

test('whitespace around a URL is tolerated', () => {
  assert.deepEqual(classifyGuestWindowOpen('  https://example.com  '), { action: 'popup' })
})

test('exports the in-app browser partition + popup window defaults', () => {
  assert.equal(IN_APP_BROWSER_PARTITION, 'persist:hermes-browser')
  assert.equal(typeof GUEST_POPUP_WINDOW.width, 'number')
  assert.equal(typeof GUEST_POPUP_WINDOW.height, 'number')
})
