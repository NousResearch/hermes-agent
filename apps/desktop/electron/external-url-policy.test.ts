import assert from 'node:assert/strict'

import { test } from 'vitest'

import { isExternalProtocolAllowed, isExternalUrlAllowed } from './external-url-policy'

test('external URL policy allows phone links and rejects unsafe protocols', () => {
  assert.equal(isExternalProtocolAllowed('tel:'), true)
  assert.equal(isExternalProtocolAllowed('TEL:'), true)
  assert.equal(isExternalProtocolAllowed('https:'), true)
  assert.equal(isExternalProtocolAllowed('mailto:'), true)
  assert.equal(isExternalProtocolAllowed('javascript:'), false)
  assert.equal(isExternalProtocolAllowed('data:'), false)
  assert.equal(isExternalProtocolAllowed('file:'), false)
})

test('external URL policy rejects unsafe telephone payloads', () => {
  assert.equal(isExternalUrlAllowed(new URL('tel:')), false)
  assert.equal(isExternalUrlAllowed(new URL('tel:1234567')), true)
  assert.equal(isExternalUrlAllowed(new URL('tel:123456789012345')), true)
  assert.equal(isExternalUrlAllowed(new URL('tel:123456')), false)
  assert.equal(isExternalUrlAllowed(new URL('tel:1234567890123456')), false)
  assert.equal(isExternalUrlAllowed(new URL('tel:0104956404')), true)
  assert.equal(isExternalUrlAllowed(new URL('tel:+46104956404')), true)
  assert.equal(isExternalUrlAllowed(new URL('tel:010-495-64-04')), false)
  assert.equal(isExternalUrlAllowed(new URL('tel:(212)5550100')), false)
  assert.equal(isExternalUrlAllowed(new URL('tel:123&calc')), false)
  assert.equal(isExternalUrlAllowed(new URL('tel:1234567?x=1')), false)
  assert.equal(isExternalUrlAllowed(new URL('tel:1234567#x')), false)
  assert.equal(isExternalUrlAllowed(new URL('tel:123%264567')), false)
  assert.equal(isExternalUrlAllowed(new URL('tel:１２３４５６７')), false)
  assert.equal(isExternalUrlAllowed(new URL('tel://1234567')), false)
  assert.equal(isExternalUrlAllowed(new URL('tel:123')), false)
})
