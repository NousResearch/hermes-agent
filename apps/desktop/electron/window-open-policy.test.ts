import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createWindowOpenHandler, describeDeniedUrl } from './window-open-policy'

test('host handler denies every request and reports the origin only', () => {
  const seen: string[] = []
  const handler = createWindowOpenHandler(origin => seen.push(origin))

  assert.deepEqual(handler({ url: 'https://evil.example/path?token=x' }), { action: 'deny' })
  assert.deepEqual(handler({ url: 'file:///etc/passwd' }), { action: 'deny' })
  // Full URLs (query credentials, paths) never reach the observer.
  assert.deepEqual(seen, ['https://evil.example', 'file:'])
})

test('host handler keeps denying when the observer throws', () => {
  const handler = createWindowOpenHandler(() => {
    throw new Error('observer blew up')
  })

  assert.deepEqual(handler({ url: 'https://evil.example/' }), { action: 'deny' })
})

test('describeDeniedUrl sanitizes unparseable and opaque origins', () => {
  assert.equal(describeDeniedUrl('https://example.com/x?y=1'), 'https://example.com')
  assert.equal(describeDeniedUrl('data:text/html,hi'), 'data:')
  assert.equal(describeDeniedUrl('not a url'), '<unparseable>')
})
