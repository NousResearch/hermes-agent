import assert from 'node:assert/strict'

import { test } from 'vitest'

import { createGuestWebviewWindowOpenHandler, createWindowOpenHandler, describeDeniedUrl } from './window-open-policy'

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

test('guest webview handler hands accepted URLs to the audited channel and still denies', () => {
  const handed: string[] = []

  const handler = createGuestWebviewWindowOpenHandler(url => {
    handed.push(url)

    return url.startsWith('https:')
  })

  // Accepted by the channel: opened in the OS browser, still no Electron popup.
  assert.deepEqual(handler({ url: 'https://www.google.com/search?q=traceback' }), { action: 'deny' })
  // Rejected by the channel (the allowlist in openExternalUrl said no): dropped.
  assert.deepEqual(handler({ url: 'javascript:alert(1)' }), { action: 'deny' })
  assert.deepEqual(handed, ['https://www.google.com/search?q=traceback', 'javascript:alert(1)'])
})

test('guest webview handler logs the origin only, and survives a throwing observer', () => {
  const seen: string[] = []

  const handler = createGuestWebviewWindowOpenHandler(
    () => true,
    origin => {
      seen.push(origin)

      if (seen.length === 2) {
        throw new Error('observer blew up')
      }
    }
  )

  assert.deepEqual(handler({ url: 'https://example.com/a?session=secret' }), { action: 'deny' })
  assert.deepEqual(handler({ url: 'https://example.com/b' }), { action: 'deny' })
  assert.deepEqual(seen, ['https://example.com', 'https://example.com'])
})

test('describeDeniedUrl sanitizes unparseable and opaque origins', () => {
  assert.equal(describeDeniedUrl('https://example.com/x?y=1'), 'https://example.com')
  assert.equal(describeDeniedUrl('data:text/html,hi'), 'data:')
  assert.equal(describeDeniedUrl('not a url'), '<unparseable>')
})
