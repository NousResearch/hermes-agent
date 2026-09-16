/**
 * Security regression for GHSA-9f4c-93c8-jc8g (CVE-2026-70608): a sandboxed
 * iframe without `allow-popups` can reach `setWindowOpenHandler` with no user
 * gesture, so the handler must always deny and must never open a URL. These
 * import the real policy module main.ts wires in.
 */

import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import { createWindowOpenHandler, describeDeniedUrl } from '../apps/desktop/electron/window-open-policy'

describe('window-open policy (GHSA-9f4c-93c8-jc8g)', () => {
  test('denies every scheme and reports only the sanitized origin', () => {
    const seen: string[] = []
    const handler = createWindowOpenHandler(origin => seen.push(origin))

    const urls = [
      'https://attacker.test/steal?token=SECRET#frag',
      'http://attacker.test:8080/x',
      'file:///etc/passwd',
      'javascript:alert(1)',
      'custom-proto://payload',
      ''
    ]

    for (const url of urls) {
      assert.deepEqual(handler({ url }), { action: 'deny' })
    }

    assert.equal(seen.length, urls.length)
    assert.equal(seen[0], 'https://attacker.test')
    assert.equal(seen[1], 'http://attacker.test:8080')
    assert.equal(describeDeniedUrl(''), '<unparseable>')
    assert.ok(seen.every(origin => !origin.includes('SECRET') && !origin.includes('/steal')))
  })

  test('a throwing observer still yields an explicit deny', () => {
    const handler = createWindowOpenHandler(() => {
      throw new Error('logging blew up')
    })

    assert.deepEqual(handler({ url: 'https://attacker.test/x' }), { action: 'deny' })
  })
})
