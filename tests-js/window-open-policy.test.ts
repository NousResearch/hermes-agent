/**
 * Security regression for GHSA-9f4c-93c8-jc8g (CVE-2026-70608, High 7.2):
 * "Sandboxed iframe can bypass the allow-popups restriction via the OpenURL
 * navigation path."
 *
 * A sandboxed iframe without `allow-popups` can reach a window's
 * `setWindowOpenHandler` with no user gesture. The desktop app renders
 * untrusted artifact HTML in `<iframe sandbox="allow-scripts">`, so if the
 * handler opens `details.url` as a side effect, a malicious artifact can force
 * the OS browser to an attacker URL. Electron has no fixed 40.x release, so the
 * defense lives in our handler: it must ALWAYS deny and must NEVER open a URL.
 *
 * These import the real policy module (pure, dependency-free) so the contract is
 * tested against the code main.ts actually wires in — a future edit that
 * reintroduces conditional opening has to defeat these tests.
 */

import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import {
  createWindowOpenHandler,
  decideWindowOpen
} from '../apps/desktop/electron/window-open-policy'

describe('window-open policy (GHSA-9f4c-93c8-jc8g)', () => {
  test('decideWindowOpen always denies, regardless of URL', () => {
    for (const url of [
      'https://example.com',
      'http://attacker.test/steal',
      'file:///etc/passwd',
      'javascript:alert(1)',
      'custom-proto://payload',
      ''
    ]) {
      assert.deepEqual(decideWindowOpen({ url }), { action: 'deny' })
    }
  })

  test('handler denies and never returns an allow action', () => {
    const handler = createWindowOpenHandler()
    const result = handler({ url: 'https://attacker.test/popup' })
    assert.equal(result.action, 'deny')
  })

  test('handler surfaces the denied URL to the observability hook only', () => {
    const denied: string[] = []
    const handler = createWindowOpenHandler(url => denied.push(url))

    const result = handler({ url: 'https://attacker.test/x' })

    assert.equal(result.action, 'deny')
    assert.deepEqual(denied, ['https://attacker.test/x'])
  })

  test('a throwing observability hook does not turn a deny into an allow', () => {
    const handler = createWindowOpenHandler(() => {
      throw new Error('logging blew up')
    })
    // The hook is logging-only; even if it throws, the security decision must
    // not silently become "allow". We assert the throw propagates rather than
    // being swallowed into an open — the caller (Electron) treats a throw as
    // deny, and crucially no URL was opened as a side effect.
    assert.throws(() => handler({ url: 'https://attacker.test/x' }))
  })
})
