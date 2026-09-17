/**
 * Security regression for the auth-flow window hardening — the GHSA-9f4c-93c8-jc8g
 * class on the OAuth/portal windows, plus the partition-level download and
 * permission guards. The auth windows (OAuth gateway login, portal sign-in,
 * silent portal renewal) load REMOTE content:
 *
 *   - the OAuth redirect chain navigates third-party IDP pages we do not initiate
 *   - the portal pages are fetched over the network
 *
 * so content-driven `window.open`, downloads, and permission requests must all
 * be denied — never opened / never granted as side effects. These tests import
 * the REAL policy module main.ts wires the windows and partitions with.
 */

import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import {
  guardAuthSessionDownloads,
  guardAuthSessionPermissions,
  wireAuthWindowOpenPolicy
} from '../apps/desktop/electron/window-open-policy'

function makeFakeAuthWindow() {
  const calls = {
    windowOpenHandlers: [] as Array<(details: { url: string }) => { action: string }>,
    logs: [] as string[]
  }

  const win = {
    webContents: {
      setWindowOpenHandler(handler: (details: { url: string }) => { action: string }) {
        calls.windowOpenHandlers.push(handler)
      }
    }
  }

  return { win, calls }
}

describe('auth window-open policy (GHSA-9f4c-93c8-jc8g class)', () => {
  test('every auth flow installs exactly one always-deny handler', () => {
    for (const label of ['oauth', 'portal', 'portal-renew']) {
      const { win, calls } = makeFakeAuthWindow()
      const logs: string[] = []

      wireAuthWindowOpenPolicy(win, label, (line: string) => logs.push(line))

      assert.equal(calls.windowOpenHandlers.length, 1)

      const handler = calls.windowOpenHandlers[0]
      assert.deepEqual(handler({ url: 'https://idp.attacker.test/login?next=steal' }), { action: 'deny' })
      assert.deepEqual(handler({ url: 'file:///etc/passwd' }), { action: 'deny' })
      assert.deepEqual(handler({ url: 'javascript:alert(1)' }), { action: 'deny' })

      // The deny log names the flow and carries origin only — a signed URL
      // or query token must never reach the persisted desktop log.
      assert.ok(logs.length >= 1)
      assert.ok(logs.every(line => line.startsWith(`[window-open] ${label} denied: `)))
      assert.ok(logs.every(line => !line.includes('next=steal')))
    }
  })

  test('a missing or throwing logger never degrades the deny', () => {
    const silent = makeFakeAuthWindow()
    wireAuthWindowOpenPolicy(silent.win, 'oauth')
    assert.deepEqual(silent.calls.windowOpenHandlers[0]({ url: 'https://x.test/' }), { action: 'deny' })

    const throwing = makeFakeAuthWindow()
    wireAuthWindowOpenPolicy(throwing.win, 'portal', () => {
      throw new Error('logging blew up')
    })
    assert.deepEqual(throwing.calls.windowOpenHandlers[0]({ url: 'https://x.test/' }), { action: 'deny' })
  })
})

describe('auth partition download guard', () => {
  test('cancels every will-download item and labels the log per partition', () => {
    const handlers: Record<string, (event: unknown, item: { cancel: () => void }) => void> = {}
    const logs: string[] = []

    const partitionSession = {
      on(event: string, handler: (event: unknown, item: { cancel: () => void }) => void) {
        handlers[event] = handler
      }
    }

    guardAuthSessionDownloads(partitionSession, 'oauth:persist:custom-1', (line: string) => logs.push(line))

    let cancelled = 0

    handlers['will-download'](null, {
      cancel: () => {
        cancelled += 1
      }
    })
    handlers['will-download'](null, {
      cancel: () => {
        cancelled += 1
      }
    })

    assert.equal(cancelled, 2)
    assert.deepEqual(logs, [
      '[auth-download] oauth:persist:custom-1 cancelled',
      '[auth-download] oauth:persist:custom-1 cancelled'
    ])
    // The guard installs exactly one will-download handler — no accumulation
    // when a partition session is reused across sign-in attempts.
    assert.equal(Object.keys(handlers).filter(k => k === 'will-download').length, 1)
  })

  test('is a no-op when session.on throws', () => {
    assert.doesNotThrow(() =>
      guardAuthSessionDownloads(
        {
          on() {
            throw new Error('Object has been destroyed')
          }
        },
        'oauth'
      )
    )
  })
})

describe('auth partition permission guard', () => {
  function makeFakePartitionSession() {
    const calls = {
      requestHandlers: [] as Array<(wc: unknown, perm: string, cb: (granted: boolean) => void, details: unknown) => void>,
      checkHandlers: [] as Array<(wc: unknown, perm: string) => boolean>
    }

    const partitionSession = {
      setPermissionRequestHandler(handler: (wc: unknown, perm: string, cb: (granted: boolean) => void, details: unknown) => void) {
        calls.requestHandlers.push(handler)
      },
      setPermissionCheckHandler(handler: (wc: unknown, perm: string) => boolean) {
        calls.checkHandlers.push(handler)
      }
    }

    return { partitionSession, calls }
  }

  test('denies every permission on both the request and the check handler', () => {
    const { partitionSession, calls } = makeFakePartitionSession()
    const logs: string[] = []

    guardAuthSessionPermissions(partitionSession, 'oauth:persist:custom-1', (line: string) => logs.push(line))

    assert.equal(calls.requestHandlers.length, 1)
    assert.equal(calls.checkHandlers.length, 1)

    const request = calls.requestHandlers[0]
    const check = calls.checkHandlers[0]

    // Every permission a compromised IDP/portal page might request is denied
    // through the request handler — including the ones the default session
    // deliberately allows for voice conversations (media).
    const permissions = [
      'notifications',
      'geolocation',
      'midi',
      'midiSysex',
      'clipboard-read',
      'media',
      'fullscreen',
      'pointerLock',
      'openExternal'
    ]

    for (const permission of permissions) {
      let granted: boolean | undefined
      request(null, permission, (value: boolean) => {
        granted = value
      }, {})
      assert.equal(granted, false, `request handler must deny ${permission}`)
      assert.equal(check(null, permission), false, `check handler must deny ${permission}`)
    }

    assert.ok(logs.length >= permissions.length)
    assert.ok(logs.every(line => line.startsWith('[auth-permission] oauth:persist:custom-1 denied: ')))
  })

  test('a partition without a check handler installs only the request handler', () => {
    const calls = {
      requestHandlers: [] as Array<(wc: unknown, perm: string, cb: (granted: boolean) => void, details: unknown) => void>
    }

    const partitionSession = {
      setPermissionRequestHandler(
        handler: (wc: unknown, perm: string, cb: (granted: boolean) => void, details: unknown) => void
      ) {
        calls.requestHandlers.push(handler)
      }
    }

    assert.doesNotThrow(() => guardAuthSessionPermissions(partitionSession, 'oauth'))

    let granted: boolean | undefined
    calls.requestHandlers[0](null, 'geolocation', (value: boolean) => {
      granted = value
    }, {})
    assert.equal(granted, false)
  })

  test('a throwing session emitter is a no-op', () => {
    assert.doesNotThrow(() =>
      guardAuthSessionPermissions(
        {
          setPermissionRequestHandler() {
            throw new Error('Object has been destroyed')
          }
        },
        'oauth'
      )
    )
  })
})
