/**
 * Security regression for the OAuth/portal partition permission guard:
 * `installMediaPermissions` wires permission handlers ONLY on
 * `session.defaultSession` (media-capture-only allowlist). The auth
 * partitions (legacy shared + per-connection jars of #92183) got Chromium's
 * default behavior, in which remote IDP/portal content in the auth windows
 * can request and receive notifications, geolocation, midi, clipboard-read,
 * and more — permission prompts from a sign-in flow the user never
 * sanctioned. No permission is ever legitimate for completing sign-in, so
 * the guard denies everything, on both the request and check handlers.
 */

import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import { guardAuthSessionPermissions } from '../apps/desktop/electron/window-open-policy'

function makeFakePartitionSession() {
  const calls = {
    requestHandlers: [] as Array<(wc: unknown, perm: string, cb: (granted: boolean) => void, details: unknown) => void>,
    checkHandlers: [] as Array<(wc: unknown, perm: string) => boolean>,
    logs: [] as string[]
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

describe('auth partition permission guard', () => {
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

    // The deny log carries the partition label and the permission name.
    assert.ok(logs.length >= permissions.length)
    assert.ok(logs.every(line => line.startsWith('[auth-permission] oauth:persist:custom-1 denied: ')))
  })

  test('a partition without a check handler installs only the request handler', () => {
    // Not every Electron version surfaces setPermissionCheckHandler on
    // partitions; the guard must not throw when it is absent.
    const calls = { requestHandlers: [] as Array<(wc: unknown, perm: string, cb: (granted: boolean) => void) => void> }

    const partitionSession = {
      setPermissionRequestHandler(handler: (wc: unknown, perm: string, cb: (granted: boolean) => void) => void) {
        calls.requestHandlers.push(handler)
      }
    }

    assert.doesNotThrow(() => guardAuthSessionPermissions(partitionSession, 'oauth'))

    let granted: boolean | undefined
    calls.requestHandlers[0](null, 'geolocation', (value: boolean) => {
      granted = value
    })
    assert.equal(granted, false)
  })

  test('a throwing session emitter is a no-op', () => {
    // Degraded environments: a destroyed session emitter must not take the
    // whole sign-in flow down with it.
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
