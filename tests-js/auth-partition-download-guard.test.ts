/**
 * Security regression for the OAuth/portal partition download guard:
 * the auth windows (OAuth login, portal sign-in, silent portal renewal)
 * load REMOTE content, and their partitions have NO will-download handler
 * (`installDownloadHandling` wires only `session.defaultSession`). A
 * Content-Disposition: attachment response on a redirect hop — or
 * download-triggering JS on a compromised IDP/portal page — would reach
 * Chromium's raw save dialog (process cwd default directory,
 * extensionless attacker-chosen filename). The auth flows exist solely to
 * complete sign-in, so the guard cancels every download outright, the same
 * exposure the link-title partition closes via `guardLinkTitleSession`.
 */

import assert from 'node:assert/strict'

import { describe, test } from 'vitest'

import { guardAuthSessionDownloads } from '../apps/desktop/electron/window-open-policy'

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
    // Degraded/headless environments: a destroyed session emitter must not
    // take the whole sign-in flow down with it.
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
