/**
 * Tests for electron/update-rebuild.ts — the retry-once policy for the desktop
 * `--build-only` rebuild during self-update.
 *
 * Run with: node --test electron/update-rebuild.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * Why this matters: a first rebuild can return nonzero on a still-settling tree
 * or a self-healed (network-blocked) Electron download. Without a second attempt
 * the updater bails before the relaunch step — the app updates but never restarts
 * (the field report behind this fix). The retry must fire on failure, not on
 * success, and must run at most twice.
 */

import assert from 'node:assert/strict'
import test from 'node:test'

import { runRebuildWithRetry, shouldRetryRebuild } from './update-rebuild.ts'

test('shouldRetryRebuild retries only on a non-success exit', () => {
  assert.equal(shouldRetryRebuild(0), false)
  assert.equal(shouldRetryRebuild(1), true)
  assert.equal(shouldRetryRebuild(null), true)
})

test('a clean first rebuild runs once and does not retry', async () => {
  const codes = []

  const result = await runRebuildWithRetry(attempt => {
    codes.push(attempt)

    return Promise.resolve({ code: 0 })
  })

  assert.deepEqual(codes, [0])
  assert.equal(result.code, 0)
})

test('a failed first rebuild retries once and succeeds', async () => {
  const codes = []

  const result = await runRebuildWithRetry(attempt => {
    codes.push(attempt)

    return Promise.resolve({ code: attempt === 0 ? 1 : 0 })
  })

  assert.deepEqual(codes, [0, 1])
  assert.equal(result.code, 0)
})

test('a rebuild that keeps failing runs at most twice and reports the failure', async () => {
  const codes = []

  const result = await runRebuildWithRetry(attempt => {
    codes.push(attempt)

    return Promise.resolve({ code: 1, error: 'rebuild-failed' })
  })

  assert.deepEqual(codes, [0, 1])
  assert.equal(result.code, 1)
  assert.equal(result.error, 'rebuild-failed')
})
