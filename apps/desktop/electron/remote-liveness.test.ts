/**
 * Regression tests for remote dashboard liveness probe behavior.
 *
 * The desktop's `hermes:connection:revalidate` IPC handler probes
 * `/api/status` on a cached remote connection. Before the fix it used a
 * flat 2.5s timeout and treated every failure as "remote is dead",
 * dropping the cached connection. On a busy single-uvicorn-worker remote
 * (sidebar refresh, SQLite I/O) that 2.5s window is easy to exceed,
 * triggering a reconnect storm.
 *
 * The fix introduces two changes that these tests guard:
 * 1. The probe timeout is raised to 60s for remote connections.
 * 2. Transient timeout errors (ETIMEDOUT, ESOCKETTIMEDOUT) keep the
 *    cached connection alive; only non-transient failures (connection
 *    refused, DNS, etc.) drop and rebuild.
 *
 * Run with: npx vitest run electron/remote-liveness.test.ts
 */

import assert from 'node:assert/strict'
import { test } from 'vitest'

import { isTransientRemoteProbeError } from './hardening'

test('ETIMEDOUT is classified as transient', () => {
  assert.ok(isTransientRemoteProbeError(new Error('ETIMEDOUT')))
})

test('ESOCKETTIMEDOUT is classified as transient', () => {
  assert.ok(isTransientRemoteProbeError(new Error('ESOCKETTIMEDOUT')))
})

test('"timed out connecting to Hermes backend" is classified as transient', () => {
  assert.ok(isTransientRemoteProbeError(new Error('timed out connecting to Hermes backend')))
})

test('ECONNREFUSED is NOT classified as transient', () => {
  assert.ok(!isTransientRemoteProbeError(new Error('ECONNREFUSED')))
})

test('DNS ENOTFOUND is NOT classified as transient', () => {
  assert.ok(!isTransientRemoteProbeError(new Error('ENOTFOUND')))
})

test('Generic HTTP 500 is NOT classified as transient', () => {
  assert.ok(!isTransientRemoteProbeError(new Error('Internal Server Error')))
})

test('undefined/null/empty are NOT classified as transient', () => {
  assert.ok(!isTransientRemoteProbeError(undefined))
  assert.ok(!isTransientRemoteProbeError(null))
  assert.ok(!isTransientRemoteProbeError(''))
})
