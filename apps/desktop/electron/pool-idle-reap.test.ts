/**
 * Tests for electron/pool-idle-reap.ts — the desktop backend pool's idle
 * reaper, gated on a backend-side busy signal (#108863).
 *
 * `lastActiveAt` only reflects renderer attention (chat WS open/streaming
 * keepalive); it says nothing about a backend running a cron job with no
 * window attached. Before this gate, the reaper killed such a backend mid-run
 * whenever it looked idle by that signal alone.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import { reapIdleBackends, selectIdleReapCandidates } from './pool-idle-reap'

const NOW = 1_000_000
const IDLE_MS = 10 * 60_000

const entry = (idleMs: number) => ({ lastActiveAt: NOW - idleMs })

test('selectIdleReapCandidates only picks entries past the idle threshold', () => {
  const entries: [string, ReturnType<typeof entry>][] = [
    ['idle', entry(IDLE_MS + 1_000)],
    ['fresh', entry(IDLE_MS - 1_000)]
  ]

  assert.deepEqual(selectIdleReapCandidates(entries, NOW, IDLE_MS), ['idle'])
})

test('#108863: an idle-by-lastActiveAt backend running a cron job is not stopped', async () => {
  const entries: [string, ReturnType<typeof entry>][] = [['default', entry(IDLE_MS + 1_000)]]
  const stopped: string[] = []

  const reaped = await reapIdleBackends(
    entries,
    NOW,
    IDLE_MS,
    async () => true, // the backend reports in-flight work (cron run)
    async key => {
      stopped.push(key)
    }
  )

  assert.deepEqual(stopped, [], 'a backend with in-flight work must not be stopped')
  assert.deepEqual(reaped, [])
})

test('a genuinely idle backend (no in-flight work) is still reaped', async () => {
  const entries: [string, ReturnType<typeof entry>][] = [['default', entry(IDLE_MS + 1_000)]]
  const stopped: string[] = []

  const reaped = await reapIdleBackends(
    entries,
    NOW,
    IDLE_MS,
    async () => false,
    async key => {
      stopped.push(key)
    }
  )

  assert.deepEqual(stopped, ['default'])
  assert.deepEqual(reaped, ['default'])
})

test('a fresh backend is never probed or reaped regardless of its busy signal', async () => {
  const entries: [string, ReturnType<typeof entry>][] = [['default', entry(1_000)]]
  let probed = false

  const reaped = await reapIdleBackends(
    entries,
    NOW,
    IDLE_MS,
    async () => {
      probed = true

      return true
    },
    async () => {
      throw new Error('must not stop a fresh backend')
    }
  )

  assert.equal(probed, false)
  assert.deepEqual(reaped, [])
})
