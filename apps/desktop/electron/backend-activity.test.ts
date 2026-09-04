/**
 * Tests for electron/backend-activity.ts — the "is this pooled backend doing
 * work?" gate in front of the idle reaper and the LRU cap eviction.
 *
 * Incident: the renderer's 60s keepalive stopped for a profile the user had
 * switched away from; ten minutes later the idle reaper SIGTERM'd a backend
 * that was running three agent turns plus an hour-long in-process
 * delegate_task subagent. The keepalive proves a renderer socket is open, not
 * that the backend is idle — so the reaper now asks the backend itself via
 * GET /api/activity and spares anything busy. Older runtimes 404 on that
 * route and wedged processes time out; both are UNKNOWN and keep the legacy
 * reap-on-idle behaviour so stale backends are still reclaimed.
 */

import assert from 'node:assert/strict'

import { test } from 'vitest'

import {
  decideIdleReap,
  DEFAULT_BUSY_GRACE_MS,
  formatSparedBusyBackendLog,
  parseBackendActivity,
  probeBackendActivity,
  withinBusyGrace
} from './backend-activity'

const NOW = 1_000_000
const IDLE_LIMIT_MS = 10 * 60_000

const busyBody = { ok: true, busy: true, running_turns: 3, active_subagents: 1, background_processes: 0 }
const idleBody = { ok: true, busy: false, running_turns: 0, active_subagents: 0, background_processes: 0 }

// ── parseBackendActivity ─────────────────────────────────────────────────────

test('parse accepts the documented payload and maps counters', () => {
  assert.deepEqual(parseBackendActivity(busyBody), {
    busy: true,
    runningTurns: 3,
    activeSubagents: 1,
    backgroundProcesses: 0
  })
  assert.deepEqual(parseBackendActivity(idleBody), {
    busy: false,
    runningTurns: 0,
    activeSubagents: 0,
    backgroundProcesses: 0
  })
})

test('parse rejects unrecognised shapes as unknown (null)', () => {
  const rejected: unknown[] = [
    null,
    undefined,
    'busy',
    42,
    [],
    {},
    { ok: true },
    { ok: false, busy: true, running_turns: 1, active_subagents: 0, background_processes: 0 },
    { ok: true, busy: 'yes', running_turns: 0, active_subagents: 0, background_processes: 0 },
    { ok: true, busy: false, running_turns: -1, active_subagents: 0, background_processes: 0 },
    { ok: true, busy: false, running_turns: '0', active_subagents: 0, background_processes: 0 },
    { ok: true, busy: false, running_turns: 0, active_subagents: 0 },
    // /api/status-like body from a backend that does not know the route.
    { ok: true, status: 'ready', version: '1.2.3' }
  ]

  for (const body of rejected) {
    assert.equal(parseBackendActivity(body), null, `expected null for ${JSON.stringify(body)}`)
  }
})

// ── decideIdleReap — decision table ──────────────────────────────────────────

const past = (overrides: Partial<Parameters<typeof decideIdleReap>[0]> = {}) => ({
  idleMs: IDLE_LIMIT_MS + 1,
  idleLimitMs: IDLE_LIMIT_MS,
  activity: null,
  busyGraceMs: DEFAULT_BUSY_GRACE_MS,
  now: NOW,
  lastBusyAt: null,
  ...overrides
})

test('idle keepalive + busy backend → keep, and hand back a busy stamp', () => {
  const decision = decideIdleReap(past({ activity: parseBackendActivity(busyBody) }))

  assert.equal(decision.reap, false)
  assert.equal(decision.reason, 'busy')
  assert.equal(decision.lastBusyAt, NOW)
})

test('idle keepalive + provably idle backend → reap', () => {
  const decision = decideIdleReap(past({ activity: parseBackendActivity(idleBody) }))

  assert.equal(decision.reap, true)
  assert.equal(decision.reason, 'idle')
})

test('idle keepalive + unknown activity (404 / timeout) → reap, legacy behaviour', () => {
  const decision = decideIdleReap(past({ activity: null }))

  assert.equal(decision.reap, true)
  assert.equal(decision.reason, 'unknown-activity')
})

test('seen busy within the grace window → keep even if this probe says idle or unknown', () => {
  const recentlyBusy = NOW - DEFAULT_BUSY_GRACE_MS / 2

  for (const activity of [null, parseBackendActivity(idleBody)]) {
    const decision = decideIdleReap(past({ activity, lastBusyAt: recentlyBusy }))

    assert.equal(decision.reap, false)
    assert.equal(decision.reason, 'busy-grace')
  }
})

test('a busy stamp older than the grace window no longer protects the backend', () => {
  const decision = decideIdleReap(past({ activity: null, lastBusyAt: NOW - DEFAULT_BUSY_GRACE_MS }))

  assert.equal(decision.reap, true)
})

test('within the idle window nothing is reaped, whatever the probe says', () => {
  for (const activity of [null, parseBackendActivity(idleBody), parseBackendActivity(busyBody)]) {
    const decision = decideIdleReap(past({ idleMs: IDLE_LIMIT_MS, activity }))

    assert.equal(decision.reap, false)
    assert.equal(decision.reason, 'keepalive-fresh')
  }
})

test('withinBusyGrace treats a missing stamp as never busy', () => {
  assert.equal(withinBusyGrace(null, NOW, DEFAULT_BUSY_GRACE_MS), false)
  assert.equal(withinBusyGrace(undefined, NOW, DEFAULT_BUSY_GRACE_MS), false)
  assert.equal(withinBusyGrace(NOW - 1, NOW, DEFAULT_BUSY_GRACE_MS), true)
})

// ── probeBackendActivity — fetch outcome mapping ─────────────────────────────

const spawnedEntry = { process: { pid: 4242 }, port: 51234, token: 'tok-secret' }

test('probe hits /api/activity on the entry port with its token and parses a good body', async () => {
  const calls: { url: string; token: string; timeoutMs: number }[] = []

  const fetch = async (url: string, token: string, options: { timeoutMs: number }) => {
    calls.push({ url, token, timeoutMs: options.timeoutMs })

    return busyBody
  }

  const activity = await probeBackendActivity(fetch, spawnedEntry, 3_000)

  assert.deepEqual(activity, parseBackendActivity(busyBody))
  assert.equal(calls.length, 1)
  assert.equal(calls[0].url, `http://127.0.0.1:${spawnedEntry.port}/api/activity`)
  assert.equal(calls[0].token, spawnedEntry.token)
  assert.equal(calls[0].timeoutMs, 3_000)
})

test('probe maps 404 (older runtime), timeout and network errors to unknown without throwing', async () => {
  const failures = [
    new Error('404: Not Found'),
    new Error('Timed out connecting to Hermes backend after 3000ms'),
    Object.assign(new Error('connect ECONNREFUSED 127.0.0.1:51234'), { code: 'ECONNREFUSED' }),
    new Error('Expected JSON from http://127.0.0.1:51234/api/activity but got HTML (status 200).')
  ]

  for (const failure of failures) {
    const fetch = async () => {
      throw failure
    }

    assert.equal(await probeBackendActivity(fetch, spawnedEntry, 3_000), null, failure.message)
  }
})

test('probe maps an unrecognised 2xx body to unknown', async () => {
  const fetch = async () => ({ ok: true, status: 'ready' })

  assert.equal(await probeBackendActivity(fetch, spawnedEntry, 3_000), null)
})

test('probe never dials a descriptor entry or a backend without a known port/token', async () => {
  let dialed = 0

  const fetch = async () => {
    dialed += 1

    return busyBody
  }

  assert.equal(await probeBackendActivity(fetch, { process: null, port: 51234, token: 'tok' }, 3_000), null)
  assert.equal(await probeBackendActivity(fetch, { process: { pid: 1 }, port: null, token: 'tok' }, 3_000), null)
  assert.equal(await probeBackendActivity(fetch, { process: { pid: 1 }, port: 51234, token: null }, 3_000), null)
  assert.equal(dialed, 0)
})

// ── desktop.log line ─────────────────────────────────────────────────────────

test('the spared-busy log line names the backend and carries every counter', () => {
  const line = formatSparedBusyBackendLog('conn:local::coder', parseBackendActivity(busyBody)!)

  assert.equal(
    line,
    'Sparing busy profile backend "conn:local::coder" (running_turns=3, active_subagents=1, background_processes=0) despite idle keepalive'
  )
})
