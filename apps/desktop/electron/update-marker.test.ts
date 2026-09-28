/**
 * Tests for electron/update-marker.ts — the in-app update mutual-exclusion
 * marker that prevents a desktop relaunched mid-update from spawning a backend
 * the updater then kills in a loop (#50238).
 *
 * Run with: node --test electron/update-marker.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * Why this matters: the gate must (a) report a live update only when the
 * updater pid is alive AND the marker is fresh, (b) treat absent/malformed/
 * dead-pid/expired markers as "no live update" so a crashed updater can't
 * strand future launches, and (c) self-heal by deleting a stale marker file.
 */

import fs from 'fs'
import assert from 'node:assert/strict'
import os from 'os'
import path from 'path'

import { test } from 'vitest'

import {
  cachedProcessStartMs,
  isPidAlive,
  markerOwnerIsLive,
  markerPath,
  parsePsEtime,
  PID_REUSE_TOLERANCE_MS,
  posixProcessState,
  probeProcessStartMs,
  readLiveUpdateMarker,
  UPDATE_MARKER_MAX_AGE_MS,
  updateHandoffConflict,
  writeUpdateMarker
} from './update-marker'

function tmpHome(tag) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `hermes-marker-${tag}-`))

  return dir
}

function writeMarker(home, pid, startedAtSec) {
  fs.writeFileSync(markerPath(home), `${pid}\n${startedAtSec}`)
}

const ALIVE: typeof process.kill = () => true // injected kill that "succeeds" => pid alive

const DEAD: typeof process.kill = () => {
  const err = new Error('no such process')

  ;(err as any).code = 'ESRCH'
  throw err
}

test('absent marker => no live update', () => {
  const home = tmpHome('absent')
  assert.equal(readLiveUpdateMarker(home, { kill: ALIVE }), null)
})

test('live pid within age ceiling => live update reported', () => {
  const home = tmpHome('live')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor(now / 1000) - 5) // 5s old
  const res = readLiveUpdateMarker(home, { kill: ALIVE, now: () => now })
  assert.ok(res, 'a fresh, alive marker is a live update')
  assert.equal(res.pid, 4242)
  assert.ok(res.ageMs >= 0 && res.ageMs < 10_000)
  assert.ok(fs.existsSync(markerPath(home)), 'a live marker is NOT deleted')
})

test('dead pid => no live update and marker is pruned', () => {
  const home = tmpHome('dead')
  writeMarker(home, 999999, Math.floor(Date.now() / 1000))
  assert.equal(readLiveUpdateMarker(home, { kill: DEAD }), null)
  assert.ok(!fs.existsSync(markerPath(home)), 'a dead-pid marker self-heals (deleted)')
})

test('zombie pid => no live update and marker is pruned', () => {
  // The kill(pid, 0) false positive: a process that exited but is still in the
  // table (parent has not reaped it) answers signal 0 like a live one. The
  // state probe must turn that into "dead" so the boot gate self-heals in
  // seconds instead of parking for the whole 20-minute ceiling.
  const home = tmpHome('zombie')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor(now / 1000) - 5)
  const res = readLiveUpdateMarker(home, { kill: ALIVE, now: () => now, processState: () => 'Z' })
  assert.equal(res, null, 'a zombie owner is not a live update')
  assert.ok(!fs.existsSync(markerPath(home)), 'a zombie-owned marker self-heals (deleted)')
})

test('a live state keeps the marker (probe answers non-Z)', () => {
  const home = tmpHome('state-live')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor(now / 1000) - 5)
  const res = readLiveUpdateMarker(home, { kill: ALIVE, now: () => now, processState: () => 'S' })
  assert.ok(res, 'an alive, non-zombie owner keeps the gate closed')
  assert.ok(fs.existsSync(markerPath(home)), 'a live marker is NOT deleted')
})

test('an unknown process state fails open to alive (keeps the marker)', () => {
  const home = tmpHome('state-unknown')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor(now / 1000) - 5)
  const res = readLiveUpdateMarker(home, { kill: ALIVE, now: () => now, processState: () => null })
  assert.ok(res, 'probe failure must keep the conservative signal-0 verdict')
  assert.ok(fs.existsSync(markerPath(home)))
})

// Owner creation times relative to the marker file's real mtime.
const UNVERIFIABLE = () => null
const STARTED_BEFORE_WRITE = () => Date.now() - 60_000
const STARTED_AFTER_WRITE = () => Date.now() + 60_000

test('expired marker with an unverifiable owner => no live update and pruned', () => {
  const home = tmpHome('expired')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor((now - UPDATE_MARKER_MAX_AGE_MS - 60_000) / 1000))
  // The pid is "alive" but its identity can't be checked: the age ceiling applies.
  assert.equal(readLiveUpdateMarker(home, { kill: ALIVE, now: () => now, ownerStartedMs: UNVERIFIABLE }), null)
  assert.ok(!fs.existsSync(markerPath(home)), 'an expired marker self-heals (deleted)')
})

test('#109795: a verified live owner past the age ceiling is still a live update', () => {
  const home = tmpHome('slow-update')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor((now - UPDATE_MARKER_MAX_AGE_MS - 60_000) / 1000))
  const res = readLiveUpdateMarker(home, { kill: ALIVE, now: () => now, ownerStartedMs: STARTED_BEFORE_WRITE })
  assert.ok(res, 'a slow update keeps its lock while its owner is alive')
  assert.equal(res.pid, 4242)
  assert.ok(fs.existsSync(markerPath(home)), 'a live marker is NOT deleted')
})

test('a recycled pid (created after the marker write) past the ceiling => pruned', () => {
  const home = tmpHome('recycled')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor((now - UPDATE_MARKER_MAX_AGE_MS - 60_000) / 1000))
  assert.equal(readLiveUpdateMarker(home, { kill: ALIVE, now: () => now, ownerStartedMs: STARTED_AFTER_WRITE }), null)
  assert.ok(!fs.existsSync(markerPath(home)))
})

test('a pending identity probe keeps the marker live (never guess a second updater in)', () => {
  const home = tmpHome('pending')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor((now - UPDATE_MARKER_MAX_AGE_MS - 60_000) / 1000))
  assert.ok(readLiveUpdateMarker(home, { kill: ALIVE, now: () => now, ownerStartedMs: () => undefined }))
  assert.ok(fs.existsSync(markerPath(home)))
})

test('inside the ceiling the identity probe is never consulted', () => {
  const home = tmpHome('young-no-probe')
  const now = 1_000_000_000_000
  let probed = false
  writeMarker(home, 4242, Math.floor(now / 1000) - 5)

  const res = readLiveUpdateMarker(home, {
    kill: ALIVE,
    now: () => now,
    ownerStartedMs: () => {
      probed = true

      return null
    }
  })

  assert.ok(res)
  assert.equal(probed, false)
})

test('markerOwnerIsLive decision table', () => {
  const past = UPDATE_MARKER_MAX_AGE_MS + 1
  const w = 1_000_000

  const cases: [Parameters<typeof markerOwnerIsLive>[0], boolean][] = [
    [{ pidAlive: false, ageMs: 5, ownerStartedMs: null, markerWrittenMs: w }, false],
    [{ pidAlive: true, ageMs: 5, ownerStartedMs: null, markerWrittenMs: w }, true],
    [{ pidAlive: true, ageMs: past, ownerStartedMs: null, markerWrittenMs: w }, false],
    [{ pidAlive: true, ageMs: past, ownerStartedMs: w - 1, markerWrittenMs: null }, false],
    [{ pidAlive: true, ageMs: past, ownerStartedMs: undefined, markerWrittenMs: w }, true],
    [{ pidAlive: true, ageMs: past, ownerStartedMs: w - 1, markerWrittenMs: w }, true],
    [{ pidAlive: true, ageMs: past, ownerStartedMs: w + PID_REUSE_TOLERANCE_MS, markerWrittenMs: w }, true],
    [{ pidAlive: true, ageMs: past, ownerStartedMs: w + PID_REUSE_TOLERANCE_MS + 1, markerWrittenMs: w }, false],
    [{ pidAlive: false, ageMs: past, ownerStartedMs: w - 1, markerWrittenMs: w }, false]
  ]

  for (const [input, live] of cases) {
    assert.equal(markerOwnerIsLive(input), live, JSON.stringify(input))
  }
})

test('parsePsEtime handles the macOS/procps etime formats', () => {
  assert.equal(parsePsEtime('05:07'), 307)
  assert.equal(parsePsEtime('  01:02:03\n'), 3723)
  assert.equal(parsePsEtime('2-00:00:01'), 172_801)
  assert.equal(parsePsEtime(''), null)
  assert.equal(parsePsEtime('abc'), null)
  assert.equal(parsePsEtime('1:2:3:4'), null)
  assert.equal(parsePsEtime('x-01:02'), null)
})

test('cachedProcessStartMs is pending first, then serves the probe result', async () => {
  const pid = 7_000_001

  let resolveProbe: (v: number) => void = () => {}
  const probe = () => new Promise<number>(r => (resolveProbe = r))
  assert.equal(cachedProcessStartMs(pid, { probe }), undefined, 'pending while the probe is in flight')
  assert.equal(cachedProcessStartMs(pid, { probe }), undefined)
  resolveProbe(1234)
  await new Promise(r => setTimeout(r, 0))
  assert.equal(cachedProcessStartMs(pid, { probe }), 1234)
})

test('cachedProcessStartMs maps a failed probe to unverifiable (null)', async () => {
  const pid = 7_000_002
  const probe = () => Promise.reject(new Error('Get-Process denied'))
  assert.equal(cachedProcessStartMs(pid, { probe }), undefined)
  await new Promise(r => setTimeout(r, 0))
  assert.equal(cachedProcessStartMs(pid, { probe }), null)
})

test.skipIf(process.platform === 'win32')('probeProcessStartMs reads this process creation time', async () => {
  const expected = Date.now() - process.uptime() * 1000
  const started = await probeProcessStartMs(process.pid)
  assert.ok(Math.abs(started - expected) <= PID_REUSE_TOLERANCE_MS, `${started} vs ${expected}`)
})

test('malformed marker => no live update and pruned', () => {
  const home = tmpHome('malformed')
  fs.writeFileSync(markerPath(home), 'not-a-pid\nnonsense')
  assert.equal(readLiveUpdateMarker(home, { kill: ALIVE }), null)
  assert.ok(!fs.existsSync(markerPath(home)))
})

test('isPidAlive: own pid is alive, impossible pid is dead', () => {
  assert.equal(isPidAlive(process.pid), true)
  assert.equal(isPidAlive(-1), false)
  assert.equal(isPidAlive(0), false)
  assert.equal(isPidAlive(NaN), false)
})

test('isPidAlive: EPERM counts as alive (process owned by another user)', () => {
  const eperm = () => {
    const err = new Error('operation not permitted')

    ;(err as any).code = 'EPERM'
    throw err
  }

  assert.equal(isPidAlive(4242, eperm), true)
})

test('posixProcessState: own pid is probeable and not a zombie; dead pid is unknown', () => {
  if (process.platform === 'win32') {
    // Windows has no zombie state and no ps stat lane; the probe is a no-op.
    assert.equal(posixProcessState(process.pid), null)

    return
  }

  const own = posixProcessState(process.pid)
  assert.ok(own, 'a live pid must be probeable on linux/darwin')
  assert.ok(!own.toUpperCase().startsWith('Z'), 'this process is not a zombie')

  // A pid nothing owns (and that kill(0) would reject) is simply unknowable —
  // callers keep their signal-0 verdict in that case.
  assert.equal(posixProcessState(2147483647), null)
})

test('writeUpdateMarker writes a marker that readLiveUpdateMarker accepts', () => {
  const home = tmpHome('write')
  const now = 1_000_000_000_000
  writeUpdateMarker(home, 4242, { now: () => now })
  // The marker should be readable and report the same pid.
  const res = readLiveUpdateMarker(home, { kill: ALIVE, now: () => now })
  assert.ok(res, 'marker written by writeUpdateMarker should be detected as live')
  assert.equal(res.pid, 4242)
  assert.ok(fs.existsSync(markerPath(home)), 'marker file should exist after write')
})

test('writeUpdateMarker preserves a live holder age across pid hand-off', () => {
  const home = tmpHome('write-handoff-age')
  const now = 1_000_000_000_000
  const startedAt = Math.floor(now / 1000) - 300

  writeMarker(home, 1010, startedAt)
  writeUpdateMarker(home, 2020, { kill: ALIVE, now: () => now })

  const [pidLine, startedLine] = fs.readFileSync(markerPath(home), 'utf8').split('\n')
  assert.equal(Number.parseInt(pidLine, 10), 2020, 'the hand-off records the new owner')
  assert.equal(Number.parseInt(startedLine, 10), startedAt, 'the holder age must not restart during hand-off')
})

test('writeUpdateMarker uses the acquisition time passed to a detached script', () => {
  const home = tmpHome('write-script-acquired-at')
  const now = 1_000_000_000_000
  const startedAt = Math.floor(now / 1000) - 300

  writeUpdateMarker(home, 2020, { now: () => now, startedAt })

  const [, startedLine] = fs.readFileSync(markerPath(home), 'utf8').split('\n')
  assert.equal(Number.parseInt(startedLine, 10), startedAt)
})

test('writeUpdateMarker is best-effort (no throw on bad path)', () => {
  // A non-existent directory should not throw.
  const badHome = path.join(os.tmpdir(), 'hermes-marker-nonexistent-' + Date.now())
  assert.doesNotThrow(() => writeUpdateMarker(badHome, 4242))
})

test('writeUpdateMarker + dead pid => self-heals on read', () => {
  const home = tmpHome('write-dead')
  writeUpdateMarker(home, 999999, { now: () => Date.now() })
  // PID 999999 is almost certainly not alive.
  const res = readLiveUpdateMarker(home, { kill: DEAD })
  assert.equal(res, null, 'a dead-pid marker from writeUpdateMarker self-heals')
  assert.ok(!fs.existsSync(markerPath(home)), 'marker file is pruned')
})

// ---------------------------------------------------------------------------
// updateHandoffConflict (#75778)
//
// A retried "Update" click must not spawn a second updater over a still-live
// one — writeUpdateMarker unconditionally overwrites the marker, so an
// unchecked hand-off clobbers the original updater's claim while it is still
// alive and mutating the checkout.
// ---------------------------------------------------------------------------

test('no marker => hand-off is not blocked', () => {
  const home = tmpHome('conflict-none')
  assert.equal(updateHandoffConflict(home, { kill: ALIVE }), null)
})

test('a different live updater already owns the marker => hand-off is blocked', () => {
  const home = tmpHome('conflict-live')
  const now = 1_000_000_000_000
  writeMarker(home, 1010, Math.floor(now / 1000) - 6) // 6s old
  const conflict = updateHandoffConflict(home, { kill: ALIVE, now: () => now })
  assert.ok(conflict, 'a live foreign updater must block a new hand-off')
  assert.equal(conflict.pid, 1010)
  assert.match(conflict.message, /already running/)
  assert.match(conflict.message, /PID 1010/)
  assert.match(conflict.message, /6s/)
})

test('a dead-pid marker does not block a hand-off (self-heals)', () => {
  const home = tmpHome('conflict-dead')
  writeMarker(home, 999999, Math.floor(Date.now() / 1000))
  assert.equal(updateHandoffConflict(home, { kill: DEAD }), null)
})

test('an expired marker with an unverifiable owner does not block a hand-off (self-heals)', () => {
  const home = tmpHome('conflict-expired')
  const now = 1_000_000_000_000
  writeMarker(home, 1010, Math.floor((now - UPDATE_MARKER_MAX_AGE_MS - 60_000) / 1000))
  assert.equal(updateHandoffConflict(home, { kill: ALIVE, now: () => now, ownerStartedMs: UNVERIFIABLE }), null)
})

test('#109795: a slow but live updater past the ceiling still blocks a second hand-off', () => {
  const home = tmpHome('conflict-slow')
  const now = 1_000_000_000_000
  writeMarker(home, 1010, Math.floor((now - UPDATE_MARKER_MAX_AGE_MS - 60_000) / 1000))
  const conflict = updateHandoffConflict(home, { kill: ALIVE, now: () => now, ownerStartedMs: STARTED_BEFORE_WRITE })
  assert.ok(conflict, 'age alone must not admit a second updater')
  assert.equal(conflict.pid, 1010)
})
