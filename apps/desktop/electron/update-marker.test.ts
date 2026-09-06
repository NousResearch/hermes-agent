/**
 * Tests for electron/update-marker.ts — the in-app update mutual-exclusion
 * marker that prevents a desktop relaunched mid-update from spawning a backend
 * the updater then kills in a loop (#50238).
 *
 * Run with: node --test electron/update-marker.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * Why this matters: the gate must (a) report a live update only when the
 * updater pid identity is proven alive, (b) keep malformed/unreadable and
 * overdue claims blocking, and (c) self-heal only a proven-dead exact claim.
 */

import fs from 'fs'
import assert from 'node:assert/strict'
import os from 'os'
import path from 'path'

import { test, vi } from 'vitest'

import {
  acquireUpdateMarker,
  isPidAlive,
  markerPath,
  probePidIdentity,
  readLiveUpdateMarker,
  releaseUpdateMarkerIfOwnedBy,
  UPDATE_HANDOFF_BRIDGE_GRACE_MS,
  UPDATE_MARKER_MAX_AGE_MS,
  updateHandoffConflict,
  writeUpdateMarker
} from './update-marker'

function tmpHome(tag) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `hermes-marker-${tag}-`))

  return dir
}

function writeMarker(home, pid, startedAtSec, kind = '') {
  const kindLine = kind ? `\n${kind}\n` : ''

  fs.writeFileSync(markerPath(home), `${pid}\n${startedAtSec}${kindLine}`)
}

const ALIVE: typeof process.kill = () => true // injected kill that "succeeds" => pid alive

const DEAD: typeof process.kill = () => {
  const err = new Error('no such process')

  ;(err as any).code = 'ESRCH'
  throw err
}

test('a concurrent reader leaves an active marker release alone', () => {
  const home = tmpHome('release-reader')
  const startedAt = Math.floor(Date.now() / 1000)
  writeMarker(home, process.pid, startedAt)
  const rename = fs.renameSync.bind(fs)
  let observed: ReturnType<typeof readLiveUpdateMarker> | undefined

  const hook = vi.spyOn(fs, 'renameSync').mockImplementation((from, to) => {
    rename(from, to)
    observed = readLiveUpdateMarker(home, { kill: ALIVE })
  })

  try {
    assert.equal(releaseUpdateMarkerIfOwnedBy(home, process.pid, startedAt), true)
    assert.equal(observed?.kind, 'unreadable')
    assert.equal(fs.existsSync(markerPath(home)), false)
    assert.deepEqual(fs.readdirSync(home), [])
  } finally {
    hook.mockRestore()
  }
})

test('an abandoned release is recovered only after its releaser is proven dead', () => {
  const home = tmpHome('abandoned-release')
  const owner = 4242
  const releaser = 4343
  const startedAt = Math.floor(Date.now() / 1000)
  const tombstone = `${markerPath(home)}.cas-release-${releaser}-abandoned`
  fs.writeFileSync(tombstone, `${owner}\n${startedAt}\n`)
  assert.equal(readLiveUpdateMarker(home, { kill: ALIVE })?.kind, 'unreadable')
  assert.equal(fs.existsSync(tombstone), true)
  const probe: typeof process.kill = (pid, signal) => pid === releaser ? DEAD(pid, signal) : ALIVE(pid, signal)
  assert.equal(readLiveUpdateMarker(home, { kill: probe })?.pid, owner)
  assert.equal(fs.existsSync(tombstone), false)
  assert.equal(fs.existsSync(markerPath(home)), true)
})

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

test('expired marker remains live but is marked overdue', () => {
  const home = tmpHome('expired')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor((now - UPDATE_MARKER_MAX_AGE_MS - 60_000) / 1000))
  const result = readLiveUpdateMarker(home, { kill: ALIVE, now: () => now })
  assert.equal(result?.kind, 'live')
  assert.equal(result?.overdue, true)
  assert.ok(fs.existsSync(markerPath(home)), 'an overdue live owner remains authoritative')
})

test('malformed marker is unreadable and remains blocking', () => {
  const home = tmpHome('malformed')
  fs.writeFileSync(markerPath(home), 'not-a-pid\nnonsense')
  assert.equal(readLiveUpdateMarker(home, { kill: ALIVE })?.kind, 'unreadable')
  assert.ok(fs.existsSync(markerPath(home)))
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

test('dead tagged hand-off bridge covers the wrapper-to-script claim gap', () => {
  const home = tmpHome('dead-handoff-bridge')
  const now = 1_000_000_000_000

  writeMarker(home, 999999, Math.floor(now / 1000) - 8, 'handoff-bridge')

  const res = readLiveUpdateMarker(home, { kill: DEAD, now: () => now })
  assert.ok(res, 'the bridge must keep the backend gate closed until PowerShell claims the marker')
  assert.ok(fs.existsSync(markerPath(home)), 'the bridge remains during the bounded claim gap')
})

test('tagged hand-off bridge expires after the claim gap even while its pid is alive', () => {
  const home = tmpHome('expired-handoff-bridge')
  const now = 1_000_000_000_000

  writeMarker(
    home,
    4242,
    Math.floor((now - UPDATE_HANDOFF_BRIDGE_GRACE_MS - 1000) / 1000),
    'handoff-bridge'
  )

  assert.equal(readLiveUpdateMarker(home, { kill: ALIVE, now: () => now }), null)
  assert.ok(!fs.existsSync(markerPath(home)), 'an unclaimed bridge cannot wedge later update attempts')
})

test('live updater claim replaces the pre-spawn Desktop bridge', () => {
  const home = tmpHome('handoff-claim-order')
  const now = 1_000_000_000_000
  const startedAt = Math.floor(now / 1000) - 8

  writeUpdateMarker(home, 1010, { now: () => now, startedAt, handoffBridge: true })

  const [bridgePidLine, bridgeStartedLine, bridgeKindLine] = fs
    .readFileSync(markerPath(home), 'utf8')
    .split('\n')

  assert.equal(Number.parseInt(bridgePidLine, 10), 1010, 'the Desktop owns the bridge')
  assert.equal(Number.parseInt(bridgeStartedLine, 10), startedAt)
  assert.equal(bridgeKindLine, 'handoff-bridge', 'the pre-spawn marker is explicitly bounded')
  assert.ok(
    readLiveUpdateMarker(home, { kill: DEAD, now: () => now }),
    'the tagged bridge remains live inside the claim grace even after its owner exits'
  )

  writeUpdateMarker(home, 2020, { now: () => now, startedAt })

  const [pidLine, startedLine, kindLine] = fs.readFileSync(markerPath(home), 'utf8').split('\n')
  assert.equal(Number.parseInt(pidLine, 10), 2020, 'the updater becomes the live marker owner')
  assert.equal(Number.parseInt(startedLine, 10), startedAt, 'the original acquisition time is preserved')
  assert.equal(kindLine, '', 'the updater claim is no longer a bounded bridge')
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

test('an expired live marker still blocks a hand-off', () => {
  const home = tmpHome('conflict-expired')
  const now = 1_000_000_000_000
  writeMarker(home, 1010, Math.floor((now - UPDATE_MARKER_MAX_AGE_MS - 60_000) / 1000))
  assert.ok(updateHandoffConflict(home, { kill: ALIVE, now: () => now }))
})

test('minutes-scale elapsed time is formatted as "Nm Ss"', () => {
  const home = tmpHome('conflict-minutes')
  const now = 1_000_000_000_000
  writeMarker(home, 1010, Math.floor(now / 1000) - 125) // 2m 5s old
  const conflict = updateHandoffConflict(home, { kill: ALIVE, now: () => now })
  assert.ok(conflict)
  assert.match(conflict.message, /2m 5s/)
})

// The Desktop holds the marker under its own pid during the holder drain so
// the gateway/serve watchdogs stay quiet, then hands the slot back before the
// hand-off script claims it with CreateNew (2026-09-05).
test('releaseUpdateMarkerIfOwnedBy removes only a marker that names the given pid', () => {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-marker-release-'))

  try {
    assert.equal(releaseUpdateMarkerIfOwnedBy(home, process.pid), false, 'nothing to release')

    writeUpdateMarker(home, process.pid, { startedAt: Math.floor(Date.now() / 1000) })
    assert.equal(fs.existsSync(markerPath(home)), true)
    assert.equal(releaseUpdateMarkerIfOwnedBy(home, process.pid + 1), false, 'another owner keeps its marker')
    assert.equal(fs.existsSync(markerPath(home)), true)
    assert.equal(releaseUpdateMarkerIfOwnedBy(home, process.pid), true)
    assert.equal(fs.existsSync(markerPath(home)), false)
  } finally {
    fs.rmSync(home, { recursive: true, force: true })
  }
})


test.each([
  [99.9, 'matching'], [100.9, 'matching'], [101, 'stale'], [null, 'unknown']
])('PID creation %s compared with marker timestamp 100 is %s', (created, expected) => {
  assert.equal(probePidIdentity(123, 100, { kill: ALIVE, getProcessCreatedAt: () => created as number | null }), expected)
})

test('proven live owner survives the advisory age ceiling', () => {
  const home = tmpHome('proven-overdue')
  writeMarker(home, 123, 100)
  assert.equal(readLiveUpdateMarker(home, { now: () => 10000000, kill: ALIVE, getProcessCreatedAt: () => 99.9 })?.kind, 'live')
  assert.ok(fs.existsSync(markerPath(home)))
})

test('recycled PID is reclaimed', () => {
  const home = tmpHome('recycled')
  writeMarker(home, 123, 100)
  assert.equal(readLiveUpdateMarker(home, { now: () => 110000, kill: ALIVE, getProcessCreatedAt: () => 102 }), null)
})

test('empty marker and unknown cleanup artifacts never age out', () => {
  const home = tmpHome('incomplete')
  fs.writeFileSync(markerPath(home), '')
  assert.equal(readLiveUpdateMarker(home, { now: () => 10000000 })?.kind, 'unreadable')
  assert.equal(fs.readFileSync(markerPath(home), 'utf8'), '')
  fs.unlinkSync(markerPath(home))
  fs.writeFileSync(markerPath(home) + '.cas-unknown', '123\n100\n')
  assert.equal(readLiveUpdateMarker(home, { now: () => 10000000 })?.kind, 'unreadable')
  assert.ok(fs.existsSync(markerPath(home) + '.cas-unknown'))
})

test('release restores a successor installed between read and isolation', () => {
  const home = tmpHome('release-race')
  const file = markerPath(home)
  writeMarker(home, process.pid, 100)
  const rename = fs.renameSync.bind(fs)

  const spy = vi.spyOn(fs, 'renameSync').mockImplementation((from, to) => {
    fs.writeFileSync(file, '4321\n101\n')

    return rename(from, to)
  })

  try {
    assert.equal(releaseUpdateMarkerIfOwnedBy(home, process.pid), false)
    assert.equal(fs.readFileSync(file, 'utf8'), '4321\n101\n')
  } finally {
    spy.mockRestore()
  }
})


test('exclusive claim refuses a second updater and binds exact release identity', () => {
  const home = tmpHome('exclusive')
  const first = acquireUpdateMarker(home)
  assert.equal(first.acquired, true)
  assert.equal(acquireUpdateMarker(home).acquired, false)
  assert.equal(releaseUpdateMarkerIfOwnedBy(home, first.owner.pid, first.owner.startedAt + 1), false)
  assert.equal(releaseUpdateMarkerIfOwnedBy(home, first.owner.pid, first.owner.startedAt), true)
})

test('exclusive claim never overwrites a contender arriving after the pre-check', () => {
  const home = tmpHome('claim-race')
  const file = markerPath(home)
  const open = fs.openSync.bind(fs)

  const spy = vi.spyOn(fs, 'openSync').mockImplementation((target, flags, mode) => {
    if (target === file && flags === 'wx') {
      fs.writeFileSync(file, '4321\n100\n')
    }

    return open(target, flags, mode)
  })

  try {
    assert.equal(acquireUpdateMarker(home).acquired, false)
    assert.equal(fs.readFileSync(file, 'utf8'), '4321\n100\n')
  } finally {
    spy.mockRestore()
  }
})
