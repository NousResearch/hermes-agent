/**
 * Tests for electron/update-marker.ts — the in-app update mutual-exclusion
 * marker that prevents a desktop relaunched mid-update from spawning a backend
 * the updater then kills in a loop (#50238).
 *
 * Run with: node --test electron/update-marker.test.ts
 * (Wired into npm test:desktop:platforms in package.json.)
 *
 * Why this matters: the gate must (a) report a live update only when the
 * updater pid identity is proven alive, (b) never wedge on a body nobody owns
 * -- empty/malformed/future markers self-heal after a dwell and an owner whose
 * identity cannot be proven expires at the age ceiling, and (c) self-heal only
 * through an exact-content CAS so a real writer never loses its claim.
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
  UPDATE_MARKER_DWELL_MS,
  UPDATE_MARKER_MAX_AGE_MS,
  updateHandoffConflict,
  writeUpdateMarker
} from './update-marker'
import { createCachedWindowsProcessCreateTimeProbe } from './windows-process-identity'

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

test('expired PROVEN owner remains live and is marked overdue', () => {
  const home = tmpHome('expired')
  const now = 1_000_000_000_000
  const startedAt = Math.floor((now - UPDATE_MARKER_MAX_AGE_MS - 60_000) / 1000)
  writeMarker(home, 4242, startedAt)
  const result = readLiveUpdateMarker(home, {
    kill: ALIVE, now: () => now, getProcessCreatedAt: () => startedAt
  })
  assert.equal(result?.kind, 'live')
  assert.equal(result?.overdue, true)
  assert.ok(fs.existsSync(markerPath(home)), 'a proven live owner stays authoritative past the ceiling')
})

// Upstream main unlinks a marker past the ceiling. The PR removed that for
// EVERY owner, so one recycled pid wedged updates and the MCP bridge for good.
// Restore it for owners we cannot prove -- never for owners we can.
test('expired UNPROVABLE owner is reclaimed at the age ceiling', () => {
  const home = tmpHome('expired-unknown')
  const now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor((now - UPDATE_MARKER_MAX_AGE_MS - 60_000) / 1000))
  assert.equal(readLiveUpdateMarker(home, { kill: ALIVE, now: () => now }), null)
  assert.ok(!fs.existsSync(markerPath(home)), 'an unprovable owner cannot hold the gate forever')
})

test('malformed marker blocks for the dwell, then self-heals', () => {
  const home = tmpHome('malformed')
  let now = 1_000_000_000_000
  fs.writeFileSync(markerPath(home), 'not-a-pid\nnonsense')

  const first = readLiveUpdateMarker(home, { kill: ALIVE, now: () => now })
  assert.equal(first?.kind, 'unreadable')
  assert.equal(first?.reason, 'malformed')
  assert.ok(fs.existsSync(markerPath(home)), 'a body seen once may still be a write in flight')
  assert.ok(String(first?.message).includes(markerPath(home)), 'the message must name the file')

  now += UPDATE_MARKER_DWELL_MS
  assert.equal(readLiveUpdateMarker(home, { kill: ALIVE, now: () => now }), null)
  assert.ok(!fs.existsSync(markerPath(home)), 'an unowned body must not block every future launch')
})

test('a body that changes inside the dwell is never reclaimed', () => {
  const home = tmpHome('malformed-inflight')
  let now = 1_000_000_000_000
  fs.writeFileSync(markerPath(home), 'partial')
  assert.equal(readLiveUpdateMarker(home, { kill: ALIVE, now: () => now })?.kind, 'unreadable')

  // The writer completes its rewrite mid-dwell: the dwell restarts and the
  // finished claim is honoured rather than deleted out from under its owner.
  now += UPDATE_MARKER_DWELL_MS
  const startedAt = Math.floor(now / 1000) - 5
  writeMarker(home, 4242, startedAt)
  assert.equal(
    readLiveUpdateMarker(home, { kill: ALIVE, now: () => now, getProcessCreatedAt: () => startedAt })?.pid,
    4242
  )
  assert.ok(fs.existsSync(markerPath(home)))
})

test('future-dated marker self-heals on the same dwell rule', () => {
  const home = tmpHome('future')
  let now = 1_000_000_000_000
  writeMarker(home, 4242, Math.floor(now / 1000) + 3_600)
  assert.equal((readLiveUpdateMarker(home, { kill: ALIVE, now: () => now }) as { reason?: string } | null)?.reason, 'future')

  now += UPDATE_MARKER_DWELL_MS
  assert.equal(readLiveUpdateMarker(home, { kill: ALIVE, now: () => now }), null)
  assert.ok(!fs.existsSync(markerPath(home)))
})

test('an empty marker -- a torn write with no attacker -- self-heals', () => {
  const home = tmpHome('empty')
  let now = 1_000_000_000_000
  fs.writeFileSync(markerPath(home), '')
  assert.equal((readLiveUpdateMarker(home, { kill: ALIVE, now: () => now }) as { reason?: string } | null)?.reason, 'malformed')

  now += UPDATE_MARKER_DWELL_MS
  assert.equal(readLiveUpdateMarker(home, { kill: ALIVE, now: () => now }), null)
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

test('an expired PROVEN live marker still blocks a hand-off', () => {
  const home = tmpHome('conflict-expired')
  const now = 1_000_000_000_000
  const startedAt = Math.floor((now - UPDATE_MARKER_MAX_AGE_MS - 60_000) / 1000)
  writeMarker(home, 1010, startedAt)
  assert.ok(updateHandoffConflict(home, { kill: ALIVE, now: () => now, getProcessCreatedAt: () => startedAt }))
})

test('a blocked hand-off names the marker path so recovery is possible', () => {
  const home = tmpHome('conflict-blocked-path')
  fs.writeFileSync(markerPath(home), 'garbage')
  const conflict = updateHandoffConflict(home, { kill: ALIVE, now: () => 1_000_000_000_000 })
  assert.ok(conflict)
  assert.ok(conflict.message.includes(markerPath(home)), conflict.message)
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
  [99.9, 'matching'], [100.9, 'matching'], [101, 'stale'], [null, 'unknown'],
  // A probe that has not answered yet is NOT an answer of "unprovable".
  [undefined, 'unknown-pending']
])('PID creation %s compared with marker timestamp 100 is %s', (created, expected) => {
  assert.equal(
    probePidIdentity(123, 100, { kill: ALIVE, getProcessCreatedAt: () => created as number | null | undefined }),
    expected
  )
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

// Inverted from the PR, which pinned "never age out" as intended behaviour.
// An empty marker names nobody and blocked every launch forever, with no
// message saying which file to delete. Unknown CAS artifacts DO still stay:
// each names a real in-flight release, so removing one resurrects a dead claim.
test('an empty marker ages out; unknown cleanup artifacts still do not', () => {
  const home = tmpHome('incomplete')
  let now = 10_000_000
  fs.writeFileSync(markerPath(home), '')
  const blocked = readLiveUpdateMarker(home, { now: () => now })
  assert.equal(blocked?.kind, 'unreadable')
  assert.ok(String(blocked?.message).includes(markerPath(home)), 'the blocking message must name the file')

  now += UPDATE_MARKER_DWELL_MS
  assert.equal(readLiveUpdateMarker(home, { now: () => now }), null)
  assert.equal(fs.existsSync(markerPath(home)), false)

  fs.writeFileSync(markerPath(home) + '.cas-unknown', '123\n100\n')
  now += UPDATE_MARKER_DWELL_MS * 10
  assert.equal(readLiveUpdateMarker(home, { now: () => now })?.kind, 'unreadable')
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


// ---------------------------------------------------------------------------
// The REAL production probe, not a synchronous stub.
//
// Every arm above injects `getProcessCreatedAt`, which answers instantly — and
// that is exactly what hid the defect. Production readers (main.ts's marker
// poll, updateHandoffConflict, acquireUpdateMarker) pass no `kill`, so they get
// createCachedWindowsProcessCreateTimeProbe, whose cold miss starts an async OS
// query and returns immediately. While that answer was indistinguishable from
// "could not prove it", the FIRST read of any overdue claim reclaimed a
// proven-live updater's marker and admitted a second updater over the same
// install. These arms drive the real probe with a controllable query.
// ---------------------------------------------------------------------------

function settlingProbe() {
  let settle!: (value: number | null) => void
  const answer = new Promise<number | null>(resolve => { settle = resolve })

  return {
    probe: createCachedWindowsProcessCreateTimeProbe({ now: () => 1_000, query: () => answer }),
    // Resolving is not enough: the cache settles in a promise callback.
    settleTo: async (value: number | null) => {
      settle(value)
      await new Promise(resolve => { setImmediate(resolve) })
    }
  }
}

const OVERDUE_NOW = 1_000_000_000_000
const OVERDUE_STARTED_AT = Math.floor((OVERDUE_NOW - UPDATE_MARKER_MAX_AGE_MS - 60_000) / 1000)

function readWithProbe(home: string, probe: (pid: number) => number | null | undefined) {
  return readLiveUpdateMarker(home, { kill: ALIVE, now: () => OVERDUE_NOW, getProcessCreatedAt: probe })
}

test('an unsettled real probe never reclaims an overdue owner, then proves it', async () => {
  const home = tmpHome('probe-unsettled')
  writeMarker(home, 4242, OVERDUE_STARTED_AT)
  const { probe, settleTo } = settlingProbe()

  const cold = readWithProbe(home, probe)

  assert.equal(cold?.kind, 'live', 'a pending probe is not evidence that the owner expired')
  assert.ok(fs.existsSync(markerPath(home)), 'the first read must not reclaim a live owner')

  // (b) settled to the owner's true creation time => matching, never expires.
  await settleTo(OVERDUE_STARTED_AT)
  const proven = readWithProbe(home, probe)

  assert.equal(proven?.kind, 'live')
  assert.equal(proven?.overdue, true, 'still past the ceiling, and still authoritative')
  assert.ok(fs.existsSync(markerPath(home)), 'a proven owner outlives the ceiling')
})

test('an overdue owner IS reclaimed once the real probe settles without proof', async () => {
  const home = tmpHome('probe-settled-null')
  writeMarker(home, 4242, OVERDUE_STARTED_AT)
  const { probe, settleTo } = settlingProbe()

  assert.equal(readWithProbe(home, probe)?.kind, 'live', 'pending holds the gate')

  await settleTo(null)

  assert.equal(readWithProbe(home, probe), null, 'a settled unprovable owner expires at the ceiling')
  assert.ok(!fs.existsSync(markerPath(home)), 'recoverability is the whole point of the ceiling')
})

test('a recycled pid is reclaimed as soon as the real probe settles', async () => {
  const home = tmpHome('probe-settled-recycled')
  writeMarker(home, 4242, OVERDUE_STARTED_AT)
  const { probe, settleTo } = settlingProbe()

  assert.equal(readWithProbe(home, probe)?.kind, 'live', 'pending holds the gate')

  // Born after the claim was stamped: this pid cannot be the claimant.
  await settleTo(OVERDUE_STARTED_AT + 60)

  assert.equal(readWithProbe(home, probe), null)
  assert.ok(!fs.existsSync(markerPath(home)), 'a proven-recycled pid never holds the gate')
})
