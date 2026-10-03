/**
 * Tests for electron/update-marker.ts — the update marker (contract C1, v2)
 * that keeps a Desktop reopened mid-update from booting a backend onto the
 * runtime being replaced, and keeps two updaters off one checkout.
 *
 * The liveness cells use REAL processes: a sleeping node child as the owner,
 * its real creation time (or a deliberately wrong one for pid reuse), and a
 * real child that takes the marker over for the hand-off cells.
 */

import fs from 'fs'
import assert from 'node:assert/strict'
import { type ChildProcess, spawn } from 'node:child_process'
import os from 'os'
import path from 'path'

import { afterEach, test } from 'vitest'

import {
  claimBridgeMarker,
  compareAndDeleteMarker,
  formatCreateTime,
  isPidAlive,
  markerPath,
  parseUpdateMarker,
  posixProcessState,
  processCreateTime,
  readLiveUpdateMarker,
  UPDATE_MARKER_MAX_AGE_MS,
  updateHandoffConflict,
  waitForHandoffClaim,
  writeUpdateMarker
} from './update-marker'

const homes: string[] = []
const children: ChildProcess[] = []

afterEach(() => {
  for (const child of children.splice(0)) {
    child.kill('SIGKILL')
  }

  for (const home of homes.splice(0)) {
    fs.rmSync(home, { recursive: true, force: true })
  }
})

function tmpHome(tag: string) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), `hermes-marker-${tag}-`))
  homes.push(dir)

  return dir
}

/** A real, live owner process (sleeps until killed). */
async function liveOwner(): Promise<ChildProcess & { pid: number }> {
  const child = spawn(process.execPath, ['-e', 'setInterval(() => {}, 1000)'], { stdio: 'ignore' })
  children.push(child)
  await new Promise(resolve => child.once('spawn', resolve))

  return child as ChildProcess & { pid: number }
}

/** A pid that existed and is now gone (reaped). */
async function deadPid(): Promise<number> {
  const child = spawn(process.execPath, ['-e', ''], { stdio: 'ignore' })
  await new Promise(resolve => child.once('exit', resolve))

  return child.pid as number
}

function minutesAgo(minutes: number) {
  return Math.floor(Date.now() / 1000) - minutes * 60
}

const HAS_CT_PROBE = process.platform === 'linux' || process.platform === 'darwin' || process.platform === 'win32'

// ---------------------------------------------------------------------------
// Parse
// ---------------------------------------------------------------------------

test('parses v1, v2 and the delegate line; garbage is null', () => {
  assert.deepEqual(parseUpdateMarker('42\n100\n'), { pid: 42, startedAt: 100, ct: null, delegate: null })
  assert.deepEqual(parseUpdateMarker('42\n100\nct:1700000000.125\ndelegate:77 ct:1700000001.500\n'), {
    pid: 42,
    startedAt: 100,
    ct: 1700000000.125,
    delegate: { pid: 77, ct: 1700000001.5 }
  })
  assert.equal(parseUpdateMarker('not-a-pid\nnonsense'), null)
})

// ---------------------------------------------------------------------------
// Liveness against REAL processes (C1 rule 3, desktop V3/V22)
// ---------------------------------------------------------------------------

test.skipIf(!HAS_CT_PROBE)('a LIVE v2 owner past 20 minutes stays live and is NOT deleted (V3)', async () => {
  const home = tmpHome('v2-old-live')
  const owner = await liveOwner()
  const ct = await processCreateTime(owner.pid)
  assert.ok(ct, 'the real owner has a probeable creation time')
  fs.writeFileSync(markerPath(home), `${owner.pid}\n${minutesAgo(25)}\nct:${formatCreateTime(ct!)}\n`)

  const live = await readLiveUpdateMarker(home)

  assert.ok(live, 'a 25-minute-old update whose owner is alive is still running')
  assert.equal(live!.pid, owner.pid)
  assert.ok(fs.existsSync(markerPath(home)), 'a live owner is never aged out')
})

test.skipIf(!HAS_CT_PROBE)('a reused pid (creation time mismatch) is dead and compare-deleted (V22)', async () => {
  const home = tmpHome('v2-reused')
  const owner = await liveOwner()
  const ct = await processCreateTime(owner.pid)
  // The marker names the pid the live process now holds, but a creation time
  // an hour earlier: the original owner died and the OS recycled its pid.
  fs.writeFileSync(markerPath(home), `${owner.pid}\n${minutesAgo(2)}\nct:${formatCreateTime(ct! - 3600)}\n`)

  assert.equal(await readLiveUpdateMarker(home), null)
  assert.ok(!fs.existsSync(markerPath(home)), 'a reused-pid marker self-heals')
})

test('a v1 marker (no creation time) keeps the legacy 20-minute ceiling', async () => {
  const home = tmpHome('v1-old')
  const owner = await liveOwner()
  fs.writeFileSync(markerPath(home), `${owner.pid}\n${Math.floor((Date.now() - UPDATE_MARKER_MAX_AGE_MS) / 1000) - 60}\n`)

  assert.equal(await readLiveUpdateMarker(home), null, 'v1 pid reuse must still self-heal')
  assert.ok(!fs.existsSync(markerPath(home)))
})

test('a dead owner with a LIVE delegate keeps the marker live (C1 rule 6)', async () => {
  const home = tmpHome('delegate')
  const gone = await deadPid()
  const delegate = await liveOwner()
  const delegateCt = await processCreateTime(delegate.pid)
  const ctPart = delegateCt === null ? '' : ` ct:${formatCreateTime(delegateCt)}`
  fs.writeFileSync(markerPath(home), `${gone}\n${minutesAgo(1)}\nct:1.000\ndelegate:${delegate.pid}${ctPart}\n`)

  const live = await readLiveUpdateMarker(home)

  assert.ok(live, 'a killed hand-off script must not hide a still-running hermes update')
  assert.equal(live!.pid, delegate.pid)
  delegate.kill('SIGKILL')
  await new Promise(resolve => delegate.once('exit', resolve))
  assert.equal(await readLiveUpdateMarker(home), null, 'both gone => dead')
})

test('dead pid / zombie => no live update; unknown state fails open', async () => {
  const home = tmpHome('dead')
  fs.writeFileSync(markerPath(home), `${await deadPid()}\n${minutesAgo(0)}\n`)
  assert.equal(await readLiveUpdateMarker(home), null)
  assert.ok(!fs.existsSync(markerPath(home)))

  fs.writeFileSync(markerPath(home), `4242\n${minutesAgo(0)}\n`)
  assert.equal(await readLiveUpdateMarker(home, { kill: () => true, processState: () => 'Z' }), null)

  fs.writeFileSync(markerPath(home), `4242\n${minutesAgo(0)}\n`)
  assert.ok(await readLiveUpdateMarker(home, { kill: () => true, processState: () => null }))
})

test('compare-and-delete never removes a claim written after the dead verdict (C1 rule 5)', () => {
  const home = tmpHome('cas')
  fs.writeFileSync(markerPath(home), '999999\n1\n')
  const judgedDead = fs.readFileSync(markerPath(home))
  fs.writeFileSync(markerPath(home), `${process.pid}\n2\nct:3.000\n`)

  assert.equal(compareAndDeleteMarker(home, judgedDead), false)
  assert.ok(fs.existsSync(markerPath(home)), 'the newer claim survives')
})

test('isPidAlive / posixProcessState basics', () => {
  assert.equal(isPidAlive(process.pid), true)
  assert.equal(isPidAlive(-1), false)
  assert.equal(
    isPidAlive(4242, () => {
      throw Object.assign(new Error('operation not permitted'), { code: 'EPERM' })
    }),
    true
  )

  if (process.platform !== 'win32') {
    assert.ok(!String(posixProcessState(process.pid)).toUpperCase().startsWith('Z'))
    assert.equal(posixProcessState(2147483647), null)
  }
})

// ---------------------------------------------------------------------------
// Writers (C1 rule 4, C2 bridge)
// ---------------------------------------------------------------------------

test('the bridge marker names THIS process with its creation time (V4)', async () => {
  const home = tmpHome('bridge')
  const claim = await claimBridgeMarker(home, { startedAt: 1234 })

  assert.ok(claim.ok)
  const marker = parseUpdateMarker(fs.readFileSync(markerPath(home), 'utf8'))!
  assert.equal(marker.pid, process.pid)
  assert.equal(marker.startedAt, 1234)

  if (HAS_CT_PROBE) {
    assert.ok(marker.ct !== null && Math.abs(marker.ct - (await processCreateTime(process.pid))!) <= 2)
  }

  assert.ok(await readLiveUpdateMarker(home), 'our own bridge claim reads live')
})

test('the bridge claim refuses a LIVE foreign owner and reclaims a dead one', async () => {
  const home = tmpHome('bridge-conflict')
  const owner = await liveOwner()
  fs.writeFileSync(markerPath(home), `${owner.pid}\n${minutesAgo(30)}\nct:${formatCreateTime((await processCreateTime(owner.pid)) ?? 0)}\n`)

  const refused = await claimBridgeMarker(home, { startedAt: 1 })
  assert.equal(refused.ok, false)
  assert.equal(!refused.ok && refused.conflict?.pid, owner.pid)
  assert.match(String(!refused.ok && refused.conflict?.message), /already running/)
  assert.equal(parseUpdateMarker(fs.readFileSync(markerPath(home), 'utf8'))!.pid, owner.pid, 'never overwritten')

  fs.writeFileSync(markerPath(home), `${await deadPid()}\n${minutesAgo(1)}\n`)
  assert.ok((await claimBridgeMarker(home, { startedAt: 2 })).ok, 'a dead claim is reclaimed')
})

test('writeUpdateMarker (staged updater) never overwrites a live claim', async () => {
  const home = tmpHome('write-live')
  const owner = await liveOwner()
  fs.writeFileSync(markerPath(home), `${owner.pid}\n${minutesAgo(1)}\n`)

  await writeUpdateMarker(home, 2020)

  assert.equal(parseUpdateMarker(fs.readFileSync(markerPath(home), 'utf8'))!.pid, owner.pid)
  assert.ok(await updateHandoffConflict(home), 'the live owner still blocks a new hand-off')
})

// ---------------------------------------------------------------------------
// Hand-off confirmation (C2, desktop V7)
// ---------------------------------------------------------------------------

test('the hand-off counts as started only when a real script process takes the marker', async () => {
  const home = tmpHome('handoff-taken')
  await claimBridgeMarker(home, { startedAt: 5 })
  const file = markerPath(home)
  // A real "script": waits, then claims the marker in its own name.
  const script = spawn(
    process.execPath,
    ['-e', `setTimeout(() => { require('fs').writeFileSync(${JSON.stringify(file)}, process.pid + '\\n5\\n'); setInterval(() => {}, 1000) }, 300)`],
    { stdio: 'ignore' }
  )
  children.push(script)

  const taken = await waitForHandoffClaim(home, process.pid, { timeoutMs: 10_000, pollMs: 50 })

  assert.deepEqual(taken, { taken: true, pid: script.pid })
})

test('a wrapper that exits 0 without the script ever claiming is NOT a hand-off', async () => {
  const home = tmpHome('handoff-never')
  await claimBridgeMarker(home, { startedAt: 5 })
  const wrapper = spawn(process.execPath, ['-e', 'process.exit(0)'], { stdio: 'ignore' })
  await new Promise(resolve => wrapper.once('exit', resolve))

  assert.deepEqual(await waitForHandoffClaim(home, process.pid, { timeoutMs: 400, pollMs: 50 }), { taken: false })
})
