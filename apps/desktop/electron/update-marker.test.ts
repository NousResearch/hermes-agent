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
import { type ChildProcess, execFileSync, spawn } from 'node:child_process'
import os from 'os'
import path from 'path'

import { afterEach, test } from 'vitest'

import {
  cachedCreateTimeProbe,
  claimBridgeMarker,
  EMPTY_MARKER_GRACE_MS,
  formatCreateTime,
  inspectUpdateMarker,
  isPidAlive,
  markerPath,
  parseUpdateMarker,
  posixProcessState,
  processCreateTime,
  psCreateTime,
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
  assert.deepEqual(parseUpdateMarker('42\n100\n'), { pid: 42, startedAt: 100, ct: null, delegate: null, run: null })
  assert.deepEqual(parseUpdateMarker('42\n100\nct:1700000000.125\ndelegate:77 ct:1700000001.500\n'), {
    pid: 42,
    startedAt: 100,
    ct: 1700000000.125,
    delegate: { pid: 77, ct: 1700000001.5 },
    run: null
  })
  assert.equal(parseUpdateMarker('not-a-pid\nnonsense'), null)
})

// A2: one positional parse in every language (Python, Rust, PowerShell, bash, TS).
test.each([
  ['BOM + CRLF v2', '\uFEFF42\r\n100\r\nct:5.250\r\n', { pid: 42, startedAt: 100, ct: 5.25, delegate: null, run: null }],
  ['fractional started_at is MALFORMED', '42\n100.5\nct:5.000\n', null],
  ['missing started_at is MALFORMED', '42\n', null],
  ['garbled started_at is MALFORMED', '42\nsoon\n', null],
  ['garbage ct => v1', '42\n100\nct:abc\n', { pid: 42, startedAt: 100, ct: null, delegate: null, run: null }],
  ['delegate on line 3 is ignored', '42\n100\ndelegate:77 ct:1.000\n', { pid: 42, startedAt: 100, ct: null, delegate: null, run: null }],
  ['space after delegate: is ignored', '42\n100\nct:1.000\ndelegate: 77 ct:1.000\n', { pid: 42, startedAt: 100, ct: 1, delegate: null, run: null }],
  ['delegate without ct is ignored', '42\n100\nct:1.000\ndelegate:77\n', { pid: 42, startedAt: 100, ct: 1, delegate: null, run: null }],
  [
    'empty ct line (Python probe failure) + delegate on line 4',
    '42\n100\n\ndelegate:77 ct:2.500\n',
    { pid: 42, startedAt: 100, ct: null, delegate: { pid: 77, ct: 2.5 }, run: null }
  ]
])('A2 parse: %s', (_name, body, expected) => {
  assert.deepEqual(parseUpdateMarker(body), expected)
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

test.skipIf(!HAS_CT_PROBE)('a reused pid (creation time mismatch) is dead and left for the script (V22, A7 rule 3)', async () => {
  const home = tmpHome('v2-reused')
  const owner = await liveOwner()
  const ct = await processCreateTime(owner.pid)
  // The marker names the pid the live process now holds, but a creation time
  // an hour earlier: the original owner died and the OS recycled its pid.
  fs.writeFileSync(markerPath(home), `${owner.pid}\n${minutesAgo(2)}\nct:${formatCreateTime(ct! - 3600)}\n`)

  const body = fs.readFileSync(markerPath(home), 'utf8')
  assert.equal(await readLiveUpdateMarker(home), null)
  assert.equal(fs.readFileSync(markerPath(home), 'utf8'), body, 'Electron never deletes; the script helper reclaims')
})

test('a v1 marker (no creation time) keeps the legacy 20-minute ceiling', async () => {
  const home = tmpHome('v1-old')
  const owner = await liveOwner()
  fs.writeFileSync(markerPath(home), `${owner.pid}\n${Math.floor((Date.now() - UPDATE_MARKER_MAX_AGE_MS) / 1000) - 60}\n`)

  assert.equal(await readLiveUpdateMarker(home), null, 'v1 pid reuse reads as not running')
  assert.ok(fs.existsSync(markerPath(home)), 'and is left for the script helper (A7 rule 3)')
})

// A1: Windows refuses creation-time queries for SYSTEM/elevated/other-user
// pids. A live pid whose ct cannot be read must not be live forever: a reused
// pid would park every Desktop boot. It is live only inside the v1 ceiling.
test('a v2 owner whose creation time is UNREADABLE is live only inside the 20-minute ceiling (A1)', async () => {
  const home = tmpHome('v2-unreadable')
  const owner = await liveOwner()
  const accessDenied = () => null

  fs.writeFileSync(markerPath(home), `${owner.pid}\n${minutesAgo(5)}\nct:1700000000.000\n`)
  assert.ok(await readLiveUpdateMarker(home, { createTime: accessDenied }), 'young: unknown ct + live pid reads live')

  fs.writeFileSync(markerPath(home), `${owner.pid}\n${minutesAgo(25)}\nct:1700000000.000\n`)
  assert.equal(await readLiveUpdateMarker(home, { createTime: accessDenied }), null, 'past the ceiling it is dead')
  assert.ok(fs.existsSync(markerPath(home)), 'not running, so boot is never parked forever — and left in place (A7 rule 3)')
})

// A1 on a real Windows host: csrss.exe is a SYSTEM (protected) process whose
// Get-Process .StartTime is access-denied. CIM still answers, so a marker
// whose pid was reused by such a process is judged by creation time — DEAD —
// instead of "unknown, live forever".
test.runIf(process.platform === 'win32')('Windows reads a SYSTEM process creation time; a reused pid is dead (A1)', async () => {
  const home = tmpHome('win-system')

  const pid = Number(
    execFileSync(
      'powershell.exe',
      ['-NoProfile', '-NonInteractive', '-Command', "(Get-CimInstance Win32_Process -Filter \"Name='csrss.exe'\" | Select-Object -First 1).ProcessId"],
      { encoding: 'utf8' }
    ).trim()
  )

  assert.ok(pid > 0, 'csrss.exe runs on every Windows session')
  const ct = await processCreateTime(pid)
  assert.ok(ct !== null && ct > 1e9 && ct <= Date.now() / 1000, `creation time of SYSTEM pid ${pid}: ${ct}`)

  fs.writeFileSync(markerPath(home), `${pid}\n${minutesAgo(1)}\nct:${formatCreateTime(ct! - 3600)}\n`)
  assert.equal(await readLiveUpdateMarker(home), null)
  assert.ok(fs.existsSync(markerPath(home)), 'left in place (A7 rule 3)')
})

test('a gate wait probes each pid creation time ONCE (A1: no powershell spawn per poll)', async () => {
  const home = tmpHome('ct-cache')
  const owner = await liveOwner()
  const ct = await processCreateTime(owner.pid)
  let probes = 0

  const createTime = cachedCreateTimeProbe(pid => {
    probes += 1

    return processCreateTime(pid)
  })

  fs.writeFileSync(markerPath(home), `${owner.pid}\n${minutesAgo(1)}\nct:${formatCreateTime(ct ?? 0)}\n`)

  for (let poll = 0; poll < 5; poll++) {
    assert.ok(await readLiveUpdateMarker(home, { createTime }))
  }

  assert.equal(probes, 1)
})

// A1 (macOS source): `ps -o lstart` is local time. Parsed by a different TZ
// engine (or in the DST fall-back hour) a live owner reads hours off its own
// record and is judged a reused pid. Printed and parsed as UTC it cannot drift.
test.skipIf(process.platform === 'win32')('ps lstart creation time is TZ-independent (A1)', async () => {
  const owner = await liveOwner()
  const savedTz = process.env.TZ

  try {
    // glibc honours this POSIX offset; V8's ICU does not and falls back.
    process.env.TZ = '<+0530>-5:30'
    const viaPs = psCreateTime(owner.pid)
    const reference = process.platform === 'linux' ? await processCreateTime(owner.pid) : Date.now() / 1000

    assert.ok(viaPs !== null && reference !== null)
    assert.ok(Math.abs(viaPs - reference) <= 2, `ps says ${viaPs}, the kernel says ${reference}`)
  } finally {
    if (savedTz === undefined) {
      delete process.env.TZ
    } else {
      process.env.TZ = savedTz
    }
  }
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
  assert.ok(fs.existsSync(markerPath(home)), 'left in place (A7 rule 3)')

  fs.writeFileSync(markerPath(home), `4242\n${minutesAgo(0)}\n`)
  assert.equal(await readLiveUpdateMarker(home, { kill: () => true, processState: () => 'Z' }), null)

  fs.writeFileSync(markerPath(home), `4242\n${minutesAgo(0)}\n`)
  assert.ok(await readLiveUpdateMarker(home, { kill: () => true, processState: () => null }))
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

  // A bridge of this very process is not someone else's update: the gate's
  // in-process flags cover the hand-off window, and a stale bridge left by a
  // failed withdraw must not park this Desktop's own backend restart.
  assert.equal((await inspectUpdateMarker(home)).state, 'ours')
  assert.equal(await readLiveUpdateMarker(home), null)
})

test('the bridge claim refuses a LIVE foreign owner and reports (never reclaims) a dead one', async () => {
  const home = tmpHome('bridge-conflict')
  const owner = await liveOwner()
  fs.writeFileSync(markerPath(home), `${owner.pid}\n${minutesAgo(30)}\nct:${formatCreateTime((await processCreateTime(owner.pid)) ?? 0)}\n`)

  const refused = await claimBridgeMarker(home, { startedAt: 1 })
  assert.equal(refused.ok, false)
  assert.equal(!refused.ok && refused.conflict?.pid, owner.pid)
  assert.match(String(!refused.ok && refused.conflict?.message), /already running/)
  assert.equal(parseUpdateMarker(fs.readFileSync(markerPath(home), 'utf8'))!.pid, owner.pid, 'never overwritten')

  const dead = `${await deadPid()}\n${minutesAgo(1)}\nct:1.000\nrun:desk-1-a-0000\n`
  fs.writeFileSync(markerPath(home), dead)
  const blocked = await claimBridgeMarker(home, { startedAt: 2 })
  assert.equal(blocked.ok, false, 'exclusive create only (A7 rule 3)')
  assert.equal(blocked.existing?.state, 'dead', 'reported for the script helper to reclaim')
  assert.equal(blocked.existing?.run, 'desk-1-a-0000')
  assert.equal(fs.readFileSync(markerPath(home), 'utf8'), dead)
})

// r1 M1: the staged Tauri updater path pre-writes the marker for the spawned
// updater. As v1 it stayed under the 20-minute ceiling (a live 25-minute
// update was aged out and deleted) and blocked the delegate line.
test.skipIf(!HAS_CT_PROBE)('writeUpdateMarker names the spawned updater with its creation time (v2)', async () => {
  const home = tmpHome('write-v2')
  const updater = await liveOwner()

  await writeUpdateMarker(home, updater.pid, { startedAt: minutesAgo(25) })

  const marker = parseUpdateMarker(fs.readFileSync(markerPath(home), 'utf8'))!
  assert.equal(marker.pid, updater.pid)
  assert.ok(marker.ct !== null && Math.abs(marker.ct - (await processCreateTime(updater.pid))!) <= 2)
  assert.ok(await readLiveUpdateMarker(home), 'a 25-minute-old live updater is still running')
  assert.ok(fs.existsSync(markerPath(home)))
})

// A3: a claim is never visible half-written, and a 0-byte claim being written
// by an O_EXCL-only writer is not mistaken for garbage and deleted.
test('a fresh 0-byte marker reads LIVE and survives; a stale one reads dead and is left (A3, A7 rule 3)', async () => {
  const home = tmpHome('empty')
  const file = markerPath(home)
  fs.writeFileSync(file, '')

  assert.ok(await readLiveUpdateMarker(home), 'a claim being written blocks')
  assert.ok(fs.existsSync(file), 'and is never deleted')
  assert.equal((await claimBridgeMarker(home, { startedAt: 1 })).ok, false, 'no claim races past it')

  const old = (Date.now() - EMPTY_MARKER_GRACE_MS - 5000) / 1000
  fs.utimesSync(file, old, old)
  assert.equal(await readLiveUpdateMarker(home), null)
  assert.ok(fs.existsSync(file))
})

test('claims publish by hard link and leave no tmp litter; a dead writer tmp is reclaimed (A3)', async () => {
  const home = tmpHome('link')
  const deadTmp = path.join(home, `.hermes-update-in-progress.${await deadPid()}.tmp`)
  fs.writeFileSync(deadTmp, 'half')

  const claim = await claimBridgeMarker(home, { startedAt: 7 })

  assert.ok(claim.ok)
  assert.equal(fs.readFileSync(markerPath(home), 'utf8'), claim.body)
  assert.deepEqual(
    fs.readdirSync(home).filter(name => name !== '.hermes-update-in-progress'),
    [],
    'no tmp sibling survives the claim'
  )
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

test('the hand-off counts as started only when a real legacy script process takes the marker', async () => {
  const home = tmpHome('handoff-taken')
  const file = markerPath(home)
  const startedAt = Math.floor(Date.now() / 1000)

  // A real old-style "script": waits, then claims the marker in its own name,
  // echoing HERMES_UPDATE_STARTED_AT on line 2.
  const script = spawn(
    process.execPath,
    [
      '-e',
      `setTimeout(() => { require('fs').writeFileSync(${JSON.stringify(file)}, process.pid + '\\n${startedAt}\\n'); setInterval(() => {}, 1000) }, 300)`
    ],
    { stdio: 'ignore' }
  )

  children.push(script)

  const taken = await waitForHandoffClaim(home, process.pid, { startedAt, timeoutMs: 10_000, pollMs: 50 })

  assert.deepEqual(taken, { taken: true, pid: script.pid })
})

test('a wrapper that exits 0 without the script ever claiming is NOT a hand-off', async () => {
  const home = tmpHome('handoff-never')
  await claimBridgeMarker(home, { startedAt: 5, runId: 'desk-1-a-0000' })
  const wrapper = spawn(process.execPath, ['-e', 'process.exit(0)'], { stdio: 'ignore' })
  await new Promise(resolve => wrapper.once('exit', resolve))

  assert.deepEqual(
    await waitForHandoffClaim(home, process.pid, { runId: 'desk-1-a-0000', timeoutMs: 400, pollMs: 50 }),
    { taken: false }
  )
})
