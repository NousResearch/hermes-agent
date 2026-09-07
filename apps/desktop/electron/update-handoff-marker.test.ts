/**
 * Hand-off marker contracts for the two update scripts.
 *
 * Both scripts share ONE contract: the marker's second line is an identity
 * token for the pid on its first line, stamped by the script that owns it.
 * Every marker reader (update-marker.ts probePidIdentity,
 * hermes_mcp_update_gate.pid_identity_status,
 * hermes_cli/update_lock.py) classifies an owner as `matching` only when
 *
 *     kernel_creation_time(marker.pid) < marker.started_at + 1
 *
 * The Desktop stamps HERMES_UPDATE_STARTED_AT *before* it spawns the script, so
 * that value can only predate the script's own birth. Copying it into the
 * marker therefore makes the script's own live claim read as `stale` and get
 * reclaimed out from under it — the 2026-09-06 incident recorded in windows.ps1
 * (release at :06.977Z, process born in :07, Desktop aborted with "did not
 * acknowledge the protected handoff"). It would also fail the ack check in
 * waitForAcknowledgedUpdaterClaim, which requires the marker timestamp and the
 * claimant's creation time to agree within 1.5s. So each script stamps its own
 * kernel creation time and keeps the Desktop's value for its log line only.
 *
 * History, because these two arms drifted apart once already:
 *
 *  - ee6a9f8326 (2026-08-14) "carry acquisition age through scripts" made both
 *    posix.sh and windows.ps1 copy the Desktop's HERMES_UPDATE_STARTED_AT into
 *    the marker, and added a single shared "preserve" assertion.
 *  - 5dd4d7a114 (2026-09-06) changed windows.ps1 to stamp its OWN kernel
 *    creation time instead, deliberately, and did not update this file. No CI
 *    job runs JS tests on Windows, so the win32 arm never executed: on a
 *    Windows host at that commit it failed with a 300s drift (expected the
 *    Desktop's acquisition time, got the script's birth time) — the fix
 *    working, not a bug.
 *  - This change gives posix.sh the same identity stamp (/proc/<pid>/stat
 *    field 22 plus /proc/stat btime on Linux, `ps -o lstart=` on macOS, wall
 *    clock as a logged fallback), so the POSIX arm below asserts the same
 *    contract as the win32 arm.
 */

import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..')
const POSIX_SCRIPT = path.join(REPO_ROOT, 'scripts', 'desktop-update', 'posix.sh')
const WINDOWS_SCRIPT = path.join(REPO_ROOT, 'scripts', 'desktop-update', 'windows.ps1')
const NONCE = 'c4'.repeat(24)

function sandbox(tag: string) {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), `hermes-handoff-marker-${tag}-`))
  const installRoot = path.join(home, 'hermes-agent')
  fs.mkdirSync(installRoot)

  return { home, installRoot, marker: path.join(home, '.hermes-update-in-progress') }
}

function markerLines(home: string): string[] {
  return fs.readFileSync(path.join(home, '.hermes-update-in-progress'), 'utf8').split('\n')
}

function markerStartedAt(home: string): number {
  return Number.parseInt(markerLines(home)[1], 10)
}

function scriptEnv(startedAt?: string): NodeJS.ProcessEnv {
  const env = { ...process.env }

  if (startedAt === undefined) {
    delete env.HERMES_UPDATE_STARTED_AT
  } else {
    env.HERMES_UPDATE_STARTED_AT = startedAt
  }

  return env
}

function runPosix(installRoot: string, startedAt?: string) {
  return spawnSync('/bin/bash', [POSIX_SCRIPT, '--daemonized', '--install-root', installRoot, '--self-test-marker'], {
    env: scriptEnv(startedAt),
    encoding: 'utf8'
  })
}

/**
 * Drive windows.ps1 step 0. -SelfTestMarker only stops the run after the
 * claim; since this PR it no longer selects a different marker body or a
 * create-only CAS, so what runs here is the production claim path.
 */
function runWindows(installRoot: string, desktopPid: number, startedAt?: string) {
  return spawnSync(
    'powershell.exe',
    [
      '-NoProfile',
      '-ExecutionPolicy',
      'Bypass',
      '-File',
      WINDOWS_SCRIPT,
      '-InstallRoot',
      installRoot,
      '-DesktopPid',
      String(desktopPid),
      '-HandoffNonce',
      NONCE,
      '-NoUi',
      '-NoMarkerCleanup',
      '-SelfTestMarker'
    ],
    { env: scriptEnv(startedAt), encoding: 'utf8' }
  )
}

/** Seed the exact Desktop claim windows.ps1 adopts under its CAS. */
function seedDesktopClaim(marker: string, desktopPid: number, startedAt: number): string {
  const body = `${desktopPid}\n${startedAt}\n`

  fs.writeFileSync(marker, body)

  return body
}

// ---------------------------------------------------------------------------
// POSIX: posix.sh stamps its own creation time, like windows.ps1.
// ---------------------------------------------------------------------------

test.skipIf(process.platform === 'win32')(
  'POSIX hand-off stamps its own creation time, not the Desktop acquisition time',
  () => {
    const claimed = sandbox('identity')
    // 300s in the past: large enough that copying it through would be visible,
    // and far outside the readers' 1s tolerance, so it would read as `stale`.
    const acquiredAt = Math.floor(Date.now() / 1000) - 300

    const before = Math.floor(Date.now() / 1000)
    const result = runPosix(claimed.installRoot, String(acquiredAt))
    const after = Math.floor(Date.now() / 1000)

    assert.equal(result.status, 0, String(result.stderr || result.stdout))

    const lines = markerLines(claimed.home)
    const ownerPid = Number.parseInt(lines[0], 10)
    const startedAt = Number.parseInt(lines[1], 10)

    // --daemonized keeps the script in the process we spawned, so the marker
    // must name exactly that pid — the one whose creation time it stamps.
    assert.equal(ownerPid, result.pid, 'the script claims the marker with its own pid')

    // THE contract: the stamped value identifies the process named on line 1.
    // Carrying the Desktop's older acquisition time through would make every
    // reader classify this live updater as stale and reclaim its marker.
    assert.notEqual(
      startedAt,
      acquiredAt,
      'the Desktop acquisition time predates the script and cannot identify it'
    )
    assert.ok(
      startedAt >= before - 5 && startedAt <= after,
      `the stamp must be the script's own creation time (got ${startedAt}, run spanned ${before}..${after})`
    )
    assert.equal(lines.length, 3, 'two lines and a trailing newline; the wire format does not grow')

    // The Desktop's value is now log-only, so no shape of it can move the
    // stamp off this process. (Both of these used to select a "fresh claim"
    // branch that no longer exists.)
    for (const [label, startedAtEnv] of [
      ['malformed', 'malformed'],
      ['oversized', '99999999999999999999'],
      ['absent', undefined]
    ] as Array<[string, string | undefined]>) {
      const box = sandbox(`ignored-${label}`)
      // A stale marker from a previous run must not survive our claim.
      fs.writeFileSync(box.marker, '999999\n1\n')

      const boxBefore = Math.floor(Date.now() / 1000)
      const boxResult = runPosix(box.installRoot, startedAtEnv)
      const boxAfter = Math.floor(Date.now() / 1000)

      assert.equal(boxResult.status, 0, `${label}: ${String(boxResult.stderr || boxResult.stdout)}`)
      assert.equal(Number.parseInt(markerLines(box.home)[0], 10), boxResult.pid, `${label}: our pid owns the marker`)
      assert.ok(
        markerStartedAt(box.home) >= boxBefore - 5 && markerStartedAt(box.home) <= boxAfter,
        `${label}: the stamp is this script's creation time regardless of the environment`
      )
    }
  },
  60_000
)

// ---------------------------------------------------------------------------
// Windows: windows.ps1 replaces the Desktop's claim with its own identity.
// ---------------------------------------------------------------------------

test.skipIf(process.platform !== 'win32')(
  'PowerShell hand-off stamps its own creation time, not the Desktop acquisition time',
  () => {
    const claimed = sandbox('identity')
    const desktopPid = process.pid
    // 300s in the past: large enough that copying it through would be visible,
    // and far outside the readers' 1s tolerance, so it would read as `stale`.
    const acquiredAt = Math.floor(Date.now() / 1000) - 300
    seedDesktopClaim(claimed.marker, desktopPid, acquiredAt)

    const before = Math.floor(Date.now() / 1000)
    const result = runWindows(claimed.installRoot, desktopPid, String(acquiredAt))
    const after = Math.floor(Date.now() / 1000)

    assert.equal(result.status, 0, String(result.stderr || result.stdout))

    const lines = markerLines(claimed.home)
    const ownerPid = Number.parseInt(lines[0], 10)
    const startedAt = Number.parseInt(lines[1], 10)

    assert.notEqual(ownerPid, desktopPid, 'the script owns the marker after adopting it')
    assert.ok(ownerPid > 0)

    // THE contract, and why the pre-existing 300s "drift" is correct: the
    // stamped value identifies the process named on line 1. Carrying the
    // Desktop's older acquisition time through would make every reader
    // classify this live updater as stale and reclaim its marker.
    assert.notEqual(
      startedAt,
      acquiredAt,
      'the Desktop acquisition time predates the script and cannot identify it'
    )
    assert.ok(
      startedAt >= before - 5 && startedAt <= after,
      `the stamp must be the script's own creation time (got ${startedAt}, run spanned ${before}..${after})`
    )
    assert.equal(lines.length, 3, 'two lines and a trailing newline; the wire format does not grow')

    // The claim is only usable because the ack ties it to this run (#B4).
    const ack = fs.readFileSync(`${claimed.marker}.ack`, 'utf8').split('\n')
    assert.equal(ack[0], NONCE)
    assert.equal(Number.parseInt(ack[1], 10), ownerPid)
    assert.equal(Number.parseInt(ack[2], 10), startedAt)
  },
  // Each PowerShell spawn costs ~1-2s; the 5s default trips under the
  // parallel full-project run, which is how this file looked broken.
  60_000
)

test.skipIf(process.platform !== 'win32')(
  'PowerShell hand-off refuses a Desktop claim it cannot authenticate',
  () => {
    // Each of these used to "start a fresh claim". Under the single CAS path a
    // script that cannot verify the Desktop's exact claim must not invent one:
    // it has no way to know whether another update is already running.
    for (const [label, startedAt] of [
      ['malformed', 'malformed'],
      ['oversized', '99999999999999999999'],
      ['absent', undefined]
    ] as Array<[string, string | undefined]>) {
      const box = sandbox(`refused-${label}`)
      const desktopPid = process.pid
      const body = seedDesktopClaim(box.marker, desktopPid, 1_700_000_000)

      const result = runWindows(box.installRoot, desktopPid, startedAt)

      assert.equal(result.status, 8, `${label}: ${String(result.stderr || result.stdout)}`)
      assert.equal(fs.readFileSync(box.marker, 'utf8'), body, `${label}: a refused claim leaves the marker alone`)
      assert.equal(fs.existsSync(`${box.marker}.ack`), false, `${label}: nothing was acknowledged`)
    }
  },
  // Each PowerShell spawn costs ~1-2s; the 5s default trips under the
  // parallel full-project run, which is how this file looked broken.
  60_000
)

test.skipIf(process.platform !== 'win32')(
  'PowerShell hand-off refuses a Desktop claim that no longer matches byte for byte',
  () => {
    const box = sandbox('mismatch')
    const desktopPid = process.pid
    const startedAt = 1_700_000_000
    // Another owner won the marker between the Desktop's claim and our adopt.
    // One second of difference is enough; the CAS is exact.
    const foreign = seedDesktopClaim(box.marker, desktopPid, startedAt - 1)

    const result = runWindows(box.installRoot, desktopPid, String(startedAt))

    assert.equal(result.status, 8, String(result.stderr || result.stdout))
    assert.equal(fs.readFileSync(box.marker, 'utf8'), foreign, 'a foreign owner keeps its claim')
  },
  // Each PowerShell spawn costs ~1-2s; the 5s default trips under the
  // parallel full-project run, which is how this file looked broken.
  60_000
)
