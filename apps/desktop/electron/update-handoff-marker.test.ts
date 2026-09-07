/**
 * Hand-off marker contracts for the two update scripts.
 *
 * These scripts NO LONGER share one timestamp contract, and this file used to
 * pretend they did. History:
 *
 *  - ee6a9f8326 (2026-08-14) "carry acquisition age through scripts" made both
 *    posix.sh and windows.ps1 copy the Desktop's HERMES_UPDATE_STARTED_AT into
 *    the marker, and added the single shared assertion.
 *  - 5dd4d7a114 (2026-09-06) changed windows.ps1 to stamp its OWN kernel
 *    creation time instead, deliberately, and did not update this file. No CI
 *    job runs JS tests on Windows, so the win32 arm has never executed: on a
 *    Windows host at that commit it fails with a 300s drift (expected the
 *    Desktop's acquisition time, got the script's birth time).
 *
 * The identity-token contract is the correct one, and that drift is the fix
 * working rather than a bug. Every marker reader (update-marker.ts
 * probePidIdentity, hermes_mcp_update_gate.pid_identity_status,
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
 * claimant's creation time to agree within 1.5s.
 *
 * So: the win32 arm now asserts the identity-token contract, and the POSIX arm
 * keeps asserting the preserve contract that posix.sh still implements.
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
// POSIX: posix.sh still carries the Desktop's acquisition time through.
// ---------------------------------------------------------------------------

test.skipIf(process.platform === 'win32')('POSIX hand-off preserves the Desktop marker acquisition time', () => {
  const preserved = sandbox('preserved')
  const acquiredAt = Math.floor(Date.now() / 1000) - 300
  const preservedResult = runPosix(preserved.installRoot, String(acquiredAt))

  assert.equal(preservedResult.status, 0, String(preservedResult.stderr || preservedResult.stdout))
  assert.equal(markerStartedAt(preserved.home), acquiredAt, 'the script must preserve the Desktop acquisition time')

  const refreshed = sandbox('refreshed')
  fs.writeFileSync(refreshed.marker, '999999\n1\n')
  const before = Math.floor(Date.now() / 1000)
  const refreshedResult = runPosix(refreshed.installRoot, 'malformed')
  const after = Math.floor(Date.now() / 1000)

  assert.equal(refreshedResult.status, 0, String(refreshedResult.stderr || refreshedResult.stdout))
  assert.ok(
    markerStartedAt(refreshed.home) >= before && markerStartedAt(refreshed.home) <= after,
    'an invalid hand-off timestamp must start a fresh claim'
  )

  const oversized = sandbox('oversized')
  const oversizedBefore = Math.floor(Date.now() / 1000)
  const oversizedResult = runPosix(oversized.installRoot, '99999999999999999999')
  const oversizedAfter = Math.floor(Date.now() / 1000)

  assert.equal(oversizedResult.status, 0, String(oversizedResult.stderr || oversizedResult.stdout))
  assert.ok(
    markerStartedAt(oversized.home) >= oversizedBefore && markerStartedAt(oversized.home) <= oversizedAfter,
    'an oversized hand-off timestamp must start a fresh claim'
  )
})

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
  }
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
  }
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
  }
)
