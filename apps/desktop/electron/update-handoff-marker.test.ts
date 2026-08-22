import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..')
const POSIX_SCRIPT = path.join(REPO_ROOT, 'scripts', 'desktop-update', 'posix.sh')
const WINDOWS_SCRIPT = path.join(REPO_ROOT, 'scripts', 'desktop-update', 'windows.ps1')

function sandbox(tag: string) {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), `hermes-handoff-marker-${tag}-`))
  const installRoot = path.join(home, 'hermes-agent')
  fs.mkdirSync(installRoot)

  return { home, installRoot }
}

function markerStartedAt(home: string): number {
  const [, startedAt] = fs.readFileSync(path.join(home, '.hermes-update-in-progress'), 'utf8').split('\n')

  return Number.parseInt(startedAt, 10)
}

function runPosix(installRoot: string, startedAt?: string) {
  const env = { ...process.env }

  if (startedAt === undefined) {
    delete env.HERMES_UPDATE_STARTED_AT
  } else {
    env.HERMES_UPDATE_STARTED_AT = startedAt
  }

  return spawnSync('/bin/bash', [POSIX_SCRIPT, '--daemonized', '--install-root', installRoot, '--self-test-marker'], {
    env,
    encoding: 'utf8'
  })
}

function runWindows(installRoot: string, startedAt?: string) {
  const env = { ...process.env }

  if (startedAt === undefined) {
    delete env.HERMES_UPDATE_STARTED_AT
  } else {
    env.HERMES_UPDATE_STARTED_AT = startedAt
  }

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
      '-NoUi',
      '-NoMarkerCleanup',
      '-SelfTestMarker'
    ],
    { env, encoding: 'utf8' }
  )
}

function assertScriptHandoff(run: (installRoot: string, startedAt?: string) => ReturnType<typeof spawnSync>) {
  const preserved = sandbox('preserved')
  const acquiredAt = Math.floor(Date.now() / 1000) - 300
  const preservedResult = run(preserved.installRoot, String(acquiredAt))

  assert.equal(preservedResult.status, 0, String(preservedResult.stderr || preservedResult.stdout))
  assert.equal(markerStartedAt(preserved.home), acquiredAt, 'the script must preserve the Desktop acquisition time')

  const refreshed = sandbox('refreshed')
  fs.writeFileSync(path.join(refreshed.home, '.hermes-update-in-progress'), '999999\n1\n')
  const before = Math.floor(Date.now() / 1000)
  const refreshedResult = run(refreshed.installRoot, 'malformed')
  const after = Math.floor(Date.now() / 1000)

  assert.equal(refreshedResult.status, 0, String(refreshedResult.stderr || refreshedResult.stdout))
  assert.ok(
    markerStartedAt(refreshed.home) >= before && markerStartedAt(refreshed.home) <= after,
    'an invalid hand-off timestamp must start a fresh claim'
  )

  const oversized = sandbox('oversized')
  const oversizedBefore = Math.floor(Date.now() / 1000)
  const oversizedResult = run(oversized.installRoot, '99999999999999999999')
  const oversizedAfter = Math.floor(Date.now() / 1000)

  assert.equal(oversizedResult.status, 0, String(oversizedResult.stderr || oversizedResult.stdout))
  assert.ok(
    markerStartedAt(oversized.home) >= oversizedBefore && markerStartedAt(oversized.home) <= oversizedAfter,
    'an oversized hand-off timestamp must start a fresh claim'
  )
}

test.skipIf(process.platform === 'win32')('POSIX hand-off preserves the Desktop marker acquisition time', () => {
  assertScriptHandoff(runPosix)
})

test.skipIf(process.platform !== 'win32')('PowerShell hand-off preserves the Desktop marker acquisition time', () => {
  assertScriptHandoff(runWindows)
})
