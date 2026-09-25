import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..')
const POSIX_SCRIPT = path.join(REPO_ROOT, 'scripts', 'desktop-update', 'posix.sh')

function sandbox(tag: string) {
  const home = fs.mkdtempSync(path.join(os.tmpdir(), `hermes-handoff-marker-${tag}-`))
  const installRoot = path.join(home, 'hermes-agent')
  fs.mkdirSync(installRoot)

  return { home, installRoot }
}

// Each sandbox is a real mkdtemp() under the OS temp directory; leaving it
// behind pollutes $TMPDIR/%TEMP% by one directory per test run (#122130).
function cleanupSandbox(home: string) {
  fs.rmSync(home, { recursive: true, force: true })
}

function markerStartedAt(home: string): number {
  const [, startedAt] = fs.readFileSync(path.join(home, '.hermes-update-in-progress'), 'utf8').split('\n')

  return Number.parseInt(startedAt, 10)
}

function runPosix(installRoot: string, startedAt?: string) {
  const env: NodeJS.ProcessEnv = { ...process.env, HERMES_HOME: path.dirname(installRoot) }

  if (startedAt === undefined) {
    delete env.HERMES_UPDATE_STARTED_AT
  } else {
    env.HERMES_UPDATE_STARTED_AT = startedAt
  }

  return spawnSync(
    '/usr/bin/env',
    ['bash', POSIX_SCRIPT, '--daemonized', '--install-root', installRoot, '--self-test-marker'],
    {
      env,
      encoding: 'utf8'
    }
  )
}

function assertScriptHandoff(run: (installRoot: string, startedAt?: string) => ReturnType<typeof spawnSync>) {
  const preserved = sandbox('preserved')
  try {
    const acquiredAt = Math.floor(Date.now() / 1000) - 300
    const preservedResult = run(preserved.installRoot, String(acquiredAt))

    assert.equal(preservedResult.status, 0, String(preservedResult.stderr || preservedResult.stdout))
    assert.equal(markerStartedAt(preserved.home), acquiredAt, 'the script must preserve the Desktop acquisition time')
  } finally {
    cleanupSandbox(preserved.home)
  }
  assert.equal(fs.existsSync(preserved.home), false, 'the self-test sandbox must not leak into the OS temp directory')

  const refreshed = sandbox('refreshed')
  try {
    fs.writeFileSync(path.join(refreshed.home, '.hermes-update-in-progress'), '999999\n1\n')
    const before = Math.floor(Date.now() / 1000)
    const refreshedResult = run(refreshed.installRoot, 'malformed')
    const after = Math.floor(Date.now() / 1000)

    assert.equal(refreshedResult.status, 0, String(refreshedResult.stderr || refreshedResult.stdout))
    assert.ok(
      markerStartedAt(refreshed.home) >= before && markerStartedAt(refreshed.home) <= after,
      'an invalid hand-off timestamp must start a fresh claim'
    )
  } finally {
    cleanupSandbox(refreshed.home)
  }
  assert.equal(fs.existsSync(refreshed.home), false, 'the self-test sandbox must not leak into the OS temp directory')

  const oversized = sandbox('oversized')
  try {
    const oversizedBefore = Math.floor(Date.now() / 1000)
    const oversizedResult = run(oversized.installRoot, '99999999999999999999')
    const oversizedAfter = Math.floor(Date.now() / 1000)

    assert.equal(oversizedResult.status, 0, String(oversizedResult.stderr || oversizedResult.stdout))
    assert.ok(
      markerStartedAt(oversized.home) >= oversizedBefore && markerStartedAt(oversized.home) <= oversizedAfter,
      'an oversized hand-off timestamp must start a fresh claim'
    )
  } finally {
    cleanupSandbox(oversized.home)
  }
  assert.equal(fs.existsSync(oversized.home), false, 'the self-test sandbox must not leak into the OS temp directory')
}

test.skipIf(process.platform === 'win32')('POSIX hand-off preserves the Desktop marker acquisition time', () => {
  assertScriptHandoff(runPosix)
})
