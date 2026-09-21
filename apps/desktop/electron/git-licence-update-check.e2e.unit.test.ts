/**
 * E2E receipt for the Xcode-licence update-check misdiagnosis.
 *
 * Drives the real resolution the desktop updater performs — run git, read its
 * exit code and stderr, decide which error kind the renderer receives — against
 * a REAL git binary wrapped in a shim that reproduces the macOS failure
 * verbatim (exit 69, licence notice on stderr, binary spawns fine).
 *
 * Without the fix this path returns 'fetch-failed', which the About card renders
 * as "We couldn't reach the update server." on a machine whose network is
 * perfectly healthy.
 */
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import path from 'node:path'

import { afterAll, beforeAll, test } from 'vitest'

import { describeGitExitFailure } from './select-runnable-binary'

let workdir: string
let licenceBlockedGit: string
let healthyGit: string

const shim = (dir: string, name: string, body: string) => {
  const file = path.join(dir, name)
  writeFileSync(file, `#!/bin/sh\n${body}\n`)
  chmodSync(file, 0o755)

  return file
}

beforeAll(() => {
  workdir = mkdtempSync(path.join(tmpdir(), 'hermes-git-licence-'))

  // Byte-for-byte what /usr/bin/git prints once Xcode updates to a new major
  // and the licence has not been accepted for it.
  licenceBlockedGit = shim(
    workdir,
    'git-licence-blocked',
    `echo "You have not agreed to the Xcode license agreements. Please run 'sudo xcodebuild -license' from within a Terminal window to review and agree to the Xcode and Apple SDKs license." >&2
exit 69`
  )

  // A git that runs but cannot reach the network — the failure the
  // update-server copy is genuinely for.
  healthyGit = shim(
    workdir,
    'git-offline',
    `echo "fatal: unable to access 'https://github.com/NousResearch/hermes-agent/': Could not resolve host: github.com" >&2
exit 128`
  )
})

afterAll(() => rmSync(workdir, { force: true, recursive: true }))

/** Mirrors runGit: capture code + stderr from a real spawned process. */
const runGit = (binary: string): Promise<{ code: number | null; stderr: string }> =>
  new Promise(resolve => {
    const child = spawn(binary, ['ls-remote', 'origin', 'refs/heads/main'], { stdio: ['ignore', 'pipe', 'pipe'] })
    let stderr = ''
    child.stderr.on('data', chunk => (stderr += chunk.toString()))
    child.once('close', code => resolve({ code, stderr }))
  })

/** The decision checkUpdatesViaLsRemote makes on a failed ls-remote. */
const errorKindFor = (stderr: string) => (describeGitExitFailure(stderr) ? 'git-unusable' : 'fetch-failed')

test('a licence-blocked git is reported as a local git problem, never as an unreachable update server', async () => {
  const { code, stderr } = await runGit(licenceBlockedGit)

  // Precondition: this is a nonzero EXIT, not a spawn failure — which is why
  // the spawn-level classifier alone could never catch it.
  assert.equal(code, 69)
  assert.match(stderr, /xcodebuild -license/)

  assert.equal(errorKindFor(stderr), 'git-unusable')
  // The remedy the user must actually run survives to the renderer.
  assert.match(describeGitExitFailure(stderr) ?? '', /xcodebuild -license/)
})

test('a real network failure still reports as a failed fetch', async () => {
  const { code, stderr } = await runGit(healthyGit)

  assert.equal(code, 128)
  assert.equal(errorKindFor(stderr), 'fetch-failed')
})
