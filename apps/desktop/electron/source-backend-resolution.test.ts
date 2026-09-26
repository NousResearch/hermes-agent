import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test, vi } from 'vitest'

import { resolveSourceInstallationBackend } from './source-backend'

// #123888: every fall-through from rung 4 (the active install root) used to be
// silent -- missing launcher and probe failures both returned null with no log
// line, so a healthy install and a broken one were indistinguishable in
// desktop.log and the app landed on the first-run setup chooser. The resolver
// now reports WHY it rejected the install via its `log` callback.

function makeFixture(root: string, launcherBody: string): string {
  fs.mkdirSync(path.join(root, 'hermes_cli'), { recursive: true })
  fs.writeFileSync(path.join(root, 'hermes_cli', 'main.py'), '')
  fs.mkdirSync(path.join(root, '.hermes', 'bin'), { recursive: true })

  const launcher: string = path.join(
    root,
    '.hermes',
    'bin',
    process.platform === 'win32' ? 'hermes.cmd' : 'hermes'
  )

  fs.writeFileSync(launcher, launcherBody, { mode: 0o755 })

  return launcher
}

function makeTempRoot(prefix: string): string {
  return fs.mkdtempSync(path.join(os.tmpdir(), prefix))
}

test('a missing launcher in an otherwise complete checkout is logged, not silent', async (): Promise<void> => {
  const root: string = makeTempRoot('desktop-resolution-')

  try {
    fs.mkdirSync(path.join(root, 'hermes_cli'), { recursive: true })
    fs.writeFileSync(path.join(root, 'hermes_cli', 'main.py'), '')

    const lines: string[] = []
    const backend: unknown = await resolveSourceInstallationBackend(root, [], { log: m => lines.push(m) })

    assert.equal(backend, null)
    assert.equal(lines.length, 1)
    assert.ok(lines[0].includes('No Hermes launcher'), lines[0])
    assert.ok(lines[0].includes(path.join(root, '.hermes', 'bin')), lines[0])
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('a failing --version probe is logged with the launcher and its exit code', async (): Promise<void> => {
  const root: string = makeTempRoot('desktop-resolution-')

  try {
    const launcher: string = makeFixture(
      root,
      process.platform === 'win32' ? '@echo off\r\nexit /b 67\r\n' : '#!/bin/sh\nexit 67\n'
    )

    const lines: string[] = []
    const backend: unknown = await resolveSourceInstallationBackend(root, [], { log: m => lines.push(m) })

    assert.equal(backend, null)
    assert.equal(lines.length, 1)
    assert.ok(lines[0].includes(launcher), lines[0])
    assert.ok(lines[0].includes('exit code 67'), lines[0])
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})

test('a probe timeout is logged as a timeout, naming the per-attempt budget', async (): Promise<void> => {
  vi.resetModules()
  vi.stubEnv('HERMES_PROBE_TIMEOUT_MS', '300')

  try {
    const { resolveSourceInstallationBackend: resolveWithShortBudget } = await import('./source-backend')
    const { PROBE_TIMEOUT_MS } = await import('./backend-probes')

    assert.equal(PROBE_TIMEOUT_MS, 300)

    const root: string = makeTempRoot('desktop-resolution-')
    const node: string = JSON.stringify(process.execPath)

    const sleep: string =
      process.platform === 'win32'
        ? `@echo off\r\n${node} -e "setTimeout(function(){},5000)"\r\n`
        : `#!/bin/sh\nexec ${node} -e "setTimeout(function(){},5000)"\n`

    try {
      const launcher: string = makeFixture(root, sleep)
      const lines: string[] = []
      const backend: unknown = await resolveWithShortBudget(root, [], { log: m => lines.push(m) })

      assert.equal(backend, null)
      assert.equal(lines.length, 1)
      assert.ok(lines[0].includes(launcher), lines[0])
      assert.ok(lines[0].includes('timed out after 300ms'), lines[0])
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  } finally {
    vi.unstubAllEnvs()
    vi.resetModules()
  }
})

// The marker only attests "a desktop bootstrap ran here"; rung 4 must keep
// launching a probe-verified runtime with NO marker at all (issue #123888:
// the stale/missing marker was suspected while the runtime was fine).
test('a probe-verified launcher resolves WITHOUT any bootstrap marker', async (): Promise<void> => {
  const root: string = makeTempRoot('desktop-resolution-')

  try {
    const launcher: string = makeFixture(
      root,
      process.platform === 'win32' ? '@echo off\r\n' : '#!/bin/sh\nexit 0\n'
    )

    const backend = await resolveSourceInstallationBackend(root, [])

    assert.ok(backend)
    assert.equal(backend.command, launcher)
    assert.equal(backend.bootstrap, false)
    assert.equal(fs.existsSync(path.join(root, '.hermes-bootstrap-complete')), false)
  } finally {
    fs.rmSync(root, { recursive: true, force: true })
  }
})
