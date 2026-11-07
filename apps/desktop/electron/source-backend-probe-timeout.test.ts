import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterAll, test, vi } from 'vitest'

// resolveSourceInstallationBackend's probe budget is captured from
// process.env at module load (backend-probes PROBE_TIMEOUT_MS), so the
// forced-timeout repro (#79087: HERMES_PROBE_TIMEOUT_MS=1 turns a healthy
// install's liveness probe into a false "not installed") needs a fresh module
// graph with the env var already set. Every test below spawns real children;
// nothing is mocked.
const FORCED_PROBE_TIMEOUT_MS = '100'

// A structurally complete install: hermes_cli/main.py plus a published
// launcher at <root>/.hermes/bin/hermes. `script` becomes the launcher's body:
// a healthy runtime answers `--version` instantly; a cold/AV-stalled one never
// answers within the probe budget.
function makeInstall(root: string, script: string): void {
  fs.mkdirSync(path.join(root, 'hermes_cli'), { recursive: true })
  fs.writeFileSync(path.join(root, 'hermes_cli', 'main.py'), '')
  fs.mkdirSync(path.join(root, '.hermes', 'bin'), { recursive: true })

  const launcher = path.join(root, '.hermes', 'bin', 'hermes')
  fs.writeFileSync(launcher, script, { mode: 0o755 })
}

const loadResolver = async () => {
  process.env.HERMES_PROBE_TIMEOUT_MS = FORCED_PROBE_TIMEOUT_MS
  vi.resetModules()

  return import('./source-backend')
}

afterAll(() => {
  delete process.env.HERMES_PROBE_TIMEOUT_MS
})

// The forced-timeout repro from #79087: a probe timeout is a fact about this
// machine right now (cold AV scan, slow disk), never about the runtime — the
// structural checks (hermes_cli/main.py + published launcher) all passed
// before the probe ran. Collapsing timeout into null routed healthy Windows
// installs to the first-run onboarding chooser.
test.skipIf(process.platform === 'win32')(
  'a probe timeout keeps the structurally complete install instead of reading as absent',
  async () => {
    const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-probe-timeout-'))
    const root = path.join(temp, 'hermes-agent')

    try {
      makeInstall(root, '#!/bin/sh\nsleep 30\n')

      const { resolveSourceInstallationBackend } = await loadResolver()
      const backend = await resolveSourceInstallationBackend(root, ['serve'], { hermesHome: temp })

      assert.ok(backend, 'a timed-out probe on a complete install must not resolve as null')
      assert.equal(backend?.probeTimedOut, true)
      assert.equal(backend?.command, path.join(root, '.hermes', 'bin', 'hermes'))
      assert.equal(backend?.bootstrap, false)
    } finally {
      fs.rmSync(temp, { recursive: true, force: true })
    }
  },
  15_000
)

// The other side of the invariant: ENOENT and a non-zero exit ARE facts about
// the runtime (broken shim, half-dead install) and must keep meaning
// "unusable", so a genuinely broken install still falls through to bootstrap.
test.skipIf(process.platform === 'win32')(
  'a probe that exits non-zero still resolves as unusable',
  async () => {
    const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-probe-exit-'))
    const root = path.join(temp, 'hermes-agent')

    try {
      makeInstall(root, '#!/bin/sh\nexit 93\n')

      const { resolveSourceInstallationBackend } = await loadResolver()
      const backend = await resolveSourceInstallationBackend(root, ['serve'], { hermesHome: temp })

      assert.equal(backend, null)
    } finally {
      fs.rmSync(temp, { recursive: true, force: true })
    }
  },
  15_000
)

// Sanity for the fixture itself: with a healthy launcher the same resolver
// returns an unmarked backend, proving the timeout mark above comes from the
// probe outcome and not from the fixture shape.
test.skipIf(process.platform === 'win32')(
  'a healthy launcher resolves unmarked',
  async () => {
    const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-probe-healthy-'))
    const root = path.join(temp, 'hermes-agent')

    try {
      makeInstall(root, '#!/bin/sh\necho hermes 1.0.0\n')

      const { resolveSourceInstallationBackend } = await loadResolver()
      const backend = await resolveSourceInstallationBackend(root, ['serve'], { hermesHome: temp })

      assert.ok(backend)
      assert.equal(backend?.probeTimedOut, undefined)
      assert.equal(backend?.command, path.join(root, '.hermes', 'bin', 'hermes'))
      // `env` must stay usable for the real spawn (PYTHONPATH etc.).
      assert.ok(backend?.env && typeof backend.env === 'object')
    } finally {
      fs.rmSync(temp, { recursive: true, force: true })
    }
  },
  15_000
)
