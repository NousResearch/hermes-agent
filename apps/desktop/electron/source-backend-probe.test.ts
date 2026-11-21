import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { resolveSourceInstallationBackend } from './source-backend'
import { chooseUpdaterArgs } from './windows-hermes-path'

/**
 * A staged launcher that counts its invocations and answers per mode:
 *   hang - never answers before the probe budget
 *   fail - exits non-zero (a definitive "this install is broken")
 *   ok   - answers cleanly
 */
function makeFixture(mode: 'hang' | 'fail' | 'ok'): {
  root: string
  invocations: () => number
} {
  const temp: string = fs.mkdtempSync(path.join(os.tmpdir(), 'desktop-probe-'))
  const root: string = path.join(temp, 'install')
  const binDir: string = path.join(root, '.hermes', 'bin')
  const counter: string = path.join(temp, 'invocations')
  const sleep: string = fs.existsSync('/usr/bin/sleep') ? '/usr/bin/sleep' : '/bin/sleep'

  fs.mkdirSync(binDir, { recursive: true })
  fs.mkdirSync(path.join(root, 'hermes_cli'), { recursive: true })
  fs.writeFileSync(path.join(root, 'hermes_cli', 'main.py'), '# probe fixture\n')
  fs.writeFileSync(
    path.join(binDir, 'hermes'),
    [
      '#!/bin/sh',
      'count=0',
      `if [ -f "${counter}" ]; then read count < "${counter}"; fi`,
      'count=$((count + 1))',
      `echo "$count" > "${counter}"`,
      `case "${mode}" in`,
      '  hang)',
      // Longer than the probe budget under test, so the probe really times out.
      `    ${sleep} 30`,
      '    exit 0',
      '    ;;',
      '  fail)',
      '    exit 3',
      '    ;;',
      '  ok)',
      `    echo "Install directory: ${root}"`,
      '    exit 0',
      '    ;;',
      'esac',
      ''
    ].join('\n')
  )
  fs.chmodSync(path.join(binDir, 'hermes'), 0o755)

  return {
    root,
    invocations: (): number =>
      fs.existsSync(counter) ? Number(fs.readFileSync(counter, 'utf8').trim()) : 0
  }
}

test('boot accepts an install whose probe never answers instead of calling it missing', { timeout: 90_000 }, async (): Promise<void> => {
  const fixture = makeFixture('hang')

  const backend = await resolveSourceInstallationBackend(fixture.root, [], {
    probeTimeoutMs: 150,
    onProbeTimeout: 'accept'
  })

  // The install is demonstrably there (`hermes_cli/main.py` and a resolvable launcher), so a
  // probe that never answers cannot be evidence that it is gone. Returning null here reaches
  // `bootstrap-needed`, which latches the first-run setup card over a working checkout (#79087).
  assert.notEqual(backend, null, 'a stalled machine must never be reported as a missing install')
  assert.equal(backend?.root, fixture.root)
  assert.equal(backend?.bootstrap, false)
})

test('the default stays strict so an unproven runtime cannot be called runnable', { timeout: 90_000 }, async (): Promise<void> => {
  const fixture = makeFixture('hang')

  const backend = await resolveSourceInstallationBackend(fixture.root, [], { probeTimeoutMs: 150 })

  // isSourceRuntimeUsable() relies on this to feed chooseUpdaterArgs(). Reporting an
  // unproven pair as runnable would pick --update for a runtime that may need --repair;
  // windows-hermes-path.ts documents that only a runnable pair may use --update.
  assert.equal(backend, null, 'the strict default must not treat a timeout as runnable')
  assert.deepEqual(
    chooseUpdaterArgs({ runtimeUsable: backend !== null }, 'release/1.2'),
    ['--repair', '--branch', 'release/1.2'],
    'an unproven runtime must still be handed to the updater as --repair'
  )
})

test('a launcher that exits non-zero fails without being treated as a timeout', { timeout: 90_000 }, async (): Promise<void> => {
  const fixture = makeFixture('fail')

  // Even the boot rung must reject this: a non-zero exit is a definitive answer that the
  // install cannot run, unlike a timeout which is merely no answer at all.
  const backend = await resolveSourceInstallationBackend(fixture.root, [], {
    probeTimeoutMs: 150,
    onProbeTimeout: 'accept'
  })

  assert.equal(backend, null, 'a broken launcher must still be reported as unusable')
})

test('a healthy launcher is accepted in both modes', { timeout: 90_000 }, async (): Promise<void> => {
  for (const onProbeTimeout of ['reject', 'accept'] as const) {
    const fixture = makeFixture('ok')

    const backend = await resolveSourceInstallationBackend(fixture.root, ['serve'], {
      probeTimeoutMs: 5_000,
      onProbeTimeout
    })

    assert.notEqual(backend, null, `a healthy launcher must be accepted (${onProbeTimeout})`)
    assert.deepEqual(backend?.args, ['serve'], 'the caller keeps ownership of the backend args')
    assert.equal(fixture.invocations(), 1, 'a healthy launcher must be probed exactly once')
  }
})

test('a stalling launcher costs one retry budget, not a multiplied one', { timeout: 90_000 }, async (): Promise<void> => {
  const fixture = makeFixture('hang')

  await resolveSourceInstallationBackend(fixture.root, [], {
    probeTimeoutMs: 150,
    onProbeTimeout: 'accept'
  })

  // execProbe already retries a timeout once internally. A caller-side loop on top of that
  // multiplied the budget (#157543 review): six invocations and ~92s at the default budget.
  assert.equal(
    fixture.invocations(),
    2,
    `expected exactly one retry from execProbe, saw ${fixture.invocations()} invocation(s)`
  )
})
