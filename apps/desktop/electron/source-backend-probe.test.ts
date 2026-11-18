import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { resolveSourceInstallationBackend } from './source-backend'

/**
 * A staged launcher that counts its invocations and answers per mode:
 *   slow-then-fast - the first two invocations stall, later ones answer cleanly
 *   fail           - exits non-zero (a definitive "this install is broken")
 *   hang           - never answers
 */
function makeFixture(mode: 'slow-then-fast' | 'fail' | 'hang'): {
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
      '  slow-then-fast)',
      '    if [ "$count" -le 2 ]; then',
      // Longer than the default probe budget (15s), so the default path really
      // times out instead of merely being slow.
      `      ${sleep} 20`,
      '    fi',
      `    echo "Install directory: ${root}"`,
      '    exit 0',
      '    ;;',
      '  fail)',
      '    exit 3',
      '    ;;',
      '  hang)',
      `    ${sleep} 30`,
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

test('a launcher that stalls is retried and accepted instead of being called a missing install', { timeout: 90_000 }, async (): Promise<void> => {
  const fixture = makeFixture('slow-then-fast')

  const backend = await resolveSourceInstallationBackend(fixture.root, [], {
    probeTimeoutMs: 150,
    probeAttempts: 3,
    probeBackoffMs: 1
  })

  assert.notEqual(backend, null, 'a stalled launcher must not be reported as an unusable install')
  assert.ok(
    fixture.invocations() >= 3,
    `expected the probe to be retried after stalling, saw ${fixture.invocations()} invocation(s)`
  )
})

test('a launcher that exits non-zero fails fast without retries', { timeout: 90_000 }, async (): Promise<void> => {
  const fixture = makeFixture('fail')

  const backend = await resolveSourceInstallationBackend(fixture.root, [], {
    probeTimeoutMs: 150,
    probeAttempts: 3,
    probeBackoffMs: 1
  })

  assert.equal(backend, null, 'a broken launcher must still be reported as unusable')
  assert.equal(fixture.invocations(), 1, 'a definitive failure must not be retried')
})

test('a launcher that never answers is rejected after the attempts are exhausted', { timeout: 90_000 }, async (): Promise<void> => {
  const fixture = makeFixture('hang')

  const backend = await resolveSourceInstallationBackend(fixture.root, [], {
    probeTimeoutMs: 150,
    probeAttempts: 2,
    probeBackoffMs: 1
  })

  assert.equal(backend, null)
  assert.ok(
    fixture.invocations() <= 4,
    `retries must stay bounded, saw ${fixture.invocations()} invocation(s)`
  )
})
