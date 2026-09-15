import assert from 'node:assert/strict'
import { createHash } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'

import { test, vi } from 'vitest'

import type { InstallStamp, PayloadRuntime } from './install-stamp'
import { bundledPayload, installIdForRoot } from './payload-backend'

function stamp(runtime: PayloadRuntime, payload: InstallStamp['payload'] = 'bundled'): InstallStamp {
  return {
    schemaVersion: 1, commit: 'a'.repeat(40), commitDate: null, branch: null,
    builtAt: null, dirty: false, source: 'ci', distribution: 'desktop-app',
    updateMechanism: 'external', baseVersion: null, displayVersion: null,
    distance: null, payload, runtime, tag: 'v1.0.0'
  }
}

test('bundled launch paths come from build metadata without filesystem access', () => {
  const runtime: PayloadRuntime = {
    repoDir: 'source-tree', toolsDir: 'binary-store',
    storePython: 'binary-store/custom-python/python',
    sitePackages: 'dependencies/lib/python3.14/site-packages',
    commands: { hermes: 'commands/run-hermes', custom: 'commands/custom' }
  }

  const probe = vi.spyOn(fs, 'existsSync').mockImplementation(() => { throw new Error('runtime probe') })
  const read = vi.spyOn(fs, 'readFileSync').mockImplementation(() => { throw new Error('runtime read') })

  try {
    for (const resources of ['/not-created/resources', '/relocated app/resources']) {
      const result = bundledPayload(resources, stamp(runtime))
      assert.ok(result)
      assert.equal(result.storePython, path.join(resources, 'agent-payload', runtime.storePython))
      assert.equal(result.sitePackages, path.join(resources, 'agent-payload', runtime.sitePackages))
      assert.equal(result.shim, result.commands.hermes)
      assert.equal(result.commands.custom, path.join(resources, 'agent-payload', runtime.commands.custom))
    }

    assert.equal(probe.mock.calls.length, 0)
    assert.equal(read.mock.calls.length, 0)
    assert.equal(bundledPayload('/resources', stamp(runtime, 'light')), null)
    assert.equal(bundledPayload('/resources', stamp(runtime, 'bootstrap')), null)
    assert.equal(bundledPayload('/resources', null), null)
  } finally {
    probe.mockRestore()
    read.mockRestore()
  }
})

// ─── update channel helpers ─────────────────────────────────────────

test('installIdForRoot matches the Python install id (sha16 of the canonical path)', () => {
  // sha256('/home/u/.hermes/hermes-agent')[:16] — recomputed independently.
  assert.equal(
    installIdForRoot('/home/u/.hermes/hermes-agent'),
    createHash('sha256').update('/home/u/.hermes/hermes-agent', 'utf8').digest('hex').slice(0, 16)
  )
  // The canonicalizer output is what gets hashed (symlinked homes).
  assert.equal(
    installIdForRoot('/link/hermes-agent', () => '/real/hermes-agent'),
    installIdForRoot('/real/hermes-agent')
  )
})

