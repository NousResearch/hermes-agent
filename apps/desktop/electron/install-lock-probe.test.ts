/**
 * The install-lock glue: does it ask about the right files, and does it stay
 * off the Restart Manager when nothing is locked?
 *
 * The lock classification itself is proved in install-mutation-set.test.ts
 * against an injected filesystem, and the gate's polling economics in
 * backend-release-gate.test.ts. What only this module can get wrong is the
 * binding: which resources an install root expands to, and whether an
 * unlocked install still pays for a PowerShell attribution child.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { describe, it, vi } from 'vitest'

import {
  attributedInstallHolders,
  installLockResources,
  isAnyInstallResourceLocked,
  probeInstallLocks,
  venvHermesShimPath
} from './install-lock-probe'

const restartManagerMocks = vi.hoisted(() => ({
  listHolders: vi.fn()
}))

vi.mock('./windows-restart-manager', () => ({
  listRestartManagerHoldersForResources: restartManagerMocks.listHolders,
  RESTART_MANAGER_DEFAULT_TIMEOUT_MS: 12_000
}))

const IS_WINDOWS = process.platform === 'win32'

function makeFakeInstall(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-install-lock-'))

  const write = (relative: string) => {
    const target = path.join(root, ...relative.split('/'))

    fs.mkdirSync(path.dirname(target), { recursive: true })
    fs.writeFileSync(target, 'x')

    return target
  }

  write('venv/Scripts/hermes.exe')
  write('venv/Scripts/python.exe')
  write('venv/Lib/site-packages/tokenizers/_native.pyd')
  write('venv/Lib/site-packages/tokenizers/README.txt')
  write('.hermes-runtime/python/3.13.1/python313.dll')

  return root
}

describe('installLockResources', () => {
  it('is the venv mutation set, not the shim alone', () => {
    const root = makeFakeInstall()
    const resources = installLockResources(root)

    // The shim-only probe let the July 2026 half-updated venv through: the
    // real interpreter runs from .hermes-runtime and keeps site-packages
    // .pyd files mapped without touching hermes.exe.
    assert.ok(resources.includes(path.join(root, 'venv', 'Scripts', 'hermes.exe')))
    assert.ok(resources.includes(path.join(root, 'venv', 'Lib', 'site-packages', 'tokenizers', '_native.pyd')))

    // Non-mutated extensions are not resources.
    assert.ok(!resources.some(resource => resource.endsWith('README.txt')))

    // .hermes-runtime is shared with foreign uv tool venvs and is never
    // rewritten in place, so a process mapping it does not block this update.
    assert.ok(!resources.some(resource => resource.includes('.hermes-runtime')))
  })

  it('is empty for a checkout with no venv, which falls back to the shim probe', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-install-lock-bare-'))

    assert.deepEqual(installLockResources(root), [])

    const locks = probeInstallLocks(root)

    // The shim does not exist, so nothing can hold it.
    assert.deepEqual(locks, { definite: [], shared: [] })
    assert.equal(
      venvHermesShimPath(root),
      path.join(root, 'venv', IS_WINDOWS ? 'Scripts' : 'bin', IS_WINDOWS ? 'hermes.exe' : 'hermes')
    )
  })
})

describe('an unlocked install', () => {
  it('reports no locks or holders without invoking Restart Manager', async () => {
    const root = makeFakeInstall()

    assert.deepEqual(probeInstallLocks(root), { definite: [], shared: [] })
    assert.equal(await isAnyInstallResourceLocked(root), false)
    assert.deepEqual(await attributedInstallHolders(root), [])
    assert.equal(restartManagerMocks.listHolders.mock.calls.length, 0)
  })
})
