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

import { describe, it } from 'vitest'

import {
  attributedInstallHolders,
  createInstallLockGateProbeForRoot,
  installLockResources,
  isAnyInstallResourceLocked,
  probeInstallLocks,
  venvHermesShimPath
} from './install-lock-probe'

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
  it('reports no locks, no holders, and an open gate', async () => {
    const root = makeFakeInstall()

    assert.deepEqual(probeInstallLocks(root), { definite: [], shared: [] })
    assert.equal(await isAnyInstallResourceLocked(root), false)
    assert.equal(await createInstallLockGateProbeForRoot(root)(), false)
  })

  it('answers without spawning a Restart Manager child', async () => {
    const root = makeFakeInstall()
    const startedAt = Date.now()
    const holders = await attributedInstallHolders(root)

    assert.deepEqual(holders, [])
    // The Restart Manager path is a PowerShell child with a multi-second
    // budget. Returning inside a second is the observable proof that the
    // no-lock short circuit fired before it.
    assert.ok(Date.now() - startedAt < 1_000, `attribution took ${Date.now() - startedAt}ms`)
  })
})
