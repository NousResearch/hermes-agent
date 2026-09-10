'use strict'

import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { afterEach, describe, it } from 'vitest'

import {
  clearInstallMutationSetCache,
  enumerateInstallMutationSet,
  findLockedInstallResources,
  getInstallMutationSet,
  installShimCandidates,
  probeInstallResourceLocks
} from './install-mutation-set'

function makeInstall(): string {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-mutation-set-'))

  const write = (relative: string) => {
    const target = path.join(root, relative)
    fs.mkdirSync(path.dirname(target), { recursive: true })
    fs.writeFileSync(target, 'x')
  }

  write('venv/Scripts/hermes.exe')
  write('venv/Scripts/python.exe')
  write('venv/Lib/site-packages/yaml/_yaml.cp311-win_amd64.pyd')
  write('venv/Lib/site-packages/yaml/__init__.py')
  write('venv/Lib/site-packages/nacl/libsodium.DLL')
  write('.hermes-runtime/python/generation-1/cpython-3.11.15-windows-x86_64-none/python.exe')
  write('.hermes-runtime/python/generation-1/cpython-3.11.15-windows-x86_64-none/DLLs/_socket.pyd')
  write('.hermes-runtime/python/generation-2/cpython-3.11.15-windows-x86_64-none/python311.dll')
  write('.hermes-runtime/python/generation-2/README.md')
  write('apps/desktop/release/win-unpacked/Hermes.exe')

  return root
}

describe('install mutation set', () => {
  afterEach(() => {
    clearInstallMutationSetCache()
  })

  it('lists shim files first, then every native module, DLL, and exe under venv, never the managed runtime', () => {
    const root = makeInstall()

    try {
      const files = enumerateInstallMutationSet(root)
      const relative = files.map(file => path.relative(root, file).split(path.sep).join('/'))

      assert.deepEqual(relative.slice(0, 2), ['venv/Scripts/hermes.exe', 'venv/Scripts/python.exe'])
      assert.ok(relative.includes('venv/Lib/site-packages/yaml/_yaml.cp311-win_amd64.pyd'))
      assert.ok(relative.includes('venv/Lib/site-packages/nacl/libsodium.DLL'))
      // The managed runtime is not rewritten in place by an update and is
      // shared with foreign uv tool venvs; its holders must not block.
      assert.ok(!relative.some(file => file.startsWith('.hermes-runtime/')))
      // Not part of the venv mutation: Python sources, docs, and the Desktop app itself.
      assert.ok(!relative.some(file => file.endsWith('.py') || file.endsWith('.md')))
      assert.ok(!relative.some(file => file.startsWith('apps/')))
      assert.equal(new Set(relative).size, relative.length)
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })

  it('tolerates a missing runtime directory and a missing shim', () => {
    const root = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-mutation-set-'))

    try {
      fs.mkdirSync(path.join(root, 'venv', 'Lib'), { recursive: true })
      fs.writeFileSync(path.join(root, 'venv', 'Lib', 'a.pyd'), 'x')

      const files = enumerateInstallMutationSet(root)

      assert.deepEqual(files, [path.join(root, 'venv', 'Lib', 'a.pyd')])
      assert.equal(installShimCandidates(root).length, 3)
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })

  it('caches the enumeration for the TTL and re-reads after it', () => {
    const root = makeInstall()

    try {
      let clock = 1_000
      const first = getInstallMutationSet(root, { now: () => clock, ttlMs: 500 })
      fs.writeFileSync(path.join(root, 'venv', 'Lib', 'late.pyd'), 'x')
      clock += 100
      const cached = getInstallMutationSet(root, { now: () => clock, ttlMs: 500 })

      assert.equal(cached, first)
      clock += 1_000
      const refreshed = getInstallMutationSet(root, { now: () => clock, ttlMs: 500 })

      assert.ok(refreshed.some(file => file.endsWith('late.pyd')))
    } finally {
      fs.rmSync(root, { recursive: true, force: true })
    }
  })

  it('reports only files that refuse an exclusive open, treats ENOENT as free, and honours the limit', () => {
    const calls: string[] = []

    const fsImpl = {
      openSync: (target: string) => {
        calls.push(target)

        if (target.endsWith('mapped.pyd') || target.endsWith('running.exe')) {
          throw Object.assign(new Error('EBUSY'), { code: 'EBUSY' })
        }

        if (target.endsWith('gone.dll')) {
          throw Object.assign(new Error('ENOENT'), { code: 'ENOENT' })
        }

        return 7
      },
      closeSync: () => undefined,
      statSync: () => ({ nlink: 1 })
    }

    const resources = ['C:\\i\\free.dll', 'C:\\i\\mapped.pyd', 'C:\\i\\gone.dll', 'C:\\i\\running.exe']

    assert.deepEqual(findLockedInstallResources(resources, { fsImpl, platform: 'win32' }), [
      'C:\\i\\mapped.pyd',
      'C:\\i\\running.exe'
    ])

    calls.length = 0
    assert.deepEqual(findLockedInstallResources(resources, { fsImpl, platform: 'win32', limit: 1 }), [
      'C:\\i\\mapped.pyd'
    ])
    assert.equal(calls.length, 2, 'stops probing once the limit is reached')
    assert.deepEqual(findLockedInstallResources(resources, { fsImpl, platform: 'linux' }), [])
  })

  it('separates locked files shared through extra hard links, which a foreign venv can map without blocking our unlink', () => {
    const stats: string[] = []

    const fsImpl = {
      openSync: (target: string) => {
        if (target.endsWith('free.pyd')) {return 3}

        throw Object.assign(new Error('EBUSY'), { code: 'EBUSY' })
      },
      closeSync: () => undefined,
      statSync: (target: string) => {
        stats.push(target)

        return { nlink: target.endsWith('tokenizers.pyd') ? 3 : 1 }
      }
    }

    const shared = 'C:\\i\\venv\\Lib\\site-packages\\tokenizers\\tokenizers.pyd'
    const own = 'C:\\i\\venv\\Lib\\site-packages\\yaml\\_yaml.pyd'
    const free = 'C:\\i\\venv\\Lib\\site-packages\\free.pyd'

    assert.deepEqual(probeInstallResourceLocks([free, shared, own], { fsImpl, platform: 'win32' }), {
      definite: [own],
      shared: [shared]
    })
    assert.deepEqual(stats, [shared, own], 'link counts are read only for files that refused the open')
    assert.deepEqual(findLockedInstallResources([free, shared, own], { fsImpl, platform: 'win32' }), [own, shared])
    assert.deepEqual(probeInstallResourceLocks([shared, own], { fsImpl, platform: 'linux' }), { definite: [], shared: [] })
  })
})
