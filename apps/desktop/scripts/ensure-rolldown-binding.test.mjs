import assert from 'node:assert/strict'
import { test } from 'vitest'

import { ensureRolldownBinding, selectRolldownBinding } from './ensure-rolldown-binding.mjs'

const packageMetadata = {
  optionalDependencies: {
    '@rolldown/binding-darwin-arm64': '1.2.1',
    '@rolldown/binding-darwin-x64': '1.2.1',
    '@rolldown/binding-linux-x64-gnu': '1.2.1',
    '@rolldown/binding-linux-x64-musl': '1.2.1'
  }
}

test('selects the exact platform binding from Rolldown metadata', () => {
  assert.deepEqual(selectRolldownBinding(packageMetadata.optionalDependencies, 'darwin', 'arm64'), [
    '@rolldown/binding-darwin-arm64',
    '1.2.1'
  ])
})

test('does nothing when Rolldown already loads', () => {
  let installs = 0
  const ok = ensureRolldownBinding({
    root: '/repo',
    probe: () => ({ status: 0 }),
    install: () => {
      installs += 1
      return { status: 0 }
    }
  })

  assert.equal(ok, true)
  assert.equal(installs, 0)
})

test('installs the missing macOS binding and verifies the repair', () => {
  let probes = 0
  const installs = []
  const ok = ensureRolldownBinding({
    root: '/repo',
    platform: 'darwin',
    arch: 'arm64',
    probe: () => ({ status: probes++ === 0 ? 1 : 0 }),
    install: (_root, spec) => {
      installs.push(spec)
      return { status: 0 }
    },
    findPackage: () => '/repo/node_modules/rolldown/package.json',
    readPackage: () => packageMetadata
  })

  assert.equal(ok, true)
  assert.deepEqual(installs, ['@rolldown/binding-darwin-arm64@1.2.1'])
  assert.equal(probes, 2)
})

test('fails without guessing when the platform binding is ambiguous', () => {
  let installs = 0
  const ok = ensureRolldownBinding({
    root: '/repo',
    platform: 'linux',
    arch: 'x64',
    probe: () => ({ status: 1 }),
    install: () => {
      installs += 1
      return { status: 0 }
    },
    findPackage: () => '/repo/node_modules/rolldown/package.json',
    readPackage: () => packageMetadata
  })

  assert.equal(ok, false)
  assert.equal(installs, 0)
})
