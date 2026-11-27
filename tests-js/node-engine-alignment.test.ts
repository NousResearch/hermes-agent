import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { load } from 'js-yaml'
// semver ships no declarations in this lock; type the two library calls used here.
const semver: { valid(version: string): string | null; satisfies(version: string, range: string): boolean } = require('semver')
import { test } from 'vitest'

interface Engines { node?: string }
interface Manifest { engines?: Engines }
interface Lockfile { packages?: Record<string, Manifest> }
interface PmLock { packages: { node: { version: string } } }

function read(relativePath: string): string {
  return fs.readFileSync(path.resolve(__dirname, '..', relativePath), 'utf8')
}

// SAFETY: these are repository-owned manifest/lock inputs, checked below.
const root = JSON.parse(read('package.json')) as Manifest
const desktop = JSON.parse(read('apps/desktop/package.json')) as Manifest
const lock = load(read('pnpm-lock.yaml')) as Lockfile
const pins = (JSON.parse(read('pm/lock.json')) as PmLock).packages

function nodeRange(manifest: Manifest): string {
  assert.ok(manifest.engines?.node, 'workspace must declare engines.node')

  return manifest.engines.node
}

test.each([
  ['26.0.0', true], ['26.12.1', true], ['27.0.0', true],
  ['22.22.0', false], ['22.23.1', false], ['24.11.0', false], ['24.18.2', false], ['25.2.1', false], ['26.0.0-rc.1', false],
] as const)('workspace Node policy for %s is %s', (version: string, accepted: boolean): void => {
  for (const manifest of [root, desktop]) {
    assert.equal(semver.satisfies(version, nodeRange(manifest)), accepted)
  }
})

test('exact independently managed Node satisfies all declared lockfile engines', (): void => {
  assert.ok(semver.valid(pins.node.version), 'node must be an exact pin')
  assert.ok(Object.keys(lock.packages ?? {}).length > 0, 'pnpm-lock.yaml has no packages')

  for (const [name, manifest] of Object.entries({ ...lock.packages, root, desktop })) {
    const range = manifest.engines?.node

    if (range) {
      assert.ok(semver.satisfies(pins.node.version, range), `${name}: pinned node ${pins.node.version} violates ${range}`)
    }
  }
})
