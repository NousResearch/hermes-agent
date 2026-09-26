import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

// semver ships no declarations in this lock; type the two library calls used here.
const semver: { valid(version: string): string | null; satisfies(version: string, range: string): boolean } = require('semver')
import { test } from 'vitest'

interface Engines { node?: string; npm?: string }
interface Manifest { engines?: Engines }
interface Lockfile { packages: Record<string, Manifest> }
interface PmLock { packages: { node: { version: string }; npm: { version: string } } }

function readJson<T>(relativePath: string): T {
  // SAFETY: these are repository-owned manifest/lock inputs, checked below.
  return JSON.parse(fs.readFileSync(path.resolve(__dirname, '..', relativePath), 'utf8')) as T
}

const root = readJson<Manifest>('package.json')
const desktop = readJson<Manifest>('apps/desktop/package.json')
const lock = readJson<Lockfile>('package-lock.json')
const pins = readJson<PmLock>('pm/lock.json').packages

function nodeRange(manifest: Manifest): string {
  assert.ok(manifest.engines?.node, 'workspace must declare engines.node')

  return manifest.engines.node
}

test.each([
  ['22.22.0', true], ['22.23.1', true], ['24.11.0', true], ['24.18.2', true], ['26.0.0', true],
  ['22.21.1', false], ['23.0.0', false], ['24.0.0', false], ['24.10.9', false], ['25.2.1', false], ['26.0.0-rc.1', false],
] as const)('workspace Node policy for %s is %s', (version: string, accepted: boolean): void => {
  for (const manifest of [root, desktop]) {
    assert.equal(semver.satisfies(version, nodeRange(manifest)), accepted)
  }
})

test.each([
  ['10.9.8', true], ['11.9.9', true], ['11.10.0', false], ['11.12.1', false],
  ['11.16.9', false], ['11.17.0', true], ['12.0.2', true],
] as const)('npm age-exclusion policy for %s is %s', (version: string, accepted: boolean): void => {
  assert.ok(root.engines?.npm)
  assert.equal(semver.satisfies(version, root.engines.npm), accepted)
})

test('exact independently managed Node and npm satisfy all declared lockfile engines', (): void => {
  for (const tool of ['node', 'npm'] as const) {
    assert.ok(semver.valid(pins[tool].version), `${tool} must be an exact pin`)
  }

  for (const [name, manifest] of Object.entries({ ...lock.packages, root, desktop })) {
    for (const tool of ['node', 'npm'] as const) {
      const range = manifest.engines?.[tool]

      if (range) {
        assert.ok(semver.satisfies(pins[tool].version, range), `${name}: pinned ${tool} ${pins[tool].version} violates ${range}`)
      }
    }
  }
})

test('lockfile workspace engine mirrors match their manifests', (): void => {
  assert.deepEqual(lock.packages[''].engines, root.engines)
  assert.deepEqual(lock.packages['apps/desktop'].engines, desktop.engines)
})
