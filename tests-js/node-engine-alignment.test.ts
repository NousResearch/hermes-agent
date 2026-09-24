import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { describe, test } from 'vitest'

const REPO_ROOT = path.resolve(__dirname, '..')

interface Manifest {
  engines?: { node?: string }
}

interface Lockfile {
  packages?: Record<string, Manifest>
}

function readJson<T>(relativePath: string): T {
  return JSON.parse(fs.readFileSync(path.join(REPO_ROOT, relativePath), 'utf-8')) as T
}

const rootManifest = readJson<Manifest>('package.json')
const desktopManifest = readJson<Manifest>('apps/desktop/package.json')
const lockfile = readJson<Lockfile>('package-lock.json')

function nodeRange(manifest: Manifest, label: string): string {
  assert.ok(manifest.engines?.node, `${label} must declare engines.node`)

  return manifest.engines.node
}

describe('Node engine alignment', () => {
  test('lockfile workspace mirrors match their manifests', () => {
    assert.equal(nodeRange(lockfile.packages?.[''] ?? {}, 'root lock entry'), nodeRange(rootManifest, 'root package.json'))
    assert.equal(
      nodeRange(lockfile.packages?.['apps/desktop'] ?? {}, 'desktop lock entry'),
      nodeRange(desktopManifest, 'apps/desktop/package.json')
    )
  })
})
