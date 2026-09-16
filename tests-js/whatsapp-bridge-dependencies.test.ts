/**
 * Security contract for the standalone WhatsApp bridge dependency graph.
 *
 * Express owns body-parser. A bridge-local override previously pinned
 * body-parser inside its vulnerable range and kept qs vulnerable on every
 * reinstall. The lockfile also carries Baileys' optional sharp peer.
 */

import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { test } from 'vitest'

const REPO_ROOT = path.resolve(__dirname, '..')
const BRIDGE = path.join(REPO_ROOT, 'scripts', 'whatsapp-bridge')

interface Manifest {
  dependencies?: Record<string, string>
  overrides?: Record<string, string>
}

interface LockPackage {
  version?: string
}

interface Lockfile {
  packages?: Record<string, LockPackage>
}

function readJson<T>(filename: string): T {
  return JSON.parse(fs.readFileSync(path.join(BRIDGE, filename), 'utf-8')) as T
}

function versionAtLeast(actual: string, minimum: string): boolean {
  const have = actual.split('-', 1)[0].split('.').map(Number)
  const want = minimum.split('-', 1)[0].split('.').map(Number)

  for (let index = 0; index < 3; index += 1) {
    if (have[index] !== want[index]) {
      return have[index] > want[index]
    }
  }

  return true
}

function assertResolvedAtLeast(packages: Record<string, LockPackage>, name: string, minimum: string): void {
  const versions = Object.entries(packages)
    .filter(([installPath]) => installPath === `node_modules/${name}` || installPath.endsWith(`/node_modules/${name}`))
    .map(([, metadata]) => metadata.version)
    .filter((version): version is string => Boolean(version))

  assert.ok(versions.length > 0, `${name} must resolve in the WhatsApp bridge lockfile`)
  assert.ok(
    versions.every(version => versionAtLeast(version, minimum)),
    `${name} must resolve at or above ${minimum}; found ${versions.join(', ')}`
  )
}

test('WhatsApp bridge resolutions stay outside the mapped advisory ranges', () => {
  const manifest = readJson<Manifest>('package.json')
  const lockfile = readJson<Lockfile>('package-lock.json')
  const packages = lockfile.packages ?? {}
  const lockRoot = packages[''] as Manifest | undefined

  assert.equal(lockRoot?.dependencies?.express, manifest.dependencies?.express)
  assert.ok(
    !Object.prototype.hasOwnProperty.call(manifest.overrides ?? {}, 'body-parser'),
    'Express must own the body-parser range; do not restore a bridge-local pin'
  )

  assertResolvedAtLeast(packages, 'express', '4.22.3')
  assertResolvedAtLeast(packages, 'body-parser', '1.20.8')
  assertResolvedAtLeast(packages, 'qs', '6.16.0')
  assertResolvedAtLeast(packages, 'sharp', '0.35.4')
})
