import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'

import { test } from 'vitest'

interface PackageManifest {
  engines?: Record<string, string>
}

interface PackageLock {
  packages?: Record<string, PackageManifest>
}

const REPO_ROOT = path.resolve(__dirname, '..')

function readJson<T>(relativePath: string): T {
  return JSON.parse(fs.readFileSync(path.join(REPO_ROOT, relativePath), 'utf-8')) as T
}

test('Node engine declarations and lockfile mirrors stay aligned', () => {
  const root = readJson<PackageManifest>('package.json')
  const desktop = readJson<PackageManifest>('apps/desktop/package.json')
  const lock = readJson<PackageLock>('package-lock.json')
  const rootNode = root.engines?.node

  assert.ok(rootNode, 'root package.json must declare engines.node')
  assert.equal(desktop.engines?.node, rootNode)
  assert.equal(lock.packages?.['']?.engines?.node, rootNode)
  assert.equal(lock.packages?.['apps/desktop']?.engines?.node, rootNode)
})

test('.nvmrc selects the declared Node engine major', () => {
  const root = readJson<PackageManifest>('package.json')
  const rootNode = root.engines?.node
  const floorMajor = Number(rootNode?.match(/\d+/)?.[0])
  const nvmMajor = Number(fs.readFileSync(path.join(REPO_ROOT, '.nvmrc'), 'utf-8').trim())

  assert.ok(Number.isInteger(floorMajor), 'engines.node must contain a major version')
  assert.equal(nvmMajor, floorMajor)
})
