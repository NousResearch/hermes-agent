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

function parseVersion(version: string): [number, number, number] {
  const [major = 0, minor = 0, patch = 0] = version.split('-', 1)[0].split('.').map(Number)
  return [major, minor, patch]
}

function compare(left: string, right: string): number {
  const have = parseVersion(left)
  const want = parseVersion(right)

  for (let index = 0; index < have.length; index += 1) {
    if (have[index] !== want[index]) {
      return have[index] - want[index]
    }
  }
  return 0
}

function satisfiesClause(version: string, clause: string): boolean {
  assert.match(clause, /^(?:\^|>=|<=|>|<|=)?\d+(?:\.\d+){0,2}$/, `unsupported semver clause: ${clause}`)
  if (clause.startsWith('^')) {
    const bound = clause.slice(1)
    return parseVersion(version)[0] === parseVersion(bound)[0] && compare(version, bound) >= 0
  }

  const match = clause.match(/^(>=|<=|>|<|=)?(.+)$/)
  assert.ok(match)
  const [, operator = '=', bound] = match
  const result = compare(version, bound)
  return operator === '>='
    ? result >= 0
    : operator === '<='
      ? result <= 0
      : operator === '>'
        ? result > 0
        : operator === '<'
          ? result < 0
          : result === 0
}

function satisfiesRange(version: string, range: string): boolean {
  return range.split('||').some(alternative =>
    alternative
      .trim()
      .split(/\s+/)
      .every(clause => satisfiesClause(version, clause))
  )
}

const rootManifest = readJson<Manifest>('package.json')
const desktopManifest = readJson<Manifest>('apps/desktop/package.json')
const lockfile = readJson<Lockfile>('package-lock.json')

function nodeRange(manifest: Manifest, label: string): string {
  assert.ok(manifest.engines?.node, `${label} must declare engines.node`)
  return manifest.engines.node
}

describe('Node engine alignment', () => {
  const rootRange = nodeRange(rootManifest, 'root package.json')
  const desktopRange = nodeRange(desktopManifest, 'apps/desktop/package.json')

  test.each(['22.22.0', '22.23.1', '24.0.0', '26.0.0'])('all workspace manifests accept supported Node %s', version => {
    assert.ok(satisfiesRange(version, rootRange))
    assert.ok(satisfiesRange(version, desktopRange))
  })

  test.each(['22.21.1', '23.0.0', '25.2.1'])(
    'all workspace manifests reject dependency-incompatible Node %s',
    version => {
      assert.ok(!satisfiesRange(version, rootRange))
      assert.ok(!satisfiesRange(version, desktopRange))
    }
  )

  test('lockfile workspace mirrors match their manifests', () => {
    assert.equal(nodeRange(lockfile.packages?.[''] ?? {}, 'root lock entry'), rootRange)
    assert.equal(nodeRange(lockfile.packages?.['apps/desktop'] ?? {}, 'desktop lock entry'), desktopRange)
  })
})
