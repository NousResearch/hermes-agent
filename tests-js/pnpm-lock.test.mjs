import { execFileSync } from 'node:child_process'
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { loadAll } from 'js-yaml'
import { afterEach, expect, test } from 'vitest'
import { pnpmCommand } from '../scripts/build/node-deps.mjs'
import { lockImporters } from '../scripts/build/pnpm-lock.mjs'

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const roots = []
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true }) })

function json(path, value) {
  mkdirSync(dirname(path), { recursive: true })
  writeFileSync(path, JSON.stringify(value))
}

function parsedImporters(text) {
  return loadAll(text)[0].importers
}

// A lock that records a pnpm pin opens with a document of its own.
const pinDocument = `---
lockfileVersion: '9.0'

importers:

  .:
    configDependencies: {}
    packageManagerDependencies:
      pnpm:
        specifier: 12.15.0
        version: 12.15.0

---
`

// A real pnpm lock with a peer-resolved dependency, a scoped (quoted) name and a
// project without dependencies.
function generatedLock() {
  const source = mkdtempSync(join(tmpdir(), 'pnpm lock with spaces-'))
  roots.push(source)
  const env = { ...process.env, pnpm_config_offline: 'true', pnpm_config_store_dir: join(source, '.store'),
    pnpm_config_cache_dir: join(source, '.cache'), pnpm_config_state_dir: join(source, '.state') }
  const packages = { peer: {}, 'with-peer': { peerDependencies: { peer: '1.0.0' } }, '@scope/scoped': {} }
  for (const [name, fields] of Object.entries(packages)) {
    const dir = join(source, 'vendor', name)
    json(join(dir, 'package.json'), { name, version: '1.0.0', ...fields })
    execFileSync(pnpmCommand(), ['pack', '--pack-destination', join(source, 'vendor')], { cwd: dir, env, stdio: 'pipe' })
  }
  json(join(source, 'package.json'), { name: 'fixture', private: true, dependencies: {
    peer: 'file:vendor/peer-1.0.0.tgz', 'with-peer': 'file:vendor/with-peer-1.0.0.tgz' } })
  json(join(source, 'apps/scoped/package.json'), { name: 'scoped-user', devDependencies: { '@scope/scoped': 'file:../../vendor/scope-scoped-1.0.0.tgz' } })
  json(join(source, 'apps/empty/package.json'), { name: 'empty' })
  writeFileSync(join(source, 'pnpm-workspace.yaml'), 'packages:\n  - apps/*\n')
  execFileSync(pnpmCommand(), ['install', '--lockfile-only'], { cwd: source, env, stdio: 'pipe' })
  return readFileSync(join(source, 'pnpm-lock.yaml'), 'utf8')
}

test('reads the workspace importers as a YAML parser does', () => {
  const text = generatedLock()
  const importers = lockImporters(text)
  expect(importers).toEqual(parsedImporters(text))
  expect(Object.keys(importers).sort()).toEqual(['.', 'apps/empty', 'apps/scoped'])
  // A peer-resolved version keeps pnpm's suffix verbatim; callers strip it.
  expect(importers['.'].dependencies['with-peer'].version).toMatch(/^file:vendor\/with-peer-1\.0\.0\.tgz\(peer@/)
  expect(importers['apps/scoped'].devDependencies['@scope/scoped'].specifier).toBe('file:../../vendor/scope-scoped-1.0.0.tgz')
})

test('reads the repository lock as a YAML parser does', () => {
  const text = readFileSync(join(repo, 'pnpm-lock.yaml'), 'utf8')
  expect(lockImporters(text)).toEqual(parsedImporters(text))
})

test('refuses a lock it does not understand instead of guessing', () => {
  const text = generatedLock()
  expect(() => lockImporters(text.replace("lockfileVersion: '9.0'", "lockfileVersion: '6.0'"))).toThrow(/lockfileVersion 6\.0/)
  expect(() => lockImporters(text.replace(/^lockfileVersion: .*$/m, ''))).toThrow(/lockfileVersion <missing>/)
  expect(() => lockImporters(text.replace(/^importers:$/m, 'projects:'))).toThrow(/no importers/)
  expect(() => lockImporters(text.replace(/^ {6}with-peer:$/m, '      - with-peer'))).toThrow(/importers line/)
  // Scanners read only the first document: a recorded pin would hide the workspace from them.
  expect(() => lockImporters(`${pinDocument}${text}`)).toThrow(/one YAML document, found 2/)
})

test('the repository lock is a single YAML document', () => {
  const text = readFileSync(join(repo, 'pnpm-lock.yaml'), 'utf8')
  expect(loadAll(text)).toHaveLength(1)
})
