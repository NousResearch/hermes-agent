import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, readdirSync, rmSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { dirname, join } from 'node:path'
import { afterEach, expect, test, vi } from 'vitest'

// A node_modules that is a mount point (a container volume) cannot be renamed, nor moved to a
// .cache on another filesystem. Fault exactly the renames of the trees a test names.
const faults = vi.hoisted(() => new Map())

vi.mock('node:fs', async importOriginal => {
  const actual = await importOriginal()
  return {
    ...actual,
    renameSync(from, to) {
      const code = faults.get(from)
      if (code) throw Object.assign(new Error(`${code}: rename '${from}' -> '${to}'`), { code })
      return actual.renameSync(from, to)
    },
  }
})

const { pnpmCommand, prepareNodeDependencies } = await import('../scripts/build/node-deps.mjs')

const roots = []
afterEach(() => {
  faults.clear()
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

function json(path, value) {
  mkdirSync(dirname(path), { recursive: true })
  writeFileSync(path, JSON.stringify(value))
}

function fixture() {
  const base = mkdtempSync(join(tmpdir(), 'node-deps cross device-'))
  roots.push(base)
  const source = join(base, 'source')
  const env = { ...process.env, pnpm_config_offline: 'true', pnpm_config_store_dir: join(base, 'store'),
    pnpm_config_cache_dir: join(base, 'cache'), pnpm_config_state_dir: join(base, 'state') }
  json(join(source, 'vendor/web-only/package.json'), { name: 'web-only', version: '1.0.0', main: 'index.cjs' })
  writeFileSync(join(source, 'vendor/web-only/index.cjs'), "module.exports = 'web-only'")
  json(join(source, 'package.json'), { name: 'fixture', private: true })
  writeFileSync(join(source, 'pnpm-workspace.yaml'), 'packages:\n  - web\n')
  json(join(source, 'web/package.json'), { name: 'web', version: '1.0.0', dependencies: { 'web-only': 'file:../vendor/web-only' } })
  execFileSync(pnpmCommand(), ['install', '--lockfile-only'], { cwd: source, env, stdio: 'pipe' })
  // PM's pnpm pin, the one node-deps requires.
  const version = execFileSync(pnpmCommand(), ['--version'], { cwd: source, env, encoding: 'utf8' }).trim()
  json(join(source, 'pm/lock.json'), { schema: 1, packages: { pnpm: { version, artifacts: {} } } })
  return { source, env }
}

// Windows reports an open file as EBUSY: deleting around it would gut the tree, so it stays a failure there.
test.each(process.platform === 'win32' ? ['EXDEV'] : ['EXDEV', 'EBUSY'])('a tree that cannot be moved aside (%s) is emptied in place and reinstalled', code => {
  const { source, env } = fixture()
  const options = { source, workspaces: ['web'], reuse: true, env }
  prepareNodeDependencies(options)
  rmSync(join(source, 'node_modules/.hermes-node-deps'))
  writeFileSync(join(source, 'web/node_modules/left-behind'), '')
  faults.set(join(source, 'web/node_modules'), code)
  prepareNodeDependencies(options)
  expect(existsSync(join(source, 'web/node_modules/left-behind'))).toBe(false)
  expect(createRequire(join(source, 'web/package.json'))('web-only')).toBe('web-only')
  prepareNodeDependencies({ ...options, install: false })
  expect(readdirSync(join(source, '.cache'))).toEqual([])
}, 30000)

// After a successful install the previous set is only garbage: a delete Windows refuses (an
// antivirus or indexer handle) must not fail the run; the next run's recovery removes it.
test('a previous set that cannot be deleted after a successful install is left for the next run', () => {
  const { source, env } = fixture()
  const options = { source, workspaces: ['web'], reuse: true, env }
  prepareNodeDependencies(options)
  rmSync(join(source, 'node_modules/.hermes-node-deps'))
  faults.set(join(source, '.cache/node-deps-previous'), 'EPERM')
  prepareNodeDependencies(options)
  expect(createRequire(join(source, 'web/package.json'))('web-only')).toBe('web-only')
  faults.clear()
  prepareNodeDependencies({ ...options, install: false })
  expect(readdirSync(join(source, '.cache'))).toEqual([])
}, 30000)
