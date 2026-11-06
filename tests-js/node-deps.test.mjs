import { execFileSync } from 'node:child_process'
import { chmodSync, cpSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, symlinkSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { delimiter, dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, expect, test } from 'vitest'
import { pnpmCommand, prepareNodeDependencies } from '../scripts/build/node-deps.mjs'

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const roots = []
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true }) })

function json(path, value) {
  mkdirSync(dirname(path), { recursive: true })
  writeFileSync(path, JSON.stringify(value))
}

function vendor(source, name, scripts) {
  json(join(source, 'vendor', name, 'package.json'), { name, version: '1.0.0', main: 'index.cjs', scripts })
  writeFileSync(join(source, 'vendor', name, 'index.cjs'), `module.exports = '${name}'`)
}

// A real pnpm workspace and lockfile. tui-only and web-only are tarballs, which pnpm
// stores and links like registry packages; web-only carries a dependency build. The
// root postinstall counts pnpm installs. Build outputs stay out of the store, as
// the repository's pnpm-workspace.yaml configures.
function fixture() {
  const base = mkdtempSync(join(tmpdir(), 'node union with spaces-'))
  roots.push(base)
  const source = join(base, 'source')
  // The store, metadata cache and state stay in the fixture, never in the user's home.
  const env = { ...process.env, pnpm_config_offline: 'true', pnpm_config_store_dir: join(base, 'store'),
    pnpm_config_cache_dir: join(base, 'cache'), pnpm_config_state_dir: join(base, 'state') }
  vendor(source, 'root-only')
  vendor(source, 'tui-only')
  vendor(source, 'web-only', { install: `node -e "require('node:fs').writeFileSync('built', require('node:crypto').randomUUID())"` })
  for (const name of ['tui-only', 'web-only']) {
    execFileSync(pnpmCommand(), ['pack', '--pack-destination', join(source, 'vendor')], { cwd: join(source, 'vendor', name), env, stdio: 'pipe' })
  }
  json(join(source, 'package.json'), { name: 'fixture', private: true, dependencies: { 'root-only': 'file:vendor/root-only' },
    scripts: { postinstall: `node -e "require('node:fs').appendFileSync('installs', 'x')"` } })
  writeFileSync(join(source, 'pnpm-workspace.yaml'), [
    'packages:', '  - ui-tui', '  - web', '  - apps/desktop', 'sideEffectsCache: false',
    'allowBuilds:', "  'web-only@file:vendor/web-only-1.0.0.tgz': true", '',
  ].join('\n'))
  json(join(source, 'ui-tui/package.json'), { name: 'tui', version: '1.0.0', dependencies: { 'tui-only': 'file:../vendor/tui-only-1.0.0.tgz' } })
  json(join(source, 'web/package.json'), { name: 'web', version: '1.0.0', dependencies: { 'web-only': 'file:../vendor/web-only-1.0.0.tgz' } })
  json(join(source, 'apps/desktop/package.json'), { name: 'desktop', version: '1.0.0', scripts: { postinstall: 'node -e "require(\'fs\').writeFileSync(\'electron-provisioned\',\'yes\')"' } })
  execFileSync(pnpmCommand(), ['install', '--lockfile-only'], { cwd: source, env, stdio: 'pipe' })
  return { source, env }
}

function installs(source) {
  return existsSync(join(source, 'installs')) ? readFileSync(join(source, 'installs'), 'utf8').length : 0
}

function webOnly(source) {
  return dirname(createRequire(join(source, 'web/package.json')).resolve('web-only/package.json'))
}

function cli(source, ...args) {
  return [join(repo, 'scripts/build/node-deps.mjs'), '--source', source, ...args]
}

test('without a pnpm on PATH, the pnpm running the script is used', () => {
  const base = mkdtempSync(join(tmpdir(), 'pnpm-execpath-'))
  roots.push(base)
  const running = join(base, process.platform === 'win32' ? 'pnpm.exe' : 'pnpm')
  writeFileSync(running, '')

  expect(pnpmCommand({ env: { npm_execpath: running, PATH: '' } })).toBe(running)
  // npm's own execpath (npm-cli.js) is not a pnpm: PATH decides, and here it has none.
  expect(() => pnpmCommand({ env: { npm_execpath: join(base, 'npm-cli.js'), PATH: '' } })).toThrow('pnpm is required on PATH')
})

test('selection fails closed on implicit, unknown or missing workspaces', () => {
  const { source, env } = fixture()
  expect(() => prepareNodeDependencies({ source, workspaces: [], env })).toThrow(/workspace/)
  // pnpm itself would ignore a filter that matches nothing.
  expect(() => prepareNodeDependencies({ source, workspaces: ['missing'], env })).toThrow(/workspace/)
  expect(() => prepareNodeDependencies({ source, workspaces: ['.'], env })).toThrow(/workspace/)
  rmSync(join(source, 'apps/desktop'), { recursive: true })
  expect(() => prepareNodeDependencies({ source, workspaces: ['apps/desktop'], env })).toThrow(/workspace/)
  expect(existsSync(join(source, 'node_modules'))).toBe(false)
}, 30000)

test('read-only dependency preparation reuses complete receipts but refuses missing inputs', () => {
  const { source, env } = fixture()
  const options = { source, workspaces: ['web'], reuse: true, env }
  expect(() => prepareNodeDependencies({ ...options, install: false })).toThrow(/disabled/)
  expect(existsSync(join(source, 'node_modules'))).toBe(false)
  prepareNodeDependencies(options)
  const receipt = readFileSync(join(source, 'node_modules/.hermes-node-deps'))
  prepareNodeDependencies({ ...options, install: false })
  prepareNodeDependencies(options)
  expect(installs(source)).toBe(1)
  // A removed store slot leaves a dangling workspace link.
  rmSync(webOnly(source), { recursive: true })
  expect(() => prepareNodeDependencies({ ...options, install: false })).toThrow(/disabled/)
  expect(readFileSync(join(source, 'node_modules/.hermes-node-deps'))).toEqual(receipt)
  prepareNodeDependencies(options)
  rmSync(join(source, 'web/node_modules/web-only'))
  expect(() => prepareNodeDependencies({ ...options, install: false })).toThrow(/disabled/)
}, 30000)

test('reuse refuses a gutted package directory and repairs it', () => {
  const { source, env } = fixture()
  const options = { source, workspaces: ['web'], reuse: true, env }
  prepareNodeDependencies(options)
  // #128935: an external cleanup or snapshot restore can leave the directory
  // while stripping its contents. pnpm trusts its record that the package is
  // installed and would not restore it, so the receipt must reject first.
  const gutted = webOnly(source)
  rmSync(gutted, { recursive: true })
  mkdirSync(gutted)
  expect(() => prepareNodeDependencies({ ...options, install: false })).toThrow(/disabled/)
  prepareNodeDependencies(options)
  expect(createRequire(join(source, 'web/package.json'))('web-only')).toBe('web-only')
}, 30000)

test('native toolchain admission rejects a changed compiler without breaking ordinary builders', () => {
  const { source, env } = fixture()
  const options = { source, workspaces: ['web'], reuse: true, env }
  prepareNodeDependencies({ ...options, nativeToolchain: 'native-first' })
  const built = readFileSync(join(webOnly(source), 'built'), 'utf8')
  prepareNodeDependencies({ ...options, install: false })
  prepareNodeDependencies({ ...options, nativeToolchain: 'native-first', install: false })
  expect(() => prepareNodeDependencies({ ...options, nativeToolchain: 'native-second', install: false })).toThrow(/disabled/)
  expect(readFileSync(join(webOnly(source), 'built'), 'utf8')).toBe(built)
  execFileSync(process.execPath, cli(source, '--workspace', 'web', '--reuse', '--native-toolchain', 'native-second'), { env, stdio: 'pipe' })
  // The dependency build ran again under the new toolchain.
  expect(readFileSync(join(webOnly(source), 'built'), 'utf8')).not.toBe(built)
  prepareNodeDependencies({ ...options, nativeToolchain: 'native-second', install: false })
  prepareNodeDependencies({ ...options, install: false })
}, 30000)

test('one locked preparation retains the requested union without provisioning desktop', () => {
  const { source, env } = fixture()
  const lock = readFileSync(join(source, 'pnpm-lock.yaml'))
  prepareNodeDependencies({ source, workspaces: ['ui-tui', 'web', 'ui-tui'], env })
  for (const [workspace, name] of [['ui-tui', 'tui-only'], ['web', 'web-only']]) {
    expect(createRequire(join(source, workspace, 'package.json'))(name)).toBe(name)
  }
  expect(createRequire(join(source, 'package.json'))('root-only')).toBe('root-only')
  expect(existsSync(join(source, 'apps/desktop/node_modules'))).toBe(false)
  expect(existsSync(join(source, 'apps/desktop/electron-provisioned'))).toBe(false)
  expect(readFileSync(join(source, 'pnpm-lock.yaml'))).toEqual(lock)
  execFileSync(process.execPath, cli(source, '--workspace', 'ui-tui', '--workspace', 'web'), { cwd: tmpdir(), env, stdio: 'pipe' })
  expect(createRequire(join(source, 'web/package.json'))('web-only')).toBe('web-only')
}, 30000)

test('electron: false installs Desktop without preparing its Electron binary', () => {
  const { source, env } = fixture()
  json(join(source, 'vendor/electron/package.json'), { name: 'electron', version: '1.0.0' })
  json(join(source, 'apps/desktop/package.json'), { name: 'desktop', version: '1.0.0', devDependencies: { electron: 'file:../../vendor/electron' } })
  execFileSync(pnpmCommand(), ['install', '--lockfile-only'], { cwd: source, env, stdio: 'pipe' })
  // Electron preparation runs after the install: this fake is not the version Desktop pins.
  expect(() => prepareNodeDependencies({ source, workspaces: ['apps/desktop'], env })).toThrow(/lock-pinned/)
  prepareNodeDependencies({ source, workspaces: ['apps/desktop'], env, electron: false })
  expect(existsSync(join(source, 'apps/desktop/node_modules/electron/package.json'))).toBe(true)
}, 30000)

test('web-only selection excludes siblings and rejects stale locks without rewriting them', () => {
  const { source, env } = fixture()
  rmSync(join(source, 'apps/desktop'), { recursive: true })
  prepareNodeDependencies({ source, workspaces: ['web'], env })
  expect(existsSync(join(source, 'ui-tui/node_modules'))).toBe(false)
  expect(existsSync(join(source, 'apps/desktop/electron-provisioned'))).toBe(false)
  const lock = readFileSync(join(source, 'pnpm-lock.yaml'))
  json(join(source, 'web/package.json'), { name: 'web', version: '1.0.0', dependencies: { 'web-only': '9.0.0' } })
  expect(() => prepareNodeDependencies({ source, workspaces: ['web'], env })).toThrow()
  expect(readFileSync(join(source, 'pnpm-lock.yaml'))).toEqual(lock)
}, 30000)

test('a restored completed install retains postinstall outputs only for matching inputs', () => {
  const { source, env } = fixture()
  prepareNodeDependencies({ source, workspaces: ['ui-tui', 'web'], env, reuse: true })
  const artifact = 'node_modules/postinstall-output'
  writeFileSync(join(source, artifact), 'built')
  const restored = join(mkdtempSync(join(tmpdir(), 'restored node union-')), 'source')
  roots.push(dirname(restored))
  cpSync(source, restored, { recursive: true, verbatimSymlinks: true })
  prepareNodeDependencies({ source: restored, workspaces: ['web', 'ui-tui'], env, reuse: true })
  expect(readFileSync(join(restored, artifact), 'utf8')).toBe('built')
  execFileSync(process.execPath, cli(restored, '--workspace', 'web', '--workspace', 'ui-tui', '--reuse'), { cwd: tmpdir(), env, stdio: 'pipe' })
  expect(readFileSync(join(restored, artifact), 'utf8')).toBe('built')
  expect(createRequire(join(restored, 'web/package.json'))('web-only')).toBe('web-only')

  prepareNodeDependencies({ source: restored, workspaces: ['web'], env, reuse: true })
  expect(existsSync(join(restored, artifact))).toBe(false)
  expect(existsSync(join(restored, 'ui-tui/node_modules'))).toBe(false)
  writeFileSync(join(restored, artifact), 'built again')
  // Even an invalid manifest must reach pnpm, not hide behind a warm receipt.
  json(join(restored, 'web/package.json'), { name: 'web', version: '1.0.0', dependencies: { 'web-only': '9.0.0' } })
  expect(() => prepareNodeDependencies({ source: restored, workspaces: ['web'], env, reuse: true })).toThrow()
  cpSync(join(source, 'web/package.json'), join(restored, 'web/package.json'))
  prepareNodeDependencies({ source: restored, workspaces: ['web'], env, reuse: true })
  expect(existsSync(join(restored, artifact))).toBe(false)
}, 30000)

test('reuse respects lifecycle configuration and repairs missing installed packages', () => {
  const { source, env } = fixture()
  const options = { source, workspaces: ['web'], env, reuse: true }
  const marker = () => readFileSync(join(webOnly(source), 'built'), 'utf8')
  prepareNodeDependencies({ ...options, env: { ...env, pnpm_config_ignore_scripts: 'true' } })
  expect(existsSync(join(webOnly(source), 'built'))).toBe(false)
  expect(installs(source)).toBe(0)
  prepareNodeDependencies(options)
  const first = marker()
  prepareNodeDependencies(options)
  expect(marker()).toBe(first)
  rmSync(webOnly(source), { recursive: true })
  prepareNodeDependencies(options)
  const repaired = marker()
  expect(repaired).not.toBe(first)
  expect(createRequire(join(source, 'web/package.json'))('web-only')).toBe('web-only')
  prepareNodeDependencies({ ...options, reuse: false })
  expect(marker()).not.toBe(repaired)
}, 30000)

test('pnpm configuration name casing does not invalidate a completed install', () => {
  const fixtureEnv = fixture()
  const { source } = fixtureEnv
  const env = Object.fromEntries(Object.entries(fixtureEnv.env).filter(([key]) => key.toLowerCase() !== 'pnpm_config_child_concurrency'))
  const options = { source, workspaces: ['web'], reuse: true }
  prepareNodeDependencies({ ...options, env: { ...env, pnpm_config_child_concurrency: '2' } })
  // The Windows runner sets lowercase; Python's os.environ returns uppercase.
  prepareNodeDependencies({ ...options, env: { ...env, PNPM_CONFIG_CHILD_CONCURRENCY: '2' } })
  expect(installs(source)).toBe(1)
  // Normalize names, not values: a genuinely changed setting still reinstalls.
  prepareNodeDependencies({ ...options, env: { ...env, PNPM_CONFIG_CHILD_CONCURRENCY: '3' } })
  expect(installs(source)).toBe(2)
}, 30000)

test('the receipt is keyed on the lockfile and the workspace settings', () => {
  const { source, env } = fixture()
  const options = { source, workspaces: ['web'], reuse: true, env }
  prepareNodeDependencies(options)
  writeFileSync(join(source, 'pnpm-workspace.yaml'), `${readFileSync(join(source, 'pnpm-workspace.yaml'), 'utf8')}childConcurrency: 2\n`)
  expect(() => prepareNodeDependencies({ ...options, install: false })).toThrow(/disabled/)
  prepareNodeDependencies(options)
  prepareNodeDependencies({ ...options, install: false })
  writeFileSync(join(source, 'pnpm-lock.yaml'), `${readFileSync(join(source, 'pnpm-lock.yaml'), 'utf8')}\n`)
  expect(() => prepareNodeDependencies({ ...options, install: false })).toThrow(/disabled/)
  prepareNodeDependencies(options)
  expect(installs(source)).toBe(3)
}, 30000)

test('an npm-installed tree is removed before the first pnpm install', () => {
  const { source, env } = fixture()
  // npm's hoisted layout: its hidden lockfile, root-level packages, nested workspace trees.
  json(join(source, 'node_modules/.package-lock.json'), { packages: {} })
  json(join(source, 'node_modules/web-only/package.json'), { name: 'web-only', version: '0.9.0' })
  json(join(source, 'web/node_modules/npm-only/package.json'), { name: 'npm-only', version: '1.0.0' })
  prepareNodeDependencies({ source, workspaces: ['web'], env })
  expect(readdirSync(join(source, 'node_modules')).filter(name => name.startsWith('.ignored'))).toEqual([])
  expect(readdirSync(join(source, 'web/node_modules')).filter(name => name.startsWith('.ignored'))).toEqual([])
  expect(existsSync(join(source, 'node_modules/.package-lock.json'))).toBe(false)
  expect(existsSync(join(source, 'web/node_modules/npm-only'))).toBe(false)
  expect(createRequire(join(source, 'web/package.json'))('web-only')).toBe('web-only')
}, 30000)

// A stand-in pnpm on PATH that records each command and delegates to the real one.
// `--version` probes exit with `probeExit`: the Job-Object failure lane from #123933,
// where the version must come from the manifest beside the resolved executable.
function fakePnpm({ manifest }) {
  const fake = mkdtempSync(join(tmpdir(), 'pnpm layout with spaces-'))
  roots.push(fake)
  const spawned = join(fake, 'spawned')
  mkdirSync(join(fake, 'real'))
  mkdirSync(join(fake, 'bin'))
  const executable = join(fake, 'real/pnpm')
  const arm = ({ probeExit }) => {
    writeFileSync(executable, [
      '#!/bin/sh',
      `printf '%s\\n' "$1" >> '${spawned}'`,
      `if [ "$1" = --version ]; then echo 12.0.0; exit ${probeExit}; fi`,
      `exec '${pnpmCommand()}' "$@"`, '',
    ].join('\n'))
    chmodSync(executable, 0o755)
  }
  const writeManifest = version => json(join(fake, 'real/package.json'), { name: 'pnpm', version })
  if (manifest) writeManifest(manifest)
  arm({ probeExit: manifest ? 3 : 0 })
  // The link's directory holds no manifest; the resolved executable's does.
  symlinkSync(executable, join(fake, 'bin/pnpm'))
  return { path: join(fake, 'bin'), spawned, arm, writeManifest }
}

test.skipIf(process.platform === 'win32')('the pnpm version reads from the manifest beside the resolved executable without a child spawn', () => {
  const { source, env } = fixture()
  const pnpm = fakePnpm({ manifest: '12.0.0' })
  const options = { source, workspaces: ['web'], reuse: true, env: { ...env, PATH: `${pnpm.path}${delimiter}${env.PATH}` } }
  prepareNodeDependencies(options)
  prepareNodeDependencies(options)
  // Both runs reached the install or the reuse short-circuit; no version probe spawned.
  expect(readFileSync(pnpm.spawned, 'utf8')).toBe('install\n')
}, 30000)

test.skipIf(process.platform === 'win32')('a pnpm layout without a readable manifest falls back to the child version probe', () => {
  const { source, env } = fixture()
  const pnpm = fakePnpm({ manifest: null })
  prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env: { ...env, PATH: `${pnpm.path}${delimiter}${env.PATH}` } })
  expect(readFileSync(pnpm.spawned, 'utf8')).toBe('--version\ninstall\n')
}, 30000)

// The receipt key hashes the pnpm version. An install completed by the probe lane
// must still count as complete once a manifest supplies the same version, without
// depending on a probe that can no longer spawn.
test.skipIf(process.platform === 'win32')('a receipt keyed by the probed version reuses under the manifest version, even when the probe fails', () => {
  const { source, env } = fixture()
  const pnpm = fakePnpm({ manifest: null })
  const options = { source, workspaces: ['web'], reuse: true, env: { ...env, PATH: `${pnpm.path}${delimiter}${env.PATH}` } }
  prepareNodeDependencies(options)
  expect(readFileSync(pnpm.spawned, 'utf8')).toBe('--version\ninstall\n')
  pnpm.writeManifest('12.0.0')
  pnpm.arm({ probeExit: 1 })
  prepareNodeDependencies({ ...options, install: false })
  expect(readFileSync(pnpm.spawned, 'utf8')).toBe('--version\ninstall\n')
  // A genuinely different pnpm still invalidates the receipt.
  pnpm.writeManifest('12.0.1')
  expect(() => prepareNodeDependencies({ ...options, install: false })).toThrow(/disabled/)
}, 30000)

// pmOnFail: ignore keeps the pin out of the lockfile, so pnpm itself never compares it.
test.skipIf(process.platform === 'win32')('a pnpm other than the root devEngines pin is refused before anything is installed', () => {
  const { source, env } = fixture()
  const manifest = JSON.parse(readFileSync(join(source, 'package.json'), 'utf8'))
  json(join(source, 'package.json'), { ...manifest, devEngines: { packageManager: { name: 'pnpm', version: '12.0.1', onFail: 'error' } } })
  writeFileSync(join(source, 'pnpm-workspace.yaml'), `${readFileSync(join(source, 'pnpm-workspace.yaml'), 'utf8')}pmOnFail: ignore\n`)
  const pnpm = fakePnpm({ manifest: '12.0.0' })
  const options = { source, workspaces: ['web'], env: { ...env, PATH: `${pnpm.path}${delimiter}${env.PATH}` } }
  expect(() => prepareNodeDependencies(options)).toThrow(/pnpm 12\.0\.1 is required .*found 12\.0\.0/)
  expect(existsSync(pnpm.spawned)).toBe(false)
  pnpm.writeManifest('12.0.1')
  prepareNodeDependencies(options)
  expect(readFileSync(pnpm.spawned, 'utf8')).toBe('install\n')
}, 30000)
