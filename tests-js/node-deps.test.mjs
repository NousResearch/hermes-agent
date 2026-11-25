import { execFileSync, spawn, spawnSync } from 'node:child_process'
import { chmodSync, cpSync, existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, symlinkSync, utimesSync, writeFileSync } from 'node:fs'
import fs from 'node:fs'
import { createRequire, syncBuiltinESMExports } from 'node:module'
import { hostname, tmpdir } from 'node:os'
import { delimiter, dirname, join, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'
import { afterEach, expect, test } from 'vitest'
import { lockTiming } from '../scripts/build/node-deps-lock.mjs'
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
  pinPnpm(source, execFileSync(pnpmCommand(), ['--version'], { cwd: source, env, encoding: 'utf8' }).trim())
  return { source, env }
}

// PM's pnpm pin, the one node-deps requires.
function pinPnpm(source, version) {
  json(join(source, 'pm/lock.json'), { schema: 1, packages: { pnpm: { version, artifacts: {} } } })
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
  // The failed install put the completed one back; its inputs match again, so it is reused.
  cpSync(join(source, 'web/package.json'), join(restored, 'web/package.json'))
  prepareNodeDependencies({ source: restored, workspaces: ['web'], env, reuse: true })
  expect(readFileSync(join(restored, artifact), 'utf8')).toBe('built again')
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

// A source launch lowers pnpm's retries (someone is waiting, and it can fall back): the trees an
// update installed with the defaults must still be reused there.
test('the download retry policy does not invalidate a completed install', () => {
  const { source, env } = fixture()
  const options = { source, workspaces: ['web'], reuse: true }
  prepareNodeDependencies({ ...options, env })
  prepareNodeDependencies({ ...options, env: { ...env, pnpm_config_fetch_retries: '1' }, install: false })
  expect(installs(source)).toBe(1)
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

// #150031: an update onto a new lockfile while the registry is unreachable. The
// first pnpm install has an empty store, so nothing new can be fetched; the
// previous install must survive the failure, and a later online run must succeed.
test('a failed install restores the previously installed trees', () => {
  const { source, env } = fixture()
  const options = { source, workspaces: ['ui-tui', 'web'], reuse: true, env }
  prepareNodeDependencies(options)
  const receipt = readFileSync(join(source, 'node_modules/.hermes-node-deps'))
  const lock = readFileSync(join(source, 'pnpm-lock.yaml'), 'utf8')
  const manifest = readFileSync(join(source, 'web/package.json'))
  json(join(source, 'web/package.json'), { name: 'web', version: '1.0.0',
    dependencies: { 'web-only': 'file:../vendor/web-only-1.0.0.tgz', picocolors: '1.1.1' } })
  writeFileSync(join(source, 'pnpm-lock.yaml'), lock
    .replace('  web:\n    dependencies:\n', '  web:\n    dependencies:\n      picocolors:\n        specifier: 1.1.1\n        version: 1.1.1\n')
    .replace('packages:\n\n', 'packages:\n\n  picocolors@1.1.1:\n    resolution: {integrity: sha512-xceH2snhtb5M9liqDsmEw56le376mTZkEX/jEb/RxNFyegNul7eNslCXP9FDj/Lcu0X8KEyMceP2ntpaHrDEVA==}\n\n')
    .replace('snapshots:\n\n', 'snapshots:\n\n  picocolors@1.1.1: {}\n\n'))
  const dead = { ...env, pnpm_config_offline: 'false', pnpm_config_registry: 'http://127.0.0.1:9/',
    pnpm_config_fetch_retries: '0', pnpm_config_store_dir: join(dirname(source), 'empty-store') }
  expect(() => execFileSync(process.execPath, cli(source, '--workspace', 'ui-tui', '--workspace', 'web', '--reuse'),
    { env: dead, stdio: 'pipe' })).toThrow(/ERR_PNPM_TARBALL_FETCH/)
  for (const [workspace, name] of [['ui-tui', 'tui-only'], ['web', 'web-only']]) {
    expect(createRequire(join(source, workspace, 'package.json'))(name)).toBe(name)
  }
  expect(createRequire(join(source, 'package.json'))('root-only')).toBe('root-only')
  // The restored receipt still names the old lockfile, so it is never reused for the new one.
  expect(readFileSync(join(source, 'node_modules/.hermes-node-deps'))).toEqual(receipt)
  expect(() => prepareNodeDependencies({ ...options, env: dead, install: false })).toThrow(/disabled/)
  expect(existsSync(join(source, '.cache/node-deps-previous'))).toBe(false)
  // Back on the old inputs, the restored install is complete as it was.
  writeFileSync(join(source, 'pnpm-lock.yaml'), lock)
  writeFileSync(join(source, 'web/package.json'), manifest)
  prepareNodeDependencies({ ...options, install: false })
  expect(installs(source)).toBe(1)
}, 30000)

// A pnpm on PATH that announces each install in `started`, then waits `delay`
// seconds before the real install runs: a slow download, or a window to kill in.
function slowPnpm(delay) {
  const dir = mkdtempSync(join(tmpdir(), 'slow pnpm-'))
  roots.push(dir)
  const started = join(dir, 'started')
  writeFileSync(join(dir, 'pnpm'), ['#!/bin/sh',
    // Backgrounded and trapped like the signalled shim below: a Ctrl-C landing between the marker
    // and a foreground sleep's exec is caught by sh's own handler in the forked child, so the
    // sleep would run its full delay and the job would outlive the signal.
    `if [ "$1" = install ]; then trap 'kill $! 2>/dev/null; exit 130' INT; echo "$$" >> '${started}'; sleep ${delay} & wait $!; trap - INT; fi`,
    `exec '${pnpmCommand()}' "$@"`, ''].join('\n'))
  chmodSync(join(dir, 'pnpm'), 0o755)
  json(join(dir, 'package.json'), { name: 'pnpm', version: execFileSync(pnpmCommand(), ['--version']).toString().trim() })
  return { started, env: env => ({ ...env, PATH: `${dir}${delimiter}${env.PATH}` }) }
}

async function until(check) {
  for (let tries = 0; !check(); tries++) {
    if (tries > 300) throw new Error('timed out')
    await new Promise(done => setTimeout(done, 50))
  }
}

// The run and every child (the pnpm shim) in one process group, killed together.
function background(source, env, ...args) {
  const child = spawn(process.execPath, cli(source, ...args), { env, stdio: 'pipe', detached: true })
  let output = ''
  child.stdout.on('data', data => { output += data })
  child.stderr.on('data', data => { output += data })
  const exit = new Promise(done => child.on('exit', (code, signal) => done({ code, signal, output })))
  return {
    exit,
    kill: () => process.kill(-child.pid, 'SIGKILL'),
    // As Ctrl-Z and `fg` do: the whole job.
    stop: () => process.kill(-child.pid, 'SIGSTOP'),
    resume: () => process.kill(-child.pid, 'SIGCONT'),
    signal: (name, group) => process.kill(group ? -child.pid : child.pid, name),
    // Anything of the job (the run, pnpm) still running.
    alive: () => { try { process.kill(-child.pid, 0); return true } catch { return false } },
  }
}

// New web inputs (as after a pull): web gains extra-only.
function addExtraOnly(source, env) {
  vendor(source, 'extra-only')
  json(join(source, 'web/package.json'), { name: 'web', version: '1.0.0',
    dependencies: { 'web-only': 'file:../vendor/web-only-1.0.0.tgz', 'extra-only': 'file:../vendor/extra-only' } })
  execFileSync(pnpmCommand(), ['install', '--lockfile-only'], { cwd: source, env, stdio: 'pipe' })
}

test.skipIf(process.platform === 'win32')('a run killed mid-install gets the previous trees back on the next run', async () => {
  const { source, env } = fixture()
  const options = { source, workspaces: ['web'], reuse: true, env }
  prepareNodeDependencies(options)
  const old = { lock: readFileSync(join(source, 'pnpm-lock.yaml')), manifest: readFileSync(join(source, 'web/package.json')) }
  addExtraOnly(source, env)
  const pnpm = slowPnpm(30)
  const run = background(source, pnpm.env(env), '--workspace', 'web', '--reuse')
  await until(() => existsSync(pnpm.started))
  run.kill()
  expect((await run.exit).signal).toBe('SIGKILL')
  // A launch on the old inputs (the pull rolled back): the trees that were in place are back.
  writeFileSync(join(source, 'pnpm-lock.yaml'), old.lock)
  writeFileSync(join(source, 'web/package.json'), old.manifest)
  prepareNodeDependencies({ ...options, install: false })
  expect(createRequire(join(source, 'web/package.json'))('web-only')).toBe('web-only')
  expect(readdirSync(join(source, '.cache'))).toEqual([])
  expect(installs(source)).toBe(1)
}, 60000)

// A read-only launch beside an install must neither swap trees under it nor let it certify
// trees it did not write.
test.skipIf(process.platform === 'win32')('a reuse check waits for an install running in the same checkout', async () => {
  const { source, env } = fixture()
  prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env })
  addExtraOnly(source, env)
  const pnpm = slowPnpm(2)
  const install = background(source, pnpm.env(env), '--workspace', 'web', '--reuse')
  await until(() => existsSync(pnpm.started))
  const reader = background(source, env, '--workspace', 'web', '--reuse', '--no-install')
  const { code, output } = await reader.exit
  expect((await install.exit).code).toBe(0)
  // It waited, then reused what the install completed.
  expect(output).toMatch(/waiting up to 10 min for another install in this checkout/)
  expect(output).toMatch(/reusing completed install/)
  expect(code).toBe(0)
  expect(createRequire(join(source, 'web/package.json'))('extra-only')).toBe('extra-only')
  expect(readdirSync(join(source, '.cache'))).toEqual([])
}, 60000)

// A run that waits for the lock reads its inputs once it holds it: an update that changed the
// dependencies meanwhile is installed, not judged against the files read before the wait.
test.skipIf(process.platform === 'win32')('a run that waited for the lock installs the inputs current when it got it', async () => {
  const { source, env } = fixture()
  prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env })
  // A live owner: a lock this test keeps fresh, as the owner's heartbeat would.
  const lock = join(source, '.cache/node-deps.lock')
  json(lock, { token: 'live', pid: process.pid, host: hostname() })
  const beat = setInterval(() => { const now = new Date(); utimesSync(lock, now, now) }, 200)
  const waiter = spawn(process.execPath, cli(source, '--workspace', 'web', '--reuse'), { env, stdio: 'pipe' })
  let output = ''
  waiter.stdout.on('data', data => { output += data })
  waiter.stderr.on('data', data => { output += data })
  const exit = new Promise(done => waiter.on('exit', done))
  try {
    await until(() => output.includes('waiting up to'))
    addExtraOnly(source, env)
  } finally {
    clearInterval(beat)
    rmSync(lock, { force: true })
  }
  expect(await exit).toBe(0)
  expect(output).not.toMatch(/reusing completed install/)
  expect(createRequire(join(source, 'web/package.json'))('extra-only')).toBe('extra-only')
  prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env, install: false })
}, 30000)

// A lock nobody refreshes is dead whatever it names: a killed run's pid now reused by another
// user's process, another host's name (a container, a renamed Mac) or another PID namespace.
test.each([
  ['a pid reused by another process', { pid: 1, host: hostname() }],
  ['another host', { pid: 4242, host: 'devcontainer-3f1a9c' }],
  ['the current process', { pid: process.pid, host: hostname() }],
])('a lock left by a killed run is broken: %s', (_, owner) => {
  const { source, env } = fixture()
  mkdirSync(join(source, '.cache'), { recursive: true })
  writeFileSync(join(source, '.cache/node-deps.lock'), JSON.stringify(owner))
  const timing = { ...lockTiming }
  Object.assign(lockTiming, { stale: 500 })
  try {
    prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env })
  } finally {
    Object.assign(lockTiming, timing)
  }
  expect(createRequire(join(source, 'web/package.json'))('web-only')).toBe('web-only')
  expect(readdirSync(join(source, '.cache'))).toEqual([])
}, 30000)

// A live owner is waited for, but not forever: a launch must fail (and fall back) rather than
// hang, and the line says what to do before anything a narrow terminal would cut.
test.skipIf(process.platform === 'win32')('waiting for a live install gives up with the remedy first', async () => {
  const { source, env } = fixture()
  prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env })
  addExtraOnly(source, env)
  const pnpm = slowPnpm(30)
  const install = background(source, pnpm.env(env), '--workspace', 'web', '--reuse')
  await until(() => existsSync(pnpm.started))
  const timing = { ...lockTiming }
  // Longer than the stale limit, so the heartbeat is what keeps the live owner's lock.
  Object.assign(lockTiming, { stale: 1_500, wait: 4_000 })
  try {
    expect(() => prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env, install: false }))
      .toThrow(/^node-deps: another install in this checkout is still running after 4 s; run again once it finishes/)
  } finally {
    Object.assign(lockTiming, timing)
    install.kill()
    await install.exit
  }
}, 60000)

// An owner stopped (Ctrl-Z) past the stale limit loses its lock to another run. Resumed while
// that run still holds the lock, it must leave the trees and the receipts to it: whether its own
// pnpm failed (a restore would put the old trees over the new owner's) or succeeded (a receipt
// would certify the tree the new owner built).
test.skipIf(process.platform === 'win32').each([['failed', 'rc=1'], ['succeeded', `'${pnpmCommand()}' "$@"; rc=$?`]])(
  'a run resumed after losing its lock touches nothing: its pnpm %s', async (_, install) => {
    const { source, env } = fixture()
    prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env, nativeToolchain: 'first' })
    const dir = dirname(source)
    const shim = (name, body) => {
      mkdirSync(join(dir, name))
      writeFileSync(join(dir, name, 'pnpm'), ['#!/bin/sh', `if [ "$1" = install ]; then ${body}; fi`, `exec '${pnpmCommand()}' "$@"`, ''].join('\n'))
      chmodSync(join(dir, name, 'pnpm'), 0o755)
      json(join(dir, name, 'package.json'), { name: 'pnpm', version: execFileSync(pnpmCommand(), ['--version']).toString().trim() })
      return { ...env, PATH: `${join(dir, name)}${delimiter}${env.PATH}` }
    }
    // A: its install stops A's run (the process waiting for pnpm), then fails or succeeds once resumed.
    const ownerEnv = shim('owner', `${install}; echo $PPID > '${dir}/owner.pid'; kill -STOP $PPID; exit $rc`)
    // A's heartbeat worker (it inherits the preload) is slowed past the test, so nothing but the
    // lock file itself can tell A that it lost the lock: after a stop, the beat lags the main thread.
    const slow = join(dir, 'slow-heartbeat.cjs')
    writeFileSync(slow, `if (!require('node:worker_threads').isMainThread) {
      const setInterval = globalThis.setInterval
      globalThis.setInterval = (fn, ms, ...args) => setInterval(fn, Math.max(ms, 600_000), ...args)
    }`)
    // Run under a shell that reaps it, since this process blocks in the new owner's pnpm.
    const owner = spawn('/bin/sh', ['-c', '"$@"; echo $? > "$EXIT_FILE"', 'sh', process.execPath,
      '--require', slow, ...cli(source, '--workspace', 'web', '--reuse', '--native-toolchain', 'third')],
    { env: { ...ownerEnv, EXIT_FILE: join(dir, 'owner.exit') }, stdio: 'pipe', detached: true })
    let output = ''
    owner.stdout.on('data', data => { output += data })
    owner.stderr.on('data', data => { output += data })
    await until(() => existsSync(join(dir, 'owner.pid')))
    // B: breaks the stale lock and installs; once its pnpm is done, and while it still holds the
    // lock, it resumes A and waits for A to exit.
    const takeover = shim('takeover', [`'${pnpmCommand()}' "$@" || exit 1`,
      `cp web/node_modules/web-only/built '${dir}/takeover.built'`,
      `kill -CONT "$(cat '${dir}/owner.pid')"`,
      `while [ ! -s '${dir}/owner.exit' ]; do sleep 0.1; done`,
      // What A left behind, before B goes on: a receipt or a restored tree would be A's.
      `cp web/node_modules/web-only/built '${dir}/after-owner.built'`,
      `ls node_modules/.hermes-node-deps* > '${dir}/after-owner.receipts' 2>/dev/null; exit 0`].join('; '))
    const timing = { ...lockTiming }
    Object.assign(lockTiming, { stale: 1_500 })
    try {
      prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env: takeover, nativeToolchain: 'second' })
    } finally {
      Object.assign(lockTiming, timing)
    }
    await new Promise(done => owner.on('exit', done))
    expect(readFileSync(join(dir, 'owner.exit'), 'utf8').trim()).not.toBe('0')
    expect(output).toMatch(/this run lost the checkout lock/)
    expect(output).not.toMatch(/restored the previously installed/)
    // A left B's fresh tree in place and wrote no receipt over it.
    expect(readFileSync(join(dir, 'after-owner.built'), 'utf8')).toBe(readFileSync(join(dir, 'takeover.built'), 'utf8'))
    expect(readFileSync(join(dir, 'after-owner.receipts'), 'utf8')).toBe('')
    // The tree is the one B built, and only B's toolchain is certified.
    expect(readFileSync(join(source, 'web/node_modules/web-only/built'), 'utf8')).toBe(readFileSync(join(dir, 'takeover.built'), 'utf8'))
    expect(readFileSync(join(source, 'node_modules/.hermes-node-deps.native-toolchain'), 'utf8')).toMatch(/"second"\n$/)
    expect(readdirSync(join(source, '.cache'))).toEqual([])
    prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env, nativeToolchain: 'second', install: false })
    expect(() => prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env, nativeToolchain: 'third', install: false })).toThrow(/disabled/)
  }, 60000)

// Ctrl-Z stops the whole job, pnpm with it, and once resumed pnpm went on where it stopped: by
// then the run that broke the stopped run's lock may have installed and certified its own trees.
// The lock watcher is not in the job, so it kills the stopped pnpm as soon as the lock is broken.
test.skipIf(process.platform === 'win32')('an owner stopped during its pnpm install cannot write into the next owner\'s trees', async () => {
  const { source, env } = fixture()
  prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env, nativeToolchain: 'first' })
  const dir = dirname(source)
  mkdirSync(join(dir, 'owner'))
  // A's pnpm: stopped (with all of A) in the middle of its work, which goes on into the tree once resumed.
  writeFileSync(join(dir, 'owner/pnpm'), ['#!/bin/sh', 'if [ "$1" = install ]; then',
    `  touch '${dir}/owner.started'; sleep 2`,
    `  echo third > node_modules/.written-by-the-stopped-pnpm; '${pnpmCommand()}' "$@"; exit $?`,
    'fi', `exec '${pnpmCommand()}' "$@"`, ''].join('\n'))
  chmodSync(join(dir, 'owner/pnpm'), 0o755)
  json(join(dir, 'owner/package.json'), { name: 'pnpm', version: execFileSync(pnpmCommand(), ['--version']).toString().trim() })
  const owner = background(source, { ...env, PATH: `${join(dir, 'owner')}${delimiter}${env.PATH}` },
    '--workspace', 'web', '--reuse', '--native-toolchain', 'third')
  await until(() => existsSync(join(dir, 'owner.started')))
  owner.stop()
  const timing = { ...lockTiming }
  Object.assign(lockTiming, { stale: 1_500 })
  try {
    prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env, nativeToolchain: 'second' })
  } finally {
    Object.assign(lockTiming, timing)
  }
  const built = readFileSync(join(webOnly(source), 'built'), 'utf8')
  owner.resume()
  const { code, output } = await owner.exit
  // Long enough for the stopped pnpm to have finished, had it survived.
  await new Promise(done => setTimeout(done, 3_000))
  expect(code).not.toBe(0)
  expect(output).toMatch(/this run lost the checkout lock/)
  expect(existsSync(join(source, 'node_modules/.written-by-the-stopped-pnpm'))).toBe(false)
  // The tree is the one the receipt certifies.
  expect(readFileSync(join(webOnly(source), 'built'), 'utf8')).toBe(built)
  expect(readFileSync(join(source, 'node_modules/.hermes-node-deps.native-toolchain'), 'utf8')).toMatch(/"second"\n$/)
  expect(readdirSync(join(source, '.cache'))).toEqual([])
  prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env, nativeToolchain: 'second', install: false })
}, 60000)

// The watcher leaves the job, pnpm does not: the terminal's Ctrl-C (SIGINT to the job's process
// group) still reaches pnpm, and so does a SIGTERM aimed at the run alone, which pnpm never got.
test.skipIf(process.platform === 'win32').each([['Ctrl-C reaches the job', 'group', 'SIGINT'], ['a SIGTERM reaches the run alone', 'run', 'SIGTERM']])(
  'pnpm gets the signal when %s', async (_, target, signal) => {
    const { source, env } = fixture()
    prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env })
    addExtraOnly(source, env)
    const dir = mkdtempSync(join(tmpdir(), 'signalled pnpm-'))
    roots.push(dir)
    writeFileSync(join(dir, 'pnpm'), ['#!/bin/sh', 'if [ "$1" = install ]; then',
      `  trap 'echo INT >> "${dir}/got"; kill $!; exit 130' INT`, `  trap 'echo TERM >> "${dir}/got"; kill $!; exit 143' TERM`,
      `  touch '${dir}/started'; sleep 30 & wait $!; exit 1`,
      'fi', `exec '${pnpmCommand()}' "$@"`, ''].join('\n'))
    chmodSync(join(dir, 'pnpm'), 0o755)
    json(join(dir, 'package.json'), { name: 'pnpm', version: execFileSync(pnpmCommand(), ['--version']).toString().trim() })
    const run = background(source, { ...env, PATH: `${dir}${delimiter}${env.PATH}` }, '--workspace', 'web', '--reuse')
    await until(() => existsSync(join(dir, 'started')))
    run.signal(signal, target === 'group')
    await run.exit
    await until(() => existsSync(join(dir, 'got')))
    expect(readFileSync(join(dir, 'got'), 'utf8')).toBe(`${signal.slice(3)}\n`)
    // Nothing of the run is left, and the next run gets the previous trees back.
    await until(() => !run.alive())
    expect(() => prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env, install: false })).toThrow(/disabled/)
    expect(createRequire(join(source, 'web/package.json'))('web-only')).toBe('web-only')
    expect(readdirSync(join(source, '.cache'))).toEqual([])
  }, 60000)

// Ctrl-C (or closing the terminal) during the install is the common way out of a slow or offline
// launch: the run puts the previous trees back and releases the lock itself, so the next run does
// not wait out the stale limit. The CLI exits as the signal would have.
test.skipIf(process.platform === 'win32').each([['SIGINT', 130], ['SIGHUP', 129]])('%s to the job during the install leaves no lock and the previous trees in place', async (signal, status) => {
  const { source, env } = fixture()
  prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env })
  addExtraOnly(source, env)
  const pnpm = slowPnpm(30)
  const run = background(source, pnpm.env(env), '--workspace', 'web', '--reuse')
  await until(() => existsSync(pnpm.started))
  run.signal(signal, true)
  const { code, output } = await run.exit
  await until(() => !run.alive())
  expect(output).toMatch(/restored the previously installed dependencies/)
  expect(code).toBe(status)
  expect(readdirSync(join(source, '.cache'))).toEqual([])
  expect(createRequire(join(source, 'web/package.json'))('web-only')).toBe('web-only')
}, 30000)

// The installer starts only once its lock watcher has: a watcher that cannot start (a process or
// descriptor limit) must not leave pnpm writing into the trees the failed run restores.
test.skipIf(process.platform === 'win32')('an install whose lock watcher cannot start never runs pnpm', async () => {
  const { source, env } = fixture()
  prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env })
  addExtraOnly(source, env)
  // Loaded by the relay: starting the watcher (node, given the lock's path) fails as a spawn would.
  const fault = join(dirname(source), 'watcher-spawn-fault.cjs')
  writeFileSync(fault, `
    const cp = require('node:child_process')
    const spawn = cp.spawn
    cp.spawn = (file, args = [], ...rest) => file === process.execPath && args.some(arg => String(arg).endsWith('node-deps.lock'))
      ? spawn('/nonexistent/node-for-the-watcher', args, ...rest)
      : spawn(file, args, ...rest)
  `)
  const pnpm = slowPnpm(1)
  const run = background(source, { ...pnpm.env(env), NODE_OPTIONS: `--require ${JSON.stringify(fault)}` }, '--workspace', 'web', '--reuse')
  const { code, output } = await run.exit
  try {
    expect(code).not.toBe(0)
    expect(output).toMatch(/cannot start the lock watcher/)
    expect(existsSync(pnpm.started)).toBe(false)
    expect(output).toMatch(/restored the previously installed dependencies/)
    expect(readdirSync(join(source, '.cache'))).toEqual([])
    expect(createRequire(join(source, 'web/package.json'))('web-only')).toBe('web-only')
  } finally {
    if (run.alive()) run.kill()
  }
}, 30000)

// A heartbeat that fails once (an antivirus or indexer handle on Windows, ESTALE on NFS, EMFILE)
// keeps beating: a waiter must not break the lock of an install that is still running.
test.skipIf(process.platform === 'win32')('a transient heartbeat error does not let another run break a live lock', async () => {
  const { source, env } = fixture()
  prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env })
  addExtraOnly(source, env)
  // Loaded in the heartbeat worker too: its second and third refreshes fail.
  const fault = join(dirname(source), 'utimes-fault.cjs')
  writeFileSync(fault, `
    const fs = require('node:fs')
    if (!require('node:worker_threads').isMainThread) {
      const utimesSync = fs.utimesSync
      let beats = 0
      fs.utimesSync = (...args) => {
        if (++beats === 2 || beats === 3) throw Object.assign(new Error('EBUSY: resource busy or locked, utime'), { code: 'EBUSY' })
        return utimesSync(...args)
      }
    }
  `)
  const pnpm = slowPnpm(30)
  const install = spawn(process.execPath, ['--require', fault, ...cli(source, '--workspace', 'web', '--reuse')],
    { env: pnpm.env(env), stdio: 'pipe', detached: true })
  await until(() => existsSync(pnpm.started))
  const timing = { ...lockTiming }
  Object.assign(lockTiming, { stale: 3_000, wait: 8_000 })
  try {
    expect(() => prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env, install: false }))
      .toThrow(/^node-deps: another install in this checkout is still running after 8 s/)
  } finally {
    Object.assign(lockTiming, timing)
    process.kill(-install.pid, 'SIGKILL')
  }
}, 60000)

// A dead lock whose replacement keeps failing (an open handle on Windows) is waited on like a live
// one: the wait still gives up at its limit instead of spinning.
test('a dead lock that cannot be replaced still ends the wait', () => {
  const { source, env } = fixture()
  json(join(source, '.cache/node-deps.lock'), { token: 'dead', pid: 1, host: 'elsewhere' })
  const rename = fs.renameSync
  fs.renameSync = (from, to) => {
    if (String(to).endsWith('node-deps.lock')) throw Object.assign(new Error('EPERM: operation not permitted, rename'), { code: 'EPERM' })
    return rename(from, to)
  }
  syncBuiltinESMExports()
  const timing = { ...lockTiming }
  Object.assign(lockTiming, { stale: 1_000, wait: 3_000 })
  try {
    expect(() => prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env, install: false }))
      .toThrow(/^node-deps: another install in this checkout is still running after 3 s/)
  } finally {
    Object.assign(lockTiming, timing)
    fs.renameSync = rename
    syncBuiltinESMExports()
  }
}, 20000)

// A guard left by a killed remover is judged by whether it changes, not by its mtime: one dated
// in the future (a network filesystem's clock ahead, or this clock stepped back) still ages out.
test('a leaked lock guard dated in the future does not block forever', () => {
  const { source, env } = fixture()
  mkdirSync(join(source, '.cache'), { recursive: true })
  writeFileSync(join(source, '.cache/node-deps.lock'), JSON.stringify({ pid: 4242, host: 'gone' }))
  writeFileSync(join(source, '.cache/node-deps.lock.guard'), '')
  const future = new Date(Date.now() + 10 * 60_000)
  utimesSync(join(source, '.cache/node-deps.lock.guard'), future, future)
  const timing = { ...lockTiming }
  Object.assign(lockTiming, { stale: 500, wait: 20_000 })
  try {
    prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env })
  } finally {
    Object.assign(lockTiming, timing)
  }
  expect(createRequire(join(source, 'web/package.json'))('web-only')).toBe('web-only')
  expect(readdirSync(join(source, '.cache'))).toEqual([])
}, 30000)

// The record of a moved set is renamed into place: a run killed while writing it (or out of
// space) leaves nothing a later run fails to parse.
test.skipIf(process.platform === 'win32')('a run killed while recording the moved trees does not break later runs', () => {
  const { source, env } = fixture()
  prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env, nativeToolchain: 'first' })
  const preload = join(dirname(source), 'kill-in-record.mjs')
  writeFileSync(preload, `
    import fs from 'node:fs'
    import { syncBuiltinESMExports } from 'node:module'
    const writeFileSync = fs.writeFileSync
    fs.writeFileSync = (path, data, options) => {
      if (String(path).includes('trees.json')) { writeFileSync(path, ''); process.kill(process.pid, 'SIGKILL') }
      return writeFileSync(path, data, options)
    }
    syncBuiltinESMExports()
  `)
  const run = spawnSync(process.execPath, ['--import', pathToFileURL(preload).href,
    ...cli(source, '--workspace', 'web', '--reuse', '--native-toolchain', 'second')], { env, stdio: 'pipe' })
  expect(run.signal).toBe('SIGKILL')
  prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env, install: false })
  expect(createRequire(join(source, 'web/package.json'))('web-only')).toBe('web-only')
  expect(readdirSync(join(source, '.cache'))).toEqual([])
}, 60000)

// The recovery text after a killed run tells the user to run pnpm install themselves.
test.skipIf(process.platform === 'win32')('a run killed mid-install does not restore its old trees over a newer install', async () => {
  const { source, env } = fixture()
  prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env })
  addExtraOnly(source, env)
  const pnpm = slowPnpm(30)
  const run = background(source, pnpm.env(env), '--workspace', 'web', '--reuse')
  await until(() => existsSync(pnpm.started))
  run.kill()
  await run.exit
  execFileSync(pnpmCommand(), ['install', '--frozen-lockfile'], { cwd: source, env, stdio: 'pipe' })
  expect(() => prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env, install: false })).toThrow(/disabled/)
  expect(createRequire(join(source, 'web/package.json'))('extra-only')).toBe('extra-only')
  expect(readdirSync(join(source, '.cache'))).toEqual([])
}, 60000)

// The previous trees are deleted after a successful install, and that delete can be killed
// halfway (a slow disk, antivirus): the half-deleted copy must never come back.
test.skipIf(process.platform === 'win32')('a run killed while deleting the previous trees keeps the fresh ones', () => {
  const { source, env } = fixture()
  prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env, nativeToolchain: 'first' })
  // At the first recursive delete under .cache: delete one file of the previous trees, then die.
  const preload = join(dirname(source), 'kill-in-delete.mjs')
  writeFileSync(preload, `
    import fs from 'node:fs'
    import { syncBuiltinESMExports } from 'node:module'
    import { join } from 'node:path'
    const rmSync = fs.rmSync
    const find = dir => { for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
      const path = join(dir, entry.name)
      if (entry.isFile() && entry.name === 'index.cjs' && path.includes('web-only')) return path
      if (entry.isDirectory()) { const found = find(path); if (found) return found } } }
    fs.rmSync = (path, options) => {
      if (options?.recursive && String(path).startsWith(process.env.KILL_UNDER) && fs.existsSync(path)) {
        const victim = find(String(path))
        if (victim) { rmSync(victim); process.kill(process.pid, 'SIGKILL') }
      }
      return rmSync(path, options)
    }
    syncBuiltinESMExports()
  `)
  // A changed native toolchain reinstalls on the same inputs.
  const run = spawnSync(process.execPath, ['--import', pathToFileURL(preload).href,
    ...cli(source, '--workspace', 'web', '--reuse', '--native-toolchain', 'second')],
  { env: { ...env, KILL_UNDER: join(source, '.cache') }, stdio: 'pipe' })
  expect(run.signal).toBe('SIGKILL')
  prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env, install: false })
  expect(execFileSync(process.execPath, ['-e', "console.log(require('web-only'))"], { cwd: join(source, 'web') }).toString().trim()).toBe('web-only')
  expect(readdirSync(join(source, '.cache'))).toEqual([])
}, 60000)

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
  pinPnpm(source, '12.0.0')
  const pnpm = fakePnpm({ manifest: '12.0.0' })
  const options = { source, workspaces: ['web'], reuse: true, env: { ...env, PATH: `${pnpm.path}${delimiter}${env.PATH}` } }
  prepareNodeDependencies(options)
  prepareNodeDependencies(options)
  // Both runs reached the install or the reuse short-circuit; no version probe spawned.
  expect(readFileSync(pnpm.spawned, 'utf8')).toBe('install\n')
}, 30000)

test.skipIf(process.platform === 'win32')('a pnpm layout without a readable manifest falls back to the child version probe', () => {
  const { source, env } = fixture()
  pinPnpm(source, '12.0.0')
  const pnpm = fakePnpm({ manifest: null })
  prepareNodeDependencies({ source, workspaces: ['web'], reuse: true, env: { ...env, PATH: `${pnpm.path}${delimiter}${env.PATH}` } })
  expect(readFileSync(pnpm.spawned, 'utf8')).toBe('--version\ninstall\n')
}, 30000)

// The receipt key hashes the pnpm version. An install completed by the probe lane
// must still count as complete once a manifest supplies the same version, without
// depending on a probe that can no longer spawn.
test.skipIf(process.platform === 'win32')('a receipt keyed by the probed version reuses under the manifest version, even when the probe fails', () => {
  const { source, env } = fixture()
  pinPnpm(source, '12.0.0')
  const pnpm = fakePnpm({ manifest: null })
  const options = { source, workspaces: ['web'], reuse: true, env: { ...env, PATH: `${pnpm.path}${delimiter}${env.PATH}` } }
  prepareNodeDependencies(options)
  expect(readFileSync(pnpm.spawned, 'utf8')).toBe('--version\ninstall\n')
  pnpm.writeManifest('12.0.0')
  pnpm.arm({ probeExit: 1 })
  prepareNodeDependencies({ ...options, install: false })
  expect(readFileSync(pnpm.spawned, 'utf8')).toBe('--version\ninstall\n')
  // A genuinely different pnpm still invalidates the receipt.
  pinPnpm(source, '12.0.1')
  pnpm.writeManifest('12.0.1')
  expect(() => prepareNodeDependencies({ ...options, install: false })).toThrow(/disabled/)
}, 30000)

// pmOnFail: ignore keeps the pin out of the lockfile, so pnpm itself never compares it.
test.skipIf(process.platform === 'win32')('a pnpm other than the pm/lock.json pin is refused before anything is installed', () => {
  const { source, env } = fixture()
  pinPnpm(source, '12.0.1')
  const pnpm = fakePnpm({ manifest: '12.0.0' })
  const options = { source, workspaces: ['web'], reuse: true, env: { ...env, PATH: `${pnpm.path}${delimiter}${env.PATH}` } }
  expect(() => prepareNodeDependencies(options)).toThrow(/pnpm 12\.0\.1 is required \(pm\/lock\.json\).*found 12\.0\.0/)
  expect(existsSync(pnpm.spawned)).toBe(false)
  pnpm.writeManifest('12.0.1')
  prepareNodeDependencies(options)
  expect(readFileSync(pnpm.spawned, 'utf8')).toBe('install\n')
  // pm/lock.json isn't a receipt input, so a re-pin must be refused before a completed install is reused.
  pinPnpm(source, '12.0.2')
  expect(() => prepareNodeDependencies(options)).toThrow(/pnpm 12\.0\.2 is required \(pm\/lock\.json\)/)
}, 30000)

test.skipIf(process.platform === 'win32')('a source without a pm/lock.json pnpm pin is refused with the reason, before anything is installed', () => {
  const { source, env } = fixture()
  const pnpm = fakePnpm({ manifest: '12.0.0' })
  const options = { source, workspaces: ['web'], env: { ...env, PATH: `${pnpm.path}${delimiter}${env.PATH}` } }
  json(join(source, 'pm/lock.json'), { schema: 1, packages: {} })
  expect(() => prepareNodeDependencies(options)).toThrow(/pm\/lock\.json pins no pnpm/)
  rmSync(join(source, 'pm/lock.json'))
  expect(() => prepareNodeDependencies(options)).toThrow(/cannot read the pnpm pin from .*pm\/lock\.json: ENOENT/)
  expect(existsSync(pnpm.spawned)).toBe(false)
}, 30000)
