import { execFileSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { load } from 'js-yaml'
import { afterEach, expect, test } from 'vitest'
import { pnpmCommand, prepareNodeDependencies } from '../scripts/build/node-deps.mjs'

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const roots = []
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true }) })

function json(path, value) {
  mkdirSync(dirname(path), { recursive: true })
  writeFileSync(path, JSON.stringify(value))
}

// How the repository declares get-windows: the apps/desktop manifest field, the allowBuilds entry (key and
// value), and the workspace settings that decide what a failed or unreviewed build does.
function repoDeclaration() {
  const manifest = JSON.parse(readFileSync(join(repo, 'apps/desktop/package.json'), 'utf8'))
  const field = ['dependencies', 'optionalDependencies', 'devDependencies'].find(name => manifest[name]?.['get-windows'])
  const version = manifest[field]['get-windows']
  const workspace = load(readFileSync(join(repo, 'pnpm-workspace.yaml'), 'utf8'))
  const allowBuilds = workspace.allowBuilds ?? {}
  // pnpm keys a file: dependency's versioned decision by its spec, not its registry version.
  const allow = `get-windows@${version}` in allowBuilds
    ? { [`get-windows@file:vendor/get-windows-${version}.tgz`]: allowBuilds[`get-windows@${version}`] }
    : 'get-windows' in allowBuilds ? { 'get-windows': allowBuilds['get-windows'] } : {}
  return { field, version, allow, settings: { strictDepBuilds: workspace.strictDepBuilds, sideEffectsCache: workspace.sideEffectsCache } }
}

// A workspace whose desktop declares a local get-windows tarball exactly as the repository declares the
// real one. Its install script fails, as node-pre-gyp's fallback build does where no prebuilt exists
// (win32-arm64); elsewhere the script never matters, because pnpm skips or ignores it.
function fixture(os) {
  const base = mkdtempSync(join(tmpdir(), 'desktop get-windows-'))
  roots.push(base)
  const source = join(base, 'source')
  const env = { ...process.env, pnpm_config_offline: 'true', pnpm_config_store_dir: join(base, 'store'),
    pnpm_config_cache_dir: join(base, 'cache'), pnpm_config_state_dir: join(base, 'state') }
  const { field, version, allow, settings } = repoDeclaration()
  const vendor = join(source, 'vendor/get-windows')
  json(join(vendor, 'package.json'), { name: 'get-windows', version, main: 'index.js',
    scripts: { install: 'node -e "process.exit(1)"' } })
  writeFileSync(join(vendor, 'index.js'), '')
  writeFileSync(join(vendor, 'main'), '')
  execFileSync(pnpmCommand(), ['pack', '--pack-destination', join(source, 'vendor')], { cwd: vendor, env, stdio: 'pipe' })
  json(join(source, 'package.json'), { name: 'fixture', private: true })
  json(join(source, 'apps/desktop/package.json'), { name: 'desktop', version: '1.0.0',
    [field]: { 'get-windows': `file:../../vendor/get-windows-${version}.tgz` } })
  // JSON is YAML: the repository's own values, verbatim.
  writeFileSync(join(source, 'pnpm-workspace.yaml'), JSON.stringify({ packages: ['apps/desktop'], ...settings,
    allowBuilds: allow, supportedArchitectures: { os: [os] } }))
  execFileSync(pnpmCommand(), ['install', '--lockfile-only'], { cwd: source, env, stdio: 'pipe' })
  // prepareNodeDependencies refuses a pnpm other than PM's pin (pm/lock.json): pin the one this test runs.
  const pnpmVersion = execFileSync(pnpmCommand(), ['--version'], { cwd: source, env, encoding: 'utf8' }).trim()
  json(join(source, 'pm/lock.json'), { schema: 1, packages: { pnpm: { version: pnpmVersion, artifacts: {} } } })
  return { source, env }
}

// #149861: pnpm >= 11.6 infers os: [win32] from the name of an optional dependency (cpu is not inferred) and
// skipped get-windows on macOS and Linux, so the macOS build lost its helper and read_window_below was gone
// everywhere but Windows. On win32 the fixture's failing script stands in for win32-arm64, which has no
// prebuilt: a failed native build must not fail the install (win32-x64 downloads one and never fails).
test.each([
  ['darwin', 'is not skipped by name'],
  ['linux', 'is not skipped by name'],
  ['win32', 'survives a failed native build'],
])('the desktop install keeps get-windows on %s: it %s', (os) => {
  const { source, env } = fixture(os)
  prepareNodeDependencies({ source, workspaces: ['apps/desktop'], env })
  // The workspace's own link, not require.resolve: `pnpm exec` sets NODE_PATH to the outer install.
  expect(existsSync(join(source, 'apps/desktop/node_modules/get-windows/main'))).toBe(true)
}, 30000)
