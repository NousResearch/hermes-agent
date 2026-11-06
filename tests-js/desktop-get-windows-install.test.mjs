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

// How the repository declares get-windows: the apps/desktop manifest field and the allowBuilds decision.
function repoDeclaration() {
  const manifest = JSON.parse(readFileSync(join(repo, 'apps/desktop/package.json'), 'utf8'))
  const field = ['dependencies', 'optionalDependencies', 'devDependencies'].find(name => manifest[name]?.['get-windows'])
  const version = manifest[field]['get-windows']
  const allowBuilds = load(readFileSync(join(repo, 'pnpm-workspace.yaml'), 'utf8')).allowBuilds ?? {}
  return { field, build: allowBuilds[`get-windows@${version}`] ?? allowBuilds['get-windows'] }
}

// A workspace whose desktop declares a local get-windows tarball exactly as the repository declares the
// real one. Its install script fails, as node-pre-gyp's fallback build does where no prebuilt exists.
function fixture(os, cpu) {
  const base = mkdtempSync(join(tmpdir(), 'desktop get-windows-'))
  roots.push(base)
  const source = join(base, 'source')
  const env = { ...process.env, pnpm_config_offline: 'true', pnpm_config_store_dir: join(base, 'store'),
    pnpm_config_cache_dir: join(base, 'cache'), pnpm_config_state_dir: join(base, 'state') }
  const vendor = join(source, 'vendor/get-windows')
  json(join(vendor, 'package.json'), { name: 'get-windows', version: '9.3.0', main: 'index.js',
    scripts: { install: 'node -e "process.exit(1)"' } })
  writeFileSync(join(vendor, 'index.js'), '')
  writeFileSync(join(vendor, 'main'), '')
  execFileSync(pnpmCommand(), ['pack', '--pack-destination', join(source, 'vendor')], { cwd: vendor, env, stdio: 'pipe' })
  const { field, build } = repoDeclaration()
  json(join(source, 'package.json'), { name: 'fixture', private: true })
  json(join(source, 'apps/desktop/package.json'), { name: 'desktop', version: '1.0.0',
    [field]: { 'get-windows': 'file:../../vendor/get-windows-9.3.0.tgz' } })
  writeFileSync(join(source, 'pnpm-workspace.yaml'), [
    'packages:', '  - apps/desktop', 'strictDepBuilds: true', 'sideEffectsCache: false',
    // pnpm keys a file: dependency's build decision by its spec, not its registry version.
    'allowBuilds:', `  'get-windows@file:vendor/get-windows-9.3.0.tgz': ${build}`,
    'supportedArchitectures:', `  os: [${os}]`, `  cpu: [${cpu}]`, '',
  ].join('\n'))
  execFileSync(pnpmCommand(), ['install', '--lockfile-only'], { cwd: source, env, stdio: 'pipe' })
  return { source, env }
}

// #149861: pnpm >= 11.6 infers os: [win32] from the name of an optional dependency and skipped get-windows
// on macOS and Linux, so the macOS build lost its helper and read_window_below was gone everywhere but
// Windows. win32-arm64 has no prebuilt: its failed native build must not fail the install either.
test.each([['darwin', 'arm64'], ['linux', 'x64'], ['win32', 'x64'], ['win32', 'arm64']])(
  'the desktop install keeps get-windows for %s-%s', (os, cpu) => {
    const { source, env } = fixture(os, cpu)
    prepareNodeDependencies({ source, workspaces: ['apps/desktop'], env })
    // The workspace's own link, not require.resolve: `pnpm exec` sets NODE_PATH to the outer install.
    expect(existsSync(join(source, 'apps/desktop/node_modules/get-windows/main'))).toBe(true)
  }, 30000)
