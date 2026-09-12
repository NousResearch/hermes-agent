import { execFileSync } from 'node:child_process'
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync, existsSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, expect, test } from 'vitest'
import { npmCommand } from '../scripts/build/node-deps.mjs'

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const roots = []
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true }) })

function json(path, value) {
  mkdirSync(dirname(path), { recursive: true })
  writeFileSync(path, JSON.stringify(value))
}

function fixture() {
  const root = mkdtempSync(join(tmpdir(), 'node union with spaces-'))
  roots.push(root)
  json(join(root, 'package.json'), { name: 'fixture', private: true, dependencies: { 'root-only': 'file:vendor/root-only' }, workspaces: ['ui-tui', 'web', 'apps/desktop'], engines: { node: '>=22', npm: '>=10' } })
  for (const name of ['tui-only', 'web-only', 'root-only']) {
    json(join(root, 'vendor', name, 'package.json'), { name, version: '1.0.0', main: 'index.cjs' })
    writeFileSync(join(root, 'vendor', name, 'index.cjs'), `module.exports = '${name}'`)
  }
  json(join(root, 'ui-tui/package.json'), { name: 'tui', version: '1.0.0', dependencies: { 'tui-only': 'file:../vendor/tui-only' } })
  json(join(root, 'web/package.json'), { name: 'web', version: '1.0.0', dependencies: { 'web-only': 'file:../vendor/web-only' } })
  json(join(root, 'apps/desktop/package.json'), { name: 'desktop', version: '1.0.0', scripts: { postinstall: 'node -e "require(\'fs\').writeFileSync(\'electron-provisioned\',\'yes\')"' } })
  const [node, npm] = npmCommand()
  execFileSync(node, [npm, 'install', '--package-lock-only', '--ignore-scripts', '--no-audit', '--no-fund', '--offline'], { cwd: root, env: { ...process.env, npm_config_cache: join(root, '.npm-cache') }, stdio: 'pipe' })
  return root
}

test('one locked preparation retains the requested union without provisioning desktop', async () => {
  const { prepareNodeDependencies } = await import('../scripts/build/node-deps.mjs')
  const source = fixture()
  const lock = readFileSync(join(source, 'package-lock.json'))
  await prepareNodeDependencies({ source, workspaces: ['ui-tui', 'web', 'ui-tui'], env: { ...process.env, npm_config_offline: 'true', npm_config_cache: join(source, '.npm-cache') } })
  for (const [workspace, name] of [['ui-tui', 'tui-only'], ['web', 'web-only']]) {
    expect(createRequire(join(source, workspace, 'package.json'))(name)).toBe(name)
  }
  expect(createRequire(join(source, 'package.json'))('root-only')).toBe('root-only')
  expect(existsSync(join(source, 'node_modules/desktop'))).toBe(false)
  expect(existsSync(join(source, 'apps/desktop/electron-provisioned'))).toBe(false)
  expect(readFileSync(join(source, 'package-lock.json'))).toEqual(lock)
  execFileSync(process.execPath, [join(repo, 'scripts/build/node-deps.mjs'), '--source', source, '--workspace', 'ui-tui', '--workspace', 'web'], { cwd: tmpdir(), env: { ...process.env, npm_config_offline: 'true', npm_config_cache: join(source, '.npm-cache') }, stdio: 'pipe' })
  expect(createRequire(join(source, 'web/package.json'))('web-only')).toBe('web-only')
}, 30000)

test('web-only selection excludes siblings and rejects stale locks without rewriting them', async () => {
  const { prepareNodeDependencies } = await import('../scripts/build/node-deps.mjs')
  const source = fixture()
  const env = { ...process.env, npm_config_offline: 'true', npm_config_cache: join(source, '.npm-cache') }
  rmSync(join(source, 'apps/desktop'), { recursive: true })
  prepareNodeDependencies({ source, workspaces: ['web'], env })
  expect(existsSync(join(source, 'node_modules/tui-only'))).toBe(false)
  expect(existsSync(join(source, 'apps/desktop/electron-provisioned'))).toBe(false)
  const lock = readFileSync(join(source, 'package-lock.json'))
  json(join(source, 'web/package.json'), { name: 'web', version: '1.0.0', dependencies: { 'web-only': '9.0.0' } })
  expect(() => prepareNodeDependencies({ source, workspaces: ['web'], env })).toThrow()
  expect(readFileSync(join(source, 'package-lock.json'))).toEqual(lock)
  expect(() => prepareNodeDependencies({ source, workspaces: [], env })).toThrow(/workspace/)
  expect(() => prepareNodeDependencies({ source, workspaces: ['missing'], env })).toThrow(/workspace/)
}, 30000)

test('a restored completed install retains postinstall outputs only for matching inputs', async () => {
  const { prepareNodeDependencies } = await import('../scripts/build/node-deps.mjs')
  const source = fixture()
  const env = { ...process.env, npm_config_offline: 'true', npm_config_cache: join(source, '.npm-cache') }
  prepareNodeDependencies({ source, workspaces: ['ui-tui', 'web'], env, reuse: true })
  const artifact = 'node_modules/postinstall-output'
  writeFileSync(join(source, artifact), 'built')
  const restored = mkdtempSync(join(tmpdir(), 'restored node union-'))
  roots.push(restored)
  cpSync(source, restored, { recursive: true, verbatimSymlinks: true })
  prepareNodeDependencies({ source: restored, workspaces: ['web', 'ui-tui'], env, reuse: true })
  expect(readFileSync(join(restored, artifact), 'utf8')).toBe('built')
  const cli = [join(repo, 'scripts/build/node-deps.mjs'), '--source', restored, '--workspace', 'web', '--workspace', 'ui-tui', '--reuse']
  execFileSync(process.execPath, cli, { cwd: tmpdir(), env, stdio: 'pipe' })
  expect(readFileSync(join(restored, artifact), 'utf8')).toBe('built')
  expect(createRequire(join(restored, 'web/package.json'))('web-only')).toBe('web-only')

  prepareNodeDependencies({ source: restored, workspaces: ['web'], env, reuse: true })
  expect(existsSync(join(restored, artifact))).toBe(false)
  expect(existsSync(join(restored, 'node_modules/tui-only'))).toBe(false)
  writeFileSync(join(restored, artifact), 'built again')
  // Even an invalid manifest must reach npm, not hide behind a warm lockfile cache.
  json(join(restored, 'web/package.json'), { name: 'web', version: '1.0.0', dependencies: { 'web-only': '9.0.0' } })
  expect(() => prepareNodeDependencies({ source: restored, workspaces: ['web'], env, reuse: true })).toThrow()
  cpSync(join(source, 'web/package.json'), join(restored, 'web/package.json'))
  prepareNodeDependencies({ source: restored, workspaces: ['web'], env, reuse: true })
  expect(existsSync(join(restored, artifact))).toBe(false)
}, 30000)

test('reuse respects lifecycle configuration and repairs missing installed packages', async () => {
  const { prepareNodeDependencies } = await import('../scripts/build/node-deps.mjs')
  const source = fixture()
  const manifest = JSON.parse(readFileSync(join(source, 'package.json'), 'utf8'))
  manifest.scripts = { postinstall: `node -e "require('node:fs').writeFileSync('node_modules/postinstall-output', require('node:crypto').randomUUID())"` }
  json(join(source, 'package.json'), manifest)
  const env = { ...process.env, npm_config_offline: 'true', npm_config_cache: join(source, '.npm-cache') }
  const options = { source, workspaces: ['web'], env, reuse: true }
  const artifact = join(source, 'node_modules/postinstall-output')
  prepareNodeDependencies({ ...options, env: { ...env, npm_config_ignore_scripts: 'true' } })
  expect(existsSync(artifact)).toBe(false)
  prepareNodeDependencies(options)
  const first = readFileSync(artifact, 'utf8')
  prepareNodeDependencies(options)
  expect(readFileSync(artifact, 'utf8')).toBe(first)
  rmSync(join(source, 'node_modules/web-only'), { recursive: true, force: true })
  prepareNodeDependencies(options)
  const repaired = readFileSync(artifact, 'utf8')
  expect(repaired).not.toBe(first)
  expect(createRequire(join(source, 'web/package.json'))('web-only')).toBe('web-only')
  prepareNodeDependencies({ ...options, reuse: false })
  expect(readFileSync(artifact, 'utf8')).not.toBe(repaired)
}, 30000)
