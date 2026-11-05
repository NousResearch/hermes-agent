import { execFileSync } from 'node:child_process'
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { delimiter, join } from 'node:path'
import { afterEach, expect, test, vi } from 'vitest'

// #147216: a PATH entry can pass existsSync and then stop resolving before
// realpathSync runs (FUSE-materialized shims, entries removed mid-scan). No
// filesystem layout reproduces that race deterministically, so fail the resolve.
const vanishing = vi.hoisted(() => new Set())
vi.mock('node:fs', async (importOriginal) => {
  const fs = await importOriginal()
  const realpathSync = (path, ...rest) => {
    if (vanishing.has(path)) throw Object.assign(new Error(`ENOENT: no such file or directory, lstat '${path}'`), { code: 'ENOENT' })
    return fs.realpathSync(path, ...rest)
  }
  return { ...fs, default: { ...fs, realpathSync }, realpathSync }
})

const { npmCommand } = await import('../scripts/build/node-deps.mjs')

const roots = []
afterEach(() => {
  vanishing.clear()
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true })
})

test('a PATH shim that stops resolving mid-scan is skipped', () => {
  const root = mkdtempSync(join(tmpdir(), 'vanishing npm-'))
  roots.push(root)
  const shim = join(root, process.platform === 'win32' ? 'npm.cmd' : 'npm')
  writeFileSync(shim, '')
  vanishing.add(shim)
  const env = { ...process.env, PATH: [root, process.env.PATH || process.env.Path].join(delimiter) }
  delete env.npm_execpath
  delete env.Path
  const [node, cli] = npmCommand({ env })
  expect(cli.startsWith(root)).toBe(false)
  expect(execFileSync(node, [cli, '--version'], { cwd: tmpdir(), env, encoding: 'utf8' }).trim()).toMatch(/^\d+\.\d+\.\d+/)
})

test('a lone vanishing shim reports missing npm, not the resolve error', () => {
  const root = mkdtempSync(join(tmpdir(), 'broken npm-'))
  roots.push(root)
  const shim = join(root, process.platform === 'win32' ? 'npm.cmd' : 'npm')
  writeFileSync(shim, '')
  const env = { ...process.env, PATH: root }
  delete env.npm_execpath
  delete env.Path
  vanishing.add(shim)
  expect(() => npmCommand({ env })).toThrow(/npm CLI is required/)
})
