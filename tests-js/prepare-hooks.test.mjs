import { spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { delimiter, join } from 'node:path'
import { afterEach, expect, test } from 'vitest'

// The shipped prepare one-liner (kept inline per a50c4ba7), not a copy of it.
const prepare = JSON.parse(readFileSync(join(import.meta.dirname, '..', 'package.json'), 'utf8')).scripts.prepare
  .replace(/^node -e\s+/, '').replace(/^"([\s\S]*)"$/, '$1')

const roots = []
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true }) })

test('a failing lefthook install warns and still exits zero (#135279)', () => {
  const root = mkdtempSync(join(tmpdir(), 'prepare with spaces-'))
  roots.push(root)
  mkdirSync(join(root, 'work/.git'), { recursive: true })
  mkdirSync(join(root, 'bin'))
  const invoked = join(root, 'invoked')
  writeFileSync(join(root, 'bin/lefthook'), '#!/bin/sh\necho invoked >> "$LEFTHOOK_INVOKED"\nexit 1\n', { mode: 0o755 })
  writeFileSync(join(root, 'bin/lefthook.cmd'), '@echo off\r\necho invoked>> "%LEFTHOOK_INVOKED%"\r\nexit /b 1\r\n')
  const result = spawnSync(process.execPath, ['-e', prepare], { cwd: join(root, 'work'),
    env: { ...process.env, PATH: `${join(root, 'bin')}${delimiter}${process.env.PATH ?? ''}`, LEFTHOOK_INVOKED: invoked }, encoding: 'utf8' })
  expect(result.status).toBe(0)
  expect(existsSync(invoked)).toBe(true)
  expect(`${result.stdout ?? ''}${result.stderr ?? ''}`).toMatch(/lefthook install failed.*continuing without git hooks/)
}, 15000)

test('the default install still runs lefthook, and no .git still skips it', () => {
  const root = mkdtempSync(join(tmpdir(), 'prepare with spaces-'))
  roots.push(root)
  mkdirSync(join(root, 'bin'))
  writeFileSync(join(root, 'bin/lefthook'), '#!/bin/sh\necho invoked >> "$LEFTHOOK_INVOKED"\nexit 0\n', { mode: 0o755 })
  writeFileSync(join(root, 'bin/lefthook.cmd'), '@echo off\r\necho invoked>> "%LEFTHOOK_INVOKED%"\r\nexit /b 0\r\n')
  const bin = `${join(root, 'bin')}${delimiter}${process.env.PATH ?? ''}`
  mkdirSync(join(root, 'hooked/.git'), { recursive: true })
  const hooked = spawnSync(process.execPath, ['-e', prepare], { cwd: join(root, 'hooked'),
    env: { ...process.env, PATH: bin, LEFTHOOK_INVOKED: join(root, 'hooked-invoked') }, encoding: 'utf8' })
  expect(hooked.status).toBe(0)
  expect(existsSync(join(root, 'hooked-invoked'))).toBe(true)
  mkdirSync(join(root, 'bare'))
  const bare = spawnSync(process.execPath, ['-e', prepare], { cwd: join(root, 'bare'),
    env: { ...process.env, PATH: bin, LEFTHOOK_INVOKED: join(root, 'bare-invoked') }, encoding: 'utf8' })
  expect(bare.status).toBe(0)
  expect(existsSync(join(root, 'bare-invoked'))).toBe(false)
}, 15000)
