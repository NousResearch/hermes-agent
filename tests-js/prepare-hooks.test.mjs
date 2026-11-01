import { spawnSync } from 'node:child_process'
import { chmodSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { delimiter, dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterEach, expect, test } from 'vitest'

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const roots = []
afterEach(() => { for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true }) })

function fakeLefthookBin(root) {
  const bin = join(root, 'fake-bin')
  mkdirSync(bin, { recursive: true })
  writeFileSync(join(bin, 'lefthook'), '#!/bin/sh\necho invoked >> "$LEFTHOOK_INVOKED"\nexit "$LEFTHOOK_STATUS"\n')
  chmodSync(join(bin, 'lefthook'), 0o755)
  writeFileSync(join(bin, 'lefthook.cmd'), '@echo off\r\necho invoked>> "%LEFTHOOK_INVOKED%"\r\nexit /b %LEFTHOOK_STATUS%\r\n')
  return bin
}

function gitEnv(root, hooksPath) {
  writeFileSync(join(root, 'gitconfig-global'), hooksPath ? `[core]\n\thooksPath = ${hooksPath}\n` : '')
  return { GIT_CONFIG_GLOBAL: join(root, 'gitconfig-global'), GIT_CONFIG_SYSTEM: process.platform === 'win32' ? 'NUL' : '/dev/null' }
}

// The prepare script must stay a one-liner inline in root package.json (no
// extra file for the Docker build context, matching a50c4ba7). Exercise the
// shipped string, not a copy of it, with a stubbed lefthook on PATH.
function runPrepare(work, { hooksPath = '', lefthookStatus = 0 } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'prepare-shipped-'))
  roots.push(root)
  mkdirSync(join(work, '.git'), { recursive: true })
  const invoked = join(root, 'invoked')
  const prepare = JSON.parse(readFileSync(join(repo, 'package.json'), 'utf8')).scripts.prepare
  const script = prepare.replace(/^node -e\s+/, '').replace(/^"|"$/g, '')
  const result = spawnSync(process.execPath, ['-e', script], {
    cwd: work,
    env: {
      ...process.env,
      ...gitEnv(root, hooksPath),
      PATH: `${fakeLefthookBin(root)}${delimiter}${process.env.PATH ?? ''}`,
      LEFTHOOK_INVOKED: invoked,
      LEFTHOOK_STATUS: String(lefthookStatus),
    },
    encoding: 'utf8',
  })
  return { result, invoked }
}

test('a custom core.hooksPath skips lefthook and exits zero', () => {
  const work = join(mkdtempSync(join(tmpdir(), 'prepare-work-')), 'work')
  roots.push(dirname(work))
  const { result, invoked } = runPrepare(work, { hooksPath: '/custom/hooks' })
  expect(result.status).toBe(0)
  expect(existsSync(invoked)).toBe(false)
}, 15000)

test('the default configuration installs hooks and exits zero', () => {
  const work = join(mkdtempSync(join(tmpdir(), 'prepare-work-')), 'work')
  roots.push(dirname(work))
  const { result, invoked } = runPrepare(work)
  expect(result.status).toBe(0)
  expect(existsSync(invoked)).toBe(true)
}, 15000)

test('a failing lefthook install still exits zero', () => {
  const work = join(mkdtempSync(join(tmpdir(), 'prepare-work-')), 'work')
  roots.push(dirname(work))
  const { result, invoked } = runPrepare(work, { lefthookStatus: 1 })
  expect(result.status).toBe(0)
  expect(existsSync(invoked)).toBe(true)
}, 15000)
