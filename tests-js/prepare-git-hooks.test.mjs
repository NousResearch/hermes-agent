import { spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { delimiter, dirname, join } from 'node:path'
import { afterEach, expect, test } from 'vitest'

const prepare = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8')).scripts.prepare
const roots = []

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true, maxRetries: 3 })
})

function runPrepare({ ci, gitLayout = 'directory', hookStatus = 0 } = {}) {
  const root = mkdtempSync(join(tmpdir(), 'hermes prepare with spaces-'))
  roots.push(root)
  const bin = join(root, 'bin')
  const log = join(root, 'hook-arguments.json')
  const probe = join(root, 'hook-probe.cjs')
  mkdirSync(bin)
  if (gitLayout === 'directory') mkdirSync(join(root, '.git'))
  if (gitLayout === 'file') {
    mkdirSync(join(root, 'git-metadata'))
    writeFileSync(join(root, '.git'), 'gitdir: git-metadata\n')
  }
  writeFileSync(probe, `
    require('node:fs').writeFileSync(process.env.PREPARE_HOOK_LOG, JSON.stringify(process.argv.slice(2)))
    process.exit(Number(process.env.PREPARE_HOOK_STATUS))
  `)
  const windows = process.platform === 'win32'
  writeFileSync(join(bin, windows ? 'lefthook.cmd' : 'lefthook'), windows
    ? `@echo off\r\n"${process.execPath}" "${probe}" %*\r\n`
    : `#!/bin/sh\nexec "${process.execPath}" "${probe}" "$@"\n`, { mode: 0o755 })
  const env = { ...process.env,
    PATH: [bin, dirname(process.execPath), process.env.PATH || ''].join(delimiter),
    PREPARE_HOOK_LOG: log, PREPARE_HOOK_STATUS: String(hookStatus) }
  delete env.CI
  if (ci !== undefined) env.CI = ci
  const result = spawnSync(prepare, { cwd: root, env, shell: true, windowsHide: true,
    timeout: 15000, encoding: 'utf8' })
  expect(result.error).toBeUndefined()
  return { status: result.status, calls: existsSync(log) ? JSON.parse(readFileSync(log, 'utf8')) : null }
}

test.each(['directory', 'file'])('managed builds skip hook installation with .git as a %s', (gitLayout) => {
  for (const ci of ['1', 'true']) {
    // A failed developer-only hook must not fail a managed dependency build.
    expect(runPrepare({ ci, gitLayout, hookStatus: 7 })).toEqual({ status: 0, calls: null })
  }
})

test.each([undefined, '', '0', 'false'])('developer installs retain hooks with CI=%s', (ci) => {
  expect(runPrepare({ ci })).toEqual({ status: 0, calls: ['install'] })
})

test('developer installs still report a hook installation failure', () => {
  expect(runPrepare({ hookStatus: 7 })).toEqual({ status: 7, calls: ['install'] })
})

test('source copies without Git metadata still skip hook installation', () => {
  expect(runPrepare({ gitLayout: 'absent', hookStatus: 7 })).toEqual({ status: 0, calls: null })
})
