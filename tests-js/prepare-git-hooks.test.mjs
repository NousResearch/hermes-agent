import { execFileSync, spawnSync } from 'node:child_process'
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { delimiter, dirname, join } from 'node:path'
import { afterEach, expect, test } from 'vitest'
import { pnpmCommand } from '../scripts/build/node-deps.mjs'

const prepare = JSON.parse(readFileSync(new URL('../package.json', import.meta.url), 'utf8')).scripts.prepare
const roots = []

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true, maxRetries: 3 })
})

function runPrepare({ ci, gitLayout = 'directory', hookStatus = 0, managedInstall = false } = {}) {
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
  if (managedInstall) {
    // A workspace with one member, locked: what a managed build installs from.
    const version = '1.0.0'
    writeFileSync(join(root, 'package.json'), JSON.stringify({
      name: 'hermes-prepare-fixture', version, private: true, scripts: { prepare } }))
    writeFileSync(join(root, 'pnpm-workspace.yaml'), 'packages:\n  - apps/*\n')
    mkdirSync(join(root, 'apps', 'demo'), { recursive: true })
    writeFileSync(join(root, 'apps', 'demo', 'package.json'), JSON.stringify({ name: 'demo', version, private: true }))
    // Written outside CI: with CI set pnpm refuses to create a lockfile.
    const lockEnv = { ...env }
    delete lockEnv.CI
    execFileSync(pnpmCommand({ env }), ['install', '--lockfile-only', '--offline'], { cwd: root, env: lockEnv, stdio: 'pipe' })
    rmSync(log, { force: true })
  }
  // The arguments scripts/build/node-deps.mjs passes pnpm for a managed build.
  const [command, ...args] = managedInstall
    ? [pnpmCommand({ env }), 'install', '--frozen-lockfile', '--trust-lockfile', '--prod=false', '--offline',
      '--filter', '{./apps/demo}...']
    : [prepare]
  const result = spawnSync(command, args, { cwd: root, env, shell: !managedInstall, windowsHide: true,
    timeout: 30000, encoding: 'utf8' })
  expect(result.error).toBeUndefined()
  return { status: result.status, calls: existsSync(log) ? JSON.parse(readFileSync(log, 'utf8')) : null }
}

test.each(['directory', 'file'])('managed builds skip hook installation with .git as a %s', (gitLayout) => {
  for (const ci of ['1', 'true']) {
    // Managed builds must not install contributor hooks into customer checkouts.
    expect(runPrepare({ ci, gitLayout, hookStatus: 7 })).toEqual({ status: 0, calls: null })
  }
})

test.each(['directory', 'file'])('developer installs retain hooks with .git as a %s', (gitLayout) => {
  for (const ci of [undefined, '', '0', 'false']) {
    expect(runPrepare({ ci, gitLayout })).toEqual({ status: 0, calls: ['install'] })
  }
})

test.each(['directory', 'file'])('developer hook failures remain non-fatal with .git as a %s', (gitLayout) => {
  expect(runPrepare({ gitLayout, hookStatus: 7 })).toEqual({ status: 0, calls: ['install'] })
})

test('source copies without Git metadata still skip hook installation', () => {
  expect(runPrepare({ gitLayout: 'absent', hookStatus: 7 })).toEqual({ status: 0, calls: null })
})

test.each(['1', 'true'])('a managed pnpm install never invokes hooks with CI=%s', (ci) => {
  expect(runPrepare({ ci, gitLayout: 'file', hookStatus: 7, managedInstall: true })).toEqual({ status: 0, calls: null })
}, 45000)

test('the same pnpm install outside a managed build still installs hooks', () => {
  // Proves the fixture reaches the root prepare script, so the CI cases above are not vacuous.
  expect(runPrepare({ gitLayout: 'file', managedInstall: true })).toEqual({ status: 0, calls: ['install'] })
}, 45000)
