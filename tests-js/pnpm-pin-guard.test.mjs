import { execFileSync, spawnSync } from 'node:child_process'
import { cpSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { tmpdir } from 'node:os'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { afterAll, expect, test } from 'vitest'
import { pnpmCommand } from '../scripts/build/node-deps.mjs'

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const semver = createRequire(import.meta.url)('semver')
const root = JSON.parse(readFileSync(join(repo, 'package.json'), 'utf8'))
const pinned = JSON.parse(readFileSync(join(repo, 'pm/lock.json'), 'utf8')).packages.pnpm.version
const scratch = mkdtempSync(join(tmpdir(), 'pnpm pin guard-'))
afterAll(() => rmSync(scratch, { recursive: true, force: true }))

// pnpm-workspace.yaml's pmOnFail: ignore means pnpm never checks its own version.
// engines.pnpm (with engineStrict) makes pnpm <= 11 refuse to install; pnpm 12 never
// compares engines.pnpm with itself, so the root preinstall refuses any pnpm but the
// managed one.
function preinstall(userAgent, cwd = repo) {
  return spawnSync(root.scripts.preinstall, {
    cwd, shell: true, encoding: 'utf8',
    env: { ...process.env, npm_config_user_agent: userAgent },
  })
}

test('engines.pnpm admits the managed pnpm and refuses older majors', () => {
  expect(semver.satisfies(pinned, root.engines?.pnpm)).toBe(true)
  for (const version of ['10.34.9', '11.28.11']) expect(semver.satisfies(version, root.engines?.pnpm)).toBe(false)
})

test.each([
  'npm/11.19.0 node/v26.12.1 linux x64 workspaces/false',
  'yarn/1.22.22 npm/? node/v26.12.1 linux x64',
  'pnpm/12.0.0 npm/? node/? linux x64',
  `pnpm/${pinned}0 npm/? node/? linux x64`,
  'corp-agent/1',
])('the root preinstall refuses %j and names the pin', userAgent => {
  const run = preinstall(userAgent)
  expect(run.stderr).toContain(`installs with pnpm ${pinned} (pm/lock.json), found ${userAgent.split(' ')[0]}`)
  expect(run.status).toBe(1)
})

test('the root preinstall accepts the managed pnpm', () => {
  const run = preinstall(`pnpm/${pinned} npm/? node/? linux x64`)
  expect(run.stderr).toBe('')
  expect(run.status).toBe(0)
})

test('without a pm/lock.json pnpm pin the root preinstall says so', () => {
  const cwd = join(scratch, 'no pin')
  mkdirSync(join(cwd, 'pm'), { recursive: true })
  expect(preinstall(`pnpm/${pinned}`, cwd).stderr).toMatch(/pm\/lock\.json pins no pnpm \(MODULE_NOT_FOUND\)/)
  writeFileSync(join(cwd, 'pm/lock.json'), JSON.stringify({ schema: 1, packages: {} }))
  const run = preinstall(`pnpm/${pinned}`, cwd)
  expect(run.stderr).toMatch(/pm\/lock\.json pins no pnpm/)
  expect(run.status).toBe(1)
})

// The guard is only as good as pnpm running the root preinstall when the root
// is not the project being installed. pnpm stamps pnpm_config_user_agent into the
// script environment as npm_config_user_agent, so a real pnpm can play any version.
function workspace() {
  const dir = join(scratch, `workspace ${Math.random().toString(36).slice(2)}`)
  mkdirSync(join(dir, 'child'), { recursive: true })
  writeFileSync(join(dir, 'package.json'), JSON.stringify({ name: 'root', private: true, scripts: { preinstall: root.scripts.preinstall } }))
  cpSync(join(repo, 'pm/lock.json'), join(dir, 'pm/lock.json'))
  writeFileSync(join(dir, 'pnpm-workspace.yaml'), 'packages:\n  - child\n')
  writeFileSync(join(dir, 'child/package.json'), JSON.stringify({ name: 'child', version: '1.0.0' }))
  return dir
}

function install(dir, cwd, args, userAgent) {
  try {
    execFileSync(pnpmCommand(), ['install', ...args], {
      cwd, encoding: 'utf8', stdio: 'pipe',
      env: { ...process.env, CI: '1', pnpm_config_offline: 'true', pnpm_config_user_agent: userAgent,
        pnpm_config_store_dir: join(dir, '.store'), pnpm_config_cache_dir: join(dir, '.cache'), pnpm_config_state_dir: join(dir, '.state') },
    })
    return { status: 0, output: '' }
  } catch (error) {
    return { status: error.status, output: `${error.stdout}${error.stderr}` }
  }
}

test.each([
  ['a filtered install', '.', ['--filter', './child']],
  ['an install from inside a workspace folder', 'child', []],
])('real pnpm runs the root preinstall on %s', (_, from, args) => {
  const dir = workspace()
  const refused = install(dir, join(dir, from), args, 'pnpm/12.0.0 npm/? node/? linux x64')
  expect(refused.output).toContain(`installs with pnpm ${pinned} (pm/lock.json), found pnpm/12.0.0`)
  expect(refused.status).not.toBe(0)
  expect(install(dir, join(dir, from), args, `pnpm/${pinned} npm/? node/? linux x64`)).toEqual({ status: 0, output: '' })
}, 60000)
