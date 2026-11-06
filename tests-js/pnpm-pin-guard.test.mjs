import { spawnSync } from 'node:child_process'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { dirname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { expect, test } from 'vitest'

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..')
const semver = createRequire(import.meta.url)('semver')
const root = JSON.parse(readFileSync(join(repo, 'package.json'), 'utf8'))
const pinned = JSON.parse(readFileSync(join(repo, 'pm/lock.json'), 'utf8')).packages.pnpm.version

// pmOnFail: ignore turns off pnpm's devEngines check. engines.pnpm (with engineStrict)
// makes pnpm <= 11 refuse to install; pnpm 12 never compares engines.pnpm with
// itself, so the root preinstall, which pnpm runs on filtered installs too,
// refuses any pnpm but the managed one.
function preinstall(userAgent) {
  return spawnSync(root.scripts.preinstall, {
    cwd: repo, shell: true, encoding: 'utf8',
    env: { ...process.env, npm_config_user_agent: userAgent },
  })
}

test('engines.pnpm admits the managed pnpm and refuses older majors', () => {
  expect(semver.satisfies(pinned, root.engines?.pnpm)).toBe(true)
  for (const version of ['10.34.9', '11.28.11']) expect(semver.satisfies(version, root.engines?.pnpm)).toBe(false)
})

test.each([
  ['npm/11.19.0 node/v26.12.1 linux x64 workspaces/false', /installs with pnpm/],
  ['yarn/1.22.22 npm/? node/v26.12.1 linux x64', /installs with pnpm/],
  ['', /installs with pnpm/],
  ['pnpm/12.0.0 npm/? node/? linux x64', /is required, found pnpm\/12\.0\.0/],
  [`pnpm/${pinned}0 npm/? node/? linux x64`, /is required/],
])('the root preinstall refuses %j', (userAgent, message) => {
  const run = preinstall(userAgent)
  expect(run.status).toBe(1)
  expect(run.stderr).toMatch(message)
})

test('the root preinstall accepts the managed pnpm', () => {
  const run = preinstall(`pnpm/${pinned} npm/? node/? linux x64`)
  expect(run.stderr).toBe('')
  expect(run.status).toBe(0)
})
