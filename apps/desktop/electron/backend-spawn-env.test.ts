import assert from 'node:assert/strict'

import { test } from 'vitest'

import { desktopBackendSpawnEnv } from './backend-spawn-env'

test('desktopBackendSpawnEnv keeps the base env and stamps stay_alive last', () => {
  const base = {
    HERMES_HOME: '/tmp/home',
    HERMES_DESKTOP: '1',
    GATEWAY_ON_ALL_ADAPTERS_DOWN: 'exit',
    PATH: '/usr/bin'
  }

  const env = desktopBackendSpawnEnv(base)

  assert.equal(env.HERMES_HOME, base.HERMES_HOME)
  assert.equal(env.HERMES_DESKTOP, base.HERMES_DESKTOP)
  assert.equal(env.PATH, base.PATH)
  // The desktop spawns `hermes serve` with no supervising service manager, so the
  // child must stay alive on all-adapters-down instead of exiting EX_TEMPFAIL
  // (#118080). Stamped unconditionally — an inherited value cannot opt the child
  // back into the failure exit.
  assert.equal(env.GATEWAY_ON_ALL_ADAPTERS_DOWN, 'stay_alive')
})

test.each(['darwin', 'linux', 'win32'] as const)('a %s backend drops foreign session identity without changing provider config', platform => {
  const base: NodeJS.ProcessEnv = {
    CODEX_THREAD_ID: 'parent-codex-thread',
    CODEX_SESSION_ID: 'parent-codex-session',
    CLAUDE_SESSION_ID: 'parent-claude-session',
    ...(platform === 'win32' ? { Codex_Thread_Id: 'case-insensitive-parent' } : {}),
    CODEX_HOME: '/tmp/codex-config',
    HERMES_CODEX_BASE_URL: 'http://localhost:1234',
    HERMES_HOME: '/tmp/hermes-profile',
    HERMES_SESSION_ID: 'hermes-owned-session',
    HERMES_SESSION_PROFILE: 'work',
    PATH: '/usr/bin'
  }

  const before = { ...base }

  const env = desktopBackendSpawnEnv(base, platform)

  assert.equal(env.CODEX_THREAD_ID, undefined)
  assert.equal(env.CODEX_SESSION_ID, undefined)
  assert.equal(env.CLAUDE_SESSION_ID, undefined)
  assert.equal(env.Codex_Thread_Id, undefined)

  for (const key of ['CODEX_HOME', 'HERMES_CODEX_BASE_URL', 'HERMES_HOME', 'HERMES_SESSION_ID', 'HERMES_SESSION_PROFILE', 'PATH']) {
    assert.equal(env[key], base[key])
  }

  assert.deepEqual(base, before)
})
