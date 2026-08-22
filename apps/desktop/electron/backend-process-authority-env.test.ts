import assert from 'node:assert/strict'
import path from 'node:path'

import { test } from 'vitest'

import {
  buildDesktopBackendEnv,
  DESKTOP_PARENT_PID_ENV,
  DESKTOP_PARENT_STARTED_AT_ENV,
  DESKTOP_PROCESS_AUTHORITY_ENV,
  DESKTOP_PROCESS_GENERATION_ENV,
  desktopProcessAuthorityBootstrapDirs,
  POSIX_PROCESS_AUTHORITY_MODE,
  WINDOWS_PROCESS_AUTHORITY_MODE
} from './backend-env'

const GENERATION = 'e2f531e4-14b1-47ff-9f87-bc278cfa816d'

test('Windows backend launch carries a generation-bound Job authority envelope', () => {
  const root = 'C:\\repo\\hermes-agent'
  const env = buildDesktopBackendEnv({
    authorityGeneration: GENERATION,
    currentEnv: { Path: 'C:\\Windows\\System32' },
    hermesHome: 'C:\\Users\\test\\AppData\\Local\\hermes',
    parentPid: 4242,
    parentStartedAtMs: 1_700_000_000_123,
    pathModule: path.win32,
    platform: 'win32',
    pythonPathEntries: [root],
    venvRoot: 'C:\\Users\\test\\AppData\\Local\\hermes\\hermes-agent\\venv'
  })

  assert.equal(env[DESKTOP_PROCESS_AUTHORITY_ENV], WINDOWS_PROCESS_AUTHORITY_MODE)
  assert.equal(env[DESKTOP_PROCESS_GENERATION_ENV], GENERATION)
  assert.equal(env[DESKTOP_PARENT_PID_ENV], '4242')
  assert.equal(env[DESKTOP_PARENT_STARTED_AT_ENV], '1700000000123')
  assert.equal(
    env.PYTHONPATH.split(';')[0],
    'C:\\repo\\hermes-agent\\hermes_cli\\desktop_bootstrap'
  )
})

test('POSIX backend launch arms the scoped session authority before imports', () => {
  const root = '/repo/hermes-agent'
  const env = buildDesktopBackendEnv({
    authorityGeneration: GENERATION,
    currentEnv: { PATH: '/usr/bin' },
    hermesHome: '/Users/test/.hermes',
    parentPid: 4242,
    parentStartedAtMs: 1_700_000_000_123,
    pathModule: path.posix,
    platform: 'darwin',
    pythonPathEntries: [root],
    venvRoot: '/Users/test/.hermes/hermes-agent/venv'
  })

  assert.equal(env[DESKTOP_PROCESS_AUTHORITY_ENV], POSIX_PROCESS_AUTHORITY_MODE)
  assert.equal(env[DESKTOP_PROCESS_GENERATION_ENV], GENERATION)
  assert.equal(env[DESKTOP_PARENT_PID_ENV], '4242')
  assert.equal(env[DESKTOP_PARENT_STARTED_AT_ENV], '1700000000123')
  assert.deepEqual(desktopProcessAuthorityBootstrapDirs([root], { pathModule: path.posix }), [
    '/repo/hermes-agent/hermes_cli/desktop_bootstrap'
  ])
  assert.equal(env.PYTHONPATH.split(':')[0], '/repo/hermes-agent/hermes_cli/desktop_bootstrap')
  assert.equal(env.PYTHONPATH.split(':')[1], root)
})

test('default Electron parent marker is stable across backend generations', () => {
  const first = buildDesktopBackendEnv({
    currentEnv: { PATH: '/usr/bin' },
    hermesHome: '/tmp/hermes',
    pathModule: path.posix,
    platform: 'linux',
    pythonPathEntries: ['/repo'],
    venvRoot: '/tmp/hermes/venv'
  })
  const second = buildDesktopBackendEnv({
    currentEnv: { PATH: '/usr/bin' },
    hermesHome: '/tmp/hermes',
    pathModule: path.posix,
    platform: 'linux',
    pythonPathEntries: ['/repo'],
    venvRoot: '/tmp/hermes/venv'
  })

  assert.notEqual(first[DESKTOP_PROCESS_GENERATION_ENV], second[DESKTOP_PROCESS_GENERATION_ENV])
  assert.equal(first[DESKTOP_PARENT_STARTED_AT_ENV], second[DESKTOP_PARENT_STARTED_AT_ENV])
})
