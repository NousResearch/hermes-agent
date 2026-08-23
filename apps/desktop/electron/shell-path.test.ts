import assert from 'node:assert/strict'

import { beforeEach, test } from 'vitest'

import {
  applyLoginShellPath,
  ensureLoginShellPath,
  extractSentinelPath,
  loginShellExecutable,
  mergeLoginShellPath,
  resetLoginShellPathForTests
} from './shell-path'

const START = '__HERMES_LOGIN_PATH_START__'
const END = '__HERMES_LOGIN_PATH_END__'

function fakeExecFile(outputsByFlags, { error = null, calls = [] }: any = {}) {
  return (file, args, _options, callback) => {
    calls.push({ file, args })
    const flags = args[0]
    const stdout = outputsByFlags[flags] ?? ''
    queueMicrotask(() => callback(error, stdout, ''))

    return { stdin: { end() {} } }
  }
}

beforeEach(() => {
  resetLoginShellPathForTests()
})

test('extractSentinelPath survives profile banner noise around the markers', () => {
  const stdout = `Welcome to my shell!\nmotd garbage\n${START}/opt/homebrew/bin:/usr/bin${END}trailing noise`
  assert.equal(extractSentinelPath(stdout), '/opt/homebrew/bin:/usr/bin')
})

test('extractSentinelPath uses the last start marker (echoed command lines cannot poison it)', () => {
  const stdout = `${START}poisoned-echo${END}\n${START}/real/path${END}`
  assert.equal(extractSentinelPath(stdout), '/real/path')
})

test('extractSentinelPath returns null when markers are missing or empty', () => {
  assert.equal(extractSentinelPath('no markers here'), null)
  assert.equal(extractSentinelPath(`${START}${END}`), null)
  assert.equal(extractSentinelPath(`${START}/half/open`), null)
})

test('mergeLoginShellPath puts login entries first and appends current-only entries', () => {
  const merged = mergeLoginShellPath(
    '/opt/homebrew/bin:/Users/u/.local/bin:/usr/bin:/bin',
    '/usr/bin:/bin:/launcher/only',
    { delimiter: ':' }
  )

  assert.equal(merged, '/opt/homebrew/bin:/Users/u/.local/bin:/usr/bin:/bin:/launcher/only')
})

test('loginShellExecutable honors $SHELL and falls back per platform', () => {
  assert.equal(loginShellExecutable({ SHELL: '/usr/local/bin/fish' }, 'darwin'), '/usr/local/bin/fish')
  assert.equal(loginShellExecutable({}, 'darwin'), '/bin/zsh')
  assert.equal(loginShellExecutable({}, 'linux'), '/bin/bash')
})

test('applyLoginShellPath merges the captured login PATH into the env', async () => {
  const env: any = { SHELL: '/bin/zsh', PATH: '/usr/bin:/bin' }
  const calls: any[] = []
  const execFileFn = fakeExecFile({ '-ilc': `${START}/opt/homebrew/bin:/Users/u/.cargo/bin:/usr/bin${END}` }, { calls })

  const result = await applyLoginShellPath({ env, platform: 'darwin', execFileFn })

  assert.equal(result.applied, true)
  assert.equal(env.PATH, '/opt/homebrew/bin:/Users/u/.cargo/bin:/usr/bin:/bin')
  assert.equal(calls.length, 1)
  assert.equal(calls[0].file, '/bin/zsh')
  assert.equal(calls[0].args[0], '-ilc')
})

test('applyLoginShellPath falls back to -lc when -ilc yields no sentinel (bash 3.2 swallow)', async () => {
  const env: any = { SHELL: '/bin/bash', PATH: '/usr/bin' }
  const calls: any[] = []
  const execFileFn = fakeExecFile({ '-ilc': 'swallowed', '-lc': `${START}/opt/homebrew/bin:/usr/bin${END}` }, { calls })

  const result = await applyLoginShellPath({ env, platform: 'darwin', execFileFn })

  assert.equal(result.applied, true)
  assert.deepEqual(
    calls.map(call => call.args[0]),
    ['-ilc', '-lc']
  )
  assert.equal(env.PATH, '/opt/homebrew/bin:/usr/bin')
})

test('applyLoginShellPath leaves the env untouched when resolution fails', async () => {
  const env: any = { SHELL: '/bin/zsh', PATH: '/usr/bin:/bin' }
  const execFileFn = fakeExecFile({}, { error: new Error('boom') })

  const result = await applyLoginShellPath({ env, platform: 'darwin', execFileFn })

  assert.equal(result.applied, false)
  assert.equal(result.reason, 'unresolved')
  assert.equal(env.PATH, '/usr/bin:/bin')
})

test('applyLoginShellPath reports unchanged when the login PATH adds nothing new', async () => {
  const env: any = { SHELL: '/bin/zsh', PATH: '/opt/homebrew/bin:/usr/bin' }
  const execFileFn = fakeExecFile({ '-ilc': `${START}/opt/homebrew/bin:/usr/bin${END}` })

  const result = await applyLoginShellPath({ env, platform: 'darwin', execFileFn })

  assert.equal(result.applied, false)
  assert.equal(result.reason, 'unchanged')
  assert.equal(env.PATH, '/opt/homebrew/bin:/usr/bin')
})

test('applyLoginShellPath is a no-op on Windows', async () => {
  const env: any = { Path: 'C:\\Windows\\System32' }
  const result = await applyLoginShellPath({ env, platform: 'win32' })

  assert.equal(result.applied, false)
  assert.equal(result.reason, 'win32')
  assert.equal(env.Path, 'C:\\Windows\\System32')
})

test('ensureLoginShellPath is single-flight — concurrent callers share one shell probe', async () => {
  const env: any = { SHELL: '/bin/zsh', PATH: '/usr/bin' }
  const calls: any[] = []
  const execFileFn = fakeExecFile({ '-ilc': `${START}/opt/homebrew/bin:/usr/bin${END}` }, { calls })

  const [first, second] = await Promise.all([
    ensureLoginShellPath({ env, platform: 'darwin', execFileFn }),
    ensureLoginShellPath({ env, platform: 'darwin', execFileFn })
  ])

  assert.equal(first.applied, true)
  assert.equal(second.applied, true)
  assert.equal(calls.length, 1)
  assert.equal(env.PATH, '/opt/homebrew/bin:/usr/bin')
})

test('ensureLoginShellPath never rejects', async () => {
  const execFileFn = () => {
    throw new Error('spawn EACCES')
  }

  const result = await ensureLoginShellPath({ env: { SHELL: '/bin/zsh' }, platform: 'darwin', execFileFn })
  assert.equal(result.applied, false)
})
