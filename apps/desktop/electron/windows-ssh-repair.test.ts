import assert from 'node:assert/strict'

import { test } from 'vitest'

import { listRemoteHermesProfiles, readRemoteInstallId } from './remote-lifecycle'
import { assertWindowsRemoteInstallUpdateClear, connectWindowsRemote, probeWindowsRemote } from './windows-remote-lifecycle'

const runtime = {
  os: 'Windows', arch: 'AMD64', hermesHome: 'C:\\Users\\A\\AppData\\Local\\hermes',
  hermesPath: 'C:\\Users\\A\\AppData\\Local\\hermes\\bin\\hermes.exe',
  python: 'C:\\Python\\python.exe'
}

function decode(command: string) {
  const token = command.match(/-EncodedCommand\s+(\S+)$/)?.[1]
  assert.ok(token, 'outer shells must receive the bootstrap as one encoded argument')

  return Buffer.from(token, 'base64').toString('utf16le')
}

test('platform and marker bootstrap survive outer-shell parsing without inlining long payloads', async () => {
  const calls: Array<{ command: string; options: { stdinData?: string } }> = []

  const ssh = { exec: async (command: string, options: { stdinData?: string }) => {
    calls.push({ command, options })

    return calls.length === 1 ? JSON.stringify(runtime) : 'CLEAR'
  } }

  await probeWindowsRemote(ssh, 'C:\\用户\\Hermes\\hermes.exe')
  await assertWindowsRemoteInstallUpdateClear(ssh, runtime.hermesHome, runtime.python)

  for (const { command, options } of calls) {
    const wrapper = decode(command)
    assert.match(wrapper, /ReadToEnd/)
    assert.match(wrapper, /FromBase64String/)
    assert.ok(command.length < 8191)
    const script = Buffer.from(String(options.stdinData).trim(), 'base64').toString('utf16le')
    assert.ok(script.length > wrapper.length)
  }
})

test('Windows roster and install id never dispatch POSIX syntax or hide a failed read', async () => {
  const calls: string[] = []

  const ssh = { exec: async (command: string, options: { stdinData?: string }) => {
    calls.push(command)
    assert.doesNotMatch(command, /^echo|^if \[/)
    const script = Buffer.from(String(options?.stdinData ?? '').trim(), 'base64').toString('utf16le') || decode(command)

    if (script.includes('.hermes-update-in-progress')) {return 'CLEAR'}

    return script.includes('install_id') ? JSON.stringify({ id: 'a'.repeat(32) }) : JSON.stringify({ profiles: ['default', 'bob', 'not legal', '.rollback', 'bob'] })
  } }

  assert.deepEqual(await listRemoteHermesProfiles(ssh, runtime), ['default', 'bob'])
  assert.equal(await readRemoteInstallId(ssh, runtime), 'a'.repeat(32))
  assert.ok(calls.length > 0)
  const failed = { exec: async () => { throw new Error('fixture disconnected') } }
  await assert.rejects(() => listRemoteHermesProfiles(failed, runtime))
})

test('Windows inventory rejects malformed helper output instead of fabricating default', async () => {
  const ssh = { exec: async (_command: string, options: { stdinData?: string }) => {
    const script = Buffer.from(String(options?.stdinData ?? '').trim(), 'base64').toString('utf16le')

    return script.includes('.hermes-update-in-progress') ? 'CLEAR' : '{}'
  } }

  await assert.rejects(() => listRemoteHermesProfiles(ssh, runtime))
})

test('a bin launcher does not require a nonexistent sibling Python and resolves after the update gate', async () => {
  const bootstrap = "import os, sys, runpy; runpy.run_module('hermes_cli.main', run_name='__main__', alter_sys=True)"

  const ssh = { exec: async (command: string, options: { stdinData?: string }) => {
    const script = Buffer.from(String(options?.stdinData || '').trim(), 'base64').toString('utf16le')

    if (script.includes('Get-Command hermes.exe')) {return JSON.stringify({ ...runtime, python: '', launcherPath: runtime.hermesPath })}

    if (script.includes('.hermes-update-in-progress')) {return 'CLEAR'}

    return JSON.stringify({ python: runtime.python, runtimeCommand: [runtime.python, '-I', '-c', bootstrap] })
  } }

  const result = await probeWindowsRemote(ssh)

  assert.equal(result.python, runtime.python)
  assert.equal(result.hermesPath, runtime.hermesPath)
})

test('the complete connect path keeps atomic spawn off argv and delivers the explicit caller JSON', async () => {
  let spawned = false

  const ssh = { exec: async (command: string, options: { stdinData?: string } = {}) => {
    const token = command.match(/-EncodedCommand\s+(\S+)$/)?.[1]
    let script = token ? Buffer.from(token, 'base64').toString('utf16le') : ''

    if (!token || script.includes('ReadToEnd')) {script = Buffer.from(String(options.stdinData || '').trim(), 'base64').toString('utf16le')}

    if (script.includes('Get-Command hermes.exe')) {return JSON.stringify(runtime)}

    if (script.includes("'spawn'")) {
      assert.ok(command.length < 8191, 'Connect must not inline its atomic spawn script')
      assert.ok(script.includes('"profile":"bob"'), 'script stdin cannot also serve as helper JSON stdin')
      assert.match(script, /' \| & /)
      spawned = true

      return JSON.stringify({ pid: 42, creationTimeNs: '1234567890123456789' })
    }

    if (script.includes('.hermes-update-in-progress')) {return 'CLEAR'}

    if (script.includes("'inspect'")) {return JSON.stringify({ supported: true, path: runtime.hermesPath, version: 'fixture' })}

    if (script.includes("'read-lock'")) {return 'null'}

    if (script.includes("'process-state'")) {return JSON.stringify({ alive: true, owned: true, indeterminate: false })}

    if (script.includes("'read-log'")) {return JSON.stringify({ content: 'HERMES_BACKEND_READY port=5555' })}

    return '{"ok":true}'
  } }

  const result = await connectWindowsRemote({
    ssh, profile: 'bob', ownershipId: 'a'.repeat(32), pickLocalPort: async () => 4444,
    forward: async () => {}, cancelForward: async () => {}, waitForHermes: async () => {}, probeReuseProof: async () => 'authenticated-ok'
  })

  assert.equal(spawned, true)
  assert.equal(result.remotePort, 5555)
})
