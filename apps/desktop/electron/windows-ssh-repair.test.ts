import assert from 'node:assert/strict'
import { spawnSync } from 'node:child_process'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'

import { test } from 'vitest'

import { listRemoteHermesProfiles, readRemoteInstallId } from './remote-lifecycle'
import {
  assertWindowsRemoteInstallUpdateClear,
  connectWindowsRemote,
  decodePowerShellCommand,
  probeWindowsFiles,
  probeWindowsRemote,
  windowsHelperOwnsRuntime
} from './windows-remote-lifecycle'

const runtime = {
  os: 'Windows',
  arch: 'AMD64',
  hermesHome: 'C:\\Users\\A\\AppData\\Local\\hermes',
  hermesPath: 'C:\\Users\\A\\AppData\\Local\\hermes\\bin\\hermes.exe',
  python: 'C:\\Python\\python.exe'
}

function decode(command: string) {
  assert.match(command, /-EncodedCommand\s+\S+$/, 'outer shells must receive the script as one encoded argument')

  return decodePowerShellCommand(command)
}

test('platform and marker scripts survive outer-shell parsing as one compressed argument without stdin', async () => {
  const calls: Array<{ command: string; options: { stdinData?: string } }> = []

  const ssh = {
    exec: async (command: string, options: { stdinData?: string } = {}) => {
      calls.push({ command, options })

      return calls.length === 1 ? JSON.stringify(runtime) : 'CLEAR'
    }
  }

  await probeWindowsRemote(ssh, 'C:\\用户\\Hermes\\hermes.exe')
  await assertWindowsRemoteInstallUpdateClear(ssh, runtime.hermesHome, runtime.python)

  for (const { command, options } of calls) {
    const outer = Buffer.from(command.split(' ').at(-1)!, 'base64').toString('utf16le')
    assert.match(outer, /DeflateStream/)
    assert.ok(command.length < 8191)
    assert.equal(options.stdinData, undefined)
    assert.ok(decode(command).length > outer.length)
  }
})

test('Windows roster and install id never dispatch POSIX syntax or hide a failed read', async () => {
  const calls: string[] = []

  const ssh = {
    exec: async (command: string) => {
      calls.push(command)
      assert.doesNotMatch(command, /^echo|^if \[/)
      const script = decode(command)

      if (script.includes('.hermes-update-in-progress')) {
        return 'CLEAR'
      }

      return JSON.stringify({
        profiles: ['default', 'bob', 'not legal', '.rollback', 'bob'],
        installId: 'a'.repeat(32)
      })
    }
  }

  assert.deepEqual(await listRemoteHermesProfiles(ssh, runtime), ['default', 'bob'])
  assert.equal(await readRemoteInstallId(ssh, runtime), 'a'.repeat(32))
  assert.ok(calls.length > 0)
  const failed = {
    exec: async () => {
      throw new Error('fixture disconnected')
    }
  }
  await assert.rejects(() => listRemoteHermesProfiles(failed, runtime))
})

test('Windows inventory rejects malformed helper output instead of fabricating default', async () => {
  const ssh = {
    exec: async (command: string) => {
      const script = decodePowerShellCommand(command)

      return script.includes('.hermes-update-in-progress') ? 'CLEAR' : '{}'
    }
  }

  await assert.rejects(() => listRemoteHermesProfiles(ssh, runtime))
})

test('a bin launcher does not require a nonexistent sibling Python and resolves after the update gate', async () => {
  const bootstrap = "import os, sys, runpy; runpy.run_module('hermes_cli.main', run_name='__main__', alter_sys=True)"

  const ssh = {
    exec: async (command: string) => {
      const script = decodePowerShellCommand(command)

      if (script.includes('Get-Command hermes -CommandType Application')) {
        return JSON.stringify({ ...runtime, python: '', launcherPath: runtime.hermesPath })
      }

      if (script.includes('.hermes-update-in-progress')) {
        return 'CLEAR'
      }

      return JSON.stringify({ python: runtime.python, runtimeCommand: [runtime.python, '-I', '-c', bootstrap] })
    }
  }

  const result = await probeWindowsRemote(ssh)

  assert.equal(result.python, runtime.python)
  assert.equal(result.hermesPath, runtime.hermesPath)
})

test('the complete connect path keeps atomic spawn off argv and delivers the explicit caller JSON', async () => {
  let spawned = false

  const ssh = {
    exec: async (command: string, options: { stdinData?: string } = {}) => {
      const script = decodePowerShellCommand(command) || ''

      if (script.includes('Get-Command hermes -CommandType Application')) {
        return JSON.stringify(runtime)
      }

      if (script.includes("'spawn'")) {
        assert.ok(command.length < 8191, "Connect's atomic spawn command must fit cmd.exe's limit")
        assert.equal(options.stdinData, undefined)
        assert.ok(script.includes('"profile":"bob"'), 'the caller JSON travels inside the spawn script')
        assert.match(script, /' \| & /)
        spawned = true

        return JSON.stringify({ pid: 42, creationTimeNs: '1234567890123456789' })
      }

      if (script.includes('.hermes-update-in-progress')) {
        return 'CLEAR'
      }

      if (script.includes("'inspect'")) {
        return JSON.stringify({ supported: true, path: runtime.hermesPath, version: 'fixture' })
      }

      if (script.includes("'read-lock'")) {
        return 'null'
      }

      if (script.includes("'process-state'")) {
        return JSON.stringify({ alive: true, owned: true, indeterminate: false })
      }

      if (script.includes("'read-log'")) {
        return JSON.stringify({ content: 'HERMES_BACKEND_READY port=5555' })
      }

      return '{"ok":true}'
    }
  }

  const result = await connectWindowsRemote({
    ssh,
    profile: 'bob',
    ownershipId: 'a'.repeat(32),
    pickLocalPort: async () => 4444,
    forward: async () => {},
    cancelForward: async () => {},
    waitForHermes: async () => {},
    probeReuseProof: async () => 'authenticated-ok'
  })

  assert.equal(spawned, true)
  assert.equal(result.remotePort, 5555)
})

test('Test and Connect share one verdict on a helper too old for a boot-selected backend', () => {
  const selected = { ...runtime, runtimeCommand: [runtime.python, '-I', '-c', 'boot'] }

  assert.equal(windowsHelperOwnsRuntime(selected, { supported: true }), false)
  assert.equal(windowsHelperOwnsRuntime(selected, { supported: true, runtimeBootstrapOwnership: 1 }), true)
  assert.equal(windowsHelperOwnsRuntime(runtime, { supported: true }), true)
})

test.runIf(process.platform === 'win32')(
  'the real Windows probe finds a .cmd-only launcher in the Hermes home off PATH',
  async () => {
    const home = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-probe-home-'))
    fs.mkdirSync(path.join(home, 'bin'))
    fs.writeFileSync(path.join(home, 'bin', 'hermes.cmd'), '@echo off\r\n')
    const system = process.env.SystemRoot || 'C:\\Windows'
    const env = {
      ...process.env,
      HERMES_HOME: home,
      PATH: [path.join(system, 'System32'), system, path.join(system, 'System32', 'WindowsPowerShell', 'v1.0')].join(
        ';'
      )
    }

    const ssh = {
      exec: async (command: string, options: { stdinData?: string } = {}) => {
        const [exe, ...args] = command.split(' ')
        const result = spawnSync(exe, args, { input: options.stdinData, encoding: 'utf8', env })

        if (result.status !== 0) {
          throw new Error(result.stderr)
        }

        return result.stdout
      }
    }

    try {
      const parsed = await probeWindowsFiles(ssh, '')
      assert.equal(parsed.hermesPath.toLowerCase(), path.join(home, 'bin', 'hermes.cmd').toLowerCase())
    } finally {
      fs.rmSync(home, { recursive: true, force: true })
    }
  }
)
