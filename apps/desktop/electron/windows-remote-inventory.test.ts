import assert from 'node:assert/strict'

import { test } from 'vitest'

import { listRemoteHermesProfiles, readRemoteInstallId } from './remote-lifecycle'

// PR #137742: a Windows remote cannot answer a POSIX inventory. Desktop used to send
// `echo "${HERMES_HOME:-$HOME/.hermes}"` and `ls -1 .../profiles` down an ssh session whose
// remote command shell is cmd.exe, get the literal text back, and reject it as an unsafe
// home. The Windows path routes through `hermes_cli.windows_ssh_runtime` instead, reached
// exactly as `windows-remote-lifecycle.ts` reaches it: a PowerShell -EncodedCommand (helper
// ops) or a base64 stdin frame (the update marker gate).

type SshExecOptions = { stdinData?: string | Buffer }

const WINDOWS_PLATFORM = {
  os: 'Windows',
  arch: 'AMD64',
  hermesHome: 'C:\\Users\\alice\\.hermes',
  hermesPath: 'C:\\Hermes\\hermes.exe',
  python: 'C:\\Hermes\\python.exe'
}

// Undo the two transports so the tests can assert on the script the remote would run.
function scriptFromInvocation(command: string, options: SshExecOptions = {}) {
  const encoded = command.match(/-EncodedCommand\s+([^\s]+)$/)?.[1]

  if (encoded) {
    return Buffer.from(encoded, 'base64').toString('utf16le')
  }

  const framed = String(options.stdinData || '').replace(/\r?\n/g, '').trim()

  return framed ? Buffer.from(framed, 'base64').toString('utf16le') : ''
}

function inventorySsh({ capabilities = ['launcher-ownership', 'profile-inventory'], profiles = ['default'], installId = null }: any = {}) {
  const calls: string[] = []
  const ops: string[] = []

  return {
    calls,
    ops,
    async exec(command: string, options: SshExecOptions = {}) {
      const script = scriptFromInvocation(command, options)
      calls.push(script)

      // The install-wide update marker gate: pure PowerShell, answered before anything
      // imports the remote checkout.
      if (script.includes('.hermes-update-in-progress')) {
        return 'CLEAR'
      }

      const operation = script.match(/'hermes_cli\.windows_ssh_runtime'\s+'([^']+)'/)?.[1]

      if (!operation) {
        throw new Error(`unexpected remote command: ${script.slice(0, 120)}`)
      }

      ops.push(operation)

      if (operation === 'inspect') {
        return JSON.stringify({ version: '1.0.0', supported: true, path: WINDOWS_PLATFORM.hermesPath, capabilities })
      }

      if (operation === 'list-profiles') {
        return JSON.stringify({ profiles })
      }

      if (operation === 'install-id') {
        return JSON.stringify({ installId })
      }

      throw new Error(`unexpected helper operation: ${operation}`)
    }
  }
}

test('Windows inventory reads profiles and the install id through the native helper', async () => {
  const ssh = inventorySsh({
    profiles: ['default', 'bob', 'bob.rollback-old'],
    installId: '0F8A1C2B3D4E5F60718293A4B5C6D7E8'
  })

  assert.deepEqual(await listRemoteHermesProfiles(ssh, WINDOWS_PLATFORM), ['default', 'bob'])
  assert.equal(await readRemoteInstallId(ssh, WINDOWS_PLATFORM), '0f8a1c2b3d4e5f60718293a4b5c6d7e8')

  // The update gate runs before every helper round trip, exactly as connectWindowsRemote does.
  assert.deepEqual(ssh.ops, ['inspect', 'list-profiles', 'inspect', 'install-id'])
  assert.equal(ssh.calls.filter(script => script.includes('.hermes-update-in-progress')).length, 2)
  // No POSIX command is ever sent to a Windows remote.
  assert.equal(
    ssh.calls.some(script => script.includes('HERMES_HOME') || script.includes('ls -1') || script.includes('uname')),
    false
  )
})

test('Windows inventory refuses to guess when the install lacks profile-inventory', async () => {
  // An older remote checkout would answer nothing (or fail): the roster must report
  // update-required instead of silently emptying. Inventory failure never blocks the
  // connection — main.ts only logs it.
  const ssh = inventorySsh({ capabilities: ['launcher-ownership'] })

  await assert.rejects(
    () => listRemoteHermesProfiles(ssh, WINDOWS_PLATFORM),
    (err: any) => err.kind === 'update-required'
  )
  await assert.rejects(
    () => readRemoteInstallId(ssh, WINDOWS_PLATFORM),
    (err: any) => err.kind === 'update-required'
  )
  assert.deepEqual(ssh.ops, ['inspect', 'inspect'])
})

test('Windows inventory surfaces a failed helper call as a transport error', async () => {
  const ssh = inventorySsh()

  ssh.exec = async (command: string, options: SshExecOptions = {}) => {
    const script = scriptFromInvocation(command, options)

    if (script.includes('.hermes-update-in-progress')) {
      return 'CLEAR'
    }

    if (script.includes("'list-profiles'")) {
      throw new Error('SSH channel closed')
    }

    return JSON.stringify({ version: '1.0.0', supported: true, capabilities: ['profile-inventory'] })
  }

  await assert.rejects(
    () => listRemoteHermesProfiles(ssh, WINDOWS_PLATFORM),
    (err: any) => err.kind === 'transient-transport-error'
  )
})

test('Windows install-id reports no id rather than a bad one', async () => {
  for (const installId of [null, '', 'not-an-id', 'ABCDEF', '0f8a1c2b3d4e5f60718293a4b5c6d7e8extra']) {
    const ssh = inventorySsh({ installId })

    assert.equal(await readRemoteInstallId(ssh, WINDOWS_PLATFORM), undefined, String(installId))
  }
})

function posixSsh() {
  const calls: string[] = []

  return {
    calls,
    async exec(command: string): Promise<string> {
      calls.push(command)

      if (command.includes('HERMES_HOME')) {
        return '/Users/alice/.hermes\n'
      }

      if (command.includes('ls -1')) {
        return 'bob\ndixie\n'
      }

      if (command.includes('install_id')) {
        return '0f8a1c2b3d4e5f60718293a4b5c6d7e8\n'
      }

      return ''
    }
  }
}

test('POSIX inventory keeps its shell path when the platform is null or non-Windows', async () => {
  for (const platform of [null, { os: 'Linux', arch: 'x86_64' }, { os: 'Darwin', arch: 'arm64' }] as any[]) {
    const ssh = posixSsh()

    assert.deepEqual(await listRemoteHermesProfiles(ssh, platform), ['default', 'bob', 'dixie'])
    assert.equal(await readRemoteInstallId(ssh, platform), '0f8a1c2b3d4e5f60718293a4b5c6d7e8')
    assert.equal(ssh.calls.some(command => command.includes('windows_ssh_runtime')), false)
  }
})

test('POSIX inventory stays unsafe-path strict with no platform', async () => {
  const ssh = posixSsh()

  ssh.exec = async (command: string) => {
    ssh.calls.push(command)

    if (command.includes('HERMES_HOME')) {
      return '/tmp/x; echo pwned\n'
    }

    return ''
  }

  await assert.rejects(
    () => listRemoteHermesProfiles(ssh),
    (err: any) => err.kind === 'unsafe-path'
  )
})
