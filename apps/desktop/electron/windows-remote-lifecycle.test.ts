import assert from 'node:assert/strict'
import crypto from 'node:crypto'

import { test } from 'vitest'

import {
  buildWindowsInteractiveCommand,
  connectWindowsRemote,
  detectRemotePlatform,
  encodedPowerShell,
  helperCommand,
  powerShellCommand,
  psLiteral,
  reusableWindowsLock,
  validLock
} from './windows-remote-lifecycle'

const ownershipId = '0123456789abcdef0123456789abcdef'

function sshWith(exec) {
  return { exec }
}

test('PowerShell transport uses UTF-16LE encoded commands and literal escaping', () => {
  assert.equal(Buffer.from(encodedPowerShell("'ok'"), 'base64').toString('utf16le'), "'ok'")
  assert.equal(psLiteral("a'b"), "'a''b'")
  assert.match(powerShellCommand('Write-Output ok'), /^powershell\.exe -NoProfile -NonInteractive .* -EncodedCommand /)
})

test('platform detection preserves POSIX and falls back to Windows PowerShell', async () => {
  assert.deepEqual(await detectRemotePlatform(sshWith(async () => 'Linux\nx86_64\n')), { os: 'Linux', arch: 'x86_64' })
  const calls: string[] = []

  const result = await detectRemotePlatform(
    sshWith(async command => {
      calls.push(command)

      if (command.startsWith('uname ')) {
        throw new Error('PowerShell does not recognize uname')
      }

      return JSON.stringify({
        os: 'Windows',
        arch: 'ARM64',
        hermesHome: 'C:\\h',
        hermesPath: 'C:\\h\\hermes.exe',
        python: 'C:\\h\\python.exe'
      })
    })
  )

  assert.equal(result.os, 'Windows')
  assert.match(calls[1], /EncodedCommand/)
})

test('platform detection surfaces transport failures as themselves, not unsupported-platform', async () => {
  // A dead/unauthorized host is a connectivity verdict; only a host that answers
  // neither probe is an unsupported platform.
  const transportErr: any = new Error('SSH connection timed out')
  transportErr.kind = 'timeout'
  await assert.rejects(
    detectRemotePlatform(
      sshWith(async () => {
        throw transportErr
      })
    ),
    (err: any) => err.kind === 'timeout'
  )
  // Probe genuinely failing on a reachable host still classifies unsupported,
  // and carries the probe detail for diagnosis.
  await assert.rejects(
    detectRemotePlatform(
      sshWith(async command => {
        if (command.startsWith('uname ')) {
          throw new Error('not recognized')
        }

        throw new Error('Hermes is not installed on the remote Windows host.')
      })
    ),
    (err: any) => err.kind === 'unsupported-platform' && /Hermes is not installed/.test(err.message)
  )
})

test('helper command uses the fixed remote Python entry point and quotes path data', () => {
  const command = helperCommand({ python: "C:\\Program Files\\Hermes's\\python.exe" }, 'inspect', [
    'C:\\x y\\hermes.exe'
  ])

  const encoded = command.split(' ').pop()!
  const script = Buffer.from(encoded, 'base64').toString('utf16le')
  assert.match(script, /-m' 'hermes_cli\.windows_ssh_runtime' 'inspect'/)
  assert.match(script, /Hermes''s/)
  assert.match(script, /C:\\x y\\hermes\.exe/)
})

test('Windows lock validation is scoped and exact', () => {
  const lock = {
    schemaVersion: 2,
    protocolVersion: 1,
    ownershipId,
    spawnNonce: '0123456789abcdef',
    pid: 10,
    creationTimeNs: '1784219690452757504',
    port: 1234,
    tokenFingerprint: 'a'.repeat(32),
    hermesPath: 'C:\\h\\hermes.exe',
    hermesHome: 'C:\\h'
  }

  assert.equal(validLock(lock, ownershipId), true)
  assert.equal(validLock({ ...lock, ownershipId: 'b'.repeat(32) }, ownershipId), false)
  assert.equal(validLock({ ...lock, creationTimeNs: '0' }, ownershipId), false)
  // port 0 = spawn-in-progress record: valid ownership proof (cleanup can act
  // on it) but the reuse gate must reject it separately.
  assert.equal(validLock({ ...lock, port: 0 }, ownershipId), true)
  assert.equal(validLock({ ...lock, port: -1 }, ownershipId), false)
})

test('Windows SSH reuse requires the requested remote profile to match the lock', () => {
  const token = 'stored-token'

  const lock = {
    schemaVersion: 2,
    protocolVersion: 1,
    ownershipId,
    spawnNonce: '0123456789abcdef',
    pid: 10,
    creationTimeNs: '1784219690452757504',
    port: 1234,
    profile: 'default',
    tokenFingerprint: crypto.createHash('sha256').update(token).digest('hex').slice(0, 32),
    hermesPath: 'C:\\h\\hermes.exe',
    hermesHome: 'C:\\h'
  }

  const state = { alive: true, owned: true }
  const runtime = { hermesPath: lock.hermesPath, hermesHome: lock.hermesHome }

  assert.equal(reusableWindowsLock(lock, state, 'default', token, runtime), true)
  assert.equal(reusableWindowsLock(lock, state, 'desktop-work', token, runtime), false)
  assert.equal(reusableWindowsLock({ ...lock, profile: '' }, state, '', token, runtime), true)
})

test('Windows integrated terminal uses encoded PowerShell and preserves cwd as literal data', () => {
  const command = buildWindowsInteractiveCommand("C:\\Users\\O'Brien\\repo")
  const script = Buffer.from(command.split(' ').pop()!, 'base64').toString('utf16le')
  assert.match(script, /Set-Location -LiteralPath 'C:\\Users\\O''Brien\\repo'/)
  assert.match(script, /powershell\.exe -NoLogo/)
})

function windowsRestartConnection(owned = true, terminated = true) {
  const oldNonce = '0123456789abcdef'
  const token = 'stored-token'

  const lock = {
    schemaVersion: 2,
    protocolVersion: 1,
    ownershipId,
    spawnNonce: oldNonce,
    pid: 333,
    creationTimeNs: '1784219690452757504',
    port: 1234,
    profile: '',
    tokenFingerprint: crypto.createHash('sha256').update(token).digest('hex').slice(0, 32),
    hermesPath: 'C:\\h\\hermes.exe',
    hermesHome: 'C:\\h'
  }

  const calls: string[] = []
  let lockReads = 0

  const ssh = {
    calls,
    async exec(command: string) {
      calls.push(command)
      const script = Buffer.from(command.split(' ').pop()!, 'base64').toString('utf16le')
      const operation = script.match(/hermes_cli\.windows_ssh_runtime' '([^']+)/)?.[1]

      if (!operation) {
        return JSON.stringify({
          os: 'Windows',
          arch: 'AMD64',
          hermesHome: 'C:\\h',
          hermesPath: 'C:\\h\\hermes.exe',
          python: 'C:\\h\\python.exe'
        })
      }

      if (operation === 'inspect') {
        return JSON.stringify({ supported: true, path: 'C:\\h\\hermes.exe', version: 'Hermes 1' })
      }

      if (operation === 'read-lock') {
        lockReads += 1

        return JSON.stringify(lock)
      }

      if (operation === 'process-state') {
        return JSON.stringify({ alive: true, owned })
      }

      if (operation === 'spawn') {
        return JSON.stringify({ pid: 999, creationTimeNs: '1784219690452757999' })
      }

      if (operation === 'read-log') {
        return JSON.stringify({ content: 'HERMES_DASHBOARD_READY port=4321\n' })
      }

      if (operation === 'terminate') {
        return JSON.stringify({ terminated })
      }

      return JSON.stringify({})
    }
  }

  return { lock, ssh, token, oldNonce, getLockReads: () => lockReads }
}

test('Windows force restart terminates owned process and spawns fresh pid and nonce', async () => {
  const fixture = windowsRestartConnection()

  const result = await connectWindowsRemote({
    ssh: fixture.ssh,
    ownershipId,
    profile: '',
    reuseToken: fixture.token,
    pickLocalPort: async () => 50001,
    forward: async () => {},
    cancelForward: async () => {},
    waitForHermes: async () => {},
    probeReuseProof: async () => 'authenticated-ok',
    forceRestart: true
  })

  assert.equal(result.reused, false)
  assert.equal(result.pid, 999)
  assert.notEqual(result.spawnNonce, fixture.oldNonce)
  assert.equal(fixture.getLockReads(), 1, 'fresh recursive connect must skip the old lock')

  const terminateCalls = fixture.ssh.calls.filter(command => {
    const script = Buffer.from(command.split(' ').pop()!, 'base64').toString('utf16le')

    return script.includes("'terminate'")
  })

  assert.equal(terminateCalls.length, 1)
})

test('Windows force restart never terminates when process ownership proof fails', async () => {
  const fixture = windowsRestartConnection(false)

  await assert.rejects(
    () =>
      connectWindowsRemote({
        ssh: fixture.ssh,
        ownershipId,
        profile: '',
        reuseToken: fixture.token,
        pickLocalPort: async () => 50001,
        forward: async () => {},
        cancelForward: async () => {},
        waitForHermes: async () => {},
        probeReuseProof: async () => 'authenticated-ok',
        forceRestart: true
      }),
    (error: any) => error.kind === 'ownership-failed'
  )
  assert.ok(
    fixture.ssh.calls.every(
      command => !Buffer.from(command.split(' ').pop()!, 'base64').toString('utf16le').includes("'terminate'")
    )
  )
})

test('Windows force restart preserves ownership artifacts when termination is not proven', async () => {
  const fixture = windowsRestartConnection(true, false)

  await assert.rejects(
    () =>
      connectWindowsRemote({
        ssh: fixture.ssh,
        ownershipId,
        profile: '',
        reuseToken: fixture.token,
        pickLocalPort: async () => 50001,
        forward: async () => {},
        cancelForward: async () => {},
        waitForHermes: async () => {},
        probeReuseProof: async () => 'authenticated-ok',
        forceRestart: true
      }),
    (error: any) => error.kind === 'ownership-failed'
  )

  const scripts = fixture.ssh.calls.map(command => Buffer.from(command.split(' ').pop()!, 'base64').toString('utf16le'))
  assert.ok(scripts.some(script => script.includes("'terminate'")))
  assert.ok(!scripts.some(script => script.includes("'remove-lock'")))
})

test('Windows force restart keeps the ownership failure when forward teardown rejects', async () => {
  const fixture = windowsRestartConnection(true)

  await assert.rejects(
    () =>
      connectWindowsRemote({
        ssh: fixture.ssh,
        ownershipId,
        profile: '',
        reuseToken: fixture.token,
        pickLocalPort: async () => 50001,
        forward: async () => {},
        cancelForward: async () => {
          throw new Error('forward teardown blew up')
        },
        waitForHermes: async () => {},
        probeReuseProof: async () => {
          throw new Error('probe transport failed')
        },
        forceRestart: true
      }),
    (error: any) =>
      error.kind === 'ownership-failed' && /Could not verify the owned SSH backend/.test(error.message)
  )
})
