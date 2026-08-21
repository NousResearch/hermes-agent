import assert from 'node:assert/strict'
import crypto from 'node:crypto'

import { test } from 'vitest'

import {
  buildWindowsInteractiveCommand,
  detectRemotePlatform,
  encodedPowerShell,
  helperCommand,
  powerShellCommand,
  probeWindowsRemote,
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

test('Windows probe selects a validated Hermes and Python runtime pair', async () => {
  let script = ''
  const runtime = await probeWindowsRemote(
    sshWith(async command => {
      script = Buffer.from(command.split(' ').pop()!, 'base64').toString('utf16le')

      return JSON.stringify({
        os: 'Windows',
        arch: 'AMD64',
        hermesHome: 'C:\\Users\\me\\AppData\\Local\\hermes',
        hermesPath: 'C:\\Users\\me\\AppData\\Local\\hermes\\hermes-agent\\venv\\Scripts\\hermes.exe',
        python: 'C:\\Users\\me\\AppData\\Local\\hermes\\hermes-agent\\venv\\Scripts\\python.exe'
      })
    })
  )

  assert.match(script, /\$runtime=\$candidates\|ForEach-Object/)
  assert.match(script, /Test-Path -LiteralPath \$hermes -PathType Leaf/)
  assert.match(script, /Test-Path -LiteralPath \$python -PathType Leaf/)
  assert.ok(script.indexOf('$runtime=$candidates') < script.indexOf('$hermes=$runtime.hermes'))
  assert.match(runtime.python, /venv\\Scripts\\python\.exe$/)
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

  await assert.rejects(
    detectRemotePlatform(
      sshWith(async command => {
        if (command.startsWith('uname ')) {
          throw new Error('not recognized')
        }

        throw new Error(
          '#< CLIXML <Objs><S S="progress">module load</S><S S="Error">The remote Hermes Python runtime was not found._x000D__x000A_</S></Objs>'
        )
      })
    ),
    (err: any) =>
      err.kind === 'unsupported-platform' &&
      /probe: The remote Hermes Python runtime was not found\./.test(err.message) &&
      !/CLIXML|module load/.test(err.message)
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
