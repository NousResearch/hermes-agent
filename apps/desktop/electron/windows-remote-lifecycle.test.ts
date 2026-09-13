import assert from 'node:assert/strict'
import crypto from 'node:crypto'

import { test } from 'vitest'

import {
  assertWindowsRemoteInstallUpdateClear,
  atomicWindowsSpawnCommand,
  buildWindowsInteractiveCommand,
  connectWindowsRemote,
  detectRemotePlatform,
  encodedPowerShell,
  helperCommand,
  powerShellCommand,
  probeWindowsRemote,
  psLiteral,
  reusableWindowsLock,
  terminateOwnedWindowsDashboardForUpdate,
  validLock
} from './windows-remote-lifecycle'

const ownershipId = '0123456789abcdef0123456789abcdef'

test('Windows spawn holds the update mutex across marker check and helper spawn', () => {
  const command = atomicWindowsSpawnCommand({
    hermesHome: 'C:\\Users\\andre\\.hermes',
    python: 'C:\\Users\\andre\\.hermes\\python.exe'
  })

  const encoded = command.match(/-EncodedCommand\s+([^\s]+)$/)?.[1]
  const script = encoded ? Buffer.from(encoded, 'base64').toString('utf16le') : ''
  assert.match(script, /\.hermes-update-in-progress/)
  assert.match(script, /\$mutexPath=\$marker\+"\.mutex"/)
  assert.match(script, /\.Lock\(0,1\)/)
  assert.match(script, /windows_ssh_runtime.*spawn/)
  assert.match(script, /remote update marker is present/)
})

test('Windows spawn publishes the initial ownership record before releasing the mutex', () => {
  const command = atomicWindowsSpawnCommand(
    {
      hermesHome: 'C:\\Users\\andre\\.hermes',
      python: 'C:\\Users\\andre\\.hermes\\python.exe'
    },
    {
      ownershipId,
      spawnNonce: '0123456789abcdef',
      profile: 'default',
      hermesPath: 'C:\\Hermes\\hermes.exe',
      hermesHome: 'C:\\Users\\andre\\.hermes',
      tokenFingerprint: 'a'.repeat(32),
      startedAt: '2026-07-14T00:00:00.000Z'
    }
  )

  const encoded = command.split(' ').at(-1) || ''
  const script = encoded ? Buffer.from(encoded, 'base64').toString('utf16le') : ''

  assert.match(script, /read-lock/)
  assert.match(script, /write-lock/)
  assert.ok(script.indexOf('write-lock') < script.indexOf('Unlock'))
})

function sshWith(exec) {
  return { exec }
}

test('PowerShell transport uses UTF-16LE encoded commands and literal escaping', () => {
  assert.equal(Buffer.from(encodedPowerShell("'ok'"), 'base64').toString('utf16le'), "'ok'")
  assert.equal(psLiteral("a'b"), "'a''b'")
  assert.match(powerShellCommand('Write-Output ok'), /^powershell\.exe -NoProfile -NonInteractive .* -EncodedCommand /)
})

test('Windows relaunch gate refuses live and uncertain markers before executing the remote runtime', async () => {
  for (const observation of ['LIVE:4242', 'UNCERTAIN']) {
    const scripts: string[] = []

    const ssh = sshWith(async command => {
      const script = Buffer.from(command.split(' ').at(-1) || '', 'base64').toString('utf16le')
      scripts.push(script)

      if (script.includes('Get-Command hermes.exe')) {
        return JSON.stringify({
          os: 'Windows',
          arch: 'AMD64',
          hermesHome: 'C:\\Users\\alice\\.hermes',
          hermesPath: 'C:\\Hermes\\hermes.exe',
          python: 'C:\\Hermes\\python.exe'
        })
      }

      if (script.includes('.hermes-update-in-progress')) {
        return observation
      }

      throw new Error(`unexpected command after update gate: ${script}`)
    })

    await assert.rejects(
      () =>
        connectWindowsRemote({
          ssh,
          ownershipId,
          pickLocalPort: async () => 50000,
          forward: async () => {},
          cancelForward: async () => {},
          waitForHermes: async () => {},
          probeReuseProof: async () => 'authenticated-ok'
        }),
      (error: any) => error.kind === 'update-in-progress'
    )
    assert.equal(
      scripts.some(script => script.includes('hermes_cli.windows_ssh_runtime')),
      false
    )
  }
})

test('Windows relaunch gate uses strict install-wide marker parsing and fail-closed PID probing', async () => {
  let script = ''

  const ssh = sshWith(async command => {
    script = Buffer.from(command.split(' ').at(-1) || '', 'base64').toString('utf16le')

    return 'CLEAR'
  })

  await assertWindowsRemoteInstallUpdateClear(ssh, 'C:\\Users\\alice\\.hermes\\profiles\\research')
  assert.match(script, /\.hermes-update-in-progress/)
  assert.match(script, /Split-Path -Leaf \$parent.*profiles/)
  assert.match(script, /UTF8Encoding.*true/)
  assert.match(script, /\\A\(\[1-9\]/)
  assert.match(script, /GetProcessById/)
  assert.doesNotMatch(script, /ErrorAction SilentlyContinue/)
})

test('Windows probe validates Hermes and Python topology before selection', async () => {
  let script = ''
  await probeWindowsRemote(
    sshWith(async command => {
      script = Buffer.from(command.split(' ').at(-1) || '', 'base64').toString('utf16le')

      return JSON.stringify({
        os: 'Windows',
        arch: 'AMD64',
        hermesHome: 'C:\\\\h',
        hermesPath: 'C:\\\\h\\\\hermes.exe',
        python: 'C:\\\\h\\\\python.exe'
      })
    }),
    'C:\\\\h\\\\hermes.exe'
  )

  const explicitCheck = script.indexOf('if($explicit){Assert-NoReparse $explicit $false;')
  const explicitPythonCheck = script.indexOf('Assert-NoReparse $explicitPython $true')
  const fallbackJoin = script.indexOf('Join-Path $hermesHome')
  const candidatePythonCheck = script.indexOf('Assert-NoReparse $candidatePython $true')
  const candidateSelection = script.indexOf('Get-Item -LiteralPath $candidate')
  const pythonSelection = script.indexOf('$python=$runtime.python')
  const pythonCheck = script.indexOf('Assert-NoReparse $python $false')
  const output = script.indexOf('[ordered]@{')

  assert.ok(explicitCheck >= 0)
  assert.ok(explicitCheck < explicitPythonCheck)
  assert.ok(explicitPythonCheck < fallbackJoin)
  assert.ok(candidatePythonCheck >= 0)
  assert.ok(candidatePythonCheck < candidateSelection)
  assert.ok(pythonSelection >= 0)
  assert.ok(pythonSelection < pythonCheck)
  assert.ok(pythonCheck < output)
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

  assert.match(script, /\$runtime=\$null/)
  assert.match(script, /Get-Item -LiteralPath \$candidate/)
  assert.match(script, /Get-Item -LiteralPath \$candidatePython/)
  assert.ok(script.indexOf('foreach($candidate in $candidates)') < script.indexOf('$hermes=$runtime.hermes'))
  assert.match(runtime.python, /venv\\Scripts\\python\.exe$/)
})

test('Windows probe names an explicitly configured runtime missing its sibling Python', async () => {
  let script = ''

  await assert.rejects(
    detectRemotePlatform(
      sshWith(async command => {
        if (command.startsWith('uname ')) {
          throw new Error('not recognized')
        }

        script = Buffer.from(command.split(' ').pop()!, 'base64').toString('utf16le')
        throw new Error('The configured Hermes path was found, but its sibling python.exe was not found.')
      }),
      'C:\\Users\\me\\hermes-agent\\venv\\Scripts\\hermes.exe'
    ),
    (err: any) =>
      err.kind === 'unsupported-platform' &&
      /configured Hermes path was found, but its sibling python\.exe was not found/.test(err.message)
  )

  assert.match(script, /\$explicitHermesExists=\[bool\]\$explicit/)
  assert.match(script, /if\(-not \$runtime\)\{if\(\$explicitHermesExists -and -not \$explicitHasPython\)/)
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

test('managed update drain preserves a Windows owner when creation time does not match', async () => {
  const lock = {
    schemaVersion: 2,
    protocolVersion: 1,
    ownershipId,
    spawnNonce: '0123456789abcdef',
    pid: 10,
    creationTimeNs: '1784219690452757504',
    port: 1234,
    profile: 'default',
    tokenFingerprint: 'a'.repeat(32),
    hermesPath: 'C:\\h\\hermes.exe',
    hermesHome: 'C:\\h'
  }

  const operations: string[] = []

  const ssh = sshWith(async command => {
    const script = Buffer.from(command.split(' ').at(-1) || '', 'base64').toString('utf16le')
    operations.push(script)

    return JSON.stringify(lock)
  })

  await assert.rejects(
    terminateOwnedWindowsDashboardForUpdate(
      ssh,
      { python: 'C:\\h\\python.exe' },
      { ...lock, creationTimeNs: '1784219690452757505' }
    ),
    /ownership record changed/
  )
  assert.equal(
    operations.some(operation => operation.includes("'terminate'")),
    false
  )
  assert.equal(
    operations.some(operation => operation.includes("'remove-lock'")),
    false
  )
})

test('managed update drain rechecks Windows PID/create-time ownership before exact terminate', async () => {
  const lock = {
    schemaVersion: 2,
    protocolVersion: 1,
    ownershipId,
    spawnNonce: '0123456789abcdef',
    pid: 10,
    creationTimeNs: '1784219690452757504',
    port: 1234,
    profile: 'default',
    tokenFingerprint: 'a'.repeat(32),
    hermesPath: 'C:\\h\\hermes.exe',
    hermesHome: 'C:\\h'
  }

  const operations: string[] = []

  const ssh = sshWith(async command => {
    const script = Buffer.from(command.split(' ').at(-1) || '', 'base64').toString('utf16le')
    operations.push(script)

    if (script.includes("'read-lock'")) {
      return JSON.stringify(lock)
    }

    if (script.includes("'process-state'")) {
      return JSON.stringify({ alive: true, owned: true, indeterminate: false })
    }

    return JSON.stringify({ ok: true })
  })

  const result = await terminateOwnedWindowsDashboardForUpdate(ssh, { python: 'C:\\h\\python.exe' }, lock)

  assert.equal(result.terminated, true)
  assert.equal(operations.filter(operation => operation.includes("'read-lock'")).length, 2)
  assert.equal(operations.filter(operation => operation.includes("'process-state'")).length, 2)
  assert.equal(operations.filter(operation => operation.includes("'terminate'")).length, 1)
  assert.equal(
    operations.some(operation => operation.includes("'remove-lock'")),
    false
  )
})
