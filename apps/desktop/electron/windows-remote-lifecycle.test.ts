import assert from 'node:assert/strict'
import { exec as execCallback, spawn } from 'node:child_process'
import crypto from 'node:crypto'
import { mkdtemp, readFile, rm } from 'node:fs/promises'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { test } from 'vitest'

import {
  assertWindowsRemoteInstallUpdateClear,
  atomicWindowsSpawnCommand,
  buildWindowsInteractiveCommand,
  connectWindowsRemote,
  detectRemotePlatform,
  encodedPowerShell,
  helper,
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
  assert.match(
    script,
    /\$marker\+"\.lock",\[IO\.FileMode\]::OpenOrCreate,\[IO\.FileAccess\]::ReadWrite,\[IO\.FileShare\]::None/
  )
  assert.match(script, /windows_ssh_runtime.*spawn/)
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
  assert.match(script, /\$lock\s*\|\s*&.*write-lock/)
  assert.doesNotMatch(script, /write-lock[^;]*\$lock\|Out-Null/)
  assert.ok(script.indexOf('write-lock') < script.lastIndexOf('$mutex.Dispose()'))
})

test.runIf(process.platform === 'win32')(
  'Windows spawn waits on <marker>.lock and judges v2 claims: live refuses, dead is deleted under the lock',
  async () => {
    const root = await mkdtemp(path.join(os.tmpdir(), 'hermes-win-marker-'))
    const marker = path.join(root, '.hermes-update-in-progress')
    const run = (command: string) => promisify(execCallback)(command, { timeout: 30_000 })
    const command = atomicWindowsSpawnCommand({ hermesHome: root, python: path.join(root, 'missing-python.exe') })

    // An updater: hold <marker>.lock sharing nothing (marker.ps1 Open-MarkerLock),
    // write its v2 claim, release, and stay alive as the live owner.
    const updater = spawn(
      'powershell.exe',
      [
        '-NoProfile',
        '-NonInteractive',
        '-EncodedCommand',
        encodedPowerShell(
          [
            `$m=${psLiteral(marker)}`,
            "$f=[IO.File]::Open($m+'.lock','OpenOrCreate','ReadWrite','None')",
            "'held'",
            'Start-Sleep -Milliseconds 500',
            '$ct=([DateTimeOffset](Get-Process -Id $PID).StartTime).ToUnixTimeMilliseconds()/1000.0',
            '[IO.File]::WriteAllText($m,"$PID`n$([DateTimeOffset]::UtcNow.ToUnixTimeSeconds())`nct:$ct`n")',
            'Start-Sleep -Milliseconds 500',
            '$f.Dispose()',
            'Start-Sleep -Seconds 30'
          ].join(';')
        )
      ],
      { stdio: ['ignore', 'pipe', 'inherit'] }
    )

    try {
      await new Promise(resolve => updater.stdout.once('data', resolve))
      await assert.rejects(run(command), (error: any) =>
        String(error.stderr).includes(`remote update marker is LIVE:${updater.pid}`)
      )

      updater.kill()
      await new Promise(resolve => updater.once('exit', resolve))
      await assertWindowsRemoteInstallUpdateClear(
        sshWith(async (probe: string) => (await run(probe)).stdout),
        root
      )
      // The dead claim is deleted inside the hold; the missing python then fails the spawn itself.
      await assert.rejects(run(command))
      await assert.rejects(readFile(marker), { code: 'ENOENT' })
    } finally {
      updater.kill()
      await rm(root, { recursive: true, force: true })
    }
  },
  60_000
)

function sshWith(exec) {
  return { exec }
}

test('PowerShell transport uses UTF-16LE encoded commands and literal escaping', () => {
  assert.equal(Buffer.from(encodedPowerShell("'ok'"), 'base64').toString('utf16le'), "'ok'")
  assert.equal(psLiteral("a'b"), "'a''b'")
  assert.match(powerShellCommand('Write-Output ok'), /^powershell\.exe -NoProfile -NonInteractive .* -EncodedCommand /)
})

test('every emitted PowerShell script keeps try blocks attached to their catch/finally handlers', async () => {
  // `;` between `try{...}` and `catch`/`finally` is a PowerShell parse error
  // (MissingCatchOrFinally), so no probe may join a handler onto a separate
  // statement. The line-oriented builders join with `;`; the pair must live
  // in one array element.
  const decode = (command: string) => Buffer.from(command.split(' ').at(-1) || '', 'base64').toString('utf16le')

  const scripts: string[] = []

  await probeWindowsRemote(
    sshWith(async command => {
      scripts.push(decode(command))

      return JSON.stringify({ os: 'Windows', arch: 'AMD64' })
    })
  )
  await assertWindowsRemoteInstallUpdateClear(
    sshWith(async command => {
      scripts.push(decode(command))

      return 'CLEAR'
    }),
    'C:\\Users\\alice\\.hermes'
  )
  scripts.push(
    decode(atomicWindowsSpawnCommand({ hermesHome: 'C:\\Users\\alice\\.hermes', python: 'C:\\py\\python.exe' })),
    decode(buildWindowsInteractiveCommand('C:\\work'))
  )

  assert.equal(scripts.length, 4)

  for (const script of scripts) {
    assert.doesNotMatch(script, /}\s*;\s*(?:catch|finally)\b/)
    // `$HOME`, `$HOST`, `$PID`, ... are read-only automatic variables: assigning
    // one throws "Cannot overwrite variable" at run time, so the probe exits 1
    // and the marker gate never observes CLEAR.
    assert.doesNotMatch(script, /\$(?:home|host|pid|profile|pwd|input|args|error)\s*=/i)
  }

  assert.ok(
    scripts.slice(0, 2).every(script => /}catch \[Management\.Automation\.ItemNotFoundException\]/.test(script))
  )
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

test('Windows gates keep a dead marker while the checkout lock or a lease byte is held', async () => {
  // Review G1: a dead claim is deleted (spawn) or reported CLEAR (probe) only
  // after the update_lock byte range -- owner byte + 16 R5b lease bytes at
  // 1048576 -- of the installer checkout and the runtime python's checkout is free.
  const decode = (command: string) => Buffer.from(command.split(' ').at(-1) || '', 'base64').toString('utf16le')
  const runtime = { hermesHome: 'C:\\Users\\alice\\.hermes', python: 'C:\\src\\hermes\\venv\\Scripts\\python.exe' }
  const spawnScript = decode(atomicWindowsSpawnCommand(runtime))
  let probeScript = ''

  const observed = await assertWindowsRemoteInstallUpdateClear(
    sshWith(async command => {
      probeScript = decode(command)

      return 'HELD'
    }),
    runtime.hermesHome,
    runtime.python
  ).catch(error => error)

  assert.equal(observed.kind, 'update-in-progress')
  assert.match(observed.message, /still holds the install/)

  for (const script of [spawnScript, probeScript]) {
    assert.match(script, /\$lockFile\.Lock\(1048576,17\)/)
    assert.match(script, /Combine\(\$gitdir,"hermes-update\.lock"\)/)
    assert.match(script, /\$checkoutRoots=@\(\[IO\.Path\]::Combine\(\$installRoot,"hermes-agent"\),/)
    assert.ok(script.includes("GetDirectoryName('C:\\src\\hermes\\venv\\Scripts\\python.exe')"))
  }

  const deleteAt = spawnScript.indexOf('[IO.File]::Delete($marker)')
  const guardAt = spawnScript.indexOf('if($verdict -eq "CLEAR" -and (Test-CheckoutLockHeld $checkoutRoots)){$verdict="HELD"}')
  assert.ok(guardAt > 0 && guardAt < deleteAt, 'the checkout probe must precede the dead-marker delete')
  assert.match(probeScript, /if\(\$result -eq "CLEAR" -and \(Test-CheckoutLockHeld \$checkoutRoots\)\)\{\$result="HELD"\}/)
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
  assert.match(script, /\$result=Get-MarkerVerdict \$text/)
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
  const explicitPythonCheck = script.indexOf('Assert-NoReparse $explicitPython $false')
  const envHome = script.indexOf('$hermesHome=$env:HERMES_HOME')
  // #118988: HERMES_HOME is trusted only when it is a directory on the remote; anything else
  // (stale User-scope value, client path leaked over SSH) falls back to the remote default.
  const envHomeGuard = script.indexOf('Test-Path -LiteralPath $hermesHome -PathType Container')
  const fallbackJoin = script.indexOf('Join-Path $hermesHome')
  const candidatePythonCheck = script.indexOf('Assert-NoReparse $candidatePython $true')
  const candidateSelection = script.indexOf('Get-Item -LiteralPath $candidate')
  const pythonJoin = script.indexOf('$python=[IO.Path]::Combine')
  const pythonCheck = script.indexOf('Assert-NoReparse $python $false')
  const output = script.indexOf('[ordered]@{')

  assert.ok(explicitCheck >= 0)
  assert.ok(explicitCheck < explicitPythonCheck)
  assert.ok(explicitPythonCheck < envHome)
  assert.ok(envHome < envHomeGuard)
  assert.ok(envHomeGuard < fallbackJoin)
  assert.ok(candidatePythonCheck >= 0)
  assert.ok(candidatePythonCheck < candidateSelection)
  assert.ok(pythonJoin >= 0)
  assert.ok(pythonJoin < pythonCheck)
  assert.ok(pythonCheck < output)
})

const CLEAN_PROBE_RESULT = JSON.stringify({
  os: 'Windows',
  arch: 'AMD64',
  hermesHome: 'C:\\h',
  hermesPath: 'C:\\h\\hermes.exe',
  python: 'C:\\h\\python.exe'
})

const CLIXML_PROGRESS =
  '#< CLIXML <Objs Version="1.1.0.1" xmlns="http://schemas.microsoft.com/maml/2004/10"><Obj S="progress" RefId="0">正在标准配置首次使用模块</Obj></Objs>'

test('Windows probe script silences the PowerShell progress stream', async () => {
  let script = ''
  await probeWindowsRemote(
    sshWith(async command => {
      script = Buffer.from(command.split(' ').at(-1) || '', 'base64').toString('utf16le')

      return CLEAN_PROBE_RESULT
    })
  )

  assert.ok(script.indexOf('$ProgressPreference="SilentlyContinue"') >= 0)
  assert.ok(script.indexOf('$ProgressPreference') < script.indexOf('$ErrorActionPreference'))
})

test('Windows probe tolerates CLIXML progress-stream pollution around the probe JSON', async () => {
  const pollutedOutputs = [
    `${CLIXML_PROGRESS}\r\n${CLEAN_PROBE_RESULT}`,
    `${CLIXML_PROGRESS}${CLEAN_PROBE_RESULT}`,
    `\uFEFF${CLEAN_PROBE_RESULT}\r\n${CLIXML_PROGRESS}`
  ]

  for (const output of pollutedOutputs) {
    const parsed = await probeWindowsRemote(sshWith(async () => output))

    assert.equal(parsed.os, 'Windows')
    assert.equal(parsed.arch, 'AMD64')
    assert.equal(parsed.hermesPath, 'C:\\h\\hermes.exe')
    assert.equal(parsed.python, 'C:\\h\\python.exe')
  }
})

test('Windows probe still rejects output that is not platform JSON', async () => {
  await assert.rejects(probeWindowsRemote(sshWith(async () => 'hermes is not installed on this host')))
  await assert.rejects(
    probeWindowsRemote(
      sshWith(async () => JSON.stringify({ os: 'Windows' }) + '\n' + JSON.stringify({ unrelated: true }))
    )
  )
})

test('the update marker gate stays CLEAR when CLIXML lands after the final Write-Output', async () => {
  // The gate is fail-closed: it only accepts a strict `CLEAR` from the last
  // line. A progress record serialized after `Write-Output $result` used to win
  // the .pop() and refuse SSH startup as 'update-in-progress' — a misleading
  // safety verdict from a cosmetic stdout race, not an actual update.
  for (const observation of [
    `CLEAR\r\n${CLIXML_PROGRESS}`,
    `${CLIXML_PROGRESS}\r\nCLEAR`,
    `\uFEFFCLEAR\r\n${CLIXML_PROGRESS}\r\n${CLIXML_PROGRESS}`
  ]) {
    await assertWindowsRemoteInstallUpdateClear(
      sshWith(async () => observation),
      'C:\\h'
    )
  }

  // The verdict itself is untouched: a LIVE marker still pauses startup.
  await assert.rejects(
    assertWindowsRemoteInstallUpdateClear(
      sshWith(async () => `LIVE:4242\r\n${CLIXML_PROGRESS}`),
      'C:\\h'
    ),
    (err: any) => err.kind === 'update-in-progress' && /4242/.test(err.message)
  )
})

test('every parsed Windows PowerShell script silences the progress stream', async () => {
  // The marker gate runs Add-Type (C# compilation) and the spawn/helper scripts
  // import the remote hermes_cli module — both emit progress records the
  // probe's Get-Item/Get-Command suppression knows nothing about, on the same
  // stdout the parsers read.
  const decode = (command: string) => Buffer.from(command.split(' ').at(-1) || '', 'base64').toString('utf16le')

  const scripts: string[] = []
  await assertWindowsRemoteInstallUpdateClear(
    sshWith(async command => {
      scripts.push(decode(command))

      return 'CLEAR'
    }),
    'C:\\h'
  )
  scripts.push(
    decode(atomicWindowsSpawnCommand({ hermesHome: 'C:\\h', python: 'C:\\py\\python.exe' })),
    decode(helperCommand({ python: 'C:\\py\\python.exe' }, 'inspect', []))
  )

  assert.equal(scripts.length, 3)

  for (const script of scripts) {
    assert.ok(script.indexOf('$ProgressPreference="SilentlyContinue"') >= 0, script.slice(0, 80))
    assert.ok(script.indexOf('$ProgressPreference') < script.indexOf('$ErrorActionPreference'), script.slice(0, 80))
  }
})

test('helper parsing ignores CLIXML progress blocks around its JSON line', async () => {
  const payload = JSON.stringify({ supported: true, version: '1.2.3' })

  for (const output of [`${CLIXML_PROGRESS}\r\n${payload}`, `\uFEFF${payload}\r\n${CLIXML_PROGRESS}`, `${payload}`]) {
    const parsed = await helper(
      sshWith(async () => output),
      { python: 'C:\\py\\python.exe' },
      'inspect',
      ['C:\\h\\hermes.exe']
    )

    assert.equal(parsed.supported, true)
    assert.equal(parsed.version, '1.2.3')
  }
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

test('platform detection preserves typed Windows probe failures', async () => {
  // `_fail()` (ssh-connection.ts) classifies exec deaths. The kinds that can
  // reach this catch with a probe error are `unknown` (signal death / stderr
  // classification) and `interactive-auth` (the Tailscale browser check
  // classifying a non-zero exec); the four TRANSPORT_KINDS are rethrown above.
  // `superseded` cannot reach it today — `exec()` passes no AbortSignal — but
  // it is pinned too: if a caller ever threads a signal through, the sentinel
  // must still surface as itself, never as the "unsupported operating system"
  // verdict.
  for (const kind of ['unknown', 'interactive-auth', 'superseded'] as const) {
    const probeErr: any = new Error('PowerShell remote command failed with exit code 1')
    probeErr.kind = kind

    await assert.rejects(
      detectRemotePlatform(
        sshWith(async command => {
          if (command.startsWith('uname ')) {
            throw new Error('PowerShell does not recognize uname')
          }

          throw probeErr
        })
      ),
      (err: any) =>
        err.kind === kind &&
        err.cause === probeErr &&
        /Windows remote probe failed/.test(err.message) &&
        /PowerShell remote command failed/.test(err.message) &&
        !/operating system is not supported/.test(err.message)
    )
  }
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
