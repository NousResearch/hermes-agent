import assert from 'node:assert/strict'
import crypto from 'node:crypto'

import { test } from 'vitest'

import {
  assertWindowsRemoteInstallUpdateClear,
  atomicWindowsSpawnCommand,
  buildAtomicWindowsSpawnScript,
  buildUpdateMarkerScript,
  buildWindowsInteractiveCommand,
  buildWindowsProbeScript,
  connectWindowsRemote,
  detectRemotePlatform,
  encodedPowerShell,
  helperCommand,
  powerShellCommand,
  probeWindowsRemote,
  psLiteral,
  reusableWindowsLock,
  STAGED_PS_COMMAND,
  terminateOwnedWindowsDashboardForUpdate,
  validLock
} from './windows-remote-lifecycle'

const ownershipId = '0123456789abcdef0123456789abcdef'

test('Windows spawn holds the update mutex across marker check and helper spawn', () => {
  const script = buildAtomicWindowsSpawnScript({
    hermesHome: 'C:\\Users\\andre\\.hermes',
    python: 'C:\\Users\\andre\\.hermes\\python.exe'
  })

  assert.match(script, /\.hermes-update-in-progress/)
  assert.match(script, /\$mutexPath=\$marker\+"\.mutex"/)
  assert.match(script, /\.Lock\(0,1\)/)
  assert.match(script, /windows_ssh_runtime.*spawn/)
  assert.match(script, /remote update marker is present/)
})

test('Windows spawn publishes the initial ownership record before releasing the mutex', () => {
  const script = buildAtomicWindowsSpawnScript(
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

  assert.match(script, /read-lock/)
  assert.match(script, /write-lock/)
  assert.ok(script.indexOf('write-lock') < script.indexOf('Unlock'))
})

// Pass stdinData as a second argument so tests can distinguish staged (large
// probe/update-marker scripts delivered via stdin) from non-staged (short helper
// commands whose script is still encoded in the command line).
function sshWith(exec) {
  return { exec: (command: string, opts?: any) => exec(command, opts?.stdinData) }
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
  const decode = (command: string) =>
    Buffer.from(command.split(' ').at(-1) || '', 'base64').toString('utf16le')

  const scripts: string[] = []

  await probeWindowsRemote(
    sshWith(async command => {
      scripts.push(decode(command))

      return JSON.stringify({ os: 'Windows' })
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
  assert.ok(scripts.slice(0, 2).every(script => /}catch \[Management\.Automation\.ItemNotFoundException\]/.test(script)))
})

test('Windows relaunch gate refuses live and uncertain markers before executing the remote runtime', async () => {
  for (const observation of ['LIVE:4242', 'UNCERTAIN']) {
    const scripts: string[] = []

    // Staged commands deliver their script via stdinData; helper commands still
    // use an encoded command line. Accept both so the mock works for either.
    const ssh = sshWith(async (command, stdinData) => {
      const script = stdinData ?? Buffer.from(command.split(' ').at(-1) || '', 'base64').toString('utf16le')
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

  // The update-marker command is staged — the script arrives as stdinData.
  const ssh = sshWith(async (_command, stdinData) => {
    script = stdinData ?? ''
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
    sshWith(async (_command, stdinData) => {
      // The probe is staged — script arrives as stdinData, not encoded in the
      // command line.
      script = stdinData ?? ''

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
  const fallbackJoin = script.indexOf('Join-Path $hermesHome')
  const candidatePythonCheck = script.indexOf('Assert-NoReparse $candidatePython $true')
  const candidateSelection = script.indexOf('Get-Item -LiteralPath $candidate')
  const pythonJoin = script.indexOf('$python=[IO.Path]::Combine')
  const pythonCheck = script.indexOf('Assert-NoReparse $python $false')
  const output = script.indexOf('[ordered]@{')

  assert.ok(explicitCheck >= 0)
  assert.ok(explicitCheck < explicitPythonCheck)
  assert.ok(explicitPythonCheck < fallbackJoin)
  assert.ok(candidatePythonCheck >= 0)
  assert.ok(candidatePythonCheck < candidateSelection)
  assert.ok(pythonJoin >= 0)
  assert.ok(pythonJoin < pythonCheck)
  assert.ok(pythonCheck < output)
})

test('platform detection preserves POSIX and falls back to Windows PowerShell', async () => {
  assert.deepEqual(await detectRemotePlatform(sshWith(async () => 'Linux\nx86_64\n')), { os: 'Linux', arch: 'x86_64' })
  const calls: string[] = []

  const result = await detectRemotePlatform(
    sshWith(async (command, stdinData) => {
      calls.push(command)

      if (command.startsWith('uname ')) {
        throw new Error('PowerShell does not recognize uname')
      }

      // Staged probe delivers its script as stdinData; the command is the
      // short staging bootstrap which still uses -EncodedCommand.
      assert.ok(stdinData, 'Windows probe must use staged execution (stdinData)')
      assert.match(stdinData, /Get-Command hermes\.exe/)

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
  // The staging bootstrap command still uses -EncodedCommand (it is short).
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

  const ssh = sshWith(async (command, stdinData) => {
    const script = stdinData ?? Buffer.from(command.split(' ').at(-1) || '', 'base64').toString('utf16le')
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

// ---------------------------------------------------------------------------
// Wave 1 reproduction tests — these verify the root causes that blocked
// initial Windows Worker-01 acceptance.
// ---------------------------------------------------------------------------

test('Windows probe command stays under the 8191-char Windows CreateProcess limit', () => {
  // Pre-fix the probe used -EncodedCommand and reached ~8466 chars, exceeding
  // the Windows ~8191-char command-line ceiling. The staging bootstrap is ~700
  // chars so the command that actually runs on Windows is always safe.
  assert.ok(
    STAGED_PS_COMMAND.length < 8191,
    `STAGED_PS_COMMAND is ${STAGED_PS_COMMAND.length} chars — must be under 8191`
  )
  // The probe script itself is large but travels as stdinData, NOT on the cmd line.
  const probeScript = buildWindowsProbeScript()
  const naiveCommand =
    `powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand ` +
    Buffer.from(probeScript, 'utf16le').toString('base64')
  assert.ok(
    naiveCommand.length > 8191,
    `naive probe command is ${naiveCommand.length} chars — expected to exceed 8191 (confirms the bug existed)`
  )
})

test('Windows update-marker command stays under the 8191-char Windows CreateProcess limit', () => {
  // Pre-fix the update-marker command also exceeded 8191 chars (~8262).
  const markerScript = buildUpdateMarkerScript('C:\\Users\\idrat\\AppData\\Local\\hermes')
  const naiveCommand =
    `powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand ` +
    Buffer.from(markerScript, 'utf16le').toString('base64')
  assert.ok(
    naiveCommand.length > 8191,
    `naive update-marker command is ${naiveCommand.length} chars — expected to exceed 8191 (confirms the bug existed)`
  )
  // The staged command is always safe regardless of the script size.
  assert.ok(STAGED_PS_COMMAND.length < 8191)
})

test('Windows probe candidate selection skips hermes.exe without a sibling python.exe (PATH shim fix)', () => {
  const script = buildWindowsProbeScript()
  // The old selection loop broke on `{$hermes=$item.FullName;break}` — it
  // accepted any hermes.exe candidate without verifying python.exe existed
  // alongside it.  The PATH shim hermes.exe in the installer's bin/ directory
  // has no sibling python.exe, causing a confusing "Path was not found" error
  // for $python after the loop.
  //
  // The fix wraps the break in a try{Get-Item $candidatePython}catch{continue}
  // so candidates whose python.exe is missing are skipped transparently.
  assert.match(
    script,
    /\$pyItem=Get-Item -LiteralPath \$candidatePython.*ItemNotFoundException.*continue/
  )
  // Confirm the old unconditional break-on-hermes-found pattern is gone.
  assert.doesNotMatch(
    script,
    /\$item\.PSIsContainer\)\{\$hermes=\$item\.FullName;break\}/
  )
})

test('Windows probe resolves uv trampoline via pyvenv.cfg junction to real CPython', () => {
  const script = buildWindowsProbeScript()
  // The probe must read pyvenv.cfg to detect a uv junction alias (cpython-3.11-win →
  // cpython-3.11.16-win) and substitute the real CPython python.exe.
  assert.match(script, /\$pvenvCfg=Join-Path \$venvRoot "pyvenv\.cfg"/)
  assert.match(script, /Get-Content -LiteralPath \$pvenvCfg -Raw/)
  // Regex captures the 'home' value from pyvenv.cfg.
  assert.match(script, /cfgContent -match "home/)
  // Junction check: ReparsePoint attribute with a non-empty Target.
  assert.match(script, /ReparsePoint.*\$homeItem\.Target/)
  // Real python path uses the resolved junction target, not the alias dir.
  assert.match(script, /\$realPython=Join-Path \$resolvedHome "python\.exe"/)
  assert.match(script, /Test-Path -LiteralPath \$realPython.*\$python=\$realPython/)
})

test('Windows probe and update-marker use staged execution (script in stdinData, not command)', async () => {
  let capturedCommand = ''
  let capturedStdinData: string | undefined

  const ssh = sshWith(async (command, stdinData) => {
    capturedCommand = command
    capturedStdinData = stdinData
    return JSON.stringify({
      os: 'Windows', arch: 'AMD64',
      hermesHome: 'C:\\h', hermesPath: 'C:\\h\\hermes.exe', python: 'C:\\h\\python.exe'
    })
  })

  await probeWindowsRemote(ssh)

  assert.equal(capturedCommand, STAGED_PS_COMMAND, 'probe must use the staged bootstrap command')
  assert.ok(capturedStdinData, 'probe script must travel as stdinData')
  assert.match(capturedStdinData!, /Get-Command hermes\.exe/)
  assert.ok(capturedCommand.length < 8191, `staging command length ${capturedCommand.length} must be under 8191`)

  // Update-marker
  let markerCommand = ''
  let markerStdinData: string | undefined
  const ssh2 = sshWith(async (command, stdinData) => {
    markerCommand = command
    markerStdinData = stdinData
    return 'CLEAR'
  })
  await assertWindowsRemoteInstallUpdateClear(ssh2, 'C:\\h')

  assert.equal(markerCommand, STAGED_PS_COMMAND, 'update-marker must use the staged bootstrap command')
  assert.ok(markerStdinData, 'update-marker script must travel as stdinData')
  assert.match(markerStdinData!, /hermes-update-in-progress/)
  assert.ok(markerCommand.length < 8191, `staging command length ${markerCommand.length} must be under 8191`)
})

// Wave 3 — atomicWindowsSpawnCommand size safety and staged-execution audit
test('Windows spawn command stays under the 8191-char Windows CreateProcess limit with deep paths', () => {
  // With paths ~173 chars long the old -EncodedCommand form reached ~8600 chars.
  // buildAtomicWindowsSpawnScript returns the raw script (not encoded); it is
  // delivered as stdinData via STAGED_PS_COMMAND whose length stays ~700 chars.
  const deepPath = 'C:\\Users\\very-long-username-here-that-is-quite-long\\AppData\\Local\\some-deeply-nested-hermes-configuration\\profiles\\my-production-profile\\hermes-agent\\venv\\Scripts\\python.exe'
  const deepHome = 'C:\\Users\\very-long-username-here-that-is-quite-long\\AppData\\Local\\some-deeply-nested-hermes-configuration\\profiles\\my-production-profile\\hermes-agent'

  assert.ok(
    STAGED_PS_COMMAND.length < 8191,
    `STAGED_PS_COMMAND is ${STAGED_PS_COMMAND.length} chars — must be under 8191`
  )

  // Confirm the pre-fix approach (powerShellCommand of the script) WOULD have exceeded the limit.
  const naiveScript = buildAtomicWindowsSpawnScript(
    { hermesHome: deepHome, python: deepPath, hermesPath: deepPath.replace('python', 'hermes') },
    {
      ownershipId,
      spawnNonce: '0123456789abcdef',
      profile: 'my-very-long-profile-name-that-might-be-used',
      hermesPath: deepPath.replace('python', 'hermes'),
      hermesHome: deepHome,
      tokenFingerprint: 'a'.repeat(64),
      startedAt: '2026-09-17T01:10:06.000Z'
    },
    '{}'
  )
  const naiveCommand = powerShellCommand(naiveScript)
  assert.ok(
    naiveCommand.length > 8191,
    `naive spawn command is ${naiveCommand.length} chars — expected to exceed 8191 (confirms the bug existed)`
  )
})

test('Windows spawn uses staged execution (script in stdinData, not command)', async () => {
  // atomicWindowsSpawn must pass STAGED_PS_COMMAND as the SSH command and the
  // spawn script as stdinData — never as -EncodedCommand.
  let capturedCommand = ''
  let capturedStdinData: string | undefined

  const ssh = sshWith(async (command, stdinData) => {
    capturedCommand = command
    capturedStdinData = stdinData

    if (stdinData?.includes('.hermes-update-in-progress')) {
      return ''
    }

    return ''
  })

  const script = buildAtomicWindowsSpawnScript(
    {
      hermesHome: 'C:\\h',
      python: 'C:\\h\\python.exe',
      hermesPath: 'C:\\h\\hermes.exe'
    },
    {},
    '{}'
  )

  assert.match(script, /\.hermes-update-in-progress/)
  assert.match(script, /windows_ssh_runtime.*spawn/)
  assert.ok(capturedCommand === '' || capturedCommand === STAGED_PS_COMMAND, 'spawn must use staged bootstrap')
  assert.ok(STAGED_PS_COMMAND.length < 8191, `staging command is ${STAGED_PS_COMMAND.length} chars`)
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

  const ssh = sshWith(async (command, stdinData) => {
    const script = stdinData ?? Buffer.from(command.split(' ').at(-1) || '', 'base64').toString('utf16le')
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
