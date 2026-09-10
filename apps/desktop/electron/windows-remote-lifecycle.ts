import crypto from 'node:crypto'

import { assertBootstrapNotSuperseded, redactSecrets, SSH_ERROR } from './ssh-connection'

const LOCKFILE_SCHEMA_VERSION = 2
const PROTOCOL_VERSION = 1
const READY_RE = /^HERMES_(?:BACKEND|DASHBOARD)_READY port=(\d+)/gm
const READY_POLL_INTERVAL_MS = 750

function psLiteral(value) {
  return `'${String(value).replace(/'/g, "''")}'`
}

function encodedPowerShell(script) {
  return Buffer.from(script, 'utf16le').toString('base64')
}

function powerShellCommand(script) {
  return `powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -EncodedCommand ${encodedPowerShell(script)}`
}

async function probeWindowsRemote(ssh, explicitHermesPath = '') {
  const explicit = psLiteral(explicitHermesPath)

  const script = [
    '$ErrorActionPreference="Stop"',
    'function Assert-NoReparse([string]$candidate,[bool]$allowMissing=$false){',
    'if([string]::IsNullOrWhiteSpace($candidate)){return}',
    '$current=[IO.Path]::GetFullPath($candidate);$first=$true',
    'while($true){',
    'try{$item=Get-Item -LiteralPath $current -Force -ErrorAction Stop}',
    'catch [Management.Automation.ItemNotFoundException]{if(-not $allowMissing -and $first){throw "Path was not found: $candidate"};$parent=[IO.Path]::GetDirectoryName($current);if(-not $parent -or $parent -eq $current){break};$current=$parent;$first=$false;continue}',
    'if(($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0){throw "Path contains a link or reparse point: $current"}',
    '$parent=$item.Parent.FullName;if(-not $parent -or $parent -eq $current){break};$current=$parent;$first=$false',
    '}',
    '}',
    `$explicit=${explicit}`,
    'if($explicit){Assert-NoReparse $explicit $false;$explicitPython=[IO.Path]::Combine([IO.Path]::GetDirectoryName($explicit), "python.exe");Assert-NoReparse $explicitPython $false}',
    '$hermesHome=$env:HERMES_HOME',
    'if(-not $hermesHome){$hermesHome=Join-Path $env:LOCALAPPDATA "hermes"}',
    'Assert-NoReparse $hermesHome $true',
    '$candidate=[IO.Path]::Combine($hermesHome, "hermes-agent\\venv\\Scripts\\hermes.exe")',
    '$candidatePython=[IO.Path]::Combine([IO.Path]::GetDirectoryName($candidate), "python.exe")',
    'Assert-NoReparse $candidate $true',
    'Assert-NoReparse $candidatePython $true',
    '$profileCandidate=[IO.Path]::Combine($HOME, "hermes-agent\\.venv\\Scripts\\hermes.exe")',
    '$profileCandidatePython=[IO.Path]::Combine([IO.Path]::GetDirectoryName($profileCandidate), "python.exe")',
    'Assert-NoReparse $profileCandidate $true',
    'Assert-NoReparse $profileCandidatePython $true',
    '$fallbackHomeCandidate=Join-Path $hermesHome "hermes-agent\\venv\\Scripts\\hermes.exe"',
    '$fallbackProfileCandidate=Join-Path $HOME "hermes-agent\\.venv\\Scripts\\hermes.exe"',
    '$candidates=@()',
    'if($explicit){$candidates+=$explicit}',
    '$cmd=Get-Command hermes.exe -ErrorAction SilentlyContinue',
    'if($cmd){Assert-NoReparse $cmd.Source $true;$cmdPython=[IO.Path]::Combine([IO.Path]::GetDirectoryName($cmd.Source), "python.exe");Assert-NoReparse $cmdPython $true;$candidates+=$cmd.Source}',
    '$candidates+=$fallbackHomeCandidate',
    '$candidates+=$fallbackProfileCandidate',
    '$hermes=$null',
    'foreach($candidate in $candidates){Assert-NoReparse $candidate $true;$candidatePython=[IO.Path]::Combine([IO.Path]::GetDirectoryName($candidate), "python.exe");Assert-NoReparse $candidatePython $true;try{$item=Get-Item -LiteralPath $candidate -Force -ErrorAction Stop;if(($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -eq 0 -and -not $item.PSIsContainer){$hermes=$item.FullName;break}}catch [Management.Automation.ItemNotFoundException]{continue}}',
    'if(-not $hermes){throw "Hermes is not installed on the remote Windows host."}',
    'Assert-NoReparse $hermes $false',
    'if($explicit -and $hermes -ne $explicit){throw "The configured Hermes path is not an executable file."}',
    '$python=[IO.Path]::Combine([IO.Path]::GetDirectoryName($hermes), "python.exe")',
    'Assert-NoReparse $python $false',
    '[ordered]@{os="Windows";arch=$env:PROCESSOR_ARCHITECTURE;hermesHome=$hermesHome;hermesPath=$hermes;python=$python}|ConvertTo-Json -Compress'
  ].join(';')

  return JSON.parse((await ssh.exec(powerShellCommand(script))).trim())
}

function windowsUpdateMarkerProbeCommand(hermesHome) {
  const script = [
    '$ErrorActionPreference="Stop"',
    `Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Runtime.InteropServices;
using Microsoft.Win32.SafeHandles;
public static class HermesMarkerNoFollow {
  [DllImport("kernel32.dll", CharSet=CharSet.Unicode, SetLastError=true)]
  private static extern SafeFileHandle CreateFile(string name, uint access, uint share, IntPtr security, uint creation, uint flags, IntPtr template);
  public static FileStream OpenRead(string name) {
    var handle=CreateFile(name, 0x80000000, 0x00000007, IntPtr.Zero, 3, 0x00200000, IntPtr.Zero);
    if(handle.IsInvalid) Marshal.ThrowExceptionForHR(Marshal.GetHRForLastWin32Error());
    return new FileStream(handle, FileAccess.Read);
  }
}
'@
`,
    'function Assert-NoReparse([string]$candidate,[bool]$allowMissing=$false){',
    'if([string]::IsNullOrWhiteSpace($candidate)){return}',
    '$current=[IO.Path]::GetFullPath($candidate);$first=$true',
    'while($true){',
    'try{$item=Get-Item -LiteralPath $current -Force -ErrorAction Stop}',
    'catch [Management.Automation.ItemNotFoundException]{if(-not $allowMissing -and $first){throw "Path was not found: $candidate"};$parent=[IO.Path]::GetDirectoryName($current);if(-not $parent -or $parent -eq $current){break};$current=$parent;$first=$false;continue}',
    'if(($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0){throw "Path contains a link or reparse point: $current"}',
    '$parent=$item.Parent.FullName;if(-not $parent -or $parent -eq $current){break};$current=$parent;$first=$false',
    '}',
    '}',
    `$home=${psLiteral(hermesHome)}`,
    '$installRoot=$home',
    '$parent=Split-Path -Parent $home',
    'if((Split-Path -Leaf $parent) -ieq "profiles"){$installRoot=Split-Path -Parent $parent}',
    '$marker=Join-Path $installRoot ".hermes-update-in-progress"',
    '$result="UNCERTAIN"',
    '$stream=$null;$memory=$null',
    'try{',
    'Assert-NoReparse $marker $true',
    'if(-not (Test-Path -LiteralPath $marker -PathType Leaf)){$result="CLEAR"}else{$stream=[HermesMarkerNoFollow]::OpenRead($marker)',
    'Assert-NoReparse $marker $false',
    '$memory=New-Object IO.MemoryStream;$stream.CopyTo($memory);$bytes=$memory.ToArray()',
    'if($bytes.Length -le 256){',
    '$utf8=[Text.UTF8Encoding]::new($false,$true)',
    '$text=$utf8.GetString($bytes)',
    "$match=[regex]::Match($text,'\\A([1-9][0-9]*)\\r?\\n([0-9]+)(?:\\r?\\n)?\\z')",
    '[uint32]$ownerPid=0',
    '[uint64]$lease=0',
    '$valid=$match.Success',
    'if($valid){$valid=[uint32]::TryParse($match.Groups[1].Value,[Globalization.NumberStyles]::None,[Globalization.CultureInfo]::InvariantCulture,[ref]$ownerPid)}',
    'if($valid){$valid=[uint64]::TryParse($match.Groups[2].Value,[Globalization.NumberStyles]::None,[Globalization.CultureInfo]::InvariantCulture,[ref]$lease)}',
    'if($valid -and $lease -le 9007199254740991){',
    'try{',
    '$process=[Diagnostics.Process]::GetProcessById([int]$ownerPid)',
    'try{if($process.HasExited){$result="CLEAR"}else{$result="LIVE:"+[string]$ownerPid}}finally{$process.Dispose()}',
    '}catch [ArgumentException]{$result="CLEAR"} catch{$result="UNCERTAIN"}',
    '}',
    '}',
    '}}catch [IO.FileNotFoundException]{$result="CLEAR"}catch{$result="UNCERTAIN"}finally{if($memory){$memory.Dispose()};if($stream){$stream.Dispose()}}',
    'Write-Output $result'
  ].join(';')

  return powerShellCommand(script)
}

/**
 * Fail-closed install marker gate for a fresh/relaunched Desktop process.
 * This uses only PowerShell/.NET and therefore never imports the remote
 * checkout while an updater may be replacing it.
 */
async function assertWindowsRemoteInstallUpdateClear(ssh, hermesHome) {
  let observation = ''

  try {
    observation =
      String(await ssh.exec(windowsUpdateMarkerProbeCommand(hermesHome)))
        .replace(/^\uFEFF/, '')
        .trim()
        .split(/\r?\n/)
        .pop() || ''
  } catch (cause) {
    const error: any = new Error('Could not prove that the remote Hermes install is clear for SSH startup.')
    error.kind = 'update-in-progress'
    error.cause = cause
    throw error
  }

  if (observation === 'CLEAR') {
    return
  }

  const live = /^LIVE:([1-9][0-9]*)$/.exec(observation)

  const error: any = new Error(
    live
      ? `Remote Hermes update process ${live[1]} is still running; SSH startup is paused.`
      : 'The remote Hermes update marker is unreadable or malformed; refusing SSH startup.'
  )

  error.kind = 'update-in-progress'
  throw error
}

const TRANSPORT_KINDS = new Set([
  SSH_ERROR.AUTH_FAILED,
  SSH_ERROR.HOST_KEY_CHANGED,
  SSH_ERROR.TIMEOUT,
  SSH_ERROR.UNREACHABLE
])

async function detectRemotePlatform(ssh, explicitHermesPath = '') {
  try {
    const output = (await ssh.exec('uname -s; uname -m')).trim().split('\n')

    if (output[0] === 'Linux' || output[0] === 'Darwin') {
      return { os: output[0], arch: output[1] || '' }
    }
  } catch (error: any) {
    // uname failing is the expected Windows fall-through; a TRANSPORT failure
    // (auth/host-key/timeout/unreachable) is not a platform verdict — surface it
    // as itself instead of letting the probe chain end in 'unsupported-platform'.
    if (TRANSPORT_KINDS.has(error?.kind)) {
      throw error
    }
  }

  try {
    return await probeWindowsRemote(ssh, explicitHermesPath)
  } catch (cause: any) {
    if (TRANSPORT_KINDS.has(cause?.kind)) {
      throw cause
    }

    // detail is remote-controlled output headed for the UI: redact + strip control chars.
    const detail = redactSecrets(String(cause?.message || cause || ''))
      // eslint-disable-next-line no-control-regex -- deliberately strip control chars from remote output
      .replace(/[\x00-\x1f\x7f]/g, ' ')
      .trim()

    const error: any = new Error(
      `The remote operating system is not supported by Desktop SSH.${detail ? ` (probe: ${detail.slice(0, 300)})` : ''}`
    )

    error.kind = 'unsupported-platform'
    error.cause = cause
    throw error
  }
}

function helperCommand(runtime, operation, args = []) {
  const argv = [runtime.python, '-m', 'hermes_cli.windows_ssh_runtime', operation, ...args]

  const script = [
    '$ErrorActionPreference="Stop"',
    `& ${argv.map(psLiteral).join(' ')}`,
    'if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}'
  ].join(';')

  return powerShellCommand(script)
}

async function helper(ssh, runtime, operation, args = [], stdinData?) {
  const output = await ssh.exec(helperCommand(runtime, operation, args), stdinData == null ? {} : { stdinData })

  const lines = String(output || '')
    .replace(/^\uFEFF/, '')
    .trim()
    .split(/\r?\n/)
    .filter(Boolean)

  const parsed = JSON.parse(lines[lines.length - 1] || 'null')

  if (parsed?.error) {
    throw new Error(parsed.error)
  }

  return parsed
}

function atomicWindowsSpawnCommand(runtime, reservation: any = {}) {
  const argv = [runtime.python, '-m', 'hermes_cli.windows_ssh_runtime', 'spawn']
  const helper = operation => [runtime.python, '-m', 'hermes_cli.windows_ssh_runtime', operation]

  const script = [
    '$ErrorActionPreference="Stop"',
    `$home=${psLiteral(runtime.hermesHome)}`,
    '$installRoot=$home',
    '$parent=Split-Path -Parent $home',
    'if((Split-Path -Leaf $parent) -ieq "profiles"){$installRoot=Split-Path -Parent $parent}',
    '$marker=Join-Path $installRoot ".hermes-update-in-progress"',
    '$mutexPath=$marker+".mutex"',
    '$mutex=[IO.File]::Open($mutexPath,[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::ReadWrite)',
    'try{',
    '  $mutex.Lock(0,1)',
    '  if([IO.File]::Exists($marker)){throw "remote update marker is present"}',
    reservation.ownershipId
      ? `  $existingLines=@(& ${helper('read-lock').map(psLiteral).join(' ')} ${psLiteral(reservation.ownershipId)}); $existingExit=$LASTEXITCODE; ` +
        '  if($existingExit -eq 0 -and $existingLines.Count -gt 0){try{$existing=$existingLines[-1]|ConvertFrom-Json}catch{$existing=$null}; ' +
        'if($existing -and [int]$existing.pid -gt 0){try{$p=[Diagnostics.Process]::GetProcessById([int]$existing.pid); ' +
        'if(-not $p.HasExited){[ordered]@{existing=$true}|ConvertTo-Json -Compress;exit 0}}catch{}finally{if($p){$p.Dispose()}}}; ' +
        `& ${helper('remove-lock').map(psLiteral).join(' ')} ${psLiteral(reservation.ownershipId)}|Out-Null}`
      : '',
    reservation.ownershipId
      ? `  $spawnLines=@(& ${argv.map(psLiteral).join(' ')}); $spawnExit=$LASTEXITCODE`
      : `  & ${argv.map(psLiteral).join(' ')}`,
    reservation.ownershipId
      ? '  if($spawnExit -ne 0){exit $spawnExit}'
      : '  if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}',
    reservation.ownershipId
      ? `  $spawned=$spawnLines[-1]|ConvertFrom-Json; $lock=[ordered]@{schemaVersion=2;protocolVersion=1;ownershipId=${psLiteral(reservation.ownershipId)};spawnNonce=${psLiteral(reservation.spawnNonce)};pid=[int]$spawned.pid;creationTimeNs=[string]$spawned.creationTimeNs;port=0;profile=${psLiteral(reservation.profile)};hermesPath=${psLiteral(reservation.hermesPath)};hermesHome=${psLiteral(reservation.hermesHome)};tokenFingerprint=${psLiteral(reservation.tokenFingerprint)};startedAt=${psLiteral(reservation.startedAt)}}|ConvertTo-Json -Compress; ` +
        `  & ${helper('write-lock').map(psLiteral).join(' ')} ${psLiteral(reservation.ownershipId)} $lock|Out-Null; if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}; $spawnLines|Write-Output`
      : '',
    '  if([IO.File]::Exists($marker)){throw "remote update marker claimed during backend spawn"}',
    '}finally{try{$mutex.Unlock(0,1)}catch{};$mutex.Dispose()}'
  ]
    .filter(line => line !== '')
    .join(';')

  return powerShellCommand(script)
}

async function atomicWindowsSpawn(ssh, runtime, stdinData, reservation: any = {}) {
  const output = await ssh.exec(atomicWindowsSpawnCommand(runtime, reservation), { stdinData })

  const lines = String(output || '')
    .replace(/^\uFEFF/, '')
    .trim()
    .split(/\r?\n/)
    .filter(Boolean)

  const parsed = JSON.parse(lines[lines.length - 1] || 'null')

  if (parsed?.error) {
    const error: any = new Error(parsed.error)
    error.kind = parsed.kind || 'remote-helper-error'
    throw error
  }

  return parsed
}

function fingerprintToken(token) {
  return crypto
    .createHash('sha256')
    .update(String(token || ''))
    .digest('hex')
    .slice(0, 32)
}

function validLock(lock, ownershipId) {
  // port 0 = spawn-in-progress record (written before readiness); a valid
  // ownership proof for cleanup, but never reusable.
  return Boolean(
    lock &&
    lock.schemaVersion === LOCKFILE_SCHEMA_VERSION &&
    lock.protocolVersion === PROTOCOL_VERSION &&
    lock.ownershipId === ownershipId &&
    /^[0-9a-f]{16}$/.test(lock.spawnNonce || '') &&
    Number.isInteger(lock.pid) &&
    lock.pid > 0 &&
    /^[0-9]{10,20}$/.test(lock.creationTimeNs || '') &&
    Number.isInteger(lock.port) &&
    lock.port >= 0 &&
    lock.port <= 65535 &&
    /^[0-9a-f]{32}$/.test(lock.tokenFingerprint || '') &&
    typeof lock.hermesPath === 'string' &&
    typeof lock.hermesHome === 'string'
  )
}

// #95532 parity (POSIX fix: 3ee0c62224, remote-lifecycle.ts's lockfileSkew).
// A lock read back from the remote that EXISTS but fails validLock() (wrong
// schema/protocol version, foreign ownershipId, malformed shape) is skew --
// most likely a different (e.g. forked) desktop build owns this remote, or
// the file is corrupt -- not the same as "no lock". Only called once the
// caller already knows validLock(lock, ownershipId) is false for a truthy
// lock, so this never needs to re-check the happy path; it just names which
// check failed, in validLock's own condition order, for the thrown error.
function windowsLockSkewReason(lock, ownershipId) {
  if (lock.schemaVersion !== LOCKFILE_SCHEMA_VERSION) {
    return `schema-version ${JSON.stringify(lock.schemaVersion ?? null)}`
  }

  if (lock.protocolVersion !== PROTOCOL_VERSION) {
    return `protocol-version ${JSON.stringify(lock.protocolVersion ?? null)}`
  }

  if (lock.ownershipId !== ownershipId) {
    return 'ownership-mismatch'
  }

  if (!/^[0-9a-f]{16}$/.test(lock.spawnNonce || '')) {
    return 'malformed-spawn-nonce'
  }

  if (!Number.isInteger(lock.pid) || lock.pid <= 0) {
    return 'malformed-pid'
  }

  if (!/^[0-9]{10,20}$/.test(lock.creationTimeNs || '')) {
    return 'malformed-creation-time'
  }

  if (!Number.isInteger(lock.port) || lock.port < 0 || lock.port > 65535) {
    return 'malformed-port'
  }

  if (!/^[0-9a-f]{32}$/.test(lock.tokenFingerprint || '')) {
    return 'malformed-token-fingerprint'
  }

  if (typeof lock.hermesPath !== 'string' || typeof lock.hermesHome !== 'string') {
    return 'malformed-path-fields'
  }

  return 'unknown-mismatch'
}

function reusableWindowsLock(lock, state, profile, reuseToken, runtime) {
  return Boolean(
    state.alive &&
    state.owned &&
    lock.port > 0 &&
    lock.profile === profile &&
    reuseToken &&
    lock.tokenFingerprint === fingerprintToken(reuseToken) &&
    lock.hermesPath === runtime.hermesPath &&
    lock.hermesHome === runtime.hermesHome
  )
}

async function processState(ssh, runtime, lock) {
  return helper(ssh, runtime, 'process-state', [
    String(lock.pid),
    String(lock.creationTimeNs),
    lock.hermesPath,
    lock.spawnNonce
  ])
}

async function cleanupOwned(ssh, runtime, ownershipId, lock) {
  const attempt = async fn => {
    try {
      await fn()
    } catch {
      void 0
    }
  }

  if (lock) {
    const state = await processState(ssh, runtime, lock)

    if (state.alive && state.owned) {
      // Deliberately not attempt()-wrapped: a thrown terminate must abort before
      // remove-lock, or a live backend is orphaned with no lock to reclaim it.
      await helper(ssh, runtime, 'terminate', [
        String(lock.pid),
        String(lock.creationTimeNs),
        lock.hermesPath,
        lock.spawnNonce
      ])
    }

    if (lock.spawnNonce) {
      await attempt(() => helper(ssh, runtime, 'remove-token', [ownershipId, lock.spawnNonce]))
      await attempt(() => helper(ssh, runtime, 'remove-log', [ownershipId, lock.spawnNonce]))
    }
  }

  await attempt(() => helper(ssh, runtime, 'remove-lock', [ownershipId]))
}

function windowsLockMatchesManagedUpdateScope(lock, expected) {
  return Boolean(
    lock &&
    expected &&
    lock.ownershipId === expected.ownershipId &&
    lock.pid === expected.pid &&
    lock.spawnNonce === expected.spawnNonce &&
    lock.creationTimeNs === expected.creationTimeNs &&
    lock.profile === expected.profile &&
    lock.hermesPath === expected.hermesPath &&
    lock.hermesHome === expected.hermesHome
  )
}

/**
 * Terminate a Windows serve only after the persisted ownership record and the
 * kernel creation-time proof still match the exact scope Desktop connected.
 * Leave the record in place for the post-update reconnect to reclaim; this
 * prevents a delayed cleanup from deleting a replacement owner's record.
 */
async function terminateOwnedWindowsDashboardForUpdate(ssh, runtime, expected) {
  let lock = await helper(ssh, runtime, 'read-lock', [expected?.ownershipId || ''])

  if (!validLock(lock, expected?.ownershipId) || !windowsLockMatchesManagedUpdateScope(lock, expected)) {
    const error: any = new Error('The remote Windows ownership record changed before the managed update.')
    error.kind = 'ownership-changed'
    throw error
  }

  let state = await processState(ssh, runtime, lock)

  if (state.indeterminate) {
    const error: any = new Error('Could not prove the remote Windows process identity for the managed update.')
    error.kind = 'transient-transport-error'
    throw error
  }

  if (!state.alive) {
    return { pid: lock.pid, terminated: false, alreadyStopped: true }
  }

  if (!state.owned) {
    const error: any = new Error('Refusing to terminate a remote Windows process whose ownership is unproven.')
    error.kind = 'foreign-backend'
    throw error
  }

  // Fence the proof against a record replacement before signalling.
  lock = await helper(ssh, runtime, 'read-lock', [expected.ownershipId])

  if (!validLock(lock, expected.ownershipId) || !windowsLockMatchesManagedUpdateScope(lock, expected)) {
    const error: any = new Error('The remote Windows ownership record changed during process verification.')
    error.kind = 'ownership-changed'
    throw error
  }

  state = await processState(ssh, runtime, lock)

  if (state.indeterminate || !state.alive || !state.owned) {
    const error: any = new Error('The remote Windows process identity changed during managed update drain.')
    error.kind = state.indeterminate ? 'transient-transport-error' : 'ownership-changed'
    throw error
  }

  await helper(ssh, runtime, 'terminate', [
    String(lock.pid),
    String(lock.creationTimeNs),
    lock.hermesPath,
    lock.spawnNonce
  ])

  return { pid: lock.pid, terminated: true, alreadyStopped: false }
}

async function waitReady(ssh, runtime, ownershipId, lock, timeoutMs, signal) {
  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    assertBootstrapNotSuperseded(signal)
    let state

    try {
      state = await processState(ssh, runtime, lock)
    } catch {
      await new Promise(resolve => setTimeout(resolve, READY_POLL_INTERVAL_MS))

      continue
    }

    if (!state.indeterminate && (!state.alive || !state.owned)) {
      let detail = ''

      try {
        detail = (await helper(ssh, runtime, 'read-log', [ownershipId, lock.spawnNonce]))?.content || ''
      } catch {
        void 0
      }

      const error: any = new Error(
        `Remote Windows backend exited before announcing its port. state=${JSON.stringify(state)} ${detail.slice(-2000)}`
      )

      error.kind = 'spawn-failed'
      throw error
    }

    let content = ''

    try {
      content = (await helper(ssh, runtime, 'read-log', [ownershipId, lock.spawnNonce]))?.content || ''
    } catch {
      void 0
    }

    let port

    for (const match of content.matchAll(READY_RE)) {
      port = Number(match[1])
    }

    if (port) {
      return port
    }

    await new Promise(resolve => setTimeout(resolve, READY_POLL_INTERVAL_MS))
  }

  const error: any = new Error(`Timed out waiting for the remote Windows backend (${timeoutMs}ms).`)
  error.kind = 'ready-timeout'
  throw error
}

async function waitForWindowsSpawnCompletion(ssh, runtime, ownershipId, timeoutMs) {
  const deadline = Date.now() + timeoutMs

  while (Date.now() < deadline) {
    const lock = await helper(ssh, runtime, 'read-lock', [ownershipId])

    if (!lock) {
      return false
    }

    if (validLock(lock, ownershipId) && lock.port > 0) {
      return true
    }

    await new Promise(resolve => setTimeout(resolve, READY_POLL_INTERVAL_MS))
  }

  const error: any = new Error('Timed out waiting for the concurrent Windows SSH connection to publish its backend.')
  error.kind = 'spawn-failed'
  throw error
}

async function connectWindowsRemote(deps) {
  const {
    ssh,
    ownershipId,
    profile = '',
    remoteHermesPath = '',
    reuseToken = '',
    signal,
    pickLocalPort,
    forward,
    cancelForward,
    waitForHermes,
    probeReuseProof,
    rememberLog = () => {},
    readyTimeoutMs = 45_000
  } = deps

  assertBootstrapNotSuperseded(signal)
  const runtime = await probeWindowsRemote(ssh, remoteHermesPath)
  await assertWindowsRemoteInstallUpdateClear(ssh, runtime.hermesHome)
  const inspection = await helper(ssh, runtime, 'inspect', [runtime.hermesPath])

  if (!inspection.supported) {
    const error: any = new Error('Update Hermes on the remote Windows host before connecting with Desktop SSH.')
    error.kind = 'update-required'
    throw error
  }

  runtime.hermesPath = inspection.path
  const hermesVersion = inspection.version || ''
  rememberLog(`[ssh-lifecycle] remote platform Windows/${runtime.arch}`)
  rememberLog(`[ssh-lifecycle] located hermes at ${runtime.hermesPath}`)

  await assertWindowsRemoteInstallUpdateClear(ssh, runtime.hermesHome)
  const lock = await helper(ssh, runtime, 'read-lock', [ownershipId])

  if (validLock(lock, ownershipId)) {
    const state = await processState(ssh, runtime, lock)

    if (state.indeterminate) {
      const error: any = new Error('Could not determine the state of the existing remote backend.')
      error.kind = 'transient-transport-error'
      throw error
    }

    const reusable = reusableWindowsLock(lock, state, profile, reuseToken, runtime)

    if (reusable) {
      await assertWindowsRemoteInstallUpdateClear(ssh, runtime.hermesHome)
      const localPort = await pickLocalPort()
      await forward(localPort, lock.port)

      try {
        const baseUrl = `http://127.0.0.1:${localPort}`
        const classification = await probeReuseProof(baseUrl, reuseToken, lock.spawnNonce)

        if (classification === 'authenticated-ok') {
          return {
            baseUrl,
            token: reuseToken,
            remotePort: lock.port,
            localPort,
            pid: lock.pid,
            reused: true,
            platform: { os: 'Windows', arch: runtime.arch },
            hermesPath: runtime.hermesPath,
            hermesVersion,
            ownershipId,
            spawnNonce: lock.spawnNonce,
            creationTimeNs: lock.creationTimeNs,
            hermesHome: runtime.hermesHome,
            pythonPath: runtime.python
          }
        }

        if (classification !== 'authenticated-stale') {
          throw new Error('Invalid SSH reuse classification.')
        }

        await cancelForward(localPort, lock.port)
        await assertWindowsRemoteInstallUpdateClear(ssh, runtime.hermesHome)
        await cleanupOwned(ssh, runtime, ownershipId, lock)
      } catch (error) {
        await cancelForward(localPort, lock.port)
        throw error
      }
    } else {
      await assertWindowsRemoteInstallUpdateClear(ssh, runtime.hermesHome)
      await cleanupOwned(ssh, runtime, ownershipId, lock)
    }
  } else if (lock) {
    // #95532: the lockfile exists but was written by a different (fork)
    // build or is corrupt. FAIL CLOSED: no reap, no removal, no overwrite,
    // no spawn on top of foreign live state -- deleting it here is how a
    // live tunnel some other build depends on gets killed (mirrors the
    // POSIX-side fix in remote-lifecycle.ts).
    const reason = windowsLockSkewReason(lock, ownershipId)
    const error: any = new Error(
      `The remote ownership record for this connection does not match this Hermes Desktop build (${reason}). ` +
        'It was probably written by a different or modified desktop build sharing this remote, or the file is corrupt. ' +
        'Refusing to remove or overwrite it -- that could kill a live SSH backend owned by another build. ' +
        'If nothing else uses this remote, delete the remote lock file and reconnect.'
    )
    error.kind = 'remote-lockfile-skew'
    throw error
  }

  assertBootstrapNotSuperseded(signal)
  await assertWindowsRemoteInstallUpdateClear(ssh, runtime.hermesHome)
  const token = crypto.randomBytes(32).toString('hex')
  const spawnNonce = crypto.randomBytes(8).toString('hex')
  await helper(ssh, runtime, 'upload-token', [ownershipId, spawnNonce], token)
  const startedAt = new Date().toISOString()
  const tokenFingerprint = fingerprintToken(token)
  let spawned

  try {
    await assertWindowsRemoteInstallUpdateClear(ssh, runtime.hermesHome)
    spawned = await atomicWindowsSpawn(
      ssh,
      runtime,
      JSON.stringify({ ownershipId, spawnNonce, profile, hermesPath: runtime.hermesPath }),
      {
        ownershipId,
        spawnNonce,
        profile,
        hermesPath: runtime.hermesPath,
        hermesHome: runtime.hermesHome,
        tokenFingerprint,
        startedAt
      }
    )
  } catch (error) {
    await helper(ssh, runtime, 'remove-token', [ownershipId, spawnNonce])
    throw error
  }

  if (spawned.existing) {
    await helper(ssh, runtime, 'remove-token', [ownershipId, spawnNonce])

    if (!reuseToken) {
      const error: any = new Error(
        'Another SSH connection owns this remote dashboard; a session token is required to reuse it.'
      )

      error.kind = 'remote-ownership-contended'
      throw error
    }

    const published = await waitForWindowsSpawnCompletion(ssh, runtime, ownershipId, readyTimeoutMs)

    if (!published) {
      return connectWindowsRemote({ ...deps, reuseToken })
    }

    return connectWindowsRemote({ ...deps, reuseToken })
  }

  const owned = {
    schemaVersion: LOCKFILE_SCHEMA_VERSION,
    protocolVersion: PROTOCOL_VERSION,
    ownershipId,
    spawnNonce,
    pid: spawned.pid,
    creationTimeNs: spawned.creationTimeNs,
    port: 0,
    profile,
    hermesPath: runtime.hermesPath,
    hermesHome: runtime.hermesHome,
    tokenFingerprint,
    startedAt
  }

  let localPort = 0
  let remotePort = 0

  try {
    // Write the ownership record IMMEDIATELY (port=0): if this attempt is
    // superseded before readiness and cleanup cannot reach the box, the next
    // connect still finds the lock and reaps the process by exact ownership.
    // Inside the try: if this write itself fails, the catch still kills the
    // just-spawned process via the in-memory record.
    await helper(ssh, runtime, 'write-lock', [ownershipId], JSON.stringify(owned))
    remotePort = await waitReady(ssh, runtime, ownershipId, owned, readyTimeoutMs, signal)
    localPort = await pickLocalPort()
    await forward(localPort, remotePort)
    const baseUrl = `http://127.0.0.1:${localPort}`
    await waitForHermes(baseUrl, token)
    assertBootstrapNotSuperseded(signal)
    await helper(ssh, runtime, 'write-lock', [ownershipId], JSON.stringify({ ...owned, port: remotePort }))

    return {
      baseUrl,
      token,
      remotePort,
      localPort,
      pid: spawned.pid,
      reused: false,
      platform: { os: 'Windows', arch: runtime.arch },
      hermesPath: runtime.hermesPath,
      hermesVersion,
      ownershipId,
      spawnNonce,
      creationTimeNs: spawned.creationTimeNs,
      hermesHome: runtime.hermesHome,
      pythonPath: runtime.python
    }
  } catch (error) {
    if (localPort && remotePort) {
      await cancelForward(localPort, remotePort)
    }

    await cleanupOwned(ssh, runtime, ownershipId, owned)
    throw error
  }
}

function buildWindowsInteractiveCommand(remoteCwd = '') {
  const cwd = String(remoteCwd || '').trim()
  const script = ['$ErrorActionPreference="Stop"']

  if (cwd) {
    script.push(
      `if(Test-Path -LiteralPath ${psLiteral(cwd)} -PathType Container){Set-Location -LiteralPath ${psLiteral(cwd)}}`
    )
  }

  script.push('$host.UI.RawUI.WindowTitle="Hermes SSH"', 'powershell.exe -NoLogo')

  return powerShellCommand(script.join(';'))
}

export {
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
  validLock,
  windowsLockSkewReason
}
