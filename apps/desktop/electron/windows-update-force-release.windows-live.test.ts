/**
 * windows-update-force-release.windows-live.test.ts
 *
 * LIVE Windows proof of the termination boundary: real PowerShell children,
 * real WMI-brokered processes, real job objects, real watcher bridges.
 *
 * These tests used to sit in windows-update-force-release.test.ts behind
 * `if (process.platform !== 'win32') {return}`, which reports PASS on Linux CI
 * having asserted nothing. They now live in their own `*.windows-live` file
 * (the convention backend-release-gate already uses) and are SKIPPED off
 * Windows, so a green run on Linux never claims this boundary was covered.
 *
 * Every child is spawned under a private temp directory and killed by the
 * test that spawned it.
 */

import assert from 'node:assert/strict'
import { type ChildProcess, execFile, spawn } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { promisify } from 'node:util'

import { describe, it } from 'vitest'

import { queryWindowsProcessCreatedAt } from './windows-process-identity'
import {
  buildExactTerminateScript,
  identitiesStillPresent,
  parseWrapperProcessMarker,
  runPowerShellWithHardBoundary,
  snapshotProcessTreeIdentities,
  TERMINATE_ACCESS_DENIED_CLASSIFIER,
  TERMINATE_JOB_WATCHER_BRIDGE,
  TERMINATE_JOB_WATCHER_COMMAND,
  TERMINATE_JOB_WRAPPER_COMMAND,
  terminateWindowsHolderExact
} from './windows-process-terminate'
import {
  type ForceReleaseHolder
} from './windows-update-force-release'

const execFileAsync = promisify(execFile)
// Absolute path for the same reason production uses one: PATH is not trustworthy here.
const taskkillPath = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'taskkill.exe')
const isWindows = process.platform === 'win32'

/**
 * TEST-ONLY fault injection.
 *
 * These used to be `forcePrimarySnapshotFailure` / `pausePhase` / `pausePid` /
 * `phaseMarkerPath` options on `buildExactTerminateScript`, which compiled a
 * `Pause-BoundaryTest` function and a `Start-Sleep -Seconds 30` into EVERY
 * real termination script the updater ran. The seam belongs here: these
 * functions patch the generated text, so production carries no test knobs.
 */
type BoundaryPhase =
  | 'after-root-assignment'
  | 'after-root-suspension'
  | 'after-child-assignment'
  | 'after-child-suspension'

const PHASE_ANCHORS: Record<BoundaryPhase, { anchor: string; pidVar: string; indent: string }> = {
  'after-root-assignment': {
    anchor: '  if (Assign-ContainedProcess $rootHandle $pidTarget) { [void]$contained.Add($pidTarget) }\n',
    pidVar: '$pidTarget',
    indent: '  '
  },
  'after-root-suspension': {
    anchor: '  [void]$suspended.Add($pidTarget)\n',
    pidVar: '$pidTarget',
    indent: '  '
  },
  'after-child-assignment': {
    anchor: '      if (Assign-ContainedProcess $childHandle $childPid) { [void]$contained.Add($childPid) }\n',
    pidVar: '$childPid',
    indent: '      '
  },
  'after-child-suspension': {
    anchor: '      [void]$suspended.Add($childPid)\n',
    pidVar: '$childPid',
    indent: '      '
  }
}

const psQuote = (value: string) => `'${value.replace(/'/g, "''")}'`

/** Publish a marker at `phase` for `pausePid`, then stall inside the boundary. */
function withBoundaryPause(script: string, phase: BoundaryPhase, pausePid: number, markerPath: string): string {
  const { anchor, pidVar, indent } = PHASE_ANCHORS[phase]

  assert.ok(script.includes(anchor), `terminate script no longer contains the ${phase} anchor`)

  const injected =
    `${indent}if (${pidVar} -eq ${Math.trunc(pausePid)}) {\n` +
    `${indent}  $hermesTestMarkerTemp = ${psQuote(`${markerPath}.tmp`)}\n` +
    `${indent}  [IO.File]::WriteAllText($hermesTestMarkerTemp, (${psQuote(phase)} + ':' + ${pidVar} + [Environment]::NewLine), [Text.UTF8Encoding]::new($false))\n` +
    `${indent}  Move-Item -LiteralPath $hermesTestMarkerTemp -Destination ${psQuote(markerPath)} -Force\n` +
    `${indent}  Start-Sleep -Seconds 30\n` +
    `${indent}}\n`

  return script.replace(anchor, anchor + injected)
}

/** Force the primary CIM process snapshot to fail so the WMI fallback runs. */
function withPrimarySnapshotFailure(script: string): string {
  const anchor = '      $script:treeRows = @(Get-CimInstance Win32_Process -ErrorAction Stop)\n'

  assert.ok(script.includes(anchor), 'terminate script no longer contains the primary snapshot call')

  return script.replace(anchor, "      throw 'forced primary snapshot failure'\n")
}

/**
 * Wall-clock slack for the boundary arms.
 *
 * What these tests prove is BOUNDEDNESS -- the boundary returns, drains and
 * never claims mutation it did not confirm -- not stopwatch accuracy. On a
 * loaded Windows host, creating one process can take seconds, and stopwatch
 * bounds turned real invariants into nondeterministic failures ("elapsed 5006
 * must be <= 5000"). Outcome and ordering assertions stay exact; only the
 * clock gets slack, and every arm is still bounded by its vitest timeout.
 */
const BOUNDARY_ELAPSED_SLACK_MS = 20_000

/** Generous bound for "a real process appeared / a marker was written". */
const LIVE_STATE_TIMEOUT_MS = 45_000

/** Per-probe budget for one liveness read; the caller's loop provides the real bound. */
const IDENTITY_PROBE_TIMEOUT_MS = 20_000

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

/** Budget for the forced-WMI-fallback termination arm; see its comment. */
const EXTERNAL_HOLDER_BUDGET_MS = 45_000

/**
 * Poll until every identity has left the process table.
 *
 * Windows reaps asynchronously: a job terminated microseconds ago can still be
 * enumerable. Asserting emptiness on the first read made these arms fail under
 * load while the boundary was behaving correctly.
 */
async function awaitIdentitiesGone(
  identities: ReadonlyArray<{ pid: number; createdAt?: number }>,
  timeoutMs = LIVE_STATE_TIMEOUT_MS
): Promise<Array<{ pid: number; createdAt?: number }>> {
  const deadline = Date.now() + timeoutMs
  // identitiesStillPresent defaults to a 2 s absolute deadline for the WHOLE
  // batch and fails closed, so on a loaded host its own PowerShell probe times
  // out and reports a terminated process as a survivor forever. Give the probe
  // room; the polling loop below is what bounds this call.
  const probe = () => identitiesStillPresent(identities as any, { deadlineAt: Date.now() + IDENTITY_PROBE_TIMEOUT_MS })
  let survivors = await probe()

  while (survivors.length > 0 && Date.now() < deadline) {
    await delay(100)
    survivors = await probe()
  }

  return survivors
}

/** Assert a set of identities is STILL live, with a probe budget that load cannot exhaust. */
function identitiesPresentNow(identities: ReadonlyArray<{ pid: number; createdAt?: number }>) {
  return identitiesStillPresent(identities as any, { deadlineAt: Date.now() + IDENTITY_PROBE_TIMEOUT_MS })
}

const holder = (overrides: Partial<ForceReleaseHolder> = {}): ForceReleaseHolder => ({
  pid: 57012,
  createdAt: 1_700_000_000,
  name: 'hermes.exe',
  cmdline: 'hermes.exe tools',
  source: 'scanner',
  ...overrides
})

async function launchPowerShellThroughWmi(ps: string, script: string): Promise<number> {
  const encodedScript = Buffer.from(script, 'utf16le').toString('base64')
  const commandLine = `"${ps}" -NoLogo -NoProfile -NonInteractive -EncodedCommand ${encodedScript}`

  const brokerScript = `
$ErrorActionPreference = 'Stop'
$commandLine = [Environment]::GetEnvironmentVariable('HERMES_TEST_WMI_COMMAND_LINE')
$startup = ([wmiclass]'Win32_ProcessStartup').CreateInstance()
$startup.ShowWindow = 0 # SW_HIDE; CREATE_NO_WINDOW is rejected by this WMI provider.
$startup.CreateFlags = 0x01000000 # CREATE_BREAKAWAY_FROM_JOB
$result = ([wmiclass]'Win32_Process').Create($commandLine, $null, $startup)
if ($null -eq $result -or [int]$result.ReturnValue -ne 0) {
  $returnValue = if ($null -eq $result) { -1 } else { [int]$result.ReturnValue }
  throw ('WMI_PROCESS_CREATE_FAILED return=' + $returnValue)
}
Write-Output ([int]$result.ProcessId)
`.trim()

  // Two retries with a real budget. A 2 s timeout could not cover PowerShell
  // start-up on a loaded host, and Win32_Process.Create is documented to
  // return 8 ("unknown failure") transiently on this class of machine -- the
  // same flake scripts/desktop-update/windows.ps1 works around. Neither is a
  // property of the boundary under test.
  let lastError: unknown

  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const { stdout, stderr } = await execFileAsync(
        ps,
        ['-NoLogo', '-NoProfile', '-NonInteractive', '-Command', brokerScript],
        {
          encoding: 'utf8',
          windowsHide: true,
          timeout: 60_000,
          env: {
            ...process.env,
            HERMES_TEST_WMI_COMMAND_LINE: commandLine
          }
        }
      )

      const pid = Number(String(stdout).trim().split(/\r?\n/).pop())

      if (!Number.isInteger(pid) || pid <= 0) {
        throw new Error(`WMI process broker returned invalid PID stdout=${stdout} stderr=${stderr}`)
      }

      return pid
    } catch (error) {
      lastError = error
      await new Promise(resolve => setTimeout(resolve, 500))
    }
  }

  throw new Error(`WMI process broker failed after 3 attempts: ${String((lastError as any)?.message ?? lastError)}`)
}

function buildHoldTargetJobScript(
  rootPid: number,
  rootCreatedAt: number,
  writerPid: number,
  writerCreatedAt: number
): string {
  return String.raw`
$ErrorActionPreference = 'Stop'
$targetJobName = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_TARGET_JOB_NAME')
if ([string]::IsNullOrWhiteSpace($targetJobName)) { throw 'missing target job name' }
Add-Type -TypeDefinition @"
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
public static class HermesTestTargetJob {
  private const uint PROCESS_QUERY_LIMITED_INFORMATION = 0x1000;
  private const uint PROCESS_SET_QUOTA = 0x0100;
  private const uint PROCESS_TERMINATE = 0x0001;
  private const uint SYNCHRONIZE = 0x00100000;
  private const uint JOB_OBJECT_ASSIGN_PROCESS = 0x0001;
  private const uint JOB_OBJECT_TERMINATE = 0x0008;
  private const uint JOB_OBJECT_QUERY = 0x0004;
  [StructLayout(LayoutKind.Sequential)]
  private struct FileTime { public uint Low; public uint High; }
  [DllImport("kernel32.dll", SetLastError = true)]
  private static extern IntPtr OpenProcess(uint access, bool inheritHandle, int pid);
  [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
  private static extern IntPtr OpenJobObject(uint access, bool inheritHandle, string name);
  [DllImport("kernel32.dll", SetLastError = true)]
  private static extern bool GetProcessTimes(IntPtr process, out FileTime creation, out FileTime exit, out FileTime kernel, out FileTime user);
  [DllImport("kernel32.dll", SetLastError = true)]
  private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
  [DllImport("kernel32.dll", SetLastError = true)]
  private static extern bool IsProcessInJob(IntPtr process, IntPtr job, out bool result);
  [DllImport("kernel32.dll", SetLastError = true)]
  private static extern bool CloseHandle(IntPtr handle);
  private static double ToUnixSeconds(FileTime time) {
    long ticks = ((long)time.High << 32) | time.Low;
    return (ticks - 116444736000000000L) / 10000000.0;
  }
  public static IntPtr OpenTargetJob(string name) {
    IntPtr job = OpenJobObject(JOB_OBJECT_ASSIGN_PROCESS | JOB_OBJECT_TERMINATE | JOB_OBJECT_QUERY | SYNCHRONIZE, false, name);
    if (job == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error());
    return job;
  }
  public static IntPtr OpenAuthenticatedProcess(int pid, double expectedUnix) {
    IntPtr process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_SET_QUOTA | PROCESS_TERMINATE | SYNCHRONIZE, false, pid);
    if (process == IntPtr.Zero) throw new Win32Exception(Marshal.GetLastWin32Error());
    FileTime creation, exit, kernel, user;
    if (!GetProcessTimes(process, out creation, out exit, out kernel, out user)) {
      int error = Marshal.GetLastWin32Error();
      CloseHandle(process);
      throw new Win32Exception(error);
    }
    if (Math.Abs(ToUnixSeconds(creation) - expectedUnix) > 1.5) {
      CloseHandle(process);
      throw new InvalidOperationException("create-time mismatch");
    }
    return process;
  }
  public static void Assign(IntPtr job, IntPtr process) {
    if (!AssignProcessToJobObject(job, process)) throw new Win32Exception(Marshal.GetLastWin32Error());
  }
  public static void AssertAssigned(IntPtr job, IntPtr process, int pid) {
    bool assigned;
    if (!IsProcessInJob(process, job, out assigned)) throw new Win32Exception(Marshal.GetLastWin32Error());
    if (!assigned) throw new InvalidOperationException("target job membership false pid=" + pid);
  }
  public static void Close(IntPtr handle) { if (handle != IntPtr.Zero) CloseHandle(handle); }
}
"@
$job = [HermesTestTargetJob]::OpenTargetJob($targetJobName)
try {
  foreach ($target in @(
    @(${Math.trunc(rootPid)}, [double]${rootCreatedAt}),
    @(${Math.trunc(writerPid)}, [double]${writerCreatedAt})
  )) {
    $process = [HermesTestTargetJob]::OpenAuthenticatedProcess([int]$target[0], [double]$target[1])
    try {
      [HermesTestTargetJob]::Assign($job, $process)
      [HermesTestTargetJob]::AssertAssigned($job, $process, [int]$target[0])
    }
    finally { [HermesTestTargetJob]::Close($process) }
  }
  # The wrapper owns the other target-job handle. Returning normally leaves the
  # authenticated target tree for the injected watcher/fallback boundary.
  Write-Output 'TERMINATED'
} finally {
  [HermesTestTargetJob]::Close($job)
}
`.trim()
}

function buildWatcherFailureCommand(): string {
  return String.raw`
const fs = require('node:fs');
const ownerPid = Number(process.env.HERMES_TERMINATE_OWNER_PID);
const readyPath = process.env.HERMES_TERMINATE_WATCHER_READY_PATH;
const nonce = process.env.HERMES_TERMINATE_WATCHER_READY_NONCE;
const deadlineAt = Number(process.env.HERMES_TERMINATE_WATCHER_DEADLINE_AT);
if (!Number.isInteger(ownerPid) || ownerPid <= 0 || !readyPath || !nonce) process.exit(87);
const tempPath = readyPath + '.tmp';
fs.writeFileSync(tempPath, 'ARMED:' + nonce, 'utf8');
fs.renameSync(tempPath, readyPath);
const ownerIsAlive = () => {
  try { process.kill(ownerPid, 0); return true; }
  catch { return false; }
};
const timer = setInterval(() => {
  if (!ownerIsAlive() || Date.now() >= deadlineAt) {
    clearInterval(timer);
    process.exit(17);
  }
}, 10);
`.trim()
}

function makeInjectedWatcherStarter(command: string) {
  let child: ChildProcess | undefined
  let artifacts: { readyPath: string; tempPath: string } | undefined

  const startWatcher = (
    ownerPid: number,
    ownerCreatedAt: number,
    targetJobName: string,
    watcherReadyPath: string,
    watcherReadyNonce: string,
    deadlineAt: number
  ) => {
    child = spawn(
      process.execPath,
      ['-e', command],
      {
        windowsHide: true,
        detached: true,
        stdio: 'ignore',
        env: {
          ...process.env,
          HERMES_TERMINATE_OWNER_PID: String(ownerPid),
          HERMES_TERMINATE_OWNER_CREATED_AT: String(ownerCreatedAt),
          HERMES_TERMINATE_TARGET_JOB_NAME: targetJobName,
          HERMES_TERMINATE_WATCHER_READY_PATH: watcherReadyPath,
          HERMES_TERMINATE_WATCHER_READY_NONCE: watcherReadyNonce,
          HERMES_TERMINATE_WATCHER_DEADLINE_AT: String(Math.trunc(deadlineAt))
        }
      }
    )
    artifacts = { readyPath: watcherReadyPath, tempPath: `${watcherReadyPath}.tmp` }

    return child
  }

  return {
    startWatcher,
    getChild: () => child,
    getArtifacts: () => artifacts
  }
}

async function queryWindowsProcessDetails(
  ps: string,
  pid: number
): Promise<{ raw: string; parentPid: number | null }> {
  try {
    const { stdout } = await execFileAsync(
      ps,
      [
        '-NoLogo',
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        `$p = Get-CimInstance Win32_Process -Filter 'ProcessId = ${Math.trunc(pid)}' -ErrorAction SilentlyContinue; if ($null -eq $p) { 'absent' } else { $p | Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress }`
      ],
      { encoding: 'utf8', windowsHide: true, timeout: 30_000 }
    )

    const raw = String(stdout).trim() || 'absent'

    if (raw === 'absent') {return { raw, parentPid: null }}
    const parsed = JSON.parse(raw) as { ParentProcessId?: number | string } | Array<{ ParentProcessId?: number | string }>
    const row = Array.isArray(parsed) ? parsed[0] : parsed
    const parentPid = Number(row?.ParentProcessId)

    return { raw, parentPid: Number.isInteger(parentPid) && parentPid > 0 ? parentPid : null }
  } catch (error) {
    return { raw: `diagnostic-error:${String((error as any)?.message ?? error)}`, parentPid: null }
  }
}

describe('target watcher transport boundary (live)', () => {
  it.skipIf(!isWindows)('publishes an authenticated wrapper self marker before waiting for READY', { timeout: 60_000 }, async () => {
    const tmp = fs.mkdtempSync(path.join((await import('node:os')).tmpdir(), 'hermes-wrapper-marker-'))
    const markerPath = path.join(tmp, 'wrapper.pid')
    const readyPath = path.join(tmp, 'ready')
    const helperScriptPath = path.join(tmp, 'helper.ps1')
    const helperGatePath = path.join(tmp, 'helper.go')
    const markerNonce = randomBytes(16).toString('hex')
    const readyNonce = randomBytes(16).toString('hex')
    const ps = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')

    const childResult = new Promise<{ error: any; stdout: string; stderr: string }>(resolve => {
      const child = execFile(
        ps,
        ['-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', TERMINATE_JOB_WRAPPER_COMMAND],
        {
          encoding: 'utf8',
          windowsHide: true,
          env: {
            ...process.env,
            HERMES_TERMINATE_SCRIPT: "Write-Output 'TERMINATED'",
            HERMES_TERMINATE_JOB_NAME: `HermesTestHelper-${randomBytes(8).toString('hex')}`,
            HERMES_TERMINATE_TARGET_JOB_NAME: `HermesTestTarget-${randomBytes(8).toString('hex')}`,
            HERMES_TERMINATE_TARGET_WAIT_MS: '500',
            HERMES_TERMINATE_DEADLINE_AT: String(Date.now() + 60_000),
            HERMES_TERMINATE_WATCHER_READY_PATH: readyPath,
            HERMES_TERMINATE_WATCHER_READY_NONCE: readyNonce,
            HERMES_TERMINATE_WRAPPER_PID_MARKER_PATH: markerPath,
            HERMES_TERMINATE_WRAPPER_PID_MARKER_NONCE: markerNonce,
            HERMES_TERMINATE_HELPER_SCRIPT_PATH: helperScriptPath,
            HERMES_TERMINATE_HELPER_GATE_PATH: helperGatePath
          }
        },
        (error, stdout, stderr) => resolve({ error, stdout: String(stdout ?? ''), stderr: String(stderr ?? '') })
      )

      const deadline = Date.now() + LIVE_STATE_TIMEOUT_MS

      const poll = async () => {
        while (Date.now() < deadline && !fs.existsSync(markerPath)) {
          await new Promise(resolve => setTimeout(resolve, 20))
        }

        if (child.exitCode == null && child.signalCode == null) {child.kill('SIGKILL')}
      }

      void poll()
    })

    try {
      const markerDeadline = Date.now() + LIVE_STATE_TIMEOUT_MS

      while (!fs.existsSync(markerPath) && Date.now() < markerDeadline) {
        await new Promise(resolve => setTimeout(resolve, 20))
      }

      assert.equal(fs.existsSync(markerPath), true, 'wrapper did not publish its marker')
      const marker = parseWrapperProcessMarker(fs.readFileSync(markerPath, 'utf8'), markerNonce)
      assert.ok(marker && marker.pid > 0 && marker.createdAt && marker.createdAt > 0)
      const result = await childResult
      assert.ok(result.error, `wrapper unexpectedly completed: ${result.stdout} ${result.stderr}`)
    } finally {
      fs.rmSync(tmp, { recursive: true, force: true })
    }
  })

  it.skipIf(!isWindows)('uses a detached Node bridge and terminalizes the bridge plus PowerShell child', { timeout: 90_000 }, async () => {
    assert.match(TERMINATE_JOB_WATCHER_BRIDGE, /spawnSync/)
    assert.match(TERMINATE_JOB_WATCHER_BRIDGE, /detached:\s*false/)
    assert.match(TERMINATE_JOB_WATCHER_BRIDGE, /HERMES_TERMINATE_WATCHER_ENCODED_COMMAND/)

    const os = await import('node:os')
    const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-watcher-bridge-'))
    const proofPath = path.join(tmp, 'powershell.pid')
    const readyPath = path.join(tmp, 'ready')
    const nonce = randomBytes(16).toString('hex')
    const ps = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')

    const powershellScript = String.raw`
$proofPath = [Environment]::GetEnvironmentVariable('HERMES_TEST_BRIDGE_PROOF_PATH')
$readyPath = [Environment]::GetEnvironmentVariable('HERMES_TEST_BRIDGE_READY_PATH')
$nonce = [Environment]::GetEnvironmentVariable('HERMES_TEST_BRIDGE_NONCE')
[IO.File]::WriteAllText($proofPath, [string]$PID)
Start-Sleep -Milliseconds 4000
[IO.File]::WriteAllText($readyPath, 'READY:' + $nonce)
exit 17
`.trim()

    const encodedCommand = Buffer.from(powershellScript, 'utf16le').toString('base64')

    const bridge = spawn(process.execPath, ['-e', TERMINATE_JOB_WATCHER_BRIDGE], {
      windowsHide: true,
      detached: true,
      stdio: 'ignore',
      env: {
        ...process.env,
        ELECTRON_RUN_AS_NODE: '1',
        HERMES_TERMINATE_WATCHER_POWERSHELL: ps,
        HERMES_TERMINATE_WATCHER_ENCODED_COMMAND: encodedCommand,
        HERMES_TEST_BRIDGE_PROOF_PATH: proofPath,
        HERMES_TEST_BRIDGE_READY_PATH: readyPath,
        HERMES_TEST_BRIDGE_NONCE: nonce
      }
    })

    const bridgePid = bridge.pid
    assert.ok(Number.isInteger(bridgePid) && bridgePid > 0, 'bridge did not return a PID')

    const bridgeResult = new Promise<{ code: number | null; signal: NodeJS.Signals | null }>(resolve => {
      bridge.once('exit', (code, signal) => resolve({ code, signal }))
      bridge.once('error', () => resolve({ code: null, signal: null }))
    })

    const waitDeadline = Date.now() + LIVE_STATE_TIMEOUT_MS

    try {
      while (!fs.existsSync(proofPath) && Date.now() < waitDeadline) {
        await new Promise(resolve => setTimeout(resolve, 20))
      }

      assert.equal(fs.existsSync(proofPath), true, 'bridge never executed its PowerShell child')
      const powershellPid = Number(fs.readFileSync(proofPath, 'utf8').trim())
      assert.ok(Number.isInteger(powershellPid) && powershellPid > 0, 'PowerShell proof PID invalid')
      const tree = await snapshotProcessTreeIdentities(bridgePid as number, { timeoutMs: 2_000 })
      assert.ok(tree.some(identity => identity.pid === bridgePid), 'bridge identity missing from snapshot')
      const powershellCreatedAt = await queryWindowsProcessCreatedAt(powershellPid, { platform: 'win32', timeoutMs: IDENTITY_PROBE_TIMEOUT_MS })
      assert.ok(powershellCreatedAt && powershellCreatedAt > 0, 'PowerShell generation unavailable')

      const authenticatedTree = [
        ...new Map(
          [...tree, { pid: powershellPid, createdAt: powershellCreatedAt }].map(identity => [identity.pid, identity] as const)
        ).values()
      ]

      const result = await bridgeResult
      assert.equal(result.code, 17, `bridge did not propagate PowerShell exit: ${JSON.stringify(result)}`)
      assert.equal(fs.readFileSync(readyPath, 'utf8'), `READY:${nonce}`)
      const absenceDeadline = Date.now() + LIVE_STATE_TIMEOUT_MS

      let remainingDetails = await Promise.all(
        authenticatedTree.map(async identity => ({
          pid: identity.pid,
          details: await queryWindowsProcessDetails(ps, identity.pid)
        }))
      )

      while (remainingDetails.some(entry => entry.details.raw !== 'absent') && Date.now() < absenceDeadline) {
        await new Promise(resolve => setTimeout(resolve, 50))
        remainingDetails = await Promise.all(
          authenticatedTree.map(async identity => ({
            pid: identity.pid,
            details: await queryWindowsProcessDetails(ps, identity.pid)
          }))
        )
      }

      assert.ok(
        remainingDetails.every(entry => entry.details.raw === 'absent'),
        `bridge descendant survived return powershellPid=${powershellPid} tree=${JSON.stringify(authenticatedTree)} details=${JSON.stringify(remainingDetails)} result=${JSON.stringify(result)}`
      )
    } finally {
      if (bridge.exitCode == null && bridge.signalCode == null) {
        try {
          bridge.kill('SIGKILL')
        } catch {
          // Best effort: the child may have exited between the state check and kill.
        }
      }

      fs.rmSync(tmp, { recursive: true, force: true })
    }
  })

  it.skipIf(!isWindows)('never treats a process older than the authenticated parent generation as a descendant', async () => {
    const rootCreatedAt = await queryWindowsProcessCreatedAt(process.pid, {
      platform: 'win32',
      timeoutMs: 2_000
    })

    assert.ok(rootCreatedAt && rootCreatedAt > 0, 'test root generation unavailable')

    const tree = await snapshotProcessTreeIdentities(process.pid, { timeoutMs: 2_000 })

    const older = tree.filter(
      identity =>
        identity.pid !== process.pid &&
        identity.createdAt != null &&
        identity.createdAt + 1.5 < rootCreatedAt
    )

    assert.deepEqual(older, [], `stale-parent PID edges entered the snapshot: ${JSON.stringify(older)}`)
  })
})

describe('termination cancellation hard boundary (live)', () => {
  it.skipIf(!isWindows)(
    'kills a real PowerShell root+descendant tree and leaves the delayed sentinel untouched',
    { timeout: 90_000 },
    async () => {
      const fs = await import('node:fs')
      const os = await import('node:os')

      const {
        killProcessTreeAndAwaitGone,
        TERMINATE_KILL_CONFIRM_MS,
        terminateKillReserveMs,
        terminateWindowsHolderExact
      } = await import('./windows-process-terminate')

      const { execFile } = await import('node:child_process')
      const ps = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')

      const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-force-tree-'))
      const sentinel = path.join(tmp, 'sentinel.txt')
      const writerPidPath = path.join(tmp, 'writer.pid')
      const writerArmedPath = path.join(tmp, 'writer.armed')
      const releasePath = path.join(tmp, 'release')

      // Root spawns a descendant writer that announces itself, then waits for a
      // gate this test opens only AFTER the tree is confirmed dead.
      //
      // The descendant used to mutate on a 1.5 s timer, which made the proof a
      // stopwatch: the abort fires ~0.4 s in and the kill has to land inside
      // the remaining ~1.1 s. Measured on this host the kill normally lands
      // well inside that, but under load it does not, and the arm then failed
      // with the sentinel present while BOTH identities were confirmed gone --
      // a lost race, not an escape. Nothing here is timed now.
      const longScript = `
$ErrorActionPreference = 'Stop'
$sentinel = ${JSON.stringify(sentinel)}
$writerPidPath = ${JSON.stringify(writerPidPath)}
$writerArmedPath = ${JSON.stringify(writerArmedPath)}
$releasePath = ${JSON.stringify(releasePath)}
$writer = Start-Process -FilePath ${JSON.stringify(ps)} -ArgumentList @(
  '-NoLogo','-NoProfile','-NonInteractive','-Command',
  ('Set-Content -LiteralPath ''' + $writerArmedPath + ''' -Value ([string]$PID); ' +
   '$deadline = (Get-Date).AddSeconds(180); ' +
   'while (-not (Test-Path -LiteralPath ''' + $releasePath + ''') -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 25 }; ' +
   'if (Test-Path -LiteralPath ''' + $releasePath + ''') { Set-Content -LiteralPath ''' + $sentinel + ''' -Value LATE_MUTATION }')
) -PassThru -WindowStyle Hidden
Set-Content -LiteralPath $writerPidPath -Value ([string]$writer.Id)
Start-Sleep -Seconds 20
Write-Output ('ROOT=' + $PID + ';CHILD=' + $writer.Id)
`.trim()

      let childPid: number | undefined
      let treeSnapshot: Array<{ pid: number; createdAt: number }> = []
      // Generous: the arm ends by ABORTING, so this budget only has to be
      // large enough that a loaded host still gets both processes started.
      const budgetMs = 60_000
      const started = Date.now()
      const controller = new AbortController()

      const run = async (_script: string, timeoutMs?: number, signal?: AbortSignal) => {
        return await new Promise<{ stdout: string; stderr: string; code: number; pid?: number }>((resolve) => {
          let settled = false

          const finish = (result: { stdout: string; stderr: string; code: number; pid?: number }) => {
            if (settled) {return}
            settled = true
            resolve(result)
          }

          const child = execFile(
            ps,
            ['-NoLogo', '-NoProfile', '-NonInteractive', '-Command', longScript],
            { encoding: 'utf8', windowsHide: true, timeout: Math.max(1, timeoutMs ?? budgetMs) },
            () => undefined
          )

          childPid = child.pid
          const absoluteDeadline = started + budgetMs

          // Capture root immediately; poll briefly for writer pid file.
          void (async () => {
            if (typeof childPid === 'number') {
              treeSnapshot = [{ pid: childPid, createdAt: Date.now() / 1000 }]
            }

            // Abort only once the tree is observably UP and ARMED: the root
            // published the descendant's PID and the descendant reached its
            // wait loop. Killing before that proves nothing about descendants.
            const pollUntil = Date.now() + LIVE_STATE_TIMEOUT_MS

            while (Date.now() < pollUntil) {
              try {
                if (fs.existsSync(writerPidPath) && fs.existsSync(writerArmedPath)) {
                  const writerPid = Number(fs.readFileSync(writerPidPath, 'utf8').trim())
                  const armedPid = Number(fs.readFileSync(writerArmedPath, 'utf8').trim())

                  if (Number.isInteger(writerPid) && writerPid > 0 && armedPid === writerPid) {
                    treeSnapshot.push({ pid: writerPid, createdAt: Date.now() / 1000 })

                    break
                  }
                }
              } catch {
                void 0
              }

              await new Promise(r => setTimeout(r, 40))
            }

            controller.abort()
          })()

          const onAbort = () => {
            void (async () => {
              if (typeof childPid !== 'number') {
                finish({ stdout: '', stderr: 'aborted', code: 1 })

                return
              }

              const killed = await killProcessTreeAndAwaitGone(childPid, {
                confirmMs: terminateKillReserveMs(budgetMs),
                preSnapshot: treeSnapshot,
                deadlineAt: absoluteDeadline
              })

              finish({
                stdout: '',
                stderr: killed.confirmed
                  ? 'aborted'
                  : `unconfirmed-tree-survivors:${killed.survivors.map(s => s.pid).join(',')}`,
                code: 1,
                pid: childPid
              })
            })()
          }

          if (signal?.aborted) {onAbort()}
          else {signal?.addEventListener('abort', onAbort, { once: true })}
        })
      }

      const result = await terminateWindowsHolderExact(holder({ pid: 1, createdAt: 1 }), {
        run,
        timeoutMs: budgetMs,
        waitMs: 100,
        signal: controller.signal,
        platform: 'win32'
      })

      const elapsed = Date.now() - started
      assert.equal(result.kind, 'failed')
      assert.notEqual(result.detail, 'unconfirmed-tree-survivors')
      assert.ok(
        elapsed <= budgetMs + BOUNDARY_ELAPSED_SLACK_MS,
        `terminateWindowsHolderExact elapsed ${elapsed} must stay bounded by ${budgetMs}`
      )
      assert.ok(typeof childPid === 'number' && childPid > 0)
      assert.ok(treeSnapshot.length >= 2, `expected root+writer snapshot, got ${JSON.stringify(treeSnapshot)}`)
      assert.ok(TERMINATE_KILL_CONFIRM_MS >= 1_000)

      assert.deepEqual(await awaitIdentitiesGone(treeSnapshot), [])

      // Causal, not timed: open the gate the descendant was waiting on. A tree
      // that survived would mutate now; a dead one can never observe the gate.
      fs.writeFileSync(releasePath, 'release')
      await delay(1_000)
      assert.deepEqual(await awaitIdentitiesGone(treeSnapshot), [])
      assert.equal(fs.existsSync(sentinel), false)

      try {
        fs.rmSync(tmp, { recursive: true, force: true })
      } catch {
        void 0
      }
    }
  )

  it.skipIf(!isWindows)(
    'uses the production job-object runner to prevent late descendant mutation',
    { timeout: 90_000 },
    async () => {
      const os = await import('node:os')
      const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-force-job-boundary-'))
      const sentinel = path.join(tmp, 'sentinel.txt')
      const releasePath = path.join(tmp, 'release')
      const rootPidPath = path.join(tmp, 'root.pid')
      const writerPidPath = path.join(tmp, 'writer.pid')
      const quotePowerShellLiteral = (value: string) => `'${value.replace(/'/g, "''")}'`
      const ps = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')

      const script = `
$ErrorActionPreference = 'Stop'
$sentinel = ${quotePowerShellLiteral(sentinel)}
$releasePath = ${quotePowerShellLiteral(releasePath)}
$rootPidPath = ${quotePowerShellLiteral(rootPidPath)}
$writerPidPath = ${quotePowerShellLiteral(writerPidPath)}
Set-Content -LiteralPath $rootPidPath -Value ([string]$PID)
$writer = Start-Process -FilePath ${quotePowerShellLiteral(ps)} -ArgumentList @(
  '-NoLogo','-NoProfile','-NonInteractive','-Command',
  ('$deadline = (Get-Date).AddSeconds(180); while (-not (Test-Path -LiteralPath ''' + $releasePath + ''') -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 50 }; if (Test-Path -LiteralPath ''' + $releasePath + ''') { Set-Content -LiteralPath ''' + $sentinel + ''' -Value LATE_MUTATION }')
) -PassThru -WindowStyle Hidden
Set-Content -LiteralPath $writerPidPath -Value ([string]$writer.Id)
Start-Sleep -Seconds 20
`.trim()

      const controller = new AbortController()
      const started = Date.now()
      // The arm ends by ABORTING; the budget only has to outlast process start-up on a loaded host.
      const runPromise = runPowerShellWithHardBoundary(script, 60_000, controller.signal)

      const waitForFile = async (filePath: string, timeoutMs: number) => {
        const deadline = Date.now() + timeoutMs

        while (Date.now() < deadline) {
          if (fs.existsSync(filePath)) {return true}
          await new Promise(resolve => setTimeout(resolve, 25))
        }

        return fs.existsSync(filePath)
      }

      try {
        const rootReady = await waitForFile(rootPidPath, LIVE_STATE_TIMEOUT_MS)

        if (!rootReady) {
          const earlyResult = await runPromise
          throw new Error(`production root did not start: code=${earlyResult.code} stdout=${earlyResult.stdout} stderr=${earlyResult.stderr}`)
        }

        assert.equal(await waitForFile(writerPidPath, LIVE_STATE_TIMEOUT_MS), true, 'production descendant did not start')
        const rootPid = Number(fs.readFileSync(rootPidPath, 'utf8').trim())
        const writerPid = Number(fs.readFileSync(writerPidPath, 'utf8').trim())
        assert.ok(Number.isInteger(rootPid) && rootPid > 0)
        assert.ok(Number.isInteger(writerPid) && writerPid > 0)
        const rootCreatedAt = await queryWindowsProcessCreatedAt(rootPid, { platform: 'win32', timeoutMs: IDENTITY_PROBE_TIMEOUT_MS })
        const writerCreatedAt = await queryWindowsProcessCreatedAt(writerPid, { platform: 'win32', timeoutMs: IDENTITY_PROBE_TIMEOUT_MS })
        assert.ok(rootCreatedAt && rootCreatedAt > 0, 'production root generation unavailable')
        assert.ok(writerCreatedAt && writerCreatedAt > 0, 'production writer generation unavailable')

        const identities = [
          { pid: rootPid, createdAt: rootCreatedAt },
          { pid: writerPid, createdAt: writerCreatedAt }
        ]

        // Abort only after the descendant is real, so the production runner's
        // tree snapshot and the delayed-write safety check cover both nodes.
        controller.abort()
        const result = await runPromise
        const elapsed = Date.now() - started
        assert.equal(result.code, 1)
        // The drain confirms absence inside a ~1.5 s kill reserve. On a loaded
        // host that reserve can genuinely expire, and reporting
        // `unconfirmed-tree-survivors` is the fail-closed answer, not a defect.
        // What must never happen is a SUCCESS claim, and the identity + late
        // mutation checks below prove the tree really died either way.
        assert.doesNotMatch(result.stdout, /TERMINATED/i)
        assert.ok(elapsed <= 5_000 + BOUNDARY_ELAPSED_SLACK_MS, `production runner elapsed ${elapsed}`)

        assert.deepEqual(await awaitIdentitiesGone(identities), [])

        fs.writeFileSync(releasePath, 'release')
        await new Promise(resolve => setTimeout(resolve, 500))
        assert.equal(fs.existsSync(sentinel), false)
        assert.deepEqual(await awaitIdentitiesGone(identities), [])
      } finally {
        controller.abort()
        await runPromise.catch(() => undefined)
        fs.rmSync(tmp, { recursive: true, force: true })
      }
    }
  )

  it.skipIf(!isWindows)(
    'waits for a failed watcher and drains its delayed target before returning',
    { timeout: 90_000 },
    async () => {
      const os = await import('node:os')
      const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-force-watcher-failure-'))
      const sentinel = path.join(tmp, 'sentinel.txt')
      const releasePath = path.join(tmp, 'release')
      const rootPidPath = path.join(tmp, 'root.pid')
      const writerPidPath = path.join(tmp, 'writer.pid')
      const watcherLog = path.join(tmp, 'watcher.log')
      const ps = path.join(process.env.SystemRoot || 'C:\\\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')
      const quotePowerShellLiteral = (value: string) => `'${value.replace(/'/g, "''")}'`

      const rootScript = `
$ErrorActionPreference = 'Stop'
$sentinel = ${quotePowerShellLiteral(sentinel)}
$releasePath = ${quotePowerShellLiteral(releasePath)}
$rootPidPath = ${quotePowerShellLiteral(rootPidPath)}
$writerPidPath = ${quotePowerShellLiteral(writerPidPath)}
Set-Content -LiteralPath $rootPidPath -Value ([string]$PID)
$writer = Start-Process -FilePath ${quotePowerShellLiteral(ps)} -ArgumentList @(
  '-NoLogo','-NoProfile','-NonInteractive','-Command',
  ('$deadline = (Get-Date).AddSeconds(180); while (-not (Test-Path -LiteralPath ''' + $releasePath + ''') -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 50 }; if (Test-Path -LiteralPath ''' + $releasePath + ''') { Set-Content -LiteralPath ''' + $sentinel + ''' -Value LATE_MUTATION }')
) -PassThru -WindowStyle Hidden
Set-Content -LiteralPath $writerPidPath -Value ([string]$writer.Id)
Start-Sleep -Seconds 20
`.trim()

      const waitForFile = async (filePath: string, timeoutMs: number) => {
        const deadline = Date.now() + timeoutMs

        while (Date.now() < deadline) {
          if (fs.existsSync(filePath)) {return true}
          await new Promise(resolve => setTimeout(resolve, 25))
        }

        return fs.existsSync(filePath)
      }

      let launchedRootPid: number | undefined

      try {
        launchedRootPid = await launchPowerShellThroughWmi(ps, rootScript)
        assert.equal(await waitForFile(rootPidPath, LIVE_STATE_TIMEOUT_MS), true, 'watcher-failure root did not start')
        assert.equal(await waitForFile(writerPidPath, LIVE_STATE_TIMEOUT_MS), true, 'watcher-failure writer did not start')
        const rootPid = Number(fs.readFileSync(rootPidPath, 'utf8').trim())
        const writerPid = Number(fs.readFileSync(writerPidPath, 'utf8').trim())
        assert.equal(rootPid, launchedRootPid, 'watcher-failure root PID mismatch')
        const rootCreatedAt = await queryWindowsProcessCreatedAt(rootPid, { platform: 'win32', timeoutMs: IDENTITY_PROBE_TIMEOUT_MS })
        const writerCreatedAt = await queryWindowsProcessCreatedAt(writerPid, { platform: 'win32', timeoutMs: IDENTITY_PROBE_TIMEOUT_MS })
        assert.ok(rootCreatedAt && rootCreatedAt > 0, 'watcher-failure root generation unavailable')
        assert.ok(writerCreatedAt && writerCreatedAt > 0, 'watcher-failure writer generation unavailable')

        const injectedWatcher = makeInjectedWatcherStarter(buildWatcherFailureCommand())
        const started = Date.now()

        const result = await runPowerShellWithHardBoundary(
          buildHoldTargetJobScript(rootPid, rootCreatedAt, writerPid, writerCreatedAt),
          90_000,
          undefined,
          undefined,
          { startWatcher: injectedWatcher.startWatcher }
        )

        const elapsed = Date.now() - started
        assert.equal(result.code, 1, `watcher failure unexpectedly cleared: ${JSON.stringify(result)}`)
        assert.ok(elapsed <= 5_000 + BOUNDARY_ELAPSED_SLACK_MS, `watcher failure elapsed ${elapsed}ms`)
        const watcherChild = injectedWatcher.getChild()

        if (!watcherChild) {throw new Error('injected watcher was not started')}
        const watcherPid = watcherChild.pid ?? 0
        assert.ok(watcherPid > 0, 'injected watcher PID unavailable')
        assert.ok(
          watcherChild.exitCode != null || watcherChild.signalCode != null,
          `injected watcher still running pid=${watcherPid}`
        )
        const watcherArtifacts = injectedWatcher.getArtifacts()
        assert.ok(watcherArtifacts, 'injected watcher artifacts unavailable')

        const identities = [
          { pid: rootPid, createdAt: rootCreatedAt },
          { pid: writerPid, createdAt: writerCreatedAt }
        ]

        assert.deepEqual(await awaitIdentitiesGone([{ pid: watcherPid }]), [], 'failed watcher leaked')
        assert.equal(fs.existsSync(watcherArtifacts.readyPath), false, 'failed watcher READY artifact remained')
        assert.equal(fs.existsSync(watcherArtifacts.tempPath), false, 'failed watcher temp artifact remained')
        assert.deepEqual(
          await awaitIdentitiesGone(identities),
          [],
          `target survived failed watcher boundary result=${JSON.stringify(result)}`
        )
        // Causal, not timed: the descendant only mutates once this gate exists.
        fs.writeFileSync(releasePath, 'release')
        await delay(1_000)
        assert.equal(fs.existsSync(sentinel), false, 'delayed writer mutated after failed watcher return')
        assert.deepEqual(await awaitIdentitiesGone([{ pid: watcherPid }]), [], 'failed watcher reappeared')
        assert.equal(fs.existsSync(watcherArtifacts.readyPath), false, 'failed watcher READY appeared late')
        assert.equal(fs.existsSync(watcherArtifacts.tempPath), false, 'failed watcher temp appeared late')
        assert.deepEqual(await awaitIdentitiesGone(identities), [], 'target generation reappeared after failed watcher')
      } finally {
        if (Number.isInteger(launchedRootPid) && (launchedRootPid as number) > 0) {
          await execFileAsync(taskkillPath, ['/PID', String(launchedRootPid), '/T', '/F'], {
            windowsHide: true,
            timeout: 30_000
          }).catch(() => undefined)
        }

        fs.rmSync(tmp, { recursive: true, force: true })
      }
    }
  )

  it.skipIf(!isWindows)(
    'fails closed when an explicitly injected watcher never publishes READY',
    { timeout: 60_000 },
    async () => {
      const injectedWatcher = makeInjectedWatcherStarter('setTimeout(() => {}, 30000)')
      const started = Date.now()

      const result = await runPowerShellWithHardBoundary(
        "Write-Output 'TERMINATED'",
        2_000,
        undefined,
        undefined,
        { startWatcher: injectedWatcher.startWatcher }
        )

        const elapsed = Date.now() - started
        assert.equal(result.code, 1, `no-READY watcher unexpectedly cleared: ${JSON.stringify(result)}`)
        assert.match(`${result.stdout}\n${result.stderr}`, /TARGET_WATCHER_NOT_ARMED|watcher|boundary/i)
        assert.ok(elapsed <= 3_000 + BOUNDARY_ELAPSED_SLACK_MS, `no-READY boundary elapsed ${elapsed}ms`)
        const watcherChild = injectedWatcher.getChild()

        if (!watcherChild) {throw new Error('no-READY watcher was not started')}
        const watcherPid = watcherChild.pid ?? 0
        assert.ok(watcherPid > 0, 'no-READY watcher PID unavailable')
        const watcherArtifacts = injectedWatcher.getArtifacts()
        assert.ok(watcherArtifacts, 'no-READY watcher artifacts unavailable')
        assert.deepEqual(await awaitIdentitiesGone([{ pid: watcherPid }]), [], 'no-READY watcher leaked')
        assert.equal(fs.existsSync(watcherArtifacts.readyPath), false, 'no-READY watcher published READY late')
        assert.equal(fs.existsSync(watcherArtifacts.tempPath), false, 'no-READY watcher temp artifact remained')
        await new Promise(resolve => setTimeout(resolve, 250))
        assert.deepEqual(await awaitIdentitiesGone([{ pid: watcherPid }]), [], 'no-READY watcher reappeared')
        assert.equal(fs.existsSync(watcherArtifacts.readyPath), false, 'no-READY watcher published READY after return')
        assert.equal(fs.existsSync(watcherArtifacts.tempPath), false, 'no-READY watcher temp appeared after return')
    }
  )

  it.skipIf(!isWindows)(
    'contains an external authenticated holder tree when the primary snapshot fails',
    { timeout: 90_000 },
    async () => {
      const os = await import('node:os')
      const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-force-external-holder-'))
      const sentinel = path.join(tmp, 'sentinel.txt')
      const releasePath = path.join(tmp, 'release')
      const rootPidPath = path.join(tmp, 'root.pid')
      const writerPidPath = path.join(tmp, 'writer.pid')
      const startupPath = path.join(tmp, 'startup.status')
      const startupErrorPath = path.join(tmp, 'startup.error')
      const watcherLog = path.join(tmp, 'watcher.log')
      const ps = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')
      const quotePowerShellLiteral = (value: string) => `'${value.replace(/'/g, "''")}'`

      const rootScript = `
$ErrorActionPreference = 'Stop'
$sentinel = ${quotePowerShellLiteral(sentinel)}
$releasePath = ${quotePowerShellLiteral(releasePath)}
$rootPidPath = ${quotePowerShellLiteral(rootPidPath)}
$writerPidPath = ${quotePowerShellLiteral(writerPidPath)}
$startupPath = ${quotePowerShellLiteral(startupPath)}
$startupErrorPath = ${quotePowerShellLiteral(startupErrorPath)}
try {
Set-Content -LiteralPath $startupPath -Value ('started:' + [string]$PID)
Set-Content -LiteralPath $rootPidPath -Value ([string]$PID)
$writer = Start-Process -FilePath ${quotePowerShellLiteral(ps)} -ArgumentList @(
  '-NoLogo','-NoProfile','-NonInteractive','-Command',
  ('$deadline = (Get-Date).AddSeconds(180); while (-not (Test-Path -LiteralPath ''' + $releasePath + ''') -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 50 }; if (Test-Path -LiteralPath ''' + $releasePath + ''') { Set-Content -LiteralPath ''' + $sentinel + ''' -Value LATE_MUTATION }')
) -PassThru -WindowStyle Hidden
Set-Content -LiteralPath $writerPidPath -Value ([string]$writer.Id)
Start-Sleep -Seconds 20
} catch {
  try { Set-Content -LiteralPath $startupErrorPath -Value $_.Exception.ToString() } catch {}
  throw
}
`.trim()

      let launchedRootPid: number | undefined

      const waitForFile = async (filePath: string, timeoutMs: number) => {
        const deadline = Date.now() + timeoutMs

        while (Date.now() < deadline) {
          if (fs.existsSync(filePath)) {return true}
          await new Promise(resolve => setTimeout(resolve, 25))
        }

        return fs.existsSync(filePath)
      }

      try {
        launchedRootPid = await launchPowerShellThroughWmi(ps, rootScript)
        const fixtureDeadline = Date.now() + LIVE_STATE_TIMEOUT_MS
        let rootParentPid: number | null = null
        let rootDetails = 'absent'

        while (Date.now() < fixtureDeadline) {
          const queriedDetails = await queryWindowsProcessDetails(ps, launchedRootPid)
          rootParentPid = queriedDetails.parentPid
          rootDetails = queriedDetails.raw

          if (
            rootParentPid != null &&
            rootParentPid !== process.pid &&
            rootDetails !== 'absent' &&
            !rootDetails.startsWith('diagnostic-error:')
          ) {
            break
          }

          await new Promise(resolve => setTimeout(resolve, 50))
        }

        if (!(await waitForFile(rootPidPath, Math.max(0, fixtureDeadline - Date.now())))) {
          const startupError = fs.existsSync(startupErrorPath)
            ? fs.readFileSync(startupErrorPath, 'utf8').trim()
            : 'none'

          assert.fail(
            `external holder root did not start launchedPid=${launchedRootPid} parentPid=${rootParentPid} details=${rootDetails} startupError=${startupError}`
          )
        }

        const startupError = fs.existsSync(startupErrorPath)
          ? fs.readFileSync(startupErrorPath, 'utf8').trim()
          : 'none'

        assert.ok(
          rootParentPid != null && rootParentPid !== process.pid && rootDetails !== 'absent',
          `external root was not live outside Vitest parent=${rootParentPid} details=${rootDetails} startupError=${startupError}`
        )
        assert.equal(
          await waitForFile(writerPidPath, Math.max(0, fixtureDeadline - Date.now())),
          true,
          'detached delayed writer did not start'
        )
        const rootPid = Number(fs.readFileSync(rootPidPath, 'utf8').trim())
        assert.equal(rootPid, launchedRootPid, 'root PID marker did not match WMI broker PID')
        const writerPid = Number(fs.readFileSync(writerPidPath, 'utf8').trim())
        const rootCreatedAt = await queryWindowsProcessCreatedAt(rootPid, { platform: 'win32', timeoutMs: IDENTITY_PROBE_TIMEOUT_MS })
        const writerCreatedAt = await queryWindowsProcessCreatedAt(writerPid, { platform: 'win32', timeoutMs: IDENTITY_PROBE_TIMEOUT_MS })
        assert.ok(rootCreatedAt && rootCreatedAt > 0, 'could not authenticate external root generation')
        assert.ok(writerCreatedAt && writerCreatedAt > 0, 'could not authenticate detached writer generation')

        const priorWatcherLog = process.env.HERMES_TERMINATE_WATCHER_LOG
        process.env.HERMES_TERMINATE_WATCHER_LOG = watcherLog
        const started = Date.now()
        let result

        try {
          // This invokes the same production callback/native runner used by
          // main.ts, while the holder was created outside its helper job.
          const { terminateWindowsHolderExact } = await import('./windows-process-terminate')
          result = await terminateWindowsHolderExact(
            holder({ pid: rootPid, createdAt: rootCreatedAt, name: 'powershell.exe', cmdline: 'external holder' }),
            {
              platform: 'win32',
              // The WMI fallback this arm forces (Get-WmiObject Win32_Process)
              // plus PowerShell start-up and Add-Type compilation is seconds of
              // real work. A 5 s budget made the OUTCOME load-dependent, which
              // is the one thing this arm must pin exactly.
              timeoutMs: EXTERNAL_HOLDER_BUDGET_MS,
              waitMs: 1_500,
              buildScript: (pid, createdAt, waitMs) =>
                withPrimarySnapshotFailure(
                  buildExactTerminateScript(pid, createdAt, waitMs, {
                    // The image-under-root proof is unconditional; this tree is
                    // powershell.exe under the Windows directory.
                    installRoot: process.env.SystemRoot || 'C:\\Windows'
                  })
                )
            }
          )
        } finally {
          if (priorWatcherLog == null) {delete process.env.HERMES_TERMINATE_WATCHER_LOG}
          else {process.env.HERMES_TERMINATE_WATCHER_LOG = priorWatcherLog}
        }

        const elapsed = Date.now() - started
        const watcherDiagnostics = fs.existsSync(watcherLog) ? fs.readFileSync(watcherLog, 'utf8') : 'none'
        assert.deepEqual(
          result,
          { kind: 'terminated' },
          'boundary result=' + JSON.stringify(result) +
            ' elapsed=' + elapsed +
            ' root=' + rootPid +
            ' writer=' + writerPid +
            ' watcher=' + watcherDiagnostics
        )
        assert.ok(
          elapsed <= EXTERNAL_HOLDER_BUDGET_MS + BOUNDARY_ELAPSED_SLACK_MS,
          `external holder termination elapsed ${elapsed}`
        )

        const identities = [
          { pid: rootPid, createdAt: rootCreatedAt },
          { pid: writerPid, createdAt: writerCreatedAt }
        ]

        assert.deepEqual(await awaitIdentitiesGone(identities), [])

        // Causal, not timed: the descendant only mutates once this gate exists.
        // A timer here made the proof depend on the termination finishing
        // faster than the writer's sleep, which no loaded host guarantees.
        fs.writeFileSync(releasePath, 'release')
        await delay(1_000)
        assert.equal(fs.existsSync(sentinel), false)
        assert.deepEqual(await awaitIdentitiesGone(identities), [])
      } finally {
        if (Number.isInteger(launchedRootPid) && (launchedRootPid as number) > 0) {
          await execFileAsync(taskkillPath, ['/PID', String(launchedRootPid), '/T', '/F'], {
            windowsHide: true,
            timeout: 30_000
          }).catch(() => undefined)
        }

        fs.rmSync(tmp, { recursive: true, force: true })
      }
    }
  )

  it.skipIf(!isWindows)(
    'closes the target job when a directly-killed helper dies at each child checkpoint',
    { timeout: 120_000 },
    async () => {
      const os = await import('node:os')
      const phases = ['after-child-assignment', 'after-child-suspension'] as const

      const ps = path.join(
        process.env.SystemRoot || 'C:\\\\Windows',
        'System32',
        'WindowsPowerShell',
        'v1.0',
        'powershell.exe'
      )

      const quotePowerShellLiteral = (value: string) => `'${value.replace(/'/g, "''")}'`

      for (const phase of phases) {
        const tmp = fs.mkdtempSync(path.join(os.tmpdir(), `hermes-force-helper-death-${phase}-`))
        const sentinel = path.join(tmp, 'sentinel.txt')
        const releasePath = path.join(tmp, 'release')
        const rootPidPath = path.join(tmp, 'root.pid')
        const writerPidPath = path.join(tmp, 'writer.pid')
        const phaseMarker = path.join(tmp, 'phase.marker')
        const watcherLog = path.join(tmp, 'watcher.log')
        const watcherReadyPath = path.join(tmp, 'watcher.ready')
        const watcherReadyNonce = randomBytes(16).toString('hex')
        const wrapperPidMarkerPath = path.join(tmp, 'wrapper.pid')
        const wrapperPidMarkerNonce = randomBytes(16).toString('hex')
        const helperScriptPath = path.join(tmp, 'helper.ps1')
        const helperGatePath = path.join(tmp, 'helper.go')
        const watcherDeadlineAt = Date.now() + 20_000
        const helperJobName = `HermesTestHelper-${randomBytes(16).toString('hex')}`
        const targetJobName = `HermesTestTarget-${randomBytes(16).toString('hex')}`

        const rootScript = `
$ErrorActionPreference = 'Stop'
$sentinel = ${quotePowerShellLiteral(sentinel)}
$releasePath = ${quotePowerShellLiteral(releasePath)}
$rootPidPath = ${quotePowerShellLiteral(rootPidPath)}
$writerPidPath = ${quotePowerShellLiteral(writerPidPath)}
Set-Content -LiteralPath $rootPidPath -Value ([string]$PID)
$writer = Start-Process -FilePath ${quotePowerShellLiteral(ps)} -ArgumentList @(
  '-NoLogo','-NoProfile','-NonInteractive','-Command',
  ('while (-not (Test-Path -LiteralPath ''' + $releasePath + ''')) { Start-Sleep -Milliseconds 25 }; Set-Content -LiteralPath ''' + $sentinel + ''' -Value LATE_MUTATION')
) -PassThru -WindowStyle Hidden
Set-Content -LiteralPath $writerPidPath -Value ([string]$writer.Id)
Start-Sleep -Seconds 30
`.trim()

        // Launch outside Vitest's inherited job. Otherwise Windows can reject
        // assignment to the named target job, and helper death cannot exercise
        // its kill-on-close boundary.
        await launchPowerShellThroughWmi(ps, rootScript)

        let rootPid: number | undefined
        let writerPid: number | undefined
        let boundaryChild: ReturnType<typeof execFile> | undefined
        let boundaryPromise: Promise<{ stdout: string; stderr: string; code: number }> | undefined
        let watcherChild: ReturnType<typeof execFile> | undefined
        let watcherPromise: Promise<{ stdout: string; stderr: string; code: number }> | undefined

        const waitForFile = async (filePath: string, timeoutMs: number) => {
          const deadline = Date.now() + timeoutMs

          while (Date.now() < deadline) {
            if (fs.existsSync(filePath)) {return true}
            await new Promise(resolve => setTimeout(resolve, 25))
          }

          return fs.existsSync(filePath)
        }

        try {
          assert.equal(await waitForFile(rootPidPath, LIVE_STATE_TIMEOUT_MS), true, `${phase}: root did not start`)
          assert.equal(await waitForFile(writerPidPath, LIVE_STATE_TIMEOUT_MS), true, `${phase}: writer did not start`)
          rootPid = Number(fs.readFileSync(rootPidPath, 'utf8').trim())
          writerPid = Number(fs.readFileSync(writerPidPath, 'utf8').trim())
          assert.ok(Number.isInteger(rootPid) && rootPid > 0, `${phase}: invalid root PID`)
          assert.ok(Number.isInteger(writerPid) && writerPid > 0, `${phase}: invalid writer PID`)

          const rootCreatedAt = await queryWindowsProcessCreatedAt(rootPid, { platform: 'win32', timeoutMs: IDENTITY_PROBE_TIMEOUT_MS })
          const writerCreatedAt = await queryWindowsProcessCreatedAt(writerPid, { platform: 'win32', timeoutMs: IDENTITY_PROBE_TIMEOUT_MS })
          assert.ok(rootCreatedAt && rootCreatedAt > 0, `${phase}: root generation unavailable`)
          assert.ok(writerCreatedAt && writerCreatedAt > 0, `${phase}: writer generation unavailable`)

          boundaryPromise = new Promise(resolve => {
            boundaryChild = execFile(
              ps,
              [
                '-NoLogo',
                '-NoProfile',
                '-NonInteractive',
                '-ExecutionPolicy',
                'Bypass',
                '-Command',
                TERMINATE_JOB_WRAPPER_COMMAND
              ],
              {
                encoding: 'utf8',
                windowsHide: true,
                env: {
                  ...process.env,
                  HERMES_TERMINATE_SCRIPT: withBoundaryPause(
                    buildExactTerminateScript(rootPid, rootCreatedAt, 1_500, {
                      installRoot: process.env.SystemRoot || 'C:\\Windows'
                    }),
                    phase,
                    writerPid,
                    phaseMarker
                  ),
                  HERMES_TERMINATE_JOB_NAME: helperJobName,
                  HERMES_TERMINATE_TARGET_JOB_NAME: targetJobName,
                  HERMES_TERMINATE_TARGET_WAIT_MS: '1500',
                  HERMES_TERMINATE_DEADLINE_AT: String(watcherDeadlineAt),
                  HERMES_TERMINATE_WATCHER_READY_PATH: watcherReadyPath,
                  HERMES_TERMINATE_WATCHER_READY_NONCE: watcherReadyNonce,
                  HERMES_TERMINATE_WRAPPER_PID_MARKER_PATH: wrapperPidMarkerPath,
                  HERMES_TERMINATE_WRAPPER_PID_MARKER_NONCE: wrapperPidMarkerNonce,
                  HERMES_TERMINATE_HELPER_SCRIPT_PATH: helperScriptPath,
                  HERMES_TERMINATE_HELPER_GATE_PATH: helperGatePath
                }
              },
              (error: any, stdout: string, stderr: string) =>
                resolve({
                  stdout: String(stdout ?? ''),
                  stderr: String(stderr ?? error?.message ?? ''),
                  code: typeof error?.code === 'number' ? error.code : error ? 1 : 0
                })
            )
          })

          assert.ok(boundaryChild && typeof boundaryChild.pid === 'number' && boundaryChild.pid > 0)
          assert.equal(
            await waitForFile(wrapperPidMarkerPath, LIVE_STATE_TIMEOUT_MS),
            true,
            `${phase}: wrapper marker missing`
          )

          const wrapperIdentity = parseWrapperProcessMarker(
            fs.readFileSync(wrapperPidMarkerPath, 'utf8'),
            wrapperPidMarkerNonce
          )

          assert.ok(wrapperIdentity?.pid && wrapperIdentity.createdAt, `${phase}: wrapper marker invalid`)
          watcherPromise = new Promise(resolve => {
            watcherChild = execFile(
              ps,
              [
                '-NoLogo',
                '-NoProfile',
                '-NonInteractive',
                '-ExecutionPolicy',
                'Bypass',
                '-Command',
                TERMINATE_JOB_WATCHER_COMMAND
              ],
              {
                encoding: 'utf8',
                windowsHide: true,
                env: {
                  ...process.env,
                  HERMES_TERMINATE_OWNER_PID: String(wrapperIdentity.pid),
                  HERMES_TERMINATE_OWNER_CREATED_AT: String(wrapperIdentity.createdAt),
                  HERMES_TERMINATE_TARGET_JOB_NAME: targetJobName,
                  HERMES_TERMINATE_WATCHER_READY_PATH: watcherReadyPath,
                  HERMES_TERMINATE_WATCHER_READY_NONCE: watcherReadyNonce,
                  HERMES_TERMINATE_WATCHER_DEADLINE_AT: String(watcherDeadlineAt),
                  HERMES_TERMINATE_WATCHER_LOG: watcherLog
                }
              },
              (error: any, stdout: string, stderr: string) =>
                resolve({
                  stdout: String(stdout ?? ''),
                  stderr: String(stderr ?? error?.message ?? ''),
                  code: typeof error?.code === 'number' ? error.code : error ? 1 : 0
                })
            )
          })
          const watcherPid = watcherChild?.pid ?? 0
          assert.ok(Number.isInteger(watcherPid) && watcherPid > 0, `${phase}: invalid watcher PID`)

          const markerReady = await waitForFile(phaseMarker, LIVE_STATE_TIMEOUT_MS)

          if (!markerReady) {
            const earlyResult = await Promise.race([
              boundaryPromise,
              new Promise<{ stdout: string; stderr: string; code: number }>(resolve =>
                setTimeout(() => resolve({ stdout: '', stderr: 'still-running', code: 1 }), 1_000)
              )
            ])

            assert.fail(`${phase}: checkpoint marker missing result=${JSON.stringify(earlyResult)}`)
          }

          assert.equal(
            fs.readFileSync(phaseMarker, 'utf8').trim(),
            `${phase}:${writerPid}`,
            `${phase}: wrong checkpoint marker`
          )
          await execFileAsync(taskkillPath, ['/PID', String(boundaryChild.pid), '/T', '/F'], {
            windowsHide: true,
            timeout: 30_000
          }).catch(() => undefined)
          const boundaryResult = await boundaryPromise
          const boundaryPid = boundaryChild.pid
          assert.equal(
            boundaryResult.code,
            1,
            `${phase}: helper unexpectedly completed pid=${boundaryPid} targetJob=${targetJobName}: ${JSON.stringify(boundaryResult)}`
          )
          const watcherResult = await watcherPromise
          assert.equal(
            watcherResult.code,
            0,
            `${phase}: named target-job watcher failed: ${JSON.stringify(watcherResult)}`
          )
          const watcherLogContents = fs.readFileSync(watcherLog, 'utf8')
          assert.match(watcherLogContents, new RegExp(`started owner=${boundaryPid} job=${targetJobName}`))
          assert.match(watcherLogContents, new RegExp(`waiting owner=${boundaryPid}`))
          assert.match(watcherLogContents, /completed result=0/)
          assert.deepEqual(
            await awaitIdentitiesGone([{ pid: watcherPid }]),
            [],
            `${phase}: target-job watcher leaked pid=${watcherPid}`
          )

          const identities = [
            { pid: rootPid, createdAt: rootCreatedAt },
            { pid: writerPid, createdAt: writerCreatedAt }
          ]

          const survivors = await awaitIdentitiesGone(identities)
          let survivorDetails = ''

          if (survivors.length > 0) {
            try {
              const details = await execFileAsync(
                ps,
                [
                  '-NoLogo',
                  '-NoProfile',
                  '-NonInteractive',
                  '-Command',
                  `$ids = @(${survivors.map(entry => entry.pid).join(',')}); Get-CimInstance Win32_Process | Where-Object { $ids -contains $_.ProcessId } | Select-Object ProcessId,ParentProcessId,Name,CommandLine | ConvertTo-Json -Compress`
                ],
                { encoding: 'utf8', windowsHide: true, timeout: 1_000 }
              )

              survivorDetails = String(details.stdout ?? '')
            } catch (error) {
              survivorDetails = String((error as any)?.message ?? error)
            }
          }

          assert.deepEqual(
            survivors,
            [],
            `${phase}: target survived helper death boundaryPid=${boundaryPid} targetJob=${targetJobName} details=${survivorDetails}`
          )
          fs.writeFileSync(releasePath, 'release')
          await new Promise(resolve => setTimeout(resolve, 500))
          assert.equal(fs.existsSync(sentinel), false, `${phase}: delayed writer mutated after helper death`)
          assert.deepEqual(await awaitIdentitiesGone(identities), [], `${phase}: target generation reappeared`)
        } finally {
          try {
            boundaryChild?.kill('SIGKILL')
            watcherChild?.kill('SIGKILL')
          } catch {
            void 0
          }

          await boundaryPromise?.catch(() => undefined)
          await watcherPromise?.catch(() => undefined)

          for (const pid of [writerPid, rootPid]) {
            if (!Number.isInteger(pid) || (pid as number) <= 0) {continue}

            try {
              await execFileAsync(taskkillPath, ['/PID', String(pid), '/T', '/F'], {
                windowsHide: true,
                timeout: 30_000
              })
            } catch {
              void 0
            }
          }

          fs.rmSync(`${phaseMarker}.tmp`, { force: true })

          fs.rmSync(tmp, { recursive: true, force: true })
        }
      }
    }
  )

  it.skipIf(!isWindows)(
    'uses the production runner to close the target job at each child checkpoint',
    { timeout: 120_000 },
    async () => {
      const os = await import('node:os')
      const phases = ['after-child-assignment', 'after-child-suspension'] as const

      const ps = path.join(
        process.env.SystemRoot || 'C:\\\\Windows',
        'System32',
        'WindowsPowerShell',
        'v1.0',
        'powershell.exe'
      )

      const quotePowerShellLiteral = (value: string) => `'${value.replace(/'/g, "''")}'`

      for (const phase of phases) {
        const tmp = fs.mkdtempSync(path.join(os.tmpdir(), `hermes-force-production-helper-death-${phase}-`))
        const sentinel = path.join(tmp, 'sentinel.txt')
        const releasePath = path.join(tmp, 'release')
        const rootPidPath = path.join(tmp, 'root.pid')
        const writerPidPath = path.join(tmp, 'writer.pid')
        const phaseMarker = path.join(tmp, 'phase.marker')
        const watcherLog = path.join(tmp, 'watcher.log')
        const namedJobLog = path.join(tmp, 'named-job.log')

        const rootScript = `
$ErrorActionPreference = 'Stop'
$sentinel = ${quotePowerShellLiteral(sentinel)}
$releasePath = ${quotePowerShellLiteral(releasePath)}
$rootPidPath = ${quotePowerShellLiteral(rootPidPath)}
$writerPidPath = ${quotePowerShellLiteral(writerPidPath)}
Set-Content -LiteralPath $rootPidPath -Value ([string]$PID)
$writer = Start-Process -FilePath ${quotePowerShellLiteral(ps)} -ArgumentList @(
  '-NoLogo','-NoProfile','-NonInteractive','-Command',
  ('while (-not (Test-Path -LiteralPath ''' + $releasePath + ''')) { Start-Sleep -Milliseconds 25 }; Set-Content -LiteralPath ''' + $sentinel + ''' -Value LATE_MUTATION')
) -PassThru -WindowStyle Hidden
Set-Content -LiteralPath $writerPidPath -Value ([string]$writer.Id)
Start-Sleep -Seconds 30
`.trim()

        // The production boundary handles an external holder. Launching this
        // tree from Vitest inherits the harness job and can make assignment to
        // the target job degrade with ERROR_ACCESS_DENIED.
        await launchPowerShellThroughWmi(ps, rootScript)

        const controller = new AbortController()
        let runPromise: Promise<{ stdout: string; stderr: string; code: number; pid?: number }> | undefined

        const waitForFile = async (filePath: string, timeoutMs: number) => {
          const deadline = Date.now() + timeoutMs

          while (Date.now() < deadline) {
            if (fs.existsSync(filePath)) {return true}
            await new Promise(resolve => setTimeout(resolve, 25))
          }

          return fs.existsSync(filePath)
        }

        const savedEnvironment = {
          watcherLog: process.env.HERMES_TERMINATE_WATCHER_LOG,
          namedJobLog: process.env.HERMES_TERMINATE_NAMED_JOB_LOG
        }

        try {
          assert.equal(await waitForFile(rootPidPath, LIVE_STATE_TIMEOUT_MS), true, `${phase}: root did not start`)
          assert.equal(await waitForFile(writerPidPath, LIVE_STATE_TIMEOUT_MS), true, `${phase}: writer did not start`)
          const rootPid = Number(fs.readFileSync(rootPidPath, 'utf8').trim())
          const writerPid = Number(fs.readFileSync(writerPidPath, 'utf8').trim())
          assert.ok(Number.isInteger(rootPid) && rootPid > 0, `${phase}: invalid root PID`)
          assert.ok(Number.isInteger(writerPid) && writerPid > 0, `${phase}: invalid writer PID`)
          const rootCreatedAt = await queryWindowsProcessCreatedAt(rootPid, { platform: 'win32', timeoutMs: IDENTITY_PROBE_TIMEOUT_MS })
          const writerCreatedAt = await queryWindowsProcessCreatedAt(writerPid, { platform: 'win32', timeoutMs: IDENTITY_PROBE_TIMEOUT_MS })
          assert.ok(rootCreatedAt && rootCreatedAt > 0, `${phase}: root generation unavailable`)
          assert.ok(writerCreatedAt && writerCreatedAt > 0, `${phase}: writer generation unavailable`)

          process.env.HERMES_TERMINATE_WATCHER_LOG = watcherLog
          process.env.HERMES_TERMINATE_NAMED_JOB_LOG = namedJobLog
          runPromise = runPowerShellWithHardBoundary(
            withBoundaryPause(
              buildExactTerminateScript(rootPid, rootCreatedAt, 1_500, {
                installRoot: process.env.SystemRoot || 'C:\\Windows'
              }),
              phase,
              writerPid,
              phaseMarker
            ),
            // Large on purpose: the arm ends by ABORTING at the checkpoint, so
            // the budget only has to outlast PowerShell start-up, the Add-Type
            // compile and one process snapshot. At 5 s a loaded host killed the
            // child before it ever published the checkpoint marker.
            90_000,
            controller.signal
          )
          const markerReady = await waitForFile(phaseMarker, LIVE_STATE_TIMEOUT_MS)

          if (!markerReady) {
            const earlyResult = await runPromise
            assert.fail(
              `${phase}: checkpoint marker missing code=${earlyResult.code} stdout=${earlyResult.stdout} stderr=${earlyResult.stderr}`
            )
          }

          assert.equal(
            fs.readFileSync(phaseMarker, 'utf8').trim(),
            `${phase}:${writerPid}`,
            `${phase}: wrong checkpoint marker`
          )
          controller.abort()
          const boundaryResult = await runPromise
          assert.equal(boundaryResult.code, 1, `${phase}: unexpected result ${JSON.stringify(boundaryResult)}`)

          const identities = [
            { pid: rootPid, createdAt: rootCreatedAt },
            { pid: writerPid, createdAt: writerCreatedAt }
          ]

          const survivors = await awaitIdentitiesGone(identities)
          const watcherDiagnostics = fs.existsSync(watcherLog) ? fs.readFileSync(watcherLog, 'utf8') : '<none>'
          const namedJobDiagnostics = fs.existsSync(namedJobLog) ? fs.readFileSync(namedJobLog, 'utf8') : '<none>'
          assert.deepEqual(
            survivors,
            [],
            `${phase}: target survived production helper death root=${rootPid} writer=${writerPid} boundary=${JSON.stringify(boundaryResult)} watcher=${watcherDiagnostics} namedJob=${namedJobDiagnostics}`
          )
          fs.writeFileSync(releasePath, 'release')
          await new Promise(resolve => setTimeout(resolve, 500))
          assert.equal(fs.existsSync(sentinel), false, `${phase}: delayed writer mutated after helper death`)
          assert.deepEqual(await awaitIdentitiesGone(identities), [], `${phase}: target generation reappeared`)
        } finally {
          controller.abort()
          await runPromise?.catch(() => undefined)

          for (const pid of [
            Number.isInteger(Number(fs.existsSync(writerPidPath) ? fs.readFileSync(writerPidPath, 'utf8').trim() : ''))
              ? Number(fs.readFileSync(writerPidPath, 'utf8').trim())
              : undefined,
            Number.isInteger(Number(fs.existsSync(rootPidPath) ? fs.readFileSync(rootPidPath, 'utf8').trim() : ''))
              ? Number(fs.readFileSync(rootPidPath, 'utf8').trim())
              : undefined
          ]) {
            if (!Number.isInteger(pid) || (pid as number) <= 0) {continue}

            try {
              await execFileAsync(taskkillPath, ['/PID', String(pid), '/T', '/F'], {
                windowsHide: true,
                timeout: 30_000
              })
            } catch {
              void 0
            }
          }

          if (savedEnvironment.watcherLog == null) {delete process.env.HERMES_TERMINATE_WATCHER_LOG}
          else {process.env.HERMES_TERMINATE_WATCHER_LOG = savedEnvironment.watcherLog}

          if (savedEnvironment.namedJobLog == null) {delete process.env.HERMES_TERMINATE_NAMED_JOB_LOG}
          else {process.env.HERMES_TERMINATE_NAMED_JOB_LOG = savedEnvironment.namedJobLog}

          fs.rmSync(tmp, { recursive: true, force: true })
        }
      }
    }
  )

  it.skipIf(!isWindows)(
    'does not inherit a foreign child spawned after target admission',
    { timeout: 90_000 },
    async () => {
      const os = await import('node:os')
      const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-target-job-breakaway-'))
      const rootPidPath = path.join(tmp, 'root.pid')
      const writerPidPath = path.join(tmp, 'writer.pid')
      const spawnGate = path.join(tmp, 'spawn')
      const releaseGate = path.join(tmp, 'release')
      const sentinel = path.join(tmp, 'sentinel')
      const phaseMarker = path.join(tmp, 'phase.marker')
      const writerScriptPath = path.join(tmp, 'foreign-writer.js')
      const ps = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')
      const quote = (value: string) => `'${value.replace(/'/g, "''")}'`

      fs.writeFileSync(
        writerScriptPath,
        "const fs=require('node:fs');const [pid,gate,out]=process.argv.slice(2);fs.writeFileSync(pid,String(process.pid));const t=setInterval(()=>{if(fs.existsSync(gate)){clearInterval(t);fs.writeFileSync(out,'SURVIVED')}},25)",
        'utf8'
      )

      const rootScript = `
$ErrorActionPreference = 'Stop'
Set-Content -LiteralPath ${quote(rootPidPath)} -Value ([string]$PID)
while (-not (Test-Path -LiteralPath ${quote(spawnGate)})) { Start-Sleep -Milliseconds 25 }
$writer = Start-Process -FilePath ${quote(process.execPath)} -ArgumentList @(${quote(writerScriptPath)},${quote(writerPidPath)},${quote(releaseGate)},${quote(sentinel)}) -PassThru -WindowStyle Hidden
Set-Content -LiteralPath ${quote(writerPidPath)} -Value ([string]$writer.Id)
Start-Sleep -Seconds 30
`.trim()

      let rootPid: number | undefined
      let writerPid: number | undefined
      const controller = new AbortController()
      let runPromise: ReturnType<typeof runPowerShellWithHardBoundary> | undefined

      const waitForFile = async (filePath: string, timeoutMs = LIVE_STATE_TIMEOUT_MS) => {
        const deadline = Date.now() + timeoutMs

        while (Date.now() < deadline) {
          if (fs.existsSync(filePath)) {return true}
          await new Promise(resolve => setTimeout(resolve, 25))
        }

        return fs.existsSync(filePath)
      }

      try {
        await launchPowerShellThroughWmi(ps, rootScript)
        assert.equal(await waitForFile(rootPidPath), true, 'authorized root did not start')
        rootPid = Number(fs.readFileSync(rootPidPath, 'utf8').trim())
        const rootCreatedAt = await queryWindowsProcessCreatedAt(rootPid, { platform: 'win32', timeoutMs: IDENTITY_PROBE_TIMEOUT_MS })
        assert.ok(rootCreatedAt && rootCreatedAt > 0, 'authorized root generation unavailable')

        runPromise = runPowerShellWithHardBoundary(
          withBoundaryPause(
            buildExactTerminateScript(rootPid, rootCreatedAt, 1_500, {
              installRoot: process.env.SystemRoot || 'C:\\Windows'
            }),
            'after-root-assignment',
            rootPid,
            phaseMarker
          ),
          8_000,
          controller.signal
        )
        assert.equal(await waitForFile(phaseMarker), true, 'root never entered the target job')

        fs.writeFileSync(spawnGate, 'spawn')
        assert.equal(await waitForFile(writerPidPath), true, 'foreign child did not start in the assignment gap')
        writerPid = Number(fs.readFileSync(writerPidPath, 'utf8').trim())
        const writerCreatedAt = await queryWindowsProcessCreatedAt(writerPid, { platform: 'win32', timeoutMs: IDENTITY_PROBE_TIMEOUT_MS })
        assert.ok(writerCreatedAt && writerCreatedAt > 0, 'foreign child generation unavailable')

        controller.abort()
        const result = await runPromise
        assert.equal(result.code, 1)
        assert.deepEqual(await awaitIdentitiesGone([{ pid: rootPid, createdAt: rootCreatedAt }]), [])
        assert.deepEqual(await identitiesPresentNow([{ pid: writerPid, createdAt: writerCreatedAt }]), [
          { pid: writerPid, createdAt: writerCreatedAt }
        ])

        fs.writeFileSync(releaseGate, 'release')
        assert.equal(
          await waitForFile(sentinel, LIVE_STATE_TIMEOUT_MS),
          true,
          'foreign child was killed by the target boundary'
        )
      } finally {
        controller.abort()
        await runPromise?.catch(() => undefined)

        for (const pid of [writerPid, rootPid]) {
          if (!Number.isInteger(pid) || (pid as number) <= 0) {continue}
          await execFileAsync(taskkillPath, ['/PID', String(pid), '/T', '/F'], { windowsHide: true, timeout: 30_000 }).catch(() => undefined)
        }

        fs.rmSync(tmp, { recursive: true, force: true })
      }
    }
  )

  it.skipIf(!isWindows)(
    'multi-identity confirmation stays within budget and fails closed when probes cannot finish',
    { timeout: 60_000 },
    async () => {
      const { killProcessTreeAndAwaitGone } = await import('./windows-process-terminate')

      const identities = [
        { pid: 900001 },
        { pid: 900002 },
        { pid: 900003 },
        { pid: 900004 }
      ]

      const budgetMs = 250
      const started = Date.now()

      const result = await killProcessTreeAndAwaitGone(900001, {
        confirmMs: budgetMs,
        preSnapshot: identities,
        deadlineAt: started + budgetMs
      })

      const elapsed = Date.now() - started
      assert.ok(
        elapsed <= budgetMs + BOUNDARY_ELAPSED_SLACK_MS,
        `elapsed ${elapsed} exceeded budget ${budgetMs}`
      )
      // Phantom PIDs: either confirmed gone (fast ENOENT path) or unconfirmed if
      // budget exhausted mid-probe. Never invent a clear success after overrunning.
      assert.ok(result.confirmed === true || result.confirmed === false)

      if (elapsed > budgetMs) {
        assert.equal(result.confirmed, false)
      }
    }
  )

  it.skipIf(!isWindows)(
    'does not start a post-deadline root probe after snapshot timeout',
    { timeout: 60_000 },
    async () => {
      const { killProcessTreeAndAwaitGone } = await import('./windows-process-terminate')
      const budgetMs = 180
      const started = Date.now()
      let rootProbeCalls = 0
      let snapshotResolved = false

      let releaseSnapshot: () => void = () => {}

      // A gate, not a sleep: the boundary must return while this read is STILL
      // outstanding. A timer would only say "it returned in under N ms", which
      // a loaded host can violate without any regression.
      const snapshotGate = new Promise<void>(resolve => {
        releaseSnapshot = resolve
      })

      let result

      try {
        result = await killProcessTreeAndAwaitGone(900101, {
          confirmMs: budgetMs,
          deadlineAt: started + budgetMs,
          snapshotProcessTree: async () => {
            // Deliberately ignore the adapter timeout; the boundary must not
            // await this read or launch a fresh default-timeout fallback.
            await snapshotGate
            snapshotResolved = true

            return []
          },
          readCreatedAt: async () => {
            rootProbeCalls += 1

            return 1
          }
        })
      } finally {
        releaseSnapshot()
      }

      assert.equal(snapshotResolved, false, 'the boundary awaited the snapshot instead of its own deadline')
      assert.equal(result.confirmed, false)
      assert.deepEqual(result.survivors.map(entry => entry.pid), [900101])
      assert.equal(rootProbeCalls, 0)
    }
  )
})


describe.skipIf(!isWindows)('identity failure classification (live)', () => {
  /** Run the production classifier over one synthetic error record. */
  async function classify(thrower: string): Promise<string> {
    const script = [
      "$ErrorActionPreference = 'Stop'",
      TERMINATE_ACCESS_DENIED_CLASSIFIER,
      `try { ${thrower} } catch { if (Test-AccessDeniedError $_) { Write-Output 'DENIED' } else { Write-Output 'OTHER' } }`
    ].join('\n')

    const { stdout } = await execFileAsync(
      path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe'),
      ['-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', script],
      { encoding: 'utf8', timeout: 30_000, windowsHide: true }
    )

    return String(stdout).trim()
  }

  it('classifies access denied by Win32 code, not by English message text', { timeout: 60_000 }, async () => {
    // The message is what a localized Windows returns. The regex this replaced
    // ('Access is denied|AccessDenied|denied') misses it, and the holder was
    // then reported ALREADY_GONE: the updater proceeded over a locked venv.
    assert.equal(
      await classify("throw [System.ComponentModel.Win32Exception]::new(5, 'Zugriff verweigert')"),
      'DENIED'
    )
    assert.equal(await classify('throw [System.ComponentModel.Win32Exception]::new(5)'), 'DENIED')
    assert.equal(await classify('throw [System.UnauthorizedAccessException]::new()'), 'DENIED')
  })

  it('does not call every other failure access denied', { timeout: 60_000 }, async () => {
    // ERROR_FILE_NOT_FOUND, and a plain "no such process" record: an exited
    // holder must still classify as ALREADY_GONE, not as elevation-worthy.
    assert.equal(await classify('throw [System.ComponentModel.Win32Exception]::new(2)'), 'OTHER')
    assert.equal(await classify("throw 'Cannot find a process with the process identifier 424242'"), 'OTHER')
  })
})

describe.skipIf(!isWindows)('shared managed-runtime authorization (live)', () => {
  function spawnSleeper(): { pid: number; kill: () => void } {
    const child = spawn(
      path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe'),
      ['-NoLogo', '-NoProfile', '-NonInteractive', '-Command', 'Start-Sleep -Seconds 90'],
      { stdio: 'ignore', windowsHide: true }
    )

    if (!child.pid) {throw new Error('sleeper failed to spawn')}

    return {
      pid: child.pid,
      kill: () => {
        try {
          child.kill()
        } catch {
          /* already gone */
        }
      }
    }
  }

  const alive = (pid: number): boolean => {
    try {
      process.kill(pid, 0)

      return true
    } catch (error: any) {
      return error?.code === 'EPERM'
    }
  }

  it(
    'refuses a holder that only runs from the excluded runtime, and terminates it once a mutation-set resource is proved',
    { timeout: 90_000 },
    async () => {
      const systemRoot = process.env.SystemRoot || 'C:\\Windows'
      // Stand-in for `<install>\.hermes-runtime`: the target's image lives
      // under it, exactly like a foreign uv tool venv running the managed
      // interpreter. `.hermes-runtime` is excluded from the update's mutation
      // set, so running from it is not evidence of blocking this update.
      const sharedRuntimeRoot = path.join(systemRoot, 'System32', 'WindowsPowerShell')
      const refused = spawnSleeper()

      try {
        const createdAt = await queryWindowsProcessCreatedAt(refused.pid, { platform: 'win32', timeoutMs: 5_000 })
        assert.ok(createdAt && createdAt > 0, 'sleeper generation unavailable')

        const outcome = await terminateWindowsHolderExact(
          holder({ pid: refused.pid, createdAt: createdAt as number, name: 'powershell.exe', cmdline: 'sleeper' }),
          { platform: 'win32', timeoutMs: 20_000, waitMs: 1_500, installRoot: systemRoot, sharedRuntimeRoot }
        )

        assert.equal(outcome.kind, 'failed')
        assert.match(String((outcome as any).detail), /TERMINATION_SHARED_RUNTIME_WITHOUT_MUTATION_PROOF/)
        assert.equal(alive(refused.pid), true, 'the refused holder must still be running')
      } finally {
        refused.kill()
      }

      // Same image, same install root, but the exclusion points elsewhere:
      // the ordinary authorization applies and the holder is terminated.
      const allowed = spawnSleeper()

      try {
        const createdAt = await queryWindowsProcessCreatedAt(allowed.pid, { platform: 'win32', timeoutMs: 5_000 })
        assert.ok(createdAt && createdAt > 0, 'sleeper generation unavailable')

        const outcome = await terminateWindowsHolderExact(
          holder({ pid: allowed.pid, createdAt: createdAt as number, name: 'powershell.exe', cmdline: 'sleeper' }),
          {
            platform: 'win32',
            timeoutMs: 20_000,
            waitMs: 1_500,
            installRoot: systemRoot,
            sharedRuntimeRoot: path.join(systemRoot, 'System32', 'hermes-no-such-runtime')
          }
        )

        assert.equal(outcome.kind, 'terminated')
      } finally {
        allowed.kill()
      }
    }
  )
})
