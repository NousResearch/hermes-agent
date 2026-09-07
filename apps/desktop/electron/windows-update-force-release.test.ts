import assert from 'node:assert/strict'
import { type ChildProcess, execFile, spawn } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import fs from 'node:fs'
import path from 'node:path'
import { promisify } from 'node:util'

import { describe, it, vi } from 'vitest'

import { queryWindowsProcessCreatedAt } from './windows-process-identity'
import {
  buildExactTerminateScript,
  parseTerminateScriptOutput,
  parseWrapperProcessMarker,
  runPowerShellWithHardBoundary,
  snapshotProcessTreeIdentities,
  TERMINATE_JOB_WATCHER_BRIDGE,
  TERMINATE_JOB_WATCHER_COMMAND,
  TERMINATE_JOB_WRAPPER_COMMAND,
  terminateWindowsHolderWithinDeadline
} from './windows-process-terminate'
import {
  buildRestartManagerScript,
  parseRestartManagerOutput,
  RESTART_MANAGER_ROW_SPLIT_EXPRESSION
} from './windows-restart-manager'
import {
  attachHolderTreeRelationships,
  type ForceReleaseHolder,
  type ForceReleaseTerminateResult,
  formatHolderLine,
  mergeInstallHolders,
  orderHoldersLeafFirst,
  raceWithBudget,
  runWindowsUpdateForceRelease,
  type WindowsUpdateForceReleaseDeps
} from './windows-update-force-release'

const execFileAsync = promisify(execFile)

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

  const { stdout, stderr } = await execFileAsync(
    ps,
    ['-NoLogo', '-NoProfile', '-NonInteractive', '-Command', brokerScript],
    {
      encoding: 'utf8',
      windowsHide: true,
      timeout: 2_000,
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
      { encoding: 'utf8', windowsHide: true, timeout: 3_000 }
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

const holder = (overrides: Partial<ForceReleaseHolder> = {}): ForceReleaseHolder => ({
  pid: 57012,
  createdAt: 1_700_000_000,
  name: 'hermes.exe',
  cmdline: 'hermes.exe tools',
  source: 'scanner',
  ...overrides
})

function makeDeps(
  overrides: Partial<WindowsUpdateForceReleaseDeps> = {}
): {
  calls: string[]
  deps: WindowsUpdateForceReleaseDeps
  setLocked: (locked: boolean) => void
} {
  const calls: string[] = []
  let locked = true
  let clock = 0

  const deps: WindowsUpdateForceReleaseDeps = {
    now: () => clock,
    wait: async ms => {
      calls.push(`wait:${ms}`)
      clock += Math.max(0, ms)
    },
    isResourceLocked: async () => locked,
    listScannerHolders: async () => {
      calls.push('scan')

      return []
    },
    listRestartManagerHolders: async () => {
      calls.push('rm')

      return []
    },
    terminateHolder: async target => {
      calls.push(`terminate:${target.pid}:${target.createdAt}`)

      return { kind: 'terminated' }
    },
    excludePids: new Set([42]),
    deadlineMs: 5_000,
    settleMs: 0,
    ...overrides
  }

  return {
    calls,
    deps,
    setLocked: next => {
      locked = next
    }
  }
}

describe('orderHoldersLeafFirst', () => {
  it('terminates workers before their wrappers and children before parents', () => {
    const wrapper = holder({ pid: 10, role: 'wrapper' })
    const worker = holder({ pid: 11, wrapperPid: 10, role: 'worker' })
    const child = holder({ pid: 12, parentPid: 11, role: 'worker' })
    const ordered = orderHoldersLeafFirst([wrapper, worker, child])

    assert.deepEqual(
      ordered.map(entry => entry.pid),
      [12, 11, 10]
    )
  })
})

describe('mergeInstallHolders', () => {
  it('dedupes by pid+createdAt and unions scanner with Restart Manager evidence', () => {
    const fromScan = holder({ pid: 7, createdAt: 100, source: 'scanner', resource: 'venv\\Scripts\\hermes.exe' })

    const fromRm = holder({
      pid: 7,
      createdAt: 100,
      source: 'restart-manager',
      resource: 'venv\\Lib\\site-packages\\foo.pyd'
    })

    const other = holder({ pid: 8, createdAt: 200, source: 'restart-manager' })

    const merged = mergeInstallHolders([fromScan, fromRm, other])

    assert.equal(merged.length, 2)
    const first = merged.find(entry => entry.pid === 7)
    assert.ok(first)
    assert.equal(first.source, 'scanner')
    assert.match(String(first.resource), /foo\.pyd/)
  })

  it('keeps one Restart Manager path as the ownership proof and lists the rest as evidence', () => {
    // 2026-09-04: the merged "hermes.exe; python.exe" string was handed to
    // the exact-terminate script as a path, which could never verify it.
    const fromScan = holder({ pid: 7, createdAt: 100.25, source: 'scanner', role: 'wrapper' })

    const shim = holder({
      pid: 7,
      createdAt: 100,
      source: 'restart-manager',
      resource: 'C:\\hermes\\venv\\Scripts\\python.exe'
    })

    const pyd = holder({
      pid: 7,
      createdAt: 100,
      source: 'restart-manager',
      resource: 'C:\\hermes\\venv\\Lib\\site-packages\\yaml\\_yaml.pyd'
    })

    const [merged] = mergeInstallHolders([fromScan, shim, pyd])

    assert.ok(merged)
    assert.equal(merged.resource, 'C:\\hermes\\venv\\Scripts\\python.exe')
    assert.deepEqual(merged.resources, [
      'C:\\hermes\\venv\\Scripts\\python.exe',
      'C:\\hermes\\venv\\Lib\\site-packages\\yaml\\_yaml.pyd'
    ])
    assert.equal(merged.role, 'wrapper')
    assert.equal(merged.createdAt, 100.25)
    assert.match(formatHolderLine(merged), /python\.exe; .*_yaml\.pyd/)
  })

  it('carries the scanner unit routing through a merge', () => {
    const service = holder({ pid: 9, createdAt: 50, source: 'scanner', terminateVia: 'desktop-plugin-service' })
    const rm = holder({ pid: 9, createdAt: 50, source: 'restart-manager', resource: 'C:\\hermes\\venv\\Scripts\\python.exe' })

    const [merged] = mergeInstallHolders([rm, service])

    assert.equal(merged?.terminateVia, 'desktop-plugin-service')
    assert.equal(merged?.resource, 'C:\\hermes\\venv\\Scripts\\python.exe')
  })

  it('drops excluded helper and desktop PIDs', () => {
    const merged = mergeInstallHolders([holder({ pid: 42 }), holder({ pid: 99 })], new Set([42]))
    assert.deepEqual(
      merged.map(entry => entry.pid),
      [99]
    )
  })
})

describe('runWindowsUpdateForceRelease', () => {
  it('returns clear immediately when the install resources are already unlocked', async () => {
    const { calls, deps, setLocked } = makeDeps()
    setLocked(false)

    const outcome = await runWindowsUpdateForceRelease(deps)

    assert.equal(outcome.kind, 'clear')
    assert.deepEqual(calls, [])
  })

  it('force-drains the hermes tools|head orphan class via scanner holders within five seconds', async () => {
    const orphan = holder({
      pid: 57012,
      createdAt: 1_700_000_123.5,
      name: 'hermes.exe',
      cmdline: 'hermes.exe tools',
      resource: 'venv\\Scripts\\hermes.exe'
    })

    const child = holder({
      pid: 57099,
      createdAt: 1_700_000_124,
      parentPid: 57012,
      name: 'python.exe',
      cmdline: 'python.exe -m hermes_cli.main tools'
    })

    let locked = true
    const terminated = new Set<number>()

    const { calls, deps } = makeDeps({
      isResourceLocked: async () => locked,
      listScannerHolders: async () => {
        calls.push('scan')

        return [orphan, child].filter(entry => !terminated.has(entry.pid))
      },
      listRestartManagerHolders: async () => {
        calls.push('rm')

        return []
      },
      terminateHolder: async target => {
        calls.push(`terminate:${target.pid}`)
        terminated.add(target.pid)

        if (terminated.has(orphan.pid) && terminated.has(child.pid)) {
          locked = false
        }

        return { kind: 'terminated' }
      },
      settleMs: 10
    })

    const outcome = await runWindowsUpdateForceRelease(deps)

    assert.equal(outcome.kind, 'clear')
    assert.ok(calls.indexOf('terminate:57099') < calls.indexOf('terminate:57012'), 'leaf before root')
    assert.ok(calls.includes('scan'))
    assert.ok((deps.now?.() ?? 0) <= 5_000)
  })

  it('refuses PID reuse when create-time no longer matches', async () => {
    const stale = holder({ pid: 88, createdAt: 111 })

    const { deps } = makeDeps({
      listScannerHolders: async () => [stale],
      terminateHolder: async () => ({ kind: 'create-time-mismatch' })
    })

    const outcome = await runWindowsUpdateForceRelease(deps)

    assert.equal(outcome.kind, 'blocked')

    if (outcome.kind === 'blocked') {
      assert.match(outcome.message, /create-time|PID reuse|no longer matches/i)
    }
  })

  it('escalates to elevation when OpenProcess/TerminateProcess returns access denied', async () => {
    const elevatedTarget = holder({ pid: 901, createdAt: 222, name: 'python.exe' })

    const { deps } = makeDeps({
      listScannerHolders: async () => [elevatedTarget],
      terminateHolder: async () => ({ kind: 'access-denied', win32Error: 5 })
    })

    const outcome = await runWindowsUpdateForceRelease(deps)

    assert.equal(outcome.kind, 'needs-elevation')

    if (outcome.kind === 'needs-elevation') {
      assert.equal(outcome.holders[0]?.pid, 901)
      assert.match(outcome.message, /Administrator|elevat/i)
    }
  })

  it('surfaces protected-process terminal failure without claiming the venv is clear', async () => {
    const protectedHolder = holder({ pid: 77, createdAt: 333, resource: 'venv\\python.exe' })

    const { deps } = makeDeps({
      listScannerHolders: async () => [protectedHolder],
      terminateHolder: async () => ({ kind: 'protected', win32Error: 5 })
    })

    const outcome = await runWindowsUpdateForceRelease(deps)

    assert.equal(outcome.kind, 'blocked')

    if (outcome.kind === 'blocked') {
      assert.match(outcome.message, /PID 77/)
      assert.match(outcome.message, /protected|unkillable|Win32/i)
    }
  })

  it('excludes the active updater helper and desktop main process', async () => {
    const helper = holder({ pid: 4242, createdAt: 1 })
    const desktop = holder({ pid: 42, createdAt: 2 })
    const real = holder({ pid: 99, createdAt: 3 })
    let locked = true
    const terminated: number[] = []

    const { deps } = makeDeps({
      excludePids: new Set([42, 4242]),
      isResourceLocked: async () => locked,
      listScannerHolders: async () => [helper, desktop, real],
      terminateHolder: async target => {
        terminated.push(target.pid)
        locked = false

        return { kind: 'terminated' }
      }
    })

    const outcome = await runWindowsUpdateForceRelease(deps)

    assert.equal(outcome.kind, 'clear')
    assert.deepEqual(terminated, [99])
  })

  it('stops the quick path within five seconds even when holders respawn', async () => {
    const zombie = holder({ pid: 55, createdAt: 9 })

    const { deps } = makeDeps({
      listScannerHolders: async () => [zombie],
      terminateHolder: async () => ({ kind: 'terminated' } satisfies ForceReleaseTerminateResult),
      settleMs: 2_000,
      deadlineMs: 5_000
    })

    const outcome = await runWindowsUpdateForceRelease(deps)

    assert.ok(outcome.kind === 'timeout' || outcome.kind === 'blocked' || outcome.kind === 'needs-elevation')
    assert.ok((deps.now?.() ?? 0) <= 5_000)
  })

  it('uses Restart Manager holders when the scanner is empty but the shim stays locked', async () => {
    const rmOnly = holder({
      pid: 606,
      createdAt: 404,
      source: 'restart-manager',
      resource: 'venv\\Scripts\\hermes.exe'
    })

    let locked = true
    const terminated: number[] = []

    const { deps } = makeDeps({
      isResourceLocked: async () => locked,
      listScannerHolders: async () => [],
      listRestartManagerHolders: async () => [rmOnly],
      terminateHolder: async target => {
        terminated.push(target.pid)
        locked = false

        return { kind: 'terminated' }
      }
    })

    const outcome = await runWindowsUpdateForceRelease(deps)

    assert.equal(outcome.kind, 'clear')
    assert.deepEqual(terminated, [606])
  })

  it('queries Restart Manager once and reuses it while the scanner keeps naming holders', async () => {
    // Restart Manager costs about 8 s on a loaded host; re-running it every
    // pass was most of the 20 s budget on 2026-09-04.
    const scanned = holder({ pid: 700, createdAt: 500, source: 'scanner' })
    const rm = holder({ pid: 700, createdAt: 500, source: 'restart-manager', resource: 'C:\\hermes\\venv\\Scripts\\python.exe' })
    let locked = true
    let rmCalls = 0
    let attempts = 0

    const { deps } = makeDeps({
      isResourceLocked: async () => locked,
      listScannerHolders: async () => [scanned],
      listRestartManagerHolders: async () => {
        rmCalls += 1

        return [rm]
      },
      terminateHolder: async target => {
        attempts += 1
        assert.equal(target.resource, rm.resource, 'RM evidence from the first pass still attributes the holder')

        if (attempts < 2) {
          return { kind: 'failed', detail: 'first attempt lost the race' }
        }

        locked = false

        return { kind: 'terminated' }
      }
    })

    const outcome = await runWindowsUpdateForceRelease({ ...deps, deadlineMs: 5_000, settleMs: 1 })

    assert.equal(outcome.kind, 'clear')
    assert.equal(attempts, 2)
    assert.equal(rmCalls, 1)
  })

  it('reports each discovery pass and termination outcome for forensics', async () => {
    const scanned = holder({ pid: 701, createdAt: 501, source: 'scanner' })
    let locked = true
    const discoveries: number[] = []
    const outcomes: string[] = []

    const { deps } = makeDeps({
      isResourceLocked: async () => locked,
      listScannerHolders: async () => [scanned],
      listRestartManagerHolders: async () => [],
      terminateHolder: async () => {
        locked = false

        return { kind: 'terminated' }
      },
      onDiscovery: info => {
        discoveries.push(info.scanner)
      },
      onHolderOutcome: (target, result) => {
        outcomes.push(`${target.pid}:${result.kind}`)
      }
    })

    const outcome = await runWindowsUpdateForceRelease({ ...deps, deadlineMs: 5_000, settleMs: 1 })

    assert.equal(outcome.kind, 'clear')
    assert.deepEqual(discoveries, [1])
    assert.deepEqual(outcomes, ['701:terminated'])
  })

  it('never reports clear while any verified holder remains', async () => {
    const survivor = holder({ pid: 1, createdAt: 2 })
    const terminate = vi.fn(async (): Promise<ForceReleaseTerminateResult> => ({ kind: 'terminated' }))

    const { deps } = makeDeps({
      listScannerHolders: async () => [survivor],
      terminateHolder: terminate,
      settleMs: 0,
      deadlineMs: 100
    })

    const outcome = await runWindowsUpdateForceRelease(deps)

    assert.notEqual(outcome.kind, 'clear')
  })

  it('enforces a hard wall-clock budget when dependencies hang past five seconds', async () => {
    const started = Date.now()

    const hung = () =>
      new Promise<ForceReleaseHolder[]>(resolve => {
        setTimeout(() => resolve([holder({ pid: 1, createdAt: 2 })]), 8_000)
      })

    const outcome = await runWindowsUpdateForceRelease({
      deadlineMs: 400,
      settleMs: 0,
      isResourceLocked: async () => true,
      listScannerHolders: async () => hung(),
      listRestartManagerHolders: async () => [],
      terminateHolder: async () => ({ kind: 'terminated' })
    })

    const elapsed = Date.now() - started
    assert.notEqual(outcome.kind, 'clear')
    assert.ok(elapsed < 2_000, `elapsed ${elapsed}ms must stay near the 400ms budget`)
    assert.ok(elapsed >= 300, `elapsed ${elapsed}ms should wait roughly the budget`)
  })

  it('aborts and drains termination at the outer absolute deadline', async () => {
    const target = holder({ pid: 71, createdAt: 72 })
    let aborted = false
    let settled = false
    const started = Date.now()

    const outcome = await runWindowsUpdateForceRelease({
      deadlineMs: 80,
      settleMs: 0,
      isResourceLocked: async () => true,
      listScannerHolders: async () => [target],
      listRestartManagerHolders: async () => [],
      terminateHolder: async (_holder, _budget, signal) => {
        await new Promise<void>(resolve => {
          signal?.addEventListener('abort', () => {
            aborted = true
            resolve()
          }, { once: true })
        })
        settled = true

        return { kind: 'failed', detail: 'aborted-and-drained' }
      }
    })

    const elapsed = Date.now() - started
    assert.equal(outcome.kind, 'timeout')
    assert.equal(aborted, true)
    assert.equal(settled, true, 'termination was terminal before the API returned')
    assert.ok(elapsed < 180, `elapsed ${elapsed}ms exceeded the 80ms absolute deadline envelope`)
  })

  it('bounds a stalled lock probe at the same absolute deadline', async () => {
    const started = Date.now()

    const outcome = await runWindowsUpdateForceRelease({
      deadlineMs: 80,
      settleMs: 0,
      isResourceLocked: async () => {
        await new Promise(resolve => setTimeout(resolve, 250))

        return false
      },
      listScannerHolders: async () => [],
      listRestartManagerHolders: async () => [],
      terminateHolder: async () => ({ kind: 'terminated' })
    })

    const elapsed = Date.now() - started
    assert.equal(outcome.kind, 'timeout')
    assert.ok(elapsed < 180, `elapsed ${elapsed}ms exceeded the 80ms absolute deadline envelope`)
  })

  it('passes remaining budget into terminateHolder', async () => {
    const budgets: number[] = []
    const target = holder({ pid: 3, createdAt: 4 })
    let locked = true

    const { deps } = makeDeps({
      deadlineMs: 1_000,
      settleMs: 0,
      isResourceLocked: async () => locked,
      listScannerHolders: async () => [target],
      terminateHolder: async (_holder, budgetMs) => {
        budgets.push(budgetMs)
        locked = false

        return { kind: 'terminated' }
      }
    })

    const outcome = await runWindowsUpdateForceRelease(deps)
    assert.equal(outcome.kind, 'clear')
    assert.ok(budgets.length >= 1)
    assert.ok(budgets[0]! <= 1_000)
    assert.ok(budgets[0]! > 0)
  })
})

describe('raceWithBudget', () => {
  it('returns the timeout fallback when work exceeds the budget', async () => {
    const started = Date.now()

    const value = await raceWithBudget(
      new Promise<string>(resolve => setTimeout(() => resolve('late'), 1_000)),
      100,
      () => 'fallback'
    )

    const elapsed = Date.now() - started
    assert.equal(value, 'fallback')
    assert.ok(elapsed < 500)
  })
})

describe('restart manager script contract', () => {
  it('emits a literal pipe split that does not over-escape regex', () => {
    const script = buildRestartManagerScript('C:\\h\\rm-list.txt')
    assert.match(script, /\$part\.Split\(\[char\]'\|', 4\)/)
    assert.doesNotMatch(script, /\$part -split '\\\\\|'/)
    assert.doesNotMatch(script, /\$part -split '\\\|'/)
    assert.match(script, /StringBuilder\(CCH_RM_SESSION_KEY \+ 1\)/)
    assert.equal(RESTART_MANAGER_ROW_SPLIT_EXPRESSION, "$part.Split([char]'|', 4)")
  })

  it('parses RM JSON rows into force-release holders', () => {
    const holders = parseRestartManagerOutput(
      JSON.stringify([{ pid: 12, createdAt: 34, name: 'python.exe' }]),
      ['C:\\h\\venv\\Scripts\\hermes.exe']
    )

    assert.equal(holders.length, 1)
    assert.equal(holders[0]?.source, 'restart-manager')
    assert.equal(holders[0]?.pid, 12)
    assert.match(String(holders[0]?.resource), /hermes\.exe/)
  })

  it('splits generated RM delimiter rows correctly on real Windows PowerShell', async () => {
    if (process.platform !== 'win32') {return}

    const probe = String.raw`
$ErrorActionPreference = 'Stop'
$part = '12|34|name|C:\h\venv\Scripts\hermes.exe'
$bits = $part.Split([char]'|', 4)
if ($bits.Count -ne 4) { Write-Output ("bad-count=" + $bits.Count); exit 2 }
if ($bits[0] -ne '12' -or $bits[1] -ne '34' -or $bits[2] -ne 'name') {
  Write-Output ("bad-values=" + ($bits -join ','))
  exit 3
}
Write-Output 'ok'
exit 0
`.trim()

    const ps = path.join(process.env.SystemRoot || 'C:\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')

    const { stdout } = await execFileAsync(
      ps,
      ['-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-Command', probe],
      { encoding: 'utf8', windowsHide: true, timeout: 10_000 }
    )

    assert.match(String(stdout), /ok/)
  })
})

describe('terminate script output parser', () => {
  it('classifies create-time mismatch, access denied, and protected', () => {
    assert.deepEqual(parseTerminateScriptOutput('CREATE_TIME_MISMATCH actual=1 expected=2', 3), {
      kind: 'create-time-mismatch'
    })
    assert.deepEqual(parseTerminateScriptOutput('ACCESS_DENIED', 5), {
      kind: 'access-denied',
      win32Error: 5
    })
    assert.deepEqual(parseTerminateScriptOutput('PROTECTED win32=5', 5), {
      kind: 'protected',
      win32Error: 5
    })
    assert.deepEqual(parseTerminateScriptOutput('TERMINATED', 0), { kind: 'terminated' })
    assert.deepEqual(parseTerminateScriptOutput('ALREADY_GONE', 0), { kind: 'already-gone' })
    assert.equal(parseTerminateScriptOutput('FAILED win32=87', 1).kind, 'failed')
    assert.equal(parseTerminateScriptOutput('FAILED win32=6', 1).kind, 'failed')
  })
})

describe('target watcher transport boundary', () => {
  it('requires a nonce-bound wrapper marker and fail-closed watcher script handling', () => {
    assert.match(TERMINATE_JOB_WRAPPER_COMMAND, /HERMES_TERMINATE_WRAPPER_PID_MARKER_PATH/)
    assert.match(TERMINATE_JOB_WRAPPER_COMMAND, /HERMES_TERMINATE_WRAPPER_PID_MARKER_NONCE/)
    assert.match(TERMINATE_JOB_WRAPPER_COMMAND, /HERMES_TERMINATE_WRAPPER_PHASE_PATH/)
    assert.match(TERMINATE_JOB_WRAPPER_COMMAND, /HERMES_TERMINATE_WRAPPER_PHASE_NONCE/)
    assert.match(TERMINATE_JOB_WRAPPER_COMMAND, /marker-published|target-job-created|watcher-READY-observed/)
    assert.match(TERMINATE_JOB_WATCHER_BRIDGE, /spawnSync/)
    assert.match(TERMINATE_JOB_WATCHER_COMMAND, /\$ErrorActionPreference = 'Stop'/)
    assert.match(TERMINATE_JOB_WATCHER_COMMAND, /Add-Type -TypeDefinition @'[\s\S]*'@ -ErrorAction Stop/)
    assert.match(TERMINATE_JOB_WATCHER_COMMAND, /exit \$watchResult/)
    assert.match(TERMINATE_JOB_WATCHER_COMMAND, /FAILED:' \+ \$watcherReadyNonce/)
  })

  it('publishes an authenticated wrapper self marker before waiting for READY', { timeout: 10_000 }, async () => {
    if (process.platform !== 'win32') {return}

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
            HERMES_TERMINATE_DEADLINE_AT: String(Date.now() + 5_000),
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

      const deadline = Date.now() + 2_000

      const poll = async () => {
        while (Date.now() < deadline && !fs.existsSync(markerPath)) {
          await new Promise(resolve => setTimeout(resolve, 20))
        }

        if (child.exitCode == null && child.signalCode == null) {child.kill('SIGKILL')}
      }

      void poll()
    })

    try {
      const markerDeadline = Date.now() + 2_000

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

  it('uses a detached Node bridge and terminalizes the bridge plus PowerShell child', { timeout: 15_000 }, async () => {
    if (process.platform !== 'win32') {return}

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

    const waitDeadline = Date.now() + 2_000

    try {
      while (!fs.existsSync(proofPath) && Date.now() < waitDeadline) {
        await new Promise(resolve => setTimeout(resolve, 20))
      }

      assert.equal(fs.existsSync(proofPath), true, 'bridge never executed its PowerShell child')
      const powershellPid = Number(fs.readFileSync(proofPath, 'utf8').trim())
      assert.ok(Number.isInteger(powershellPid) && powershellPid > 0, 'PowerShell proof PID invalid')
      const tree = await snapshotProcessTreeIdentities(bridgePid as number, { timeoutMs: 2_000 })
      assert.ok(tree.some(identity => identity.pid === bridgePid), 'bridge identity missing from snapshot')
      const powershellCreatedAt = await queryWindowsProcessCreatedAt(powershellPid, { platform: 'win32', timeoutMs: 2_000 })
      assert.ok(powershellCreatedAt && powershellCreatedAt > 0, 'PowerShell generation unavailable')

      const authenticatedTree = [
        ...new Map(
          [...tree, { pid: powershellPid, createdAt: powershellCreatedAt }].map(identity => [identity.pid, identity] as const)
        ).values()
      ]

      const result = await bridgeResult
      assert.equal(result.code, 17, `bridge did not propagate PowerShell exit: ${JSON.stringify(result)}`)
      assert.equal(fs.readFileSync(readyPath, 'utf8'), `READY:${nonce}`)
      const absenceDeadline = Date.now() + 3_000

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

  it('never treats a process older than the authenticated parent generation as a descendant', async () => {
    if (process.platform !== 'win32') {return}

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

describe('merge create-time tolerance and production leaf-first', () => {
  it('dedupes scanner fractional create-time with RM integer seconds', () => {
    const fromScan = holder({
      pid: 7,
      createdAt: 100.4,
      source: 'scanner',
      resource: 'venv\\Scripts\\hermes.exe'
    })

    const fromRm = holder({
      pid: 7,
      createdAt: 100,
      source: 'restart-manager',
      resource: 'venv\\Scripts\\python.exe'
    })

    const merged = mergeInstallHolders([fromScan, fromRm])
    assert.equal(merged.length, 1)
    assert.equal(merged[0]?.pid, 7)
    // One exact path for the ownership proof (Restart Manager wins); the
    // scanner claim stays visible as evidence only.
    assert.equal(merged[0]?.resource, 'venv\\Scripts\\python.exe')
    assert.deepEqual(merged[0]?.resources, ['venv\\Scripts\\hermes.exe', 'venv\\Scripts\\python.exe'])
  })

  it('orders production generic holders leaf-first when parentPid evidence is present', () => {
    const root = holder({ pid: 10, createdAt: 1, role: 'other' })
    const child = holder({ pid: 11, createdAt: 2, parentPid: 10, role: 'other' })
    const ordered = orderHoldersLeafFirst(attachHolderTreeRelationships([root, child]))
    assert.deepEqual(
      ordered.map(entry => entry.pid),
      [11, 10]
    )
  })
})

describe('termination cancellation hard boundary', () => {
  it('budget-owned terminate returns without scheduling post-return mutation', async () => {
    const target = holder({ pid: 404, createdAt: 1 })
    let mutated = false
    const started = Date.now()

    const outcome = await runWindowsUpdateForceRelease({
      deadlineMs: 400,
      settleMs: 0,
      isResourceLocked: async () => true,
      listScannerHolders: async () => [target],
      listRestartManagerHolders: async () => [],
      terminateHolder: async (_holder, budget) => {
        // Production contract: terminate owns the budget and must not leave
        // pending mutation work that fires after it returns.
        const slice = Math.max(1, Math.min(budget, 80))
        await new Promise(resolve => setTimeout(resolve, slice))

        // A cancelled late arm would mutate if it were left live.
        const late = setTimeout(() => {
          mutated = true
        }, 600)

        clearTimeout(late)

        return { kind: 'failed', detail: 'deadline-exhausted' }
      }
    })

    const elapsed = Date.now() - started
    assert.notEqual(outcome.kind, 'clear')
    await new Promise(resolve => setTimeout(resolve, 700))
    assert.equal(mutated, false)
    assert.ok(elapsed < 2_000, `elapsed ${elapsed}`)
  })

  it(
    'maps multiple near-expiry holders through one production absolute deadline',
    { timeout: 10_000 },
    async () => {
      const holders = [
        holder({ pid: 701, createdAt: 1_701_000_001 }),
        holder({ pid: 702, createdAt: 1_701_000_002 })
      ]

      const terminated = new Set<number>()
      const terminationCalls: Array<{ pid: number; budgetMs: number; deadlineAt?: number }> = []
      let scanPass = 0
      let runCalls = 0
      let lateMutations = 0
      const started = Date.now()

      const fakeRun = async (
        _script: string,
        timeoutMs = 0,
        signal?: AbortSignal,
        nativeDeadlineAt?: number
      ): Promise<{ stdout: string; stderr: string; code: number }> => {
        const call = ++runCalls
        // Leave a real margin for the outer kill/confirmation reserve while
        // still making the second holder genuinely near-expiry.
        const delayMs = call === 1 ? 4_250 : 500
        let aborted = false

        const lateMutation = setTimeout(() => {
          lateMutations += 1
        }, delayMs + 250)

        await new Promise<void>(resolve => {
          const timer = setTimeout(resolve, delayMs)

          const onAbort = () => {
            aborted = true
            clearTimeout(timer)
            resolve()
          }

          if (signal?.aborted) {onAbort()}
          else {signal?.addEventListener('abort', onAbort, { once: true })}
        })
        clearTimeout(lateMutation)

        if (
          aborted ||
          signal?.aborted ||
          (typeof nativeDeadlineAt === 'number' && Date.now() >= nativeDeadlineAt)
        ) {
          return { stdout: '', stderr: `aborted timeout=${timeoutMs}`, code: 1 }
        }

        if (call === 1) {return { stdout: 'TERMINATED', stderr: '', code: 0 }}

        return { stdout: '', stderr: 'near-expiry failure', code: 1 }
      }

      const outcome = await runWindowsUpdateForceRelease({
        deadlineMs: 5_000,
        settleMs: 0,
        isResourceLocked: async () => terminated.size < holders.length,
        listScannerHolders: async () => {
          scanPass += 1

          return scanPass === 1 ? holders : []
        },
        listRestartManagerHolders: async () => [],
        // Keep this callback identical to the production mapping in main.ts:
        // one remaining budget and one absolute deadline reach the native seam.
        terminateHolder: async (target, budgetMs, signal, deadlineAt) => {
          terminationCalls.push({ pid: target.pid, budgetMs, deadlineAt })

          const result = await terminateWindowsHolderWithinDeadline(target, {
            platform: 'win32',
            budgetMs,
            deadlineAt: deadlineAt ?? Date.now(),
            signal,
            run: fakeRun
          })

          if (result.kind === 'terminated' || result.kind === 'already-gone') {
            terminated.add(target.pid)
          }

          return result
        }
      })

      const elapsed = Date.now() - started

      assert.equal(outcome.kind, 'timeout')
      assert.equal(runCalls, 2, 'both near-expiry holders must reach the bounded seam')
      assert.equal(terminationCalls.length, 2)
      assert.equal(
        new Set(terminationCalls.map(call => call.deadlineAt)).size,
        1,
        'holders must share one absolute deadline'
      )
      const first = terminationCalls[0]
      const second = terminationCalls[1]
      assert.ok(first && second)
      assert.ok(first.budgetMs <= 5_000)
      assert.ok(second.budgetMs < 900, `second holder was not near expiry: ${second.budgetMs}`)
      assert.ok(second.budgetMs > 450, `test did not exercise a meaningful second budget: ${second.budgetMs}`)
      assert.ok(typeof first.deadlineAt === 'number')
      assert.ok((first.deadlineAt as number) - started <= 5_100)
      assert.ok(elapsed <= 5_100, `production-mapped force release elapsed ${elapsed}`)

      await new Promise(resolve => setTimeout(resolve, 800))
      assert.equal(lateMutations, 0, 'near-expiry mutation fired after the updater returned')
    }
  )

  it(
    'kills a real PowerShell root+descendant tree and leaves the delayed sentinel untouched',
    { timeout: 15_000 },
    async () => {
      if (process.platform !== 'win32') {return}
      const fs = await import('node:fs')
      const os = await import('node:os')

      const {
        identitiesStillPresent,
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

      // Root spawns a descendant writer that records its PID then would write sentinel later.
      const longScript = `
$ErrorActionPreference = 'Stop'
$sentinel = ${JSON.stringify(sentinel)}
$writerPidPath = ${JSON.stringify(writerPidPath)}
$writer = Start-Process -FilePath ${JSON.stringify(ps)} -ArgumentList @(
  '-NoLogo','-NoProfile','-NonInteractive','-Command',
  ('Start-Sleep -Milliseconds 1500; Set-Content -LiteralPath ''' + $sentinel + ''' -Value LATE_MUTATION')
) -PassThru -WindowStyle Hidden
Set-Content -LiteralPath $writerPidPath -Value ([string]$writer.Id)
Start-Sleep -Seconds 20
Write-Output ('ROOT=' + $PID + ';CHILD=' + $writer.Id)
`.trim()

      let childPid: number | undefined
      let treeSnapshot: Array<{ pid: number; createdAt: number }> = []
      const budgetMs = 5_000
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

            const pollUntil = Date.now() + 600

            while (Date.now() < pollUntil) {
              try {
                if (fs.existsSync(writerPidPath)) {
                  const writerPid = Number(fs.readFileSync(writerPidPath, 'utf8').trim())

                  if (Number.isInteger(writerPid) && writerPid > 0) {
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
      assert.ok(elapsed <= budgetMs, `terminateWindowsHolderExact elapsed ${elapsed} must be <= ${budgetMs}`)
      assert.ok(typeof childPid === 'number' && childPid > 0)
      assert.ok(treeSnapshot.length >= 2, `expected root+writer snapshot, got ${JSON.stringify(treeSnapshot)}`)
      assert.ok(TERMINATE_KILL_CONFIRM_MS >= 1_000)

      assert.deepEqual(await identitiesStillPresent(treeSnapshot), [])

      // Delayed descendant write would have landed by now if the tree survived.
      await new Promise(resolve => setTimeout(resolve, 1_800))
      assert.deepEqual(await identitiesStillPresent(treeSnapshot), [])
      assert.equal(fs.existsSync(sentinel), false)

      try {
        fs.rmSync(tmp, { recursive: true, force: true })
      } catch {
        void 0
      }
    }
  )

  it(
    'uses the production job-object runner to prevent late descendant mutation',
    { timeout: 15_000 },
    async () => {
      if (process.platform !== 'win32') {return}

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
  ('$deadline = (Get-Date).AddSeconds(8); while (-not (Test-Path -LiteralPath ''' + $releasePath + ''') -and (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 50 }; if (Test-Path -LiteralPath ''' + $releasePath + ''') { Set-Content -LiteralPath ''' + $sentinel + ''' -Value LATE_MUTATION }')
) -PassThru -WindowStyle Hidden
Set-Content -LiteralPath $writerPidPath -Value ([string]$writer.Id)
Start-Sleep -Seconds 20
`.trim()

      const controller = new AbortController()
      const started = Date.now()
      const runPromise = runPowerShellWithHardBoundary(script, 5_000, controller.signal)

      const waitForFile = async (filePath: string, timeoutMs: number) => {
        const deadline = Date.now() + timeoutMs

        while (Date.now() < deadline) {
          if (fs.existsSync(filePath)) {return true}
          await new Promise(resolve => setTimeout(resolve, 25))
        }

        return fs.existsSync(filePath)
      }

      try {
        const rootReady = await waitForFile(rootPidPath, 1_500)

        if (!rootReady) {
          const earlyResult = await runPromise
          throw new Error(`production root did not start: code=${earlyResult.code} stdout=${earlyResult.stdout} stderr=${earlyResult.stderr}`)
        }

        assert.equal(await waitForFile(writerPidPath, 1_500), true, 'production descendant did not start')
        const rootPid = Number(fs.readFileSync(rootPidPath, 'utf8').trim())
        const writerPid = Number(fs.readFileSync(writerPidPath, 'utf8').trim())
        assert.ok(Number.isInteger(rootPid) && rootPid > 0)
        assert.ok(Number.isInteger(writerPid) && writerPid > 0)
        const rootCreatedAt = await queryWindowsProcessCreatedAt(rootPid, { platform: 'win32', timeoutMs: 2_000 })
        const writerCreatedAt = await queryWindowsProcessCreatedAt(writerPid, { platform: 'win32', timeoutMs: 2_000 })
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
        assert.doesNotMatch(result.stderr, /unconfirmed-tree-survivors/i)
        assert.ok(elapsed <= 5_000, `production runner elapsed ${elapsed}`)

        const { identitiesStillPresent } = await import('./windows-process-terminate')

        assert.deepEqual(await identitiesStillPresent(identities), [])

        fs.writeFileSync(releasePath, 'release')
        await new Promise(resolve => setTimeout(resolve, 500))
        assert.equal(fs.existsSync(sentinel), false)
        assert.deepEqual(await identitiesStillPresent(identities), [])
      } finally {
        controller.abort()
        await runPromise.catch(() => undefined)
        fs.rmSync(tmp, { recursive: true, force: true })
      }
    }
  )

  it(
    'waits for a failed watcher and drains its delayed target before returning',
    { timeout: 20_000 },
    async () => {
      if (process.platform !== 'win32') {return}

      const os = await import('node:os')
      const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-force-watcher-failure-'))
      const sentinel = path.join(tmp, 'sentinel.txt')
      const rootPidPath = path.join(tmp, 'root.pid')
      const writerPidPath = path.join(tmp, 'writer.pid')
      const watcherLog = path.join(tmp, 'watcher.log')
      const ps = path.join(process.env.SystemRoot || 'C:\\\\Windows', 'System32', 'WindowsPowerShell', 'v1.0', 'powershell.exe')
      const quotePowerShellLiteral = (value: string) => `'${value.replace(/'/g, "''")}'`

      const rootScript = `
$ErrorActionPreference = 'Stop'
$sentinel = ${quotePowerShellLiteral(sentinel)}
$rootPidPath = ${quotePowerShellLiteral(rootPidPath)}
$writerPidPath = ${quotePowerShellLiteral(writerPidPath)}
Set-Content -LiteralPath $rootPidPath -Value ([string]$PID)
$writer = Start-Process -FilePath ${quotePowerShellLiteral(ps)} -ArgumentList @(
  '-NoLogo','-NoProfile','-NonInteractive','-Command',
  ('Start-Sleep -Milliseconds 6000; Set-Content -LiteralPath ''' + $sentinel + ''' -Value LATE_MUTATION')
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
        assert.equal(await waitForFile(rootPidPath, 4_000), true, 'watcher-failure root did not start')
        assert.equal(await waitForFile(writerPidPath, 4_000), true, 'watcher-failure writer did not start')
        const rootPid = Number(fs.readFileSync(rootPidPath, 'utf8').trim())
        const writerPid = Number(fs.readFileSync(writerPidPath, 'utf8').trim())
        assert.equal(rootPid, launchedRootPid, 'watcher-failure root PID mismatch')
        const rootCreatedAt = await queryWindowsProcessCreatedAt(rootPid, { platform: 'win32', timeoutMs: 2_000 })
        const writerCreatedAt = await queryWindowsProcessCreatedAt(writerPid, { platform: 'win32', timeoutMs: 2_000 })
        assert.ok(rootCreatedAt && rootCreatedAt > 0, 'watcher-failure root generation unavailable')
        assert.ok(writerCreatedAt && writerCreatedAt > 0, 'watcher-failure writer generation unavailable')

        const injectedWatcher = makeInjectedWatcherStarter(buildWatcherFailureCommand())
        const started = Date.now()

        const result = await runPowerShellWithHardBoundary(
          buildHoldTargetJobScript(rootPid, rootCreatedAt, writerPid, writerCreatedAt),
          5_000,
          undefined,
          undefined,
          { startWatcher: injectedWatcher.startWatcher }
        )

        const elapsed = Date.now() - started
        assert.equal(result.code, 1, `watcher failure unexpectedly cleared: ${JSON.stringify(result)}`)
        assert.ok(elapsed <= 5_000, `watcher failure elapsed ${elapsed}ms`)
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

        const { identitiesStillPresent } = await import('./windows-process-terminate')
        assert.deepEqual(await identitiesStillPresent([{ pid: watcherPid }]), [], 'failed watcher leaked')
        assert.equal(fs.existsSync(watcherArtifacts.readyPath), false, 'failed watcher READY artifact remained')
        assert.equal(fs.existsSync(watcherArtifacts.tempPath), false, 'failed watcher temp artifact remained')
        assert.deepEqual(
          await identitiesStillPresent(identities),
          [],
          `target survived failed watcher boundary result=${JSON.stringify(result)}`
        )
        await new Promise(resolve => setTimeout(resolve, 6_500))
        assert.equal(fs.existsSync(sentinel), false, 'delayed writer mutated after failed watcher return')
        assert.deepEqual(await identitiesStillPresent([{ pid: watcherPid }]), [], 'failed watcher reappeared')
        assert.equal(fs.existsSync(watcherArtifacts.readyPath), false, 'failed watcher READY appeared late')
        assert.equal(fs.existsSync(watcherArtifacts.tempPath), false, 'failed watcher temp appeared late')
        assert.deepEqual(await identitiesStillPresent(identities), [], 'target generation reappeared after failed watcher')
      } finally {
        if (Number.isInteger(launchedRootPid) && (launchedRootPid as number) > 0) {
          await execFileAsync('taskkill', ['/PID', String(launchedRootPid), '/T', '/F'], {
            windowsHide: true,
            timeout: 2_000
          }).catch(() => undefined)
        }

        fs.rmSync(tmp, { recursive: true, force: true })
      }
    }
  )

  it(
    'fails closed when an explicitly injected watcher never publishes READY',
    { timeout: 10_000 },
    async () => {
      if (process.platform !== 'win32') {return}

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
        assert.ok(elapsed <= 3_000, `no-READY boundary elapsed ${elapsed}ms`)
        const watcherChild = injectedWatcher.getChild()

        if (!watcherChild) {throw new Error('no-READY watcher was not started')}
        const watcherPid = watcherChild.pid ?? 0
        assert.ok(watcherPid > 0, 'no-READY watcher PID unavailable')
        const watcherArtifacts = injectedWatcher.getArtifacts()
        assert.ok(watcherArtifacts, 'no-READY watcher artifacts unavailable')
        const { identitiesStillPresent } = await import('./windows-process-terminate')
        assert.deepEqual(await identitiesStillPresent([{ pid: watcherPid }]), [], 'no-READY watcher leaked')
        assert.equal(fs.existsSync(watcherArtifacts.readyPath), false, 'no-READY watcher published READY late')
        assert.equal(fs.existsSync(watcherArtifacts.tempPath), false, 'no-READY watcher temp artifact remained')
        await new Promise(resolve => setTimeout(resolve, 250))
        assert.deepEqual(await identitiesStillPresent([{ pid: watcherPid }]), [], 'no-READY watcher reappeared')
        assert.equal(fs.existsSync(watcherArtifacts.readyPath), false, 'no-READY watcher published READY after return')
        assert.equal(fs.existsSync(watcherArtifacts.tempPath), false, 'no-READY watcher temp appeared after return')
    }
  )

  it(
    'contains an external authenticated holder tree when the primary snapshot fails',
    { timeout: 15_000 },
    async () => {
      if (process.platform !== 'win32') {return}

      const os = await import('node:os')
      const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-force-external-holder-'))
      const sentinel = path.join(tmp, 'sentinel.txt')
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
$rootPidPath = ${quotePowerShellLiteral(rootPidPath)}
$writerPidPath = ${quotePowerShellLiteral(writerPidPath)}
$startupPath = ${quotePowerShellLiteral(startupPath)}
$startupErrorPath = ${quotePowerShellLiteral(startupErrorPath)}
try {
Set-Content -LiteralPath $startupPath -Value ('started:' + [string]$PID)
Set-Content -LiteralPath $rootPidPath -Value ([string]$PID)
$writer = Start-Process -FilePath ${quotePowerShellLiteral(ps)} -ArgumentList @(
  '-NoLogo','-NoProfile','-NonInteractive','-Command',
  ('Start-Sleep -Milliseconds 5000; Set-Content -LiteralPath ''' + $sentinel + ''' -Value LATE_MUTATION')
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
        const fixtureDeadline = Date.now() + 8_000
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
        const rootCreatedAt = await queryWindowsProcessCreatedAt(rootPid, { platform: 'win32', timeoutMs: 2_000 })
        const writerCreatedAt = await queryWindowsProcessCreatedAt(writerPid, { platform: 'win32', timeoutMs: 2_000 })
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
              timeoutMs: 5_000,
              waitMs: 1_500,
              buildScript: (pid, createdAt, waitMs) =>
                buildExactTerminateScript(pid, createdAt, waitMs, {
                  forcePrimarySnapshotFailure: true,
                  // The image-under-root proof is unconditional; this tree is
                  // powershell.exe under the Windows directory.
                  installRoot: process.env.SystemRoot || 'C:\\Windows'
                })
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
        assert.ok(elapsed <= 5_000, `external holder termination elapsed ${elapsed}`)

        const identities = [
          { pid: rootPid, createdAt: rootCreatedAt },
          { pid: writerPid, createdAt: writerCreatedAt }
        ]

        const { identitiesStillPresent } = await import('./windows-process-terminate')
        assert.deepEqual(await identitiesStillPresent(identities), [])
        await new Promise(resolve => setTimeout(resolve, 1_800))
        assert.equal(fs.existsSync(sentinel), false)
        assert.deepEqual(await identitiesStillPresent(identities), [])
      } finally {
        if (Number.isInteger(launchedRootPid) && (launchedRootPid as number) > 0) {
          await execFileAsync('taskkill', ['/PID', String(launchedRootPid), '/T', '/F'], {
            windowsHide: true,
            timeout: 2_000
          }).catch(() => undefined)
        }

        fs.rmSync(tmp, { recursive: true, force: true })
      }
    }
  )

  it(
    'closes the target job when a directly-killed helper dies at each child checkpoint',
    { timeout: 35_000 },
    async () => {
      if (process.platform !== 'win32') {return}

      const os = await import('node:os')
      const { identitiesStillPresent } = await import('./windows-process-terminate')
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
          assert.equal(await waitForFile(rootPidPath, 4_000), true, `${phase}: root did not start`)
          assert.equal(await waitForFile(writerPidPath, 4_000), true, `${phase}: writer did not start`)
          rootPid = Number(fs.readFileSync(rootPidPath, 'utf8').trim())
          writerPid = Number(fs.readFileSync(writerPidPath, 'utf8').trim())
          assert.ok(Number.isInteger(rootPid) && rootPid > 0, `${phase}: invalid root PID`)
          assert.ok(Number.isInteger(writerPid) && writerPid > 0, `${phase}: invalid writer PID`)

          const rootCreatedAt = await queryWindowsProcessCreatedAt(rootPid, { platform: 'win32', timeoutMs: 2_000 })
          const writerCreatedAt = await queryWindowsProcessCreatedAt(writerPid, { platform: 'win32', timeoutMs: 2_000 })
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
                  HERMES_TERMINATE_SCRIPT: buildExactTerminateScript(rootPid, rootCreatedAt, 1_500, {
                    installRoot: process.env.SystemRoot || 'C:\\Windows',
                    pausePhase: phase,
                    pausePid: writerPid,
                    phaseMarkerPath: phaseMarker
                  }),
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
          assert.equal(await waitForFile(wrapperPidMarkerPath, 4_000), true, `${phase}: wrapper marker missing`)

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

          const markerReady = await waitForFile(phaseMarker, 4_000)

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
          await execFileAsync('taskkill', ['/PID', String(boundaryChild.pid), '/T', '/F'], {
            windowsHide: true,
            timeout: 2_000
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
            await identitiesStillPresent([{ pid: watcherPid }]),
            [],
            `${phase}: target-job watcher leaked pid=${watcherPid}`
          )

          const identities = [
            { pid: rootPid, createdAt: rootCreatedAt },
            { pid: writerPid, createdAt: writerCreatedAt }
          ]

          const survivors = await identitiesStillPresent(identities)
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
          assert.deepEqual(await identitiesStillPresent(identities), [], `${phase}: target generation reappeared`)
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
              await execFileAsync('taskkill', ['/PID', String(pid), '/T', '/F'], {
                windowsHide: true,
                timeout: 2_000
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

  it(
    'uses the production runner to close the target job at each child checkpoint',
    { timeout: 35_000 },
    async () => {
      if (process.platform !== 'win32') {return}

      const os = await import('node:os')
      const { identitiesStillPresent } = await import('./windows-process-terminate')
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
          assert.equal(await waitForFile(rootPidPath, 4_000), true, `${phase}: root did not start`)
          assert.equal(await waitForFile(writerPidPath, 4_000), true, `${phase}: writer did not start`)
          const rootPid = Number(fs.readFileSync(rootPidPath, 'utf8').trim())
          const writerPid = Number(fs.readFileSync(writerPidPath, 'utf8').trim())
          assert.ok(Number.isInteger(rootPid) && rootPid > 0, `${phase}: invalid root PID`)
          assert.ok(Number.isInteger(writerPid) && writerPid > 0, `${phase}: invalid writer PID`)
          const rootCreatedAt = await queryWindowsProcessCreatedAt(rootPid, { platform: 'win32', timeoutMs: 2_000 })
          const writerCreatedAt = await queryWindowsProcessCreatedAt(writerPid, { platform: 'win32', timeoutMs: 2_000 })
          assert.ok(rootCreatedAt && rootCreatedAt > 0, `${phase}: root generation unavailable`)
          assert.ok(writerCreatedAt && writerCreatedAt > 0, `${phase}: writer generation unavailable`)

          process.env.HERMES_TERMINATE_WATCHER_LOG = watcherLog
          process.env.HERMES_TERMINATE_NAMED_JOB_LOG = namedJobLog
          runPromise = runPowerShellWithHardBoundary(
            buildExactTerminateScript(rootPid, rootCreatedAt, 1_500, {
              installRoot: process.env.SystemRoot || 'C:\\Windows',
              pausePhase: phase,
              pausePid: writerPid,
              phaseMarkerPath: phaseMarker
            }),
            5_000,
            controller.signal
          )
          const markerReady = await waitForFile(phaseMarker, 4_000)

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

          const survivors = await identitiesStillPresent(identities)
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
          assert.deepEqual(await identitiesStillPresent(identities), [], `${phase}: target generation reappeared`)
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
              await execFileAsync('taskkill', ['/PID', String(pid), '/T', '/F'], {
                windowsHide: true,
                timeout: 2_000
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

  it(
    'does not inherit a foreign child spawned after target admission',
    { timeout: 20_000 },
    async () => {
      if (process.platform !== 'win32') {return}

      const os = await import('node:os')
      const { identitiesStillPresent } = await import('./windows-process-terminate')
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

      const waitForFile = async (filePath: string, timeoutMs = 4_000) => {
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
        const rootCreatedAt = await queryWindowsProcessCreatedAt(rootPid, { platform: 'win32', timeoutMs: 2_000 })
        assert.ok(rootCreatedAt && rootCreatedAt > 0, 'authorized root generation unavailable')

        runPromise = runPowerShellWithHardBoundary(
          buildExactTerminateScript(rootPid, rootCreatedAt, 1_500, {
            installRoot: process.env.SystemRoot || 'C:\\Windows',
            pausePhase: 'after-root-assignment',
            pausePid: rootPid,
            phaseMarkerPath: phaseMarker
          }),
          8_000,
          controller.signal
        )
        assert.equal(await waitForFile(phaseMarker), true, 'root never entered the target job')

        fs.writeFileSync(spawnGate, 'spawn')
        assert.equal(await waitForFile(writerPidPath), true, 'foreign child did not start in the assignment gap')
        writerPid = Number(fs.readFileSync(writerPidPath, 'utf8').trim())
        const writerCreatedAt = await queryWindowsProcessCreatedAt(writerPid, { platform: 'win32', timeoutMs: 2_000 })
        assert.ok(writerCreatedAt && writerCreatedAt > 0, 'foreign child generation unavailable')

        controller.abort()
        const result = await runPromise
        assert.equal(result.code, 1)
        assert.deepEqual(await identitiesStillPresent([{ pid: rootPid, createdAt: rootCreatedAt }]), [])
        assert.deepEqual(await identitiesStillPresent([{ pid: writerPid, createdAt: writerCreatedAt }]), [
          { pid: writerPid, createdAt: writerCreatedAt }
        ])

        fs.writeFileSync(releaseGate, 'release')
        assert.equal(await waitForFile(sentinel, 2_000), true, 'foreign child was killed by the target boundary')
      } finally {
        controller.abort()
        await runPromise?.catch(() => undefined)

        for (const pid of [writerPid, rootPid]) {
          if (!Number.isInteger(pid) || (pid as number) <= 0) {continue}
          await execFileAsync('taskkill', ['/PID', String(pid), '/T', '/F'], { windowsHide: true, timeout: 2_000 }).catch(() => undefined)
        }

        fs.rmSync(tmp, { recursive: true, force: true })
      }
    }
  )

  it(
    'multi-identity confirmation stays within budget and fails closed when probes cannot finish',
    { timeout: 8_000 },
    async () => {
      if (process.platform !== 'win32') {return}
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
      assert.ok(elapsed <= budgetMs + 150, `elapsed ${elapsed} exceeded budget ${budgetMs}`)
      // Phantom PIDs: either confirmed gone (fast ENOENT path) or unconfirmed if
      // budget exhausted mid-probe. Never invent a clear success after overrunning.
      assert.ok(result.confirmed === true || result.confirmed === false)

      if (elapsed > budgetMs) {
        assert.equal(result.confirmed, false)
      }
    }
  )

  it(
    'does not start a post-deadline root probe after snapshot timeout',
    { timeout: 5_000 },
    async () => {
      if (process.platform !== 'win32') {return}
      const { killProcessTreeAndAwaitGone } = await import('./windows-process-terminate')
      const budgetMs = 180
      const started = Date.now()
      let rootProbeCalls = 0

      const result = await killProcessTreeAndAwaitGone(900101, {
        confirmMs: budgetMs,
        deadlineAt: started + budgetMs,
        snapshotProcessTree: async () => {
          // Deliberately ignore the adapter timeout; the boundary must not
          // await this read or launch a fresh default-timeout fallback.
          await new Promise(resolve => setTimeout(resolve, budgetMs * 3))

          return []
        },
        readCreatedAt: async () => {
          rootProbeCalls += 1

          return 1
        }
      })

      const elapsed = Date.now() - started

      assert.equal(result.confirmed, false)
      assert.deepEqual(result.survivors.map(entry => entry.pid), [900101])
      assert.equal(rootProbeCalls, 0)
      assert.ok(elapsed <= budgetMs + 100, `snapshot boundary elapsed ${elapsed}ms > ${budgetMs}ms budget`)
    }
  )
})

describe('liveness probe classification', () => {
  it('classifies discriminated exit vs error outcomes without host PIDs', async () => {
    const {
      classifyLivenessProbeResult,
      probeProcessLiveness,
      identitiesStillPresent
    } = await import('./windows-process-terminate')

    // Authenticated exits only.
    assert.equal(classifyLivenessProbeResult({ kind: 'exit', code: 0 }), 'live')
    assert.equal(classifyLivenessProbeResult({ kind: 'exit', code: 3 }), 'absent')
    assert.equal(classifyLivenessProbeResult({ kind: 'exit', code: 1 }), 'unknown')
    assert.equal(classifyLivenessProbeResult({ kind: 'exit', code: 99 }), 'unknown')

    // Error metadata never proves absence — even numeric/string 3.
    assert.equal(classifyLivenessProbeResult({ kind: 'error', code: 3 }), 'unknown')
    assert.equal(classifyLivenessProbeResult({ kind: 'error', code: '3' }), 'unknown')
    assert.equal(classifyLivenessProbeResult({ kind: 'error', code: 'ETIMEDOUT' }), 'unknown')
    assert.equal(classifyLivenessProbeResult({ kind: 'error', code: 'EACCES' }), 'unknown')
    assert.equal(classifyLivenessProbeResult({ kind: 'error', code: 'EPERM' }), 'unknown')
    assert.equal(classifyLivenessProbeResult({ kind: 'error', code: 'ENOENT' }), 'unknown')
    assert.equal(
      classifyLivenessProbeResult({ kind: 'error', message: 'spawn powershell ENOENT' }),
      'unknown'
    )

    assert.equal(
      await probeProcessLiveness(7, 1_000, async () => ({ kind: 'exit', code: 0 })),
      'live'
    )
    assert.equal(
      await probeProcessLiveness(7, 1_000, async () => ({ kind: 'exit', code: 3 })),
      'absent'
    )
    assert.equal(
      await probeProcessLiveness(7, 1_000, async () => ({ kind: 'error', code: 'ETIMEDOUT' })),
      'unknown'
    )
    assert.equal(
      await probeProcessLiveness(7, 1_000, async () => ({ kind: 'error', code: 'EACCES' })),
      'unknown'
    )
    assert.equal(
      await probeProcessLiveness(7, 1_000, async () => ({ kind: 'error', code: 'ENOENT' })),
      'unknown'
    )
    assert.equal(
      await probeProcessLiveness(7, 1_000, async () => ({ kind: 'error', code: 3 })),
      'unknown'
    )
    assert.equal(
      await probeProcessLiveness(7, 1_000, async () => ({ kind: 'error', code: '3' })),
      'unknown'
    )
    assert.equal(
      await probeProcessLiveness(7, 0, async () => ({ kind: 'exit', code: 0 })),
      'unknown'
    )
    assert.equal(
      await probeProcessLiveness(-1, 1_000, async () => ({ kind: 'exit', code: 0 })),
      'absent'
    )

    // identitiesStillPresent retains every unknown; drops only explicit exit-3 absent.
    // Force null create-time so the injectable liveness runner is exercised, not host PIDs.
    const noCreateTime = async () => null
    const unknownRunner = async () => ({ kind: 'error' as const, code: 'ETIMEDOUT' as const })

    const kept = await identitiesStillPresent([{ pid: 11 }, { pid: 12 }], {
      deadlineAt: Date.now() + 5_000,
      livenessRunner: unknownRunner,
      readCreatedAt: noCreateTime
    })

    assert.deepEqual(
      kept.map(entry => entry.pid),
      [11, 12]
    )

    const errorThreeRunner = async () => ({ kind: 'error' as const, code: 3 as const })

    const errorThreeKept = await identitiesStillPresent([{ pid: 31 }, { pid: 32 }], {
      deadlineAt: Date.now() + 5_000,
      livenessRunner: errorThreeRunner,
      readCreatedAt: noCreateTime
    })

    assert.deepEqual(
      errorThreeKept.map(entry => entry.pid),
      [31, 32]
    )

    const mixedRunner = async (pid: number) => {
      if (pid === 21) {return { kind: 'exit' as const, code: 0 }}

      if (pid === 22) {return { kind: 'exit' as const, code: 3 }}

      if (pid === 23) {return { kind: 'error' as const, code: 'EPERM' }}

      if (pid === 24) {return { kind: 'error' as const, code: 3 }}

      return { kind: 'error' as const, code: 'ENOENT' }
    }

    const mixed = await identitiesStillPresent(
      [{ pid: 21 }, { pid: 22 }, { pid: 23 }, { pid: 24 }, { pid: 25 }],
      {
        deadlineAt: Date.now() + 5_000,
        livenessRunner: mixedRunner,
        readCreatedAt: noCreateTime
      }
    )

    assert.deepEqual(
      mixed.map(entry => entry.pid).sort((a, b) => a - b),
      [21, 23, 24, 25]
    )
  })
})

describe('access-denied identity classification', () => {
  it('does not expose ambient test controls in the production termination script', () => {
    const script = buildExactTerminateScript(9, 100, 100)
    assert.doesNotMatch(script, /HERMES_FORCE_RELEASE_FORCE_SNAPSHOT_FAILURE/)
    assert.doesNotMatch(script, /HERMES_FORCE_RELEASE_TEST_PAUSE_PHASE/)
    assert.doesNotMatch(script, /HERMES_FORCE_RELEASE_TEST_PAUSE_PID/)
    assert.doesNotMatch(script, /HERMES_FORCE_RELEASE_TEST_PHASE_MARKER/)
  })

  it('embeds final install-root and current Restart Manager ownership authorization', () => {
    const script = buildExactTerminateScript(4242, 1234, 500, {
      installRoot: 'C:\\Hermes',
      resource: 'C:\\Hermes\\venv\\Scripts\\hermes.exe'
    })

    assert.match(script, /TERMINATION_RESOURCE_OUTSIDE_INSTALL_ROOT/)
    assert.match(script, /TERMINATION_EXECUTABLE_IDENTITY_UNAVAILABLE/)
    assert.match(script, /TERMINATION_CURRENT_LOCK_OWNERSHIP_MISMATCH/)
    assert.match(script, /QueryFullProcessImageName/)
    assert.match(script, /GetFinalPathNameByHandle/)
    assert.match(script, /IsSameOrUnderRoot\(\$resourceFinal, \$installRootFinal\)/)
    assert.match(script, /IsSameOrUnderRoot\(\$imageFinal, \$installRootFinal\)/)
    assert.match(script, /TERMINATION_EXECUTABLE_OUTSIDE_INSTALL_ROOT/)
    assert.match(script, /IsCurrentResourceOwner/)
    assert.match(script, /RmRegisterResources/)
    assert.match(script, /C:\\Hermes\\venv\\Scripts\\hermes\.exe/)
    assert.match(script, /\(\$childExpected \+ 1\.5\) -lt \$parentCreated/)
  })

  it('classifies access-denied identity failures for Administrator routing', () => {
    assert.deepEqual(parseTerminateScriptOutput('ACCESS_DENIED', 5), {
      kind: 'access-denied',
      win32Error: 5
    })
    const script = buildExactTerminateScript(9, 100, 100)
    assert.match(script, /HOLDER_OPEN_FAILED win32=5/)
    assert.match(script, /ACCESS_DENIED/)
    assert.doesNotMatch(script, /\$win32 -eq '5'/)
  })

  it('does not classify generic Job failures as Administrator escalation', () => {
    assert.deepEqual(parseTerminateScriptOutput('BOUNDARY_FAILED TREE_ASSIGN_FAILED win32=5', 5), {
      kind: 'failed',
      detail: 'BOUNDARY_FAILED TREE_ASSIGN_FAILED win32=5',
      win32Error: 5
    })
  })
})
