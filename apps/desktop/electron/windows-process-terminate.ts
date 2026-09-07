/**
 * SuperF4-style exact process termination for Windows update force-release.
 *
 * Behavioral reference (not copied): stefansundin/superf4 commit
 * 6b677d422553e6b908b9eeaff4333b8b457e7bef, superf4.c lines 236-289.
 * SuperF4 is GPL-3.0 — do not copy its implementation. This module follows
 * Microsoft Win32 documentation for:
 *   OpenProcess(PROCESS_TERMINATE | SYNCHRONIZE)
 *   TerminateProcess
 *   WaitForSingleObject
 *   CloseHandle
 * and optionally AdjustTokenPrivileges(SeDebugPrivilege) when present.
 *
 * The PowerShell child that performs TerminateProcess is itself bounded by a
 * hard wall-clock budget. On expiry the child tree is killed and the call
 * returns only after the child PID is confirmed gone (or a Windows Job Object
 * KILL_ON_JOB_CLOSE terminal is applied). No late mutation after return.
 */

import { type ChildProcess, execFile, spawn } from 'node:child_process'
import { randomBytes } from 'node:crypto'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { promisify } from 'node:util'

import { installSharedRuntimeRoot } from './install-mutation-set'
import { windowsPowerShellExecutable, windowsSystem32Executable } from './windows-powershell-path'
import { psLiteral } from './windows-remote-lifecycle'
import type { ForceReleaseHolder, ForceReleaseTerminateResult, WindowsUpdateForceReleaseDeps } from './windows-update-force-release'

const execFileAsync = promisify(execFile)

// A Job handle does not signal when its last process exits. Query accounting
// under the same deadline to prove that termination drained every member.
const JOB_DRAIN_NATIVE = String.raw`
    [StructLayout(LayoutKind.Sequential)]
    private struct JobAccounting {
        public long TotalUserTime, TotalKernelTime, ThisPeriodTotalUserTime, ThisPeriodTotalKernelTime;
        public uint TotalPageFaultCount, TotalProcesses, ActiveProcesses, TotalTerminatedProcesses;
    }
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool QueryInformationJobObject(IntPtr job, int infoClass, out JobAccounting info, uint length, IntPtr returnLength);
    private static int WaitForJobEmpty(IntPtr job, int waitMs) {
        var timer = System.Diagnostics.Stopwatch.StartNew();
        while (true) {
            JobAccounting info;
            if (!QueryInformationJobObject(job, 1, out info, (uint)Marshal.SizeOf(typeof(JobAccounting)), IntPtr.Zero)) {
                int error = Marshal.GetLastWin32Error();
                return error == 0 ? -1 : -error;
            }
            if (info.ActiveProcesses == 0) return 0;
            long remaining = Math.Max(0, waitMs) - timer.ElapsedMilliseconds;
            if (remaining <= 0) return -258;
            System.Threading.Thread.Sleep((int)Math.Min(10, remaining));
        }
    }
`

export type StartTargetJobWatcher = (
  ownerPid: number,
  ownerCreatedAt: number,
  targetJobName: string,
  watcherReadyPath: string,
  watcherReadyNonce: string,
  deadlineAt: number
) => ChildProcess | undefined

export type HardBoundaryDependencies = {
  startWatcher?: StartTargetJobWatcher
}

const WRAPPER_MARKER_POLL_MS = 10

const WRAPPER_PHASE_NAMES = [
  'marker-published',
  'target-job-created',
  'target-boundary-armed',
  'inner-started',
  'inner-exited',
  'target-terminated',
  'finally-close',
  'handles-closed'
] as const

type TargetJobWatcherDiagnostics = {
  spawnedAt: number
  pid?: number
  errorAt?: number
  errorCode?: string | number
  errorMessage?: string
  exitAt?: number
  exitCode?: number | null
  signalCode?: NodeJS.Signals | null
}

export type RunPowerShell = (
  script: string,
  timeoutMs?: number,
  signal?: AbortSignal,
  deadlineAt?: number,
  dependencies?: HardBoundaryDependencies
) => Promise<{ stdout: string; stderr: string; code: number; pid?: number }>

/** Shared kill/identity-confirm reserve used by the terminate boundary. */
export const TERMINATE_KILL_CONFIRM_MS = 1_500
export const TERMINATE_KILL_CONFIRM_MIN_MS = 400
export const TERMINATE_KILL_CONFIRM_RATIO = 0.3

export function terminateKillReserveMs(budgetMs: number): number {
  const budget = Math.max(0, Math.trunc(budgetMs))

  if (budget <= 0) {return 0}

  return Math.min(
    budget,
    TERMINATE_KILL_CONFIRM_MS,
    Math.max(Math.min(TERMINATE_KILL_CONFIRM_MIN_MS, budget), Math.floor(budget * TERMINATE_KILL_CONFIRM_RATIO))
  )
}

/**
 * The mutating PowerShell script runs behind a KILL_ON_JOB_CLOSE supervisor.
 * If the supervisor is killed during cancellation, Windows closes its job
 * handle and terminates the nested script as a hard terminal boundary. The
 * target script travels through an environment value and a temporary file so
 * it is never interpolated into the supervisor's PowerShell source.
 */
export const TERMINATE_JOB_WRAPPER_COMMAND = String.raw`
$ErrorActionPreference = 'Stop'
$targetScript = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_SCRIPT')
$helperJobName = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_JOB_NAME')
$targetJobName = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_TARGET_JOB_NAME')
$targetWaitText = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_TARGET_WAIT_MS')
$deadlineAtText = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_DEADLINE_AT')
$wrapperPidMarkerPath = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_WRAPPER_PID_MARKER_PATH')
$wrapperPidMarkerNonce = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_WRAPPER_PID_MARKER_NONCE')
$wrapperPhasePath = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_WRAPPER_PHASE_PATH')
$wrapperPhaseNonce = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_WRAPPER_PHASE_NONCE')
$helperScriptPath = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_HELPER_SCRIPT_PATH')
$helperGatePath = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_HELPER_GATE_PATH')
if (
    [string]::IsNullOrWhiteSpace($targetScript) -or
    [string]::IsNullOrWhiteSpace($helperJobName) -or
    [string]::IsNullOrWhiteSpace($targetJobName) -or
    [string]::IsNullOrWhiteSpace($helperScriptPath) -or
    [string]::IsNullOrWhiteSpace($helperGatePath)
) { exit 87 }
function Write-WrapperPidMarker {
    $hasMarkerPath = -not [string]::IsNullOrWhiteSpace($wrapperPidMarkerPath)
    $hasMarkerNonce = -not [string]::IsNullOrWhiteSpace($wrapperPidMarkerNonce)
    if (-not $hasMarkerPath -and -not $hasMarkerNonce) { return }
    if (-not $hasMarkerPath -or -not $hasMarkerNonce) { throw 'WRAPPER_MARKER_INVALID_INPUT' }
    $process = Get-Process -Id $PID -ErrorAction Stop
    $createdAtMs = [DateTimeOffset]::new($process.StartTime.ToUniversalTime()).ToUnixTimeMilliseconds()
    $markerValue = 'PID:' + [string]$PID + ';CREATED_AT_MS:' + [string]$createdAtMs + ';NONCE:' + $wrapperPidMarkerNonce
    $markerTempPath = $wrapperPidMarkerPath + '.tmp'
    $stream = $null
    $writer = $null
    try {
        $stream = [IO.File]::Open($markerTempPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        $writer = [IO.StreamWriter]::new($stream, [Text.UTF8Encoding]::new($false))
        $writer.Write($markerValue)
        $writer.Flush()
        $writer.Dispose()
        $writer = $null
        $stream = $null
        [IO.File]::Move($markerTempPath, $wrapperPidMarkerPath)
    } catch {
        if ($null -ne $writer) { try { $writer.Dispose() } catch {} }
        elseif ($null -ne $stream) { try { $stream.Dispose() } catch {} }
        try { Remove-Item -LiteralPath $markerTempPath -Force -ErrorAction SilentlyContinue } catch {}
        throw
    }
}
function Write-WrapperPhase([string]$phase, [string]$detail = '') {
    if ([string]::IsNullOrWhiteSpace($wrapperPhasePath) -or [string]::IsNullOrWhiteSpace($wrapperPhaseNonce)) { return }
    $safePhase = $phase -replace '[^A-Za-z0-9_-]', '_'
    $phasePath = $wrapperPhasePath + '.' + $safePhase + '.marker'
    $phaseTempPath = $phasePath + '.tmp'
    $safeDetail = ([string]$detail) -replace '[\r\n;]', '_'
    $phaseValue = 'PHASE:' + $wrapperPhaseNonce + ';NAME:' + $safePhase + ';TICKS:' + [Diagnostics.Stopwatch]::GetTimestamp() + ';DETAIL:' + $safeDetail
    $stream = $null
    $writer = $null
    try {
        $stream = [IO.File]::Open($phaseTempPath, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write, [IO.FileShare]::None)
        $writer = [IO.StreamWriter]::new($stream, [Text.UTF8Encoding]::new($false))
        $writer.Write($phaseValue)
        $writer.Flush()
        $writer.Dispose()
        $writer = $null
        $stream = $null
        [IO.File]::Move($phaseTempPath, $phasePath)
    } catch {
        if ($null -ne $writer) { try { $writer.Dispose() } catch {} }
        elseif ($null -ne $stream) { try { $stream.Dispose() } catch {} }
        try { Remove-Item -LiteralPath $phaseTempPath -Force -ErrorAction SilentlyContinue } catch {}
    }
}
Write-WrapperPidMarker
Write-WrapperPhase 'marker-published'
$targetWaitMs = 1500
$parsedTargetWaitMs = 0
if ([int]::TryParse($targetWaitText, [ref]$parsedTargetWaitMs)) {
    $targetWaitMs = [Math]::Max(0, $parsedTargetWaitMs)
}
$deadlineAt = 0L
$parsedDeadlineAt = 0L
if ([long]::TryParse($deadlineAtText, [ref]$parsedDeadlineAt)) {
    $deadlineAt = [Math]::Max(0L, $parsedDeadlineAt)
}
function Get-TargetWaitMs {
    if ($deadlineAt -le 0) { return $targetWaitMs }
    $remaining = $deadlineAt - [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()
    if ($remaining -le 0) { return -1 }
    return [Math]::Min($targetWaitMs, [int][Math]::Min($remaining, [long][int]::MaxValue))
}

Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;

public static class HermesTerminateJob {
${JOB_DRAIN_NATIVE}
    [StructLayout(LayoutKind.Sequential)]
    private struct BasicLimits {
        public long PerProcessUserTimeLimit, PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize, MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass, SchedulingClass;
    }
    [StructLayout(LayoutKind.Sequential)]
    private struct IoCounters {
        public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount;
        public ulong ReadTransferCount, WriteTransferCount, OtherTransferCount;
    }
    [StructLayout(LayoutKind.Sequential)]
    private struct ExtendedLimits {
        public BasicLimits BasicLimitInformation;
        public IoCounters IoInfo;
        public UIntPtr ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemoryUsed, PeakJobMemoryUsed;
    }
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObject(IntPtr attributes, string name);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool TerminateJobObject(IntPtr job, uint exitCode);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint WaitForSingleObject(IntPtr handle, uint milliseconds);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetInformationJobObject(IntPtr job, int infoClass, ref ExtendedLimits info, uint length);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool IsProcessInJob(IntPtr process, IntPtr job, out bool result);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr handle);

    public static IntPtr CreateKillOnClose(string name, bool allowDescendantBreakaway) {
        IntPtr job = CreateJobObject(IntPtr.Zero, name);
        if (job == IntPtr.Zero) throw new Win32Exception();
        var limits = new ExtendedLimits();
        limits.BasicLimitInformation.LimitFlags = 0x00002000U | (allowDescendantBreakaway ? 0x00001000U : 0U);
        if (!SetInformationJobObject(job, 9, ref limits, (uint)Marshal.SizeOf(typeof(ExtendedLimits)))) {
            int error = Marshal.GetLastWin32Error();
            CloseHandle(job);
            throw new Win32Exception(error);
        }
        return job;
    }
    public static void Assign(IntPtr job, IntPtr process) {
        if (!AssignProcessToJobObject(job, process)) throw new Win32Exception();
    }
    public static int TerminateAndWait(IntPtr job, int waitMs) {
        if (!TerminateJobObject(job, 1)) {
            int error = Marshal.GetLastWin32Error();
            return error == 0 ? -1 : -error;
        }
        return WaitForJobEmpty(job, waitMs);
    }
    public static void Close(IntPtr job) {
        if (job != IntPtr.Zero) CloseHandle(job);
    }
    public static string QuoteArgument(string value) {
        if (value == null) return "\"\"";
        var quoted = new StringBuilder("\"");
        int slashes = 0;
        foreach (char current in value) {
            if (current == '\\') { slashes++; continue; }
            if (current == '"') {
                quoted.Append('\\', slashes * 2 + 1).Append('"');
                slashes = 0;
                continue;
            }
            quoted.Append('\\', slashes).Append(current);
            slashes = 0;
        }
        quoted.Append('\\', slashes * 2).Append('"');
        return quoted.ToString();
    }
}
'@ -ErrorAction Stop

$helperJob = [IntPtr]::Zero
$targetJob = [IntPtr]::Zero
$child = $null
$tempScript = $helperScriptPath
$targetTerminationComplete = $false
$finalExitCode = 1
try {
    $gatePrologue = @'
$helperGatePath = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_HELPER_GATE_PATH')
$helperGateDeadlineText = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_DEADLINE_AT')
$helperGateDeadline = 0L
[void][long]::TryParse($helperGateDeadlineText, [ref]$helperGateDeadline)
while (-not (Test-Path -LiteralPath $helperGatePath)) {
    if ($helperGateDeadline -gt 0 -and [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() -ge $helperGateDeadline) { exit 87 }
    Start-Sleep -Milliseconds 5
}
'@
    [IO.File]::WriteAllText(
        $tempScript,
        ($gatePrologue + [Environment]::NewLine + $targetScript),
        [Text.UTF8Encoding]::new($false)
    )
    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
    $psi.Arguments = '-NoLogo -NoProfile -NonInteractive -ExecutionPolicy Bypass -File ' + [HermesTerminateJob]::QuoteArgument($tempScript)
    $psi.UseShellExecute = $false
    $psi.CreateNoWindow = $true
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.EnvironmentVariables['HERMES_TERMINATE_HELPER_GATE_PATH'] = $helperGatePath

    # The wrapper owns two independent kill-on-close jobs. The gated inner
    # helper is assigned to helperJob before its script can run; the inner script
    # opens targetJob only to assign the separately authenticated external target
    # tree. Wrapper death therefore contains helper descendants without changing
    # the target tree's job membership contract.
    # Target descendants must be admitted explicitly after identity and image
    # authorization. The helper job stays closed so its subprocess cannot
    # escape the cancellation boundary.
    $targetJob = [HermesTerminateJob]::CreateKillOnClose($targetJobName, $true)
    $helperJob = [HermesTerminateJob]::CreateKillOnClose($helperJobName, $false)
    Write-WrapperPhase 'target-job-created'
    # The wrapper owns the only persistent target-Job handle. Wrapper death
    # closes it and applies KILL_ON_JOB_CLOSE to every assigned target.
    Write-WrapperPhase 'target-boundary-armed'
    $child = [System.Diagnostics.Process]::Start($psi)
    [HermesTerminateJob]::Assign($helperJob, $child.Handle)
    [IO.File]::WriteAllText($helperGatePath, 'GO', [Text.UTF8Encoding]::new($false))
    Write-WrapperPhase 'inner-started' ('pid=' + [string]$child.Id)
    $stdoutTask = $child.StandardOutput.ReadToEndAsync()
    $stderrTask = $child.StandardError.ReadToEndAsync()
    $child.WaitForExit()
    $childStdout = $stdoutTask.Result
    $childStderr = $stderrTask.Result
    Write-WrapperPhase 'inner-exited' ('code=' + [string]$child.ExitCode)
    # A successful inner TERMINATED marker means the inner script already
    # called TerminateJobObject and waited for the shared target job to drain.
    # Do not start a second relative drain after the absolute deadline has
    # expired; the wrapper's owned handle still closes in finally.
    if ($child.ExitCode -eq 0 -and $childStdout -match '(?m)^\s*TERMINATED\s*$') {
        $targetTerminationComplete = $true
        Write-WrapperPhase 'target-terminated'
        [Console]::Out.Write($childStdout)
        [Console]::Error.Write($childStderr)
        $finalExitCode = $child.ExitCode
    } else {
        $drainWaitMs = [int](Get-TargetWaitMs)
        if ($drainWaitMs -lt 0) {
            [Console]::Out.Write($childStdout)
            [Console]::Error.Write($childStderr)
            [Console]::Error.WriteLine('TARGET_JOB_DEADLINE_EXHAUSTED')
            $finalExitCode = 1
        } else {
            $targetCode = [HermesTerminateJob]::TerminateAndWait($targetJob, $drainWaitMs)
            if ($targetCode -ne 0) {
                [Console]::Out.Write($childStdout)
                [Console]::Error.Write($childStderr)
                [Console]::Error.WriteLine('TARGET_JOB_TERMINATE_FAILED win32=' + (-$targetCode))
                $finalExitCode = 1
            } else {
                $targetTerminationComplete = $true
                Write-WrapperPhase 'target-terminated'
                [Console]::Out.Write($childStdout)
                [Console]::Error.Write($childStderr)
                $finalExitCode = $child.ExitCode
            }
        }
    }
} catch {
    if ($null -ne $child) {
        try { if (!$child.HasExited) { $child.Kill() } } catch {}
    }
    $failure = [string]$_.Exception.Message
    if ($targetJob -ne [IntPtr]::Zero -and -not $targetTerminationComplete) {
        try {
            $drainWaitMs = [int](Get-TargetWaitMs)
            if ($drainWaitMs -lt 0) {
                $failure += ' TARGET_JOB_DEADLINE_EXHAUSTED'
            } else {
                $targetCode = [HermesTerminateJob]::TerminateAndWait($targetJob, $drainWaitMs)
                if ($targetCode -ne 0) {
                    $failure += ' TARGET_JOB_TERMINATE_FAILED win32=' + (-$targetCode)
                }
            }
        } catch {
            $failure += ' TARGET_JOB_TERMINATE_FAILED ' + [string]$_.Exception.Message
        }
    }
    [Console]::Error.WriteLine($failure)
    $finalExitCode = 1
} finally {
    Write-WrapperPhase 'finally-close'
    if ($null -ne $child) { $child.Dispose() }
    [HermesTerminateJob]::Close($targetJob)
    [HermesTerminateJob]::Close($helperJob)
    Remove-Item -LiteralPath $helperGatePath -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $tempScript -Force -ErrorAction SilentlyContinue
    Write-WrapperPhase 'handles-closed'
}
[Environment]::Exit($finalExitCode)
`.trim()

/**
 * Direct sibling termination of the persistent target job. This is invoked
 * before killing the wrapper so cancellation does not depend on observing the
 * wrapper's final handle teardown or on taskkill's process-tree traversal.
 */
export const TERMINATE_NAMED_JOB_COMMAND = String.raw`
$ErrorActionPreference = 'Stop'
$targetJobName = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_TARGET_JOB_NAME')
$targetWaitText = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_TARGET_WAIT_MS')
$diagnosticLog = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_NAMED_JOB_LOG')
function Write-Diagnostic([string]$message) {
    if ([string]::IsNullOrWhiteSpace($diagnosticLog)) { return }
    try { [IO.File]::AppendAllText($diagnosticLog, ($message + [Environment]::NewLine)) } catch {}
}
if ([string]::IsNullOrWhiteSpace($targetJobName)) { Write-Diagnostic 'invalid-input'; exit 87 }
Write-Diagnostic ('started job=' + $targetJobName)
$targetWaitMs = 500
$parsedTargetWaitMs = 0
if ([int]::TryParse($targetWaitText, [ref]$parsedTargetWaitMs)) {
    $targetWaitMs = [Math]::Max(0, $parsedTargetWaitMs)
}
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class HermesTerminateNamedJob {
${JOB_DRAIN_NATIVE}
    private const uint JOB_OBJECT_TERMINATE = 0x0008;
    private const uint JOB_OBJECT_QUERY = 0x0004;
    private const uint SYNCHRONIZE = 0x00100000;
    private const uint WAIT_OBJECT_0 = 0;
    private const uint WAIT_TIMEOUT = 258;
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr OpenJobObject(uint desiredAccess, bool inheritHandle, string name);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool TerminateJobObject(IntPtr job, uint exitCode);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint WaitForSingleObject(IntPtr handle, uint milliseconds);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr handle);
    public static int OpenAndTerminate(string name, int waitMs) {
        IntPtr job = OpenJobObject(JOB_OBJECT_TERMINATE | JOB_OBJECT_QUERY | SYNCHRONIZE, false, name);
        if (job == IntPtr.Zero) {
            int openError = Marshal.GetLastWin32Error();
            return openError == 2 || openError == 6 ? 0 : -openError;
        }
        try {
            if (!TerminateJobObject(job, 1)) {
                int terminateError = Marshal.GetLastWin32Error();
                return terminateError == 0 ? -1 : -terminateError;
            }
            return WaitForJobEmpty(job, waitMs);
        } finally {
            CloseHandle(job);
        }
    }
}
'@ -ErrorAction Stop
$exitCode = 1
try {
    $result = [HermesTerminateNamedJob]::OpenAndTerminate($targetJobName, $targetWaitMs)
    Write-Diagnostic ('result=' + $result)
    if ($result -eq 0) {
        $exitCode = 0
    } else {
        [Console]::Error.WriteLine('TARGET_JOB_TERMINATE_FAILED win32=' + (-$result))
    }
} catch {
    Write-Diagnostic ('exception=' + [string]$_.Exception.Message)
    [Console]::Error.WriteLine('TARGET_JOB_TERMINATE_FAILED ' + [string]$_.Exception.Message)
}
exit $exitCode
`.trim()

/**
 * Detached sibling watcher for the named target job. It opens an authenticated
 * process handle for the wrapper, waits for that exact process object to exit,
 * then terminates the target job. This covers the narrow window where a direct
 * supervisor kill can finish before the wrapper's last job handle is observed
 * closed by the terminating caller.
 */
export const TERMINATE_JOB_WATCHER_COMMAND = String.raw`
$ErrorActionPreference = 'Stop'
$ownerPidText = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_OWNER_PID')
$ownerCreatedAtText = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_OWNER_CREATED_AT')
$targetJobName = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_TARGET_JOB_NAME')
$watcherLog = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_WATCHER_LOG')
$watcherReadyPath = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_WATCHER_READY_PATH')
$watcherReadyNonce = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_WATCHER_READY_NONCE')
$watcherDeadlineAtText = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_WATCHER_DEADLINE_AT')
function Write-WatcherLog([string]$message) {
    if ([string]::IsNullOrWhiteSpace($watcherLog)) { return }
    try { [IO.File]::AppendAllText($watcherLog, ($message + [Environment]::NewLine)) } catch {}
}
function Write-WatcherReady([string]$value) {
    if ([string]::IsNullOrWhiteSpace($watcherReadyPath) -or [string]::IsNullOrWhiteSpace($watcherReadyNonce)) { return }
    $tempPath = $watcherReadyPath + '.tmp'
    try {
        [IO.File]::WriteAllText($tempPath, $value)
        [IO.File]::Move($tempPath, $watcherReadyPath)
    } catch {
        try { Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue } catch {}
    }
}
$ownerPid = 0
$ownerCreatedAt = 0.0
$watcherDeadlineAt = 0L
[long]::TryParse($watcherDeadlineAtText, [ref]$watcherDeadlineAt) | Out-Null
[double]::TryParse($ownerCreatedAtText, [ref]$ownerCreatedAt) | Out-Null
if (
    -not [int]::TryParse($ownerPidText, [ref]$ownerPid) -or
    $ownerCreatedAt -le 0 -or
    [string]::IsNullOrWhiteSpace($targetJobName) -or
    [string]::IsNullOrWhiteSpace($watcherReadyPath) -or
    [string]::IsNullOrWhiteSpace($watcherReadyNonce) -or
    $watcherDeadlineAt -le 0
) {
    Write-WatcherLog 'invalid-input'
    exit 87
}
Write-WatcherLog ('started owner=' + $ownerPid + ' job=' + $targetJobName)
Write-WatcherLog ('waiting owner=' + $ownerPid)
try {
Add-Type -TypeDefinition @'
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading;
public static class HermesTerminateWatch {
${JOB_DRAIN_NATIVE}
    private const uint PROCESS_QUERY_LIMITED_INFORMATION = 0x1000;
    private const uint SYNCHRONIZE = 0x00100000;
    private const uint JOB_OBJECT_TERMINATE = 0x0008;
    private const uint JOB_OBJECT_QUERY = 0x0004;
    private const uint WAIT_OBJECT_0 = 0;
    private const uint WAIT_TIMEOUT = 258;
    private const uint WAIT_FAILED = 0xFFFFFFFF;
    [StructLayout(LayoutKind.Sequential)]
    private struct FileTime { public uint Low; public uint High; }
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool GetProcessTimes(IntPtr process, out FileTime creation, out FileTime exit, out FileTime kernel, out FileTime user);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern IntPtr OpenProcess(uint desiredAccess, bool inheritHandle, int processId);
    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr OpenJobObject(uint desiredAccess, bool inheritHandle, string name);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint WaitForSingleObject(IntPtr handle, uint milliseconds);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool TerminateJobObject(IntPtr job, uint exitCode);
    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CloseHandle(IntPtr handle);
    private static long NowUnixMilliseconds() {
        return DateTimeOffset.UtcNow.ToUnixTimeMilliseconds();
    }
    private static uint Remaining(long deadlineAt) {
        long remaining = deadlineAt - NowUnixMilliseconds();
        if (remaining <= 0) return 0;
        return (uint)Math.Min(remaining, 0xFFFFFFFEL);
    }
    private static double ToUnixSeconds(FileTime time) {
        long fileTicks = ((long)time.High << 32) | time.Low;
        return (fileTicks - 116444736000000000L) / 10000000.0;
    }
    public static int ReadCreatedAt(IntPtr process, out double createdAt) {
        createdAt = 0;
        FileTime creation, exit, kernel, user;
        if (!GetProcessTimes(process, out creation, out exit, out kernel, out user)) {
            int error = Marshal.GetLastWin32Error();
            return error == 0 ? -1 : -error;
        }
        createdAt = ToUnixSeconds(creation);
        return 0;
    }
    private static IntPtr OpenTargetJobUntilDeadline(string jobName, long deadlineAt, out int error) {
        error = 0;
        while (true) {
            IntPtr job = OpenJobObject(JOB_OBJECT_TERMINATE | JOB_OBJECT_QUERY | SYNCHRONIZE, false, jobName);
            if (job != IntPtr.Zero) return job;
            error = Marshal.GetLastWin32Error();
            if (error != 2 && error != 6) return IntPtr.Zero;
            uint remaining = Remaining(deadlineAt);
            if (remaining == 0) {
                error = (int)WAIT_TIMEOUT;
                return IntPtr.Zero;
            }
            Thread.Sleep((int)Math.Min(10U, remaining));
        }
    }
    private static int Failure(string readyPath, string readyNonce, string stage, int error) {
        TryWriteReady(readyPath, "FAILED:" + readyNonce + " " + stage + " win32=" + error);
        return error == 0 ? -1 : -error;
    }
    private static bool TryWriteReady(string path, string value) {
        if (String.IsNullOrWhiteSpace(path)) return false;
        string tempPath = path + "." + System.Diagnostics.Process.GetCurrentProcess().Id + ".tmp";
        try {
            using (var stream = new FileStream(tempPath, FileMode.CreateNew, FileAccess.Write, FileShare.None))
            using (var writer = new StreamWriter(stream)) {
                writer.Write(value);
                writer.Flush();
            }
            File.Move(tempPath, path);
            return true;
        } catch {
            try { File.Delete(tempPath); } catch {}
            return false;
        }
    }
    public static int WaitForOwnerThenTerminate(int ownerPid, double expectedOwnerCreatedAt, string jobName, string readyPath, string readyNonce, long deadlineAt) {
        IntPtr owner = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | SYNCHRONIZE, false, ownerPid);
        if (owner == IntPtr.Zero) {
            return Failure(readyPath, readyNonce, "owner-open", Marshal.GetLastWin32Error());
        }
        double actualOwnerCreatedAt = 0;
        int ownerGenerationCode = ReadCreatedAt(owner, out actualOwnerCreatedAt);
        if (ownerGenerationCode != 0) {
            CloseHandle(owner);
            return Failure(readyPath, readyNonce, "owner-generation", -ownerGenerationCode);
        }
        if (Math.Abs(actualOwnerCreatedAt - expectedOwnerCreatedAt) > 1.5) {
            CloseHandle(owner);
            return Failure(readyPath, readyNonce, "owner-generation", 0x10001);
        }
        int openJobError = 0;
        IntPtr job = OpenTargetJobUntilDeadline(jobName, deadlineAt, out openJobError);
        if (job == IntPtr.Zero) {
            CloseHandle(owner);
            return Failure(readyPath, readyNonce, "job-open", openJobError);
        }
        try {
            if (!TryWriteReady(readyPath, "ARMED:" + readyNonce)) return Failure(readyPath, readyNonce, "ready-write", 1);
            uint ownerRemaining = Remaining(deadlineAt);
            if (ownerRemaining == 0) return Failure(readyPath, readyNonce, "owner-deadline", (int)WAIT_TIMEOUT);
            uint ownerWait = WaitForSingleObject(owner, ownerRemaining);
            if (ownerWait == WAIT_TIMEOUT) return Failure(readyPath, readyNonce, "owner-wait", (int)WAIT_TIMEOUT);
            if (ownerWait == WAIT_FAILED) return Failure(readyPath, readyNonce, "owner-wait", Marshal.GetLastWin32Error());
            if (!TerminateJobObject(job, 1)) {
                return Failure(readyPath, readyNonce, "job-terminate", Marshal.GetLastWin32Error());
            }
            uint jobRemaining = Remaining(deadlineAt);
            if (jobRemaining == 0) return Failure(readyPath, readyNonce, "job-deadline", (int)WAIT_TIMEOUT);
            int result = WaitForJobEmpty(job, (int)Math.Min(1500U, jobRemaining));
            if (result == 0) return 0;
            return Failure(readyPath, readyNonce, "job-wait", -result);
        } finally {
            CloseHandle(job);
            CloseHandle(owner);
        }
    }
}
'@ -ErrorAction Stop
  $watchResult = [HermesTerminateWatch]::WaitForOwnerThenTerminate($ownerPid, $ownerCreatedAt, $targetJobName, $watcherReadyPath, $watcherReadyNonce, $watcherDeadlineAt)
  Write-WatcherLog ('completed result=' + $watchResult)
  if ($watchResult -ne 0) {
    Write-WatcherReady ('FAILED:' + $watcherReadyNonce + ' watcher-result=' + $watchResult)
  }
  exit $watchResult
} catch {
  Write-WatcherLog ('exception=' + [string]$_.Exception.Message)
  Write-WatcherReady ('FAILED:' + $watcherReadyNonce + ' exception=' + [string]$_.Exception.Message)
  exit 1
}
`.trim()

const TERMINATE_JOB_WATCHER_BOOTSTRAP = String.raw`
$watcherScript = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_WATCHER_SCRIPT')
if ([string]::IsNullOrWhiteSpace($watcherScript)) { exit 87 }
& ([ScriptBlock]::Create($watcherScript))
`.trim()

export const TERMINATE_JOB_WATCHER_BRIDGE = String.raw`
const { spawnSync } = require('node:child_process')
const executable = process.env.HERMES_TERMINATE_WATCHER_POWERSHELL
const encodedCommand = process.env.HERMES_TERMINATE_WATCHER_ENCODED_COMMAND
if (!executable || !encodedCommand) process.exit(87)
const result = spawnSync(
  executable,
  ['-NoLogo', '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass', '-EncodedCommand', encodedCommand],
  { detached: false, windowsHide: true, stdio: 'ignore', env: process.env }
)
process.exit(Number.isInteger(result.status) ? result.status : 1)
`.trim()

export type ProcessIdentity = { pid: number; createdAt?: number }

export function parseWrapperProcessMarker(value: string, expectedNonce: string): ProcessIdentity | null {
  const match = /^PID:(\d+);CREATED_AT_MS:(\d+);NONCE:([A-Za-z0-9_-]+)$/.exec(String(value ?? '').trim())

  if (!match || match[3] !== expectedNonce) {return null}
  const pid = Number(match[1])
  const createdAtMs = Number(match[2])

  if (!Number.isSafeInteger(pid) || pid <= 0 || !Number.isSafeInteger(createdAtMs) || createdAtMs <= 0) {return null}

  return { pid, createdAt: createdAtMs / 1_000 }
}

async function waitForWrapperProcessMarker(
  markerPath: string,
  markerNonce: string,
  deadlineAt: number
): Promise<ProcessIdentity | null> {
  while (Date.now() < deadlineAt) {
    try {
      if (fs.existsSync(markerPath)) {
        const identity = parseWrapperProcessMarker(fs.readFileSync(markerPath, 'utf8'), markerNonce)

        if (identity && identity.createdAt != null) {return identity}
      }
    } catch {
      // A partial or stale marker is not an authenticated wrapper identity.
    }

    const remaining = Math.max(0, deadlineAt - Date.now())

    if (remaining <= 0) {break}
    await new Promise(resolve => setTimeout(resolve, Math.min(WRAPPER_MARKER_POLL_MS, remaining)))
  }

  return null
}

function readWrapperPhaseDiagnostics(phasePath: string, phaseNonce: string): string {
  const summary = WRAPPER_PHASE_NAMES.map(phase => {
    const markerPath = `${phasePath}.${phase}.marker`
    const expectedPrefix = `PHASE:${phaseNonce};NAME:${phase};TICKS:`

    try {
      if (!fs.existsSync(markerPath)) {return `${phase}=missing`}
      const value = fs.readFileSync(markerPath, 'utf8').trim().replace(/[\r\n]+/g, ' ')

      return value.startsWith(expectedPrefix) ? `${phase}=${value.slice(0, 512)}` : `${phase}=invalid`
    } catch {
      return `${phase}=unreadable`
    }
  })

  return `wrapper-phases ${summary.join(' ')}`.slice(0, 4_096)
}

function hasSuccessfulWrapperReceipt(phasePath: string, phaseNonce: string): boolean {
  const readPhase = (phase: (typeof WRAPPER_PHASE_NAMES)[number]): string | undefined => {
    try {
      const value = fs.readFileSync(`${phasePath}.${phase}.marker`, 'utf8').trim()
      const expectedPrefix = `PHASE:${phaseNonce};NAME:${phase};TICKS:`

      return value.startsWith(expectedPrefix) ? value : undefined
    } catch {
      return undefined
    }
  }

  const innerExited = readPhase('inner-exited')

  return Boolean(
    innerExited?.endsWith(';DETAIL:code=0') &&
      readPhase('target-terminated') &&
      readPhase('handles-closed')
  )
}

function hasTargetBoundaryReceipt(phasePath: string, phaseNonce: string): boolean {
  const hasPhase = (phase: 'target-terminated' | 'handles-closed'): boolean => {
    try {
      const value = fs.readFileSync(`${phasePath}.${phase}.marker`, 'utf8').trim()

      return value.startsWith(`PHASE:${phaseNonce};NAME:${phase};TICKS:`)
    } catch {
      return false
    }
  }

  return hasPhase('target-terminated') && hasPhase('handles-closed')
}

function readExactWatcherReadyValue(watcherReadyPath: string): string | undefined {
  try {
    if (!fs.existsSync(watcherReadyPath)) {return undefined}

    return fs.readFileSync(watcherReadyPath, 'utf8').trim().replace(/[\r\n]+/g, ' ').slice(0, 512)
  } catch {
    return undefined
  }
}

async function readProcessCreatedAt(
  pid: number,
  timeoutMs = 2_000
): Promise<number | null> {
  if (!Number.isInteger(pid) || pid <= 0) {return null}
  const budget = Math.trunc(timeoutMs)

  if (budget <= 0) {return null}

  if (process.platform !== 'win32') {return null}

  try {
    const { stdout } = await execFileAsync(
      windowsPowerShellExecutable(),
      [
        '-NoLogo',
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        `$p = Get-Process -Id ${Math.trunc(pid)} -ErrorAction Stop; ` +
          'if ($p.HasExited -or $p.WaitForExit(0)) { exit 3 }; ' +
          '[DateTimeOffset]::new($p.StartTime.ToUniversalTime()).ToUnixTimeSeconds()'
      ],
      { encoding: 'utf8', windowsHide: true, timeout: budget }
    )

    const value = Number(String(stdout).trim())

    return Number.isFinite(value) && value > 0 ? value : null
  } catch {
    return null
  }
}

export type ProcessLiveness = 'live' | 'absent' | 'unknown'

/**
 * Discriminated liveness probe outcome.
 * Only authenticated child-process exits may prove live/absent.
 * Error metadata (timeout/access/spawn) is never treated as an exit code.
 */
export type LivenessProbeResult =
  | { kind: 'exit'; code: number }
  | { kind: 'error'; code?: string | number; message?: string }

/** Injectable runner for process liveness probes (tests inject fakes). */
export type LivenessProbeRunner = (
  pid: number,
  timeoutMs: number
) => Promise<LivenessProbeResult>

type DeadlineProbeResult<T> =
  | { completed: true; value: T }
  | { completed: false }

/**
 * Resolve a read-only probe only while the shared deadline remains. The
 * underlying native operation is deliberately consumed after a timeout so a
 * late rejection cannot become an unhandled promise, but its value is never
 * allowed back into the mutation boundary.
 */
async function resolveBeforeDeadline<T>(
  operation: () => Promise<T>,
  deadlineAt: number
): Promise<DeadlineProbeResult<T>> {
  const remaining = Math.max(0, deadlineAt - Date.now())

  if (remaining <= 0) {return { completed: false }}

  const pending = Promise.resolve().then(operation)
  let timer: NodeJS.Timeout | undefined

  const timeout = new Promise<DeadlineProbeResult<T>>(resolve => {
    timer = setTimeout(() => resolve({ completed: false }), Math.max(1, remaining - 1))
  })

  try {
    const outcome = await Promise.race([
      pending.then(value => ({ completed: true as const, value })),
      timeout
    ])

    if (!outcome.completed || Date.now() > deadlineAt) {return { completed: false }}

    return outcome
  } catch {
    return { completed: false }
  } finally {
    if (timer) {clearTimeout(timer)}
    void pending.catch(() => undefined)
  }
}

/**
 * Pure classification of a liveness probe outcome.
 * live only for kind=exit/code=0; absent only for kind=exit/code=3.
 * Every kind=error is unknown regardless of embedded code metadata.
 */
export function classifyLivenessProbeResult(result: LivenessProbeResult): ProcessLiveness {
  if (result.kind === 'exit') {
    if (result.code === 0) {return 'live'}

    if (result.code === 3) {return 'absent'}

    return 'unknown'
  }

  // kind=error: timeout/access/spawn/malformed — never prove absence from error metadata.
  return 'unknown'
}

function execFileFailureToLivenessResult(error: any): LivenessProbeResult {
  // Timeout/killed probes are errors even if a numeric code is present.
  if (error?.killed === true || error?.signal) {
    return {
      kind: 'error',
      code: typeof error?.code === 'string' || typeof error?.code === 'number' ? error.code : undefined,
      message: String(error?.message ?? error)
    }
  }

  // Node execFile puts authenticated child exit status on error.code as a number.
  if (typeof error?.code === 'number') {
    return { kind: 'exit', code: error.code }
  }

  // Some paths expose exit status on .status while .code is a string errno.
  if (typeof error?.status === 'number') {
    return { kind: 'exit', code: error.status }
  }

  return {
    kind: 'error',
    code: typeof error?.code === 'string' || typeof error?.code === 'number' ? error.code : undefined,
    message: String(error?.message ?? error)
  }
}

async function defaultLivenessProbeRunner(
  pid: number,
  timeoutMs: number
): Promise<LivenessProbeResult> {
  if (process.platform !== 'win32') {
    try {
      process.kill(pid, 0)

      return { kind: 'exit', code: 0 }
    } catch (error: any) {
      // ESRCH is an authenticated "no such process" from the kill(2) probe.
      if (error?.code === 'ESRCH') {return { kind: 'exit', code: 3 }}

      return {
        kind: 'error',
        code: error?.code,
        message: String(error?.message ?? error)
      }
    }
  }

  try {
    await execFileAsync(
      windowsPowerShellExecutable(),
      [
        '-NoLogo',
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        // A terminated process object can remain enumerable while another
        // process still owns a handle. Its zero-time wait is nevertheless
        // authoritative terminal-state evidence. Exit 0 = runnable, 3 =
        // absent/terminated, anything else = unknown.
        `$p = Get-Process -Id ${Math.trunc(pid)} -ErrorAction SilentlyContinue; ` +
          'if ($null -eq $p) { exit 3 }; ' +
          'try { if ($p.HasExited -or $p.WaitForExit(0)) { exit 3 }; exit 0 } catch { exit 4 }'
      ],
      { windowsHide: true, timeout: timeoutMs }
    )

    return { kind: 'exit', code: 0 }
  } catch (error: any) {
    return execFileFailureToLivenessResult(error)
  }
}

export async function probeProcessLiveness(
  pid: number,
  timeoutMs = 2_000,
  runner: LivenessProbeRunner = defaultLivenessProbeRunner
): Promise<ProcessLiveness> {
  if (!Number.isInteger(pid) || pid <= 0) {return 'absent'}
  const budget = Math.trunc(timeoutMs)

  if (budget <= 0) {return 'unknown'}
  const result = await runner(pid, budget)

  return classifyLivenessProbeResult(result)
}

export async function identitiesStillPresent(
  identities: readonly ProcessIdentity[],
  {
    deadlineAt,
    livenessRunner,
    readCreatedAt
  }: {
    deadlineAt?: number
    livenessRunner?: LivenessProbeRunner
    readCreatedAt?: (pid: number, timeoutMs: number) => Promise<number | null>
  } = {}
): Promise<ProcessIdentity[]> {
  const survivors: ProcessIdentity[] = []

  const absoluteDeadline =
    typeof deadlineAt === 'number' && Number.isFinite(deadlineAt) ? deadlineAt : Date.now() + 2_000

  const remaining = () => Math.max(0, absoluteDeadline - Date.now())
  const runner = livenessRunner ?? defaultLivenessProbeRunner
  const createdAtReader = readCreatedAt ?? readProcessCreatedAt

  for (const identity of identities) {
    const left = remaining()

    if (left <= 0) {
      // No time for an absence probe: keep known/unknown identities as survivors.
      survivors.push(identity)

      continue
    }

    const slice = Math.max(1, Math.floor(left / Math.max(1, identities.length - survivors.length)))

    const createdAtResult = await resolveBeforeDeadline(
      () => createdAtReader(identity.pid, slice),
      absoluteDeadline
    )

    if (!createdAtResult.completed) {
      survivors.push(identity)

      continue
    }

    const createdAt = createdAtResult.value

    if (createdAt == null) {
      // Do not infer absence from a null create-time read. Probe liveness with
      // remaining budget; only explicit not-found proves absence.
      const liveLeft = remaining()

      if (liveLeft <= 0) {
        survivors.push(identity)

        continue
      }

      const livenessResult = await resolveBeforeDeadline(
        () => probeProcessLiveness(identity.pid, liveLeft, runner),
        absoluteDeadline
      )

      if (!livenessResult.completed || livenessResult.value !== 'absent') {
        // live, unknown, or an expired probe => survivor
        survivors.push(identity)
      }

      continue
    }

    // Unknown generation: any completed create-time read is still only
    // identity evidence; a matching live generation remains a survivor.
    if (identity.createdAt == null || !Number.isFinite(identity.createdAt)) {
      survivors.push({ pid: identity.pid, createdAt })

      continue
    }

    if (Math.abs(createdAt - identity.createdAt) <= 1.5) {
      survivors.push(identity)
    }
  }

  return survivors
}

/**
 * Snapshot a Windows process tree (root + descendants) as PID + create-time
 * identities so post-kill verification can detect PID reuse.
 */
export async function snapshotProcessTreeIdentities(
  rootPid: number,
  {
    timeoutMs = 1_500,
    deadlineAt
  }: { timeoutMs?: number; deadlineAt?: number } = {}
): Promise<ProcessIdentity[]> {
  if (!Number.isInteger(rootPid) || rootPid <= 0) {return []}
  const requestedBudget = Math.max(0, Math.trunc(timeoutMs))

  const remainingBudget =
    typeof deadlineAt === 'number' && Number.isFinite(deadlineAt)
      ? Math.max(0, Math.trunc(deadlineAt - Date.now()))
      : requestedBudget

  const budget = Math.min(requestedBudget, remainingBudget)

  // A zero/negative caller budget is a hard no-probe condition on every
  // platform. In particular, do not fall through to a fresh default-timeout
  // probe after the shared deadline is exhausted.
  if (budget <= 0) {
    return [{ pid: rootPid }]
  }

  if (process.platform !== 'win32') {
    const createdAt = await readProcessCreatedAt(rootPid, budget)

    return createdAt == null ? [] : [{ pid: rootPid, createdAt }]
  }

  const startedAt = Date.now()

  try {
    const { stdout } = await execFileAsync(
      windowsPowerShellExecutable(),
      [
        '-NoLogo',
        '-NoProfile',
        '-NonInteractive',
        '-Command',
        `
$ErrorActionPreference = 'Stop'
$root = ${Math.trunc(rootPid)}
$seen = @{}
$outputRows = New-Object 'System.Collections.Generic.List[string]'
function Get-RowCreationUnix($row) {
  $raw = $row.CreationDate
  if ($null -eq $raw -or [string]::IsNullOrWhiteSpace([string]$raw)) { throw 'TREE_SNAPSHOT_MISSING_CREATE_TIME' }
  if ($raw -is [DateTime]) {
    return [DateTimeOffset]::new(([DateTime]$raw).ToUniversalTime()).ToUnixTimeSeconds()
  }
  return [DateTimeOffset]::new(
    [System.Management.ManagementDateTimeConverter]::ToDateTime([string]$raw).ToUniversalTime()
  ).ToUnixTimeSeconds()
}
function Add-Tree($row, [double]$parentCreated, [bool]$hasParent) {
  if ($null -eq $row) { return }
  $pidVal = [int]$row.ProcessId
  if ($pidVal -le 0 -or $seen.ContainsKey([string]$pidVal)) { return }
  $created = Get-RowCreationUnix $row
  # ParentProcessId alone is not generation-safe after PID reuse. Ignore an
  # older process that merely retains the reused parent PID.
  if ($hasParent -and (($created + 1.5) -lt $parentCreated)) { return }
  $seen[[string]$pidVal] = $true
  [void]$outputRows.Add(("{0}|{1}" -f $pidVal, $created))
  foreach ($child in @(Get-CimInstance Win32_Process -Filter ("ParentProcessId = $pidVal") -ErrorAction Stop)) {
    Add-Tree $child $created $true
  }
}
$rootRow = Get-CimInstance Win32_Process -Filter ("ProcessId = $root") -ErrorAction Stop | Select-Object -First 1
Add-Tree $rootRow 0.0 $false
$outputRows -join ';'
`.trim()
      ],
      { encoding: 'utf8', windowsHide: true, timeout: budget }
    )

    const identities: ProcessIdentity[] = []

    for (const part of String(stdout || '')
      .trim()
      .split(';')
      .filter(Boolean)) {
      const [pidText, createdText] = part.split('|')
      const pid = Number(pidText)
      const createdAt = Number(createdText)

      if (Number.isInteger(pid) && pid > 0 && Number.isFinite(createdAt) && createdAt > 0) {
        identities.push({ pid, createdAt })
      }
    }

    return identities
  } catch {
    // If the tree query failed, spend only the caller's remaining slice on an
    // exact root-generation read; never fall back to readProcessCreatedAt's
    // independent two-second default and overrun the absolute deadline.
    const elapsed = Date.now() - startedAt
    const fallbackBudget = Math.max(0, Math.min(budget, elapsed >= budget ? 0 : budget - elapsed))

    if (fallbackBudget <= 0) {return [{ pid: rootPid }]}
    const createdAt = await readProcessCreatedAt(rootPid, fallbackBudget)

    return createdAt == null ? [{ pid: rootPid }] : [{ pid: rootPid, createdAt }]
  }
}

async function runTaskkillWithinDeadline(
  pid: number,
  deadlineAt: number
): Promise<{ completed: boolean; succeeded: boolean }> {
  const remaining = Math.max(0, deadlineAt - Date.now())

  if (remaining <= 0) {return { completed: false, succeeded: false }}

  return await new Promise(resolve => {
    let settled = false
    let timer: NodeJS.Timeout | undefined

    const finish = (result: { completed: boolean; succeeded: boolean }) => {
      if (settled) {return}
      settled = true

      if (timer) {clearTimeout(timer)}
      resolve(result)
    }

    const child = execFile(
      // Absolute path. PATH is attacker-influenced, and during an update the
      // venv's own Scripts directory is prepended to it and rewritten in flight.
      windowsSystem32Executable('taskkill.exe'),
      ['/PID', String(pid), '/T', '/F'],
      {
        windowsHide: true,
        timeout: Math.max(1, remaining - 1),
        killSignal: 'SIGKILL'
      },
      (error: any) => finish({ completed: true, succeeded: !error })
    )

    child.once('error', () => finish({ completed: true, succeeded: false }))
    timer = setTimeout(() => {
      // This command only supervises the already-contained mutation child.
      // If taskkill itself cannot settle in time, stop waiting and fail closed;
      // never report a confirmed tree after an unbounded native command.
      try {
        child.kill('SIGKILL')
      } catch {
        void 0
      }

      finish({ completed: false, succeeded: false })
    }, Math.max(1, remaining - 1))
  })
}

/**
 * Kill a Windows process tree and wait until every pre-captured identity is gone
 * or reused. Uses one absolute deadline shared by taskkill + identity polling.
 * Returns confirmed=false when any identity remains — callers must treat that as
 * a hard terminal-boundary failure, not a settled success.
 */
export async function killProcessTreeAndAwaitGone(
  pid: number,
  {
    confirmMs = TERMINATE_KILL_CONFIRM_MS,
    pollMs = 50,
    preSnapshot,
    deadlineAt,
    snapshotProcessTree,
    readCreatedAt
  }: {
    confirmMs?: number
    pollMs?: number
    preSnapshot?: readonly ProcessIdentity[]
    /** Absolute Date.now() deadline; overrides confirmMs window when provided. */
    deadlineAt?: number
    /** Injectable only to force snapshot failure in the real boundary canary. */
    snapshotProcessTree?: typeof snapshotProcessTreeIdentities
    /** Injectable create-time reader for deadline regressions. */
    readCreatedAt?: (pid: number, timeoutMs: number) => Promise<number | null>
  } = {}
): Promise<{ confirmed: boolean; identities: ProcessIdentity[]; survivors: ProcessIdentity[] }> {
  if (!Number.isInteger(pid) || pid <= 0) {
    return { confirmed: true, identities: [], survivors: [] }
  }

  const absoluteDeadline =
    typeof deadlineAt === 'number' && Number.isFinite(deadlineAt)
      ? deadlineAt
      : Date.now() + Math.max(0, Math.trunc(confirmMs))

  const remaining = () => Math.max(0, absoluteDeadline - Date.now())

  if (remaining() <= 0) {
    // No time for probes: fail closed with known/unknown identities as survivors.
    const identities =
      preSnapshot && preSnapshot.length > 0 ? [...preSnapshot] : [{ pid }]

    return { confirmed: false, identities, survivors: identities }
  }

  const snapshot = snapshotProcessTree ?? snapshotProcessTreeIdentities
  let captured: ProcessIdentity[] = []
  let snapshotTimedOut = false

  if (preSnapshot && preSnapshot.length > 0) {
    captured = [...preSnapshot]
  } else {
    const snapshotBudget = remaining()

    if (snapshotBudget <= 0) {
      return { confirmed: false, identities: [{ pid }], survivors: [{ pid }] }
    }

    const snapshotPromise = Promise.resolve().then(() =>
      snapshot(pid, {
        timeoutMs: Math.min(500, snapshotBudget),
        deadlineAt: absoluteDeadline
      })
    )

    let snapshotTimer: NodeJS.Timeout | undefined

    try {
      // Native execFile has its own timeout, but keep the boundary safe even
      // if an injected/native adapter ignores that option. Snapshotting is
      // read-only, so abandoning its late result cannot leave mutation work.
      captured = await Promise.race([
        snapshotPromise,
        new Promise<ProcessIdentity[]>(resolve => {
          snapshotTimer = setTimeout(() => {
            snapshotTimedOut = true
            resolve([])
          }, Math.max(1, snapshotBudget - 1))
        })
      ])
    } catch {
      // Snapshot failure is intentionally represented by the unknown root.
      // The caller's Job Object boundary, when present, is the hard fallback.
      captured = []
    } finally {
      if (snapshotTimer) {clearTimeout(snapshotTimer)}
    }

    // A timed-out adapter may settle later; consume its rejection without
    // extending this boundary or creating an unhandled-rejection side effect.
    void snapshotPromise.catch(() => undefined)

    if (snapshotTimedOut) {
      // The shared deadline was consumed by snapshotting. Do not launch a
      // second native probe for a fabricated/fresh root generation.
      captured = []
    }
  }

  const identities = captured
  const createdAtReader = readCreatedAt ?? readProcessCreatedAt

  // Always include the root identity. Prefer a real create-time; if unavailable,
  // keep an unknown-generation identity (createdAt omitted) so liveness alone
  // keeps confirmation false while the PID lives.
  if (!identities.some(entry => entry.pid === pid)) {
    const left = remaining()
    let createdAt: number | null = null

    if (!snapshotTimedOut && left > 0) {
      const rootResult = await resolveBeforeDeadline(
        () => createdAtReader(pid, left),
        absoluteDeadline
      )

      if (rootResult.completed) {createdAt = rootResult.value}
    }

    identities.push(createdAt != null ? { pid, createdAt } : { pid })
  }

  const taskkillBudget = Math.min(remaining(), Math.max(0, Math.trunc(confirmMs)))

  if (taskkillBudget > 0) {
    const taskkillResult = await runTaskkillWithinDeadline(pid, absoluteDeadline)

    if (!taskkillResult.completed) {
      return { confirmed: false, identities, survivors: identities }
    }

    if (!taskkillResult.succeeded) {
      try {
        process.kill(pid, 'SIGKILL')
      } catch {
        void 0
      }
    }
  }

  // Also hard-kill every known identity in case /T missed a detached Start-Process child.
  for (const identity of identities) {
    const left = remaining()

    if (left <= 0) {break}

    if (identity.pid === pid) {continue}
    const taskkillResult = await runTaskkillWithinDeadline(identity.pid, absoluteDeadline)

    if (!taskkillResult.completed) {
      return { confirmed: false, identities, survivors: identities }
    }

    if (!taskkillResult.succeeded) {
      try {
        process.kill(identity.pid, 'SIGKILL')
      } catch {
        void 0
      }
    }
  }

  if (remaining() <= 0) {
    return { confirmed: false, identities, survivors: identities }
  }

  let survivors = await identitiesStillPresent(identities, { deadlineAt: absoluteDeadline })

  while (survivors.length > 0 && remaining() > 0) {
    await new Promise(resolve => setTimeout(resolve, Math.max(1, Math.min(pollMs, remaining()))))

    if (remaining() <= 0) {break}
    survivors = await identitiesStillPresent(identities, { deadlineAt: absoluteDeadline })
  }

  if (remaining() <= 0 && survivors.length > 0) {
    return { confirmed: false, identities, survivors }
  }

  return {
    confirmed: survivors.length === 0,
    identities,
    survivors
  }
}

function waitForChildExit(child: ChildProcess, timeoutMs: number): Promise<boolean> {
  if (child.exitCode != null || child.signalCode != null) {
    return Promise.resolve(true)
  }

  return new Promise(resolve => {
    let settled = false

    const done = (ok: boolean) => {
      if (settled) {return}
      settled = true
      resolve(ok)
    }

    const timer = setTimeout(() => done(false), Math.max(1, timeoutMs))
    child.once('exit', () => {
      clearTimeout(timer)
      done(true)
    })
    child.once('error', () => {
      clearTimeout(timer)
      done(false)
    })
  })
}

async function terminateNamedTargetJobWithinDeadline(
  targetJobName: string,
  deadlineAt: number,
  maxBudgetMs = 1_000
): Promise<boolean> {
  if (process.platform !== 'win32' || !targetJobName) {return false}
  const remaining = Math.max(0, deadlineAt - Date.now())
  const budget = Math.min(Math.max(0, Math.trunc(maxBudgetMs)), remaining)

  if (budget <= 0) {return false}

  try {
    await execFileAsync(
      windowsPowerShellExecutable(),
      [
        '-NoLogo',
        '-NoProfile',
        '-NonInteractive',
        '-ExecutionPolicy',
        'Bypass',
        '-Command',
        TERMINATE_NAMED_JOB_COMMAND
      ],
      {
        windowsHide: true,
        timeout: Math.max(1, budget - 1),
        env: {
          ...process.env,
          HERMES_TERMINATE_TARGET_JOB_NAME: targetJobName,
          HERMES_TERMINATE_TARGET_WAIT_MS: String(Math.max(0, budget - 100))
        }
      }
    )

    return true
  } catch {
    return false
  }
}

function startTargetJobWatcher(
  ownerPid: number,
  ownerCreatedAt: number,
  targetJobName: string,
  watcherReadyPath: string,
  watcherReadyNonce: string,
  deadlineAt: number
): ReturnType<StartTargetJobWatcher> {
  if (process.platform !== 'win32' || !Number.isInteger(ownerPid) || ownerPid <= 0) {return undefined}
  const encodedWatcherCommand = Buffer.from(TERMINATE_JOB_WATCHER_BOOTSTRAP, 'utf16le').toString('base64')

  try {
    const watcher = spawn(
      process.execPath,
      ['-e', TERMINATE_JOB_WATCHER_BRIDGE],
      {
        windowsHide: true,
        detached: true,
        stdio: 'ignore',
        env: {
          ...process.env,
          ELECTRON_RUN_AS_NODE: '1',
          HERMES_TERMINATE_WATCHER_POWERSHELL: windowsPowerShellExecutable(),
          HERMES_TERMINATE_WATCHER_ENCODED_COMMAND: encodedWatcherCommand,
          HERMES_TERMINATE_OWNER_PID: String(ownerPid),
          HERMES_TERMINATE_OWNER_CREATED_AT: String(ownerCreatedAt),
          HERMES_TERMINATE_TARGET_JOB_NAME: targetJobName,
          HERMES_TERMINATE_WATCHER_READY_PATH: watcherReadyPath,
          HERMES_TERMINATE_WATCHER_READY_NONCE: watcherReadyNonce,
          // The watcher body was originally authored for direct Windows argv
          // transport, where escaped C# quotes are consumed before PowerShell
          // parses the here-string. ScriptBlock/environment transport preserves
          // those backslashes, so normalize only escaped double quotes here.
          HERMES_TERMINATE_WATCHER_SCRIPT: TERMINATE_JOB_WATCHER_COMMAND.replace(/\\"/g, '"'),
          HERMES_TERMINATE_WATCHER_DEADLINE_AT: String(Math.trunc(deadlineAt))
        }
      }
    )

    return watcher
  } catch {
    return undefined
  }
}

function boundedWatcherErrorMessage(message: unknown): string {
  return String(message ?? '')
    .replace(/[\r\n]+/g, ' ')
    .slice(0, 256)
}

function observeTargetJobWatcher(child: ChildProcess, spawnedAt: number): TargetJobWatcherDiagnostics {
  const diagnostics: TargetJobWatcherDiagnostics = {
    spawnedAt,
    pid: child.pid
  }

  child.on('error', error => {
    if (diagnostics.errorAt == null) {diagnostics.errorAt = Date.now()}

    if (diagnostics.errorCode == null) {
      const errorCode = (error as NodeJS.ErrnoException).code

      if (typeof errorCode === 'string' || typeof errorCode === 'number') {diagnostics.errorCode = errorCode}
    }

    if (!diagnostics.errorMessage) {diagnostics.errorMessage = boundedWatcherErrorMessage(error?.message)}
  })
  child.on('exit', (code, signal) => {
    if (diagnostics.exitAt == null) {diagnostics.exitAt = Date.now()}
    diagnostics.exitCode = code
    diagnostics.signalCode = signal
  })

  if (child.exitCode != null || child.signalCode != null) {
    diagnostics.exitAt = Date.now()
    diagnostics.exitCode = child.exitCode
    diagnostics.signalCode = child.signalCode
  }

  return diagnostics
}

function formatTargetJobWatcherDiagnostics(diagnostics?: TargetJobWatcherDiagnostics): string {
  if (!diagnostics) {return 'watcher-diagnostics=none'}
  const now = Date.now()
  const errorElapsedMs = diagnostics.errorAt == null ? 'none' : Math.max(0, diagnostics.errorAt - diagnostics.spawnedAt)
  const exitElapsedMs = diagnostics.exitAt == null ? 'none' : Math.max(0, diagnostics.exitAt - diagnostics.spawnedAt)

  const elapsedMs = diagnostics.exitAt != null
    ? exitElapsedMs
    : diagnostics.errorAt != null
      ? errorElapsedMs
      : Math.max(0, now - diagnostics.spawnedAt)

  return [
    `watcher-spawned-at=${diagnostics.spawnedAt}`,
    `watcher-pid=${diagnostics.pid ?? 'none'}`,
    `watcher-error-code=${diagnostics.errorCode ?? 'none'}`,
    `watcher-error-message=${JSON.stringify(diagnostics.errorMessage ?? '')}`,
    `watcher-exit-code=${diagnostics.exitCode ?? 'none'}`,
    `watcher-signal=${diagnostics.signalCode ?? 'none'}`,
    `watcher-error-elapsed-ms=${errorElapsedMs}`,
    `watcher-exit-elapsed-ms=${exitElapsedMs}`,
    `watcher-elapsed-ms=${elapsedMs}`
  ].join(' ')
}

function sanitizeBoundaryDiagnostics(value: unknown): string {
  const safeLines: string[] = []

  for (const rawLine of String(value ?? '').split(/\r?\n/)) {
    const line = rawLine.trim()

    if (!line || /Command failed:/i.test(line)) {continue}

    if (/watcher-(?:spawned|exited|stalled|error|exit|signal|elapsed)/i.test(line)) {
      safeLines.push(line.slice(0, 1_200))

      continue
    }

    if (/^(?:TARGET_[A-Z0-9_ -]+|unconfirmed-tree-survivors:[0-9,]+|exit -?\d+|timeout|killed|aborted|deadline-exhausted)\b/i.test(line)) {
      safeLines.push(line.slice(0, 512))
    }
  }

  return [...new Set(safeLines)].slice(0, 8).join('\n')
}

function deadlineFailureDetail(stderr: unknown): string {
  const diagnostics = sanitizeBoundaryDiagnostics(stderr)

  return diagnostics ? `deadline-exhausted ${diagnostics}` : 'deadline-exhausted'
}

function sanitizeRunnerStderr(value: unknown): string {
  const stderr = String(value ?? '').replace(/\0/g, '')

  if (/Command failed:/i.test(stderr)) {return sanitizeBoundaryDiagnostics(stderr)}

  return stderr.slice(0, 4_096)
}

const TARGET_JOB_WATCHER_GRACE_MS = 400

type TargetJobWatcherResult = {
  terminal: boolean
  healthy: boolean
  targetBoundaryConfirmed: boolean
  detail: string
}

/**
 * Wait for the detached target-job watcher without allowing it to become an
 * unobserved mutator. A watcher that exits unsuccessfully or misses its short
 * settle window is killed and the named target job is terminated within the
 * same absolute deadline. The caller must not remove watcher artifacts until
 * this function reports terminal=true.
 */
async function waitForTargetJobWatcher(
  watcher: ChildProcess | undefined,
  targetJobName: string,
  deadlineAt: number,
  startupFailure?: string,
  diagnostics?: TargetJobWatcherDiagnostics,
  skipGrace = false
): Promise<TargetJobWatcherResult> {
  if (!watcher || !Number.isInteger(watcher.pid) || (watcher.pid as number) <= 0) {
    const remaining = Math.max(0, deadlineAt - Date.now())

    const targetJobAttempt =
      remaining > 0
        ? await terminateNamedTargetJobWithinDeadline(targetJobName, deadlineAt, Math.min(1_000, remaining))
        : false

    return {
      terminal: true,
      healthy: false,
      targetBoundaryConfirmed: targetJobAttempt,
      detail: `${startupFailure ?? 'watcher-missing'} ${formatTargetJobWatcherDiagnostics(diagnostics)} target-job-termination=${targetJobAttempt ? 'confirmed' : 'not-confirmed'}`
    }
  }

  const watcherPid = watcher.pid
  const settleBudget = skipGrace ? 0 : Math.min(TARGET_JOB_WATCHER_GRACE_MS, Math.max(0, deadlineAt - Date.now()))

  if (settleBudget > 0 && (await waitForChildExit(watcher, settleBudget))) {
    if (watcher.exitCode === 0 && watcher.signalCode == null) {
      return {
        terminal: true,
        healthy: true,
        targetBoundaryConfirmed: true,
        detail: `watcher-exited-cleanly ${formatTargetJobWatcherDiagnostics(diagnostics)}`
      }
    }

    const remaining = Math.max(0, deadlineAt - Date.now())

    const targetJobAttempt =
      remaining > 0
        ? await terminateNamedTargetJobWithinDeadline(targetJobName, deadlineAt, Math.min(1_000, remaining))
        : false

    return {
      terminal: true,
      healthy: false,
      targetBoundaryConfirmed: targetJobAttempt,
      detail: `watcher-exited-with-failure code=${String(watcher.exitCode)} signal=${String(watcher.signalCode)} ${formatTargetJobWatcherDiagnostics(diagnostics)} target-job-termination=${targetJobAttempt ? 'confirmed' : 'not-confirmed'}`
    }
  }

  // The watcher is stalled. Kill its exact process tree and close the named
  // target job in parallel, then observe the watcher exit before returning.
  const remaining = Math.max(0, deadlineAt - Date.now())

  const watcherTreeSnapshot =
    remaining > 0
      ? await snapshotProcessTreeIdentities(watcherPid, {
          timeoutMs: Math.min(300, remaining),
          deadlineAt
        })
      : []

  const targetJobTermination =
    remaining > 0
      ? terminateNamedTargetJobWithinDeadline(targetJobName, deadlineAt, Math.min(1_000, remaining))
      : Promise.resolve(false)

  try {
    watcher.kill('SIGKILL')
  } catch {
    void 0
  }

  const watcherTreeTermination = killProcessTreeAndAwaitGone(watcherPid, {
    confirmMs: Math.min(500, remaining),
    preSnapshot: watcherTreeSnapshot.length > 0 ? watcherTreeSnapshot : [{ pid: watcherPid }],
    deadlineAt
  })

  const [targetJobAttempt, watcherTreeResult] = await Promise.all([targetJobTermination, watcherTreeTermination])
  const remainingAfterKill = Math.max(0, deadlineAt - Date.now())

  const watcherExited =
    remainingAfterKill > 0
      ? await waitForChildExit(watcher, remainingAfterKill)
      : watcher.exitCode != null || watcher.signalCode != null

  return {
    // Native exact-generation absence is terminal proof even when Node has not
    // delivered the child `exit` event yet (for example while inherited stdio
    // handles finish closing).
    terminal: watcherExited || watcherTreeResult.confirmed,
    healthy: false,
    targetBoundaryConfirmed: targetJobAttempt,
    detail: `watcher-stalled tree-confirmed=${watcherTreeResult.confirmed} ${formatTargetJobWatcherDiagnostics(diagnostics)} target-job-termination=${targetJobAttempt ? 'confirmed' : 'not-confirmed'}`
  }
}

/**
 * Run PowerShell under a hard budget. On abort/timeout the child tree is killed
 * and the promise resolves only after every pre-captured tree identity is
 * confirmed gone. Confirmation failure is a hard boundary failure.
 */
async function defaultRunPowerShell(
  script: string,
  timeoutMs = 4_000,
  signal?: AbortSignal,
  deadlineAt?: number,
  dependencies?: HardBoundaryDependencies
): Promise<{ stdout: string; stderr: string; code: number; pid?: number }> {
  const requestedBudget = Math.max(0, Math.trunc(timeoutMs))
  const startedAt = Date.now()
  // Leave a small scheduler/IPC margin so the public return stays inside the
  // caller's hard wall-clock budget even when native callbacks settle late.
  const deadlineSafetyMarginMs = Math.min(100, Math.floor(requestedBudget / 10))

  const requestedDeadline =
    typeof deadlineAt === 'number' && Number.isFinite(deadlineAt)
      ? Math.min(Math.trunc(deadlineAt), startedAt + requestedBudget)
      : startedAt + requestedBudget

  const absoluteDeadline = Math.max(startedAt, requestedDeadline - deadlineSafetyMarginMs)
  const budget = Math.max(0, absoluteDeadline - startedAt)

  if (budget <= 0) {
    return { stdout: '', stderr: 'aborted', code: 1 }
  }

  if (signal?.aborted) {
    return { stdout: '', stderr: 'aborted', code: 1 }
  }

  // Reserve part of the budget for kill + confirmed absence of the whole tree.
  const killReserveMs = terminateKillReserveMs(budget)
  const wrapperDeadline = absoluteDeadline - killReserveMs
  // Keep a bounded outer tail for one final PID-not-found proof when Node's exit
  // event lags behind a successfully delivered watcher kill.
  const watcherSettleReserveMs = Math.min(250, Math.floor(killReserveMs / 4))
  const watcherDeadline = absoluteDeadline - watcherSettleReserveMs
  const runMs = Math.max(0, wrapperDeadline - startedAt)

  if (runMs <= 0) {
    return { stdout: '', stderr: 'deadline-exhausted', code: 1 }
  }

  const jobName = `HermesTerminateHelper-${randomBytes(16).toString('hex')}`
  const targetJobName = `HermesTerminateTarget-${randomBytes(16).toString('hex')}`
  const targetWaitMs = Math.max(0, Math.min(1_500, runMs - 100))
  // Production relies on the wrapper-owned kill-on-close target/helper Jobs.
  // A watcher can be injected only by boundary tests; it is never an ambient
  // or production lifecycle dependency.
  const watcherStarter = dependencies?.startWatcher
  const watcherEnabled = typeof watcherStarter === 'function'
  const watcherReadyDirectory = fs.mkdtempSync(path.join(os.tmpdir(), 'hermes-terminate-watcher-'))
  const watcherReadyPath = path.join(watcherReadyDirectory, 'ready')
  const watcherReadyNonce = randomBytes(16).toString('hex')
  const wrapperPidMarkerPath = path.join(watcherReadyDirectory, 'wrapper.pid')
  const wrapperPidMarkerNonce = randomBytes(16).toString('hex')
  const wrapperPhasePath = path.join(watcherReadyDirectory, 'wrapper.phase')
  const wrapperPhaseNonce = randomBytes(16).toString('hex')
  const helperScriptPath = path.join(watcherReadyDirectory, 'helper.ps1')
  const helperGatePath = path.join(watcherReadyDirectory, 'helper.go')

  return await new Promise(resolve => {
    let settled = false
    let childPid: number | undefined
    let treeSnapshot: ProcessIdentity[] = []
    let killing = false
    let terminalizing: Promise<void> | null = null
    let childProcess: ChildProcess | undefined
    let wrapperProcessIdentity: ProcessIdentity | undefined
    let wrapperMarkerSetup: Promise<void> | undefined
    let watcherProcess: ChildProcess | undefined
    let watcherDiagnostics: TargetJobWatcherDiagnostics | undefined
    let watcherStartupFailure: string | undefined

    const cleanupWatcherArtifacts = () => {
      // Every writer path is parent-known and nonce-directory-bound. Never
      // recursively remove the directory, because unrelated files must not be swept.
      const exactPaths = [
        watcherReadyPath,
        `${watcherReadyPath}.tmp`,
        wrapperPidMarkerPath,
        `${wrapperPidMarkerPath}.tmp`,
        helperScriptPath,
        helperGatePath,
        ...WRAPPER_PHASE_NAMES.flatMap(phase => [
          `${wrapperPhasePath}.${phase}.marker`,
          `${wrapperPhasePath}.${phase}.marker.tmp`
        ])
      ].filter(
        (entry): entry is string => typeof entry === 'string'
      )

      for (const exactPath of exactPaths) {
        try {
          fs.rmSync(exactPath, { force: true })
        } catch {
          void 0
        }
      }

      try {
        fs.rmdirSync(watcherReadyDirectory)
      } catch {
        // A non-empty directory is intentionally retained rather than removed
        // recursively. The caller still receives the terminal failure.
        void 0
      }
    }

    const finish = async (result: { stdout: string; stderr: string; code: number; pid?: number }) => {
      if (settled || terminalizing) {return}
      terminalizing = (async () => {
        let finalResult = { ...result, pid: childPid }
        const mustKill = killing || signal?.aborted === true

        if (wrapperMarkerSetup) {await wrapperMarkerSetup}

        const watcherReadyAtFinish = watcherEnabled ? readExactWatcherReadyValue(watcherReadyPath) : undefined

        if (
          watcherEnabled &&
          watcherReadyAtFinish !== `ARMED:${watcherReadyNonce}` &&
          watcherProcess?.exitCode == null &&
          watcherProcess?.signalCode == null
        ) {
          try {
            watcherProcess?.kill('SIGKILL')
          } catch {
            void 0
          }
        }

        const watcherResult: TargetJobWatcherResult = watcherEnabled
          ? await waitForTargetJobWatcher(
              watcherProcess,
              targetJobName,
              absoluteDeadline,
              watcherStartupFailure,
              watcherDiagnostics,
              watcherReadyAtFinish !== `ARMED:${watcherReadyNonce}`
            )
          : {
              terminal: true,
              healthy: true,
              targetBoundaryConfirmed: hasTargetBoundaryReceipt(wrapperPhasePath, wrapperPhaseNonce),
              detail: 'wrapper-owned-job-boundary'
            }

        let targetBoundaryConfirmed = watcherResult.targetBoundaryConfirmed

        // On cancellation, drain the named target Job while the wrapper still
        // owns its persistent handle. Only then kill and confirm the wrapper /
        // helper process tree. This removes the close-before-open race.
        if (!watcherEnabled && mustKill && !targetBoundaryConfirmed && Date.now() < absoluteDeadline) {
          targetBoundaryConfirmed = await terminateNamedTargetJobWithinDeadline(
            targetJobName,
            absoluteDeadline,
            Math.min(1_000, Math.max(0, absoluteDeadline - Date.now()))
          )
        }

        let killResult =
          mustKill && typeof childPid === 'number'
            ? await killProcessTreeAndAwaitGone(childPid, {
                confirmMs: killReserveMs,
                preSnapshot: treeSnapshot,
                deadlineAt: absoluteDeadline
              })
            : undefined

        let watcherTerminal = watcherResult.terminal

        if (
          watcherEnabled &&
          !watcherTerminal &&
          Number.isInteger(watcherProcess?.pid) &&
          (watcherProcess?.pid as number) > 0 &&
          Date.now() < absoluteDeadline
        ) {
          const watcherSurvivors = await identitiesStillPresent([{ pid: watcherProcess?.pid as number }], {
            deadlineAt: absoluteDeadline
          })

          watcherTerminal = watcherSurvivors.length === 0
        }

        if (watcherEnabled && !watcherTerminal && watcherProcess) {
          watcherTerminal = watcherProcess.exitCode != null || watcherProcess.signalCode != null
        }

        let wrapperBoundaryRequired = mustKill

        // Only the injected legacy watcher can authenticate wrapper absence.
        // Production requires the normal wrapper-tree liveness proof.
        if (watcherEnabled && watcherResult.healthy && killResult && wrapperProcessIdentity) {
          const survivors = killResult.survivors.filter(identity => {
            if (identity.pid !== wrapperProcessIdentity?.pid) {return true}

            if (identity.createdAt == null || wrapperProcessIdentity.createdAt == null) {return false}

            return Math.abs(identity.createdAt - wrapperProcessIdentity.createdAt) > 1.5
          })

          killResult = { ...killResult, survivors, confirmed: survivors.length === 0 }
        }

        // execFile completion can lag exact process death. Accept only the
        // nonce-bound success receipt written after target drain and handle
        // closure; watcher READY is required only for an explicitly injected
        // watcher test path.
        if (
          finalResult.code !== 0 &&
          targetBoundaryConfirmed &&
          (!watcherEnabled || watcherReadyAtFinish === `ARMED:${watcherReadyNonce}`) &&
          hasSuccessfulWrapperReceipt(wrapperPhasePath, wrapperPhaseNonce)
        ) {
          finalResult = { stdout: 'TERMINATED\n', stderr: '', code: 0, pid: childPid }
        }

        if (!watcherResult.healthy || !targetBoundaryConfirmed) {
          wrapperBoundaryRequired = true
          killing = true
          const remaining = Math.max(0, absoluteDeadline - Date.now())

          if (!targetBoundaryConfirmed && remaining > 0) {
            targetBoundaryConfirmed = await terminateNamedTargetJobWithinDeadline(
              targetJobName,
              absoluteDeadline,
              Math.min(1_000, remaining)
            )
          }

          if (typeof childPid === 'number' && (!killResult || !killResult.confirmed)) {
            killResult = await killProcessTreeAndAwaitGone(childPid, {
              confirmMs: Math.min(killReserveMs, Math.max(0, absoluteDeadline - Date.now())),
              preSnapshot: treeSnapshot,
              deadlineAt: absoluteDeadline
            })
          }
        }

        if (watcherEnabled && !watcherResult.healthy) {
          finalResult = {
            ...finalResult,
            stderr: [watcherStartupFailure, watcherResult.detail, finalResult.stderr]
              .filter(Boolean)
              .join('\n'),
            code: 1,
            pid: childPid
          }
        }

        if (watcherEnabled && watcherDiagnostics && finalResult.code !== 0) {
          finalResult = {
            ...finalResult,
            stderr: [formatTargetJobWatcherDiagnostics(watcherDiagnostics), finalResult.stderr]
              .filter(Boolean)
              .join('\n'),
            pid: childPid
          }
        }

        if (!targetBoundaryConfirmed) {
          finalResult = {
            ...finalResult,
            stderr: [finalResult.stderr, 'target-boundary-unconfirmed'].filter(Boolean).join('\n'),
            code: 1,
            pid: childPid
          }
        }

        // Node's exit status is evidence for the exact child it spawned, even
        // when a fresh PowerShell liveness probe exhausts the remaining budget.
        // The target receipt separately proves the external holder job drained.
        if (
          killResult && !killResult.confirmed && targetBoundaryConfirmed &&
          childProcess && childProcess.pid === wrapperProcessIdentity?.pid &&
          (childProcess.exitCode != null || childProcess.signalCode != null) &&
          hasSuccessfulWrapperReceipt(wrapperPhasePath, wrapperPhaseNonce)
        ) {
          const survivors = killResult.survivors.filter(identity => identity.pid !== childProcess.pid)
          killResult = { ...killResult, survivors, confirmed: survivors.length === 0 }
        }

        if (killResult && !killResult.confirmed) {
          finalResult = {
            ...finalResult,
            stderr: [
              finalResult.stderr,
              `unconfirmed-tree-survivors:${killResult.survivors.map(s => s.pid).join(',')}`
            ]
              .filter(Boolean)
              .join('\n'),
            code: 1,
            pid: childPid
          }
        }

        if (!watcherTerminal) {
          finalResult = {
            ...finalResult,
            stderr: [finalResult.stderr, 'watcher-terminal-state-unconfirmed'].filter(Boolean).join('\n'),
            code: 1,
            pid: childPid
          }
        }

        // Native confirmation can observe the process object gone before Node
        // delivers its exit event. Drain that event under the same deadline.
        if (wrapperBoundaryRequired && childProcess && absoluteDeadline > Date.now()) {
          await waitForChildExit(childProcess, absoluteDeadline - Date.now())
        }

        // Capture the exact authenticated READY value and the ordered,
        // nonce-validated wrapper phases before any artifact cleanup. These
        // bounded diagnostics identify which side of the boundary stalled
        // without making cleanup depend on a wildcard directory scan.
        const wrapperPhaseDiagnostics = readWrapperPhaseDiagnostics(wrapperPhasePath, wrapperPhaseNonce)

        if (finalResult.code !== 0) {
          finalResult = {
            ...finalResult,
            stderr: [
              watcherEnabled
                ? watcherReadyAtFinish
                  ? 'watcher-ready=armed'
                  : 'watcher-ready=missing'
                : 'boundary=wrapper-owned-job',
              wrapperPhaseDiagnostics,
              finalResult.stderr
            ]
              .filter(Boolean)
              .join('\n'),
            pid: childPid
          }
        }

        // Do not remove READY or its PID-qualified temp file while the watcher
        // can still publish FAILED or terminate the target job.
        if (watcherTerminal) {cleanupWatcherArtifacts()}
        settled = true
        signal?.removeEventListener('abort', onAbort)
        resolve(finalResult)
      })().catch(error => {
        // Preserve the exact artifacts if watcher terminalization itself failed;
        // recursive cleanup here could permit a late watcher mutation.
        settled = true
        signal?.removeEventListener('abort', onAbort)
        resolve({
          stdout: result.stdout,
          stderr: `termination-boundary-error:${String(error?.message ?? error)}`,
          code: 1,
          pid: childPid
        })
      })
    }

    childProcess = execFile(
      windowsPowerShellExecutable(),
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
        timeout: runMs,
        windowsHide: true,
        maxBuffer: 1024 * 1024,
        killSignal: 'SIGTERM',
        env: {
          ...process.env,
          HERMES_TERMINATE_SCRIPT: script,
          HERMES_TERMINATE_JOB_NAME: jobName,
          HERMES_TERMINATE_TARGET_JOB_NAME: targetJobName,
          HERMES_TERMINATE_TARGET_WAIT_MS: String(targetWaitMs),
          HERMES_TERMINATE_DEADLINE_AT: String(Math.trunc(wrapperDeadline)),
          HERMES_TERMINATE_WATCHER_READY_PATH: watcherReadyPath,
          HERMES_TERMINATE_WATCHER_READY_NONCE: watcherReadyNonce,
          HERMES_TERMINATE_WRAPPER_PID_MARKER_PATH: wrapperPidMarkerPath,
          HERMES_TERMINATE_WRAPPER_PID_MARKER_NONCE: wrapperPidMarkerNonce,
          HERMES_TERMINATE_WRAPPER_PHASE_PATH: wrapperPhasePath,
          HERMES_TERMINATE_WRAPPER_PHASE_NONCE: wrapperPhaseNonce,
          HERMES_TERMINATE_HELPER_SCRIPT_PATH: helperScriptPath,
          HERMES_TERMINATE_HELPER_GATE_PATH: helperGatePath
        }
      },
      (error: any, stdout, stderr) => {
        void (async () => {
          const errorStdout = String(stdout ?? error?.stdout ?? '')
          const capturedStderr = sanitizeRunnerStderr(stderr ?? error?.stderr ?? '')
          const errorMessage = String(error?.message ?? '')

          const lifecycleFact =
            error?.killed === true || /ETIMEDOUT|timeout|killed/i.test(errorMessage) ? 'timeout' : ''

          const errorStderr = [
            capturedStderr,
            typeof error?.code === 'number' ? `exit ${error.code}` : '',
            lifecycleFact
          ]
            .filter(Boolean)
            .join('\n')

          if (!error) {
            await finish({ stdout: String(stdout ?? ''), stderr: String(stderr ?? ''), code: 0 })

            return
          }

          if (
            typeof childPid === 'number' &&
            (error?.killed || /ETIMEDOUT|timeout/i.test(String(error?.message ?? '')))
          ) {
            killing = true
            await finish({
              stdout: errorStdout,
              stderr: errorStderr || 'timeout',
              code: typeof error?.code === 'number' ? error.code : 1
            })

            return
          }

          await finish({
            stdout: errorStdout,
            stderr: errorStderr,
            code: typeof error?.code === 'number' ? error.code : 1
          })
        })()
      }
    )

    if (typeof childProcess.pid === 'number' && childProcess.pid > 0) {
      childPid = childProcess.pid
      const launchedPid = childPid
      // The execFile PID is only a launch hint. The wrapper writes an exact,
      // nonce-bound self marker before opening the target job; authenticate that
      // generation before starting the detached watcher.
      wrapperMarkerSetup = (async () => {
        wrapperProcessIdentity = await waitForWrapperProcessMarker(
          wrapperPidMarkerPath,
          wrapperPidMarkerNonce,
          wrapperDeadline
        )

        if (!wrapperProcessIdentity) {
          watcherStartupFailure = `wrapper-marker-unavailable launch-pid=${launchedPid}`
          killing = true
          treeSnapshot = [{ pid: launchedPid }]

          try {
            childProcess?.kill('SIGKILL')
          } catch {
            void 0
          }

          return
        }

        if (watcherStarter) {
          const watcherSpawnedAt = Date.now()
          watcherProcess = watcherStarter(
            wrapperProcessIdentity.pid,
            wrapperProcessIdentity.createdAt ?? 0,
            targetJobName,
            watcherReadyPath,
            watcherReadyNonce,
            watcherDeadline
          )

          if (watcherProcess) {watcherDiagnostics = observeTargetJobWatcher(watcherProcess, watcherSpawnedAt)}

          if (!watcherProcess) {
            watcherStartupFailure = `watcher-spawn-failed wrapper-pid=${wrapperProcessIdentity.pid} launch-pid=${launchedPid}`
            killing = true

            return
          }
        }

        // Capture identities promptly after spawn with a bounded slice; abort must
        // not wait for a fresh snapshot before killing. Keep the authenticated
        // wrapper generation in the same confirmation set even when the launch
        // PID is a short-lived proxy.
        const snapshot = await snapshotProcessTreeIdentities(launchedPid, {
          timeoutMs: Math.min(400, killReserveMs)
        })

        // Never let the launch hint or an unknown-generation snapshot entry
        // replace the exact nonce-authenticated wrapper generation.
        treeSnapshot = [
          ...snapshot.filter(identity => identity.pid !== wrapperProcessIdentity?.pid),
          { pid: wrapperProcessIdentity.pid, createdAt: wrapperProcessIdentity.createdAt }
        ]
      })()
      // A marker failure kills the launch child; make sure the public promise
      // still enters terminalization even if execFile reports no callback.
      void wrapperMarkerSetup.then(() => {
        if (watcherStartupFailure && !settled && !terminalizing) {
          void finish({ stdout: '', stderr: watcherStartupFailure, code: 1, pid: childPid })
        }
      })
    }

    const onAbort = () => {
      if (settled || terminalizing) {return}
      killing = true

      if (typeof childPid === 'number') {
        // Kill immediately with already-captured identities (+ root). Never
        // delay kill for a fresh unbounded snapshot, and never synthesize
        // create-time. Unknown generation keeps confirmation fail-safe.
        if (treeSnapshot.length === 0) {
          treeSnapshot = [{ pid: childPid }]
        }
      }

      // finish() first drains the wrapper-owned named target Job, then kills and
      // confirms the authenticated wrapper/helper tree. Do not pre-kill the
      // wrapper here or its only persistent target-Job handle disappears before
      // the bounded drain can open it.
      void finish({ stdout: '', stderr: 'aborted', code: 1, pid: childPid })
    }

    if (signal) {
      if (signal.aborted) {
        onAbort()
      } else {
        signal.addEventListener('abort', onAbort, { once: true })
      }
    }
  })
}

/** Testable production runner; callers should use terminateWindowsHolderExact. */
export async function runPowerShellWithHardBoundary(
  script: string,
  timeoutMs = 4_000,
  signal?: AbortSignal,
  deadlineAt?: number,
  dependencies?: HardBoundaryDependencies
): Promise<{ stdout: string; stderr: string; code: number; pid?: number }> {
  return defaultRunPowerShell(script, timeoutMs, signal, deadlineAt, dependencies)
}

/**
 * Access denied is Win32 error 5. Exception MESSAGES are localized: the
 * English-only regex this replaced never fired on a German or Japanese
 * Windows, so a protected holder we could not even open was reported
 * ALREADY_GONE and the updater proceeded over a still-locked venv. Classify on
 * the numeric code, walking the inner-exception chain (Get-Process wraps a
 * Win32Exception).
 *
 * Exported because the generated script embeds it verbatim and the live suite
 * runs it against synthetic exception objects under real PowerShell.
 */
export const TERMINATE_ACCESS_DENIED_CLASSIFIER = `
function Test-AccessDeniedError($errorRecord) {
  $exception = $errorRecord.Exception
  $depth = 0
  while ($null -ne $exception -and $depth -lt 8) {
    if ($exception -is [System.ComponentModel.Win32Exception]) {
      if ([int]$exception.NativeErrorCode -eq 5) { return $true }
    }
    if ($exception -is [System.UnauthorizedAccessException]) { return $true }
    $hresult = 0
    try { $hresult = [int]$exception.HResult } catch { $hresult = 0 }
    # HRESULT_FROM_WIN32(ERROR_ACCESS_DENIED) = 0x80070005.
    if ($hresult -eq -2147024891) { return $true }
    $exception = $exception.InnerException
    $depth++
  }
  return $false
}
`.trim()

/**
 * Build a self-contained PowerShell script that terminates one PID only when
 * its create-time ticks still match the expected generation.
 */
export type ExactTerminateScriptOptions = {
  installRoot?: string
  resource?: string
  /**
   * Directory tree inside the install that this termination does NOT own.
   * `.hermes-runtime` is excluded from the update's mutation set because
   * foreign uv tool venvs borrow the managed interpreter as their base Python
   * (install-mutation-set.ts). Running from it therefore proves nothing about
   * blocking THIS update, so an image under this root is authorized only with
   * a re-proved lock on a file the update will actually rewrite.
   */
  sharedRuntimeRoot?: string
}

export function buildExactTerminateScript(
  pid: number,
  createdAtUnixSeconds: number,
  waitMs = 1_500,
  options?: ExactTerminateScriptOptions
): string {
  // createdAt from psutil is epoch seconds (float). Compare at second resolution.
  const expected = Number(createdAtUnixSeconds)
  const installRootClaim = psLiteral(options?.installRoot ?? '')
  const resourceClaim = psLiteral(options?.resource ?? '')
  const sharedRuntimeClaim = psLiteral(options?.sharedRuntimeRoot ?? '')

  return `
$ErrorActionPreference = 'Stop'
$pidTarget = ${Math.trunc(pid)}
$expectedUnix = [double]${expected}
$waitMs = ${Math.max(0, Math.trunc(waitMs))}
$installRootClaim = ${installRootClaim}
$resourceClaim = ${resourceClaim}
$sharedRuntimeClaim = ${sharedRuntimeClaim}
Add-Type -TypeDefinition @"
using System;
using System.Text;
using System.Runtime.InteropServices;
public static class HermesForceReleaseNative {
${JOB_DRAIN_NATIVE}
  public const uint PROCESS_TERMINATE = 0x0001;
  public const uint PROCESS_SET_QUOTA = 0x0100;
  public const uint PROCESS_SUSPEND_RESUME = 0x0800;
  public const uint PROCESS_QUERY_LIMITED_INFORMATION = 0x1000;
  public const uint SYNCHRONIZE = 0x00100000;
  public const uint JOB_OBJECT_ASSIGN_PROCESS = 0x0001;
  public const uint JOB_OBJECT_TERMINATE = 0x0008;
  public const uint JOB_OBJECT_QUERY = 0x0004;
  public const uint JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000;
  public const uint JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK = 0x00001000;
  public const int JobObjectExtendedLimitInformation = 9;
  public const uint WAIT_OBJECT_0 = 0;
  public const uint WAIT_TIMEOUT = 258;
  public const uint FILE_SHARE_ALL = 0x00000007;
  public const uint OPEN_EXISTING = 3;
  public const uint FILE_FLAG_BACKUP_SEMANTICS = 0x02000000;
  public const int CCH_RM_SESSION_KEY = 32;
  public const int CCH_RM_MAX_APP_NAME = 255;
  public const int CCH_RM_MAX_SVC_NAME = 63;
  [StructLayout(LayoutKind.Sequential)]
  public struct BasicLimits {
    public long PerProcessUserTimeLimit, PerJobUserTimeLimit;
    public uint LimitFlags;
    public UIntPtr MinimumWorkingSetSize, MaximumWorkingSetSize;
    public uint ActiveProcessLimit;
    public UIntPtr Affinity;
    public uint PriorityClass, SchedulingClass;
  }
  [StructLayout(LayoutKind.Sequential)]
  public struct IoCounters {
    public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount;
    public ulong ReadTransferCount, WriteTransferCount, OtherTransferCount;
  }
  [StructLayout(LayoutKind.Sequential)]
  public struct ExtendedLimits {
    public BasicLimits BasicLimitInformation;
    public IoCounters IoInfo;
    public UIntPtr ProcessMemoryLimit, JobMemoryLimit, PeakProcessMemoryUsed, PeakJobMemoryUsed;
  }
  [StructLayout(LayoutKind.Sequential)]
  public struct FileTime { public uint Low; public uint High; }
  [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)]
  public struct RmUniqueProcess { public int ProcessId; public FileTime ProcessStartTime; }
  [StructLayout(LayoutKind.Sequential, CharSet=CharSet.Unicode)]
  public struct RmProcessInfo {
    public RmUniqueProcess Process;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst=CCH_RM_MAX_APP_NAME+1)] public string AppName;
    [MarshalAs(UnmanagedType.ByValTStr, SizeConst=CCH_RM_MAX_SVC_NAME+1)] public string ServiceShortName;
    public int ApplicationType;
    public uint AppStatus;
    public uint TerminalSessionId;
    [MarshalAs(UnmanagedType.Bool)] public bool Restartable;
  }
  [DllImport("kernel32.dll", SetLastError = true)]
  public static extern IntPtr OpenProcess(uint desiredAccess, bool inheritHandle, int processId);
  [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
  public static extern IntPtr CreateJobObject(IntPtr attributes, string name);
  [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
  public static extern IntPtr OpenJobObject(uint desiredAccess, bool inheritHandle, string name);
  [DllImport("kernel32.dll", SetLastError = true)]
  public static extern bool SetInformationJobObject(IntPtr job, int infoClass, ref ExtendedLimits info, uint length);
  [DllImport("kernel32.dll", SetLastError = true)]
  public static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
  [DllImport("kernel32.dll", SetLastError = true)]
  public static extern bool TerminateProcess(IntPtr process, uint exitCode);
  [DllImport("kernel32.dll", SetLastError = true)]
  public static extern bool IsProcessInJob(IntPtr process, IntPtr job, out bool result);
  [DllImport("kernel32.dll", SetLastError = true)]
  public static extern bool TerminateJobObject(IntPtr job, uint exitCode);
  [DllImport("kernel32.dll", SetLastError = true)]
  public static extern uint WaitForSingleObject(IntPtr handle, uint milliseconds);
  [DllImport("kernel32.dll", SetLastError = true)]
  public static extern bool GetProcessTimes(IntPtr process, out FileTime creation, out FileTime exit, out FileTime kernel, out FileTime user);
  [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
  public static extern bool QueryFullProcessImageName(IntPtr process, uint flags, StringBuilder path, ref uint size);
  [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
  public static extern IntPtr CreateFile(string fileName, uint desiredAccess, uint shareMode, IntPtr securityAttributes, uint creationDisposition, uint flagsAndAttributes, IntPtr templateFile);
  [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
  public static extern uint GetFinalPathNameByHandle(IntPtr file, StringBuilder path, uint pathLength, uint flags);
  [DllImport("kernel32.dll", SetLastError = true)]
  public static extern bool CloseHandle(IntPtr handle);
  [DllImport("ntdll.dll")]
  public static extern int NtSuspendProcess(IntPtr process);
  [DllImport("ntdll.dll")]
  public static extern int NtResumeProcess(IntPtr process);
  [DllImport("rstrtmgr.dll", CharSet = CharSet.Unicode)]
  public static extern int RmStartSession(out uint sessionHandle, int sessionFlags, StringBuilder sessionKey);
  [DllImport("rstrtmgr.dll")]
  public static extern int RmEndSession(uint sessionHandle);
  [DllImport("rstrtmgr.dll", CharSet = CharSet.Unicode)]
  public static extern int RmRegisterResources(uint sessionHandle, uint fileCount, string[] files, uint appCount, IntPtr apps, uint serviceCount, string[] services);
  [DllImport("rstrtmgr.dll")]
  public static extern int RmGetList(uint sessionHandle, out uint needed, ref uint count, [In,Out] RmProcessInfo[] processes, ref uint rebootReasons);
  public static double ToUnixSeconds(FileTime time) {
    long fileTicks = ((long)time.High << 32) | time.Low;
    return (fileTicks - 116444736000000000L) / 10000000.0;
  }
  public static IntPtr OpenAuthenticatedProcess(int pid, double expectedUnix, out double actualUnix, out int error) {
    actualUnix = 0;
    error = 0;
    IntPtr process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION | PROCESS_SET_QUOTA | PROCESS_SUSPEND_RESUME | PROCESS_TERMINATE | SYNCHRONIZE, false, pid);
    if (process == IntPtr.Zero) {
      error = Marshal.GetLastWin32Error();
      return IntPtr.Zero;
    }
    FileTime creation, exit, kernel, user;
    if (!GetProcessTimes(process, out creation, out exit, out kernel, out user)) {
      error = Marshal.GetLastWin32Error();
      CloseHandle(process);
      return IntPtr.Zero;
    }
    actualUnix = ToUnixSeconds(creation);
    if (Math.Abs(actualUnix - expectedUnix) > 1.5) {
      error = 0x10001;
      CloseHandle(process);
      return IntPtr.Zero;
    }
    return process;
  }
  public static string ReadImagePath(IntPtr process) {
    uint size = 32768;
    var buffer = new StringBuilder((int)size);
    return QueryFullProcessImageName(process, 0, buffer, ref size) ? buffer.ToString() : "";
  }
  public static string ReadFinalPath(string path) {
    if (String.IsNullOrWhiteSpace(path)) return "";
    IntPtr handle = CreateFile(path, 0, FILE_SHARE_ALL, IntPtr.Zero, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, IntPtr.Zero);
    if (handle == new IntPtr(-1)) return "";
    try {
      var buffer = new StringBuilder(32768);
      uint length = GetFinalPathNameByHandle(handle, buffer, (uint)buffer.Capacity, 0);
      if (length == 0 || length >= buffer.Capacity) return "";
      string value = buffer.ToString();
      if (value.StartsWith(@"\\\\?\\UNC\\", StringComparison.OrdinalIgnoreCase)) value = @"\\\\" + value.Substring(8);
      else if (value.StartsWith(@"\\\\?\\", StringComparison.OrdinalIgnoreCase)) value = value.Substring(4);
      return value.TrimEnd('\\\\');
    } finally {
      CloseHandle(handle);
    }
  }
  public static bool IsSameOrUnderRoot(string path, string root) {
    if (String.IsNullOrWhiteSpace(path) || String.IsNullOrWhiteSpace(root)) return false;
    string cleanRoot = root.TrimEnd('\\\\');
    return path.Equals(cleanRoot, StringComparison.OrdinalIgnoreCase) ||
      path.StartsWith(cleanRoot + "\\\\", StringComparison.OrdinalIgnoreCase);
  }
  public static bool IsCurrentResourceOwner(string resource, int pid, double expectedUnix) {
    if (String.IsNullOrWhiteSpace(resource)) return false;
    uint session;
    var key = new StringBuilder(CCH_RM_SESSION_KEY + 1);
    if (RmStartSession(out session, 0, key) != 0) return false;
    try {
      if (RmRegisterResources(session, 1, new string[]{ resource }, 0, IntPtr.Zero, 0, null) != 0) return false;
      uint needed = 0, count = 0, reboot = 0;
      int rc = RmGetList(session, out needed, ref count, null, ref reboot);
      if (rc != 234 || needed == 0) return false;
      count = needed;
      var rows = new RmProcessInfo[count];
      rc = RmGetList(session, out needed, ref count, rows, ref reboot);
      if (rc != 0) return false;
      for (int i = 0; i < count; i++) {
        if (rows[i].Process.ProcessId != pid) continue;
        double actualUnix = ToUnixSeconds(rows[i].Process.ProcessStartTime);
        if (Math.Abs(actualUnix - expectedUnix) <= 1.5) return true;
      }
      return false;
    } finally {
      RmEndSession(session);
    }
  }
  public static int SuspendProcess(IntPtr process) {
    int status = NtSuspendProcess(process);
    return status == 0 ? 0 : status;
  }
  public static int ResumeProcess(IntPtr process) {
    int status = NtResumeProcess(process);
    return status == 0 ? 0 : status;
  }
  public static int ReadCreatedAt(IntPtr process, out double createdUnix) {
    createdUnix = 0;
    FileTime creation, exit, kernel, user;
    if (!GetProcessTimes(process, out creation, out exit, out kernel, out user)) {
      int error = Marshal.GetLastWin32Error();
      return error == 0 ? -1 : -error;
    }
    createdUnix = ToUnixSeconds(creation);
    return 0;
  }
  public static IntPtr CreateKillOnCloseJob() {
    return CreateNamedKillOnCloseJob(null);
  }
  public static IntPtr CreateNamedKillOnCloseJob(string name) {
    IntPtr job = CreateJobObject(IntPtr.Zero, name);
    if (job == IntPtr.Zero) return IntPtr.Zero;
    var limits = new ExtendedLimits();
    // Descendants are excluded until this scanner authenticates and assigns
    // them explicitly. This prevents an unreviewed executable spawned between
    // assignment and suspension from inheriting a lethal target boundary.
    limits.BasicLimitInformation.LimitFlags =
      JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE | JOB_OBJECT_LIMIT_SILENT_BREAKAWAY_OK;
    if (!SetInformationJobObject(job, JobObjectExtendedLimitInformation, ref limits, (uint)Marshal.SizeOf(typeof(ExtendedLimits)))) {
      CloseHandle(job);
      return IntPtr.Zero;
    }
    return job;
  }
  public static int AssignProcessHandle(IntPtr job, IntPtr process) {
    if (AssignProcessToJobObject(job, process)) return 0;
    int error = Marshal.GetLastWin32Error();
    return error == 0 ? -1 : -error;
  }
  public static int ProcessIsInJob(IntPtr job, IntPtr process) {
    bool contained = false;
    if (!IsProcessInJob(process, job, out contained)) {
      int error = Marshal.GetLastWin32Error();
      return error == 0 ? -1 : -error;
    }
    return contained ? 1 : 0;
  }
  public static int ProcessIsInAnyJob(IntPtr process) {
    return ProcessIsInJob(IntPtr.Zero, process);
  }
  // Degraded containment: an authenticated holder that already lives in a job
  // whose hierarchy refuses ours is terminated by handle after suspension.
  public static int TerminateProcessAndWait(IntPtr process, int waitMs) {
    if (!TerminateProcess(process, 1)) {
      int error = Marshal.GetLastWin32Error();
      return error == 0 ? -1 : -error;
    }
    uint result = WaitForSingleObject(process, (uint)Math.Max(0, waitMs));
    return result == 0 ? 0 : -258;
  }
  public static IntPtr OpenNamedTargetJob(string name, out int error) {
    IntPtr job = OpenJobObject(JOB_OBJECT_ASSIGN_PROCESS | JOB_OBJECT_TERMINATE | JOB_OBJECT_QUERY | SYNCHRONIZE, false, name);
    error = job == IntPtr.Zero ? Marshal.GetLastWin32Error() : 0;
    return job;
  }
  public static int TerminateJobAndWait(IntPtr job, int waitMs) {
    if (!TerminateJobObject(job, 1)) {
      int error = Marshal.GetLastWin32Error();
      return error == 0 ? -1 : -error;
    }
    return WaitForJobEmpty(job, waitMs);
  }
}
"@

function Get-IdentityUnix([int]$targetPid) {
  $p = Get-Process -Id $targetPid -ErrorAction Stop
  return [DateTimeOffset]::new($p.StartTime.ToUniversalTime()).ToUnixTimeSeconds()
}

${TERMINATE_ACCESS_DENIED_CLASSIFIER}

try {
  $actualUnix = Get-IdentityUnix $pidTarget
} catch {
  if (Test-AccessDeniedError $_) {
    Write-Output ('ACCESS_DENIED ' + [string]$_.Exception.Message)
    exit 5
  }
  Write-Output 'ALREADY_GONE'
  exit 0
}
if ([math]::Abs($actualUnix - $expectedUnix) -gt 1.5) {
  Write-Output ("CREATE_TIME_MISMATCH actual=" + $actualUnix + " expected=" + $expectedUnix)
  exit 3
}

$job = [IntPtr]::Zero
$targetJobName = [Environment]::GetEnvironmentVariable('HERMES_TERMINATE_TARGET_JOB_NAME')
$externalTargetJob = -not [string]::IsNullOrWhiteSpace($targetJobName)
$handles = @{}
$suspended = New-Object 'System.Collections.Generic.List[int]'
$contained = New-Object 'System.Collections.Generic.HashSet[int]'
$degraded = New-Object 'System.Collections.Generic.List[int]'
$foreign = New-Object 'System.Collections.Generic.List[int]'
$success = $false
$exitCode = 1
$treeRows = $null

function Get-TreeChildren([int]$parentPid) {
  if ($null -eq $script:treeRows) {
    try {
      # Capture one coherent process snapshot. Re-querying WMI for every
      # generation consumes the mutation deadline and widens PID-reuse races.
      $script:treeRows = @(Get-CimInstance Win32_Process -ErrorAction Stop)
    } catch {
      # A primary CIM snapshot failure still has one bounded provider fallback.
      # If this fallback also fails, the caller's finally kills only the
      # already-contained root and reports failure; it never claims clearance.
      $script:treeRows = @(Get-WmiObject Win32_Process -ErrorAction Stop)
    }
  }
  return @($script:treeRows | Where-Object { [int]$_.ParentProcessId -eq $parentPid })
}

function Get-TreeRowCreationUnix($row) {
  $raw = $row.CreationDate
  if ($null -eq $raw -or [string]::IsNullOrWhiteSpace([string]$raw)) {
    throw 'TREE_SNAPSHOT_MISSING_CREATE_TIME'
  }
  try {
    if ($raw -is [DateTime]) {
      return [DateTimeOffset]::new(([DateTime]$raw).ToUniversalTime()).ToUnixTimeSeconds()
    }
    return [DateTimeOffset]::new(
      [System.Management.ManagementDateTimeConverter]::ToDateTime([string]$raw).ToUniversalTime()
    ).ToUnixTimeSeconds()
  } catch {
    throw 'TREE_SNAPSHOT_INVALID_CREATE_TIME'
  }
}

function Assert-NativeSuccess([int]$code, [string]$operation) {
  if ($code -ne 0) { throw ($operation + ' win32=' + (-$code)) }
}

function Assign-ContainedProcess([IntPtr]$processHandle, [int]$currentPid) {
  # Returns $true when the process is inside our job, $false when containment
  # is degraded for it: ERROR_ACCESS_DENIED (5) is how Windows refuses to nest
  # a process that already lives in a job whose hierarchy does not admit ours
  # (a service launcher's job, a supervised backend). It is still the exact,
  # authenticated holder: it stays suspended (no job needed for that) and is
  # terminated by handle instead of through the job. 2026-09-05: the gateway's
  # venv trampoline failed here with win32=5 and the whole force-release
  # reported failure while the holder kept the venv locked.
  $assignCode = [HermesForceReleaseNative]::AssignProcessHandle($job, $processHandle)
  if ($assignCode -ne 0) {
    if ((-$assignCode) -eq 5 -and [HermesForceReleaseNative]::ProcessIsInAnyJob($processHandle) -eq 1) {
      [void]$degraded.Add($currentPid)
      return $false
    }
    throw ('TREE_ASSIGN_FAILED pid=' + $currentPid + ' win32=' + (-$assignCode))
  }
  $membershipCode = [HermesForceReleaseNative]::ProcessIsInJob($job, $processHandle)
  if ($membershipCode -ne 1) {
    Assert-NativeSuccess $membershipCode ('TREE_ASSIGN_MEMBERSHIP_FAILED pid=' + $currentPid)
    throw ('TREE_ASSIGN_MEMBERSHIP_FAILED pid=' + $currentPid + ' win32=0')
  }
  return $true
}

try {
  if ($externalTargetJob) {
    $targetJobOpenError = 0
    $job = [HermesForceReleaseNative]::OpenNamedTargetJob($targetJobName, [ref]$targetJobOpenError)
    if ($job -eq [IntPtr]::Zero) { throw ('TREE_JOB_OPEN_FAILED win32=' + $targetJobOpenError) }
  } else {
    $job = [HermesForceReleaseNative]::CreateKillOnCloseJob()
    if ($job -eq [IntPtr]::Zero) { throw 'TREE_JOB_CREATE_FAILED win32=5' }
  }

  # Authenticate and assign each generation from one handle before suspending
  # it. A helper death after assignment closes this job and kills the member;
  # no suspended process can remain outside the terminal boundary.
  $rootActual = 0.0
  $rootOpenError = 0
  $rootHandle = [HermesForceReleaseNative]::OpenAuthenticatedProcess($pidTarget, $expectedUnix, [ref]$rootActual, [ref]$rootOpenError)
  if ($rootHandle -eq [IntPtr]::Zero) {
    if ($rootOpenError -eq 0x10001) { throw ("CREATE_TIME_MISMATCH root=" + $pidTarget) }
    throw ('HOLDER_OPEN_FAILED win32=' + $rootOpenError)
  }
  # PID and creation time prove identity, not authority to terminate. Every
  # target must also execute from this installation.
  if ([string]::IsNullOrWhiteSpace($installRootClaim)) {
    throw 'TERMINATION_AUTHORIZATION_CLAIM_MISSING'
  }
  $installRootFinal = [HermesForceReleaseNative]::ReadFinalPath($installRootClaim)
  if ([string]::IsNullOrWhiteSpace($installRootFinal)) {
    throw 'TERMINATION_FINAL_PATH_UNAVAILABLE'
  }
  $imagePath = [HermesForceReleaseNative]::ReadImagePath($rootHandle)
  $imageFinal = [HermesForceReleaseNative]::ReadFinalPath($imagePath)
  if ([string]::IsNullOrWhiteSpace($imageFinal)) {
    throw 'TERMINATION_EXECUTABLE_IDENTITY_UNAVAILABLE'
  }
  if (-not [HermesForceReleaseNative]::IsSameOrUnderRoot($imageFinal, $installRootFinal)) {
    throw 'TERMINATION_EXECUTABLE_OUTSIDE_INSTALL_ROOT'
  }
  # The managed runtime under the install root is SHARED. It is deliberately
  # excluded from the update's mutation set because unrelated uv tool venvs
  # (an MCP server another agent spawned) borrow the managed interpreter as
  # their base Python and map its DLLs without touching our venv. Their image
  # therefore lives under our install root while they block nothing we
  # rewrite, and "image under install root" alone would authorize killing
  # them. Under the shared runtime, authorization needs a re-proved lock on a
  # file this update actually mutates.
  $imageInSharedRuntime = $false
  if (-not [string]::IsNullOrWhiteSpace($sharedRuntimeClaim)) {
    $sharedRuntimeFinal = [HermesForceReleaseNative]::ReadFinalPath($sharedRuntimeClaim)
    if (-not [string]::IsNullOrWhiteSpace($sharedRuntimeFinal)) {
      $imageInSharedRuntime = [HermesForceReleaseNative]::IsSameOrUnderRoot($imageFinal, $sharedRuntimeFinal)
    }
  }
  # Only the Restart Manager ownership re-proof stays conditional: a
  # scanner-only holder names no file it holds.
  if (-not [string]::IsNullOrWhiteSpace($resourceClaim)) {
    $resourceFinal = [HermesForceReleaseNative]::ReadFinalPath($resourceClaim)
    if ([string]::IsNullOrWhiteSpace($resourceFinal)) {
      throw 'TERMINATION_FINAL_PATH_UNAVAILABLE'
    }
    if (-not [HermesForceReleaseNative]::IsSameOrUnderRoot($resourceFinal, $installRootFinal)) {
      throw 'TERMINATION_RESOURCE_OUTSIDE_INSTALL_ROOT'
    }
    if ($imageInSharedRuntime -and -not [string]::IsNullOrWhiteSpace($sharedRuntimeFinal) -and
        [HermesForceReleaseNative]::IsSameOrUnderRoot($resourceFinal, $sharedRuntimeFinal)) {
      # A lock on the shared runtime itself is not a lock on the mutation set.
      throw 'TERMINATION_SHARED_RUNTIME_WITHOUT_MUTATION_PROOF'
    }
    if (-not [HermesForceReleaseNative]::IsCurrentResourceOwner($resourceFinal, $pidTarget, $expectedUnix)) {
      throw 'TERMINATION_CURRENT_LOCK_OWNERSHIP_MISMATCH'
    }
  } elseif ($imageInSharedRuntime) {
    throw 'TERMINATION_SHARED_RUNTIME_WITHOUT_MUTATION_PROOF'
  }
  $handles[[string]$pidTarget] = $rootHandle
  if (Assign-ContainedProcess $rootHandle $pidTarget) { [void]$contained.Add($pidTarget) }
  $suspendRoot = [HermesForceReleaseNative]::SuspendProcess($rootHandle)
  Assert-NativeSuccess $suspendRoot 'TREE_SUSPEND_FAILED'
  [void]$suspended.Add($pidTarget)
  $rootRevalidated = 0.0
  Assert-NativeSuccess ([HermesForceReleaseNative]::ReadCreatedAt($rootHandle, [ref]$rootRevalidated)) 'TREE_REVALIDATE_FAILED'
  if ([math]::Abs($rootRevalidated - $expectedUnix) -gt 1.5) {
    throw ("CREATE_TIME_MISMATCH root=" + $pidTarget)
  }

  $rows = New-Object 'System.Collections.Generic.List[object]'
  $seen = @{}
  $queue = New-Object 'System.Collections.Generic.Queue[object]'
  $seen[[string]$pidTarget] = $true
  [void]$queue.Enqueue([pscustomobject]@{ pid = $pidTarget; created = $rootRevalidated })
  while ($queue.Count -gt 0) {
    $parent = $queue.Dequeue()
    $parentPid = [int]$parent.pid
    $parentCreated = [double]$parent.created
    foreach ($child in @(Get-TreeChildren $parentPid)) {
      $childPid = [int]$child.ProcessId
      if ($childPid -le 0 -or $seen.ContainsKey([string]$childPid)) { continue }
      # Use the creation timestamp from the same process-tree row that yielded
      # this PID. Never turn a stale/reused PID into a fresh authenticated
      # generation by probing it again before opening its boundary handle.
      $childExpected = Get-TreeRowCreationUnix $child
      # Windows can retain a stale ParentProcessId after the original parent PID
      # exits and is reused. An older process is not a descendant of this exact
      # authenticated parent generation and must never be opened or assigned.
      if (($childExpected + 1.5) -lt $parentCreated) { continue }
      $seen[[string]$childPid] = $true
      $childActual = 0.0
      $childOpenError = 0
      $childHandle = [HermesForceReleaseNative]::OpenAuthenticatedProcess($childPid, $childExpected, [ref]$childActual, [ref]$childOpenError)
      if ($childHandle -eq [IntPtr]::Zero) {
        if ($childOpenError -eq 0x10001) { throw ("CREATE_TIME_MISMATCH child=" + $childPid) }
        throw ('TREE_OPEN_FAILED win32=' + $childOpenError)
      }
      # A descendant is authorized by the same executable-under-install-root
      # proof as the root. A foreign image (the gateway's terminal tool ran an
      # editor, git, or a user shell) is never assigned to the job: it stays
      # alive, its subtree is not walked, and it is reported.
      $childImageFinal = [HermesForceReleaseNative]::ReadFinalPath([HermesForceReleaseNative]::ReadImagePath($childHandle))
      if ([string]::IsNullOrWhiteSpace($childImageFinal) -or -not [HermesForceReleaseNative]::IsSameOrUnderRoot($childImageFinal, $installRootFinal)) {
        [HermesForceReleaseNative]::CloseHandle($childHandle) | Out-Null
        [void]$foreign.Add($childPid)
        continue
      }
      $handles[[string]$childPid] = $childHandle
      if (Assign-ContainedProcess $childHandle $childPid) { [void]$contained.Add($childPid) }
      $childSuspend = [HermesForceReleaseNative]::SuspendProcess($childHandle)
      Assert-NativeSuccess $childSuspend 'TREE_SUSPEND_FAILED'
      [void]$suspended.Add($childPid)
      $childRevalidated = 0.0
      Assert-NativeSuccess ([HermesForceReleaseNative]::ReadCreatedAt($childHandle, [ref]$childRevalidated)) 'TREE_REVALIDATE_FAILED'
      if ([math]::Abs($childRevalidated - $childExpected) -gt 1.5) {
        throw ("CREATE_TIME_MISMATCH child=" + $childPid)
      }
      [void]$rows.Add([pscustomobject]@{ pid = $childPid; created = $childRevalidated })
      [void]$queue.Enqueue([pscustomobject]@{ pid = $childPid; created = $childRevalidated })
    }
  }

  if ($contained.Count -gt 0) {
    $terminateCode = [HermesForceReleaseNative]::TerminateJobAndWait($job, $waitMs)
    Assert-NativeSuccess $terminateCode 'TREE_TERMINATE_FAILED'
  }
  # Every member is suspended, so terminating the degraded ones by handle
  # after the job cannot let a survivor spawn past the boundary.
  foreach ($degradedPid in $degraded) {
    $degradedHandle = $handles[[string]$degradedPid]
    $degradedCode = [HermesForceReleaseNative]::TerminateProcessAndWait($degradedHandle, $waitMs)
    Assert-NativeSuccess $degradedCode ('TREE_TERMINATE_FAILED degraded pid=' + $degradedPid)
  }
  $allRows = @([pscustomobject]@{ pid = $pidTarget; created = $rootRevalidated }) + @($rows.ToArray())
  if (-not $externalTargetJob) {
    foreach ($row in $allRows) {
      $live = Get-Process -Id ([int]$row.pid) -ErrorAction SilentlyContinue
      if ($null -ne $live) {
        $liveCreated = Get-IdentityUnix ([int]$row.pid)
        if ([math]::Abs($liveCreated - [double]$row.created) -le 1.5) {
          throw ("TREE_SURVIVOR pid=" + [int]$row.pid)
        }
      }
    }
  }
  $success = $true
  $exitCode = 0
  Write-Output 'TERMINATED'
  if ($degraded.Count -gt 0) { Write-Output ('CONTAINMENT_DEGRADED pids=' + ($degraded -join ',')) }
  if ($foreign.Count -gt 0) { Write-Output ('FOREIGN_DESCENDANTS_LEFT_ALIVE pids=' + ($foreign -join ',')) }
} catch {
  $message = [string]$_.Exception.Message
  # Elevation is authorized only when opening the exact authenticated holder
  # failed with access denied. Generic Job create/assign/terminate failures do
  # not prove protected holder ownership and must remain ordinary failures.
  if ($message -match '^HOLDER_OPEN_FAILED win32=5$') {
    Write-Output ('ACCESS_DENIED ' + $message)
    $exitCode = 5
  } elseif ($message -match 'CREATE_TIME_MISMATCH') {
    Write-Output $message
    $exitCode = 3
  } else {
    Write-Output ('BOUNDARY_FAILED ' + $message)
    $exitCode = 1
  }
} finally {
  if ($job -ne [IntPtr]::Zero) {
    if (-not $success -and $contained.Count -gt 0) { [HermesForceReleaseNative]::TerminateJobAndWait($job, $waitMs) | Out-Null }
    [HermesForceReleaseNative]::CloseHandle($job) | Out-Null
  }
  for ($index = $suspended.Count - 1; $index -ge 0; $index--) {
    $suspendedPid = [int]$suspended[$index]
    if (-not $contained.Contains($suspendedPid)) {
      $handle = $handles[[string]$suspendedPid]
      if ($null -ne $handle) { [HermesForceReleaseNative]::ResumeProcess($handle) | Out-Null }
    }
  }
  foreach ($entry in $handles.GetEnumerator()) {
    [HermesForceReleaseNative]::CloseHandle([IntPtr]$entry.Value) | Out-Null
  }
}
exit $exitCode
`.trim()
}

export function parseTerminateScriptOutput(stdout: string, code: number): ForceReleaseTerminateResult {
  const text = String(stdout || '').trim()

  if (/PROTECTED/i.test(text)) {
    const win32 = text.match(/win32=(\d+)/i)

    return { kind: 'protected', win32Error: win32 ? Number(win32[1]) : 5 }
  }

  if (/ALREADY_GONE/i.test(text) || (code === 0 && /TERMINATED/i.test(text))) {
    if (/TERMINATED/i.test(text)) {return { kind: 'terminated' }}

    if (/ALREADY_GONE/i.test(text)) {return { kind: 'already-gone' }}
  }

  if (/CREATE_TIME_MISMATCH/i.test(text) || code === 3) {
    return { kind: 'create-time-mismatch' }
  }

  if (/ACCESS_DENIED/i.test(text)) {
    const marker = text.match(/ACCESS_DENIED(?:\s+([\s\S]*))?/i)
    const detail = marker?.[1]?.trim()

    return detail ? { kind: 'access-denied', win32Error: 5, detail } : { kind: 'access-denied', win32Error: 5 }
  }

  const win32 = text.match(/win32=(\d+)/i)

  if (win32) {
    const err = Number(win32[1])

    // Do not treat 6/87 as already-gone: the process was observed live above.
    return { kind: 'failed', detail: text || `win32=${err}`, win32Error: err }
  }

  if (code === 0 && /TERMINATED/i.test(text)) {return { kind: 'terminated' }}

  return { kind: 'failed', detail: text || `exit ${code}` }
}

export async function terminateWindowsHolderExact(
  target: ForceReleaseHolder,
  {
    platform = process.platform,
    run = defaultRunPowerShell,
    waitMs = 1_500,
    timeoutMs,
    signal,
    deadlineAt,
    installRoot,
    sharedRuntimeRoot,
    buildScript = buildExactTerminateScript
  }: {
    platform?: NodeJS.Platform
    run?: RunPowerShell
    waitMs?: number
    /** Hard wall-clock budget for the PowerShell child including kill/confirm. */
    timeoutMs?: number
    /** When aborted, kill the child tree, await confirmed absence, then return. */
    signal?: AbortSignal
    /** Absolute deadline shared with the caller's orchestration budget. */
    deadlineAt?: number
    /** Canonical update root for final resource-authorization validation. */
    installRoot?: string
    /** Tree inside the install that the mutation set excludes; see ExactTerminateScriptOptions. */
    sharedRuntimeRoot?: string
    /** Explicit test seam; production uses buildExactTerminateScript. */
    buildScript?: (
      pid: number,
      createdAt: number,
      waitMs: number,
      options?: ExactTerminateScriptOptions
    ) => string
  } = {}
): Promise<ForceReleaseTerminateResult> {
  if (platform !== 'win32') {
    return { kind: 'failed', detail: 'windows-only' }
  }

  if (signal?.aborted) {
    return { kind: 'failed', detail: 'deadline-exhausted' }
  }

  if (!Number.isInteger(target.pid) || target.pid <= 0) {
    return { kind: 'failed', detail: 'invalid pid' }
  }

  if (!Number.isFinite(target.createdAt) || target.createdAt <= 0) {
    return { kind: 'failed', detail: 'invalid createdAt' }
  }

  const requestedBudget = Math.max(0, Math.trunc(timeoutMs ?? Math.max(2_000, waitMs + 1_000)))

  const remainingBudget =
    typeof deadlineAt === 'number' && Number.isFinite(deadlineAt)
      ? Math.max(0, Math.trunc(deadlineAt - Date.now()))
      : requestedBudget

  const budget = Math.min(requestedBudget, remainingBudget)
  // Keep TerminateProcess wait short enough that kill-reserve still fits.
  const killReserveMs = terminateKillReserveMs(budget)
  const runBudget = Math.max(1, budget - killReserveMs)
  const effectiveWait = Math.max(0, Math.min(Math.trunc(waitMs), Math.max(0, runBudget - 250)))

  if (budget <= 50) {
    return { kind: 'failed', detail: 'deadline-exhausted' }
  }

  const script = buildScript(target.pid, target.createdAt, effectiveWait, {
    installRoot,
    resource: target.resource,
    sharedRuntimeRoot: sharedRuntimeRoot ?? (installRoot ? installSharedRuntimeRoot(installRoot) : undefined)
  })

  const result = await run(script, budget, signal, deadlineAt)

  if (signal?.aborted) {
    // Child tree must already be confirmed gone by run(); never claim mutation.
    return { kind: 'failed', detail: deadlineFailureDetail(result.stderr) }
  }

  if (/unconfirmed-tree-survivors/i.test(result.stderr || '')) {
    return { kind: 'failed', detail: sanitizeBoundaryDiagnostics(result.stderr) || 'unconfirmed-tree-survivors' }
  }

  // Timed-out/killed child: do not parse a partial TerminateProcess success.
  if (/aborted|ETIMEDOUT|timeout/i.test(result.stderr || '') && !/TERMINATED|ACCESS_DENIED|PROTECTED|CREATE_TIME/i.test(result.stdout || '')) {
    return { kind: 'failed', detail: deadlineFailureDetail(result.stderr) }
  }

  return parseTerminateScriptOutput(result.stdout + '\n' + result.stderr, result.code)
}

/**
 * Execute exact termination using the caller's remaining absolute deadline.
 * This is the production adapter used by the updater path; keeping the
 * deadline calculation here prevents a stale per-holder timeout from
 * extending the overall force-release contract.
 */
export async function terminateWindowsHolderWithinDeadline(
  target: ForceReleaseHolder,
  {
    platform = process.platform,
    run = defaultRunPowerShell,
    budgetMs,
    deadlineAt,
    installRoot,
    sharedRuntimeRoot,
    signal
  }: {
    platform?: NodeJS.Platform
    run?: RunPowerShell
    budgetMs: number
    deadlineAt: number
    installRoot?: string
    sharedRuntimeRoot?: string
    signal?: AbortSignal
  }
): Promise<ForceReleaseTerminateResult> {
  const requestedBudget = Math.max(0, Math.trunc(budgetMs))
  const absoluteDeadline = Number.isFinite(deadlineAt) ? Math.trunc(deadlineAt) : Date.now() + requestedBudget
  const remainingBudget = Math.max(0, absoluteDeadline - Date.now())
  const budget = Math.min(requestedBudget, remainingBudget)

  if (budget <= 50 || signal?.aborted) {
    return { kind: 'failed', detail: 'deadline-exhausted' }
  }

  return terminateWindowsHolderExact(target, {
    platform,
    run,
    timeoutMs: budget,
    waitMs: Math.max(0, Math.min(1_500, budget - 250)),
    signal,
    deadlineAt: absoluteDeadline,
    installRoot,
    ...(sharedRuntimeRoot ? { sharedRuntimeRoot } : {})
  })
}

/** Route plugin units through their owner and authorize every other holder by install root. */
export function createWindowsHolderTerminator(
  installRoot: string,
  stopPluginService: (service: NonNullable<ForceReleaseHolder['service']>) => Promise<boolean>,
  terminate = terminateWindowsHolderWithinDeadline
): WindowsUpdateForceReleaseDeps['terminateHolder'] {
  return async (holder, budgetMs, signal, deadlineAt) => {
    if (holder.terminateVia === 'desktop-plugin-service' && holder.service) {
      const stopped = await stopPluginService(holder.service)

      return stopped ? { kind: 'terminated' } : { kind: 'failed', detail: 'plugin service unit not stopped' }
    }

    return terminate(holder, {
      budgetMs,
      deadlineAt: deadlineAt ?? Date.now() + Math.max(0, budgetMs),
      signal,
      installRoot
    })
  }
}
