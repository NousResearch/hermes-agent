/**
 * Embedded program text for the Windows termination boundary.
 *
 * Everything in this module is PowerShell (plus the C# compiled by Add-Type)
 * or the Node bridge stub: source text that TypeScript tooling cannot lint,
 * type-check or format. It lives apart from the executor so
 * windows-process-terminate.ts reads as control flow rather than as a
 * thousand lines of quoted script, and so a change to the script text shows up
 * in review as a change to the script text.
 *
 * Only truncated integers and psLiteral-quoted strings are ever interpolated
 * into these templates; every other value reaches the child through the
 * environment. See buildExactTerminateScript for the authorization contract.
 */

import { psLiteral } from './windows-remote-lifecycle'

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

export const TERMINATE_JOB_WATCHER_BOOTSTRAP = String.raw`
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

/**
 * Walk a Windows process tree from a root PID and emit `pid|createdAtUnix`
 * rows joined by ';'. A missing CreationDate throws rather than yielding an
 * identity-free row, and a child whose creation predates its parent is dropped
 * as PID reuse. The truncated root PID is the only interpolated value.
 */
export function buildProcessTreeSnapshotScript(rootPid: number): string {
  return `
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
