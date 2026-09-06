# Native work belongs to the installer until every descendant has exited.
# Assign the suspended launcher to a job before it can create children. Closing
# this private job is a final kill-on-close safeguard, never a process-name kill.
function Initialize-InstallerNativeProcess {
    if ('HermesInstaller.NativeCommand' -as [type]) { return }
    Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;

namespace HermesInstaller {
    public sealed class NativeCommand : IDisposable {
        [StructLayout(LayoutKind.Sequential)] struct Security {
            public int Length; public IntPtr Descriptor; public int Inherit;
        }
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)] struct Startup {
            public int Size; public string Reserved, Desktop, Title;
            public int X, Y, XSize, YSize, XChars, YChars, Fill, Flags;
            public short Show, ReservedSize; public IntPtr ReservedBytes, Input, Output, Error;
        }
        [StructLayout(LayoutKind.Sequential)] struct ProcessInfo {
            public IntPtr Process, Thread; public int ProcessId, ThreadId;
        }
        [StructLayout(LayoutKind.Sequential)] struct BasicLimits {
            public long ProcessTime, JobTime; public uint Flags;
            public UIntPtr MinimumWorkingSet, MaximumWorkingSet;
            public uint ActiveProcessLimit; public UIntPtr Affinity;
            public uint PriorityClass, SchedulingClass;
        }
        [StructLayout(LayoutKind.Sequential)] struct IoCounters {
            public ulong ReadOperations, WriteOperations, OtherOperations, ReadBytes, WriteBytes, OtherBytes;
        }
        [StructLayout(LayoutKind.Sequential)] struct ExtendedLimits {
            public BasicLimits Basic; public IoCounters Io;
            public UIntPtr ProcessMemory, JobMemory, PeakProcessMemory, PeakJobMemory;
        }
        [StructLayout(LayoutKind.Sequential)] struct Accounting {
            public long UserTime, KernelTime, PeriodUserTime, PeriodKernelTime;
            public uint PageFaults, TotalProcesses, ActiveProcesses, TerminatedProcesses;
        }
        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        static extern IntPtr CreateJobObject(IntPtr attributes, string name);
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool SetInformationJobObject(IntPtr job, int kind, ref ExtendedLimits value, uint length);
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool QueryInformationJobObject(IntPtr job, int kind, out Accounting value, uint length, IntPtr returned);
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool TerminateJobObject(IntPtr job, uint code);
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool TerminateProcess(IntPtr process, uint code);
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern uint ResumeThread(IntPtr thread);
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool GetExitCodeProcess(IntPtr process, out uint code);
        [DllImport("kernel32.dll", SetLastError = true)]
        static extern bool CloseHandle(IntPtr handle);
        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        static extern IntPtr CreateFile(string name, uint access, uint share, ref Security security, uint creation, uint flags, IntPtr template);
        [DllImport("kernel32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
        static extern bool CreateProcess(string application, StringBuilder command, IntPtr processSecurity, IntPtr threadSecurity,
            bool inherit, uint flags, IntPtr environment, string directory, ref Startup startup, out ProcessInfo process);

        IntPtr job, process;
        public int ProcessId { get; private set; }
        static Exception Error(string action) { return new Win32Exception(Marshal.GetLastWin32Error(), action); }
        static bool Valid(IntPtr handle) { return handle != IntPtr.Zero && handle != new IntPtr(-1); }
        public static string Quote(string value) {
            // CommandLineToArgvW/CRT quoting, including empty args and trailing backslashes.
            var result = new StringBuilder("\""); int slashes = 0;
            foreach (char c in value) {
                if (c == '\\') { slashes++; continue; }
                if (c == '"') { result.Append('\\', slashes * 2 + 1); result.Append(c); }
                else { result.Append('\\', slashes); result.Append(c); }
                slashes = 0;
            }
            result.Append('\\', slashes * 2); return result.Append('"').ToString();
        }
        public static NativeCommand Start(string executable, string[] arguments, string directory, string output, string error) {
            var owner = new NativeCommand();
            IntPtr input = IntPtr.Zero, stdout = IntPtr.Zero, stderr = IntPtr.Zero;
            ProcessInfo info = new ProcessInfo();
            try {
                owner.job = CreateJobObject(IntPtr.Zero, null);
                if (!Valid(owner.job)) throw Error("create installer process job");
                var limits = new ExtendedLimits(); limits.Basic.Flags = 0x2000; // KILL_ON_JOB_CLOSE
                if (!SetInformationJobObject(owner.job, 9, ref limits, (uint)Marshal.SizeOf(limits))) throw Error("configure installer process job");
                var security = new Security { Length = Marshal.SizeOf(typeof(Security)), Inherit = 1 };
                input = CreateFile("NUL", 0x80000000, 3, ref security, 3, 0, IntPtr.Zero);
                stdout = CreateFile(output, 0x40000000, 7, ref security, 2, 0, IntPtr.Zero);
                stderr = CreateFile(error, 0x40000000, 7, ref security, 2, 0, IntPtr.Zero);
                if (!Valid(input) || !Valid(stdout) || !Valid(stderr)) throw Error("open installer process streams");
                var startup = new Startup { Size = Marshal.SizeOf(typeof(Startup)), Flags = 0x100, Input = input, Output = stdout, Error = stderr };
                var command = new StringBuilder(Quote(executable));
                foreach (string argument in arguments) command.Append(' ').Append(Quote(argument));
                // CREATE_SUSPENDED | CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT.
                if (!CreateProcess(executable, command, IntPtr.Zero, IntPtr.Zero, true, 0x08000404, IntPtr.Zero, directory, ref startup, out info)) throw Error("launch installer command");
                owner.process = info.Process; owner.ProcessId = info.ProcessId;
                if (!AssignProcessToJobObject(owner.job, owner.process)) throw Error("own installer command before execution");
                if (ResumeThread(info.Thread) == UInt32.MaxValue) throw Error("resume installer command");
                return owner;
            } catch {
                // An unassigned process is still suspended and has no descendants.
                if (Valid(owner.process)) TerminateProcess(owner.process, 125);
                owner.Dispose(); throw;
            } finally {
                if (Valid(info.Thread)) CloseHandle(info.Thread);
                if (Valid(input)) CloseHandle(input);
                if (Valid(stdout)) CloseHandle(stdout);
                if (Valid(stderr)) CloseHandle(stderr);
            }
        }
        public uint ActiveProcesses {
            get {
                Accounting value;
                if (!QueryInformationJobObject(job, 1, out value, (uint)Marshal.SizeOf(typeof(Accounting)), IntPtr.Zero)) throw Error("observe installer process ownership");
                return value.ActiveProcesses;
            }
        }
        public int ExitCode {
            get { uint value; if (!GetExitCodeProcess(process, out value)) throw Error("read installer exit code"); return unchecked((int)value); }
        }
        public void Terminate() { if (!TerminateJobObject(job, 124)) throw Error("terminate installer process job"); }
        public void Dispose() {
            if (Valid(job)) { CloseHandle(job); job = IntPtr.Zero; }
            if (Valid(process)) { CloseHandle(process); process = IntPtr.Zero; }
        }
    }
}
'@
}

# A build gets a generous hard limit, not an output-idle limit: healthy compiler
# and antivirus work can legitimately be quiet for minutes. Tests replace this
# table in their isolated stage process; it is not a new user configuration API.
$script:InstallerCommandTimeouts = @{
    Desktop = 3600; Electron = 600; Packages = 600; NodeDeps = 600; ComputerUse = 660
}

function Invoke-ProcessWithWallClockTimeout {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$ArgumentList = @(),
        [Parameter(Mandatory = $true)][ValidateRange(1, 86400)][int]$TimeoutSec,
        [string]$RedirectStandardOutput,
        [string]$RedirectStandardError,
        [string]$WorkingDirectory = (Get-Location).Path,
        [string]$Label = 'Installer command'
    )
    if ($script:InstallerNativeCleanupFailure) { throw $script:InstallerNativeCleanupFailure }
    Initialize-InstallerNativeProcess
    $temporary = @()
    $owner = $null
    $readers = @()
    $output = [Text.StringBuilder]::new()
    $clock = [Diagnostics.Stopwatch]::StartNew()
    $timedOut = $false
    $lastProgress = 0.0
    $cleanupError = $null
    try {
        if (-not $RedirectStandardOutput) { $RedirectStandardOutput = [IO.Path]::GetTempFileName(); $temporary += $RedirectStandardOutput }
        if (-not $RedirectStandardError) { $RedirectStandardError = [IO.Path]::GetTempFileName(); $temporary += $RedirectStandardError }
        if ([IO.Path]::GetFullPath($RedirectStandardOutput) -eq [IO.Path]::GetFullPath($RedirectStandardError)) { throw 'Native stdout and stderr paths must be distinct' }
        $command = Get-Command $FilePath -CommandType Application, ExternalScript -ErrorAction Stop | Select-Object -First 1
        $executable = $command.Source
        if ([IO.Path]::GetExtension($executable) -in @('.ps1', '.cmd', '.bat')) {
            # Passing an encoded script preserves paths/arguments without constructing
            # a cmd.exe command string. The script host and native grandchildren all
            # stay in the same job, even when a batch shim exits first.
            $literalArgs = @($executable) + $ArgumentList | ForEach-Object { "'" + $_.Replace("'", "''") + "'" }
            $invocation = '$ProgressPreference = ''SilentlyContinue''; [Console]::OutputEncoding = [Text.UTF8Encoding]::new($false); $global:LASTEXITCODE = 0; try { & ' + ($literalArgs -join ' ') + '; exit $global:LASTEXITCODE } catch { [Console]::Error.WriteLine([string]$_); exit 1 }'
            $executable = Join-Path $PSHOME $(if ($PSVersionTable.PSEdition -eq 'Core') { 'pwsh.exe' } else { 'powershell.exe' })
            $ArgumentList = @('-NoProfile', '-NonInteractive', '-OutputFormat', 'Text', '-EncodedCommand', [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($invocation)))
        }
        $owner = [HermesInstaller.NativeCommand]::Start($executable, $ArgumentList, $WorkingDirectory, $RedirectStandardOutput, $RedirectStandardError)
        foreach ($file in @($RedirectStandardOutput, $RedirectStandardError)) {
            $stream = [IO.File]::Open($file, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete)
            $readers += [IO.StreamReader]::new($stream, [Text.Encoding]::UTF8, $true)
        }
        do {
            foreach ($reader in $readers) {
                $chunk = $reader.ReadToEnd()
                if ($chunk) { [void]$output.Append($chunk); Write-Host -NoNewline $chunk }
            }
            $active = $owner.ActiveProcesses
            if ($active -eq 0) { break }
            if ($clock.Elapsed.TotalSeconds -ge $TimeoutSec) {
                $timedOut = $true
                $owner.Terminate()
                $cleanup = [Diagnostics.Stopwatch]::StartNew()
                while ($owner.ActiveProcesses -ne 0 -and $cleanup.Elapsed.TotalSeconds -lt 10) { Start-Sleep -Milliseconds 50 }
                if ($owner.ActiveProcesses -ne 0) { throw 'Installer process job did not become empty after termination' }
                Write-Warn "$Label timed out after ${TimeoutSec}s; its process tree has exited."
                break
            }
            if ($clock.Elapsed.TotalSeconds - $lastProgress -ge 15) {
                Write-Info "$Label is still running ($([int]$clock.Elapsed.TotalSeconds)s elapsed; ${TimeoutSec}s limit)."
                $lastProgress = $clock.Elapsed.TotalSeconds
            }
            Start-Sleep -Milliseconds 100
        } while ($true)
        foreach ($reader in $readers) {
            $chunk = $reader.ReadToEnd()
            if ($chunk) { [void]$output.Append($chunk); Write-Host -NoNewline $chunk }
        }
        $code = if ($timedOut) { 124 } else { $owner.ExitCode }
        $global:LASTEXITCODE = $code
        return @{ TimedOut = $timedOut; ExitCode = $code; ProcessId = $owner.ProcessId; Output = $output.ToString(); CleanupComplete = $true }
    } finally {
        if ($owner) {
            try {
                if ($owner.ActiveProcesses -ne 0) {
                    $owner.Terminate()
                    $cleanup = [Diagnostics.Stopwatch]::StartNew()
                    while ($owner.ActiveProcesses -ne 0 -and $cleanup.Elapsed.TotalSeconds -lt 10) { Start-Sleep -Milliseconds 50 }
                    if ($owner.ActiveProcesses -ne 0) { throw 'Installer native process teardown could not be verified' }
                }
            } catch {
                # Optional callers may catch an exception. No subsequent native
                # attempt may start after cleanup failed, even in a fallback.
                $script:InstallerNativeCleanupFailure = $_
                $cleanupError = $_
            } finally { $owner.Dispose() }
        }
        foreach ($reader in $readers) { $reader.Dispose() }
        foreach ($file in $temporary) { Remove-Item -LiteralPath $file -Force -ErrorAction SilentlyContinue }
        if ($cleanupError) { throw $cleanupError }
    }
}
$script:InstallerNativeCleanupFailure = $null
