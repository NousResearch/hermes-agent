# update-job.ps1 -- the Win32 job/process helper windows.ps1 runs every
# hermes step through (dot-sourced after the marker claim, never before it:
# Add-Type is not allowed in the claim path). StartAssigned creates the step
# SUSPENDED inside a private job (kill-on-close until Resume), so windows.ps1
# can publish the update marker's delegate before the first instruction runs.

function Initialize-HermesUpdateJob {
    # Compile the helper once per PowerShell session (Add-Type cannot redefine a type).
    if ("HermesUpdateJob" -as [type]) { return }
    Add-Type -TypeDefinition @'
using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading;
using Microsoft.Win32.SafeHandles;

public static class HermesUpdateJob {
    public sealed class StartedProcess {
        public Process Process;
        public StreamReader StandardOutput;
        public StreamReader StandardError;
        public IntPtr Job;
        public IntPtr Thread;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct BasicLimitInformation {
        public long PerProcessUserTimeLimit;
        public long PerJobUserTimeLimit;
        public uint LimitFlags;
        public UIntPtr MinimumWorkingSetSize;
        public UIntPtr MaximumWorkingSetSize;
        public uint ActiveProcessLimit;
        public UIntPtr Affinity;
        public uint PriorityClass;
        public uint SchedulingClass;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct ExtendedLimitInformation {
        public BasicLimitInformation Basic;
        public ulong ReadOperationCount, WriteOperationCount, OtherOperationCount;
        public ulong ReadTransferCount, WriteTransferCount, OtherTransferCount;
        public UIntPtr ProcessMemoryLimit;
        public UIntPtr JobMemoryLimit;
        public UIntPtr PeakProcessMemoryUsed;
        public UIntPtr PeakJobMemoryUsed;
    }

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetInformationJobObject(IntPtr job, int informationClass, ref ExtendedLimitInformation information, uint length);

    // JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE (0x2000) on, or every limit off.
    private static bool SetKillOnClose(IntPtr job, bool on) {
        ExtendedLimitInformation information = new ExtendedLimitInformation();
        information.Basic.LimitFlags = on ? 0x2000u : 0u;
        return SetInformationJobObject(job, 9, ref information, (uint)Marshal.SizeOf(typeof(ExtendedLimitInformation)));
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct SecurityAttributes {
        public int Length;
        public IntPtr SecurityDescriptor;
        public bool InheritHandle;
    }

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    private struct StartupInfo {
        public int Size;
        public string Reserved;
        public string Desktop;
        public string Title;
        public int X;
        public int Y;
        public int XSize;
        public int YSize;
        public int XCountChars;
        public int YCountChars;
        public int FillAttribute;
        public int Flags;
        public short ShowWindow;
        public short Reserved2;
        public IntPtr Reserved2Ptr;
        public IntPtr StdInput;
        public IntPtr StdOutput;
        public IntPtr StdError;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct ProcessInformation {
        public IntPtr Process;
        public IntPtr Thread;
        public int ProcessId;
        public int ThreadId;
    }

    [StructLayout(LayoutKind.Sequential)]
    private struct BasicAccountingInformation {
        public long TotalUserTime;
        public long TotalKernelTime;
        public long ThisPeriodTotalUserTime;
        public long ThisPeriodTotalKernelTime;
        public uint TotalPageFaultCount;
        public uint TotalProcesses;
        public uint ActiveProcesses;
        public uint TotalTerminatedProcesses;
    }

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateJobObject(IntPtr attributes, string name);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool AssignProcessToJobObject(IntPtr job, IntPtr process);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool CreatePipe(out IntPtr read, out IntPtr write, ref SecurityAttributes attributes, int size);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool SetHandleInformation(IntPtr handle, int mask, int flags);

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr CreateFile(
        string fileName, uint desiredAccess, uint shareMode, ref SecurityAttributes attributes,
        uint creationDisposition, uint flagsAndAttributes, IntPtr templateFile
    );

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern bool CreateProcess(
        string applicationName, StringBuilder commandLine,
        IntPtr processAttributes, IntPtr threadAttributes, bool inheritHandles,
        int creationFlags, IntPtr environment, string currentDirectory,
        ref StartupInfo startupInfo, out ProcessInformation processInformation
    );

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern uint ResumeThread(IntPtr thread);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool TerminateProcess(IntPtr process, uint exitCode);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool TerminateJobObject(IntPtr job, uint exitCode);

    [DllImport("kernel32.dll", SetLastError = true)]
    private static extern bool QueryInformationJobObject(
        IntPtr job,
        int informationClass,
        out BasicAccountingInformation information,
        uint informationLength,
        IntPtr returnLength
    );

    [DllImport("kernel32.dll", EntryPoint = "QueryInformationJobObject", SetLastError = true)]
    private static extern bool QueryInformationJobObjectRaw(
        IntPtr job, int informationClass, IntPtr information, uint informationLength, IntPtr returnLength);

    [DllImport("kernel32.dll")]
    private static extern bool CloseHandle(IntPtr handle);

    // Pids still assigned to the job (JobObjectBasicProcessIdList).
    public static int[] ProcessIds(IntPtr job) {
        if (job == IntPtr.Zero) return new int[0];
        const int capacity = 1024;
        int size = 8 + IntPtr.Size * capacity;
        IntPtr buffer = Marshal.AllocHGlobal(size);
        try {
            Marshal.WriteInt64(buffer, 0, 0);
            if (!QueryInformationJobObjectRaw(job, 3, buffer, (uint)size, IntPtr.Zero)) return new int[0];
            int count = Marshal.ReadInt32(buffer, 4);
            int[] ids = new int[count];
            for (int i = 0; i < count; i++) ids[i] = (int)Marshal.ReadIntPtr(buffer, 8 + i * IntPtr.Size).ToInt64();
            return ids;
        } finally { Marshal.FreeHGlobal(buffer); }
    }

    // CPU time plus I/O bytes the job's processes have spent so far
    // (JobObjectBasicAndIoAccountingInformation: 48-byte basic accounting,
    // then IO_COUNTERS). Any change is progress; -1 when unavailable.
    public static long Activity(IntPtr job) {
        if (job == IntPtr.Zero) return -1;
        const int size = 96;
        IntPtr buffer = Marshal.AllocHGlobal(size);
        try {
            if (!QueryInformationJobObjectRaw(job, 8, buffer, size, IntPtr.Zero)) return -1;
            return Marshal.ReadInt64(buffer, 0) + Marshal.ReadInt64(buffer, 8)
                + Marshal.ReadInt64(buffer, 72) + Marshal.ReadInt64(buffer, 80) + Marshal.ReadInt64(buffer, 88);
        } finally { Marshal.FreeHGlobal(buffer); }
    }

    public static StartedProcess StartAssigned(string executable, string arguments) {
        IntPtr job = IntPtr.Zero;
        IntPtr outRead = IntPtr.Zero, outWrite = IntPtr.Zero;
        IntPtr errRead = IntPtr.Zero, errWrite = IntPtr.Zero;
        IntPtr nullInput = new IntPtr(-1);
        ProcessInformation pi = new ProcessInformation();
        try {
            job = CreateJobObject(IntPtr.Zero, null);
            if (job == IntPtr.Zero) throw new InvalidOperationException("CreateJobObject failed");
            // Until Resume: should this script die while the child is still
            // suspended, closing the job kills it before it ran anything.
            if (!SetKillOnClose(job, true)) throw new InvalidOperationException("SetInformationJobObject failed");
            SecurityAttributes sa = new SecurityAttributes();
            sa.Length = Marshal.SizeOf(typeof(SecurityAttributes));
            sa.InheritHandle = true;
            if (!CreatePipe(out outRead, out outWrite, ref sa, 0) ||
                !CreatePipe(out errRead, out errWrite, ref sa, 0))
                throw new InvalidOperationException("CreatePipe failed");
            if (!SetHandleInformation(outRead, 1, 0) || !SetHandleInformation(errRead, 1, 0))
                throw new InvalidOperationException("SetHandleInformation failed");
            // Steps read NUL, never the hand-off console. A step that sees a
            // console asks its question into the captured stdout, where the
            // user cannot see it, and waits for an answer that never comes.
            nullInput = CreateFile("NUL", 0x80000000, 0x00000003, ref sa, 3, 0, IntPtr.Zero);
            if (nullInput == new IntPtr(-1))
                throw new InvalidOperationException("CreateFile(NUL) failed");

            StartupInfo si = new StartupInfo();
            si.Size = Marshal.SizeOf(typeof(StartupInfo));
            si.Flags = 0x00000100; // STARTF_USESTDHANDLES
            si.StdInput = nullInput;
            si.StdOutput = outWrite;
            si.StdError = errWrite;
            StringBuilder commandLine = new StringBuilder("\"" + executable + "\" " + arguments);
            if (!CreateProcess(executable, commandLine, IntPtr.Zero, IntPtr.Zero, true,
                    0x00000004 | 0x08000000, IntPtr.Zero, null, ref si, out pi))
                throw new InvalidOperationException("CreateProcess failed");
            if (!AssignProcessToJobObject(job, pi.Process)) {
                TerminateProcess(pi.Process, 1);
                throw new InvalidOperationException("AssignProcessToJobObject failed");
            }

            Process process = Process.GetProcessById(pi.ProcessId);
            // Force Process to open its own stable query handle before the raw
            // CreateProcess handle is closed; PS 5.1 otherwise reports a null
            // ExitCode after fast children have already disappeared.
            IntPtr stableProcessHandle = process.Handle;
            StreamReader stdout = new StreamReader(new FileStream(
                new SafeFileHandle(outRead, true), FileAccess.Read, 4096, false), Encoding.UTF8);
            StreamReader stderr = new StreamReader(new FileStream(
                new SafeFileHandle(errRead, true), FileAccess.Read, 4096, false), Encoding.UTF8);
            outRead = IntPtr.Zero;
            errRead = IntPtr.Zero;
            CloseHandle(outWrite); outWrite = IntPtr.Zero;
            CloseHandle(errWrite); errWrite = IntPtr.Zero;
            // Still suspended: the caller publishes the marker delegate, then Resume().
            StartedProcess started = new StartedProcess { Process = process, StandardOutput = stdout, StandardError = stderr, Job = job, Thread = pi.Thread };
            pi.Thread = IntPtr.Zero;
            return started;
        } catch {
            if (pi.Process != IntPtr.Zero) TerminateProcess(pi.Process, 1);
            if (job != IntPtr.Zero) CloseHandle(job);
            throw;
        } finally {
            if (pi.Thread != IntPtr.Zero) CloseHandle(pi.Thread);
            if (pi.Process != IntPtr.Zero) CloseHandle(pi.Process);
            if (outRead != IntPtr.Zero) CloseHandle(outRead);
            if (outWrite != IntPtr.Zero) CloseHandle(outWrite);
            if (errRead != IntPtr.Zero) CloseHandle(errRead);
            if (errWrite != IntPtr.Zero) CloseHandle(errWrite);
            if (nullInput != new IntPtr(-1)) CloseHandle(nullInput);
        }
    }

    public static void Resume(StartedProcess started) {
        // Detached services a successful step starts must outlive the job
        // handle: drop kill-on-close before the first instruction runs.
        try {
            if (!SetKillOnClose(started.Job, false)) {
                TerminateJobObject(started.Job, 1);
                throw new InvalidOperationException("SetInformationJobObject failed");
            }
            if (ResumeThread(started.Thread) == 0xffffffff) {
                TerminateJobObject(started.Job, 1);
                throw new InvalidOperationException("ResumeThread failed");
            }
        } finally {
            if (started.Thread != IntPtr.Zero) CloseHandle(started.Thread);
            started.Thread = IntPtr.Zero;
        }
    }

    public static bool TerminateAndWait(IntPtr job, uint exitCode, int timeoutMs) {
        if (job == IntPtr.Zero || !TerminateJobObject(job, exitCode)) return false;
        Stopwatch clock = Stopwatch.StartNew();
        BasicAccountingInformation information;
        do {
            if (!QueryInformationJobObject(
                    job, 1, out information,
                    (uint)Marshal.SizeOf(typeof(BasicAccountingInformation)),
                    IntPtr.Zero)) return false;
            if (information.ActiveProcesses == 0) return true;
            Thread.Sleep(50);
        } while (clock.ElapsedMilliseconds < timeoutMs);
        return false;
    }

    public static void Close(IntPtr job) {
        if (job != IntPtr.Zero) CloseHandle(job);
    }
}
'@
}

Initialize-HermesUpdateJob
