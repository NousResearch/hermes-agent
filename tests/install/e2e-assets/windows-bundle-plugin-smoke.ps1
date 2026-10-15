# Install and enable a plugin with Python dependencies through the installed
# package's CLI execution alias, as a basic (non-admin) user. Dot-sourced by
# windows-bundle-smoke.ps1 after the chat smoke; throws on failure.
#
# GitHub runners are administrators, and Administrators may execute files in a
# package directory without package identity. A normal user may not: a process
# outside the package (a uv venv's Scripts\python.exe redirector) cannot start
# the packaged interpreter and exits 101 (#135236). The Safer NORMALUSER token
# below turns the admin group deny-only, so the runner sees what a user sees.

Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;
using System.Text;

public static class HermesBasicUser {
    [DllImport("advapi32.dll", SetLastError = true)]
    static extern bool SaferCreateLevel(uint scope, uint level, uint flags, out IntPtr handle, IntPtr reserved);
    [DllImport("advapi32.dll", SetLastError = true)]
    static extern bool SaferComputeTokenFromLevel(IntPtr level, IntPtr inToken, out IntPtr outToken, uint flags, IntPtr reserved);
    [DllImport("advapi32.dll")]
    static extern bool SaferCloseLevel(IntPtr handle);
    [DllImport("advapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    static extern bool ConvertStringSidToSidW(string sid, out IntPtr psid);
    [DllImport("advapi32.dll")]
    static extern uint GetLengthSid(IntPtr psid);
    [DllImport("advapi32.dll", SetLastError = true)]
    static extern bool SetTokenInformation(IntPtr token, int infoClass, IntPtr info, uint length);
    [DllImport("advapi32.dll", SetLastError = true, CharSet = CharSet.Unicode)]
    static extern bool CreateProcessAsUserW(IntPtr token, string application, StringBuilder commandLine,
        IntPtr processAttributes, IntPtr threadAttributes, bool inheritHandles, uint flags,
        IntPtr environment, string directory, ref StartupInfo startup, out ProcessInformation info);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern uint WaitForSingleObject(IntPtr handle, uint milliseconds);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool GetExitCodeProcess(IntPtr handle, out uint code);
    [DllImport("kernel32.dll", SetLastError = true)]
    static extern bool TerminateProcess(IntPtr handle, uint code);
    [DllImport("kernel32.dll")]
    static extern bool CloseHandle(IntPtr handle);

    [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
    public struct StartupInfo {
        public int cb; public string reserved; public string desktop; public string title;
        public int x, y, xSize, ySize, xChars, yChars, fill, flags;
        public short show, reserved2Size; public IntPtr reserved2, stdIn, stdOut, stdErr;
    }
    [StructLayout(LayoutKind.Sequential)]
    public struct ProcessInformation { public IntPtr process, thread; public int processId, threadId; }
    [StructLayout(LayoutKind.Sequential)]
    struct SidAndAttributes { public IntPtr sid; public uint attributes; }

    // Run commandLine under a Safer NORMALUSER token at medium integrity; returns its exit code.
    public static int Run(string commandLine, string directory, uint timeoutMs) {
        IntPtr level, token;
        if (!SaferCreateLevel(2 /* USER */, 0x20000 /* NORMALUSER */, 1 /* OPEN */, out level, IntPtr.Zero))
            throw new Win32Exception();
        try {
            if (!SaferComputeTokenFromLevel(level, IntPtr.Zero, out token, 0, IntPtr.Zero)) throw new Win32Exception();
        } finally { SaferCloseLevel(level); }
        IntPtr medium;
        if (!ConvertStringSidToSidW("S-1-16-8192", out medium)) throw new Win32Exception();
        SidAndAttributes label = new SidAndAttributes { sid = medium, attributes = 0x20 /* SE_GROUP_INTEGRITY */ };
        int size = Marshal.SizeOf(label);
        IntPtr buffer = Marshal.AllocHGlobal(size);
        Marshal.StructureToPtr(label, buffer, false);
        if (!SetTokenInformation(token, 25 /* TokenIntegrityLevel */, buffer, (uint)size + GetLengthSid(medium)))
            throw new Win32Exception();
        StartupInfo startup = new StartupInfo();
        startup.cb = Marshal.SizeOf(startup);
        ProcessInformation info;
        if (!CreateProcessAsUserW(token, null, new StringBuilder(commandLine), IntPtr.Zero, IntPtr.Zero, false,
                0x08000000 /* CREATE_NO_WINDOW */, IntPtr.Zero, directory, ref startup, out info))
            throw new Win32Exception();
        try {
            if (WaitForSingleObject(info.process, timeoutMs) != 0) {
                TerminateProcess(info.process, 124);
                throw new TimeoutException("basic-user command timed out: " + commandLine);
            }
            uint code;
            if (!GetExitCodeProcess(info.process, out code)) throw new Win32Exception();
            return (int)code;
        } finally { CloseHandle(info.thread); CloseHandle(info.process); CloseHandle(token); }
    }
}
'@

# The batch starts from this process's environment. The CI driver exports its own PM home
# (HERMES_HOME, HERMES_RUNTIME_DIR, HERMES_PYTHON); a CLI that inherits them builds on the
# driver's Python instead of the package's, and the bug never shows. Start clean, as a user does.
function Get-UserEnvironmentReset {
    $lines = @(Get-ChildItem env: | Where-Object { $_.Name -match '^(HERMES_|UV_|PYTHON|VIRTUAL_ENV|CONDA)' } |
        ForEach-Object { 'set "' + $_.Name + '="' })
    $driverRoots = @($env:HERMES_HOME, $env:HERMES_RUNTIME_DIR) | Where-Object { $_ }
    $path = @($env:PATH -split ';' | Where-Object { $entry = $_; $_ -and -not @($driverRoots | Where-Object {
        $entry.StartsWith($_, [StringComparison]::OrdinalIgnoreCase) }).Count })
    return $lines + @('set "PATH=' + ($path -join ';') + '"')
}

function Invoke-BasicUserBatch([string]$Name, [string[]]$Lines, [string]$Directory, [int]$TimeoutMinutes) {
    $batch = Join-Path $Directory "$Name.cmd"
    Set-Content -LiteralPath $batch -Encoding ASCII -Value (@('@echo off') + (Get-UserEnvironmentReset) + $Lines)
    return [HermesBasicUser]::Run("cmd.exe /d /c `"$batch`"", $Directory, [uint32]($TimeoutMinutes * 60000))
}

function Test-BundlePluginInstall([string]$Root, [string]$Out) {
    $manifest = Get-Content -Raw -LiteralPath (Join-Path $Root 'manifest.json') | ConvertFrom-Json
    $alias = Join-Path $env:LOCALAPPDATA ('Microsoft\WindowsApps\' + (Split-Path -Leaf $manifest.runtime.commands.hermes))
    if (-not (Test-Path -LiteralPath $alias)) { throw "Package CLI execution alias missing: $alias" }
    $python = Join-Path $Root $manifest.runtime.storePython
    # Under the profile, which the basic-user token can write; the runner's temp may be admin-only.
    # The CLI uses its default home (%LOCALAPPDATA%\hermes), as an installed user's does.
    $scratch = Join-Path $env:LOCALAPPDATA ('hermes-plugin-smoke-' + [Guid]::NewGuid().ToString('N'))
    New-Item -ItemType Directory -Path $scratch | Out-Null
    try {
        # The token must reproduce a user's package ACL, or a green below proves nothing.
        $denied = Invoke-BasicUserBatch 'probe' @("`"$python`" -c `"print(1)`" > probe.log 2>&1") $scratch 2
        if ($denied -eq 0) { throw 'Basic-user token could start the packaged Python directly; it does not reproduce a user' }

        $plugin = Join-Path $scratch 'msix-smoke-plugin'
        New-Item -ItemType Directory -Path $plugin | Out-Null
        Set-Content -LiteralPath (Join-Path $plugin 'plugin.yaml') -Encoding ASCII -Value @(
            'name: msix-smoke-plugin', 'version: 1.0.0', 'description: MSIX plugin dependency smoke',
            'python_dependencies:', '  - pyjokes==0.8.3')
        Set-Content -LiteralPath (Join-Path $plugin '__init__.py') -Encoding ASCII -Value @(
            'import pyjokes  # noqa: F401', '', 'def register(ctx):', '    pass')
        & git -C $plugin init -q
        & git -C $plugin -c user.name=smoke -c user.email=smoke@invalid add -A
        & git -C $plugin -c user.name=smoke -c user.email=smoke@invalid commit -q -m smoke
        if ($LASTEXITCODE -ne 0) { throw 'Could not commit the smoke plugin' }
        $url = 'file:///' + ($plugin -replace '\\', '/')

        $code = Invoke-BasicUserBatch 'install' @(
            "`"$alias`" plugins install `"$url`" --enable --yes-deps > install.log 2>&1",
            "`"$alias`" plugins list --json > list.json 2> list.err") $scratch 25
        foreach ($name in 'probe.log', 'install.log', 'list.json', 'list.err') {
            $file = Join-Path $scratch $name
            if (Test-Path -LiteralPath $file) { Copy-Item -LiteralPath $file -Destination (Join-Path $Out "plugin-smoke-$name") }
        }
        $listing = if (Test-Path -LiteralPath (Join-Path $scratch 'list.json')) { Get-Content -Raw -LiteralPath (Join-Path $scratch 'list.json') } else { '' }
        # Startup warnings may precede the JSON array; it is the block from a line starting '['.
        $json = [regex]::Match($listing, '(?ms)^\[.*^\]')
        $rows = if ($json.Success) { @($json.Value | ConvertFrom-Json) } else { @() }
        $row = @($rows | Where-Object { $_.name -ceq 'msix-smoke-plugin' })
        $installLog = if (Test-Path -LiteralPath (Join-Path $scratch 'install.log')) { Get-Content -Raw -LiteralPath (Join-Path $scratch 'install.log') } else { '' }
        foreach ($driverRoot in @($env:HERMES_HOME, $env:HERMES_RUNTIME_DIR) | Where-Object { $_ }) {
            if ($installLog.IndexOf($driverRoot, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
                throw "Plugin install used the CI driver's PM home ($driverRoot), not the package's"
            }
        }
        if ($row.Count -ne 1 -or $row[0].status -cne 'enabled') {
            $log = Join-Path $scratch 'install.log'
            $tail = if (Test-Path -LiteralPath $log) { (Get-Content -LiteralPath $log -Tail 40) -join "`n" } else { '(no install log)' }
            throw "Plugin with Python dependencies did not enable as a basic user (batch exit $code):`n$tail"
        }
    } finally {
        Remove-Item -LiteralPath $scratch -Recurse -Force -ErrorAction SilentlyContinue
    }
}
