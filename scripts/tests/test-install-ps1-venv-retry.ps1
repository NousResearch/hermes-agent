param(
    [Parameter(Mandatory=$true)][string]$Root,
    [string]$Installer,
    [string]$Stage,
    [string]$Mode = 'ok',
    [switch]$BuildTool,
    [switch]$NoVenv
)

$ErrorActionPreference = 'Stop'
$rootPath = [IO.Path]::GetFullPath($Root).TrimEnd('\', '/')
$env:HERMES_INSTALL_FIXTURE_ROOT = $rootPath
$env:HERMES_INSTALL_FIXTURE_MODE = $Mode

if ($BuildTool) {
    # A real executable exercises ProcessStartInfo and native exit-code plumbing.
    # Compile once with Windows PowerShell5.1; the same executable is invoked by
    # both host versions. No Python imports or package resolutions happen here:
    # recognized native operations return explicit synthetic responses.
    $source = @'
using System;
using System.IO;
using System.Diagnostics;
public class InstallerFixture {
    private static void Record(string root, string operation) {
        File.AppendAllText(Path.Combine(root, "native-events.txt"), "operation:" + operation + Environment.NewLine);
    }
    private static void Validation(string root, string install, string operation) {
        bool pending = File.Exists(Path.Combine(install, "venv.pending-backup"));
        File.AppendAllText(Path.Combine(root, "validation-events.txt"), operation + ":pending=" + pending + Environment.NewLine);
    }
    public static int Main(string[] args) {
        string root = Environment.GetEnvironmentVariable("HERMES_INSTALL_FIXTURE_ROOT");
        string mode = Environment.GetEnvironmentVariable("HERMES_INSTALL_FIXTURE_MODE");
        string install = Path.Combine(root, "home", "hermes-agent");
        string self = Process.GetCurrentProcess().MainModule.FileName;
        string joined = String.Join(" ", args);
        File.AppendAllText(Path.Combine(root, "native-events.txt"), Path.GetFileName(self) + " " + joined + Environment.NewLine);
        if (Path.GetFileName(self).Equals("python.exe", StringComparison.OrdinalIgnoreCase)) {
            if (args.Length == 2 && args[0] == "-c" && args[1] == "import dotenv, openai, rich, prompt_toolkit") {
                Record(root, "baseline-import");
                Validation(root, install, "baseline");
                if (mode == "import-fail") return 8;
                if (mode == "final-import-fail" && File.ReadAllText(Path.Combine(root, "validation-events.txt")).Split(new string[] { "baseline:pending=" }, StringSplitOptions.None).Length > 2) return 8;
                File.WriteAllText(Path.Combine(install, "venv", "generation.txt"), "VALIDATED_REPLACEMENT");
                return 0;
            }
            if (args.Length == 2 && args[0] == "-c" && args[1].Contains("data['project']['optional-dependencies']['all']")) {
                Record(root, "toml-extras");
                Console.WriteLine("web,browser");
                return 0;
            }
            if (args.Length == 2 && args[0] == "-c" && args[1].Contains(".get('scripts', {})")) {
                Record(root, "toml-entrypoints");
                Console.WriteLine("hermes");
                return 0;
            }
            if (args.Length == 2 && args[0] == "-c" && args[1] == "import fastapi, uvicorn") {
                Record(root, "dashboard-import");
                return mode == "optional-web-fail" ? 10 : 0;
            }
            if (args.Length == 3 && args[0] == "-m" && args[1] == "py_compile" && Path.GetFileName(args[2]) == "web_server.py") {
                Record(root, "dashboard-syntax");
                Validation(root, install, "dashboard-syntax");
                return mode == "dashboard-syntax-fail" ? 11 : 0;
            }
            Console.Error.WriteLine("Unexpected synthetic Python operation: " + joined);
            return 98;
        }
        if (args.Length >= 2 && args[0] == "python" && args[1] == "find") {
            Record(root, "uv-python-find");
            if (mode == "find-fail") return 7;
            Console.WriteLine(Path.Combine(install, ".hermes-runtime", "python", "fixture", "python.exe"));
            return 0;
        }
        if (args.Length > 0 && args[0] == "venv") {
            Record(root, "uv-venv");
            string scripts = Path.Combine(install, "venv", "Scripts");
            Directory.CreateDirectory(scripts);
            if (mode != "venv-no-interpreter") File.Copy(self, Path.Combine(scripts, "python.exe"), true);
            File.Copy(self, Path.Combine(scripts, "hermes.exe"), true);
            File.WriteAllText(Path.Combine(install, "venv", "generation.txt"), "PARTIAL_REPLACEMENT");
            return mode == "venv-fail" ? 9 : 0;
        }
        if (args.Length > 0 && (args[0] == "pip" || args[0] == "sync")) {
            bool webRepair = args.Length == 4 && args[0] == "pip" && args[1] == "install" && args[2] == "-e" && args[3] == ".[web]";
            Record(root, webRepair ? "uv-web-repair" : args[0] == "sync" ? "uv-sync" : "uv-pip-install");
            if (webRepair && mode == "optional-web-fail") return 12;
            if (mode == "deps-fail") File.WriteAllText(Path.Combine(install, "venv", "generation.txt"), "DAMAGED_BY_FAILED_DEPENDENCY_INSTALL");
            return mode == "deps-fail" || mode == "rollback-park-fail" || mode == "rollback-restore-fail" || mode == "rollback-marker-clear-fail" || mode == "crash-after-restore" || mode == "crash-after-rollback-clear" ? 7 : 0;
        }
        throw new Exception("Unexpected provisioning command: " + joined);
    }
}
'@
    Add-Type -TypeDefinition $source -OutputAssembly (Join-Path $rootPath 'fixture-uv.exe') -OutputType ConsoleApplication
    exit 0
}

# These are external boundaries, not installer implementation overrides. Never
# allow the full stage invocation to enumerate or stop the user's real services.
function global:taskkill {
    [IO.File]::AppendAllText((Join-Path $rootPath 'intercepted-boundaries.txt'), "taskkill`n")
    $global:LASTEXITCODE = 0
}
function global:schtasks {
    [IO.File]::AppendAllText((Join-Path $rootPath 'intercepted-boundaries.txt'), "schtasks`n")
    $global:LASTEXITCODE = 0
}
function global:Get-CimInstance {
    [IO.File]::AppendAllText((Join-Path $rootPath 'intercepted-boundaries.txt'), "Get-CimInstance`n")
    @()
}
function global:Invoke-WebRequest { throw 'Network access is forbidden in installer transaction tests' }
function global:Invoke-RestMethod { throw 'Network access is forbidden in installer transaction tests' }

function Get-FixturePath([string]$Candidate) {
    if (-not [IO.Path]::IsPathRooted($Candidate)) { $Candidate = Join-Path (Get-Location).Path $Candidate }
    $absolute = [IO.Path]::GetFullPath($Candidate)
    if ($absolute -ine $rootPath -and -not $absolute.StartsWith($rootPath + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Fixture filesystem operation escaped its root: $absolute"
    }
    return $absolute
}

function global:Rename-Item {
    [CmdletBinding()] param([string]$Path, [string]$LiteralPath, [Parameter(Mandatory=$true)][string]$NewName, [switch]$Force)
    $source = Get-FixturePath $(if ($LiteralPath) { $LiteralPath } else { $Path })
    $leaf = [IO.Path]::GetFileName($source)
    $parkingOriginal = $leaf -eq 'venv' -and $NewName -like 'venv.stale.*'
    if ($Mode -eq 'crash-before-park' -and $parkingOriginal) { [Environment]::Exit(91) }
    if ($Mode -eq 'rollback-park-fail' -and $leaf -eq 'venv' -and $NewName -like 'venv.failed.*') { throw 'fixture: replacement parking denied' }
    if ($Mode -eq 'rollback-restore-fail' -and $leaf -like 'venv.stale.*' -and $NewName -eq 'venv') { throw 'fixture: backup restore denied' }
    if ($Mode -eq 'marker-write-fail' -and $NewName -eq 'venv.pending-backup') { throw 'fixture: marker publication denied' }
    Microsoft.PowerShell.Management\Rename-Item @PSBoundParameters
    if ($Mode -eq 'crash-after-park' -and $parkingOriginal) { [Environment]::Exit(92) }
    if ($Mode -eq 'crash-after-restore' -and $leaf -like 'venv.stale.*' -and $NewName -eq 'venv') { [Environment]::Exit(-1) }
}

function global:Move-Item {
    [CmdletBinding()] param([string]$Path, [string]$LiteralPath, [string]$Destination, [switch]$Force)
    [void](Get-FixturePath $(if ($LiteralPath) { $LiteralPath } else { $Path }))
    [void](Get-FixturePath $Destination)
    if ($Mode -eq 'marker-write-fail' -and [IO.Path]::GetFileName($Destination) -eq 'venv.pending-backup') { throw 'fixture: marker publication denied' }
    Microsoft.PowerShell.Management\Move-Item @PSBoundParameters
}

function global:Set-Content {
    [CmdletBinding()] param([string]$Path, [string]$LiteralPath, $Value, [string]$Encoding, [switch]$NoNewline, [switch]$Force)
    $target = Get-FixturePath $(if ($LiteralPath) { $LiteralPath } else { $Path })
    if ($Mode -eq 'marker-write-fail' -and [IO.Path]::GetFileName($target) -eq 'venv.pending-backup') { throw 'fixture: marker publication denied' }
    Microsoft.PowerShell.Management\Set-Content @PSBoundParameters
}

function global:Remove-Item {
    [CmdletBinding()] param([Parameter(Position=0)][string[]]$Path, [string[]]$LiteralPath, [switch]$Recurse, [switch]$Force)
    foreach ($candidate in $(if ($LiteralPath) { $LiteralPath } else { $Path })) {
        if ($candidate -like 'Env:*') { continue }
        $target = Get-FixturePath $candidate
        if ($Mode -in @('commit-marker-clear-fail', 'rollback-marker-clear-fail') -and [IO.Path]::GetFileName($target) -eq 'venv.pending-backup') {
            throw 'fixture: marker clearing denied'
        }
        if ($Mode -eq 'crash-before-commit-cleanup' -and [IO.Path]::GetFileName($target) -like 'venv.stale.*') {
            [Environment]::Exit(93)
        }
    }
    Microsoft.PowerShell.Management\Remove-Item @PSBoundParameters
    if ($Mode -eq 'crash-after-rollback-clear' -and [IO.Path]::GetFileName($target) -eq 'venv.pending-backup') {
        [Environment]::Exit(-1)
    }
}

$homePath = Join-Path $rootPath 'home'
$installPath = Join-Path $homePath 'hermes-agent'
$env:HERMES_HOME = $homePath
# The clean CI runner does not forward OS. This fixture only runs on Windows;
# preserve the native branch while intercepting every global process boundary.
$env:OS = 'Windows_NT'
$env:PATHEXT = '.COM;.EXE;.BAT;.CMD'
$env:TEMP = $rootPath
$env:TMP = $rootPath
$env:LOCALAPPDATA = Join-Path $rootPath 'localappdata'
$env:APPDATA = Join-Path $rootPath 'appdata'
$env:USERPROFILE = Join-Path $rootPath 'profile'
[IO.File]::AppendAllText((Join-Path $rootPath 'stage-processes.txt'), "$PID $Stage $Mode`n")

# Invoke the full public stage protocol. The installer remains self-contained;
# tests neither extract its functions nor assert on its source text.
& $Installer -HermesHome $homePath -InstallDir $installPath -Stage $Stage -Json -NonInteractive -NoVenv:$NoVenv
exit $LASTEXITCODE
