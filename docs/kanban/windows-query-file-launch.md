# Windows Query-File Launch

Use a query file for long goals, kanban worker prompts, and recovery prompts on
Windows. This avoids fragile quoting through PowerShell, `Start-Process`, `.cmd`
shims, and paths that contain spaces.

## Foreground Launch

```powershell
$repo = "C:\Users\Admin\AppData\Local\hermes\hermes-agent"
$exe = Join-Path $repo "venv\Scripts\python.exe"
$goalFile = "C:\Users\Admin\Documents\Hermes monitoring\goals\next-goal.txt"

$moduleArgs = @(
    "-m", "hermes_cli.main",
    "chat",
    "--query-file", $goalFile,
    "--cli"
)

Push-Location $repo
try {
    & $exe @moduleArgs
} finally {
    Pop-Location
}
```

## Background Launch With Startup Smoke

```powershell
$repo = "C:\Users\Admin\AppData\Local\hermes\hermes-agent"
$exe = Join-Path $repo "venv\Scripts\python.exe"
$goalFile = "C:\Users\Admin\Documents\Hermes monitoring\goals\next-goal.txt"
$runDir = Join-Path $env:TEMP ("hermes-run-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
New-Item -ItemType Directory -Path $runDir -Force | Out-Null

$stdout = Join-Path $runDir "stdout.log"
$stderr = Join-Path $runDir "stderr.log"
$moduleArgs = @(
    "-m", "hermes_cli.main",
    "chat",
    "--query-file", $goalFile,
    "--cli"
)

$proc = Start-Process `
    -FilePath $exe `
    -ArgumentList $moduleArgs `
    -WorkingDirectory $repo `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr `
    -WindowStyle Hidden `
    -PassThru

Start-Sleep -Seconds 2
$logs = @($stdout, $stderr) | Where-Object { Test-Path $_ }
$badLaunch = $false
if ($logs.Count -gt 0) {
    $badLaunch = Select-String `
        -Path $logs `
        -Pattern "usage: hermes|unrecognized arguments|Traceback" `
        -Quiet
}
if (($proc.HasExited -and $proc.ExitCode -ne 0) -or $badLaunch) {
    Get-Content $logs -Tail 80
    throw "Hermes exited or printed a startup failure during launch smoke."
}

"Hermes started: pid=$($proc.Id), logs=$runDir"
```

## Stdin Variant

```powershell
$repo = "C:\Users\Admin\AppData\Local\hermes\hermes-agent"
$exe = Join-Path $repo "venv\Scripts\python.exe"
$goalFile = "C:\Users\Admin\Documents\Hermes monitoring\goals\next-goal.txt"

Get-Content -Raw $goalFile | & $exe -m hermes_cli.main chat --stdin-query --cli
```

For local recovery slash commands, prefer the explicit slash path:

```powershell
& $exe -m hermes_cli.main chat --slash "/goal resume" --cli
```
