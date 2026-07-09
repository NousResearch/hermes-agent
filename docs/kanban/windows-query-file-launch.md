# Windows Query-File Launch

Use a query file for long goals, kanban worker prompts, and recovery prompts on
Windows. This avoids fragile quoting through PowerShell, `Start-Process`, `.cmd`
shims, multiline prompt strings, and paths that contain spaces.

## Foreground Resume Launch

```powershell
$repo = "C:\Users\Admin\AppData\Local\hermes\hermes-agent"
$sessionId = "20260704_211738_60f6ef"
$promptPath = "C:\Users\Admin\Documents\Hermes monitoring\runs\prompt.txt"
$args = @("-m", "hermes_cli.main", "chat", "--resume", $sessionId, "--model", "gpt-5.5", "--query-file", $promptPath)

Push-Location $repo
try {
    & .\venv\Scripts\python.exe @args
} finally {
    Pop-Location
}
```

## Background Launch With Startup Smoke

```powershell
$repo = "C:\Users\Admin\AppData\Local\hermes\hermes-agent"
$exe = Join-Path $repo "venv\Scripts\python.exe"
$sessionId = "20260704_211738_60f6ef"
$promptPath = "C:\Users\Admin\Documents\Hermes monitoring\runs\prompt.txt"
$runDir = Join-Path $env:TEMP ("hermes-run-" + (Get-Date -Format "yyyyMMdd-HHmmss"))
New-Item -ItemType Directory -Path $runDir -Force | Out-Null

$stdout = Join-Path $runDir "stdout.log"
$stderr = Join-Path $runDir "stderr.log"
$args = @("-m", "hermes_cli.main", "chat", "--resume", $sessionId, "--model", "gpt-5.5", "--query-file", $promptPath)

$proc = Start-Process `
    -FilePath $exe `
    -ArgumentList $args `
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
        -Pattern "usage: hermes|unrecognized arguments" `
        -Quiet
}
if (($proc.HasExited -and $proc.ExitCode -ne 0) -or $badLaunch) {
    Get-Content $logs -Tail 80
    throw "Hermes launch failed before creating a session DB message; treat this as a launcher/parser failure, not a Hermes agent failure."
}

"Hermes started: pid=$($proc.Id), logs=$runDir"
```

## Stdin Variant

```powershell
$repo = "C:\Users\Admin\AppData\Local\hermes\hermes-agent"
$exe = Join-Path $repo "venv\Scripts\python.exe"
$promptPath = "C:\Users\Admin\Documents\Hermes monitoring\runs\prompt.txt"

Get-Content -Raw $promptPath | & $exe -m hermes_cli.main chat --stdin-query --cli
```

For local recovery slash commands, prefer the explicit slash path:

```powershell
& $exe -m hermes_cli.main chat --slash "/goal resume" --cli
```
