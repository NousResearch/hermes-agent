<#
  Smoke test for hermes-update-rehearsal.ps1 (this directory).
  Builds a synthetic install in a temp tree and drives pre/post/status for real.
#>
[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'

$Script = Join-Path $PSScriptRoot 'hermes-update-rehearsal.ps1'
# A python is needed for the sqlite assertions. Prefer an explicit checkout, then
# a hermes-agent checkout next to this kit (the usual layout while developing).
$Checkout = $env:HERMES_REHEARSAL_CHECKOUT
if (-not $Checkout) {
  $sibling = Join-Path (Split-Path -Parent $PSScriptRoot) 'hermes-agent'
  if (Test-Path -LiteralPath $sibling) { $Checkout = $sibling }
}
if (-not (Test-Path -LiteralPath $Script)) { throw "missing $Script" }

# The kit runs under whichever host runs this smoke: `pwsh -File` tests pwsh,
# `powershell -File` tests Windows PowerShell.
$PSExe = (Get-Process -Id $PID).Path
Write-Host "host: $PSExe"

$errors = $null
[void][System.Management.Automation.Language.Parser]::ParseFile($Script, [ref]$null, [ref]$errors)
if ($errors -and $errors.Count) { Write-Host 'SYNTAX ERRORS:'; $errors | ForEach-Object { Write-Host "  $_" }; exit 1 }
Write-Host 'syntax OK'

$RealUserPath = [Environment]::GetEnvironmentVariable('Path', 'User')

function Find-RealPython {
  $cands = @(
    (Join-Path $env:USERPROFILE '.hermes\hermes-agent\venv\Scripts\python.exe')
  )
  if ($Checkout) {
    $cands += (Join-Path $Checkout '.venv\Scripts\python.exe')
    $cands += (Join-Path $Checkout 'venv\Scripts\python.exe')
  }
  $wa = Join-Path $env:ProgramFiles 'WindowsApps'
  if (Test-Path -LiteralPath $wa) {
    $cands += Get-ChildItem -LiteralPath $wa -Directory -ErrorAction SilentlyContinue |
      Where-Object { $_.Name -like 'NousResearch.Hermes*' } |
      ForEach-Object {
        Get-ChildItem -LiteralPath (Join-Path $_.FullName 'app\resources\agent-payload\tools') -Directory -ErrorAction SilentlyContinue |
          Where-Object { $_.Name -like 'python-*' } | ForEach-Object { Join-Path $_.FullName 'python.exe' }
      }
  }
  foreach ($root in @($env:LOCALAPPDATA + '\Programs\Python', 'C:\', $env:ProgramFiles)) {
    $cands += Get-ChildItem -Path (Join-Path $root 'Python*') -Directory -ErrorAction SilentlyContinue |
      ForEach-Object { Join-Path $_.FullName 'python.exe' }
    $cands += Get-ChildItem -Path (Join-Path $root 'python*') -Directory -ErrorAction SilentlyContinue |
      ForEach-Object { Join-Path $_.FullName 'python.exe' }
  }
  # PATH and the py launcher too: that is where a CI runner keeps python.
  $cands += (Get-Command python.exe -All -ErrorAction SilentlyContinue | ForEach-Object { $_.Source })
  $pyLauncher = (Get-Command py.exe -ErrorAction SilentlyContinue).Source
  if ($pyLauncher) {
    try {
      $fromLauncher = (& $pyLauncher -3 -c 'import sys;print(sys.executable)' 2>$null | Out-String).Trim()
      if ($fromLauncher) { $cands += $fromLauncher }
    }
    catch { }
  }
  foreach ($c in $cands) {
    if (-not $c) { continue }
    if ($c -like '*\Microsoft\WindowsApps\*') { continue }   # Store alias stub
    if (-not (Test-Path -LiteralPath $c)) { continue }
    try { & $c -c 'import sqlite3' *> $null; if ($LASTEXITCODE -eq 0) { return $c } }
    catch { }
  }
  return $null
}

function Find-RealGit {
  foreach ($c in @("$env:ProgramFiles\Git\cmd\git.exe", "$env:ProgramFiles\Git\bin\git.exe",
      "${env:ProgramFiles(x86)}\Git\cmd\git.exe")) {
    if ($c -and (Test-Path -LiteralPath $c)) {
      try { & $c --version *> $null; if ($LASTEXITCODE -eq 0) { return $c } } catch { }
    }
  }
  return $null
}

$RealPython = Find-RealPython
if (-not $RealPython) { throw 'no usable python.exe found; the db-count assertions need one' }
$env:PATH = (Split-Path -Parent $RealPython) + ';' + $env:PATH
Write-Host "python: $RealPython"
$RealGit = Find-RealGit
if ($RealGit) { $env:PATH = (Split-Path -Parent $RealGit) + ';' + $env:PATH; Write-Host "git: $RealGit" }

$id = [Security.Principal.WindowsIdentity]::GetCurrent()
if (-not (New-Object Security.Principal.WindowsPrincipal $id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
  throw 'run this smoke test elevated: pre/post take a Volume Shadow Copy snapshot'
}

$Root = Join-Path $env:LOCALAPPDATA ("Temp\rehearsal-ps-" + [guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Force -Path $Root | Out-Null
$env:USERPROFILE = Join-Path $Root 'home'
New-Item -ItemType Directory -Force -Path $env:USERPROFILE | Out-Null
$env:GIT_CONFIG_GLOBAL = Join-Path $Root 'gitconfig-test'
Set-Content -LiteralPath $env:GIT_CONFIG_GLOBAL -Value @()
$env:HERMES_HOME = Join-Path $env:USERPROFILE '.hermes'
$env:HERMES_DESKTOP_USER_DATA_DIR = Join-Path $Root 'electron-user-data'
Remove-Item Env:\HERMES_DATA_DIR_SUFFIX -ErrorAction SilentlyContinue

$H = $env:HERMES_HOME
$Install = Join-Path $H 'hermes-agent'
$Backups = Join-Path $Root 'backups'
Write-Host "fixture under $Root"

$pass = 0; $fail = 0
function Check { param([string]$Msg, [bool]$Ok) if ($Ok) { Write-Host "  PASS $Msg"; $script:pass++ } else { Write-Host "  FAIL $Msg"; $script:fail++ } }

# Every path, file size and content hash under a tree, so post can be checked
# for exactness against the tree as it was before pre.
function Get-TreeListing {
  param([string]$Root)
  if (-not (Test-Path -LiteralPath $Root)) { return '' }
  $lines = New-Object System.Collections.Generic.List[string]
  $stack = New-Object System.Collections.Stack
  $stack.Push($Root)
  while ($stack.Count -gt 0) {
    $cur = $stack.Pop()
    foreach ($it in @(Get-ChildItem -LiteralPath $cur -Force -ErrorAction SilentlyContinue)) {
      $rel = $it.FullName.Substring($Root.Length).TrimStart('\')
      if ($it.PSIsContainer) {
        if ($it.Attributes -band [IO.FileAttributes]::ReparsePoint) { $lines.Add("link`t$rel`t$($it.Target)") }
        else { $lines.Add("dir`t$rel"); $stack.Push($it.FullName) }
      }
      else {
        $lines.Add("file`t$rel`t$($it.Length)`t$((Get-FileHash -LiteralPath $it.FullName -Algorithm SHA256).Hash)")
      }
    }
  }
  return (($lines | Sort-Object) -join "`n")
}

function Invoke-Rehearsal {
  param([string[]]$Arguments, [switch]$AllowFailure)
  # PS 5.1 turns a child's stderr into a TERMINATING error record when EAP is
  # Stop and stderr is merged; relax it just for the child call.
  $prev = $ErrorActionPreference
  $ErrorActionPreference = 'Continue'
  try {
    $out = & $PSExe -NoProfile -ExecutionPolicy Bypass -File $Script @Arguments 2>&1
    $code = $LASTEXITCODE
  }
  finally { $ErrorActionPreference = $prev }
  if (-not $AllowFailure -and $code -ne 0) {
    Write-Host '--- child output ---'
    Write-Host (($out | Out-String).Trim())
    Write-Host '--- end child output ---'
  }
  return [pscustomobject]@{ Code = $code; Out = ($out | Out-String) }
}

function New-Fixture {
  New-Item -ItemType Directory -Force -Path `
    (Join-Path $H 'plugins\mnemosyne-wrapper'), (Join-Path $H 'memories'),
    (Join-Path $H 'skills\foo'), (Join-Path $H 'cron'), (Join-Path $H 'logs'),
    (Join-Path $H 'photon\sidecar\node_modules'), (Join-Path $Root 'external-mnemosyne') | Out-Null
  Set-Content -LiteralPath (Join-Path $H 'plugins\mnemosyne-wrapper\plugin.yaml') -Value 'name: mnemosyne-wrapper'
  Set-Content -LiteralPath (Join-Path $H 'plugins\mnemosyne-wrapper\mnemosyne-wrapper.json') -Value '{"wrapper":true}'
  Set-Content -LiteralPath (Join-Path $Root 'external-mnemosyne\witness.txt') -Value 'witness'
  Set-Content -LiteralPath (Join-Path $H 'config.yaml') -Value 'timezone: utc'
  Set-Content -LiteralPath (Join-Path $H '.env') -Value 'NOUS_API_KEY=xxx'
  Set-Content -LiteralPath (Join-Path $H 'auth.json') -Value '{"tokens":{}}'
  Set-Content -LiteralPath (Join-Path $H 'memories\note.md') -Value 'recall'
  Set-Content -LiteralPath (Join-Path $H 'cron\jobs.json') -Value 'jobs'
  Set-Content -LiteralPath (Join-Path $H 'photon\sidecar\index.mjs') -Value 'console.log(1)'
  Set-Content -LiteralPath (Join-Path $H 'photon\sidecar\package.json') -Value '{"name":"sidecar"}'
  Set-Content -LiteralPath (Join-Path $H 'photon\sidecar\package-lock.json') -Value '{"lockfileVersion":3}'
  Set-Content -LiteralPath (Join-Path $H 'photon\sidecar\node_modules\.package-lock.json') -Value '{"lockfileVersion":3}'

  # SQL goes through a temp script FILE: a `-c` argument carrying quotes gets
  # reshaped by PowerShell's native-argument handling and fails to parse.
  $sqlPy = Join-Path $Root 'exec_sql.py'
  @'
import sqlite3, sys
con = sqlite3.connect(sys.argv[1])
con.executescript(sys.argv[2])
con.commit()
con.close()
'@ | Set-Content -LiteralPath $sqlPy -Encoding ASCII
  $script:SqlPy = $sqlPy
  $script:DbPath = Join-Path $H 'state.db'
  & $RealPython $sqlPy $script:DbPath "CREATE TABLE sessions(id TEXT); CREATE TABLE messages(id TEXT); INSERT INTO sessions VALUES('s1'); INSERT INTO sessions VALUES('s2');"
  if ($LASTEXITCODE -ne 0) { throw 'state.db fixture failed' }

  New-Item -ItemType Directory -Force -Path $Install | Out-Null
  & git -C $Install init -q -b main
  & git -C $Install config user.email t@example.com
  & git -C $Install config user.name test
  Set-Content -LiteralPath (Join-Path $Install 'module_name.py') -Value 'print(1)'
  & git -C $Install add -A | Out-Null
  & git -C $Install -c commit.gpgsign=false commit -qm initial | Out-Null
  & git -C $Install remote add origin https://github.com/NousResearch/hermes-agent.git
  New-Item -ItemType Directory -Force -Path (Join-Path $Install '.hermes\bin'), (Join-Path $Install '.hermes-runtime\python') | Out-Null
  Set-Content -LiteralPath (Join-Path $Install '.hermes\bin\hermes.cmd') -Value "@echo off`r`necho hermes 0.0.0"
  Set-Content -LiteralPath (Join-Path $Install '.hermes-runtime\python\interpreter.bin') -Value 'big'
  New-Item -ItemType Directory -Force -Path (Join-Path $H 'bin') | Out-Null
  # A fake launcher that understands `backup -o <zip>`: pre calls it for the data backup.
  @'
@echo off
if /i "%~1"=="backup" if /i "%~2"=="-o" (
  > "%~3" echo fake-zip
  exit /b 0
)
echo hermes
'@ | Set-Content -LiteralPath (Join-Path $H 'bin\hermes.cmd') -Encoding ASCII

  New-Item -ItemType Directory -Force -Path (Join-Path $env:HERMES_DESKTOP_USER_DATA_DIR 'Local Storage\leveldb'), (Join-Path $env:HERMES_DESKTOP_USER_DATA_DIR 'Cache') | Out-Null
  Set-Content -LiteralPath (Join-Path $env:HERMES_DESKTOP_USER_DATA_DIR 'Preferences') -Value '{"window":{}}'
  Set-Content -LiteralPath (Join-Path $env:HERMES_DESKTOP_USER_DATA_DIR 'connection.json') -Value '{"session":"t"}'
  Set-Content -LiteralPath (Join-Path $env:HERMES_DESKTOP_USER_DATA_DIR 'Local Storage\leveldb\000001.ldb') -Value 'x'
  Set-Content -LiteralPath (Join-Path $env:HERMES_DESKTOP_USER_DATA_DIR 'Cache\data.bin') -Value 'junk'
}

function Get-ShadowCapC {
  $vol = Get-CimInstance Win32_Volume | Where-Object { $_.Name -eq 'C:\' }
  $st = Get-CimInstance Win32_ShadowStorage | Where-Object { $_.Volume.DeviceID -eq $vol.DeviceID } | Select-Object -First 1
  if ($st) { return [UInt64]$st.MaxSpace } else { return $null }
}

try {
  New-Fixture
  $HeadSha = (& git -C $Install rev-parse HEAD | Out-String).Trim()
  Write-Host "checkout HEAD: $HeadSha"
  $CapBefore = Get-ShadowCapC
  Write-Host "shadow storage cap on C: before pre: $CapBefore"

  Write-Host "`n--- pre (source = the fixture repo, so no network) ---"
  $statusBefore = (& git -C $Install status --porcelain | Out-String)
  # Both trees as they are right now: post is checked against this for exactness.
  $homeBefore = Get-TreeListing $H
  $userDataBefore = Get-TreeListing $env:HERMES_DESKTOP_USER_DATA_DIR
  $r = Invoke-Rehearsal -Arguments @('pre', '-Source', $Install, '-BackupRoot', $Backups)
  Check 'pre exits 0' ($r.Code -eq 0)
  $Snap = (Get-ChildItem -LiteralPath $Backups -Directory | Sort-Object Name)[-1].FullName
  foreach ($f in @('hermes-backup.zip', 'shadows.txt', 'manifest.json', 'hermes-home.txt', 'target-sha')) {
    Check "backup artifact $f" (Test-Path -LiteralPath (Join-Path $Snap $f))
  }
  $ShadowIds = @(Get-Content -LiteralPath (Join-Path $Snap 'shadows.txt') | Where-Object { $_.Trim() } | ForEach-Object { ($_ -split "`t")[1] })
  Check 'one snapshot recorded' ($ShadowIds.Count -eq 1)
  Check 'the recorded snapshot exists' ([bool](Get-CimInstance Win32_ShadowCopy | Where-Object { $ShadowIds -contains $_.ID }))
  Check 'shadow storage cap on C: is at least 128 GB while the snapshot lives' ((Get-ShadowCapC) -ge [UInt64]128GB)

  Write-Host "`n--- pre points the install at the rehearsal copy ---"
  $cfg = (& git -C $Install config --local --get-regexp 'insteadOf' 2>$null | Out-String)
  Check 'two insteadOf entries written (repo-local)' ((([regex]::Matches($cfg, 'insteadOf', 'IgnoreCase')).Count) -eq 2)
  Check 'upstream-prompt marker created' (Test-Path -LiteralPath (Join-Path $H '.skip_upstream_prompt'))
  $getUrl = (& git -C $Install remote get-url origin | Out-String).Trim()
  Check 'remote get-url resolves to -Source' ($getUrl -eq $Install)
  $configured = (& git -C $Install config --get remote.origin.url | Out-String).Trim()
  Check 'config --get remote.origin.url stays official' ($configured -match 'NousResearch')

  Write-Host "`n--- pre changed nothing else ---"
  Check 'checkout untouched by pre' (((& git -C $Install rev-parse HEAD | Out-String).Trim()) -eq $HeadSha)
  Check 'pre changed no files in the checkout' (((& git -C $Install status --porcelain | Out-String)) -eq $statusBefore)
  Check 'pre says it did not update anything' ($r.Out -match 'nothing has been updated yet')

  Write-Host "`n--- status (read-only) ---"
  $r = Invoke-Rehearsal -Arguments @('status', '-BackupRoot', $Backups)
  Check 'status reports what it prepared' ($r.Out -match [regex]::Escape($HeadSha))
  Check 'status reports the snapshot present' ($r.Out -match 'snapshot\s+\S+ present')

  Write-Host "`n--- simulate an update: modify, add and delete in both trees ---"
  Set-Content -LiteralPath (Join-Path $H 'config.yaml') -Value 'timezone: changed-by-update'
  Remove-Item -LiteralPath (Join-Path $H 'memories\note.md')
  Set-Content -LiteralPath (Join-Path $Install 'added_by_update.py') -Value 'print(2)'
  New-Item -ItemType Directory -Force -Path (Join-Path $H 'photon\sidecar\node_modules\newdep') | Out-Null
  Set-Content -LiteralPath (Join-Path $H 'photon\sidecar\node_modules\newdep\index.js') -Value 'x'
  Set-Content -LiteralPath (Join-Path $env:HERMES_DESKTOP_USER_DATA_DIR 'Preferences') -Value '{"window":{"changed":true}}'
  Set-Content -LiteralPath (Join-Path $env:HERMES_DESKTOP_USER_DATA_DIR 'new-after-update.json') -Value '{}'
  Check 'the simulated update changed HERMES_HOME' ((Get-TreeListing $H) -ne $homeBefore)

  Write-Host "`n--- post ---"
  $r = Invoke-Rehearsal -Arguments @('post', '-BackupRoot', $Backups, '-Yes')
  Check 'post exits 0' ($r.Code -eq 0)
  Check 'config.yaml restored' (Test-Path -LiteralPath (Join-Path $H 'config.yaml'))
  Check '.env restored' (Test-Path -LiteralPath (Join-Path $H '.env'))
  Check 'memories restored' (Test-Path -LiteralPath (Join-Path $H 'memories\note.md'))
  Check 'plugin marker restored' (Test-Path -LiteralPath (Join-Path $H 'plugins\mnemosyne-wrapper\mnemosyne-wrapper.json'))
  Check 'photon sidecar marker restored' (Test-Path -LiteralPath (Join-Path $H 'photon\sidecar\node_modules\.package-lock.json'))
  Check 'PM store restored' (Test-Path -LiteralPath (Join-Path $Install '.hermes-runtime\python\interpreter.bin'))
  Check 'checkout restored' (Test-Path -LiteralPath (Join-Path $Install '.git'))
  Check 'checkout HEAD restored' (((& git -C $Install rev-parse HEAD | Out-String).Trim()) -eq $HeadSha)
  Check 'origin remote restored' (((& git -C $Install config --get remote.origin.url | Out-String).Trim()) -eq 'https://github.com/NousResearch/hermes-agent.git')
  Check 'userData connection.json restored' (Test-Path -LiteralPath (Join-Path $env:HERMES_DESKTOP_USER_DATA_DIR 'connection.json'))
  $cfg2 = (& git -C $Install config --local --get-regexp 'insteadOf' 2>$null | Out-String)
  if (-not $cfg2) { $cfg2 = '' }
  Check 'no stale insteadOf left in the checkout' ((([regex]::Matches($cfg2, 'insteadOf', 'IgnoreCase')).Count) -eq 0)
  Check 'upstream-prompt marker removed' (-not (Test-Path -LiteralPath (Join-Path $H '.skip_upstream_prompt')))
  Check 'origin resolves officially again' (((& git -C $Install remote get-url origin | Out-String).Trim()) -match 'NousResearch')
  Check 'bin shim restored' (Test-Path -LiteralPath (Join-Path $H 'bin\hermes.cmd'))
  Check 'post deleted the snapshot' (-not (Get-CimInstance Win32_ShadowCopy | Where-Object { $ShadowIds -contains $_.ID }))
  Check 'post removed its mount link' (-not (Test-Path -LiteralPath (Join-Path $Snap 'vss-C')))
  Check 'post put the shadow storage cap back exactly' ((Get-ShadowCapC) -eq $CapBefore)
  Write-Host "`n--- the acceptance criterion: every file identical before/after ---"
  $homeAfter = Get-TreeListing $H
  Check 'HERMES_HOME identical to before pre' ($homeAfter -eq $homeBefore)
  if ($homeAfter -ne $homeBefore) {
    Compare-Object ($homeBefore -split "`n") ($homeAfter -split "`n") |
      ForEach-Object { Write-Host "    $($_.SideIndicator) $($_.InputObject)" }
  }
  $userDataAfter = Get-TreeListing $env:HERMES_DESKTOP_USER_DATA_DIR
  Check 'userData identical to before pre' ($userDataAfter -eq $userDataBefore)
  if ($userDataAfter -ne $userDataBefore) {
    Compare-Object ($userDataBefore -split "`n") ($userDataAfter -split "`n") |
      ForEach-Object { Write-Host "    $($_.SideIndicator) $($_.InputObject)" }
  }
}
finally {
  # Never leave the real user PATH touched by a test.
  [Environment]::SetEnvironmentVariable('Path', $RealUserPath, 'User')
  if ($env:SMOKE_KEEP -eq '1') { Write-Host "root kept: $Root" }
  else { Remove-Item -LiteralPath $Root -Recurse -Force -ErrorAction SilentlyContinue }
}

Write-Host "`n=== smoke: $pass passed, $fail failed ==="
exit ([int]($fail -gt 0))
