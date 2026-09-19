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

$PSExe = (Get-Command powershell.exe -ErrorAction SilentlyContinue).Source
if (-not $PSExe) { $PSExe = (Get-Command pwsh.exe -ErrorAction SilentlyContinue).Source }
if (-not $PSExe) { throw 'no powershell host found' }
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
# tar lives in System32; put it ahead of any WindowsApps payload on PATH.
if ($env:SystemRoot) { $env:PATH = (Join-Path $env:SystemRoot 'System32') + ';' + $env:PATH }

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
  Set-Content -LiteralPath (Join-Path $H 'bin\hermes.cmd') -Value "@echo off`r`necho hermes"

  New-Item -ItemType Directory -Force -Path (Join-Path $env:HERMES_DESKTOP_USER_DATA_DIR 'Local Storage\leveldb'), (Join-Path $env:HERMES_DESKTOP_USER_DATA_DIR 'Cache') | Out-Null
  Set-Content -LiteralPath (Join-Path $env:HERMES_DESKTOP_USER_DATA_DIR 'Preferences') -Value '{"window":{}}'
  Set-Content -LiteralPath (Join-Path $env:HERMES_DESKTOP_USER_DATA_DIR 'connection.json') -Value '{"session":"t"}'
  Set-Content -LiteralPath (Join-Path $env:HERMES_DESKTOP_USER_DATA_DIR 'Local Storage\leveldb\000001.ldb') -Value 'x'
  Set-Content -LiteralPath (Join-Path $env:HERMES_DESKTOP_USER_DATA_DIR 'Cache\data.bin') -Value 'junk'
}

try {
  New-Fixture
  $HeadSha = (& git -C $Install rev-parse HEAD | Out-String).Trim()
  Write-Host "checkout HEAD: $HeadSha"

  Write-Host "`n--- pre (source = the fixture repo, so no network) ---"
  $statusBefore = (& git -C $Install status --porcelain | Out-String)
  # Both trees as they are right now: post is checked against this for exactness.
  $homeBefore = Get-TreeListing $H
  $userDataBefore = Get-TreeListing $env:HERMES_DESKTOP_USER_DATA_DIR
  $r = Invoke-Rehearsal -Arguments @('pre', '-Source', $Install, '-Ref', 'main', '-BackupRoot', $Backups)
  Check 'pre exits 0' ($r.Code -eq 0)
  $Snap = (Get-ChildItem -LiteralPath $Backups -Directory | Sort-Object Name)[-1].FullName
  foreach ($f in @('hermes-home.tar', 'electron-userdata.tar', 'manifest.json', 'hermes-home.txt', 'target-sha')) {
    Check "backup artifact $f" (Test-Path -LiteralPath (Join-Path $Snap $f))
  }
  $tarList = (& tar.exe -tf (Join-Path $Snap 'hermes-home.tar') | Out-String)
  Check 'whole home: checkout .git in the tar' ($tarList -match '(?m)^\./hermes-agent/\.git/config\s*$')
  Check 'whole home: PM store in the tar' ($tarList -match '(?m)^\./hermes-agent/\.hermes-runtime/python/interpreter\.bin\s*$')
  Check 'whole home: config.yaml in the tar' ($tarList -match '(?m)^\./config\.yaml\s*$')
  $udList = (& tar.exe -tf (Join-Path $Snap 'electron-userdata.tar') | Out-String)
  Check 'whole userData: nothing filtered out of the tar' ($udList -match '(?m)^\./Cache/data\.bin\s*$')

  Write-Host "`n--- pre points the install at the rehearsal copy ---"
  $served = (& git -C (Join-Path $Snap 'serve.git') rev-parse refs/heads/main | Out-String).Trim()
  Check 'serve.git main is the custom ref' ($served -eq $HeadSha)
  $cfg = (& git -C $Install config --local --get-regexp 'insteadOf' 2>$null | Out-String)
  Check 'two insteadOf entries written (repo-local)' ((([regex]::Matches($cfg, 'insteadOf', 'IgnoreCase')).Count) -eq 2)
  Check 'upstream-prompt marker created' (Test-Path -LiteralPath (Join-Path $H '.skip_upstream_prompt'))
  $getUrl = (& git -C $Install remote get-url origin | Out-String).Trim()
  Check 'remote get-url resolves to the rehearsal copy' ($getUrl -match 'serve\.git')
  $configured = (& git -C $Install config --get remote.origin.url | Out-String).Trim()
  Check 'config --get remote.origin.url stays official' ($configured -match 'NousResearch')

  Write-Host "`n--- pre changed nothing else ---"
  Check 'checkout untouched by pre' (((& git -C $Install rev-parse HEAD | Out-String).Trim()) -eq $HeadSha)
  Check 'pre changed no files in the checkout' (((& git -C $Install status --porcelain | Out-String)) -eq $statusBefore)
  Check 'pre says it did not update anything' ($r.Out -match 'nothing has been updated yet')

  Write-Host "`n--- status (read-only) ---"
  $r = Invoke-Rehearsal -Arguments @('status', '-BackupRoot', $Backups)
  Check 'status reports what it prepared' ($r.Out -match [regex]::Escape($HeadSha))

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
