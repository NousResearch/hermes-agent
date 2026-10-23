#Requires -Version 5.1
# Smoke test for the ast-grep skill on Windows (PowerShell 5.1+).
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$SkillDir  = Split-Path -Parent $ScriptDir
$Helper    = Join-Path $SkillDir 'scripts/ast_grep_helper.py'
$Python    = if (Get-Command py -ErrorAction SilentlyContinue) { 'py' } else { 'python' }

$Output = Join-Path $env:TEMP ("ast-grep-skill-smoke-" + [guid]::NewGuid().ToString('N').Substring(0,8))
New-Item -ItemType Directory -Path $Output -Force | Out-Null

function Pass([string]$msg) { Write-Host "PASS: $msg" }
function Fail([string]$msg) { Write-Host "FAIL: $msg" -ForegroundColor Red; Remove-Item -Recurse -Force $Output -ErrorAction SilentlyContinue; exit 1 }

function Run([string[]]$Args) {
    $stdoutFile = Join-Path $Output ("out-" + [guid]::NewGuid().ToString('N').Substring(0,8) + ".txt")
    $proc = Start-Process -FilePath $Python -ArgumentList (@($Helper) + $Args) -NoNewWindow -PassThru -Wait -RedirectStandardOutput $stdoutFile -RedirectStandardError "$stdoutFile.err"
    $stdout = if (Test-Path $stdoutFile) { Get-Content $stdoutFile -Raw } else { '' }
    $stderr = if (Test-Path "$stdoutFile.err") { Get-Content "$stdoutFile.err" -Raw } else { '' }
    return [pscustomobject]@{
        ExitCode = $proc.ExitCode
        Stdout = $stdout
        Stderr = $stderr
        Combined = "$stdout`n$stderr"
    }
}

try {
    # 1. --version
    $r = Run @('--version')
    if ($r.Combined -notmatch 'ast-grep-helper') { Fail '--version output missing' }
    Pass '--version'

    # 2. langs (must list >=25)
    $r = Run @('langs')
    $langCount = ($r.Stdout -split "`n" | Where-Object { $_ -match '^  [a-z]' }).Count
    if ($langCount -lt 25) { Fail "langs listed only $langCount (expected >=25)" }
    Pass 'langs lists at least 25 languages'

    # 3. regex misuse: \w+
    $r = Run @('validate', '\w+', '--lang', 'ts')
    if ($r.ExitCode -ne 2) { Fail "validate '\w+' should exit 2, got $($r.ExitCode)" }
    if ($r.Combined -notmatch 'regex') { Fail "validate '\w+' should mention regex" }
    Pass 'validate detects \w regex misuse'

    # 4. valid pattern
    $r = Run @('validate', 'console.log($MSG)', '--lang', 'ts')
    if ($r.ExitCode -ne 0) { Fail "validate 'console.log(`$MSG)' should exit 0, got $($r.ExitCode)" }
    Pass 'validate accepts plausible pattern'

    # 5. Python trailing colon
    $r = Run @('validate', 'def $F($$$):', '--lang', 'py')
    if ($r.ExitCode -ne 2) { Fail "validate 'def `$F(`$`$`$):' should exit 2, got $($r.ExitCode)" }
    if ($r.Combined -notmatch 'colon|trailing') { Fail 'validate should mention trailing colon' }
    Pass 'validate detects Python trailing colon'

    # 6. Incomplete TS function
    $r = Run @('validate', 'function $N', '--lang', 'ts')
    if ($r.ExitCode -ne 2) { Fail "validate 'function `$N' should exit 2, got $($r.ExitCode)" }
    if ($r.Combined -notmatch 'incomplete|params|body') { Fail 'validate should hint about params/body' }
    Pass 'validate detects incomplete TS function'

    # 7. Alternation pipe
    $r = Run @('validate', 'foo|bar', '--lang', 'ts')
    if ($r.ExitCode -ne 2) { Fail "validate 'foo|bar' should exit 2, got $($r.ExitCode)" }
    if ($r.Combined -notmatch 'alternation|regex') { Fail 'validate should mention alternation' }
    Pass 'validate detects literal | alternation'

    # 8. doctor
    $r = Run @('doctor')
    if ($r.Combined -notmatch 'ast-grep-helper') { Fail 'doctor missing helper version line' }
    Pass 'doctor produces output'

    # 9. search w/o binary
    $r = Run @('-q', 'search', 'foo()', '--lang', 'ts', 'C:/nonexistent-path-xyzzy')
    switch ($r.ExitCode) {
        { $_ -in 0,1,4 } { Pass "search runs (rc=$($r.ExitCode), ast-grep available)" }
        3 {
            if ($r.Combined -notmatch 'install') { Fail 'search rc=3 should print install hint' }
            Pass 'search without binary prints install hint'
        }
        default { Fail "search returned unexpected rc=$($r.ExitCode): $($r.Combined)" }
    }

    # 10. install.ps1 syntax (parse-check via PowerShell tokenizer)
    $tokens = $null
    $errors = $null
    [System.Management.Automation.Language.Parser]::ParseFile(
        (Join-Path $SkillDir 'install.ps1'), [ref]$tokens, [ref]$errors) | Out-Null
    if ($errors -and $errors.Count -gt 0) { Fail "install.ps1 has parse errors: $($errors -join '; ')" }
    Pass 'install.ps1 parses cleanly'

    # 11. SKILL.md frontmatter
    $skill = Get-Content (Join-Path $SkillDir 'SKILL.md') -Raw
    if (-not $skill.StartsWith("---`n") -and -not $skill.StartsWith("---`r`n")) {
        Fail 'SKILL.md must start with YAML frontmatter'
    }
    $endIdx = $skill.IndexOf("`n---`n", 4)
    if ($endIdx -lt 0) { $endIdx = $skill.IndexOf("`r`n---`r`n", 4) }
    if ($endIdx -lt 0) { Fail 'SKILL.md missing closing ---' }
    $fm = $skill.Substring(4, $endIdx - 4)
    if ($fm -notmatch '(?m)^name:\s*ast-grep\s*$') { Fail 'frontmatter missing name: ast-grep' }
    if ($fm -notmatch '(?m)^description:') { Fail 'frontmatter missing description' }
    Pass 'SKILL.md frontmatter shape'

    # 12. All required reference files exist
    $required = @(
        'references/install.md', 'references/patterns.md', 'references/pitfalls.md',
        'references/recipes.md', 'references/cli.md', 'references/yaml-rules.md',
        'references/sgconfig.md'
    )
    foreach ($f in $required) {
        if (-not (Test-Path (Join-Path $SkillDir $f))) { Fail "missing $f" }
    }
    Pass 'all references present'

    Write-Host ''
    Write-Host 'all smoke tests passed'
}
finally {
    Remove-Item -Recurse -Force $Output -ErrorAction SilentlyContinue
}
