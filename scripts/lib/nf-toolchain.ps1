# scripts/lib/nf-toolchain.ps1 - dot-source me; not a standalone script.
#
# ONE canonical answer to "is a self-contained Python toolchain shipped on the
# drive alongside this checkout, ready to build a venv with NOTHING on the host
# PATH?"
#
# DECISION-2026-09-09-001: a provisioned North Forge drive carries its own
# toolchain - uv.exe + a matching python-build-standalone CPython 3.11 - in a
# SIBLING folder of the checkout, mirroring the -venv / -data siblings:
#
#     <parent>\<leaf>-toolchain\uv\uv.exe
#     <parent>\<leaf>-toolchain\python\python.exe
#
# The admin prepares one master copy and copies it onto each drive during
# provisioning (see docs\BUILDING-A-DRIVE.md). It is NEVER git-tracked and NEVER
# fetched at recipient first-launch. When present, bootstrap uses it silently;
# when absent, bootstrap falls back to uv/python on the host PATH (admin/dev
# only, logged distinctly) and then to its existing "nothing available" error.
#
# READ-ONLY: this file locates and sanity-checks, it never downloads, never
# writes, never mutates, never exits, never emits. Dot-sourced: it defines
# functions and must not change the caller's execution preferences.
#
# NOT part of R1. The toolchain folder is an INPUT to venv creation, never a
# venv: nf-venv-state.ps1 (ownership) and nf-readiness.ps1 (readiness) never look
# at it, and it is a fourth sibling of the checkout so it cannot overlap the
# checkout, the venv, or the data dir - none of bootstrap's path-safety guards
# involve it.

function Test-NfExecutable {
    <# True when $Path is an existing, non-empty regular file with a .exe
       extension. A cheap presence/sanity gate - NOT an execution probe (the venv
       build itself and the readiness probe are the real proof). Never throws. #>
    param([Parameter(Mandatory = $true)][AllowEmptyString()][AllowNull()][string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
    try {
        $it = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
        if ($it.PSIsContainer) { return $false }
        if ($it.Length -le 0) { return $false }
        return ($it.Extension -and $it.Extension.ToLowerInvariant() -eq '.exe')
    } catch {
        return $false
    }
}

function Get-NfBundledToolchain {
    <# Locate a drive-native toolchain shipped as a sibling of the checkout.
       Returns [pscustomobject]:
         .Root    - the <parent>\<leaf>-toolchain path that was checked (always
                    set when $RepoRoot has a real parent; $null for a drive root)
         .UvExe   - full path to a present, non-empty uv.exe under Root\uv\, else $null
         .PyExe   - full path to a present, non-empty python.exe under Root\python\, else $null
         .Present - $true only when BOTH .UvExe and .PyExe are usable
       Never throws. Never writes. Never downloads. #>
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][string]$RepoRoot)

    $result = [pscustomobject]@{ Root = $null; UvExe = $null; PyExe = $null; Present = $false }

    try { $repoFull = [System.IO.Path]::GetFullPath($RepoRoot) } catch { return $result }
    $parent = Split-Path -Parent $repoFull
    $leaf   = Split-Path -Leaf   $repoFull
    if ([string]::IsNullOrWhiteSpace($parent) -or [string]::IsNullOrWhiteSpace($leaf)) {
        return $result   # a drive root ("D:\") has no parent - no sibling toolchain possible
    }

    $root = Join-Path $parent ($leaf + '-toolchain')
    $result.Root = $root
    if (-not (Test-Path -LiteralPath $root -PathType Container)) { return $result }

    $uv = Join-Path $root 'uv\uv.exe'
    $py = Join-Path $root 'python\python.exe'
    if (Test-NfExecutable $uv) {
        try { $result.UvExe = (Resolve-Path -LiteralPath $uv).Path } catch { $result.UvExe = $uv }
    }
    if (Test-NfExecutable $py) {
        try { $result.PyExe = (Resolve-Path -LiteralPath $py).Path } catch { $result.PyExe = $py }
    }
    $result.Present = [bool]($result.UvExe -and $result.PyExe)
    return $result
}
