# scripts/lib/nf-venv-state.ps1 - dot-source me; not a standalone script.
#
# ONE canonical, READ-ONLY answer to "what IS this would-be venv directory, and
# do we have the right to rebuild it in place?" - a decision that is SEPARATE
# from readiness (scripts/lib/nf-readiness.ps1). A failed readiness probe never
# by itself authorizes deletion; ownership does.
#
# Sole caller: scripts/bootstrap-north-forge.ps1, which is the single cleanup
# authority for the sibling VenvDir. This file NEVER deletes and NEVER writes -
# it only classifies. bootstrap runs readiness and this classifier independently,
# then decides create / rebuild / refuse per its state table.
#
# Ownership evidence, strongest first (see logs/ledger - R1):
#   1. a readable .nf-bootstrapped with a repo= line - matching OR foreign/stale,
#      either still proves North-Forge created this directory.
#   2. a pyvenv.cfg that PARSES as a real venv config: home= plus at least one of
#      include-system-site-packages= / version= / version_info= / executable= .
#      Filename presence alone is not enough.
#   3. a bare partial layout (a Scripts\ or Lib\ tree, an interrupted extract) is
#      NOT sufficient evidence - refuse, do not guess. -Force cannot override.
#
# States returned (.State):
#   Absent                 - the directory does not exist yet
#   EmptyDirectory         - exists, no entries
#   OwnedNorthForgeVenv    - evidence (1)
#   RecognizablePythonVenv - evidence (2), no North-Forge marker
#   UnknownDirectory       - has content, no (1) and no (2)  => bootstrap refuses
#   UnsafePath             - reparse point / junction / symlink, a file, or an
#                            unparseable path                 => bootstrap refuses
#
# Dot-sourced: defines functions only; must not change the caller's execution
# preferences, must not exit, must not emit.

if (-not (Get-Command Get-NfMarkerRepo -ErrorAction SilentlyContinue)) {
    . (Join-Path $PSScriptRoot 'nf-readiness.ps1')   # reuse Get-NfMarkerRepo / Get-NfCanonicalDir
}

function Test-NfReparsePoint {
    <# True when the item at $Path exists and carries the ReparsePoint attribute
       (a junction, a directory symlink, or a file symlink). Never throws. #>
    param([Parameter(Mandatory = $true)][AllowEmptyString()][AllowNull()][string]$Path)
    if ([string]::IsNullOrWhiteSpace($Path)) { return $false }
    try {
        $it = Get-Item -LiteralPath $Path -Force -ErrorAction Stop
        return (($it.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0)
    } catch { return $false }
}

function Get-NfVenvState {
    <# Read-only ownership/layout classifier for a would-be venv directory.
       Returns [pscustomobject] { State; Path; Evidence }.
       NEVER deletes, NEVER writes, NEVER throws. #>
    [CmdletBinding()]
    param([Parameter(Mandatory = $true)][AllowEmptyString()][AllowNull()][string]$VenvDir)

    $canon = Get-NfCanonicalDir $VenvDir
    if (-not $canon) {
        return [pscustomobject]@{ State = 'UnsafePath'; Path = "$VenvDir"; Evidence = 'path is null / blank / unparseable' }
    }

    $item = $null
    try { $item = Get-Item -LiteralPath $VenvDir -Force -ErrorAction Stop } catch { $item = $null }
    if ($null -eq $item) {
        return [pscustomobject]@{ State = 'Absent'; Path = $canon; Evidence = 'directory does not exist' }
    }
    if (-not $item.PSIsContainer) {
        return [pscustomobject]@{ State = 'UnsafePath'; Path = $canon; Evidence = 'path exists but is a file, not a directory' }
    }
    if (($item.Attributes -band [System.IO.FileAttributes]::ReparsePoint) -ne 0) {
        return [pscustomobject]@{ State = 'UnsafePath'; Path = $canon; Evidence = 'directory is a reparse point (junction / symlink)' }
    }

    $children = @(Get-ChildItem -LiteralPath $VenvDir -Force -ErrorAction SilentlyContinue)
    if ($children.Count -eq 0) {
        return [pscustomobject]@{ State = 'EmptyDirectory'; Path = $canon; Evidence = 'directory exists and is empty' }
    }

    # (1) strongest evidence: a readable North-Forge marker with a repo= line.
    $markerRepo = Get-NfMarkerRepo -MarkerPath (Join-Path $VenvDir '.nf-bootstrapped')
    if ($null -ne $markerRepo) {
        return [pscustomobject]@{
            State = 'OwnedNorthForgeVenv'; Path = $canon
            Evidence = "readable .nf-bootstrapped (repo=$markerRepo)"
        }
    }

    # (2) a pyvenv.cfg that actually parses as a venv config.
    $cfg = Join-Path $VenvDir 'pyvenv.cfg'
    if (Test-Path -LiteralPath $cfg -PathType Leaf) {
        $kv = @{}
        try {
            foreach ($ln in [System.IO.File]::ReadAllLines($cfg)) {
                if ($ln -match '^\s*([A-Za-z][A-Za-z0-9_\- ]*?)\s*=\s*(.*?)\s*$') {
                    $kv[$Matches[1].Trim().ToLowerInvariant()] = $Matches[2]
                }
            }
        } catch { $kv = @{} }
        $hasHome = $kv.ContainsKey('home') -and -not [string]::IsNullOrWhiteSpace([string]$kv['home'])
        $hasVenvKey = $kv.ContainsKey('include-system-site-packages') -or
                      $kv.ContainsKey('version') -or $kv.ContainsKey('version_info') -or
                      $kv.ContainsKey('executable')
        if ($hasHome -and $hasVenvKey) {
            return [pscustomobject]@{
                State = 'RecognizablePythonVenv'; Path = $canon
                Evidence = "valid pyvenv.cfg (home=$($kv['home']))"
            }
        }
    }

    # (3) content, but nothing that proves ownership - refuse, do not guess.
    $names = (($children | Select-Object -First 8 | ForEach-Object { $_.Name }) -join ', ')
    return [pscustomobject]@{
        State = 'UnknownDirectory'; Path = $canon
        Evidence = "content present, no readable .nf-bootstrapped and no valid pyvenv.cfg (contains: $names)"
    }
}
