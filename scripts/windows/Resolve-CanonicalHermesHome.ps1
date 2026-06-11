# Resolve the canonical user install HERMES_HOME (~/.hermes).
# Rejects repo-local "<checkout>/.hermes" so dev shells and Cursor isolation
# cannot shadow the real user install and steal OAuth refresh-token rotation.

function Resolve-CanonicalHermesHome {
    param(
        [string]$Preferred = "",
        [string]$RepoRoot = ""
    )

    $canonical = Join-Path $env:USERPROFILE ".hermes"

    if ($Preferred -and $Preferred.Trim()) {
        $candidate = $Preferred.Trim()
    }
    elseif ($env:HERMES_HOME -and $env:HERMES_HOME.Trim()) {
        $candidate = $env:HERMES_HOME.Trim()
    }
    else {
        return $canonical
    }

    $candidateFull = [System.IO.Path]::GetFullPath($candidate)

    if ($RepoRoot -and (Test-Path -LiteralPath $RepoRoot)) {
        $repoLocal = [System.IO.Path]::GetFullPath((Join-Path $RepoRoot ".hermes"))
        if ($candidateFull -ieq $repoLocal) {
            Write-Warning "Ignoring repo-local HERMES_HOME ($candidate). Using $canonical"
            return $canonical
        }
    }

    return $candidateFull
}
