# install-obsidian.ps1
#
# Uruchamiane przez instalator NSIS Agent Czesiek (apps/desktop/scripts/installer.nsh,
# makro `customInstall`) po skopiowaniu plików aplikacji.
#
# Co robi: sprawdza, czy Obsidian jest już na maszynie; jeśli nie — instaluje go
# z OFICJALNEGO źródła wydawcy:
#   1) `winget install -e --id Obsidian.Obsidian` (manifest publikuje Obsidian
#      w repozytorium winget; winget pobiera instalator od wydawcy),
#   2) awaryjnie: pobranie oficjalnego instalatora Windows z obsidian.md
#      (strona odsyła do wydania obsidianmd/obsidian-releases na GitHubie) i cicha
#      instalacja `/S`.
#
# Czego NIE robi: nie pakuje i nie redystrybuuje binarek Obsidiana. Obsidian jest
# darmowy, ale zamknięty — jego licencja nie pozwala na redystrybucję, więc
# instalator Go ŚCIĄGA u użytkownika, a nie rozsyła.
#
# Kontrakt dla instalatora:
#   exit 0  -> Obsidian był już zainstalowany ALBO udało się go zainstalować.
#   exit 2  -> nie udało się (brak internetu, brak winget, odmowa/UAC, blokada).
#              To NIE jest błąd instalacji Cześka — NSIS tylko pokazuje komunikat.
# Wszystko jest opakowane w try/catch: skrypt nigdy nie rzuca wyjątku na zewnątrz.

[CmdletBinding()]
param(
    # Docelowy vault Cześka — dziś tylko logujemy, tworzy go aplikacja przy starcie.
    [string]$VaultPath = '',
    # Ścieżka logu (instalator podaje ...\hermes-home\logs\obsidian-install.log).
    [string]$LogPath = '',
    [int]$TimeoutSeconds = 300
)

$ErrorActionPreference = 'Continue'
$script:LogFile = $LogPath
$script:Timeout = if ($TimeoutSeconds -gt 0) { $TimeoutSeconds } else { 300 }

function Write-CzesiekLog {
    param([string]$Message)

    $line = '[{0}] {1}' -f (Get-Date).ToString('s'), $Message
    Write-Host $line

    if ($script:LogFile) {
        try {
            $dir = Split-Path -Parent $script:LogFile
            if ($dir -and -not (Test-Path -LiteralPath $dir)) {
                New-Item -ItemType Directory -Path $dir -Force | Out-Null
            }
            Add-Content -LiteralPath $script:LogFile -Value $line -Encoding UTF8
        } catch {
            # log to dodatek — jego brak nie może przerwać instalacji
        }
    }
}

function Test-ObsidianInstalled {
    # 1. Typowe lokalizacje pliku wykonywalnego (per-user i per-machine).
    $candidates = @()
    if ($env:LOCALAPPDATA) {
        $candidates += (Join-Path $env:LOCALAPPDATA 'Obsidian\Obsidian.exe')
        $candidates += (Join-Path $env:LOCALAPPDATA 'Programs\Obsidian\Obsidian.exe')
    }
    if ($env:ProgramFiles) { $candidates += (Join-Path $env:ProgramFiles 'Obsidian\Obsidian.exe') }
    if (${env:ProgramFiles(x86)}) { $candidates += (Join-Path ${env:ProgramFiles(x86)} 'Obsidian\Obsidian.exe') }

    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            Write-CzesiekLog "Obsidian znaleziony: $candidate"
            return $true
        }
    }

    # 2. Wpisy odinstalowania (nazwane i GUID-owe; HKCU, HKLM i widok WOW6432).
    $roots = @(
        'HKCU:\Software\Microsoft\Windows\CurrentVersion\Uninstall',
        'HKLM:\Software\Microsoft\Windows\CurrentVersion\Uninstall',
        'HKLM:\Software\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall'
    )
    foreach ($root in $roots) {
        if (-not (Test-Path $root)) { continue }
        foreach ($sub in (Get-ChildItem -Path $root -ErrorAction SilentlyContinue)) {
            try {
                $name = (Get-ItemProperty -Path $sub.PSPath -Name DisplayName -ErrorAction SilentlyContinue).DisplayName
                if ($name -and $name -like 'Obsidian*') {
                    Write-CzesiekLog "Obsidian znaleziony w rejestrze: $($sub.PSChildName) ($name)"
                    return $true
                }
            } catch {
                continue
            }
        }
    }

    # 3. winget list — ostatnia siatka bezpieczeństwa.
    try {
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if ($winget) {
            $listed = & winget list --id Obsidian.Obsidian -e --accept-source-agreements 2>$null
            if ($listed -and (($listed -join "`n") -match 'Obsidian\.Obsidian')) {
                Write-CzesiekLog 'Obsidian znaleziony przez winget list.'
                return $true
            }
        }
    } catch {
        # brak winget / błąd zapytania nie jest problemem
    }

    return $false
}

function Install-ObsidianViaWinget {
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (-not $winget) {
        Write-CzesiekLog 'winget niedostępny na tej maszynie.'
        return $false
    }

    # Dwie próby: z --scope user (manifest Obsidiana jest per-user) i bez scope,
    # gdy starsza wersja winget nie zna tego przełącznika albo manifest go nie
    # deklaruje. Obie ciche i z zaakceptowanymi umowami, żeby nic nie wisiało.
    $attempts = @(
        @('install', '-e', '--id', 'Obsidian.Obsidian', '--scope', 'user', '--silent', '--accept-package-agreements', '--accept-source-agreements'),
        @('install', '-e', '--id', 'Obsidian.Obsidian', '--silent', '--accept-package-agreements', '--accept-source-agreements')
    )

    foreach ($args in $attempts) {
        Write-CzesiekLog ('winget ' + ($args -join ' '))
        try {
            $output = & winget @args 2>&1 | Out-String
            $exit = $LASTEXITCODE
        } catch {
            Write-CzesiekLog "winget zgłosił wyjątek: $($_.Exception.Message)"
            continue
        }

        if ($output) { Write-CzesiekLog ('winget: ' + ($output.Trim())) }

        if ($exit -eq 0 -or (Test-ObsidianInstalled)) {
            Write-CzesiekLog 'winget zakończył się sukcesem.'
            return $true
        }

        Write-CzesiekLog "winget zwrócił kod $exit — następna próba/fallback."
    }

    return $false
}

function Install-ObsidianFromOfficialDownload {
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
    } catch {
        # starsze .NET — zostawiamy domyślne ustawienie
    }

    # Oficjalna strona pobierania — stąd bierzemy BIEŻĄCY, oficjalny URL wydania.
    $url = $null
    try {
        $page = Invoke-WebRequest -Uri 'https://obsidian.md/download' -UseBasicParsing -TimeoutSec 60
        $match = [regex]::Match([string]$page.Content, 'https://github\.com/obsidianmd/obsidian-releases/releases/download/[^"''\s<>]+/Obsidian-[0-9][0-9.]*\.exe')
        if ($match.Success) { $url = $match.Value }
    } catch {
        Write-CzesiekLog "Nie udało się odczytać obsidian.md/download: $($_.Exception.Message)"
    }

    if (-not $url) {
        Write-CzesiekLog 'Nie znaleziono oficjalnego URL instalatora Obsidiana na obsidian.md (brak internetu?).'
        return $false
    }

    $dest = Join-Path $env:TEMP ('Obsidian-' + [Guid]::NewGuid().ToString('N') + '.exe')
    Write-CzesiekLog "Pobieram oficjalny instalator Obsidiana: $url"

    try {
        $progressPreference = 'SilentlyContinue'
        Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing -TimeoutSec $script:Timeout
    } catch {
        Write-CzesiekLog "Pobranie nie powiodło się: $($_.Exception.Message)"
        Remove-Item -LiteralPath $dest -Force -ErrorAction SilentlyContinue
        return $false
    }

    if (-not (Test-Path -LiteralPath $dest)) {
        Write-CzesiekLog 'Pobrany plik nie istnieje — przerywam.'
        return $false
    }

    Write-CzesiekLog 'Uruchamiam oficjalny instalator Obsidiana cicho (/S).'
    try {
        $process = Start-Process -FilePath $dest -ArgumentList '/S' -PassThru -Wait
        Write-CzesiekLog "Instalator Obsidiana zakończył się kodem $($process.ExitCode)."
    } catch {
        Write-CzesiekLog "Uruchomienie instalatora nie powiodło się: $($_.Exception.Message)"
    } finally {
        # Nie zostawiamy pobranej binarki na dysku użytkownika.
        Remove-Item -LiteralPath $dest -Force -ErrorAction SilentlyContinue
    }

    return (Test-ObsidianInstalled)
}

$exitCode = 2

try {
    Write-CzesiekLog '— Agent Czesiek: krok Obsidian (pamięć ogólna) —'
    if ($VaultPath) { Write-CzesiekLog "Docelowy vault (tworzy go aplikacja przy pierwszym starcie): $VaultPath" }

    if (Test-ObsidianInstalled) {
        Write-CzesiekLog 'Obsidian jest już zainstalowany — pomijam instalację.'
        $exitCode = 0
    } elseif (Install-ObsidianViaWinget) {
        $exitCode = 0
    } elseif (Install-ObsidianFromOfficialDownload) {
        $exitCode = 0
    } else {
        Write-CzesiekLog 'Nie udało się automatycznie zainstalować Obsidiana. Vault i pamięć Cześka działają dalej — Obsidiana można doinstalować ręcznie:'
        Write-CzesiekLog '  winget install -e --id Obsidian.Obsidian'
        Write-CzesiekLog 'albo ze strony https://obsidian.md/download'
        $exitCode = 2
    }
} catch {
    Write-CzesiekLog "Nieoczekiwany błąd kroku Obsidian (nie-fatalny): $($_.Exception.Message)"
    $exitCode = 2
}

exit $exitCode
