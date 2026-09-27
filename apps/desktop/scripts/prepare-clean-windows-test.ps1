param(
  [Parameter(Mandatory=$true)][string]$Installer,
  [Parameter(Mandatory=$true)][string]$OutputDirectory
)
$ErrorActionPreference = 'Stop'
$sourceInstaller = (Resolve-Path -LiteralPath $Installer).Path
if ([IO.Path]::GetExtension($sourceInstaller) -ne '.exe') { throw 'Wskaż instalator EXE.' }
$testRoot = [IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $testRoot) { throw 'Wybierz nowy, pusty katalog testu. Istniejących danych nie nadpisujemy.' }
$inputPath = Join-Path $testRoot 'input'
$resultsPath = Join-Path $testRoot 'results'
New-Item -ItemType Directory -Path $inputPath,$resultsPath | Out-Null
Copy-Item -LiteralPath $sourceInstaller -Destination (Join-Path $inputPath 'setup.exe')
@'
$ErrorActionPreference = 'Stop'
$result = @{ installed=$false; firstStart=$false; restart=$false; noSystemPython=(-not (Get-Command python -ErrorAction SilentlyContinue)); error=$null }
try {
  $env:HERMES_DESKTOP_USER_DATA_DIR='C:\CzesiekTestData'
  New-Item -ItemType Directory -Path $env:HERMES_DESKTOP_USER_DATA_DIR -Force | Out-Null
  [IO.File]::WriteAllText((Join-Path $env:HERMES_DESKTOP_USER_DATA_DIR 'runtime-collaborator.json'), '{"mode":"bundled"}', (New-Object Text.UTF8Encoding $false))
  $installer=Start-Process 'C:\CzesiekInput\setup.exe' -ArgumentList '/S','/D=C:\CzesiekApp' -PassThru -Wait -WindowStyle Hidden
  if ($installer.ExitCode -ne 0) { throw "Installer exit $($installer.ExitCode)" }
  $exe='C:\CzesiekApp\AI Evolution Jarvis.exe'
  $result.installed=Test-Path -LiteralPath $exe
  if (-not $result.installed) { throw 'Nie znaleziono aplikacji.' }
  $log='C:\CzesiekTestData\hermes-home\logs\desktop.log'
  foreach ($attempt in 1,2) {
    $before=if(Test-Path $log){(Get-Content $log -Raw).Length}else{0}
    $app=Start-Process -FilePath $exe -PassThru -WindowStyle Hidden
    $ready=$false
    foreach ($second in 1..120) {
      Start-Sleep -Seconds 1
      if(Test-Path $log){
        $text=(Get-Content $log -Raw).Substring($before)
        if($text.Contains('starting first-launch bootstrap')) { throw 'Rozpoczęto instalację silnika zamiast użycia pakietu.' }
        if($text.Contains('backend is ready')) { $ready=$true; break }
      }
    }
    if(-not $ready){throw 'Backend nie osiągnął gotowości.'}
    if($attempt -eq 1){$result.firstStart=$true}else{$result.restart=$true}
    & taskkill.exe /PID $app.Id /T /F | Out-Null
    Start-Sleep -Seconds 2
  }
} catch { $result.error=$_.Exception.Message }
finally {
  $result | ConvertTo-Json | Set-Content -Encoding utf8 'C:\CzesiekResults\result.json'
  if(Test-Path 'C:\CzesiekTestData\hermes-home\logs\desktop.log'){Copy-Item 'C:\CzesiekTestData\hermes-home\logs\desktop.log' 'C:\CzesiekResults\desktop.log'}
}
'@ | Set-Content -Encoding utf8 (Join-Path $inputPath 'test.ps1')
$escapedInput = [Security.SecurityElement]::Escape($inputPath)
$escapedResults = [Security.SecurityElement]::Escape($resultsPath)
@"
<Configuration>
 <Networking>Disable</Networking><ClipboardRedirection>Disable</ClipboardRedirection><VGpu>Disable</VGpu><MemoryInMB>4096</MemoryInMB>
 <MappedFolders>
  <MappedFolder><HostFolder>$escapedInput</HostFolder><SandboxFolder>C:\CzesiekInput</SandboxFolder><ReadOnly>true</ReadOnly></MappedFolder>
  <MappedFolder><HostFolder>$escapedResults</HostFolder><SandboxFolder>C:\CzesiekResults</SandboxFolder><ReadOnly>false</ReadOnly></MappedFolder>
 </MappedFolders>
 <LogonCommand><Command>powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\CzesiekInput\test.ps1</Command></LogonCommand>
</Configuration>
"@ | Set-Content -Encoding utf8 (Join-Path $testRoot 'Czesiek-clean-install.wsb')
Write-Output "Gotowy test: $testRoot\Czesiek-clean-install.wsb. Wynik pojawi się w results\result.json. Utworzenie pliku nie oznacza wykonania testu."
