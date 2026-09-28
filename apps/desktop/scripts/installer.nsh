; Launch the installed binary, independent of shortcut migration and shell context.
!macro customInstall
  StrCpy $launchLink "$INSTDIR\${APP_EXECUTABLE_FILENAME}"

  ; ---------------------------------------------------------------------------
  ; Pamięć ogólna Cześka: Obsidian z OFICJALNEGO źródła (bez redystrybucji).
  ;
  ; Obsidian jest darmowy, ale zamknięty — jego licencja nie pozwala rozsyłać
  ; binarek, dlatego NIE pakujemy jego instalatora ani do repo, ani do naszego
  ; instalatora. Ten krok tylko URUCHAMIA instalację u użytkownika:
  ;   1. resources\bootstrap\install-obsidian.ps1 -> winget (manifest wydawcy,
  ;      winget pobiera instalator od wydawcy),
  ;   2. awaryjnie: pobranie oficjalnego instalatora Windows z obsidian.md
  ;      (strona odsyła do wydania obsidianmd/obsidian-releases) i cicha
  ;      instalacja /S; plik tymczasowy jest po niej kasowany.
  ; Skrypt sam wykrywa, że Obsidian już jest (typowe ścieżki pliku, wpisy
  ; odinstalowania w rejestrze, `winget list`) i wtedy nic nie robi.
  ;
  ; Krok jest NIE-FATALNY. Kod 2 oznacza „nie udało się" (brak winget, brak
  ; internetu, blokada/UAC) i tylko pokazuje użytkownikowi komunikat z komendą
  ; do ręcznej instalacji. Instalacja Cześka leci dalej, a vault i tak powstanie
  ; przy pierwszym starcie aplikacji (electron/vault-seed.ts).
  ;
  ; Log kroku: %LOCALAPPDATA%\AI Evolution Jarvis\hermes-home\logs\obsidian-install.log
  ; ---------------------------------------------------------------------------
  StrCpy $0 "2"

  IfFileExists "$INSTDIR\resources\bootstrap\install-obsidian.ps1" 0 czesiekObsidianNoScript
    nsExec::ExecToLog '"$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "$INSTDIR\resources\bootstrap\install-obsidian.ps1" -VaultPath "$DOCUMENTS\Czesiek Vault" -LogPath "$LOCALAPPDATA\AI Evolution Jarvis\hermes-home\logs\obsidian-install.log"'
    Pop $0
    Goto czesiekObsidianDone

  czesiekObsidianNoScript:
    DetailPrint "Pomijam Obsidiana: brak resources\bootstrap\install-obsidian.ps1"

  czesiekObsidianDone:
    ${If} $0 == "2"
      ${IfNot} ${Silent}
        MessageBox MB_OK|MB_ICONINFORMATION "Nie udało się automatycznie zainstalować Obsidiana.$\r$\n$\r$\nPamięć ogólna Cześka działa dalej — Obsidiana możesz doinstalować jednym poleceniem:$\r$\n$\r$\n    winget install -e --id Obsidian.Obsidian$\r$\n$\r$\nalbo ze strony https://obsidian.md/download"
      ${EndIf}
    ${EndIf}
!macroend
