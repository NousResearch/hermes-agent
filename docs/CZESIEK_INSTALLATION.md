# Instalacja Agent Czesiek na nowym komputerze

Nie trzeba wcześniej instalować Hermes Agent, Pythona ani konfigurować CLI.
Instalacja składa się z dwóch etapów:

1. Instalator Windows NSIS instaluje okno aplikacji, tworzy skrót **Agent Czesiek**
   na pulpicie i w menu Start, a po zakończeniu pozwala uruchomić program.
2. Przy pierwszym starcie Czesiek pobiera instalator silnika przypięty do commita
   wydania z `aievolutionpl/AGENT_CZESIEK`. Kreator przeprowadza instalację
   potrzebnych narzędzi, Pythona, środowiska i zależności. Następnie użytkownik
   wybiera model i podaje własny klucz API.

Pierwszy start wymaga internetu (GitHub, źródła pakietów i zależności).
To instalacja online, nie pełny pakiet offline. Czas zależy od sieci i komputera.
Samo otwarcie okna nie potwierdza zakończenia instalacji silnika.

## Własne środowisko

Pakiet Cześka używa własnego katalogu danych:

- Windows: `%LOCALAPPDATA%\AI Evolution Jarvis\hermes-home`.
- macOS/Linux: `~/.ai-evolution-jarvis/hermes-home`.

Historyczna nazwa katalogu pozostaje dla zgodności z aktualizacjami i zapisanymi
profilami. Widoczna nazwa produktu to Agent Czesiek. Pakiet nie wybiera
automatycznie Hermesa znalezionego na PATH ani modułu z systemowego Pythona.
Jawne połączenie zdalne lub override deweloperski pozostają osobną możliwością.
Instalacja deweloperska ze źródeł ma własne reguły wykrywania backendu.

## Ikona i skróty

Ikona pochodzi z logo dostarczonego przez AI Evolution Polska. Plik źródłowy
`apps/desktop/assets/icon-master.png` został przygotowany narzędziem imagegen:
zachowano szklaną literę C, centralny orb, siedem słupków głosu oraz kolory
cyan/niebieski/fiolet; uporządkowano krawędzie i pozostawiono przezroczyste tło.

Wersje techniczne generuje na Windows:

```powershell
./apps/desktop/scripts/build-brand-icons.ps1
```

- `assets/icon.ico`: Windows, rozmiary od 16 do 256 px.
- `assets/icon.icns`: macOS, rozmiary od 16 do 1024 px.
- `assets/icon.png`: Linux i zasób ogólny, 1024 px.
- `public/apple-touch-icon.png`: ikona okna i fallback, 180 px.

Windows EXE ma nazwę produktu i opis **Agent Czesiek**, firmę **AI Evolution Polska**
oraz informację o silniku Hermes na MIT. Nieudane osadzenie ikony zatrzymuje
pakowanie zamiast publikować plik z domyślnym znakiem Electron.
Identyfikator aplikacji, nazwa techniczna EXE i protokół pozostają kompatybilne
ze starszymi wydaniami. Nie należy ich zmieniać bez migracji aktualizatora.

## Weryfikacja wydania

Build aplikacji i testy routingu/instalatora nie zastępują instalacji na czystym
Windows. Przed publikacją sprawdź instalator w nowym koncie lub maszynie wirtualnej:

1. Brak zainstalowanego Hermesa, Pythona i Node.js.
2. Instalacja NSIS, obecność skrótu i poprawna ikona.
3. Pierwszy start: postęp pobierania, instalacja silnika, dojście do onboardingu.
4. Zapis klucza przez ustawienia i rzeczywista odpowiedź modelu.
5. Ponowny start bez ponownej instalacji silnika.
6. Aktualizacja zachowuje profil, a istniejąca oddzielna instalacja Hermesa działa dalej.

Podpisywanie i publikacja instalatora są osobnym etapem opisanym w
[pipeline wydania](product/RELEASE_PIPELINE.md). Merge kodu nie aktualizuje
automatycznie plików dostępnych w Releases.

### Wynik lokalnej próby — 27 września 2026

- Zbudowano renderer, Electron i instalator NSIS x64; TypeScript bez błędów.
- Testy ikon, bootstrapa i polityki runtime: 45 zaliczonych, 1 pominięty.
- Dodatkowe testy instalatora Python: 9 zaliczonych, 1 niezaliczony
  (`test_python_find_timeout_kills_uv_and_fails_stage`, timeout 45 s).
  Ten sam błąd odtworzono oddzielnie na kodzie sprzed zmian; nie uznajemy
  obsługi tego przypadku za zweryfikowaną. Job CI „PowerShell installer tests”
  dla PR #69 zakończył się sukcesem, ale obejmuje inny zakres.
- Rzeczywiste pobranie przypiętego `install.ps1` z repozytorium Cześka: sukces.
- W pustym katalogu wykonano etapy `repository`, `python`, `venv`, `dependencies`.
  Pobrano własnego Pythona 3.11.16, a nowy backend odpowiedział HTTP 200
  na `/api/health`. Nie korzystał z istniejącego środowiska Hermesa.
- Pierwsza próba wykryła błąd długich ścieżek Windows. Po włączeniu
  `core.longpaths` w procesie instalatora ponowna próba zakończyła się sukcesem.
  HTTPS tego procesu korzysta z magazynu certyfikatów Windows (`schannel`).
- Instalacja zależności użyła istniejącego fallbacku PyPI, ponieważ lokalny uv
  odrzucił synchronizację lockfile w trybie `--locked`. Importy backendu i health
  przeszły, ale ta próba nie potwierdza odtwarzalności wersji z lockfile.
- Zweryfikowano w gotowym EXE nazwę, firmę i wersję produktu 0.17.5 oraz zgodność
  dołączonej ikony ICO. PNG/ICO/ICNS mają poprawne formaty i kanał alpha.

To była próba izolowanego runtime na istniejącym Windows, nie pełny przebieg
instalatora na czystej maszynie. Utworzony lokalnie instalator jest **niepodpisany**
i służy do testu; nie został opublikowany w Releases. Nie sprawdzano płatnego API
ani rzeczywistego mikrofonu w tej próbie.
