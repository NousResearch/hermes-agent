# Instalacja Agent Czesiek na nowym komputerze

Nie trzeba wcześniej instalować Hermes Agent, Pythona ani konfigurować CLI.
Pełny instalator Windows zawiera aplikację, kod silnika, Python, jego zależności,
Git Bash i Node.js. Pierwszy start nie klonuje repozytorium ani nie uruchamia pip.
Dla nowego współpracownika kod startuje z `resources/runtime`, a dane użytkownika pozostają poza instalacją.

Do instalacji i uruchomienia lokalnego backendu nie potrzeba internetu. Rozmowy
z Gemini, OpenRouter i innymi usługami API wymagają internetu i własnych kluczy.
Opcjonalne narzędzia, modele lokalne i integracje mogą mieć dodatkowe wymagania;
nie są obietnicą całkowicie offline działającego asystenta.

## Wybór współpracownika przy pierwszym starcie

Czesiek prowadzi rozmowę i planuje pracę. Hermes jest silnikiem wykonującym zadania — jego współpracownikiem.

- **Przygotuj nowego współpracownika**: pełny pakiet Windows uruchamia dołączony silnik bez pobierania repozytorium i bez pip.
- **Mam już Hermesa**: aplikacja sprawdza standardowe katalogi instalacji oraz środowisko Python. Możesz również wskazać folder zawierający `hermes_cli` i `venv` lub `.venv`.

Wybór jest zapisywany w `runtime-collaborator.json` w katalogu danych aplikacji.
Kolejny start używa tego samego silnika, bez ponownej instalacji. Jeśli zapisany
folder zniknie lub zapis będzie uszkodzony, wybór pojawi się ponownie. Opcja naprawy
po błędzie pozwala ponownie wybrać współpracownika.

Istniejąca instalacja oznacza ponowne użycie jej kodu i Pythona, a nie przejęcie
uruchomionej sesji Hermesa. Czesiek uruchamia własny proces z własnym profilem.
Nie kopiuje kluczy, historii ani ustawień innego Hermesa. Zewnętrzna wersja silnika
może mieć inny zestaw funkcji; dołączony silnik jest wariantem testowanym z tym wydaniem.

## Własne środowisko

Pakiet Cześka używa własnego katalogu danych:

- Windows: `%LOCALAPPDATA%\AI Evolution Jarvis\hermes-home`.
- macOS/Linux: `~/.ai-evolution-jarvis/hermes-home`.

Historyczna nazwa katalogu pozostaje dla zgodności z aktualizacjami i zapisanymi
profilami. Widoczna nazwa produktu to Agent Czesiek. Pakiet nie wybiera
automatycznie Hermesa znalezionego na PATH ani modułu z systemowego Pythona.
Jawne połączenie zdalne lub override deweloperski pozostają osobną możliwością.
Instalacja deweloperska ze źródeł ma własne reguły wykrywania backendu.

## Pamięć ogólna Cześka (vault + Obsidian)

Domyślna pamięć ogólna to vault Obsidiana w:

- Windows: `%USERPROFILE%\Documents\Czesiek Vault`
- macOS/Linux: `~/Documents/Czesiek Vault`

Dlaczego tam: Obsidian domyślnie otwiera vaulty z Documents, katalog jest
per-użytkownik (bez uprawnień administratora, pisanie po prostu działa) i przeżywa
aktualizację aplikacji. Zawartość to zwykły markdown, który użytkownik może czytać
i edytować ręcznie — vault jest wspólnym zeszytem, nie zamkniętym systemem.

### Co robi instalator

Instalator NSIS (`apps/desktop/scripts/installer.nsh`, makro `customInstall`) po
skopiowaniu plików aplikacji uruchamia `resources\bootstrap\install-obsidian.ps1`,
który:

1. sprawdza, czy Obsidian już jest (typowe ścieżki `Obsidian.exe`, wpisy
   odinstalowania w rejestrze, `winget list`) — jeśli tak, nic nie robi;
2. próbuje `winget install -e --id Obsidian.Obsidian --scope user --silent
   --accept-package-agreements --accept-source-agreements` (winget pobiera
   instalator od wydawcy), a gdy to zawiedzie — drugą próbę bez `--scope user`;
3. awaryjnie pobiera oficjalny instalator Windows z `https://obsidian.md/download`
   (strona odsyła do wydania `obsidianmd/obsidian-releases`) i uruchamia go cicho
   (`/S`), po czym kasuje plik tymczasowy;
4. jeśli wszystko zawiedzie — **nie przerywa instalacji Cześka**: zapisuje
   komunikat w `%LOCALAPPDATA%\AI Evolution Jarvis\hermes-home\logs\obsidian-install.log`
   i pokazuje okno z komendą do ręcznej instalacji.

**Nie pakujemy binarek Obsidiana do instalatora ani do repozytorium.** Obsidian
jest darmowy, ale zamknięty; jego licencja nie pozwala na redystrybucję. Instalacja
odbywa się zawsze z oficjalnego źródła, po stronie użytkownika.

### Co robi aplikacja przy pierwszym starcie

`apps/desktop/electron/vault-seed.ts`, wołane z `electron/main.ts` przy każdym
starcie (idempotentnie i nie-fatalnie):

- kopiuje szablon pamięci z `resources/vault-seed` (w repo:
  `apps/desktop/build/vault-seed`) do vaultu, tworząc brakujące katalogi i **nie
  nadpisując** istniejących plików użytkownika;
- dopisuje `OBSIDIAN_VAULT_PATH=<vault>` do `…\hermes-home\.env`, jeśli tego klucza
  jeszcze tam nie ma (reszta pliku bez zmian);
- dopisuje protokół pamięci do `…\hermes-home\AGENTS.md` (tworzy plik, jeśli go nie
  ma; istniejącej treści nie nadpisuje);
- przy pierwszym zasianiu vaultu otwiera go w Obsidianie przez
  `obsidian://open?path=…`.

Pliki startowe vaultu: `README.md`, `00_KIM_JESTEM.md`, `01_AKTUALNY_KONTEKST.md`,
`02_PAMIEC_TRWALA.md`, `03_LUDZIE/README.md`, `04_PROJEKTY/README.md`,
`05_RYTUALY/DZIENNIK.md`, `05_RYTUALY/PORANNY_PRZEGLAD.md`,
`06_DECYZJE/README.md`, `99_ZADANIA.md`.

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
3. Pierwszy start: wybór nowego lub istniejącego współpracownika, dojście do onboardingu bez pobierania silnika.
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


### Poprawka pierwszego startu po próbie instalacji

- Skrypty instalacji są w `resources/bootstrap`, więc ich uruchomienie nie wymaga
  pobierania z raw.githubusercontent.com (zgłoszony HTTP 429).
- Ekran pierwszej instalacji oferuje instalację lokalną z logo Cześka.
- Nowe profile zaczynają w jasnym motywie; zapisany wybór pozostaje zachowany.
- Przycisk zakończenia NSIS uruchamia EXE bezpośrednio, bez zależności od skrótu
  w menu Start. Skróty nadal są tworzone przez instalator.
- Internet nadal jest wymagany do pobrania silnika i zależności.


## Budowanie pełnego instalatora Windows

`apps/desktop/scripts/stage-windows-runtime.mjs` pakuje wyłącznie śledzone pliki
kodu z listy katalogów runtime oraz czystą dystrybucję Pythona i przetestowane
`site-packages`. Nie wskazuj profilu użytkownika jako źródła. Klucze, czaty,
konfiguracje i pliki `.env` nie mogą być częścią materiału do wydania.

```powershell
node apps/desktop/scripts/stage-windows-runtime.mjs --python-root=<standalone-Python> --site-packages=<clean-venv/Lib/site-packages> --git-root=<Git-distribution> --node-root=<Node-distribution> --node-license=<matching-Node-LICENSE>
```

Opcjonalne `--output` pozwala użyć innego dysku; `apps/desktop/build/runtime`
musi wtedy wskazywać wynikowy katalog. Źródłowy interpreter musi być pełną,
przenośną dystrybucją, nie samym launcherem z venv. Skrypt usuwa powiązania
editable z maszyną budującą i zapisuje wersje pakietów w `manifest.json`.
Przygotuj zależności w czystym środowisku zgodnie z `pyproject.toml`, a przed
wydaniem sprawdź manifest i przetestuj wynik po przeniesieniu do nowej ścieżki.
Pakowanie odrzuca runtime z innego commita lub architektury.

Następnie wykonaj build i pakowanie NSIS. Aktualizacja wymienia runtime razem
z aplikacją. Nie aktualizuj silnika przez git/pip w katalogu zainstalowanego
programu. Licencje Hermesa, Pythona, Git, Node i pakietów pozostają w paczce.


## macOS: instalator DMG

Dla macOS (Apple silicon i Intel) instalatorem jest plik `.dmg` z Releases; zwykła
instalacja to przeciągnięcie aplikacji do Programów. Jednym poleceniem (pobiera
wydanie dla Twojej architektury, kopiuje do `/Applications` lub `~/Applications`):

```bash
curl -fsSL https://raw.githubusercontent.com/aievolutionpl/AGENT_CZESIEK/main/scripts/install-jarvis.sh | bash
```

### Budowanie DMG na Macu

```bash
scripts/build-macos-installer.sh            # DMG z wbudowanym silnikiem (bez internetu przy pierwszym starcie)
scripts/build-macos-installer.sh --online   # mniejszy DMG; silnik pobiera się przy pierwszym starcie
scripts/build-macos-installer.sh --sign     # podpis Developer ID z kluczenika (zob. RELEASE_SIGNING.md)
```

Skrypt pobiera przenośnego Pythona 3.11 (uv), instaluje zależności z `uv.lock`,
składa `apps/desktop/build/runtime` przez `stage-macos-runtime.mjs` i buduje
aplikację. Budujesz architekturę tego Maca; drugą zbuduj na drugim Macu. Pakowanie
odrzuca runtime z innego commita lub architektury (`before-pack.mjs`).
Bez `runtime/manifest.json` aplikacja działa jak dotąd: silnik instaluje
`bootstrap/install.sh` przy pierwszym starcie (wymaga internetu).

Wbudowany silnik macOS zawiera Python i zależności; używa systemowego `bash` i `git`
(Xcode Command Line Tools) i nie pakuje Node.js (opcjonalne `--node-root` w
`stage-macos-runtime.mjs`). Dane użytkownika pozostają w `~/.ai-evolution-jarvis/hermes-home`.

### Obsidian na macOS

Bez instalatora NSIS robi to aplikacja przy pierwszym zasianiu vaultu:
`resources/bootstrap/install-obsidian.sh` w tle (Homebrew `--cask obsidian`, a bez
niego oficjalny DMG do `~/Applications`), z logiem w `…/hermes-home/logs/obsidian-install.log`.
Niepowodzenie nie blokuje aplikacji. Binarek Obsidiana nie redystrybuujemy.

### Wydanie w CI

`release-desktop.yml` buduje osobno Windows x64, macOS arm64 (`macos-latest`),
macOS x64 (`macos-15-intel`) i Linux. Legi Windows i macOS najpierw składają
wbudowany silnik (`scripts/stage-windows-runtime.sh`, `scripts/stage-macos-runtime.sh`),
bo silnik jest zależny od architektury. Bramka na poziomie legu sprawdza tylko jego
architekturę (`verify-release-gate.mjs --archs`); bramka zbiorcza nadal wymaga
obu DMG i obu ZIP. Nie uruchamiano tego workflow: pierwszy przebieg (najlepiej
`workflow_dispatch` bez publikacji) jest próbą, a nazwa runnera Intela może wymagać korekty.

### Niepodpisany build

Bez certyfikatu Developer ID DMG jest podpisany ad-hoc. Gatekeeper pokaże
ostrzeżenie: kliknij aplikację prawym przyciskiem → Otwórz. Instalacja skryptem
`install-jarvis.sh` (curl) nie ustawia flagi kwarantanny. Potwierdzenie na czystym
Macu (nowe konto, brak Pythona/Hermesa) nadal jest wymagane przed publikacją —
ta zmiana nie była uruchamiana na macOS.
