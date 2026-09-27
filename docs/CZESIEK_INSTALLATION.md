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
