<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤
<p align="center">
  <a href="https://hermes-agent.nousresearch.com/">Hermes Agent</a> | <a href="https://hermes-agent.nousresearch.com/">Hermes Desktop</a>
</p>
<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Dokumentacja-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Dokumentacja"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/Licencja-MIT-green?style=for-the-badge" alt="Licencja: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Tworzone%20przez-Nous%20Research-blueviolet?style=for-the-badge" alt="Tworzone przez Nous Research"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/Lang-English-blue?style=for-the-badge" alt="English"></a>
</p>

**Samodoskonalący się agent AI zbudowany przez [Nous Research](https://nousresearch.com).** To jedyny agent z wbudowaną pętlą uczenia — tworzy umiejętności na podstawie doświadczenia, ulepsza je w trakcie pracy, sam przypomina sobie o utrwalaniu wiedzy, przeszukuje własne wcześniejsze rozmowy i z sesji na sesję buduje coraz głębszy model tego, kim jesteś. Uruchomisz go na VPS-ie za 5 dolarów, na klastrze GPU albo na infrastrukturze serverless, która w bezczynności kosztuje niemal nic. Nie jest przywiązany do Twojego laptopa — możesz rozmawiać z nim przez Telegram, podczas gdy pracuje na maszynie w chmurze.

Używaj dowolnego modelu — [Nous Portal](https://portal.nousresearch.com), OpenRouter, OpenAI, własny endpoint i [wiele innych](https://hermes-agent.nousresearch.com/docs/integrations/providers). Przełączasz się poleceniem `hermes model` — bez zmian w kodzie, bez uzależnienia od dostawcy.

<table>
<tr><td><b>Prawdziwy interfejs terminalowy</b></td><td>Pełny TUI z edycją wielolinijkową, autouzupełnianiem poleceń ukośnikowych, historią rozmów, przerywaniem i przekierowywaniem pracy oraz strumieniowanym wyjściem narzędzi.</td></tr>
<tr><td><b>Jest tam, gdzie Ty</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal i CLI — wszystko z jednego procesu bramy. Transkrypcja notatek głosowych, ciągłość rozmowy między platformami.</td></tr>
<tr><td><b>Zamknięta pętla uczenia</b></td><td>Pamięć kurowana przez agenta z okresowymi przypomnieniami. Samodzielne tworzenie umiejętności po złożonych zadaniach. Umiejętności same się doskonalą w trakcie użycia. Wyszukiwanie w sesjach przez FTS5 z podsumowaniami LLM dla pamięci międzysesyjnej. Dialektyczne modelowanie użytkownika przez <a href="https://github.com/plastic-labs/honcho">Honcho</a>. Zgodność z otwartym standardem <a href="https://agentskills.io">agentskills.io</a>.</td></tr>
<tr><td><b>Zaplanowane automatyzacje</b></td><td>Wbudowany harmonogram cron z dostarczaniem na dowolną platformę. Codzienne raporty, nocne kopie zapasowe, cotygodniowe audyty — wszystko w języku naturalnym, działające bez nadzoru.</td></tr>
<tr><td><b>Deleguje i zrównolegla</b></td><td>Uruchamiaj odizolowane podagenty do równoległych nurtów pracy. Pisz skrypty w Pythonie, które wywołują narzędzia przez RPC, zwijając wieloetapowe procesy do tur niekosztujących kontekstu.</td></tr>
<tr><td><b>Działa wszędzie, nie tylko na laptopie</b></td><td>Siedem backendów terminala — lokalny, Docker, SSH, Singularity, Modal, Daytona i Vercel Sandbox. Daytona i Modal oferują serverlessową trwałość — środowisko agenta hibernuje w bezczynności i budzi się na żądanie, kosztując między sesjami prawie nic. Uruchom go na VPS-ie za 5 dolarów albo na klastrze GPU.</td></tr>
<tr><td><b>Gotowy do badań</b></td><td>Wsadowe generowanie trajektorii i ich kompresja do trenowania kolejnej generacji modeli wywołujących narzędzia.</td></tr>
</table>

---

## Szybka instalacja

### Linux, macOS, WSL2, Termux

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

### Windows (natywnie, PowerShell)

> **Uwaga:** Natywny Windows uruchamia Hermesa bez WSL — CLI, brama, TUI i narzędzia działają natywnie. Jeśli wolisz WSL2, powyższa komenda dla Linuksa/macOS zadziała również tam. Znalazłeś błąd? [Zgłoś go](https://github.com/NousResearch/hermes-agent/issues).

Uruchom to w PowerShellu:

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

Instalator zajmuje się wszystkim: uv, Python 3.11, Node.js, ripgrep, ffmpeg **oraz przenośny Git Bash** (MinGit, rozpakowany do `%LOCALAPPDATA%\hermes\git` — bez uprawnień administratora, całkowicie odizolowany od systemowej instalacji Gita). Hermes używa tego dołączonego Git Basha do uruchamiania poleceń powłoki.

Jeśli masz już zainstalowanego Gita, instalator go wykryje i użyje. W przeciwnym razie wystarczy pobranie MinGita (~45 MB) — nie ruszy ono ani nie zakłóci żadnej systemowej instalacji Gita.

> **Android / Termux:** Przetestowana ścieżka ręcznej instalacji jest opisana w [przewodniku Termux](https://hermes-agent.nousresearch.com/docs/getting-started/termux). W Termuksie Hermes instaluje wyselekcjonowany dodatek `.[termux]`, ponieważ pełny dodatek `.[all]` pociąga obecnie zależności głosowe niezgodne z Androidem.
>
> **Windows:** Natywny Windows jest w pełni wspierany — powyższa komenda PowerShella instaluje wszystko. Jeśli wolisz WSL2, komenda dla Linuksa też tam zadziała. Natywna instalacja na Windowsie trafia do `%LOCALAPPDATA%\hermes`; instalacje WSL2 do `~/.hermes`, tak jak na Linuksie.

Po instalacji:

```bash
source ~/.bashrc    # przeładuj powłokę (albo: source ~/.zshrc)
hermes              # zacznij rozmawiać!
```

### Rozwiązywanie problemów

#### Windows Defender lub antywirus oznacza `uv.exe` jako złośliwe oprogramowanie

Jeśli Twój antywirus (Bitdefender, Windows Defender itp.) przenosi do kwarantanny `uv.exe` z katalogu `bin` Hermesa (`%LOCALAPPDATA%\hermes\bin\uv.exe`), jest to **fałszywy alarm**. Ten plik to `uv` od Astral — napisany w Ruście menedżer pakietów Pythona, który Hermes dołącza do zarządzania swoim środowiskiem. Silniki antywirusowe oparte na uczeniu maszynowym często oznaczają niepodpisane binaria Rusta, które pobierają i instalują pakiety.

**Aby zweryfikować autentyczność swojej kopii:**

```powershell
# Zainstaluj GitHub CLI, jeśli trzeba
winget install --id GitHub.cli

# Zaloguj się do GitHuba
gh auth login

# Uruchom weryfikację
$uv = "$env:LOCALAPPDATA\hermes\bin\uv.exe"
$ver = (& $uv --version).Split(' ')[1]
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$zip = "$env:TEMP\uv.zip"
Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$ver/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip -UseBasicParsing
gh attestation verify $zip --repo astral-sh/uv
Expand-Archive $zip "$env:TEMP\uv_x" -Force
(Get-FileHash "$env:TEMP\uv_x\uv.exe").Hash -eq (Get-FileHash $uv).Hash
```

Jeśli poświadczenie zwróci „Verification succeeded”, a ostatnia linia wypisze `True`, wszystko jest w porządku.

**Aby dodać Hermesa do wyjątków:**
- **Windows Defender:** Uruchom PowerShell jako administrator → `Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\hermes\bin"`
- **Bitdefender:** Dodaj wyjątek w konsoli Bitdefendera (Ochrona > Antywirus > Ustawienia > Zarządzaj wyjątkami)
- Dodaj do wyjątków **katalog**, a nie skrót pliku — Hermes aktualizuje `uv`, więc skrót zmienia się z każdą wersją

Więcej kontekstu znajdziesz w zgłoszeniach u Astral: [astral-sh/uv#13553](https://github.com/astral-sh/uv/issues/13553), [astral-sh/uv#15011](https://github.com/astral-sh/uv/issues/15011), [astral-sh/uv#10079](https://github.com/astral-sh/uv/issues/10079).

---

## Pierwsze kroki

```bash
hermes              # Interaktywne CLI — rozpocznij rozmowę
hermes model        # Wybierz dostawcę LLM i model
hermes tools        # Skonfiguruj, które narzędzia są włączone
hermes config set   # Ustaw pojedyncze wartości konfiguracji
hermes config get   # Wypisz pojedyncze wartości konfiguracji
hermes gateway      # Uruchom bramę komunikatorów (Telegram, Discord itd.)
hermes setup        # Uruchom pełnego kreatora konfiguracji (ustawia wszystko naraz)
hermes claw migrate # Migracja z OpenClaw (jeśli przychodzisz z OpenClaw)
hermes update       # Zaktualizuj do najnowszej wersji
hermes doctor       # Zdiagnozuj problemy
```

📖 **[Pełna dokumentacja →](https://hermes-agent.nousresearch.com/docs/)**

---

## Po polsku od pierwszego uruchomienia

Hermes mówi po polsku. Ustaw język raz, a prośby o zgodę w CLI, odpowiedzi bramy na polecenia ukośnikowe, panel webowy i aplikacja desktopowa przechodzą na polski:

```yaml
# ~/.hermes/config.yaml
display:
  language: pl
```

Albo dla jednej sesji:

```bash
HERMES_LANGUAGE=pl hermes
```

W panelu webowym i w Hermes Desktop wybierzesz „Polski” prosto z przełącznika języka. Odpowiedzi samego agenta podążają za językiem, w którym do niego piszesz — jeśli chcesz to przypiąć na stałe, napisz mu o tym w SOUL.md albo w prompcie systemowym.

---

## Pomiń zbieranie kluczy API — Nous Portal

Hermes działa z dowolnym dostawcą, którego wybierzesz — to się nie zmienia. Ale jeśli wolisz nie zbierać pięciu osobnych kluczy API do modelu, wyszukiwania w sieci, generowania obrazów, syntezy mowy i przeglądarki w chmurze, **[Nous Portal](https://portal.nousresearch.com)** obejmuje je wszystkie jedną subskrypcją:

- **Ponad 300 modeli** — wybierzesz dowolny przez `/model <nazwa>`
- **Tool Gateway** — wyszukiwanie w sieci (Firecrawl), generowanie obrazów (FAL), synteza mowy (OpenAI), przeglądarka w chmurze (Browser Use) — wszystko przez Twoją subskrypcję. Bez dodatkowych kont.

Jedno polecenie od świeżej instalacji:

```bash
hermes setup --portal
```

Zaloguje Cię przez OAuth, ustawi Nous jako dostawcę i włączy Tool Gateway. Co jest podłączone, sprawdzisz w każdej chwili poleceniem `hermes portal info`. Szczegóły na [stronie dokumentacji Tool Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway).

Nadal możesz używać własnych kluczy dla poszczególnych narzędzi — brama działa per backend, a nie na zasadzie „wszystko albo nic”.

---

## CLI kontra komunikatory — szybkie porównanie

Hermes ma dwa punkty wejścia: uruchom interfejs terminalowy poleceniem `hermes` albo uruchom bramę i rozmawiaj z nią przez Telegram, Discord, Slack, WhatsApp, Signal lub e-mail. Gdy jesteś już w rozmowie, wiele poleceń ukośnikowych działa tak samo w obu interfejsach.

| Akcja                                    | CLI                                           | Platformy komunikacyjne                                                                |
| ---------------------------------------- | --------------------------------------------- | -------------------------------------------------------------------------------------- |
| Rozpocząć rozmowę                        | `hermes`                                      | Uruchom `hermes gateway setup` + `hermes gateway start`, a potem napisz do bota          |
| Rozpocząć nową rozmowę                   | `/new` albo `/reset`                          | `/new` albo `/reset`                                                                     |
| Zmienić model                            | `/model [dostawca:model]`                     | `/model [dostawca:model]`                                                                |
| Ustawić osobowość                        | `/personality [nazwa]`                        | `/personality [nazwa]`                                                                   |
| Ponowić albo cofnąć ostatnią turę        | `/retry`, `/undo`                             | `/retry`, `/undo`                                                                        |
| Skompresować kontekst / sprawdzić zużycie | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [dni]`                                                 |
| Przeglądać umiejętności                  | `/skills` albo `/<nazwa-umiejętności>`        | `/<nazwa-umiejętności>`                                                                  |
| Przerwać bieżącą pracę                   | `Ctrl+C` albo wyślij nową wiadomość           | `/stop` albo wyślij nową wiadomość                                                       |
| Status zależny od platformy              | `/platforms`                                  | `/status`, `/sethome`                                                                    |

Pełne listy poleceń znajdziesz w [przewodniku CLI](https://hermes-agent.nousresearch.com/docs/user-guide/cli) i [przewodniku bramy komunikatorów](https://hermes-agent.nousresearch.com/docs/user-guide/messaging).

---

## Dokumentacja

Cała dokumentacja mieszka pod adresem **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)**:

| Sekcja                                                                                              | Co obejmuje                                                      |
| --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------- |
| [Szybki start](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart)               | Instalacja → konfiguracja → pierwsza rozmowa w 2 minuty          |
| [Użycie CLI](https://hermes-agent.nousresearch.com/docs/user-guide/cli)                             | Polecenia, skróty klawiszowe, osobowości, sesje                  |
| [Konfiguracja](https://hermes-agent.nousresearch.com/docs/user-guide/configuration)                 | Plik konfiguracyjny, dostawcy, modele, wszystkie opcje           |
| [Brama komunikatorów](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)              | Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant       |
| [Bezpieczeństwo](https://hermes-agent.nousresearch.com/docs/user-guide/security)                    | Zatwierdzanie poleceń, parowanie DM, izolacja w kontenerach      |
| [Narzędzia i zestawy](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools)         | Ponad 40 narzędzi, system zestawów, backendy terminala           |
| [System umiejętności](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills)        | Pamięć proceduralna, Skills Hub, tworzenie umiejętności          |
| [Pamięć](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)                     | Pamięć trwała, profile użytkownika, dobre praktyki               |
| [Integracja MCP](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp)                | Podłącz dowolny serwer MCP, aby rozszerzyć możliwości            |
| [Harmonogram cron](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron)             | Zaplanowane zadania z dostarczaniem na platformy                 |
| [Pliki kontekstu](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files)     | Kontekst projektu kształtujący każdą rozmowę                     |
| [Architektura](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture)             | Struktura projektu, pętla agenta, kluczowe klasy                 |
| [Współtworzenie](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing)           | Konfiguracja środowiska, proces PR, styl kodu                    |
| [Referencja CLI](https://hermes-agent.nousresearch.com/docs/reference/cli-commands)                 | Wszystkie polecenia i flagi                                      |
| [Zmienne środowiskowe](https://hermes-agent.nousresearch.com/docs/reference/environment-variables)  | Pełna referencja zmiennych środowiskowych                        |

---

## Migracja z OpenClaw

Jeśli przychodzisz z OpenClaw, Hermes potrafi automatycznie zaimportować Twoje ustawienia, wspomnienia, umiejętności i klucze API.

**Podczas pierwszej konfiguracji:** Kreator (`hermes setup`) automatycznie wykrywa `~/.openclaw` i proponuje migrację jeszcze przed rozpoczęciem konfiguracji.

**W dowolnym momencie po instalacji:**

```bash
hermes claw migrate              # Migracja interaktywna (pełny zestaw)
hermes claw migrate --dry-run    # Podgląd tego, co zostałoby zmigrowane
hermes claw migrate --preset user-data   # Migracja bez sekretów
hermes claw migrate --overwrite  # Nadpisz istniejące konflikty
```

Co jest importowane:

- **SOUL.md** — plik osobowości
- **Wspomnienia** — wpisy z MEMORY.md i USER.md
- **Umiejętności** — umiejętności utworzone przez użytkownika → `~/.hermes/skills/openclaw-imports/`
- **Lista dozwolonych poleceń** — wzorce zatwierdzeń
- **Ustawienia komunikatorów** — konfiguracje platform, dozwoleni użytkownicy, katalog roboczy
- **Klucze API** — sekrety z listy dozwolonych (Telegram, OpenRouter, OpenAI, Anthropic, ElevenLabs)
- **Zasoby TTS** — pliki audio z przestrzeni roboczej
- **Instrukcje przestrzeni roboczej** — AGENTS.md (z `--workspace-target`)

Wszystkie opcje znajdziesz w `hermes claw migrate --help` albo skorzystaj z umiejętności `openclaw-migration`, aby przejść migrację interaktywnie, z podglądem na sucho.

---

## Współtworzenie

Chętnie przyjmujemy wkład! Zajrzyj do [przewodnika dla współtwórców](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) po konfigurację środowiska, styl kodu i proces PR.

Szybki start dla współtwórców — użyj standardowego instalatora, a potem pracuj na pełnym checkoucie gita, który tworzy w `$HERMES_HOME/hermes-agent` (zwykle `~/.hermes/hermes-agent`). Odpowiada to układowi używanemu przez `hermes update`, zarządzane venv, leniwe zależności, bramę i narzędzia dokumentacji.

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

Awaryjna ścieżka z ręcznym klonowaniem (do jednorazowych klonów i CI, gdzie świadomie nie chcesz zarządzanego układu instalacji):

Utwórz venv poza sklonowanym drzewem źródeł — venv wewnątrz katalogu, w którym działa agent, może zostać skasowany przez polecenie z relatywną ścieżką uruchomione przez agenta na własnym checkoucie, niszcząc działające środowisko w trakcie sesji.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/.hermes/venvs/hermes-dev --python 3.11
source ~/.hermes/venvs/hermes-dev/bin/activate
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

---

## Społeczność

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Zgłoszenia](https://github.com/NousResearch/hermes-agent/issues)
- 🔌 [computer-use-linux](https://github.com/avifenesh/computer-use-linux) — serwer MCP do sterowania pulpitem Linuksa dla Hermesa i innych hostów MCP, z drzewami dostępności AT-SPI, obsługą wejścia w Wayland/X11, zrzutami ekranu i celowaniem w okna kompozytora.
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — społecznościowy mostek WeChat: uruchom Hermes Agent i OpenClaw na tym samym koncie WeChat.

---

## Licencja

MIT — zobacz [LICENSE](LICENSE).

Tworzone przez [Nous Research](https://nousresearch.com).
