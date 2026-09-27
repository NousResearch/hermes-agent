# Instrukcja dla agenta AI — jak uruchomić ten projekt

Ten plik jest dla **agenta** (Claude Code, Codex, Cursor, Hermes), który dostaje to repo i ma je uruchomić oraz sprawdzić. Przed publikacją wydania obowiązuje osobna [bramka wydania](product/RELEASE_PIPELINE.md). Człowiek ma własną, krótką instrukcję: [README → Pierwsze uruchomienie](../README.md#pierwsze-uruchomienie). Szybką mapę kodu znajdziesz w [REPO_MAP.md](REPO_MAP.md).

Zasady, które obowiązują zawsze: [AGENTS.md](../AGENTS.md) (root), [apps/desktop/AGENTS.md](../apps/desktop/AGENTS.md), [skills/AGENTS.md](../skills/AGENTS.md).

## 1. Co to jest

Aplikacja desktopowa (Electron + React) na silniku Hermes Agent. Produkt nazywa się **Agent Czesiek** (wcześniej AI Evolution Jarvis). Frontend: `apps/desktop` (Electron + Vite + React). Backend: Python w `hermes_cli/` — uruchamiany jako proces obok okna, rozmawia z interfejsem po `127.0.0.1` (port wybiera sam i ogłasza go w `HERMES_DESKTOP_REMOTE_URL`).

## 2. Wymagania

| Rzecz | Wersja | Po co |
| --- | --- | --- |
| Node.js + npm | 22 | interfejs, Electron, testy JS |
| Python | 3.11+ | silnik (agent, narzędzia, bramy) |
| Git | dowolny | klonowanie, wydania |
| (opcjonalnie) `xvfb` | — | E2E i zrzuty ekranu na Linuksie bez ekranu |

```bash
node -v && npm -v && python3 -V
```

## 3. Uruchomienie krok po kroku

```bash
git clone https://github.com/aievolutionpl/AGENT_CZESIEK.git
cd AGENT_CZESIEK
npm install                     # całe monorepo (workspaces)

cd apps/desktop
npm run dev                     # okno aplikacji w trybie deweloperskim
```

`npm run dev` sprawdza też środowisko Pythona w `.venv` i przy pierwszym starcie instaluje zależności backendu. Dzięki temu samo `npm install` nie pozostawia okna z błędem `No module named 'yaml'`. Instalacja Pythona wymaga dostępu do PyPI.

Zanim uznasz, że „nie działa": pierwszy start buduje renderer i podnosi silnik — to trwa. Sprawdź stan po kodzie wyjścia i po plikach, nie po tym, że okno jest puste.

## 4. Weryfikacja przed każdym commitem

```bash
cd apps/desktop
npm run typecheck               # TypeScript
npm run lint                    # ESLint (0 błędów; ostrzeżenia są dozwolone)
npx vitest run                  # testy interfejsu i logiki
npm run build                   # build produkcyjny
npx playwright test e2e/jarvis-shell-vertical.spec.ts   # E2E w prawdziwym Electronie

cd ../..
scripts/run_tests.sh tests/hermes_cli/    # testy backendu — zawsze tym skryptem
scripts/run_tests.sh tests/skills/test_authoring_standards.py   # standardy umiejętności
```

Na Linuksie bez ekranu: `xvfb-run -a npx playwright test …`.

## 5. Bramka wydania (zanim cokolwiek wypuścisz)

```bash
cd apps/desktop
npm run release:manifest        # zbuduj manifest artefaktów
npm run release:gate:dry        # sprawdź bramkę bez publikowania
```

Bramka (`apps/desktop/scripts/verify-release-gate.mjs`) odrzuca: przycięty instalator, brak `arm64 dmg`, wersję niezgodną z tagiem, niepodpisany instalator w wydaniu tagowanym. Małe pliki aktualizatora (`latest*.yml`, `*.blockmap`) są legalne — nie zgłaszaj ich jako błędu.

## 6. Wydanie

Wydanie powstaje **tylko** przez workflow `Release Desktop`:

1. Zakładka **Actions → Release Desktop → Run workflow** (pole `version`, `platforms`, `publish`) **albo** push taga `vX.Y.Z`.
2. Workflow buduje instalatory, przepuszcza je przez bramkę i tworzy **draft** wydania.
3. Podpisanie: sekrety `APPLE_SIGNING_ENABLED` / `CSC_*` / `WIN_*` / `AZURE_*` + `HERMES_REQUIRE_SIGNING=1` na tagach — szczegóły i koszty: [RELEASE_SIGNING.md](RELEASE_SIGNING.md).
4. **Publikacja draftu to decyzja człowieka** — dopiero ona uruchamia test aktualizacji ze starszej wersji. Nie publikuj sam.

## 7. Jakie klucze trzeba podpiąć

Klucze wpisuje się w **Ustawieniach aplikacji** (szyfrowane lokalnie), nigdy do rozmowy ani do repo.

| Klucz | Wymagany? | Do czego |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | **tak** (minimum) | dostęp do modeli: GPT, Claude, Gemini, DeepSeek — jeden klucz na start |
| `OPENAI_API_KEY` | nie | rozmowa głosem Live (OpenAI Realtime) |
| `GEMINI_API_KEY` | nie | alternatywny głos Live i modele Google |
| `ELEVENLABS_API_KEY` | nie | naturalne głosy (TTS) |
| `TAVILY_API_KEY` | nie | wyszukiwanie w internecie |
| `ANTHROPIC_API_KEY` | nie | modele Claude bez OpenRoutera |

Bez żadnego klucza aplikacja wstaje, ale nie odpowie modelem — onboarding prowadzi wtedy przez krok „Silnik".

Klucze nigdy nie trafiają do repozytorium, testów ani logów. W kodzie czytaj je z konfiguracji silnika, nie z `process.env` na sztywno.

## 8. Gdzie czego szukać

| Temat | Plik |
| --- | --- |
| Zasady pracy w repo | [AGENTS.md](../AGENTS.md) |
| Powłoka, onboarding, pulpit | `apps/desktop/src/app/jarvis/` |
| Umiejętności agenta | `skills/`, [skills/AGENTS.md](../skills/AGENTS.md) |
| Silnik i narzędzia | `agent/`, `tools/`, `hermes_cli/` |
| Bramka i manifest wydania | `apps/desktop/scripts/verify-release-gate.mjs`, `release-manifest.mjs` |
| CI | `.github/workflows/` |
| Architektura produktu | [jarvis-architecture.md](jarvis-architecture.md) |

## 9. Zanim powiesz „gotowe"

- [ ] `npm run typecheck`, `npm run lint`, `npx vitest run` — zielone (albo wyjaśnione).
- [ ] Zmiana działa w prawdziwej aplikacji (`npm run dev` lub E2E), nie tylko w teście jednostkowym.
- [ ] Żaden klucz, hasło ani ścieżka lokalna (`/home/<użytkownik>/`) nie weszła do repo.
- [ ] Wersja w `apps/desktop/package.json` zgadza się z tagiem wydania.
- [ ] Napisałeś, co sprawdziłeś i czym — dowód, nie zapewnienie.
