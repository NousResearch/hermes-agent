# Mapa repozytorium Agent Czesiek

Zacznij od tej strony. Wybierz obszar z tabeli, przeczytaj jego `AGENTS.md` i
dopiero wtedy szukaj symbolu w tym katalogu (`rg -n 'nazwa' <katalog>`). Nie
ładuj całego repo, całej dokumentacji ani wszystkich testów na starcie.
Główne [AGENTS.md](../AGENTS.md) zawiera zasady wspólne, a
[AGENT_SETUP.md](AGENT_SETUP.md) — uruchomienie i kontrole.

| Zmiana dotyczy | Punkt wejścia | Instrukcje obszaru |
| --- | --- | --- |
| Okno desktopowe, onboarding, orb, menu, ustawienia | `apps/desktop/src/app/`, `apps/desktop/src/plugins/`, `apps/desktop/electron/` | [`apps/desktop/AGENTS.md`](../apps/desktop/AGENTS.md), [`apps/desktop/src/AGENTS.md`](../apps/desktop/src/AGENTS.md) |
| Transport desktop ↔ backend | `apps/shared/`, `tui_gateway/` | [`tui_gateway/AGENTS.md`](../tui_gateway/AGENTS.md) |
| Rozmowa, model, pamięć i pętla agenta | `run_agent.py`, `agent/` | [`agent/AGENTS.md`](../agent/AGENTS.md) |
| Narzędzia, przeglądarka, delegowanie | `model_tools.py`, `toolsets.py`, `tools/` | [`tools/AGENTS.md`](../tools/AGENTS.md) |
| Konfiguracja, profile, komendy CLI | `cli.py`, `hermes_cli/` | [`hermes_cli/AGENTS.md`](../hermes_cli/AGENTS.md) |
| Komunikatory i gateway | `gateway/`, `gateway/platforms/` | [`gateway/AGENTS.md`](../gateway/AGENTS.md) |
| Zadania cykliczne | `cron/` | [`cron/AGENTS.md`](../cron/AGENTS.md) |
| Rozszerzenia i umiejętności | `plugins/`, `skills/` | [`plugins/AGENTS.md`](../plugins/AGENTS.md), [`skills/AGENTS.md`](../skills/AGENTS.md) |
| Panel webowy | `web/`, `hermes_cli/web_routers/` | [`web/AGENTS.md`](../web/AGENTS.md) |
| Strona sprzedażowa Cześka | `sales-site/dist/` | [`sales-site/README.md`](../sales-site/README.md); osobna statyczna strona, bez zmian w aplikacji |
| Testy i pipeline | `tests/`, `tests-js/`, `.github/workflows/`, `scripts/` | [testowanie w głównym AGENTS](../AGENTS.md#testing-applies-everywhere) |

Najkrótsza ścieżka zmiany: znajdź właściciela zachowania w tabeli → przeczytaj
lokalne instrukcje → odtwórz błąd → zmień najmniejszy właściwy moduł → uruchom
testy tego obszaru. Testy Pythona uruchamiaj **wyłącznie** przez
`scripts/run_tests.sh`; dla desktopu użyj skryptów z `apps/desktop/package.json`.
Nie zmieniaj identyfikatorów instalacji, protokołu i profili tylko po to, by
usunąć starszą nazwę — mogą być potrzebne do zgodności z aktualizacjami.

Wydanie jest osobnym etapem: [pipeline](product/RELEASE_PIPELINE.md),
[podpisywanie](RELEASE_SIGNING.md) i bramka
`apps/desktop/scripts/verify-release-gate.mjs`. Zielony test lokalny ani scalony
PR nie potwierdzają podpisanego instalatora. Kod Hermes ma warunki [MIT](../LICENSE),
a autorskie zasoby marki mają [osobne warunki](../CZESIEK-ASSETS-LICENSE.md).
Sekrety i dane profilu pozostają poza repozytorium.

## Własne skille, role i orb

- Import z tekstu i kreator: `apps/desktop/src/app/skills/skill-create.tsx`;
  walidacja/podgląd: `skill-draft.ts`; zapis: `src/api/skills.ts` →
  `hermes_cli/web_routers/skills.py` → istniejący `tools/skill_manager_tool.py`.
  Profil jest częścią routingu żądania; test integracyjny:
  `tests/hermes_cli/test_web_server_skill_editor.py`.
- Role asystenta: `apps/desktop/src/app/agents/presets.ts`, `preset-store.ts`
  (zapis CAS), `launch-preset.ts` (przygotowanie nowego zadania).
- Kolor i ruch orba: `apps/desktop/src/app/jarvis/plasma.ts`,
  `plasma-palette.ts`, `particle-orb.ts`; podpis pływającej kuli:
  `src/app/pet-overlay/orb-overlay.tsx`.

Instrukcja użytkownika: [własne umiejętności i role](SKILLS_AND_ASSISTANT.md).


### Windows: silnik w instalatorze i wybór współpracownika

- `apps/desktop/electron/runtime-collaborator.ts`: wykrywanie gotowego środowiska i zapis wyboru; własny profil Cześka pozostaje oddzielny.
- `apps/desktop/src/components/collaborator-picker.tsx`: automatyczne wykrywanie i ręczny wybór folderu w pierwszym uruchomieniu.

- `apps/desktop/electron/bundled-runtime.ts`: wybór dołączonego silnika przed starym bootstrapem.
- `apps/desktop/scripts/stage-windows-runtime.mjs`: pakowanie kodu, Pythona, zależności, Git i Node.
- `apps/desktop/scripts/before-pack.mjs`: zgodność commita i architektury pakietu.
- `docs/CZESIEK_INSTALLATION.md`: instrukcja budowania i granice działania offline.

### macOS: DMG i wbudowany silnik
- `scripts/build-macos-installer.sh`: jedna komenda → DMG/ZIP; `apps/desktop/scripts/stage-macos-runtime.mjs` składa `build/runtime` (odpowiednik wersji Windows).
- `apps/desktop/electron/bundled-runtime.ts`: na darwin używa runtime tylko gdy jest `manifest.json`, inaczej bootstrap online; `before-pack.mjs` pilnuje commita i architektury.
- `scripts/install-obsidian.sh` + `installObsidianOnMac` w `electron/main.ts`: Obsidian w tle przy pierwszym starcie.


## Diagnostyka, współpraca i odzyskiwanie

- `apps/desktop/src/app/settings/assistant-health.tsx`: panel „Czy wszystko działa?”, testy API/audio bez ujawniania sekretów; `health-error.ts` tłumaczy błędy.
- `apps/desktop/src/app/chat/composer/hooks/use-realtime-conversation.ts`: szybkie przyjęcie zlecenia głosowego, sterowanie podagentami i raporty; `status-stack/work-results.tsx` zachowuje wyniki po zakończeniu.
- `apps/desktop/electron/update-recovery.ts`: kopia aplikacji i danych, przywracanie z zachowaniem obecnej wersji; `recovery-controller.ts` zatrzymuje backend i blokuje równoległy start podczas kopii.
- `apps/desktop/src/app/settings/recovery-settings.tsx`: wspólny panel kopii w ustawieniach i przy błędzie startu.
- `apps/desktop/scripts/prepare-clean-windows-test.ps1`: przygotowanie testu w Windows Sandbox. Samo przygotowanie nie jest potwierdzeniem instalacji.

### Rozmowa Live i pulpit
- `apps/desktop/src/app/jarvis/live-model-picker.tsx`: osobny wybór modelu głosu, zapis przypięty do połączenia i profilu; model wykonawczy pozostaje bez zmian.
- `apps/desktop/src/app/jarvis/use-live-autostart.ts`: start rozmowy po konfiguracji, raz na profil/połączenie podczas uruchomienia aplikacji; wymaga dostępnego klucza i zgody na mikrofon.
- `apps/desktop/src/app/jarvis/home-hero.tsx`: orb, przyciski rozmowy i propozycje zadań.
- Starszy współpracownik bez adaptera Live jest pomijany; aplikacja korzysta z dołączonego silnika. `scripts/smoke-bundled-runtime.mjs` sprawdza także status Live i brak klucza (400 zamiast 405).

### Pamięć w vaulcie Obsidiana, głos i kursor
- `agent/vault_notes.py` + `hermes_cli/web_routers/vault.py` (`/api/vault/*`): notatki markdown vaultu (`OBSIDIAN_VAULT_PATH`, domyślnie `~/Documents/Czesiek Vault`) jako graf — węzły to notatki, krawędzie to `[[wikilinki]]`; odczyt, zapis, tworzenie, usuwanie do `.trash`. Ścieżki spoza vaultu są odrzucane. Test: `tests/agent/test_vault_notes.py`.
- `apps/desktop/src/app/starmap/vault-view.tsx` (widok + edytor notatki), `vault-graph.tsx` (graf sił na canvasie), `src/api/vault.ts`: domyślna zakładka „Mapa wiedzy”; obok stare zakładki Umiejętności i Pamięć (graf nauczonych umiejętności).
- `apps/desktop/src/app/jarvis/voice-aura.tsx`: pierścień widma głosu wokół orba, napędzany realnym `$micLevel`; dok głosu (`voice-controls.tsx`) to tylko cztery minimalne przyciski.
- `tools/browser_cursor_overlay.py`: kursor agenta w oknie przeglądarki (gradientowa strzałka z plakietką „Czesiek”, halo, ogon, fala przy kliknięciu); natywny kursor sterownika komputera: `tools/computer_use/cursor_overlay.py`.

### Aktualizacje: skąd aplikacja wie o nowej wersji
- `hermes_cli/distribution.py` (silnik) i `apps/desktop/electron/distribution.ts` (aplikacja): **jedyne** miejsca, które nazywają oficjalne repozytorium tej dystrybucji (`aievolutionpl/AGENT_CZESIEK`). Sprawdzanie aktualizacji, wykrywanie forka, pobieranie ZIP i linki do wydań z nich wynikają. Nie wpisuj adresu repo nigdzie indziej; dla lustra ustaw `HERMES_DISTRIBUTION_REPO=owner/name`.
- Zainstalowana aplikacja nie ma checkoutu gita, więc „Sprawdź aktualizacje” w Ustawieniach → O aplikacji (`settings/release-update-card.tsx`) czyta ostatnie Wydanie z GitHuba (`electron/release-check.ts`, kanał `hermes:updates:release-check`), porównuje wersje i otwiera instalator dla tego systemu. 404 = repozytorium prywatne albo brak wydania; aplikacja mówi to wprost. Samo wydanie nadal buduje i podpisuje [pipeline](product/RELEASE_PIPELINE.md).
- `scripts/upstream_watch.py` + `.github/workflows/upstream-watch.yml` (co poniedziałek): lista zmian w upstream Hermes od ostatniego przeglądu, pogrupowana po obszarach, z plikami zmienionymi po obu stronach (tam będą konflikty). Historia gita forka jest niezależna od upstreamu, więc punkt startu zapisuje `docs/upstream/WATCH.json` (SHA po pierwszym przeglądzie: `python scripts/upstream_watch.py --mark-reviewed <sha>`). Wynik trafia do jednego issue z etykietą `upstream-watch`; nic nie jest scalane automatycznie.

### Kula na pulpicie, dźwięki i wygląd
- `apps/desktop/src/app/pet-overlay/orb-overlay.tsx` + `orb-overlay.css`: kula na pulpicie (okno przezroczyste). Rozmiar to jedna liczba (`--orb-scale`, 0.5–1.6): kółko myszy nad kulą, uchwyt na jej krawędzi albo przyciski −/+. Rozmiar zmienia się wokół środka, kula nie ucieka poza ekran i po puszczeniu przy krawędzi „przykleja się” do niej. Reguły (skala, środek, kotwiczenie, przyciąganie): `app/jarvis/desktop-orb-geometry.ts`; zapis rozmiaru: `desktop-orb-state.ts` (`czesiek.orb-scale.v1`, wspólny dla obu okien).
- `apps/desktop/src/lib/ui-sound.ts`: ciche dźwięki interfejsu (WebAudio, bez plików). Podpięte pod istniejące `triggerHaptic(intent)`, więc dźwięk dostaje każde miejsce, które już oznaczało świadome kliknięcie; przycisk wyciszenia na pasku tytułu wycisza też je, a Ustawienia → Wygląd → „Dźwięki interfejsu” wyłącza tylko dźwięki. Strumieniowanie ma własne dźwięki, więc jest pominięte.
- Kolory: `--czesiek-grad` i `--czesiek-glow` w `app/jarvis/glass.css` to jeden gradient marki (przycisk główny, znacznik aktywnej pozycji menu, logotyp).

### Połączenia: Google, role, przeglądarka
- Google bez ręcznej konfiguracji przez agenta: `agent/google_connect.py` + `hermes_cli/web_routers/google_connect.py` (`/api/google/*`) prowadzą istniejący skill `google-workspace` (status z plików, zapis pliku klienta OAuth, dwa kroki logowania, prawdziwy test odczytu kalendarza i poczty). Kreator: `apps/desktop/src/app/connections/google-connect-dialog.tsx` (wznawia od ostatniego kroku). Własna aplikacja OAuth marki (logowanie jednym kliknięciem) wymaga weryfikacji Google i nie jest zrobiona.
- Stan połączeń (`agent/connection_status.py`, `/api/connections/status`): tylko obecność plików i poświadczeń, bez sieci i bez wartości; co nieznane, to `unknown` (bez plakietki). Używają go plakietki na stronie Integracje i karta „Połącz więcej” w panelu bocznym (`jarvis/rail-connect-card.tsx`, `suggestConnections`).
- Onboarding: wybór roli (`connections-catalog.ts`: `JARVIS_ROLE_CONNECTIONS`, `selectionForRole`) zaznacza polecane narzędzia; każdy wybór można zmienić.
- Raport dnia dostaje sekcję Google (dzisiejsze spotkania i nieprzeczytane maile) z `/api/briefing`, gdy Google jest połączony; treść maili idzie do modelu jako dane, nie polecenia.
- Ustawienia → Narzędzia → Przeglądarka: „Zablokowane strony” (`settings/browser-sites-panel.tsx`) edytuje `security.website_blocklist`, które backend egzekwuje przy nawigacji przeglądarki i w narzędziach sieciowych. Pytanie „czy wejść na tę stronę” przed wizytą nie istnieje.

## Browser modes (agent's browser + Google login)

- `agent/browser_modes.py` — status / `set_mode` (`managed` | `own` | `copy`) / `open_sign_in` / `import_now` / `clear_*`. Reuses `hermes_cli/browser_connect.py` (own `chrome-debug` profile, real-profile snapshot). Login check reads cookie **names/hosts only** from a temp copy; values are never returned.
- `hermes_cli/web_routers/browser.py` — `/api/browser/*`. Desktop: `src/api/browser.ts`, `src/app/settings/browser-modes-panel.tsx` (Skills → Browser). Tests: `tests/agent/test_browser_modes.py`.
- Driving the live default profile is intentionally not offered (Chrome ≥136 blocks CDP on it); `copy` is the supported cookie import.

## Integrations that pay off at once

- **Raport dnia** — `hermes_cli/web_routers/briefing.py` adds open kanban tasks (`summarize_tasks`) next to Google mail/calendar, news, sessions and jobs; the prompt lives in `apps/desktop/src/app/jarvis/briefing.ts`.
- **Dysk → pamięć** — `agent/drive_memory.py` (folders → `Drive/<folder>/*.md` notes in the vault, link to source, incremental by `modifiedTime`, `STEP_LIMIT` files per call), router `web_routers/drive_memory.py`, UI `src/app/connections/drive-memory-panel.tsx` (shown on the connected Google card).
- **Sklep MCP** — already present: `src/app/skills/mcp-tab.tsx` + `/api/mcp/catalog`.
- **Polski kontekst** — `optional-skills/finance/fakturownia` (read-only, API not yet run against a live account).

## Security foundation (untrusted content, secrets)

- `agent/external_content.py` — `fence()` wraps text written by others in `<external-data source="…">` (tags inside are neutralised). Used by `vault_notes.create_note(external_source=…)` (news saved from the rail), `agent/drive_memory.py`; `read_note` reports `external`. The agent's standing rule is the second block in `AGENTS.md` (`electron/vault-seed.ts`, `AGENTS_EXTERNAL_MARKER`), appended to existing installs too.
- `agent/secret_audit.py` + `web_routers/secrets_audit.py` — presence-only audit of credential files/folders and one-click owner-only permissions; UI `connections/secrets-audit-card.tsx` (Keys tab). OS keychain storage is not implemented yet.

## Premium UI pass (dashboard rail, composer, pages)

- `apps/desktop/src/components/model-brand-icon.tsx` — `ModelBrandIcon` / `resolveModelBrand`: one maker mark per model (simple-icons where they exist, drawn glyphs for GPT, DeepSeek, Hermes, free); used by the model card, the live-voice picker and preset rows.
- `src/app/jarvis/glass.css` (bottom sections) — gradient-hairline `.jarvis-panel`, `.jarvis-icon-chip`, `.model-brand-tile`, `.jarvis-choice`, `.jarvis-segment`, `.jarvis-menu`, `.jarvis-rise` entrance animation; composer glass controls `.jarvis-ctl*` (classes set in `chat/composer/control-classes.ts`); list/detail surfaces `.jarvis-page-list|detail` and `.jarvis-row` (set in `master-detail.tsx`, `overlays/panel.tsx`). All motion respects `prefers-reduced-motion`.

## Tips and first-run experience

- `apps/desktop/src/app/jarvis/setup-progress.ts` + `rail-start-card.tsx` — "Zacznij tutaj" card at the top of the dashboard rail: progress from real state (model, Google, vault, browser login, first task, daily report), hides itself when done or dismissed.
- `playbook-extras.ts` + `tips.tsx` — connection-aware tips ("with what you have connected": mail/calendar need Google, file search needs the vault, the "Tatuś wrócił" voice tip) shown above the regular playbook; the launcher asks what is connected each time the window opens.
- Onboarding wizard (`onboarding.tsx`, `onboarding-welcome.tsx`): glass panel, gradient step markers and progress bar, per-step entrance animation.
