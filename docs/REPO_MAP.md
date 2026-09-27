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


### Windows: silnik w instalatorze

- `apps/desktop/electron/bundled-runtime.ts`: wybór dołączonego silnika przed starym bootstrapem.
- `apps/desktop/scripts/stage-windows-runtime.mjs`: pakowanie kodu, Pythona, zależności, Git i Node.
- `apps/desktop/scripts/before-pack.mjs`: zgodność commita i architektury pakietu.
- `docs/CZESIEK_INSTALLATION.md`: instrukcja budowania i granice działania offline.
