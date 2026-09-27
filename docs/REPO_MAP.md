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
