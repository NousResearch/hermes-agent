# PR 1/3: agents-os-runtime-control-plane-clean

**Branch:** `feat/agents-os-runtime-control-plane-clean`
**Base:** `main` @ `4d39a60`
**Scope:** Originalni Agents OS runtime control plane + današnji escape-safe onclick fix.
**Source:** Izdvojeno iz #42341 (commits `6999b065` + `3370c74e`).

## Files
- `hermes_cli/agents_os.py` (+1848)
- `hermes_cli/agents_os_tui.py` (+356)
- `hermes_cli/agents_os_web.py` (+1250) — **uključuje današnji fix**
- `hermes_cli/commands.py` (+3)
- `hermes_cli/main.py` (+44)
- `scripts/launch_agents_os_mission_control.sh` (+59)
- `tests/hermes_cli/test_agents_os.py` (+143)
- `tests/hermes_cli/test_agents_os_tui.py` (+123)
- `tests/hermes_cli/test_agents_os_web.py` (+503)

## Što radi
- Lokalni Agents OS runtime s Mission Control web UI-jem (port 18790)
- CLI komanda `agents-os` za start/stop/status
- Mission Control paneli: tasks, approvals, artifacts (detail view), launch script
- TUI mode za isti panel set bez web dependency-ja
- **Današnji fix:** data-detail-id + dataset.detailId umjesto string-escape onclick (mc-debug.js: 0 pageerror, 12 tasks + 3 approvals render)

## Što NE uključuje
- Jarvis/STT/TTS scope (→ PR 2)
- Wiki lifecycle helpers (→ PR 3)
- SEO/idea_factory panels (→ PR 2)

## Acceptance za maintainere
- `pytest tests/hermes_cli/test_agents_os.py tests/hermes_cli/test_agents_os_tui.py tests/hermes_cli/test_agents_os_web.py -q` mora proći
- `bash scripts/launch_agents_os_mission_control.sh` starta UI na portu 18790
- mc-debug.js (browser konzola) — 0 pageerror nakon učitavanja
- Svi artifact i approval detail paneli se renderiraju s validnim JSON-om

## Notes
- Ovaj PR je minimalni, deterministic-verify scope. Razdvajanje je dogovoreno jer je #42341 narastao na 11 commit-ova i 17 fajlova (scope drift disclaimer).
- Originalni fix-commit (`3370c74e`) nosi i `.gitignore` update za `recall-like-mvp/`. Taj red je izdvojen iz PR-a i pripada lokalnom mirror workflow-u — ne treba ići u upstream PR.
