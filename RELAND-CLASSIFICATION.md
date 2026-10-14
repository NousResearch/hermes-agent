# RE-LAND classification (v0.22.0 merged tree)

2026-10-14. Branch `merge/v0.22.0` in the clone. Each of the 8 RE-LAND commits was re-checked against the
merged (= v0.22.0) code. **Cherry-picking does not work** — v0.22.0 rewrote the target seams, so
`git cherry-pick` produced an 8,000-line spurious conflict on `web_server.py`. Classification below is by
reading each commit's diff against the new structure.

## SUPERSEDED by upstream → DROP (no work)

| Commit | Was | Why DROP (evidence in v0.22.0) |
|---|---|---|
| `4e710be696` | flat-json provider config file path | `hermes_cli/web_server_memory.py:214` already tries `(hermes_home/f"{name}.json", hermes_home/name/"config.json")` — upstream reads the native `<name>.json` natively. |
| `17b5ae7214` | gateway: adopt held lock as identity when `gateway.pid` gone | `gateway/status.py` rewritten with a full identity/lock system (`_scope_hash`, `_get_scope_lock_path`, identity matchers, #107002). |
| `bcd3a78953` | gateway: detached self-replacement on Windows exit-75 | `gateway/shutdown_watchdog.py` rewritten: supervisor-restart exit code + `state/gateway.heartbeat` so supervisors distinguish alive-vs-hung. |
| `871ae1ee32` | desktop: kill orphaned hermes_cli python on quit | `electron/main.ts` rewritten with `close-stop-kill`, `no-console-git` (`killTimedGitChildren`), singleton-lock PID handling. |
| `85e4ecc2fd` | perf(tools): dedupe tool prompt text | perf-only micro-opt; upstream reworked the tool schemas/prompts wholesale. Re-measure before considering. |

## STILL NEEDED → PORT (upstream lacks it; seams moved)

| Commit | Item | New home | Effort |
|---|---|---|---|
| `3a55a0aaf4` | `memory_extraction` aux slot in Models UI | v0.22.0 split this out: `apps/desktop/src/app/settings/aux-task-rows.tsx` + per-locale `i18n/*_aux_tasks.ts` (en/de/es/fr/ja/ru + zh/zh-hant settings) + `hermes_cli/web_server_config.py`. `memory_extraction` appears nowhere. | M (multi-file) |
| `23d13c2954` + `bdc1d54534` | pre_llm_call hook routes the answerer model (+ log the pre-switch model) | v0.22.0 refactored the seam into `agent/turn_context.py::_collect_pre_llm_call_context()` (returns joined text only, lines 794-845). Port = surface `{model, provider}` from hook results and switch in `build_turn_context` (1173); `switch_model` now lives in `agent/agent_runtime_helpers.py`. | M (logic port) |
| `baa4825bda` | EOL-normalize fuzzy patch matching for CRLF checkouts | `tools/fuzzy_match.py` is a rewritten 719-line strategy engine with **no** EOL normalization (splits on `'\n'`, so CRLF files carry `\r`). Port via TDD against the new strategy structure. | M (TDD) |

## Method notes
- `git cherry-pick` is the wrong tool here: `-x` of `4e710be696` conflicted across lines 1005–9147 of `web_server.py`.
- `web_server.py` is now split into ~20 modules (`web_server_memory.py`, `web_server_config.py`, ...) — any port must target the new module.
- Desktop settings i18n is now per-locale files (`*_aux_tasks.ts`), not one big `en.ts`/`ar.ts`.
