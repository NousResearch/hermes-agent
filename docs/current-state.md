# Current State

_Last updated: 2026-04-14_

## Dashboard / ops work

- The existing Hermes web UI already ships a FastAPI + React dashboard for status, sessions, analytics, logs, cron, skills, config, and env management.
- `hermes_cli/ops_dashboard.py` now provides the first Jax-ops-specific read-model ingestion layer over raw session JSON files in `HERMES_HOME/sessions/`.
- The module normalizes each session into:
  - a lightweight `SessionRecord`
  - a lightweight `RunRecord`
- Supported inputs currently include:
  - regular `session_*.json` files
  - cron-style `session_cron_<jobid>_*.json` files
- Best-effort run status heuristics are implemented for `running`, `completed`, and `failed`.

## Validation

- `python -m pytest tests/hermes_cli/test_ops_dashboard.py -q`
- `python -m pytest tests/hermes_cli/test_web_server.py -q`

## Next logical steps

- PAB-127: expose the normalized run/session read model through dedicated read-only ops endpoints.
- PAB-128: build overview + active-work UI surfaces on top of the read model.
- PAB-129/PAB-130: extend the same read model with process/alert ingestion and drilldown views.
