# Extensions — Agent Instructions

## Adding capability (footprint order)

1. Extend existing plugin or skill.
2. CLI + skill (`hermes <cmd>` documented in `SKILL.md`).
3. Service-gated tool (`check_fn` in `tools/`).
4. Plugin under `plugins/<name>/` with `plugin.yaml` + `register(ctx)`.
5. MCP catalog server.
6. New core tool — last resort; update `toolsets.py` **and** merge policy overlay.

## Plugin checklist

- `plugin.yaml` with `name`, `version`, `description`
- `register(ctx)` must not edit core files (`run_agent.py`, `cli.py`, …)
- Tool handlers accept `task_id` or `**kwargs`
- Secrets in `~/.hermes/.env` only; behaviour in `config.yaml`

## Fork tool files (preserve_custom)

These are **not** plugins — they ship in `tools/` and merge as fork-owned:

- `tools/harness_tools.py`
- `tools/vrchat_osc_tool.py`
- `tools/voicevox_tts_tool.py`
- `tools/shinka_evolve_tool.py`
- `tools/ai_scientist_tool.py`

When upstream adds overlapping capability, diff for API parity before dropping fork copies.

## VRChat safety

`vrchat-autonomy` move/speak paths require explicit user ACK before `dry_run=false`.

## Skills

- `description` ≤ 60 characters in `SKILL.md` frontmatter.
- Reference Hermes tools by name in prose, not raw shell utilities.
- Tests: `tests/skills/test_<skill>_skill.py` via `scripts/run_tests.sh`.

## Upstream PRs

Ship generic fixes upstream; keep Hakua/VRChat/Windows-only paths in this fork.
