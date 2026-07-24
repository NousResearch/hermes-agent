# OpenClaw plugin governance implementation notes

## Scope decision

Vladimir approved the full plugin roadmap. The first safe vertical slice is a local Hermes plugin that governs skill-to-plugin migration before any live Bitrix/Telegram/Google/Ozon actions.

Why this slice first:
- It has no external side effects and needs no credentials.
- It turns the approved roadmap into machine-readable plugin candidates and plans.
- It creates reusable tools that can drive the later Bitrix, Telegram, digest, watchdog, Google, procurement, document, Ozon, broker, GitHub, and Employee Access plugins.

## Source grounding

Official in-repo docs consulted:
- `website/docs/guides/build-a-hermes-plugin.md`: plugin manifest, `register(ctx)`, tool schema, JSON-string handler contract.
- `hermes_cli/plugins.py`: bundled standalone plugins are discovered but opt-in via `plugins.enabled`; tool registration uses `ctx.register_tool`.

## Safety boundaries

This slice only adds local plugin code, tests, and docs/artifacts. It does not send Telegram/Bitrix messages, deploy production, mutate credentials, or import marketplace products.

## Verification log

- RED: `python -m pytest tests/plugins/test_skill_governance_plugin.py -q` failed as expected because `plugins/skill_governance/` did not exist yet (10 failures: missing catalog/tools and discovery key).
- GREEN targeted: `python -m pytest tests/plugins/test_skill_governance_plugin.py -q` -> `10 passed, 1 warning in 0.29s`.
- Ruff targeted: `python -m ruff check plugins/skill_governance tests/plugins/test_skill_governance_plugin.py` -> `All checks passed!`.
- Plugin regression: `python -m pytest tests/plugins/test_skill_governance_plugin.py tests/plugins/test_disk_cleanup_plugin.py -q` -> `56 passed, 1 warning in 0.86s`.
- Full suite attempt before project venv sync: `python -m pytest tests/ -o 'addopts=' -q` failed during collection with missing `acp` dependency.
- Project venv sync: `uv sync --frozen --extra dev --extra all` succeeded and installed `agent-client-protocol` plus dev/all extras in the worktree `.venv`.
- Full suite attempt after sync: `uv run python -m pytest tests/ -o 'addopts=' -q` timed out after 600s with many unrelated pre-existing failures/errors across the repository; not used as release gate for this scoped plugin slice.
- Plugin-suite gate after sync: `uv run python -m pytest tests/plugins -q` -> `1071 passed, 1 warning in 34.35s`.
- Ruff after sync: `uv run python -m ruff check plugins/skill_governance tests/plugins/test_skill_governance_plugin.py` -> `All checks passed!`.
- Manual plugin dispatch proof with temp `HERMES_HOME`: plugin enabled, registered `skills_audit,skills_find_plugin_candidates,skills_to_plugin_plan`, `skills_audit` returned `12`, and `skills_to_plugin_plan(bitrix_ops)` returned candidate id `bitrix_ops`.
- Independent pre-commit review: `passed=true`; no security concerns, no logic errors. Two suggestions were handled: plugin handlers now normalize malformed args to preserve JSON contract, and `skills_audit` now exposes current registered tool names separately from future roadmap tool names.
- Post-review RED: `uv run python -m pytest tests/plugins/test_skill_governance_plugin.py -q` failed with 2 expected failures for malformed-args handling and missing `registered_tools` audit field.
- Post-review GREEN+ruff: `uv run python -m pytest tests/plugins/test_skill_governance_plugin.py -q && uv run python -m ruff check plugins/skill_governance tests/plugins/test_skill_governance_plugin.py` -> `11 passed, 1 warning in 0.48s`; `All checks passed!`.
- Final plugin-suite gate: `uv run python -m pytest tests/plugins -q` -> `1072 passed, 1 warning in 30.88s`.

## PR #40712 CI follow-up

- Root cause found before GitHub Actions could run: the fork PR workflows are in `action_required`, and the pushed commit used a contributor email that was not yet mapped in `scripts/release.py`.
- Fix: added the contributor mapping and renamed the bundled plugin directory to `plugins/skill_governance/` while keeping the public plugin manifest name `skill-governance`; this preserves plugin discovery and avoids new `ty` unresolved-import diagnostics from a hyphenated package path.
- Local proof after the fix: history check passed, contributor attribution check passed, supply-chain scan found no critical patterns, `tests/plugins` passed (`1072 passed, 1 warning`), release targeted tests passed (`3 passed`), full ruff passed, Windows footgun scan passed, and targeted `ty` passed.
