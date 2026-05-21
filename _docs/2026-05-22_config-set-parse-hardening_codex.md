# Config Set Parse Hardening

Date: 2026-05-22
Branch: `codex/hermes-env-config-hardening-20260522`

## Summary

This change hardens `hermes config set` so it refuses to write when the existing
`config.yaml` cannot be parsed or is not a top-level mapping. The goal is to
preserve the user's on-disk configuration instead of treating an unreadable file
as an empty config and writing over it.

## GitHub Triage

Related public items:

- #5214: Improve handling for locked or invalid config.yaml writes.
- #14276: Broader open PR for config parse failure guards and recovery.
- #29655: ruamel fallback for round-trip config writes.

This PR keeps the scope narrow: only the direct `hermes config set` mutation path
is changed, with regression tests that prove malformed or non-mapping YAML stays
untouched.

## Changes

- `set_config_value()` now returns `True`/`False`.
- Secret writes to `.env` still return success after saving.
- Existing malformed `config.yaml` now triggers a refusal message and no write.
- Existing non-mapping YAML now triggers a refusal message and no write.
- Nested key navigation or write failures are surfaced without rewriting the
  original file.

## Verification

```powershell
uv run --extra dev python -m pytest -o addopts= tests/hermes_cli/test_set_config_value.py tests/cli/test_cli_save_config_value.py
uv run --extra dev ruff check hermes_cli/config.py tests/hermes_cli/test_set_config_value.py
```

Results:

- `43 passed in 9.08s`
- `ruff` passed.
