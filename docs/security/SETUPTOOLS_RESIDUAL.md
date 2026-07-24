# Setuptools Residual Advisory (2026-07-18)

## Status: RESIDUAL — known, documented, not bumped in this PR

## Decision

`setuptools` remains pinned at **`==81.0.0`** in the `dev` extra and the
build-system floor stays at **`>=77.0,<83`**, matching the in-file comment
that has been on `pyproject.toml` since this floor was introduced.

## Why we are NOT bumping to 83.0.0

The current scanner fix target is **`83.0.0`** (`PYSEC-2026-3447`, as reported by both `pip-audit --strict` and `hermes security audit`). Hermes cannot take it without coordinated changes upstream:

- **Torch cap.** Per the in-file comment on the `dev` extra, `torch >=2.11`
  has not yet shipped compatibility for `setuptools>=82`. Several
  lazy-installable extra dependencies (model-serving and TTS backends) pull
  in `torch` transitively. Bumping `setuptools` to `83.x` would force
  every model-serving / TTS installer to either upgrade torch first
  (out of scope for a security PR) or pin an older setuptools at runtime,
  which defeats the bump on the very installs that need the patch.

- **Build-system ceiling.** `setuptools>=77.0,<83` intentionally excludes the
  scanner's `83.0.0` fix target. Both that ceiling and the `==81.0.0` `dev`
  pin must move together once the Torch compatibility constraint is resolved.

## Active exposure

- `setuptools 81.0.0` ships with the PEP 639 SPDX license handling used
  by our `license = "MIT"` declaration and is the version most users
  install under `[all]` today.
- Any setuptools CVE published against `81.0.0` itself (rather than
  older releases) is therefore the relevant exposure surface.
- Current scanner findings (2026-07-18, `hermes security audit` /
  `pip-audit --strict`, both querying OSV.dev):
  - `PYSEC-2026-3447` — affects `setuptools<83.0.0`, fix version 83.0.0.
  - This is the single residual advisory reported against the venv
    after the 2026-07-18 remediation. No other package on the
    isolated `[dev,mcp,computer-use,web,google]` install is flagged.

## Follow-up

- Re-evaluate when Torch ships setuptools-83 compatibility, OR
  when a Hermes security audit confirms that none of the lazy-install
  backends reach `torch` on the install paths we actually exercise
  (text-only LLM backends, Telegram/Discord/Slack gateways, browser
  tools, file tools).
- At that point, bump:
  - `dev` extra: `setuptools==81.0.0` → `setuptools==83.0.0` (or the
    current `83.x.y`, see PYSEC-2026-3447)
  - Run `uv lock` to regenerate.
  - Run the full `pytest -q` suite plus `hermes security audit` in CI
    before merging.

## Tracking

- `pyproject.toml` `dev` extra line: search for `setuptools==81.0.0`
- Build-system floor: `pyproject.toml` line 5
- This file: `docs/security/SETUPTOOLS_RESIDUAL.md`
