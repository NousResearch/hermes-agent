# BRANDING.md — what North Forge owns, what stays Hermes

Source of truth for any agent or contributor touching identity in this repo. It
resolves the recurring question: *"is this string supposed to say North Forge, or
is it deliberately left as Hermes?"*

North Forge is a fork of [`NousResearch/hermes-agent`](https://github.com/NousResearch/hermes-agent),
kept rebased on upstream, engine used **unmodified**. The rebrand
(`DECISION-2026-09-06-001`, Option B — see `logs/ledger/`) changed identity and
workflow surfaces only. Three categories:

---

## 1. North-Forge-owned — keep these branded

Change these to match North Forge; a stray "Hermes" here is a bug.

| Surface | Notes |
| --- | --- |
| `README.md` — H1, tagline, badges, lead description, provenance paragraph, License footer | `CHG-2026-09-06-020`. The install steps, `hermes …` command list, feature table, docs links, Contributing/Community are **category 3** (upstream docs) and stay. |
| `README.md` — "Drive class", "Customizing your agent" sections | `CHG-2026-09-06-026`. North Forge deployment concepts. |
| `SOUL.md` (repo root) | The generic North Forge chassis voice — adaptive tone, honest about uncertainty, direct, **industry-neutral**. Not read at runtime (installers seed from `hermes_cli/default_soul.py`); it is the identity reference. Its `DEFAULT_SOUL_MD` seed now opens with the same North Forge identity line (`CHG-2026-09-07-007`). `CHG-2026-09-06-021`, re-generic'd this pass. |
| `editions/**` | North Forge profile overlays (e.g. `editions/field-service/`). Optional, additive, never a replacement for root `SOUL.md`. |
| `hermes_cli/default_soul.py` `DEFAULT_SOUL_MD` + `agent/prompt_builder.py` `DEFAULT_AGENT_IDENTITY` | The runtime persona **identity line**: the text seeded into `HERMES_HOME/SOUL.md` on first run (`DEFAULT_SOUL_MD`) and the in-memory fallback used when no SOUL.md is present (`DEFAULT_AGENT_IDENTITY`, e.g. `skip_context_files` / subagent). Opens *"You are North Forge, an adaptive AI agent (built on the Hermes Agent engine by Nous Research)."* The two are kept **byte-identical**. `CHG-2026-09-07-007` — moved here from category 2. The rest of the behaviour spec (reply-sizing rule, named prohibitions, earned-depth) is upstream's and stays; the `_LEGACY_TEMPLATE_SOULS` / `_SCAFFOLD_*` detection strings in the same file stay Hermes-verbatim (category 2). |
| `hermes_cli/banner.py` `format_banner_version_label()` + `hermes_cli/_parser.py` top-level parser `description` | Display name in `hermes --version` (*"North Forge v0.21.0 …"*) and the `hermes --help` header (*"North Forge - AI assistant with tool-calling capabilities"*). `CHG-2026-09-07-007` — moved here from category 2. The version number, the `· upstream <sha>` suffix, and the `Install directory` / `Install method` / `Python` / `OpenAI SDK` lines are **not** identity — they stay. Every `hermes …` example in the epilogue stays (that is the command name — category 2). Degraded-path echoes in `hermes_cli/_startup_fast.py` and `cli.py` (`HERMES_FAST_STARTUP_BANNER=1`) match. |
| `LICENSE` | MIT. Both copyright lines: `© 2025 Nous Research` (engine) **and** `© 2026 Kenneth C. Walker Jr.` (North Forge). Never remove Nous's line. `CHG-2026-09-06-024`. |
| `package.json` — `name`, `repository`, `homepage`, `bugs` | `north-forge-agent`, `kwalker7631/north-forge-agent`. `CHG-2026-09-06-022`. Root `package-lock.json` root `name` mirrors it. |
| `pyproject.toml` — `[project.urls]` | `Homepage` / `Repository` → `kwalker7631/north-forge-agent`. `CHG-2026-09-06-023`. |
| `logs/ledger/**` | The North Forge project ledger. Upstream has nothing here. |
| `assets/banner.png` | North Forge brand banner (compass + anvil + flame, "NORTH FORGE"). Final art, `CHG-2026-09-07-006` (retired the `CHG-2026-09-06-025` placeholder). |
| `assets/icons/**` | `north-forge.ico` (Windows multi-resolution), `icon-512.png`, `icon-256.png`, `splash-alt.png` (held for a future loading screen — not wired yet). `CHG-2026-09-07-006`. A future Start-Menu/desktop shortcut should point at `assets/icons/north-forge.ico`. |
| `.githooks/content-scan`, `.github/workflows/nf-secret-scan.yml` | North Forge secret-scanning layer. |
| `scripts/collect-logs.*`, `scripts/redact_handoff.py`, `scripts/bootstrap-north-forge.ps1`, `north-forge.cmd` | North Forge tooling. |
| `README.zh-CN.md`, `README.es.md`, `README.ur-pk.md` | Short landing pages → the English `README.md` + upstream's translated docs. Not full translations. |

---

## 2. Intentionally Hermes-compatible — do NOT rebrand

These deliberately keep the Hermes name. Renaming them is churn with no outward
benefit and breaks compatibility or self-references. Each is a settled carve-out.

| Surface | Why it stays |
| --- | --- |
| `pyproject.toml` `name = "hermes-agent"` (the **distribution** name) | Never published (`setup.py` blocks wheel/sdist builds), referenced ~19× by the self-referential `hermes-agent[...]` optional-dependency extras, pinned in `uv.lock` and the installed venv. Explicit carve-out in `DECISION-2026-09-06-001`. |
| `hermes`, `hermes-agent`, `hermes-acp` console scripts; the repo-root `./hermes` launcher | The command names users and docs already know. `bootstrap-north-forge.ps1` / `north-forge.cmd` wrap them; they don't replace them. |
| Workspace package names — `hermes-tui`, `hermes`, `@hermes/root-tests` | Internal build ids; `"private": true`, never published. |
| `pyproject.toml` `authors = [{ name = "Nous Research" }]` | Engine authorship. North Forge attribution is carried by `LICENSE`. |
| `agent/prompt_builder.py` `HERMES_AGENT_HELP_GUIDANCE`; `hermes_cli/default_soul.py` `_LEGACY_TEMPLATE_SOULS` / `_SCAFFOLD_*` | Engine-help constant (*"You run on Hermes Agent (by Nous Research)…"*, points at `hermes-agent.nousresearch.com/docs`) and the legacy-SOUL auto-upgrade detection strings — both must match upstream **verbatim** to keep functioning. The persona **identity line** (`DEFAULT_SOUL_MD` / `DEFAULT_AGENT_IDENTITY`) moved to category 1 — `CHG-2026-09-07-007`. |
| `HERMES_HOME`, `~/.hermes`, `$HERMES_*` env vars, `%LOCALAPPDATA%\hermes` | Runtime/data layout owned by the engine. `bootstrap-north-forge.ps1` sets `HERMES_HOME` to a sibling folder but does not rename the variable. |
| `docker/SOUL.md`, `Dockerfile`, `docker-compose*.yml` | Upstream's container assets. |

---

## 3. Upstream documentation — left pointing at Nous

North Forge has **no docs site of its own**. Everything below still points at
`hermes-agent.nousresearch.com` and upstream repos on purpose — the engine is
used unmodified, so upstream's docs are correct for it. Fork-specific issues go to
`github.com/kwalker7631/north-forge-agent/issues`; engine issues go upstream.

- `README.md` install one-liners, `hermes …` command reference, the documentation
  table, the feature table, Contributing, Community, Migrating-from-OpenClaw.
- `AGENTS.md`, `CONTRIBUTING.md` / `CONTRIBUTING.es.md`, `SECURITY.md` / `SECURITY.es.md`,
  `COMPAT_MANIFEST.md`, everything under `website/`.
- Upstream code comments, docstrings, and log strings that say "Hermes".

Rule of thumb: **identity and workflow surfaces = North Forge; how-to-run-it and
engine internals = Hermes.** When a change would touch category 1 or 2 in a way
this file doesn't cover, that's a `DECISION-` in `logs/ledger/decisions/`, not a
judgement call to make in passing.
