# 04 — Rebrand Log: Hermes → Aro

Sep 30, 2026 · commit `2ed24927` — *"rebrand: Aro family surface pass (samjuniors)"* on `/home/z/SamAgent` (fork of `NousResearch/hermes-agent`).

**Scope rule:** surface rebrand only — every user-visible string. Deep identifiers (`~/.hermes`, `hermes://`, `hermes_cli` modules, `HERMES_*` env vars) intentionally untouched; kept as the next phase.

## What changed (22 files, +125/−250)

| Area | File(s) | Change |
|---|---|---|
| Desktop identity | `apps/desktop/product-identity.cjs` | Variants → **Aro** (default), **Aro Light** (remote-only), **Aro Agent** (bundled); `appId` → `com.samjuniors.aro*`; `cliName` → `aro` / `aro-light`; MSIX org → `Samjuniors.*`; store identity → `Samjuniors.AroAgent` (upstream Nous publisher values kept as reference with a NOTE comment — samjuniors must register its own Partner Center identity); new env `ARO_DESKTOP_VARIANT` (legacy `HERMES_DESKTOP_VARIANT` still honored) |
| CLI entry points | `pyproject.toml` | `[project.scripts]`: added primary **`aro`** (`hermes_cli.main:main`), **`aro-agent`** (`agent.legacy_cli:main`), **`aro-acp`** (`acp_adapter.entry:main`); upstream `hermes*` aliases kept for compatibility |
| CLI banner | `hermes_cli/banner.py` | New **`ARO_AGENT_LOGO`** — minimal emerald ASCII "A" (`#10B981/#34D399/#6EE7B7` ramp) + "agent by samjuniors" byline; replaces gold caduceus `HERMES_AGENT_LOGO` as default; version labels → "Aro Agent v…" (desktop-stamp + base paths) |
| Version strings | `agent/legacy_cli.py`, `acp_adapter/commands.py` | `--version` / `/version` output → "Aro Agent v…" |
| Desktop package | `apps/desktop/package.json`, `apps/desktop/index.html` | npm name → **`aro-desktop`**; description → "Aro Desktop — native desktop shell for Aro Agent, by samjuniors."; index.html `<title>` → **Aro** |
| Desktop build | `apps/desktop/electron-builder.config.cjs` | Default GitHub repo fallback → `samjuniors/SamAgent` |
| Web dashboard | `web/index.html` | Title → "Aro Agent - Dashboard" |
| Website | `website/docusaurus.config.ts` | Title → "Aro Agent"; org/URLs → samjuniors/SamAgent |
| README | `README.md` | Rewritten for the Aro family: samjuniors byline, MIT fork attribution, 4-product table (Aro Agent / Aro CLI / Aro Desktop / Aro Harness), `aro` quickstart, Aro roadmap (surface ✓ / new UI / deep rebrand / own infra) |
| Attribution | `NOTICE` (new) | "Aro Agent, Copyright © 2026 samjuniors… fork of Hermes Agent by Nous Research… upstream notices retained; 'Hermes' marks belong to their owners; user-facing surfaces branded Aro" |
| Persona | `SOUL.md` | "You are Aro Agent, built by samjuniors (based on Hermes Agent by Nous Research)." — voice unchanged (direct, terse) |
| Translations | `README.zh-CN.md`, `README.es.md`, `README.ur-pk.md` | Fork-notice banners pointing to the English README for current branding |
| Dev guide | `AGENTS.md` | Top banner: user-facing surfaces branded Aro; internal identifiers unchanged this phase; see roadmap |
| Tests | `tests/acp_adapter/test_runtime_identity.py`, `tests/agent/test_legacy_cli.py`, `tests/e2e/core/windows/test_boot_lifecycle.py`, `tests/e2e/core/windows_update/test_fresh_install_update.py`, `tests/e2e/core/windows_update/test_paths_and_acls.py` | String assertions updated "Hermes Agent v" → "Aro Agent v" |

**Kept on purpose:** `hermes`/`hermes-agent`/`hermes-acp` console aliases; `HERMES_DESKTOP_VARIANT` env fallback; gold caduceus skin asset (`HERMES_CADUCEUS`) as legacy skin option; all internal `hermes` identifiers.

## Remaining deep-rebrand TODO

| # | Item | Notes / risk |
|---|---|---|
| 1 | **`~/.aro` home directory** | Migrate `~/.hermes` (sessions DB, memory, skills, config). Needs migration path + `hermes` alias compat window. Breaks docs/scripts that hardcode the path |
| 2 | **`aro://` deep-link protocol** | Replace `hermes://`; must re-register protocol handlers (macOS `Info.plist`, Windows registry/MSIX, Linux `.desktop`) |
| 3 | **`hermes_cli` module rename** | 479 files / 253k LOC touch every import; also `hermes_state_*.py` siblings (60+), `hermes` launcher, package metadata. Largest single item — mechanical but wide |
| 4 | **`HERMES_*` env vars** | Rename to `ARO_*` with a deprecation shim reading old names; audit CI, docs, install scripts |
| 5 | **Update feed: R2 → own bucket** | electron-updater feed currently points at upstream Nous Research R2. samjuniors must stand up its own generic feed (or GitHub releases) before shipping updates |
| 6 | **Code-signing certs for samjuniors** | macOS Developer ID + notarization, and Azure Trusted Signing (or equivalent) for Windows — under the samjuniors identity. Without them, auto-update and SmartScreen reputation break |
| 7 | **MSIX publisher registration** | Partner Center publisher identity `CN=…` is still Nous Research's (kept as reference in `product-identity.cjs`); store submission blocked until samjuniors registers |
| 8 | **~717 desktop files with internal `hermes` refs** | Classify: internal identifiers (leave), user-visible strings (fix), asset names/icons (replace), file names (optional). Sweep + CI lint to prevent regressions |
| 9 | Install scripts (`setup-hermes.sh/.ps1` → `setup-aro.*`), `install.sh` upstream URL, Docker/Nix/Termux surfaces | README already flags "install scripts still point at upstream infrastructure" |
| 10 | Website copy + i18n beyond title; dashboard in-app strings beyond `<title>`; docs site domain | Aro docs site under samjuniors infra (roadmap item) |
| 11 | Assets: banner.png, favicons, app icons, store artwork | Currently upstream Hermes artwork; needs the Aro mark (see `docs/brand/BRAND.md` §6) |
| 12 | `ARO_AGENT_LOGO` as *skin default* parity | Some banner-skin paths still reference the legacy logo variable — verify all skins resolve the Aro logo |

Order of attack: **5+6+7 first** (shipping pipeline — an unsigned/unupdatable fork is dead on arrival) → **1+2** (user-facing identity) → **3+4** (code-wide, one mechanical PR each) → **8–12** cleanup sweeps with a CI lint (`ban-hermes-strings` on user-facing surfaces).
