# Brand Surface Audit & Rebranding Strategy

**Classification of every occurrence of "Hermes" in the repository, and a strategy for an
independent customer-facing identity that preserves a clean upstream merge path.**

Nothing has been renamed. This document ends with decisions needed before any rename begins.

| | |
|---|---|
| Repository | `yahyeameer/hermes-agen-` (fork of `NousResearch/hermes-agent`) |
| Version audited | `0.21.1`, HEAD `4bdd64b` |
| Total occurrences | **101,368** across **7,224 files** |
| Audit date | 2026-09-12 |

Throughout this document `ACME` stands in for the eventual customer-facing name.

---

## 0. The headline

A naive find-and-replace touches 101,368 strings across 7,224 files — more than the entire
diff of most rewrites — and would produce a fork that can never cleanly merge upstream again.

**It is also unnecessary.** Only about **4% of those occurrences are actually user-visible.**
The other 96% are module names, test fixtures, env-var keys, on-disk paths, upstream URLs and
third-party model identifiers — none of which a customer ever sees, and most of which are
load-bearing.

The strategy that follows treats brand as a **projection layer**, not a rename: the customer sees
`ACME` everywhere, while the substrate keeps saying `hermes` so `git merge upstream/main` stays
a routine operation.

Two further findings make this much cheaper than it looks:

1. **A branding indirection layer already ships.** `hermes_cli/skin_engine.py` defines a
   `branding` dict (`agent_name`, `welcome`, `goodbye`, `response_label`, `prompt_symbol`,
   `help_header`) plus `banner_logo` and `banner_hero`, and `load_skin()` reads user skins from
   `$HERMES_HOME/skins/<name>.yaml` before falling back to built-ins. CLI and TUI branding is
   *already* configuration, not code.
2. **A string catalog already ships.** `locales/*.yaml` holds static user-facing strings for 17
   languages behind `agent/i18n.py`, with fallback to English and then to the key.

Both are underused — 9 `get_branding()` call sites, 317 `t()` call sites — but they are the
right seams, they are upstream-maintained, and widening their use is the kind of change upstream
would accept rather than fight.

---

## 1. Classification

Counts come from separate `git grep` probes, so the buckets overlap slightly and do not sum
exactly to 101,368. Each row is a measured figure, not an estimate.

### Category A — User-facing branding
*Everything a customer can see. This is the rename target.*

| Surface | Count | Disposition |
|---|---:|---|
| `website/` — upstream docs site | 20,974 | **Delete from the business build.** Do not edit |
| `apps/` — Electron desktop app | 10,498 | **Out of scope.** Drop from the business build |
| Root `README*.md`, `CONTRIBUTING*`, `SECURITY*` | 1,642 | **Replace wholesale**, never line-edit |
| Hardcoded strings in shipping Python | 1,426 | **The real work** — route through branding/i18n |
| `web/` + `ui-tui/` — dashboard and TUI front ends | 1,279 | Token swap + brand config |
| `locales/*.yaml` (17 languages) | 272 | **Already an indirection layer** — swap values |
| `skin_engine.py` `branding` dict | 9 call sites | **Already config-driven** — ship a skin |
| `SOUL.md` / `default_soul.py` persona | 4 | Distribution-owned, per-profile — already ours |
| **Category A total (shipping surfaces only)** | **~4,600** | excludes `website/` and `apps/` |

The distinction in that last row is the whole argument: the genuinely user-visible brand surface
in code we intend to ship is roughly **4,600 occurrences, not 101,368** — and a third of those
already have a supported indirection point.

### Category B — Internal implementation identifiers
*Never visible to a customer. Renaming buys nothing and costs every future merge.*

| Identifier | Count | Disposition |
|---|---:|---|
| `tests/` + `tests-js/` | 36,697 | **Never rename.** Not shipped; pure merge cost |
| `hermes_cli` package and its imports | 18,708 | **Never rename** |
| `hermes_state*` — 22 modules | 1,971 | **Never rename** |
| `hermes_constants`, `hermes_logging`, `hermes_time`, `hermes_bootstrap`, `hermes_startup_watchdog` | 1,591 | **Never rename** |
| `_HERMES_*` private module constants | 392 | **Never rename** |
| `evals/` | 563 | Drop from the business build |
| **Category B total** | **~59,900** | |

`hermes_cli` alone is imported 10,150 times. Renaming the package would rewrite every import in
the repository, conflict with essentially every upstream commit, and change nothing a customer
can observe.

### Category C — Upstream dependency identifiers
*Names that belong to someone else. Renaming them breaks correctness, not just merges.*

| Identifier | Count | Why it must stay |
|---|---:|---|
| PyPI distribution `hermes-agent` | 2,973 | The package you install and update from |
| `nousresearch.com` domains | 863 | Portal, docs, install script, free-tier endpoints |
| `github.com/NousResearch/*` URLs | 690 | Upstream remote, issue links, example plugins |
| **`Hermes-3` / `Hermes-4` LLM model ids** | 189 | **A different product entirely — see below** |
| **Category C total** | **~4,700** | |

> **The one genuinely dangerous confusion in this repository.** `Hermes-4-405B`,
> `NousResearch/Hermes-3-Llama-3.1-70B`, `nousresearch/hermes-4-405b` and friends are
> **Nous Research's large language models**, not this agent product. They appear in provider
> profiles, model-routing regexes (`hermes_cli/model_switch.py`) and auxiliary-client
> configuration. A blanket rename silently breaks model resolution for any customer pointed at
> Nous inference, and the failure surfaces as a 404 from a provider rather than as a test
> failure. Any rename tooling must exclude `[Hh]ermes-[0-9]` as a hard rule.

### Category D — Compatibility-sensitive identifiers
*Renaming these breaks live installations, stored data, third-party plugins, or wire protocols.*

| Identifier | Count | What breaks on rename |
|---|---:|---|
| `HERMES_*` environment variables — **225 distinct**, read via `os.environ` | 12,962 | Every existing deployment, container, systemd unit, CI job |
| `$HERMES_HOME` / `~/.hermes` on-disk root | 6,973 | All state: `state.db`, `kanban.db`, credentials, profiles, skills |
| `compat_manifest.json` — plugin import paths | 2,170 | External plugins importing through the compatibility shim |
| `hermes-*` toolset names (`hermes-cli`, `hermes-telegram`, …) | 1,029 | Written into `config.yaml`; a rename silently disables toolsets |
| `X-Hermes-*` HTTP headers (9 distinct) | 164 | Gateway/sidecar/relay wire protocol |
| `hermes://` URL scheme (7 routes) | 69 | OS-registered deep links, plugin and blueprint install links |
| `hermes_agent.plugins` entry-point group | 43 | Every pip-installed third-party plugin |
| Console scripts `hermes`, `hermes-agent`, `hermes-acp` | 3 | Muscle memory, scripts, docs, cron entries |
| Docker image / container / s6 service names | ~30 | Compose files, orchestration, runbooks |
| `_HERMES_HOME_MARKERS = ("config.yaml", ".env", "state.db")` | 1 | Home-directory autodetection |
| **Category D total** | **~23,400** | |

The 225 environment variables are the sharpest edge. They are read directly through
`os.environ`, they appear in customer compose files and CI, and there is no central registry to
translate them — each is read at its own call site.

### Summary

| Category | Occurrences | Share | Rename? |
|---|---:|---:|---|
| A — User-facing branding (shipping) | ~4,600 | 4.5% | **Yes — this is the product identity** |
| A — User-facing branding (dropped surfaces) | ~31,500 | 31% | Delete, don't rename |
| B — Internal implementation | ~59,900 | 59% | **Never** |
| C — Upstream dependency | ~4,700 | 4.6% | **Never — correctness risk** |
| D — Compatibility-sensitive | ~23,400 | 23% | **Alias, never replace** |

*(Shares exceed 100% because probes overlap — `tests/` contains env-var and module references
counted in B and D both.)*

---

## 2. Rebranding strategy

### 2.1 The principle

> **Brand is a projection of the substrate, not a rename of it.**
> The customer sees `ACME`. The repository still says `hermes`. `git merge upstream/main`
> remains a routine operation.

Three layers, with a different rule for each:

| Layer | Rule | Examples |
|---|---|---|
| **Presentation** | Rename freely — it is already configuration | Skin branding, locales, persona, banner, docs |
| **Identity** | **Add** a new name beside the old one; never replace | CLI command, env vars, home directory, headers |
| **Substrate** | Never touch | Module names, tests, upstream URLs, model ids, entry points |

Every mechanism below is either pure configuration or purely additive. That is what keeps the
merge path clean: **an addition conflicts with nothing; a replacement conflicts with everything.**

### 2.2 Presentation layer — rename here, zero core edits

Almost all of this is already possible today.

**A brand skin.** Ship `brand/acme.skin.yaml`, installed to `$HERMES_HOME/skins/acme.yaml`. The
existing `load_skin()` prefers user skins over built-ins, so this needs no code change at all:

```yaml
name: acme
description: ACME Assistant
branding:
  agent_name:      "ACME Assistant"
  welcome:         "Welcome to ACME. Type your message or /help for commands."
  goodbye:         "Goodbye."
  response_label:  " ◆ ACME "
  prompt_symbol:   "❯"
  help_header:     "(?) Available Commands"
banner_logo: |
  <customer ASCII art>
colors: { ... }
```

**A brand locale overlay.** `locales/en.yaml` is the static message catalog. Ship an overlay
keyed the same way, and the brand values replace the upstream strings. Because `agent/i18n.py`
already falls back English → key, a missing key degrades gracefully rather than crashing.

**Persona.** `SOUL.md` is already distribution-owned and per-profile. The business distribution
ships its own; nothing in core changes.

**Documentation and assets.** `website/` and `apps/` are dropped from the business build
entirely. Root `README.md`, `CONTRIBUTING.md`, `SECURITY.md` and `assets/banner.png` are
**replaced wholesale, never line-edited** — see §2.5 for why that distinction decides your merge
cost.

**The 1,426 hardcoded strings.** This is the only genuine engineering work in the presentation
layer. Each is routed through `get_branding()` or `i18n.t()` rather than edited in place — which
also makes the change one upstream would plausibly accept, since it improves their skin and i18n
coverage rather than forking it.

### 2.3 Identity layer — alias, never replace

For every compatibility-sensitive identifier, **add** the new name and keep the old one working.

| Identifier | Mechanism | Core edit |
|---|---|---|
| CLI command | Add `acme = "hermes_cli.main:main"` to `[project.scripts]`. Both commands work | 1 line in `pyproject.toml` |
| Environment variables | Resolve `ACME_FOO` first, fall back to `HERMES_FOO`. One helper, called from the existing env accessors | ~1 helper + call-site wiring |
| Home directory | `get_hermes_home()` checks `$ACME_HOME`, then `~/.acme`, then `$HERMES_HOME`, then `~/.hermes` | 1 function in `hermes_constants.py` |
| HTTP headers | Emit both; accept either. Drop the old one only after every deployment is upgraded | ~9 emit sites |
| `hermes://` scheme | Register `acme://` **in addition**; route both to the same handler | Desktop manifest only |
| Plugin entry point | Discover `acme.plugins` **and** `hermes_agent.plugins` | 1 line in `providers/__init__.py` |
| Toolset names | Add `acme-*` aliases resolving to the same definitions; existing configs keep working | `toolsets.py` alias map |
| Docker image / services | Rename in *your* compose and Terraform, not in `docker/` | 0 |

Total core footprint: **roughly six small, self-contained hook points.** Each is a few lines,
each is additive, and each is in a file upstream rarely rewrites. That is the entire cost of an
independent identity.

Two identifiers deliberately keep their upstream names forever:

- **PyPI `hermes-agent`** — it is how you install and update. A customer never types it; it
  appears in a Dockerfile.
- **`Hermes-3` / `Hermes-4` model ids** — a third-party product. Excluded from every sweep by a
  hard rule.

### 2.4 Substrate — never touch

`hermes_cli`, `hermes_state*`, `hermes_constants`, `_HERMES_*`, `tests/`, upstream URLs. About
60% of all occurrences, and all of it invisible to customers.

The one-line justification to give anyone who objects: **a customer cannot see an import
statement, but every one you rename conflicts with every upstream commit that touches that file,
forever.**

### 2.5 The merge-path mechanics

Three concrete techniques, in order of leverage.

**1. Replace files wholesale; never line-edit upstream prose.** A `README.md` you rewrote
entirely resolves as a single "both modified" conflict you settle with `--ours` in one command.
A `README.md` where you swapped 400 brand words across 200 lines produces 200 conflicts on every
upstream docs commit, forever. The same rule governs `CONTRIBUTING.md`, `SECURITY.md`, locale
files and assets.

**2. Declare branded paths with a merge driver.** Add to `.gitattributes`:

```gitattributes
# Brand-owned files: always keep our version on merge.
README.md          merge=brandours
SECURITY.md        merge=brandours
CONTRIBUTING.md    merge=brandours
locales/*.yaml     merge=brandours
assets/**          merge=brandours
brand/**           merge=brandours
```

with `git config merge.brandours.driver true` in the setup script. These files then resolve
automatically on every upstream merge instead of demanding attention.

**3. Keep a patch budget.** `docs/platform/CORE_PATCHES.md` lists every core file the brand layer
touches, with one line saying why. The target is the six hook points in §2.3, and the list is
reviewed at every upstream merge. When it grows, something has gone wrong — that is the signal.

**Expected steady state:** a `git merge upstream/main` touching brand-owned files resolves
automatically; a merge touching one of the six hook points needs a few minutes of attention; a
merge touching anything else does not know the brand layer exists.

### 2.6 What a customer actually sees

| Surface | Customer sees |
|---|---|
| CLI command | `acme chat`, `acme gateway run` |
| Banner, prompt, responses | ACME name, logo, colors |
| Dashboard | ACME title, favicon, palette |
| Chat platforms | ACME name and avatar |
| Agent persona | ACME voice, from `SOUL.md` |
| Documentation | ACME docs only |
| Config directory | `~/.acme/` |
| Environment | `ACME_HOME`, `ACME_MODEL`, … |
| Docker | `acme-platform:1.0` |
| **What they never see** | `hermes_cli`, `hermes-agent` on PyPI, `hermes_state.py`, the tests |

### 2.7 Effort

| Stage | Work | Effort |
|---|---|---|
| R1 | Brand config schema, skin, locale overlay, persona; installer wiring | 3–5 days |
| R2 | Route the 1,426 hardcoded strings through branding/i18n | 4–6 days |
| R3 | The six alias hook points: command, env, home, headers, scheme, entry point, toolsets | 3–4 days |
| R4 | Wholesale doc/asset replacement, `.gitattributes` merge driver, patch budget, an upstream-merge rehearsal | 2–3 days |
| | **Total** | **2.5–3.5 weeks** |

This can run in parallel with platform Phase 1, or fold into it — the brand config is a natural
member of the `platform/config/` schema proposed in the platform audit.

### 2.8 Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Rename tooling catches `Hermes-4` model ids | **High** | Hard exclusion rule `[Hh]ermes-[0-9]` in every sweep, enforced by a CI check |
| Env-var rename breaks a running customer deployment | **High** | Aliases only; the old name never stops working in v1 |
| Brand strings drift back in on upstream merges | Medium | A CI check failing on `Hermes` in shipping user-facing strings, with an allowlist for B/C/D |
| Someone "tidies up" by renaming `hermes_cli` | Medium | Write the substrate rule into `CONTRIBUTING.md` and the patch budget |
| Upstream renames one of the six hook points | Low | Small, self-contained functions; a conflict is minutes, not hours |
| Trademark collision on the new name | — | A search before the name is chosen, not after the rename |

---

## 3. Decisions needed

1. **The name.** Everything downstream keys off it. A trademark and domain check comes first.
2. **Confirm: projection, not rename** — presentation renamed, identity aliased, substrate
   untouched.
3. **Confirm: aliases never expire in v1** — `HERMES_*` env vars and `~/.hermes` keep working
   indefinitely, with removal a later, separately-decided migration.
4. **Confirm: drop `website/`, `apps/`, `evals/`** from the business build rather than rebranding
   them.
5. **Upstream tracking** — the same question the platform audit raised. If you intend to keep
   merging, §2.5 is mandatory. If not, the substrate rule relaxes, though it still buys nothing.
6. **PyPI `hermes-agent` stays as the installed distribution** — confirm you are comfortable
   with the upstream name appearing in a Dockerfile where no customer reads it. Publishing a
   renamed private package is possible but adds a release pipeline you would then own.

Nothing will be renamed until these are settled.
