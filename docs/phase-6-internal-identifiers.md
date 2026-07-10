# Phase 6 — Internal Identifier Rebrand (backward-compatible)

**Status:** In progress — the alias foundation, entrypoint hardening, the
backward-compatible `~/.hermes` → `~/.ht-ai-agent` data-dir migration, and the
single-file module-name aliases (`ht_constants`, `ht_state`, `ht_time`,
`ht_logging`, `ht_bootstrap`) have landed. Only the full internal *package*
rename (`hermes_cli` → `ht_cli`, …) remains a deferred follow-up increment.

Phase 6 rebrands the fork's **load-bearing internal identifiers** from the
`Hermes` name to `HT`. Unlike Phases 1–5 (user-visible surfaces, safe to change
outright), these identifiers are inter-process, on-disk, and third-party-client
contracts:

- `HERMES_*` environment variables (~329, many are launcher⇄gateway protocol)
- the `~/.hermes` data directory (holds every user's sessions, skills, memory)
- `X-Hermes-*` HTTP headers (read/written by external OpenAI-compatible clients)
- Python module names (`hermes_cli`, `hermes_state`, `hermes_constants`, …)

The chosen strategy is **backward-compatible migration**: introduce the new
`HT_*` / `X-HT-*` names *additively*, keep the legacy names working, and never
break a running install or an existing client.

---

## What has landed (this increment)

A single, dependency-free compatibility module, `ht_compat.py`, plus the wiring
that makes both namespaces equivalent.

### Environment variables — `HERMES_*` ⇆ `HT_*`

`ht_compat.mirror_brand_env()` mirrors the two namespaces by a pure **prefix
rule** (`HERMES_FOO` ⇆ `HT_FOO`), so all ~329 vars are covered without an
enumerated table that would drift. It is:

- **non-clobbering** — a counterpart is only filled when absent, so setting both
  keeps both;
- **idempotent** — running twice changes nothing;
- **bidirectional** — set either name, the other appears.

It runs once at process startup — at every standalone entrypoint, not just the
CLI — so both names resolve process-wide and every spawned subprocess inherits
both:

  * `hermes_cli/main.py:main` (the `ht` / `hermes` CLI; the gateway and MCP
    server run in-process under it, so they are covered too),
  * `run_agent.py:main` (the `hermes-agent` console script),
  * `acp_adapter/entry.py:main` (the `hermes-acp` console script),
  * `batch_runner.py` `__main__` (`python batch_runner.py`).

Each uses the same guarded, idempotent call, so an `HT_*`-only environment works
even when these are launched directly rather than as a child of the CLI.

`ht_compat.resolve_env(name, default)` reads a single var honouring both
spellings, with the new `HT_*` value winning when both are set. `hermes_constants`
uses it directly for the load-bearing path vars so they resolve correctly even
before `mirror_brand_env` runs (it is called at import time from 30+ sites):

| Legacy | New alias |
|---|---|
| `HERMES_HOME` | `HT_HOME` |
| `HERMES_REAL_HOME` | `HT_REAL_HOME` |
| `HERMES_OPTIONAL_SKILLS` | `HT_OPTIONAL_SKILLS` |
| `HERMES_OPTIONAL_MCPS` | `HT_OPTIONAL_MCPS` |
| `HERMES_BUNDLED_SKILLS` | `HT_BUNDLED_SKILLS` |
| `HERMES_NODE_TARGET_MAJOR` | `HT_NODE_TARGET_MAJOR` |
| …all other `HERMES_*` | …`HT_*` (via `mirror_brand_env`) |

`gateway/status.py` (which keeps its own process-level home resolver) reads the
alias too.

### HTTP headers — `X-Hermes-*` ⇆ `X-HT-*`

Same prefix rule via `mirror_brand_headers()` (dual-emit) and
`read_brand_header()` (dual-read). Wired into:

- **`gateway/platforms/api_server.py`** (the OpenAI-compatible API — the
  third-party contract): every response now carries both `X-Hermes-*` and
  `X-HT-*` for `Session-Id`, `Session-Key`, `Completed`, `Partial`, `Error`;
  request ingestion accepts either spelling of `Session-Id` / `Session-Key`.
- **`hermes_cli/web_server.py`**: the loopback dashboard auth header accepts
  both `X-HT-Session-Token` and `X-Hermes-Session-Token`.

Legacy `X-Hermes-*` headers are always preserved, so existing clients are
unaffected; the mirror is additive and non-clobbering.

### Tests

- `tests/test_ht_compat.py` — full unit coverage of the mapping, env mirror, and
  header helpers (precedence, non-clobbering, idempotence, case-insensitivity).
- `tests/test_hermes_constants.py::TestHtHomeAlias` — `HT_HOME` precedence /
  fallback / equivalence with `HERMES_HOME`.
- `tests/gateway/test_session_api.py::test_session_chat_accepts_ht_headers_and_mirrors_them`
  — end-to-end: a request sent with only `X-HT-Session-Key` is honoured and the
  response carries both header spellings.

---

## Physical `~/.hermes` → `~/.ht-ai-agent` data-dir migration (landed)

The on-disk default is now the HT-branded `~/.ht-ai-agent` (POSIX) /
`%LOCALAPPDATA%\ht-ai-agent` (Windows), with a backward-compatible migration.

**Resolver** (`_get_platform_default_hermes_home`, side-effect-free — safe to
call at import from 30+ sites): prefer `~/.ht-ai-agent` if it exists, else an
existing legacy `~/.hermes` *in place* (so an un-migrated install never loses
its home), else the new location for fresh installs. Env resolution is unchanged
and still legacy-authoritative (`HT_HOME` → `HERMES_HOME` → default).

**Migration** (`maybe_migrate_home`, explicit, run once at startup from
**every** standalone entrypoint — the CLI, `hermes-agent`, `hermes-acp`, and
`python batch_runner.py` — never at import, so the bridge exists no matter
which entry launches first): reconciles the default dir:
- *Existing install* (`~/.hermes` is a populated real dir): atomic `os.rename`
  to `~/.ht-ai-agent` + a back-compat symlink `~/.hermes → ~/.ht-ai-agent`.
  If the symlink fails the rename is rolled back and the legacy home stays
  authoritative. In the one unavoidable split (rollback *also* fails, or the
  process dies between rename and symlink) the move is reported honestly and
  the missing bridge is repaired on the next start.
- *Fresh install* (neither exists): create `~/.ht-ai-agent` and point
  `~/.hermes` at it via symlink. A **dangling** `~/.hermes` symlink (deleted
  target) is repaired the same way rather than dead-ending; a *valid* custom
  symlink is left alone.
- *Repairs*: a missing bridge next to an existing `~/.ht-ai-agent` is
  re-created, and an **empty** `~/.ht-ai-agent` stub next to a populated
  un-migrated `~/.hermes` is cleared so the real migration runs (the resolver
  likewise never lets an empty new dir shadow a populated legacy home). When
  both homes hold data the state is ambiguous and nothing is touched.

Either way, the `~/.hermes → ~/.ht-ai-agent` symlink means the ~25 code paths
that still hardcode the legacy `os.environ.get("HERMES_HOME", ~/.hermes)`
fallback resolve to the **same** directory as `get_hermes_home()` — no split
data dir — without needing to touch each callsite. Migration skips entirely
when a home override is set (`HT_HOME`/`HERMES_HOME`/context override), when
`HT_SKIP_HOME_MIGRATION` is set, and under pytest (so tests never move real
dirs). On Windows without symlink privilege the migration rolls back / the
bridge is skipped (existing installs keep `~/.hermes`); the env alias still
works.

**Tests:** `tests/test_home_migration.py` covers the resolver preference,
atomic migrate-with-rollback, fresh-provision, and every `maybe_migrate_home`
guard; the bare-default assertions in `tests/test_hermes_constants.py` and
`tests/test_hermes_home_profile_warning.py` were updated to the new default.

**Fallback tidy-up:** the node-bootstrap home in `hermes_cli/main.py` now
routes through `get_hermes_home()`. The dotenv loader
(`hermes_cli/env_loader.py`) and the secret-cache fallback
(`agent/secret_sources/_cache.py`) resolve the **environment** home
(`HERMES_HOME` with the `HT_HOME` alias, else the platform default) —
deliberately *without* the profile contextvar override, so the home the loader
threads through and the cache's fallback always name the same directory (an
ambient profile context must not silently relocate the secret cache). Left
as-is on purpose:
`hermes_cli/dashboard_auth/audit.py` (kept import-light on a hot auth path; it
reads the env directly, which stays correct via the startup `HT_HOME` →
`HERMES_HOME` mirror and the `~/.hermes → ~/.ht-ai-agent` symlink — there is no
import cycle to avoid, `hermes_constants` importing only `ht_compat` + stdlib),
and the resolver-with-fallback sites whose *primary* path already calls
`get_hermes_home()` and only drop to the hardcoded string if that call fails:
`mcp_serve.py` / `tools/mcp_oauth.py` (guarded by `except ImportError`) and
`hermes_cli/slack_cli.py` (a broader `except Exception`). One further direct
read — the pre-import `_config_default_interface_early` probe in
`hermes_cli/main.py`, which runs before the resolver is importable — is
intentional and is bridged by the same symlink. The remaining hardcoded
`~/.hermes` fallbacks in `scripts/`,
`optional-skills/`, and some plugins are bridged by the `~/.hermes →
~/.ht-ai-agent` symlink for migrated and fresh installs, so they resolve to the
same directory without a callsite change.

## Module-name aliases (`ht_*` → `hermes_*`) — safe subset landed

The HT-branded names for the **top-level single-file modules** are importable
as aliases: `ht_constants`, `ht_state`, `ht_time`, `ht_logging`, `ht_bootstrap`.
Each is a thin shim that replaces itself in `sys.modules` with the real
`hermes_<name>` module, so `import ht_constants` and `import hermes_constants`
return the **same object** — no duplicate module state, singletons, or
isinstance surprises. `hermes_<name>` stays canonical (all internal code and any
pickled references keep using it); the alias just makes the new name work too.
They are declared in `[tool.setuptools] py-modules` so the wheel/image ship them
(guarded by `tests/test_module_aliases.py`).

**Deliberately not done — package aliasing / full internal rename.** Aliasing
the `hermes_cli` **package** (and the ~12,300 internal references) is *not*
safe as a thin shim: `import ht_cli.config` double-imports the submodule
(`ht_cli.config is not hermes_cli.config`), duplicating module state. Doing it
correctly needs a process-wide import hook (a `sys.meta_path` finder), which is
real risk in the hot path of every import for essentially no user-visible gain —
no code or user references the new module names. The full internal rename
remains a large, low-value pass to do only if there's a concrete reason.

## Left unchanged on purpose

The `hermes-tools` codex config key (written into users' `~/.codex/config.toml`),
the `hermes-agent` ACP registry id and OAuth `client_id`, and the "Nous Hermes"
model-family names are **identifiers other systems key on**, not branding —
renaming them breaks integrations for no user-visible gain. See the PR-1 rebrand
notes for the full list.
