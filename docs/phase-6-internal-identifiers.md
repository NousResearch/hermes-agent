# Phase 6 — Internal Identifier Rebrand (backward-compatible)

**Status:** In progress — the alias foundation has landed; the physical data-dir
migration and Python module renames are deferred follow-up increments.

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

## Deliberately deferred (each its own increment)

These are **not** in this increment because doing them wrong is destructive and
they each need dedicated care. They are safe to do next, on this same branch.

### 1. Physical `~/.hermes` → `~/.ht-ai-agent` data-dir migration

The `HT_HOME` env alias lands now, but the **on-disk default stays `~/.hermes`**.
Flipping the default is a separate change because:

- `_get_platform_default_hermes_home()` is called at import time from 30+ sites;
  a data *move* must never happen there.
- ~250 test files reference `.hermes` paths and several assert the literal
  default — the flip must reconcile all of them.

**Planned design:** resolver preference `HT_HOME → HERMES_HOME →
~/.ht-ai-agent (if exists) → ~/.hermes (if exists) → ~/.ht-ai-agent`, plus an
explicit `maybe_migrate_home()` run once from controlled startup (not at import)
that does an atomic `os.rename` of `~/.hermes → ~/.ht-ai-agent` and leaves a
back-symlink so anything still referencing the old path keeps working. Falls
back to keep-using-old if the rename can't be done (e.g. cross-device).

### 2. Python module renames (`hermes_cli` → `ht_cli`, …)

The bulk of the ~12,300 occurrences, and purely internal — no user value, high
churn. The backward-compatible way is a **shim**: create the new module and
leave the old name re-exporting from it (so `import hermes_cli` and any pickled
session state referencing the old module path keep working). This is a
mechanical, self-contained pass best done in isolation with its own review.

### Left unchanged on purpose

The `hermes-tools` codex config key (written into users' `~/.codex/config.toml`),
the `hermes-agent` ACP registry id and OAuth `client_id`, and the "Nous Hermes"
model-family names are **identifiers other systems key on**, not branding —
renaming them breaks integrations for no user-visible gain. See the PR-1 rebrand
notes for the full list.
