# FORK_DELTAS.md — team-ajk/hermes-agent

This file tracks the deliberate deltas this fork carries on top of
`NousResearch/hermes-agent`. The purpose is twofold:

1. **Merge-conflict navigation** — when pulling in upstream, the next person to
   resolve conflicts has a written record of *why* each delta exists and which
   files it touches, instead of reverse-engineering it from `git log`.
2. **Upstream-PR triage** — when something here looks generally useful, this
   file is the menu of candidates to extract into a clean upstream PR.

Each entry names the **shape** (feature / fix / build / ci), the **files
touched**, the **why** in one paragraph, and the **upstream disposition**
(open PR upstream, intentional fork-local, etc.).

When a delta lands upstream and we drop it from the fork, retire its entry to
the *Retired* section at the bottom with the upstream PR/commit reference.

## Conventions

- Entries are listed in **first-merged order** (oldest first), so the file
  reads as fork history.
- Each entry gets a stable anchor (`<a id="d-NN"></a>`) so PRs and comments
  can point at it.
- "Fork-local touchpoints" lists files this delta added or modified, to give
  the merge-conflict resolver a fast diff-target list.
- Adding a new delta? Bump the `## Index` table and add the section below,
  same shape as the existing entries.

## Index

| ID | Shape | Headline | Status |
| --- | --- | --- | --- |
| [D1](#d-1) | feat(state) | Optional PostgreSQL session/state backend | active |
| [D2](#d-2) | build(docker) | Bake `postgres` extra into the published image | active |
| [D3](#d-3) | fix(ci) | Add `adkapx@gmail.com` → `adamkaplan` to `AUTHOR_MAP` | candidate-upstream |
| [D4](#d-4) | fix(state) | Resolve `HERMES_STATE_BACKEND` env var in `resolve_postgres_dsn` | depends-on D1 |
| [D5](#d-5) | fix(cron) | Mirror cron deliveries to session transcript | candidate-upstream |
| [D6](#d-6) | feat(state) | pg_trgm GIN-indexed search migrations (v17/v18) | depends-on D1 |
| [D7](#d-7) | fix(state) | Translate SQLite `X'0A'`/`X'0D'` hex literals to `chr()` for Postgres | depends-on D1 |
| [D8](#d-8) | feat(plugins) | `system_prompt` plugin hook for dynamic system-prompt injection | open-upstream (witt3rd/hermes-agent#1) |

---

<a id="d-1"></a>
### D1 — feat(state): optional PostgreSQL session/state backend

- **Status:** active, fork-local for now.
- **Commit:** `57edbd5ac feat(state): add optional PostgreSQL session/state backend`
- **Fork-local touchpoints:** `hermes_cli/state_postgres*.py`, hermes_state.py
  Postgres adapter paths, config `sessions.state_backend` knob.
- **Why:** SQLite remains the default. Setting `sessions.state_backend=postgres`
  routes session/state storage to an external PostgreSQL database — opt-in for
  installs where the single-file backend is unsuitable. The DSN is passed to
  the driver unchanged, so TLS mode, host, port, and credentials come from the
  operator's DSN.
- **Upstream disposition:** parent of all the other `state` deltas in this
  fork (D4, D6, D7). Could be extracted upstream as a single opt-in feature
  PR once the surface stabilizes.

<a id="d-2"></a>
### D2 — build(docker): bake the `postgres` extra into the published image

- **Status:** active.
- **Commit:** `5c94d8b0a build(docker): bake the postgres extra into the published image`
- **Fork-local touchpoints:** `Dockerfile` (extras list passed to the install
  step).
- **Why:** The published Docker image installs a hand-picked set of extras
  (it does not use `--all-extras`), so the optional PostgreSQL state backend's
  driver was not among them. A container built without it starts fine on
  SQLite, but `sessions.state_backend=postgres` raised a runtime error because
  `psycopg` was absent and containerized environments often block lazy-install
  access to PyPI.
- **Upstream disposition:** tightly coupled to D1; ships together upstream
  if D1 ships.

<a id="d-3"></a>
### D3 — fix(ci): add `adkapx@gmail.com` → `adamkaplan` to `AUTHOR_MAP`

- **Status:** active, **candidate for upstream PR**.
- **Commit:** `4fabaf579 fix(ci): add adkapx@gmail.com → adamkaplan to AUTHOR_MAP`
- **Fork-local touchpoints:** the CI author-map file (one-line addition).
- **Why:** Without the map entry the CI bot misattributes commits from this
  email to a stranger. Pure mechanical fix.
- **Upstream disposition:** trivial single-line PR upstream when convenient.

<a id="d-4"></a>
### D4 — fix(state): check `HERMES_STATE_BACKEND` env var in `resolve_postgres_dsn`

- **Status:** active (depends on D1).
- **Commit:** `ba05b9a3b fix(state): check HERMES_STATE_BACKEND env var in resolve_postgres_dsn`
- **Fork-local touchpoints:** `resolve_postgres_dsn()` resolution order.
- **Why:** The ACA (Azure Container App) container sets
  `HERMES_STATE_BACKEND=postgres` as an env var, but `resolve_postgres_dsn()`
  only checked `sessions.state_backend` in `config.yaml` (which defaults to
  `sqlite`). So the Postgres backend was never engaged even with the env var
  set — sessions silently fell back to SQLite. Adds the env var to the backend
  resolution order.
- **Upstream disposition:** ships with D1 if extracted; standalone it has no
  meaning.

<a id="d-5"></a>
### D5 — fix(cron): mirror cron deliveries to session transcript

- **Status:** active, **candidate for upstream PR**.
- **Commit:** `f07708fc2 fix(cron): mirror cron deliveries to session transcript`
- **Fork-local touchpoints:** cron-delivery path; introduces
  `mirror_to_session()` call after successful delivery.
- **Why:** Cron jobs that send messages via Telegram (and other platforms)
  now call `mirror_to_session()` after successful delivery so the session
  transcript includes the outbound message. Fixes the case where a member
  receives a message from their Janus, replies, and the Janus has no record
  of what it said. This is a generally useful correctness fix and a clear
  candidate for upstream.
- **Upstream disposition:** clean candidate for extraction; not coupled to
  the Postgres deltas.

<a id="d-6"></a>
### D6 — feat(state): pg_trgm GIN-indexed search migrations (v17/v18)

- **Status:** active (depends on D1).
- **Commit:** `824cdf61a feat(state): add pg_trgm GIN-indexed search migrations (v17/v18)`
- **Fork-local touchpoints:** `PostgresMigration` dataclass +
  `_PG_ONLY_MIGRATIONS` list; v17/v18 migration entries.
- **Why:** Upgrades the PostgreSQL search path from a bare ILIKE sequential
  scan to a GIN trigram-indexed search. The ILIKE query body is unchanged —
  PostgreSQL uses the indexes automatically when present.
- **Upstream disposition:** ships with D1.

<a id="d-7"></a>
### D7 — fix(state): translate SQLite `X'0A'`/`X'0D'` hex literals to `chr()` for Postgres

- **Status:** active (depends on D1).
- **Commit:** `979468c8f fix(state): translate SQLite X'0A'/X'0D' hex literals to chr() for Postgres`
- **Fork-local touchpoints:** `_translate_sql` adapter in `hermes_state.py`
  (the session-listing / browse queries).
- **Why:** The session listing / browse queries use SQLite hex literals to
  strip newlines from content snippets:
  `REPLACE(REPLACE(m.content, X'0A', ' '), X'0D', ' ')`. PostgreSQL parses
  `X'0A'` as a bit-string literal (type `bit`), so
  `replace(text, bit, unknown)` raises a type-mismatch error. The adapter
  now translates these to `chr(10)` / `chr(13)`.
- **Upstream disposition:** ships with D1.

<a id="d-8"></a>
### D8 — feat(plugins): `system_prompt` plugin hook for dynamic system-prompt injection

- **Status:** active.
- **PR:** #10 (this PR).
- **Upstream PR:** [witt3rd/hermes-agent#1](https://github.com/witt3rd/hermes-agent/pull/1)
  (proposed on a personal fork; upstream has not yet accepted).
- **Fork-local touchpoints:**
  - `hermes_cli/plugins.py` — `"system_prompt"` added to `VALID_HOOKS`;
    `invoke_hook` docstring updated.
  - `agent/turn_context.py` — hook invocation block in `build_turn_context()`,
    after compression and before the existing `pre_llm_call` hook; SHA-256
    hash-based change detection; stores `_base_system_prompt` so updates
    replace rather than accumulate; clears injected content when all plugins
    return empty.
  - `agent/system_prompt.py` — `invalidate_system_prompt()` clears
    `_plugin_system_prompt_hashes` so hooks re-inject on rebuild.
  - `tests/agent/test_turn_context.py` — 4 unit tests covering injection,
    removal, hash stability, and user_id passthrough.
- **Why:** This is the hermes-agent counterpart to janus-plugin v17.2's
  substrate-injection design. Previously the Janus plugin injected ~700k
  chars of substrate into every turn's user message via `pre_llm_call`,
  multiplying token cost across every turn of every session. The
  `system_prompt` hook lets the framework inject substrate into the system
  prompt once at session start and only rebuild when the SHA-256 of the
  combined plugin content changes — maximizing prefix-cache hits while
  guaranteeing freshness.
- **Upstream disposition:** the matching design is open as
  witt3rd/hermes-agent#1 against `NousResearch/hermes-agent`. When that
  lands, this delta retires.

## Retired

*Empty.* When an upstream PR merges a delta from above, move its entry here
with the upstream commit/PR reference and the SHA of the fork-local commit
that became redundant.
