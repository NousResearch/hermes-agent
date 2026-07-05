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
| [D9](#d-9) | fix(docker) | Honor passwd home in container wrappers instead of hardcoding `/opt/data` | candidate-upstream |
| [D10](#d-10) | feat(gateway) | `X-Hermes-User-Id` header → agent `user_id` on the api_server platform (interlocutor identity) | candidate-upstream |
| [D11](#d-11) | fix(agent) | Allow ZWJ inside emoji grapheme clusters in context/memory/skills scanners | **pending upstream** — NousResearch/hermes-agent#12673 |
| [D12](#d-12) | feat(gateway) | `GroupContext` sidecar — unified group context injection for all platforms | **pending upstream** — candidate for NousResearch |

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

<a id="d-9"></a>
### D9 — fix(docker): honor passwd home in container wrappers instead of hardcoding `/opt/data`

- **Status:** active, **candidate for upstream PR**.
- **Fork-local touchpoints:**
  - `docker/main-wrapper.sh` — the `export HOME=/opt/data` override before
    privilege drop.
  - `docker/hermes-exec-shim.sh` — same pattern, mirroring `main-wrapper.sh`.
  - `docker/s6-rc.d/dashboard/run` — same pattern in the dashboard service.
  - `tests/test_docker_home_override_scripts.py` — the regression test that
    guards the HOME-reset contract (reconciled in PR #17; see note below).
- **Why:** All three container entrypoints unconditionally `export
  HOME=/opt/data` before dropping privileges, to keep `HOME`-anchored writes
  (XDG caches, lockfiles, `.config`) from landing under `/root` when
  `with-contenv` repopulates `HOME` from `/init`'s context. The override is
  necessary; the hardcoded value is the bug. Multi-tenant embedders that
  point the hermes user's passwd home at a per-tenant path (e.g. a layout
  that splits durable identity in one path from ephemeral per-session work
  in another) end up with `$HOME` contradicting `/etc/passwd` — the user's
  passwd home is created and chowned, then silently never honored as
  `$HOME`. Replace the hardcoded path with a passwd lookup that falls back
  to `/opt/data` when the lookup fails, so the stock Hermes image (which
  sets the hermes user's passwd home to `/opt/data`) behaves identically
  and embedders get the home dir they declared.
- **Test reconciliation (PR #17, forge-witt3rd):** the original passwd-home
  change (commit `361da380b`, merged as PR #15) edited
  `docker/s6-rc.d/dashboard/run` from `export HOME=/opt/data` to
  `export HOME="${_hermes_passwd_home:-/opt/data}"` but left
  `test_dashboard_run_resets_home_before_dropping_privileges` asserting the
  old literal substring `"export HOME=/opt/data"`, which no longer appears
  verbatim. That broke the test on `main`, so every PR opened afterward
  inherited the red regardless of what it touched. PR #17 reconciles the
  test to a behavior contract (per AGENTS.md "behavior contracts over
  snapshots"): assert that HOME is exported, anchored to the `/opt/data`
  default (directly or as the `${...:-/opt/data}` fallback), and reset
  *before* the `s6-setuidgid` privilege drop (verified via index ordering).
  The test edit should have ridden along with the original D9 script change;
  PR #17 is the catch-up.
- **Upstream disposition:** clean candidate for extraction; the fix is
  general (honoring passwd over hardcoded paths is the obviously-correct
  shape) and not coupled to any other fork-local feature. If extracted
  upstream, the script change and the reconciled test ship together.

<a id="d-10"></a>
### D10 — feat(gateway): `X-Hermes-User-Id` header → agent `user_id` on the api_server platform

- **Status:** active, **candidate for upstream PR**.
- **PR:** team-ajk/hermes-agent#16
- **Fork-local touchpoints:**
  - `gateway/platforms/api_server.py` — new `_parse_user_id_header()`
    (mirrors `_parse_session_key_header`'s auth/validation shape); a
    `gateway_user_id` parameter threaded through `_create_agent`,
    `_run_agent`, and the per-handler dispatch sites; sets `agent._user_id`
    after construction.
  - `tests/gateway/test_api_server.py` — `TestUserIdHeader` + a
    `_create_agent` user-id forwarding test.
- **Why:** Every other gateway platform (Telegram, Discord, Slack, …)
  constructs the agent with `user_id=source.user_id`, so plugins that
  resolve *who is speaking* from `agent._user_id` work. The api_server
  platform was the only one that never set it — it hardcoded
  `platform="api_server"` and passed only `gateway_session_key`. As a
  result, web-originated callers always presented an empty `sender_id`,
  and the janus interlocutor `system_prompt` hook (D8) could never
  distinguish the actual sender from the member — every web chat resolved
  to the member's relational pack. This delta adds the missing inbound
  identity: a caller sends `X-Hermes-User-Id: <stable-id>` and it lands on
  `agent._user_id`, giving the api_server platform the same per-sender
  identity surface every other platform already carries. Security mirrors
  `X-Hermes-Session-Key`: the header is an impersonation surface, so it is
  rejected unless `API_SERVER_KEY` is configured; control chars and
  over-long values are rejected. Absent header ⇒ unchanged member-fallback
  behaviour (fully backward compatible). Also adds `X-Hermes-User-Id` —
  and the pre-existing siblings `X-Hermes-Session-Key` /
  `X-Hermes-Session-Id` — to the CORS `Access-Control-Allow-Headers` list,
  so browser clients can send any Hermes custom header without a preflight
  failure (the sibling gap predated this PR; fixed here as the same class).
- **Upstream disposition:** clean candidate for extraction. It closes a
  real cross-platform parity gap (api_server is the only platform missing
  user identity) and is independent of the janus plugin — any plugin or
  memory provider keying on `agent._user_id` benefits. Depends conceptually
  on nothing fork-local; the consumer that motivated it is D8, but the
  mechanism is generic.

<a id="d-11"></a>
### D11 — fix(agent): allow ZWJ inside emoji grapheme clusters in context/memory/skills scanners

- **Status:** active. **This is a forward-port of a pending upstream PR** —
  see disposition below. Once the upstream PR merges this delta retires.
- **Upstream PR:** [NousResearch/hermes-agent#12673](https://github.com/NousResearch/hermes-agent/pull/12673)
  (open, authored by witt3rd). The upstream PR was also carried in the
  witt3rd personal fork as branch `fix/emoji-zwj-scan-false-positive`.
- **Fork-local touchpoints:**
  - `utils.py` — new shared helper `find_unsafe_invisibles()`,
    `_is_pictographic()`, `_zwj_in_emoji_context()`, and exported
    `BLOCKLIST_INVISIBLES` constant.
  - `tools/skills_guard.py` — `scan_file()` calls `find_unsafe_invisibles()`
    instead of iterating `char in INVISIBLE_CHARS`.
  - `tools/threat_patterns.py` — `scan_for_threats()` calls
    `find_unsafe_invisibles()` instead of a raw set intersection.
  - `tests/agent/test_prompt_builder.py` — 6 new test cases covering
    legitimate emoji ZWJ sequences, injection ZWJ outside emoji, and
    mixed content.
- **Why:** The invisible-unicode blocklist in `scan_for_threats()` (and the
  `skills_guard` scanner) categorically rejected ZWJ (`U+200D`). But ZWJ is
  a required component of emoji grapheme clusters — any gendered emoji
  (`🧙‍♂️`, `👩‍⚕️`), family emoji (`👨‍👩‍👧`), or multi-pictograph emoji uses
  `char + ZWJ + char [+ VS16]`. Any context file (`SOUL.md`, `AGENTS.md`,
  `.hermes.md`), memory entry, or skill file containing such emoji was
  silently blocked with a `[BLOCKED: … contained potential prompt injection
  (invisible unicode U+200D)]` message. The fix adds a shared helper
  `find_unsafe_invisibles()` that whitelists ZWJ only when it sits between
  two pictographic codepoints (skipping variation selectors), and rejects it
  everywhere else — preserving full injection protection while allowing the
  emoji that Janus substrate files use extensively (e.g. `🧙‍♂️` in SOUL.md).
  Upstream PR #28589 had already fixed the same false-positive for the cron
  prompt scanner; this PR completes the job for the three remaining scanners
  and factors the logic into a single shared helper.
- **Upstream disposition:** **pending upstream merge** as
  [NousResearch/hermes-agent#12673](https://github.com/NousResearch/hermes-agent/pull/12673).
  This fork-local copy is a forward-port to keep team-ajk current. When
  the upstream PR merges, drop this branch, retire this entry, and pull
  upstream normally.

## Retired

*Empty.* When an upstream PR merges a delta from above, move its entry here
with the upstream commit/PR reference and the SHA of the fork-local commit
that became redundant.

<a id="d-12"></a>
### D12 — feat(gateway): `GroupContext` sidecar for unified group context injection

- **Status:** active. **Candidate for upstream contribution.**
- **Branch:** `feat/group-context-sidecar` (team-ajk/hermes-agent)
- **Fork-local touchpoints:**
  - `gateway/platforms/base.py` — new `GroupContext` dataclass; optional
    `group_context: GroupContext | None = None` field on `MessageEvent`.
    Fully backward-compatible: `None` preserves all existing behaviour.
  - `gateway/run.py` — shared processing site after `is_shared_multi_user`
    sender-prefix path. Reads `event.group_context` and applies Layer 1
    (group identity block into `channel_prompt`, session-level, cache-stable)
    and Layer 2 (`[SenderName]: ` prefix on message text). Existing fallback
    path preserved for adapters not yet ported.
  - `plugins/platforms/telegram/adapter.py` — populates `GroupContext` in
    `_build_message_event()` for group/supergroup messages. Adapter no longer
    mutates `channel_prompt` or message text directly.
  - `gateway/platforms/api_server.py` — reads `X-Hermes-Group-Id`,
    `X-Hermes-Group-Name`, `X-Hermes-Sender-Display` headers and populates
    `GroupContext`. Enables Spire UI to stop packing metadata into the message
    body and send structured headers instead. *(In progress.)*
- **Why:** Without this patch, agents in group settings receive raw message
  text with no group context and no sender attribution — they cannot distinguish
  a group conversation from a 1:1 DM and implicitly treat every message as
  coming from their primary member. This is the foundational change for
  multi-being group presence (Telegram groups, Discord servers, Spire UI rooms,
  GitHub issue threads). Each platform adapter fills in what it knows; one place
  in `gateway/run.py` processes it uniformly. No double-attribution risk.
  Prompt cache preserved: same group = same Layer 1 block = stable prefix.
- **Upstream disposition:** strong candidate for NousResearch upstream. Platform-
  neutral design benefits every Hermes deployment with multi-platform or
  multi-being group presence. Discord, GitHub, and future adapters populate
  the sidecar; injection behavior is free. See AMBIENT-ROOM-INTELLIGENCE.md
  for the full design rationale.
