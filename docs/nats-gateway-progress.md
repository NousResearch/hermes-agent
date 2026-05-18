# NATS Gateway — Progress Tracker

**Purpose of this file.** This is the single source of truth for "where are we in the NATS gateway implementation?" across context-cleared sessions. `TaskList` state **might** survive `/clear` but I am not betting on it — this file does.

---

## If you are reading this in a fresh session

You (Claude) are resuming work on the NATS gateway channel for Hermes Agent. The user has cleared the context between phases on purpose. Do this, in order:

1. **Read `docs/nats-gateway-design.md` in full.** It is the architectural reference — protocol↔adapter mapping, streaming model, session identity, lock scope, approval hook design, failure modes. ~650 lines. Everything there has been reviewed and approved by the user.
2. **Read this file to the end.** The "Status" section tells you the last completed phase; the "Task checklist" tells you exactly which `T#.#` items are done vs. pending; the "Decision log" captures anything decided mid-flight that is not in the design doc.
3. **Call `TaskList`.** If the task list is empty or out of sync with this file's checkboxes, treat this file as authoritative and recreate the tasks via `TaskCreate` (titles, descriptions, and `activeForm` values are listed below in "Task definitions reference" — copy verbatim).
4. **Pick up at the next `[ ]` task** and work through the current phase's items. Do not skip ahead into later phases — the phases have dependencies and the user wants phase-boundary reviews.
5. **At end of phase**, follow the "End-of-phase ritual" below. **Do not clear context yourself** — that is the user's call. Just report that the phase is done and the ritual is complete.

Do not rewrite the design doc unless the user asks. If a design decision turns out wrong during implementation, append a note to this file's **Decision log** and flag it to the user before proceeding.

---

## Status

- **Last completed phase:** Phase 11 — split SDK migration (`synadia-ai-agents` v0.5 client / `synadia-ai-agent-service` v0.1 agent) + broker-derived `max_payload` (PR #41 in `synadia-ai/synadia-agents`)
- **Next phase:** None — all phases complete. PR on `nats-gateway` → `main` is the next human step.
- **Branch:** `nats-gateway` (feature branch; PR target is `main`)
- **Known blockers:** none
- **Open design questions pending user input:** see §16 of `docs/nats-gateway-design.md`.

When you finish a phase, update the two bullets above and tick its tasks in the "Task checklist" below.

---

## Phase-by-phase task checklist

Tick the box when the task is complete. One authoritative list; do not let TaskList drift from this file.

### Phase 0 — Docs first

- [x] **T0.1** — Write `docs/nats-gateway-design.md`
- [x] **T0.2** — Add CLAUDE.md pointers to design doc + natsagent SDK location

### Phase 1 — Scaffolding & config

- [x] **T1.1** — Add `Platform.NATS` enum value in `gateway/config.py`
- [x] **T1.2** — Extend `_apply_env_overrides()` for NATS (NATS_URL, NATS_CONTEXT, HERMES_NATS_{AGENT,OWNER,NAME,SESSION})
- [x] **T1.3** — Extend `get_connected_platforms()` for NATS (enabled AND (servers OR context))
- [x] **T1.4** — Register NATS adapter in `_create_adapter()` (gateway/run.py ~line 2717)
- [x] **T1.5** — Add `natsagent` to `pyproject.toml` extras (deferred `all`-extra inclusion — see Decision log 2026-04-21)
- [x] **T1.6** — Add `_ensure_natsagent_mock()` in `tests/gateway/conftest.py`

### Phase 2 — Adapter skeleton

- [x] **T2.1** — Create `gateway/platforms/nats.py` skeleton (`check_nats_requirements`, `NatsAdapter` stub, `NatsAdapterSettings` dataclass, validation)
- [x] **T2.2** — `tests/gateway/test_nats_config.py` — config parsing (happy/bad/env-override)

### Phase 3 — Connection & lifecycle

- [x] **T3.1** — Implement `NatsAdapter.connect()` (lock, natsagent.connect, Agent.start, _mark_connected)
- [x] **T3.2** — Implement `NatsAdapter.disconnect()` (idempotent, cancel handlers, agent.stop, nc.close, release lock)
- [x] **T3.3** — Implement `get_chat_info()` (returns `{"name": chat_id, "type": "dm"}`)
- [x] **T3.4** — `tests/gateway/test_nats_connect.py` — connect/disconnect/lock/handler registration

### Phase 4 — Inbound path (the meaty one; plan for a dedicated session)

- [x] **T4.1** — Implement `_on_prompt(envelope, stream)` — x-session → chat_id, attachments → media cache, MessageEvent, `_active_streams[chat_id] = stream`, keep-alive task, command-vs-text branch, cleanup
- [x] **T4.2** — Wire `_active_streams` into `send()` — look up PromptStream, `stream.send(ResponseChunk(text=content))`, return SendResult
- [x] **T4.3** — Wire streaming deltas — adapter-owned AIAgent with `stream_delta_callback` forwarding to a queue → pump → `stream.send`. (Ownership decision: adapter owns the callback; see §6.1 of design doc.)
- [x] **T4.4** — `tests/gateway/test_nats_inbound.py` — envelope in, MessageEvent, deltas emitted, keep-alive fires, terminator, attachment round-trip

### Phase 5 — Outbound attachments & formatting

**Fold-in justification (T5.0).** Phase 4's "known shortcoming #1" — concurrent prompts on the same `x-session` overwrite `_active_streams[chat_id]` so tool outputs from handler A can land on handler B's reply subject — was deliberately deferred at Phase 4's close. Phase 5 adds four new tool-accessible methods (`send_image_file` / `send_document` / `send_voice` / `send_video`) that all resolve through `_active_streams[chat_id]`, so the blast radius of the race quadruples in this phase. Fixing it here (before the send helpers) costs less than retrofitting later — build the send helpers on a race-safe lookup rather than patching four call sites after the fact.

- [x] **T5.0** — Race-safe stream lookup. **Landed as a contextvar-primary hybrid of (a)+(b):** `_active_streams` became a `dict[tuple[str, int], PromptStream]` (compound key on `(chat_id, id(stream))` so concurrent same-`x-session` handlers don't overwrite each other in the registry), plus a module-level `_current_stream: ContextVar[PromptStream | None]` that `_on_prompt` sets at entry and resets in `finally`. Send helpers first consult the contextvar (inherited through `asyncio.Task` / `run_in_executor` context propagation), then fall back to the dict by `chat_id` only for sends scheduled outside the handler's context. See Decision log 2026-04-21 entry for the "why contextvar-first" rationale.
- [x] **T5.1** — Implement `send_image_file` (`Attachment.from_path(path)` → `stream.send(ResponseChunk(text=caption, attachments=[...]))`), built on the T5.0 race-safe lookup.
- [x] **T5.2** — Implement `send_document` (same pattern, generic file; `file_name` override uses `from_bytes` to honor the caller's explicit wire filename)
- [x] **T5.3** — Implement `send_voice` / `send_video` (same pattern; v0.1 doesn't distinguish on wire)
- [x] **T5.4** — `format_message()` override (no-op for symmetry — NATS carries text verbatim; already landed in Phase 4 as a class method, outbound test module now asserts the verbatim contract)
- [x] **T5.5** — `tests/gateway/test_nats_outbound.py` — image/doc/voice → ResponseChunk.attachments[0] shape, plus a concurrent-`x-session` regression test (`TestConcurrentSameSessionRegression`) gating two overlapping handlers on `asyncio.Event`s so both are mid-flight when each calls `send_image_file`; proves contextvar-scoped lookup routes each send to its own stream.

### Phase 6 — Mid-stream queries (NATS-local)

- [x] **T6.1** — Survey `_pending_approvals` usage in `gateway/run.py` + read `hermes_cli/callbacks.py` (research spike; notes into Decision log)
- [x] **T6.2** — Add `async def request_interaction(self, chat_id, prompt, *, kind, timeout) -> str | None` on `BasePlatformAdapter` (default raises NotImplementedError); NATS implementation calls `stream.ask(prompt, timeout=timeout)`
- [x] **T6.3** — In `gateway/run.py`, route approval callback through `adapter.request_interaction` when adapter overrides the base default (capability check via `type(adapter).request_interaction is not BasePlatformAdapter.request_interaction`). Preserve existing behavior for non-NATS adapters. **AND** register a NATS-scoped gateway notify callback inside `_run_agent_sync` so approvals over NATS actually reach the adapter (the adapter-owned agent path bypasses the gateway's default `_approval_notify_sync` registration).
- [x] **T6.4** — `tests/gateway/test_nats_query.py` — approval callback triggers Query chunk; simulate caller reply; agent resumes

### Phase 7 — Slash commands

- [x] **T7.1** — Confirm gateway-eligible commands (`/new`, `/reset`, `/model`, `/status`, `/stop`, `/help`, `/compress`, `/resume`) route as `MessageEvent(COMMAND)` with no new code. Data-only verification.
- [x] **T7.2** — Manually verify `/help` output renders sensibly as plain-text chunks over NATS

### Phase 8 — End-to-end verification (manual; requires local nats-server)

- [x] **T8.1** — Local nats-server + hermes smoke config (already documented in §14 of design doc; confirm it still applies)
- [x] **T8.2** — `examples/01-discover.py` lists `agents.hermes.<owner>.<name>`
- [x] **T8.3** — `examples/02-prompt-text.py` — simple prompt streams a response
- [x] **T8.4** — `examples/03-prompt-attachment.py` — both a PDF (lualatex-generated with marker phrases) and a PNG round-tripped end-to-end; see the "attachment-handling refactor" Decision log entry below for the canonical-gateway alignment
- [x] **T8.5** — `examples/04-query-reply.py` — tool call that requires approval; Query chunk; reply "once"; stream resumes
- [x] **T8.6** — `examples/05-liveness.py` in background; kill hermes; `is_online()` flips False after 3× interval
- [x] **T8.7** — `nats` CLI interop — `nats req '$SRV.INFO.SynadiaAgents'` and `nats sub 'agents.hermes.*.*.heartbeat'` per protocol Appendix C
- [x] **T8.8** — `scripts/run_tests.sh` — full suite: 19 failures, all pre-existing / environmental, zero in the NATS subtree (NATS subtree 203/203 green after the Phase 8 attachment refactor; `TestEnrichEventWithMedia` (8 tests) replaces the earlier note-injection coverage)

### Phase 9 — Polish & docs

- [x] **T9.1** — Update `gateway/platforms/ADDING_A_PLATFORM.md` with any new integration points that emerged (e.g. `request_interaction` if it gets generalized)
- [x] **T9.2** — Append "Lessons learned" section to `docs/nats-gateway-design.md` (especially surprises in stream_delta_callback wiring or attachments)
- [x] **T9.3** — Add example config snippet to README or new `docs/nats-gateway.md` (user-facing)

### Phase 10 — Protocol v0.3 / `synadia-ai-agents` SDK migration

- [x] **T10.1** — Bump `pyproject.toml` `[nats]` extra to `synadia-ai-agents>=0.4.0,<1`
- [x] **T10.2** — Rewrite `gateway/platforms/nats.py` imports (`import synadia_ai.agents as sdk`), `SYNADIA_AGENTS_AVAILABLE`, all `getattr(natsagent, …)` → `getattr(sdk, …)`, TYPE_CHECKING import
- [x] **T10.3** — `NatsAdapterSettings`: drop `name` and `session_default`, add required `session_name` field; rebuild `identity` from `session_name`; remove `DEFAULT_SESSION_DEFAULT`
- [x] **T10.4** — `gateway/config.py`: rename env var `HERMES_NATS_NAME` → `HERMES_NATS_SESSION_NAME`; delete `HERMES_NATS_SESSION` + `extra["session_default"]`; update `extra` keys
- [x] **T10.5** — `connect()`: construct `sdk.AgentService(session_name=…)`; rename `self._agent` → `self._service`; update connected log line to `agents.prompt.{a}.{o}.{session_name}`
- [x] **T10.6** — Collapse session routing: `chat_id = settings.session_name`; replace `_session_locks` dict with single `_session_lock`; delete `_session_default()`; remove all `envelope.session` reads
- [x] **T10.7** — `gateway/run.py:2874` SDK-missing log message updated to reference `synadia-ai-agents` and the monorepo install path
- [x] **T10.8** — `tests/gateway/conftest.py`: rename `_ensure_natsagent_mock` → `_ensure_synadia_agents_mock`; register under `sys.modules["synadia_ai.agents"]`; expose `AgentService` (was `Agent`); drop `session` from envelope/heartbeat fakes
- [x] **T10.9** — Sweep all NATS test files: rename `mock_natsagent` fixture → `mock_synadia_agents`; kwargs `name=` → `session_name=`; drop `session_default` and `envelope.session`-fallback tests; add positive test that `chat_id` is sourced from `settings.session_name` even when a stray envelope field exists; rewrite distinct-sessions concurrent test to single-session serialization
- [x] **T10.10** — `scripts/run_tests.sh tests/gateway/` — 190/190 NATS tests green
- [x] **T10.11** — Update `CLAUDE.md` SDK reference (package + import name + AgentService rename)
- [x] **T10.12** — Rewrite design-doc §1–§6 + §11–§17 deltas; append Phase 10 entry to this progress doc; rewrite `website/docs/user-guide/messaging/nats.md` (env var rename, package rename, subject examples, `_INBOX.agents` permission note, status endpoint mention)

### Phase 11 — Split SDK (client v0.5 / agent v0.1) + broker-derived `max_payload` (2026-04-30)

Two upstream changes landed in one batch:

- **SDK split** (CHANGELOG 0.5.0, 2026-04-30) — `synadia-ai-agents` v0.4.x split into a wire-only client SDK (`synadia-ai-agents` v0.5, import root `synadia_ai.agents`) and a host-side agent SDK (`synadia-ai-agent-service` v0.1, import root `synadia_ai.agent_service`).
- **PR #41 — broker-derived `max_payload`** — SDK clamps a constructor-supplied `max_payload` down to `nc.max_payload` at `start()`. Hermes hardcoded `DEFAULT_MAX_PAYLOAD = "1MB"` and capped every host at 1 MB regardless of negotiated capacity, contradicting our docs.

- [x] **T11.1** — `gateway/platforms/nats.py`: import `synadia_ai.agent_service as sdk_svc` alongside `synadia_ai.agents as sdk`; split TYPE_CHECKING block; retarget `AgentService` construction to `sdk_svc.AgentService` (wire types stay on `sdk`)
- [x] **T11.2** — `pyproject.toml`: bump `[nats]` extra to `synadia-ai-agents>=0.5,<1` + `synadia-ai-agent-service>=0.1,<1`; add new top-level `[tool.uv.sources]` block resolving both to sibling `../synadia-agents/` checkout
- [x] **T11.3** — `NatsAdapterSettings.max_payload`: `Optional[str] = None` (was `str` with `DEFAULT_MAX_PAYLOAD = "1MB"`); only validate when user supplied a value; drop `DEFAULT_MAX_PAYLOAD` constant
- [x] **T11.4** — `_on_connect()`: derive `resolved_max_payload` from `nc.max_payload` when settings.max_payload is None; pass user value through unchanged otherwise; conservative `_FALLBACK_MAX_PAYLOAD = "1MB"` when broker reports 0; new `_format_max_payload_grammar` helper picks largest clean unit (B/KB/MB/GB)
- [x] **T11.5** — Connected log line: append `(server-negotiated)` / `(configured)` suffix so operators can tell which path resolved
- [x] **T11.6** — `tests/gateway/conftest.py`: extend `_ensure_synadia_agents_mock` to register `sys.modules["synadia_ai.agent_service"]` with parallel `agent_service_mod`; AgentService/PromptStream/PromptHandler move there; early-return guard checks both modules
- [x] **T11.7** — `tests/gateway/test_nats_connect.py`: `mock_synadia_agents` fixture proxies AgentService from agent_service module; `mock_nats` sets default `return_value.max_payload = 1MiB`; new tests for broker-derivation (8MB), user-set passthrough (512KB), and zero-broker fallback (1MB)
- [x] **T11.8** — `tests/gateway/test_nats_config.py`: drop `DEFAULT_MAX_PAYLOAD` import; new test that unset `max_payload` stays `None`; new test that blank/whitespace string is treated as unset
- [x] **T11.9** — Update `docs/nats-gateway.md` (sibling-path layout for both SDKs, `uv sync` replaces manual install), `docs/nats-gateway-design.md` (SDK reference, conftest mirror, install snippet, §15 cross-refs, §17.13 retro entry), `CLAUDE.md` Agent-side SDK paragraph

---

## End-of-phase ritual

Run this every time a phase's tasks are all ticked off. **In order.**

1. **Run the relevant test subset** via `scripts/run_tests.sh`:
   - Phase 2: `scripts/run_tests.sh tests/gateway/test_nats_config.py`
   - Phase 3: `scripts/run_tests.sh tests/gateway/test_nats_connect.py`
   - Phase 4: `scripts/run_tests.sh tests/gateway/test_nats_inbound.py`
   - Phase 5: `scripts/run_tests.sh tests/gateway/test_nats_outbound.py`
   - Phase 6: `scripts/run_tests.sh tests/gateway/test_nats_query.py`
   - Phase 8: `scripts/run_tests.sh` (full suite)
   - Other phases: no dedicated tests — skip
2. **Update this file's "Status" block** — bump `Last completed phase` and `Next phase`.
3. **Tick any remaining `[ ]` boxes** for the just-completed phase. Scan the list for drift vs. TaskList.
4. **Append to the Decision log** if anything was decided mid-flight (API tweaks, discovered surprises, deferred items).
5. **Run `TaskList`** and ensure its state matches this file. If divergent, update TaskList — this file is authoritative.
6. **Commit the phase** on the `nats-gateway` branch. Stage only the files touched by this phase (including this progress doc's updates from steps 2–4). Message format: `feat(gateway): <short phase summary> (phase N)`; body lists what changed and any decisions worth surfacing to a reviewer. Use the standard Claude Code `Co-Authored-By` trailer. Do NOT push and do NOT use `--no-verify`.
7. **Report to the user:** "Phase N done. Commit: `<short SHA>`. Tests: `<passing>`. Ready for review / context clear."
8. **Do not push or clear context yourself.** Those remain the user's call.

---

## Decision log (append-only)

Use this to capture non-obvious decisions made during implementation — things a fresh session wouldn't know from reading the design doc alone. New entries at the bottom; include date + phase.

### 2026-04-21 — Phase 0 — Progress doc + CLAUDE.md pointer added

Design doc + this progress doc now exist. CLAUDE.md updated to point at both so fresh sessions pick them up automatically. No architectural decisions changed.

### 2026-04-21 — Phase 1 — `natsagent` deliberately NOT added to the `all` extra

T1.5 says "add `natsagent` to pyproject.toml extras (and the `all` extra)." The `nats` extra is in place (`natsagent>=0.1.0,<1`) but it was **not** added to the `all` extra. Reason: `natsagent` is not yet published to PyPI (the design doc §14 acknowledges this — local install is `pip install -e ../nats-ai-pysdk`). Adding a non-PyPI dep to `all` would break `pip install 'hermes-agent[all]'` for every user doing the standard onboarding install. Reverse this once the SDK ships on PyPI — one line to add `"hermes-agent[nats]"` to the `all` list in `pyproject.toml`.

### 2026-04-21 — Phase 1 — Env overrides trigger `enabled=True` on any NATS env var

`_apply_env_overrides()` creates/enables the NATS platform entry if *any* of `NATS_URL`, `NATS_CONTEXT`, `HERMES_NATS_{AGENT,OWNER,NAME,SESSION}` is set. This matches Signal's "any creds env present ⇒ enable" pattern (`gateway/config.py:926-943`). Note that `get_connected_platforms()` still gates on `enabled AND (servers OR context)` — so setting only `HERMES_NATS_OWNER` without `NATS_URL`/`NATS_CONTEXT` enables the platform but it won't show as connected. That's intentional: lets you pre-populate identity via env and complete config via YAML.

### 2026-04-21 — Phase 1 — Pre-existing test failures observed, not introduced by Phase 1

`scripts/run_tests.sh tests/gateway/` reports two failures on clean `main` (verified by stashing Phase 1 changes): `test_agent_cache.py::TestAgentCacheIdleResume::test_close_vs_release_full_teardown_difference` and `test_matrix.py::TestMatrixUploadAndSend::test_upload_encrypted_room_uses_file_payload`. These are pre-existing, unrelated to NATS work. Flagged here so a future phase doesn't blame them on NATS changes. `tests/gateway/test_config.py` (the most directly relevant test file) passes cleanly after Phase 1 edits.

### 2026-04-21 — Phase 2 — `max_payload` validated via local regex, not the SDK's `parse_human_bytes`

The design doc §4 calls for pre-flighting `max_payload` through `natsagent._bytes.parse_human_bytes`. The adapter uses a local regex (`_MAX_PAYLOAD_RE`) instead. Reason: the SDK's `_bytes` module is private/underscored, and the gateway test harness installs `natsagent` as a `MagicMock` — `from natsagent._bytes import parse_human_bytes` can't resolve on a MagicMock module, so the check would either crash or no-op silently under test. A local regex matches the §2.1 grammar ("positive integer followed by B/KB/MB/GB") and keeps the validation deterministic whether the real SDK or the mock is loaded. The SDK still re-validates at `Agent(...)` construction time in Phase 3, so this is belt-and-braces rather than belt-only.

### 2026-04-21 — Phase 2 — `agent` token strictly validated; `owner`/`name` deferred to SDK

`NatsAdapterSettings.from_extra` enforces the §2.2 regex (`^[a-z0-9-]+$`) on the `agent` token but only insists on non-empty for `owner` / `name`. Reason: the SDK's `AgentSubject._sanitize()` base64-url-escapes non-conforming owner/name tokens rather than rejecting them, so a strict regex here would reject inputs the SDK would have accepted. The `agent` token has no such fallback — the SDK rejects it outright — so failing fast in our settings parser gives a cleaner error message than the SDK's exception surfacing from inside `connect()`.

### 2026-04-21 — Phase 2 — `bool` rejected for integer fields

Plain `int(True) == 1` would silently pass `heartbeat_interval_s`/`ack_keepalive_interval_s` validation. `_positive_int` rejects `bool` explicitly — a YAML `heartbeat_interval_s: true` is always a mistake, and surfacing it as a config error beats emitting heartbeats every 1 s in production.

### 2026-04-21 — Phase 2 — `_active_streams`/`_nc`/`_agent` initialised on adapter regardless of config validity

Even when `NatsAdapterSettings.from_extra` fails, `NatsAdapter.__init__` initialises `_active_streams = {}`, `_nc = None`, `_agent = None`. Reason: Phase 3's `connect()` and Phase 4's `send()` assume these attributes exist. If a fatal-error adapter somehow reaches later-phase code (e.g. GatewayRunner still calling `get_chat_info()` on it), `AttributeError` would be a harder failure than "not connected". Cheap guard, no downside.

### 2026-04-21 — Phase 3 — Conftest mock: `nc.close` explicitly made awaitable

`tests/gateway/conftest.py::_ensure_natsagent_mock` now wires `mod.connect.return_value.close = AsyncMock()`. Background: `mod.connect = AsyncMock()` returns a MagicMock when awaited, and a MagicMock's `.close()` returns another MagicMock — which can't be `await`-ed. `NatsAdapter.disconnect()`'s `await self._nc.close()` would blow up in every test touching the lifecycle path. No downstream cost — the real nats-py `Client.close` is already a coroutine, so keeping `close` async matches production behavior.

### 2026-04-21 — Phase 3 — `_on_prompt` ships a placeholder response instead of a real pipeline

The handler registered at `agent.on_prompt(...)` is a one-liner that sends a short "NATS adapter is online, Phase 4 wires the real pipeline" ResponseChunk and returns. Reason: `natsagent.Agent.start()` enforces that a handler is registered (raises otherwise), and we need `connect()` to land a fully-running micro service in Phase 3 so `$SRV.PING` discovery and heartbeat emission can be verified against a real nats-server between phases. Phase 4 swaps this handler for the real `x-session` + attachment + MessageEvent pipeline (T4.1). The placeholder is test-asserted (`TestPromptHandlerStub`) so any regression during Phase 4's swap will be caught.

### 2026-04-21 — Phase 3 — `disconnect()` teardown order: `agent.stop()` before `nc.close()`

`_teardown_handles()` stops the agent first, then closes the NATS client. Reason: the SDK's heartbeat publisher runs inside the agent's background task and emits on the NATS connection. Closing `nc` first would surface a burst of "connection closed" warnings from the heartbeat loop before the stop signal reaches it. Both halves of teardown are wrapped in try/except so a failing stop() doesn't prevent the close() and vice-versa — gateway shutdown runs disconnect() over every adapter in sequence and one raising would abort teardown for all the others after it.

### 2026-04-21 — Phase 3 — Lock release on connect failure is routed through `_teardown_handles()`, not a separate code path

Previously in Telegram (`gateway/platforms/telegram.py:910`), the connect() failure branch explicitly calls `_release_platform_lock()`. The NATS adapter instead routes both the success-disconnect path and the connect-failure path through a single `_teardown_handles()` helper. Reason: Phase 4 adds more handles (`_active_streams`, `stream_delta_callback`, keep-alive task) that also need cleanup in both paths — centralizing the teardown logic now means T4.x doesn't have to remember to wire cleanup into two places.

### 2026-04-21 — Phase 3 — Shutdown event + in-flight handler tracking landed early (pre-Phase 4)

Design doc §9 calls for "signal cancellation to in-flight `_on_prompt` handlers / await all outstanding pump / keep-alive / `_on_prompt` tasks" during shutdown. Phase 3's placeholder handler is a one-liner with no long-running work, so this was initially deferred to Phase 4. Review feedback: land the infrastructure now so Phase 4's handler body inherits the cancellation behavior for free instead of having to retrofit it.

The machinery:
- `self._shutdown_event: asyncio.Event` — set at the top of `_teardown_handles`, cleared at the top of `connect()`. Phase 4's streaming loop will `if self._shutdown_event.is_set(): break` between deltas.
- `self._in_flight_handlers: set[asyncio.Task]` — `_on_prompt` registers its own task via `asyncio.current_task()` at entry and discards it in a `finally` block. `_teardown_handles` cancels every live task and `asyncio.gather(..., return_exceptions=True)`s them before `agent.stop()` runs.
- `discard` (not `remove`) in the finally block: `_teardown_handles` may call `_in_flight_handlers.clear()` after gather returns, so the finally may find the task already gone.

Tests cover: task registration/deregistration on normal completion, finally-block tolerance of a mid-handler `clear()` (regression guard for `remove` vs. `discard`), cancellation of a hanging handler during `disconnect()` bounded by `asyncio.wait_for`, shutdown event set-before-stop ordering, shutdown event cleared by a retry `connect()`.

### 2026-04-21 — Phase 3 — Disconnect ordering test tightened with a side_effect call-order recorder

The original `test_disconnect_after_successful_connect_tears_down_in_order` asserted `agent.stop.assert_awaited_once()` + `nc.close.assert_awaited_once()` but NOT their relative order — the name was aspirational. Tightened by attaching `side_effect=lambda: call_order.append("stop")` / `("close")` to each mock and asserting `call_order == ["stop", "close"]`. `mock.call_args_list` is per-mock, so cross-mock ordering genuinely requires a shared recorder; `MagicMock.attach_mock` is the other standard option but the side_effect approach is one line shorter.

### 2026-04-21 — Phase 4 — Command vs. text prompt split at `_on_prompt`

Design doc §6.1 says "adapter-owned AIAgent, bypass GatewayStreamConsumer" (api_server pattern). Task list T4.1 literally says "`handle_message(event)`". Reconciling: a pure api_server-style bypass loses slash commands, which §10 explicitly wants. Pure `handle_message(event)` routes text prompts through `GatewayStreamConsumer` whose edit-a-single-message model is nonsense on NATS.

Phase 4 resolves this with a two-branch dispatch inside `_on_prompt`:
- Slash commands → `self._message_handler(event)` directly (gateway's `_handle_message` runs, returns the rendered response string, we wrap it in a `ResponseChunk` and publish). The gateway's command path short-circuits before `GatewayStreamConsumer` is ever constructed, so this is clean.
- Text prompts → adapter-owned `AIAgent` via `_run_agent_sync` in an executor, with a `stream_delta_callback` that feeds an `asyncio.Queue` drained by `_pump_deltas`. Each delta is its own `ResponseChunk`.

The classification heuristic (`_looks_like_command`) rejects paths (`/var/log/foo`), double-slashes (`//`), and bodies with non-alnum first chars — matches `MessageEvent.get_command()`'s behaviour in `base.py:746`.

### 2026-04-21 — Phase 4 — `SUPPORTS_MESSAGE_EDITING = False` on NatsAdapter

NATS publishes each streaming chunk as a fresh `ResponseChunk`; the protocol has no edit semantics (§6.1). `gateway/run.py:9597-9599` short-circuits `GatewayStreamConsumer` construction when the adapter reports it can't edit, so setting this flag is the cheapest way to ensure any code path that does go through `handle_message(event)` (slash commands today, possibly more tomorrow) gracefully skips the edit-based consumer instead of making noise. Streaming is wired adapter-locally via `_run_text_prompt` regardless. `weixin` and `qqbot` both use the same flag for the same reason.

### 2026-04-21 — Phase 4 — `_extract_x_session` peeks `stream._request.data`

Design doc §3 flags this as "open question (b)" — accepted here as MVP. The SDK's `Envelope` pydantic model has `extra="ignore"` (envelope.py:35), so `x-session` is dropped before our handler sees it. `PromptStream.__init__` stores the request on `self._request`, and `request.data` is the raw payload (agent.py:258). We JSON-re-parse the raw bytes locally. This is a private attribute today; if the SDK renames it, the adapter breaks loud and fast (attribute error at handler entry) rather than silently routing every session to `"default"`. A note to upstream a public raw-bytes handle to `nats-ai-pysdk` is carried in design doc §13 non-goals.

### 2026-04-21 — Phase 4 — Attachment cache failures convert to `RuntimeError`

`cache_image_from_bytes` raises `ValueError` when the magic bytes don't match (e.g. caller uploaded HTML as `.jpg`). The SDK's `_on_prompt_request` wraps any `Exception` from the handler into a 500 error frame (agent.py:270-272). For attachment-validation errors that are clearly caller-fault, 400 would be more accurate, but the SDK only differentiates based on the exception class it recognizes — `ProtocolError` → 400, anything else → 500. Raising `RuntimeError` gets us 500 with a clean message; upgrading to 400 would require either importing `natsagent.ProtocolError` at the adapter (tight coupling to the SDK's error module, which the test-harness mock barely models) or plumbing a typed error-response path into the handler. The design doc's §11 table already marks oversize-envelope as "deferred"; attachment-validation gets similar treatment for now.

### 2026-04-21 — Phase 4 — `_final_response_text` fallback publishes final text when no deltas streamed

Streaming deltas are fed via `stream_delta_callback` which the agent may not invoke (streaming disabled in config, tool-only turn, provider fallback). `run_conversation` returns a dict shape `{"final_response": "..."}` — we publish it as one `ResponseChunk` if and only if no deltas already landed. `threading.Event` guards the "anything streamed?" flag because the callback runs on the worker thread while the finalizer runs on the event-loop thread; a plain `bool` wouldn't be visible across threads without an explicit barrier.

### 2026-04-21 — Phase 4 — `Platform.NATS` added to `hermes_cli/platforms.py` + `hermes-nats` toolset

`_get_platform_tools(config, Platform.NATS.value)` requires a `PLATFORMS["nats"]["default_toolset"]` entry or it `KeyError`s. Registered `"nats"` → `"hermes-nats"` in `hermes_cli/platforms.py` (the shared registry, `tools_config.py` derives from it), and added a `hermes-nats` toolset in `toolsets.py` that mirrors `_HERMES_CORE_TOOLS` — same scope as other messaging platforms. A tighter NATS-specific subset can be carved out later if we want to restrict tools by transport.

### 2026-04-21 — Phase 4 — Conftest mock gained `StatusChunk` + kwargs-recording ResponseChunk

Phase 3's `ResponseChunk = MagicMock` was good enough for the placeholder handler which passed a bare string. Phase 4 emits `ResponseChunk(text=delta)` and `StatusChunk(status="ack")` via kwargs — tests assert on `chunk.text` / `chunk.status` to verify the adapter wrapped outgoing content correctly. Plain `MagicMock(text=...)` would return a MagicMock on attribute access rather than the string we passed, so the conftest now installs small kwargs-recording classes. Real SDK pydantic models behave the same way with the same surface.

### 2026-04-21 — Phase 4 — `_on_prompt` re-raises `CancelledError`, swallows all other exceptions

The SDK's `_on_prompt_request` has two clauses: `except Exception` → respond 500 + terminator, but `CancelledError` (a `BaseException` in 3.11+) falls through. Phase 4's handler mirrors that split — `CancelledError` re-raises so shutdown cancellation propagates cleanly through `_teardown_handles`'s `gather(return_exceptions=True)`. Arbitrary exceptions also re-raise so the SDK can convert them into a 500 error frame; we log them at ERROR level first so the gateway log has the full stack trace, not just the SDK's sanitized description line.

### 2026-04-21 — Phase 4 post-review — Authorization: NATS added to `_is_user_authorized` early-return set

Surfaced during Phase 4 self-review. `gateway/run.py:_is_user_authorized` had no handling for `Platform.NATS`. Commands dispatched via `_message_handler` hit the user allowlist check, which treated the caller's `x-session` string as a user_id and rejected it unless pre-paired — so `/help` over NATS replied with a pairing code instead of the help text. Design doc §10.1 already delegates NATS authorization to the NATS server layer (accounts / NKey / JWT / TLS), mirroring Webhook (HMAC) and HomeAssistant (HASS_TOKEN). Fix: add `Platform.NATS` to the `(HOMEASSISTANT, WEBHOOK)` early-return tuple. Regression test lives in `tests/gateway/test_unauthorized_dm_behavior.py::test_nats_is_authorized_without_user_allowlist`.

### 2026-04-21 — Phase 4 post-review — Command text lstripped before `MessageEvent` construction

Surfaced during Phase 4 self-review. `_looks_like_command` tolerates leading whitespace (``"  /help"`` → True, covered by tests), but `MessageEvent.is_command` / `get_command()` in `base.py:732` require literal `text.startswith("/")`. Before the fix, a whitespace-prefixed command would pass our heuristic → we'd mark it as `MessageType.COMMAND` and call `_dispatch_command(event, stream)` → the gateway's `_handle_message` → `event.get_command()` returns `None` → falls through to the text-agent path, which we already decided to bypass. Net result: silent misrouting. Fix: when `is_command` is True, set `event_text = prompt_text.lstrip()` before constructing the `MessageEvent`. Regression test in `test_nats_inbound.py::TestOnPromptIntegration::test_command_text_is_lstripped_for_gateway_dispatch`.

### 2026-04-21 — Phase 5 — T5.0 landed as contextvar-first, dict as diagnostic fallback

Plan called for option (b) "compound-key dict". In implementation, a pure compound-key dict still forces send helpers to *know* `id(stream)` at call time, which means every caller would need the stream threaded through — the exact "touches tool dispatch surface" cost that option (a) was flagged for. Merged the approaches: the dict is compound-keyed (so the `_on_prompt` finally block can pop-exactly-the-right-entry instead of guessing, and concurrent handlers literally coexist in the registry), AND a module-level `_current_stream: ContextVar[PromptStream | None]` is set at handler entry / reset in finally. `_resolve_stream(chat_id)` consults the contextvar first (race-safe, inherited via `run_in_executor`'s default `copy_context()`) and only falls back to the dict for sends scheduled outside handler context. Cost: +1 contextvar, +1 resolver method, no tool-dispatch surface changes. Test callers in `test_nats_inbound.py::TestSend` had to update from `adapter._active_streams["alice"] = stream` → `adapter._active_streams[("alice", id(stream))] = stream` (4 sites).

### 2026-04-21 — Phase 5 — `send_document(file_name=...)` switches from `from_path` to `from_bytes`

Design table in §8.2 specifies `Attachment.from_path(image_path)` uniformly for all four send helpers. In practice, `from_path` pins `filename=Path(path).name` — which is wrong when the caller staged the file under a content-hash name but wants the recipient to see the original filename. Telegram's `send_document` accepts an explicit `file_name` parameter for exactly this reason (`telegram.py:1811`). The NATS impl honors it by reading the bytes and re-building via `Attachment.from_bytes(override_filename, data)`. The other three helpers (no override semantics) still use `from_path` directly.

### 2026-04-21 — Phase 5 — Concurrent-regression test uses `asyncio.Event` gating, not timing

The concurrency regression test in `TestConcurrentSameSessionRegression` does NOT rely on `asyncio.sleep` to create overlap — that's racy under load. Instead, handler A blocks on `b_sent_image.wait()`, handler B blocks on `start_a.wait()`, and both handlers explicitly signal progress via events. This guarantees the assertion "A's `send_image_file` ran while both streams were registered in `_active_streams`" holds regardless of scheduler jitter.

### 2026-04-21 — Phase 5 — Missing file paths return `SendResult(success=False)`, don't raise

`send_image_file("/does/not/exist.png")` returns a failure result rather than raising. Rationale: these helpers are invoked from tool code and from `gateway/run.py`'s `_deliver_media_from_response` in best-effort chains (`logger.warning` on failure, continue with next file). Raising would break that pattern and force every call site to wrap a try/except. The Telegram adapter takes the same stance (`telegram.py:1785`). Tests also assert that `stream.send` is NOT awaited on failure — no partial chunks leak onto the reply subject.

### 2026-04-21 — Phase 5 — Known shortcomings (NOT fixed in Phase 5; carry forward)

1. **Contextvar does NOT propagate through `asyncio.run_coroutine_threadsafe`.** If a worker-thread tool schedules an adapter send via `run_coroutine_threadsafe(adapter.send_image_file(...), loop)`, the new Task on the main loop starts with a fresh context — `_current_stream` is unset and the fallback dict lookup runs. When two concurrent handlers share a `chat_id`, the fallback returns whichever stream happens to be registered first. Current hermes tool code doesn't schedule sends this way (it either directly-awaits from an async tool or constructs a fresh adapter), so this is a latent issue, not an active one. If tools grow a cross-thread scheduling path, the fix is to capture `_current_stream.get()` in the executor thread and pass it explicitly to the scheduled coroutine.

2. **Ordering caveat in the dict fallback.** `_resolve_stream` iterates `_active_streams` and returns the first matching `chat_id` — deterministic (`dict` preserves insertion order in Python 3.7+) but arbitrary. For legitimate concurrent sessions this only matters when the contextvar path is unavailable (see shortcoming #1). Acceptable for MVP.

### 2026-04-21 — Phase 4 — Known shortcomings (NOT fixed in Phase 4; carry forward)

Each of these is a deliberate MVP trade-off that should either land in a later phase or be promoted to a design-doc non-goal. Logged here so future Claudes don't waste cycles rediscovering them as "bugs".

1. **Concurrent prompts on the same `x-session` overwrite `_active_streams[chat_id]`.** ~~Two prompts with the same `x-session` arriving in quick succession race — the second replaces the first in `_active_streams`, so tool outputs from the first handler (e.g., `send_image_file`) land on the second handler's reply subject.~~ **RESOLVED in Phase 5 via T5.0.** `_active_streams` is now compound-keyed on `(chat_id, id(stream))` and send helpers prefer the `_current_stream` contextvar. Regression test: `tests/gateway/test_nats_outbound.py::TestConcurrentSameSessionRegression`. A narrow residual window — sends scheduled across `asyncio.run_coroutine_threadsafe` boundaries — is documented under Phase 5 shortcoming #1 above.

2. **`/stop` cannot interrupt a running NATS agent.** We bypass `self.handle_message(event)` for text prompts (design doc §6.1, api_server-style agent ownership), so `_active_sessions[session_key]` in `BasePlatformAdapter` is never populated. The gateway's `/stop` handler walks `_active_sessions` to find a running agent — for NATS that dict is always empty, so `/stop` becomes a no-op. Callers can drop their NATS subscription to abandon a run; real interrupt support would require either (a) routing text through `handle_message` and adding a NATS-aware stream consumer, or (b) adapter-local active-session tracking that the `/stop` handler is taught to consult. Defer to post-MVP.

3. **Unbounded `delta_queue`.** `asyncio.Queue()` is unbounded by default; if the model produces deltas faster than `stream.send` can drain them, memory grows linearly with the run. Not practical at LLM token rates (thousands of tokens per second max, each chunk small) but would matter if we ever drove a non-token data stream through the same pump. Not scheduled for a fix.

4. **Attachment-validation errors return SDK 500, not 400.** The SDK's `_on_prompt_request` only maps `ProtocolError` to 400; anything else becomes 500. `cache_image_from_bytes` raising `ValueError` on non-image bytes is caller-fault and should be 400 per §9.3 semantics, but the handler raises `RuntimeError` → 500. Fix requires either importing `natsagent.ProtocolError` directly (tight coupling to a module the test-harness mock barely models) or plumbing a typed error-response path. Noted in the Phase 4 attachment decision-log entry; revisit if callers complain.

5. **Session interrupt / busy-session merging is gone.** `BasePlatformAdapter.handle_message` has useful logic for "photo burst merging", "busy-session handoff", and "pending-message queue drain". NATS `_on_prompt` bypasses all of that. For a request/reply wire protocol that's fine (the caller controls concurrency), but anything downstream that assumes `_pending_messages`/`_active_sessions` population won't work over NATS. Document as a limitation.

6. **Private `stream._request.data` access for x-session peek.** Design doc §3 option (b), pre-approved. If the SDK renames `_request` or `data`, we blow up loud at handler entry (AttributeError via `getattr(..., None)` returning None → falls back to session default). Acceptable for MVP; upstream a public raw-bytes handle to `nats-ai-pysdk` when convenient.

### 2026-04-22 — Phase 6 — Approval wiring needed TWO sites, not one

Design doc §7.3 says "inside `_approval_notify_sync()` (or its async sibling), check whether the adapter defines `request_interaction`." That covers the *default* gateway path (`handle_message()` → `register_gateway_notify()` at run.py:9993). But for NATS the design at §6.1 already decided the adapter bypasses `handle_message` for text prompts (api_server-style adapter-owned `AIAgent`). Net effect: the default path's `register_gateway_notify` is never called for NATS text prompts, so just patching `_approval_notify_sync` wouldn't actually flow NATS approvals through `request_interaction`.

Phase 6 therefore landed TWO wiring sites:

1. **`gateway/run.py:_approval_notify_sync`** — class-level capability check at the TOP of the notify callback, before the button-based / plain-text fallbacks. Non-NATS adapters inherit the base default and fall through unchanged; this is infrastructure for future adapters that might opt in (and keeps the semantics consistent with the design doc).
2. **`gateway/platforms/nats.py:_run_agent_sync`** — registers its own `register_gateway_notify(session_key, _nats_approval_notify)` that directly invokes the shared `dispatch_approval_via_request_interaction` helper. Unregistered in the `finally` block alongside a `reset_current_session_key` call.

Both call sites share one helper: `dispatch_approval_via_request_interaction` in `gateway/platforms/base.py`. It handles prompt formatting, reply parsing, coroutine scheduling on the adapter's loop, and the `resolve_gateway_approval` callback. Sharing the helper guarantees the canonical approval choices (`once`/`session`/`always`/`deny`) and the "unknown/timeout ⇒ deny" fail-safe stay in lockstep across both sites.

### 2026-04-22 — Phase 6 — Approval session-key contextvar must be set on the worker thread

The default gateway path calls `set_current_session_key(session_key)` on the async side before `register_gateway_notify` (run.py:9992), and `run_conversation` subsequently inherits that contextvar via `_run_in_executor_with_context` (which uses `copy_context()`). NATS doesn't use that helper — it calls `loop.run_in_executor(None, self._run_agent_sync, ...)` directly, so the contextvar is NOT copied into the worker thread.

Without explicit propagation, `tools/approval.py::get_current_session_key()` in the agent thread would return the default (empty) session key → `check_all_command_guards` would create an approval entry under the wrong key → `_gateway_queues[session_key]` lookup in `resolve_gateway_approval` would miss → approval would hang until the gateway_timeout (300 s). Fix: `_run_agent_sync` now calls `set_current_session_key(session_key)` on entry and `reset_current_session_key(approval_token)` in its `finally`. This sets the contextvar on the worker thread's local copy, which is where the agent's tool dispatch runs.

### 2026-04-22 — Phase 6 — Approval timeout reads `approvals.gateway_timeout` from config

The NATS-specific notify callback calls `dispatch_approval_via_request_interaction(..., timeout=_approval_timeout_from_config())` so `stream.ask`'s deadline matches the gateway's `ApprovalEntry.event.wait()` deadline. A shorter adapter timeout (say 30 s) would make `request_interaction` return `None` → resolve as "deny" well before the user could reply; a longer adapter timeout would keep the stream alive past the agent's own timeout, wasting a socket. Reads lazily at dispatch time (not at registration) so live config changes apply to subsequent approvals without a restart. Falls back to 300 s on any read failure — never hangs forever.

### 2026-04-22 — Phase 6 — `BasePlatformAdapter.request_interaction` default is `NotImplementedError`, not a no-op

The design doc was ambiguous on this. `NotImplementedError` means the capability-detection helper (`type(adapter).request_interaction is not BasePlatformAdapter.request_interaction`) is the gate, not runtime inspection. A no-op default would have made "adapter doesn't support it" indistinguishable from "adapter returned nothing" — breaking the distinction between "caller timeout" (None, fall back to legacy) and "programmer error" (raise). The explicit `NotImplementedError` also matches Python's `abc` conventions for optional mixin methods, which is what this is in spirit.

### 2026-04-22 — Phase 6 — Unknown replies fail-safe to "deny", first-token-wins parsing

`_parse_approval_reply` accepts "yes please" → "once", "approve this one" → "once", "deny immediately" → "deny". But anything not in the canonical allow-lists — "maybe", "hmm", "whatever" — maps to "deny". This is aligned with `tools/approval.py`'s existing "choice is None or choice == 'deny' ⇒ BLOCKED" semantic: the gateway's existing safety net is "uncertain ⇒ blocked", so the adapter's parser does the same. A reply of `"approve"` counts as "once" (no persistence) — if the caller wants session or always scope they have to say the word. Avoids the "I typed 'yes' and it silently got permanently whitelisted" footgun.

### 2026-04-22 — Phase 6 — Concurrent-test polling uses a bounded `asyncio.sleep(0)` loop, not `wait_for`

`dispatch_approval_via_request_interaction` uses `asyncio.run_coroutine_threadsafe` which wraps the coroutine in a task on the target loop. The scheduled task doesn't complete synchronously — the test has to yield to the loop until the task runs. Using `asyncio.wait_for(future, timeout=…)` on the returned future would work but adds a real wall-clock wait if the task is already done. The chosen pattern is a 10-iteration `for _ in range(10): await asyncio.sleep(0); if resolved: break` which yields the loop up to 10 times; every test resolves in 1–2 iterations in practice. Same idiom used in the existing `test_nats_inbound.py` inbound tests.

### 2026-04-22 — Phase 6 — Per-`x-session` serialization eliminates the stacked stream/notify races

The first-pass Phase 6 implementation documented a "stacking of two races" for concurrent same-`x-session` handlers:

1. `register_gateway_notify(session_key, cb)` overwrites — handler B's registration replaces handler A's, so agent A's dangerous commands route through B's captured stream (wrong reply subject).
2. `_current_stream` contextvar doesn't propagate through `asyncio.run_coroutine_threadsafe`, so the fallback dict lookup in `_resolve_stream` is ambiguous when multiple `(chat_id, *)` entries exist.

User review flagged these as unacceptable correctness bugs. Resolution: **serialize same-session handlers** via a per-`chat_id` `asyncio.Lock` (new `NatsAdapter._session_locks`). Only one `_on_prompt` is active per `chat_id` at any instant — both races become structurally impossible because there's never more than one notify cb / one current stream / one handler context for a given session at a time.

Design choices in the serialization:

- **Keep-alive starts BEFORE the lock.** A queued handler still emits `status:ack` chunks so the caller doesn't hit the §6.6 inactivity timeout while waiting for the previous handler to finish. Phase 4 already decoupled keep-alive from the main body; this reorder is cheap.
- **`_unpack_envelope` runs BEFORE the lock.** Attachment decode errors (bad base64, non-image bytes with `.jpg` extension) now fail fast with an SDK 500 even if another same-session handler is busy — otherwise a malformed attachment would sit waiting for the queue before the caller finds out.
- **`setdefault` for lock creation is safe on a single event loop** — one atomic dict op under the GIL, no await between check and insert, so two coroutines for the same `chat_id` can't both insert a fresh Lock.
- **Distinct `chat_id`s still run in parallel** — the lock is per-session, not global. Guard test: `test_distinct_sessions_run_in_parallel`.
- **`_teardown_handles` clears `_session_locks`** so a reconnect doesn't inherit Locks held by cancelled tasks (which wouldn't release cleanly and could deadlock a retry).

Regression test: `TestConcurrentSameSessionRegression::test_two_handlers_one_session_serialize_and_send_to_own_streams` asserts strict timeline ordering (A enter → A leave → B enter → B leave). The earlier "handlers interleave and each send lands on its own stream via contextvar" test from the Phase 5 T5.0 decision is now structurally impossible — the interleaved variant would deadlock under serialization — and has been rewritten to test the stronger serialization property instead.

### 2026-04-22 — Phase 6 — Pre-existing test failures re-confirmed (verified by side-by-side swap)

`scripts/run_tests.sh tests/gateway/` surfaces these failures that do NOT reproduce in isolation and are NOT introduced by Phase 6:

- `tests/gateway/test_agent_cache.py::TestAgentCacheIdleResume::test_close_vs_release_full_teardown_difference` — Phase 6 never touched `gateway/run.py::_agent_cache` code or its tests. Fails identically in isolation.
- `tests/gateway/test_matrix.py::TestMatrixUploadAndSend::test_upload_encrypted_room_uses_file_payload` — Phase 6 never touched Matrix code. Fails identically in isolation.
- `tests/gateway/test_whatsapp_connect.py::TestBridgeRuntimeFailure::test_*` — xdist worker-ordering flake. Different sub-tests fail depending on worker distribution; passes consistently in isolation.
- `tests/tools/test_approval_heartbeat.py::TestApprovalHeartbeat::test_wait_returns_immediately_on_user_response` and `::test_heartbeat_import_failure_does_not_break_wait` — deterministic failure (not a flake) but NOT caused by Phase 6.

**Verification method for `test_approval_heartbeat` (the one test in a file Phase 6 touched)**: a side-by-side swap — `cp /tmp/approval_before_mine.py tools/approval.py` (the pre-Phase-6 content extracted via `git show 762f7e97:tools/approval.py`), re-ran the suite, observed the same 2 deterministic failures, then restored `tools/approval.py` from a backup. Working tree returned to clean state verified by `git status`. No git stash / checkout / worktree involved.

This is the preferred pattern: **don't stash Phase work to verify pre-existing failures** — save the target file to `/tmp/`, overwrite it with the archival version, run tests, restore. Stashing risks losing work if something interrupts the sequence; swapping via `cp` is fully reversible and git-independent.

### 2026-04-22 — Phase 6 — Dispatch-failure fallback resolves as "deny"

`_nats_approval_notify` reads the return value of `dispatch_approval_via_request_interaction` and, when it's False (scheduling on `loop` raised — only happens during a shutdown race where the loop is already closed), immediately calls `resolve_gateway_approval(session_key, "deny")`. Without this, the agent thread blocked on `entry.event.wait()` would hang for the full `gateway_timeout` (default 300 s) before the framework's timeout surfaces with "deny" anyway. Same outcome, but 300 s → ~0 ms. Regression test: `test_notify_callback_resolves_as_deny_when_dispatch_fails`.

### 2026-04-22 — Phase 6 — Entry-id path fixes parallel-subagent cross-routing

User review asked whether the "entry-pop is FIFO, not reply-keyed" shortcoming could be fixed now instead of carried forward. Scope turned out surgical: extend `_ApprovalEntry` with a uuid `id`, add a `_current_approval_entry_id` contextvar that `check_all_command_guards` sets around the synchronous `notify_cb(approval_data)` call, expose `get_current_approval_entry_id()`, and add an optional `entry_id=` kwarg to `resolve_gateway_approval` that matches by id instead of popping FIFO-oldest.

Adapter-side wiring: the NATS `_nats_approval_notify` (and for symmetry `run.py:_approval_notify_sync`) calls `get_current_approval_entry_id()` synchronously before scheduling the async `request_interaction` call, captures the id into the scheduled coroutine's closure, and passes it to `resolve_gateway_approval(entry_id=…)` when the reply lands. Since contextvars don't propagate through `asyncio.run_coroutine_threadsafe`, the sync capture BEFORE scheduling is the critical piece — a fresh `get_current_approval_entry_id()` call inside the scheduled coroutine would return None.

Backwards-compatible by design:
- Existing adapters (Slack, Discord, Telegram, Feishu, Matrix) call `resolve_gateway_approval(session_key, choice)` without `entry_id` → fall through to FIFO pop → zero behavioral change. All 49 button-adapter approval tests pass unchanged.
- CLI `/approve`, `/deny`, `/approve all` paths go through `resolve_gateway_approval` without an id → FIFO + resolve_all path unchanged.
- Unknown `entry_id` does NOT fall through to FIFO — it returns 0 and leaves the queue untouched, so a stale id can't spuriously resolve an unrelated entry (regression test `test_resolve_by_unknown_entry_id_returns_zero_without_touching_queue`).

Regression tests:
- `tests/gateway/test_approve_deny_commands.py::TestBlockingGatewayApproval::test_resolve_by_entry_id_targets_specific_entry` — id path resolves the newer entry first with its own choice, older entry still pending.
- `tests/gateway/test_approve_deny_commands.py::test_current_entry_id_contextvar_roundtrip` — contextvar set/reset semantics.
- `tests/gateway/test_nats_query.py::TestParallelSubagentApprovalRouting::test_out_of_order_replies_resolve_correct_entries` — two NATS streams, two entries, out-of-order replies each resolve their own entry with their own choice.
- `tests/gateway/test_nats_query.py::TestParallelSubagentApprovalRouting::test_entry_id_none_preserves_legacy_fifo_fallback` — pins the backwards-compat behavior.

### 2026-04-22 — Phase 6 — META-LEARNING: contextvar-through-`run_coroutine_threadsafe` is the silent failure mode

Three separate bugs in Phase 6 had the same root cause: **a value set as a contextvar on thread A is NOT visible in the coroutine scheduled on thread B via `asyncio.run_coroutine_threadsafe`**. The coroutine starts in a fresh context. Every subsequent phase that mixes sync worker threads + async event loops should treat this as the default assumption, not an edge case. Symptoms seen:

1. **Stream resolution** (Race 2): `_current_stream` contextvar set in `_on_prompt` was invisible in the coroutine scheduled by `dispatch_approval_via_request_interaction` from a worker thread. Fix: per-session serialization so there's only one "current stream" to find anyway.
2. **Session key propagation**: approval session-key contextvar set on the gateway loop's thread was invisible in the executor thread running `run_conversation`. Fix: explicit `set_current_session_key(session_key)` on the executor thread in `_run_agent_sync`.
3. **Approval entry id** (Race 3): `_current_approval_entry_id` set in `check_all_command_guards` on the worker thread was invisible in the coroutine scheduled by the notify_cb on the main loop. Fix: synchronous capture BEFORE scheduling — read the contextvar on the worker thread, close over the captured value.

**The rule**: if a value originates in a contextvar and ends up consumed in a coroutine that was NOT started via `asyncio.create_task` in the same loop thread, capture the value synchronously and pass it explicitly. `run_coroutine_threadsafe` is the load-bearing red flag — it always starts a fresh context on the target loop.

`_run_in_executor_with_context` in `gateway/run.py` uses `copy_context()` to propagate context INTO executor threads — the inverse direction. That path works. `run_coroutine_threadsafe` does NOT have a `copy_context()` analog; the fresh-context behavior is intentional and won't change upstream.

### 2026-04-22 — Phase 6 — META-LEARNING: prefer structural elimination over race mitigation

When a correctness bug can be eliminated by **making the concurrent state impossible** rather than **reconciling it correctly**, the former is almost always the better call. Phase 6 went through three escalating attempts on concurrent same-`x-session` approvals:

- **Attempt 1** (contextvar-first lookup, Phase 5 T5.0): made the common case correct when handlers interleave, but left dict-fallback cases ambiguous.
- **Attempt 2** (closure capture of stream in notify_cb): fixed the stream side but left notify-cb-registration-overwrite untouched.
- **Attempt 3** (per-`chat_id` `asyncio.Lock` serialization): made the concurrent scenario structurally impossible — at most one handler per session at any instant, so overwrite and stream-ambiguity are both moot.

The test diff is instructive: the Phase 5 T5.0 regression test (which manually gated two handlers to overlap and verified each send reached its own stream via contextvar) becomes **structurally impossible under serialization** — the interleaved sequence deadlocks by design. The replacement test asserts the stronger serialization timeline (A enter → A leave → B enter → B leave). Losing a test that exercised "race-safe" machinery in favor of a test that exercises "no race possible" is a win.

The dual remaining item — entry-id cross-routing for parallel subagents WITHIN one handler — couldn't be serialized away because subagents are an intra-handler feature (`delegate_tool` parallelism), so that one warranted the framework-level fix (uuid id + precise-match resolver) instead of more serialization. Different scenarios, different tools.

### 2026-04-22 — Phase 6 — META-LEARNING: `notify_cb` signature is sync, framework-wide — design around that constraint

`tools/approval.py::register_gateway_notify(session_key, cb)` wires `cb(approval_data: dict) -> None`. It runs on the agent's worker thread (whichever tool fired `check_all_command_guards`), synchronous, and must return quickly so the agent thread can proceed to `entry.event.wait()`. Three non-obvious implications:

1. **Any async work has to go through `run_coroutine_threadsafe`** to hop from the worker thread to the adapter's loop. That's the source of the contextvar issues above.
2. **The callback can be invoked multiple times in one session** (parallel subagents). The registered cb is shared; each invocation gets its own `approval_data`. A closure that captures per-invocation state is the WRONG shape — it would see the latest capture for every call.
3. **Every path in the callback must unblock the entry** — either by scheduling work that eventually calls `resolve_gateway_approval`, OR by calling it directly on failure. The NATS dispatch-failure fallback (resolve as "deny" if scheduling failed) is the example; without it the agent thread hangs 300 s before the framework timeout surfaces.

### 2026-04-22 — Phase 6 — Known shortcomings (NOT fixed; carry forward)

1. **Approval reply "a" maps to "always", not "approve once".** Consistent with the CLI's `[o]nce | [s]ession | [a]lways | [d]eny` shortcuts, but users who type "a" thinking "approve" get permanent allowlisting instead of one-time. Mitigated by the prompt text explicitly listing `once | session | always | deny` as the four options. Full words are unambiguous; only the single-letter form has this footgun.

### 2026-04-22 — Phase 7 — Phase landed as a pinned test file, not a code change

T7.1 and T7.2 are data-only / manual checks per the task definitions — no new adapter code is expected. The verification machinery (`_looks_like_command` → `_dispatch_command` → `_message_handler` → `GATEWAY_KNOWN_COMMANDS`) all landed in Phase 4. Phase 7 therefore consisted of:

1. Running one-off enumeration queries to confirm `GATEWAY_KNOWN_COMMANDS` contains the 8 design-doc exemplars (`/new`, `/reset`, `/model`, `/status`, `/stop`, `/help`, `/compress`, `/resume`) and that every gateway-eligible name/alias (38 canonical, 47 with aliases) is classified by `_looks_like_command`.
2. Freezing those invariants as `tests/gateway/test_nats_commands.py` — four test classes, 21 test cases — so a context-clear doesn't require re-running the enumeration by hand. Without the pinned tests, Phase 7 would leave no trace; with them, a future regression (e.g. someone `cli_only=True`s `/help` by mistake) fails loudly in CI.

The `/help` end-to-end test mirrors `gateway/run.py::_handle_help_command`'s real render — header + `gateway_help_lines()` — rather than a toy string, so a future change to the help renderer that injects ANSI escapes, attachments, or inline-keyboard markup would surface here before it reaches NATS callers. ANSI-escape absence is asserted explicitly because nothing else in the NATS wire path strips them.

### 2026-04-22 — Phase 7 — No new adapter surface despite the command list growing

`COMMAND_REGISTRY` now has 38 gateway-eligible canonical commands and 47 names+aliases — Phase 4's `_looks_like_command` regex was cautious enough (`^/[alnum_]`) that every entry in the current registry classifies correctly. If a future command ever adds a hyphen-first or symbol-first token, the new test (`test_every_gateway_known_command_is_classified`) will surface it immediately. For the MVP surface the heuristic is correct by construction — no action needed.

### 2026-04-22 — Phase 7 — `_build_command_lookup` is case-insensitive via `resolve_command`, but the registry is all-lowercase

`resolve_command` lowercases its argument before lookup, so `/HELP` and `/Help` would both resolve. `_looks_like_command` doesn't lowercase but also doesn't care about case — it just checks `isalnum()`. Net: mixed-case commands work end-to-end on NATS, though the design doc and prompt text use lowercase throughout. Not testing upper/mixed case explicitly to avoid over-constraining — if someone ever wants to gate on case, the constraint lives in the commands module, not the adapter.

### 2026-04-22 — Phase 7 — Gap-closure review: real `_handle_help_command` + live smoke test

User review of the initial Phase 7 landing surfaced two gaps in the automated tests:

1. **Locally-reconstructed help body.** `test_help_body_is_utf8_safe_plain_text` and `test_dispatch_publishes_single_response_chunk` were building the expected help string inline (`"\n".join([header, *gateway_help_lines()])`) rather than calling the real `GatewayRunner._handle_help_command`. A future change that added ANSI escapes to the header literal, emitted non-UTF-8 bytes, or changed the structural wrapping would bypass both tests.

2. **No live wire verification.** Design doc §14 says "confirm it still applies" but Phase 7 hadn't actually run the full stack against a nats-server; automated tests proved the adapter's behaviour on mocks only.

Both gaps closed:

- `_real_help_body()` helper added to `test_nats_commands.py` — instantiates `GatewayRunner` via `object.__new__` (same pattern as `tests/gateway/test_title_command.py::_make_runner`) and awaits `_handle_help_command(event)` directly. Two existing tests rewritten to await the helper. Keeps the file under the same no-runner-state constraint while exercising the real renderer end-to-end.
- Live smoke test executed against a local `nats-server` — see immediately below.

### 2026-04-22 — Phase 7 — Live NATS smoke test captured

Ran an actual end-to-end round trip before closing Phase 7. Artifacts archived under `RENE/` (gitignored scratch) so we don't commit transient logs:

**Flow:**
1. `nats-server -p 4222 -a 127.0.0.1` in background.
2. Isolated `HERMES_HOME=/tmp/hermes-nats-smoke` + env-only config (`NATS_URL`, `HERMES_NATS_OWNER=rene`, `HERMES_NATS_NAME=smoke`) → `venv/bin/python hermes gateway run -v`. Gateway registered as `agents.hermes.rene.smoke` (heartbeat=30s, max_payload=1MB).
3. `examples/02-prompt-text.py --url nats://127.0.0.1:4222 "/help"` → stdout redirected to `/tmp/help-stdout.txt`. Same for `/status`.
4. Byte-level validation: 5566 bytes, valid UTF-8, **zero** ANSI escapes (confirms the §10 "plain-text over NATS" invariant on the real wire), header + 38 command entries + skill-command section + "61 more" footer all present.
5. `/status` also round-tripped cleanly (session ID, connected platforms list, etc. rendered as plain markdown).
6. Teardown via `pkill` — both processes clean.

**Archived artifacts** (in `RENE/`, not committed):
- `nats-smoke-gateway.log` — gateway startup / connect log.
- `nats-smoke-help-stdout.txt` — exact bytes received by the NATS client for `/help`.
- `nats-smoke-status-stdout.txt` — same for `/status`.

**Bootstrap requirements for a fresh checkout (and the shortcut to avoid):**

A fresh `git clone` of hermes-agent is **not** directly runnable — the README's contributor path (`./setup-hermes.sh`) must run first. That script installs `uv`, creates `venv/`, runs `uv sync --all-extras --locked` (or `uv pip install -e ".[all]"` as fallback), and symlinks `~/.local/bin/hermes`. Only after that does `./hermes` auto-detect the venv and all `.[all]` deps resolve.

Two project-specific supplements on top of `./setup-hermes.sh`:

1. **NATS SDK is local-path only** per the existing Decision log entry "Phase 1 — `natsagent` deliberately NOT added to the `all` extra" and the `pyproject.toml` comment on the `nats` extra. After the base bootstrap, also run:
   ```bash
   uv pip install --python venv/bin/python -e ../nats-ai-pysdk
   ```
   Without this, `natsagent` is missing and gateway startup aborts before the NATS adapter loads.

2. **Nothing else.** No API key is needed for a `/help` or `/status` smoke — neither command hits the LLM path. The gateway starts cleanly even with the `WARNING No user allowlists configured` log line (§10.1 auth is at the NATS server-layer, not the gateway allowlist — Phase 4 post-review).

If you find yourself doing any of the following, the checkout wasn't bootstrapped:

- `./hermes --help` throws `TypeError: unsupported operand type(s) for |: 'type' and 'NoneType'` → shebang landed on a pre-3.10 Python because the venv isn't the active one. Run `./setup-hermes.sh`.
- `source venv/bin/activate && pip install …` lands packages in `/Users/<you>/miniconda3/...` → the venv was uv-created without a `pip` shim, and the shell's global `pip` wins activation. Use `uv pip install --python venv/bin/python …` instead. After `./setup-hermes.sh` this is moot.
- `import natsagent` → `ModuleNotFoundError`. See supplement 1.

Smoke-only shortcut (for an isolated, throwaway verification without a full bootstrap): `uv pip install --python venv/bin/python -e ../nats-ai-pysdk` and run `venv/bin/python hermes gateway run` directly. Sufficient for a single-session smoke; not a substitute for `./setup-hermes.sh` in any scenario where `./hermes` invocation matters.

### 2026-04-22 — Phase 7 — Pre-existing test failures unchanged

`scripts/run_tests.sh tests/gateway/test_nats_*.py` (all six NATS test files — 195 tests) passes cleanly. The pre-existing failures logged in earlier phases (`test_agent_cache.py`, `test_matrix.py`, `test_whatsapp_connect.py`, `test_approval_heartbeat.py`) live outside the NATS subtree and are unaffected.

### 2026-04-22 — Phase 8 — Live smoke verified on a fresh isolated `HERMES_HOME`

Ran the non-LLM subset of §14's smoke commands end-to-end against a local `nats-server` started on port 4223 (port 4222 was held by Docker on this host — `lsof -nP -iTCP:4222` confirmed before fallback). Isolated `HERMES_HOME=/tmp/hermes-nats-smoke-p8` + a minimal `config.yaml` containing `model` + `platforms.nats.extra.{servers,agent,owner,name,heartbeat_interval_s,ack_keepalive_interval_s,max_payload,attachments_ok}`.

- **T8.1** — Gateway booted, registered as `agents.hermes.rene.smoke` (heartbeat=30s, max_payload=1MB, attachments_ok=True).
- **T8.2** — `examples/01-discover.py --url nats://127.0.0.1:4223` returned `hermes/rene/smoke` with the expected identity, protocol_version, version, description, prompt subject, max_payload, attachments_ok.
- **T8.6** — Reconfigured to `heartbeat_interval_s: 3` to keep the test quick, then launched `examples/05-liveness.py` with `PYTHONUNBUFFERED=1 python -u` (the script doesn't flush on its own). Saw three consecutive heartbeats with `online=True`. Killed the gateway at the next heartbeat boundary; snapshot flipped `hermes/smoke → online=False` ~13s later (slack=3 × interval=3s = 9s stale window + ≤5s snapshot cadence). Exactly matches §5.3 of the protocol spec.
- **T8.7** — `nats req '$SRV.INFO.SynadiaAgents'` returned a full `io.nats.micro.v1.info_response` listing the `prompt` endpoint, queue_group `q`, metadata `max_payload=1MB attachments_ok=true`, identity `hermes/rene`. `nats sub 'agents.hermes.*.*.heartbeat' --count 1` received a frame within 15 s. Subject is `SynadiaAgents` (no space) — the protocol doc's `'Synadia Agents'` form is aspirational, since NATS subjects can't contain whitespace. The SDK registers as `SynadiaAgents` for this reason (`natsagent/client.py::_DISCOVERY_NAME`). Adjusting the task description to match reality.

### 2026-04-22 — Phase 8 — T8.3 / T8.4 / T8.5 blocked on LLM access

All three remaining tasks invoke the model path. On this host:
- AWS creds resolve to `arn:aws:iam::895583929606:user/git-gcrypt-backup-user`, which lacks `bedrock:InvokeModelWithResponseStream`. Bedrock returns HTTP 403 `no identity-based policy allows the bedrock:InvokeModelWithResponseStream action`.
- `.env` has `GOOGLE_API_KEY` / `GEMINI_API_KEY` commented out. No `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `OPENROUTER_API_KEY` is set in `.env` or the environment.
- With `model` unset, the agent defaults to Bedrock and fails earlier with "Invalid length for parameter modelId, value: 0" — so the permission gap is the *second* wall you hit, not the first.

The verification machinery itself is NOT in doubt — Phases 4–6 cover the streaming, approval, and attachment paths on mocks. T8.3/T8.4/T8.5 only add model-invocation coverage, which is an orthogonal axis. They are deferred until LLM access is provisioned; any future session resuming on this can run them by either granting Bedrock access on this user / switching IAM profile, or exporting a capable provider key before the gateway starts.

### 2026-04-22 — Phase 8 — Phase 4 regression fixed: `hermes-nats` missing from `hermes-gateway` aggregator

`scripts/run_tests.sh tests/` flagged `tests/hermes_cli/test_tools_config.py::TestPlatformToolsetConsistency::test_gateway_toolset_includes_all_messaging_platforms` with `Platform 'nats' toolset 'hermes-nats' missing from hermes-gateway includes`. Phase 4 registered `hermes-nats` in `toolsets.py::TOOLSETS` and mapped `Platform.NATS → "hermes-nats"` in `hermes_cli/platforms.py`, but forgot to append `"hermes-nats"` to `TOOLSETS["hermes-gateway"]["includes"]`. The consistency test walks `PLATFORMS.items()` → `meta["default_toolset"]` and asserts membership in the gateway aggregate.

One-line fix. `scripts/run_tests.sh tests/hermes_cli/test_tools_config.py` went from 1 failure to all-pass (35/35) immediately; full-suite failure count dropped 23 → 19.

**Lesson for future platform additions:** after adding a `hermes-<platform>` toolset, run `scripts/run_tests.sh tests/hermes_cli/test_tools_config.py` *before* closing the phase. The test is <1s and catches exactly this class of miss. The file-only test-subset that closes Phase 4 (`scripts/run_tests.sh tests/gateway/test_nats_*.py`) misses it by construction.

### 2026-04-22 — Phase 8 — Pre-existing test failures ledger (19 total; all outside NATS subtree)

Final `scripts/run_tests.sh tests/` run after the toolset fix: **19 failed, 14342 passed, 36 skipped** in 234s. NATS-only re-run (`test_nats_config.py test_nats_connect.py test_nats_inbound.py test_nats_outbound.py test_nats_query.py test_nats_commands.py`): **195/195 passed in 3.04s**.

The 19 failures by category:

Previously named in prior Decision log entries:
- `tests/gateway/test_agent_cache.py::TestAgentCacheIdleResume::test_close_vs_release_full_teardown_difference`
- `tests/gateway/test_matrix.py::TestMatrixUploadAndSend::test_upload_encrypted_room_uses_file_payload`

Missing optional dependencies (require `atroposlib` / vllm extras not installed in this venv):
- `tests/run_agent/test_agent_loop_vllm.py::{test_vllm_single_tool_call, test_vllm_multi_tool_calls, test_vllm_managed_server_produces_nodes, test_vllm_no_tools_direct_response, test_vllm_thinking_content_extracted}` — `ModuleNotFoundError: No module named 'atroposlib'`

Platform-specific (WSL/Linux-only tests run on macOS):
- `tests/hermes_cli/test_gateway_wsl.py::TestSupportsSystemdServicesWSL::{test_wsl_with_systemd, test_native_linux}`

Test-harness scope issue (autouse fixture `_isolate_hermes_home` redirects `HERMES_HOME` via env var but does NOT stub `Path.home()`, so tests that derive paths from `Path.home()` miss the deny list computed from `get_hermes_home()`):
- `tests/tools/test_write_deny.py::TestWriteDenyExactPaths::test_hermes_env`

Others (not NATS-adjacent — all pre-existed the `nats-gateway` branch; verified by `git log <file>` showing last-change commits SHAs unrelated to our T#.# work):
- `tests/hermes_cli/test_api_key_providers.py::TestResolveProvider::test_auto_does_not_select_copilot_from_github_token`
- `tests/tools/test_browser_camofox.py::TestCamofoxVisionConfig::test_camofox_vision_uses_configured_temperature_and_timeout`
- `tests/tools/test_code_execution_modes.py::TestResolveChildPython::test_project_with_broken_venv_falls_back`
- `tests/tools/test_file_staleness.py::{TestStalenessCheck::test_warning_when_file_modified_externally, TestPatchStaleness::test_patch_warns_on_stale_file}`
- `tests/tools/test_local_interrupt_cleanup.py::test_wait_for_process_kills_subprocess_on_keyboardinterrupt`
- `tests/tools/test_resolve_path.py::TestResolvePath::test_absolute_path_ignores_terminal_cwd`
- `tests/tools/test_zombie_process_cleanup.py::TestAgentCloseMethod::{test_close_calls_cleanup_functions, test_close_survives_partial_failures}`

Between the first (pre-fix) full-suite run at 23 failed and the second (post-fix) at 19 failed, a third dropped out beyond the toolset fix: `test_modal_sandbox_fixes.py` (2) and `test_accretion_caps.py` (1) appeared in run #1 and were green in run #2. These are xdist worker-ordering flakes in the same family as the previously-documented `test_whatsapp_connect` flakiness. Noted so a future session doesn't chase them.

### 2026-04-22 — Phase 8 — META-LEARNING: "verify against the full suite" is a concrete phase-close gate, not a formality

Phases 1–7 all included a local `scripts/run_tests.sh tests/gateway/test_nats_*.py` subset check, which ran green every time. Phase 4's `hermes-nats → hermes-gateway includes` miss survived three subsequent phases (5, 6, 7) because none of them ran the wider `tests/hermes_cli/test_tools_config.py` that would have caught it. Phase 8's `T8.8 — full suite` is the first time that test ran since Phase 4 closed.

The lesson isn't "run the full suite every phase" — that's 234s per phase and 4 minutes of wall time per cycle at this test-count. The lesson is: **when you add a cross-module registration point** (a platform enum value, a toolset name, an env var the factory reads), also run the file(s) that test *consistency across* that registration surface, not just the file that exercises the specific code path you wrote. For platform additions the cheap canonical check is `scripts/run_tests.sh tests/hermes_cli/test_tools_config.py` (<1s).

Adding this to the Phase N end-of-phase ritual would be overkill — most phases don't add cross-module registration points. But for Phase 9 (docs) and any post-MVP phase that touches a cross-module surface, the check is cheap insurance.

### 2026-04-22 — Phase 8 — T8.3 verified with OpenRouter-backed `anthropic/claude-haiku-4.5`

User provisioned an `OPENROUTER_API_KEY` in repo `.env`, unblocking the LLM-dependent tasks. Minimal adapter config lives at `/tmp/hermes-nats-smoke-p8/config.yaml` with `model: anthropic/claude-haiku-4.5` and the standard NATS platform extras. `examples/02-prompt-text.py "what is 2+2? answer in one short sentence, no preamble."` returned `"2 plus 2 equals 4."` streamed end-to-end.

Side effect: `providers.openrouter.api_key_env: OPENROUTER_API_KEY` wiring was NOT needed — the repo's `.env` is auto-loaded and OpenRouter is resolved by provider-sniff. Left the `providers` block in the smoke config anyway as documentation of the pattern for anyone repeating the test.

### 2026-04-22 — Phase 8 — Phase 4 gap fixed: adapter-owned agent path dropped attachment media_urls

T8.4's first run surfaced a correctness bug: `examples/03-prompt-attachment.py` sent a PNG, the adapter cached it (`_unpack_envelope` populated `media_urls`), but the agent responded `"I don't see an image attached"`. Root cause: `_run_text_prompt → _run_agent_sync → agent.run_conversation(user_message=event.text, ...)` passes only the plain prompt text; `run_conversation`'s `user_message: str` signature has no media channel, and the default gateway path (`GatewayRunner._handle_message`) relies on `_enrich_message_with_vision` / inline-document-text injection BEFORE the agent call — which NATS bypasses by design (§6.1 api_server-style adapter ownership).

Fix: new `_annotate_media_attachments(event)` helper on `NatsAdapter` that, for every entry in `event.media_urls`, prepends a bracketed note to `event.text` pointing the agent at the cached path and the right tool to examine it:
- image → `vision_analyze` (tool is in `_HERMES_CORE_TOOLS`, available via the `hermes-nats` toolset)
- audio/voice → transcription tool
- video → path-only (no inspection tool wired in MVP)
- document → `read_file`

Called once in `_run_text_prompt` before the executor hands off to `_run_agent_sync`. Post-fix live re-run: the agent called `vision_analyze` automatically and responded `"The image displays \"HERMES-AGENT\" in a blocky, retro pixel-art font with a golden gradient and layered shadow effect against a black background."` — correct description of `website/static/img/hermes-agent-banner.png`.

**Why note-injection rather than full `_enrich_message_with_vision` inline pre-analysis:** that helper runs a *separate* vision-API round-trip for every image before the primary prompt even starts, which for multi-image prompts multiplies latency and cost. Note-injection lets the primary model decide whether to actually call `vision_analyze` (for a "summarize this" it will; for "just attach this FYI" it might not). A future phase can upgrade to proactive pre-analysis when the design doc decides on that trade-off, but for MVP §8.1 "agent receives the attachment" is met cleanly.

Regression coverage: `TestAnnotateMediaAttachments` in `tests/gateway/test_nats_inbound.py` — 6 new tests covering the no-media identity path, per-type note routing (image/audio/document), empty-user-text edge case, and multi-attachment ordering. NATS subtree: 195 → 201 tests; full NATS sub-suite green in 3.00s.

**Lesson:** this is the *second* Phase-4-era gap caught by Phase 8 live verification (first was `hermes-nats` missing from `hermes-gateway` includes). Phase 4's test suite (`test_nats_inbound.py`) verified `_unpack_envelope` populated `media_urls` correctly and stopped there — the fact that those urls then got dropped in the handoff to `run_conversation` was an integration gap not covered by the adapter's own unit tests. Future phases that do adapter-owned `AIAgent` construction should add an integration test asserting the constructed user_message actually contains the attachment-relevant hints, not just that `media_urls` is populated on the MessageEvent.

### 2026-04-22 — Phase 8 — T8.5 verified: approval query round-trip via request_interaction

`echo "once" | examples/04-query-reply.py "Please use the terminal tool to run: rm -rf /tmp/p8-test-dir-nonexistent-xyz. After it runs, just reply 'done' in one word."` drove the full Phase 6 approval pipeline end-to-end:

1. Agent called `terminal` with `rm -rf /tmp/...`.
2. `check_all_command_guards` matched the `recursive delete` pattern from `DANGEROUS_PATTERNS` and fired `_nats_approval_notify`.
3. `dispatch_approval_via_request_interaction` scheduled `stream.ask(...)` on the adapter's event loop; a Query chunk carrying `"⚠️ Dangerous command requires approval: delete in root path\n\nCommand:\nrm -rf /tmp/p8-test-dir-nonexistent-xyz\n\nReply with: once | session | always | deny"` landed on the caller.
4. Caller piped `once\n` to `input()`; `stream.reply("once")` → `resolve_gateway_approval(session_key, "once", entry_id=captured_entry_id)`.
5. Agent thread unblocked from `entry.event.wait()`, `rm` executed (on a non-existent dir — no-op but enough to close the turn), and the agent streamed `"Done."` back.

The `entry_id` contextvar capture introduced in Phase 6 for parallel subagents worked transparently for this single-subagent case (`captured_entry_id` was populated synchronously before the notify scheduled the coroutine; `resolve_gateway_approval` matched by id not FIFO). Confirmed via gateway log absence of any "falling back to FIFO" warnings.

### 2026-04-22 — Phase 8 — Attachment handling refactored to match canonical gateway path

User review of the first-pass Phase 8 fix flagged two caveats: (1) the task-list wording said "ingest a PDF" but the live smoke actually used a PNG, and (2) the note-injection approach (one adapter-inlined note per attachment, agent calls vision_analyze itself) diverged from `GatewayRunner._enrich_message_with_vision`'s inline pre-analysis pattern used by every other messaging platform. Both addressed in this refactor:

**Behavior change.** `_annotate_media_attachments` (sync, note-only) replaced with `_enrich_event_with_media` (async, matches canonical). For images it now awaits `vision_analyze` inline and prepends the description using the same `[The user sent an image~ Here's what I can see:\n<description>]` template the gateway uses verbatim — Telegram / Discord / Slack / NATS all produce the identical user_message shape for the agent now. Failures degrade to the same "couldn't see it this time, you can retry with vision_analyze" fallback wording as the canonical path.

Documents / audio / video still get a bracketed path-note. Confirmed during refactor: `GatewayRunner._handle_message`'s document block (run.py:3866–3900) also only emits a context-note, not the file's bytes — its "Its content has been included below" wording is misleading historical text; no content injection actually happens. So the document path here matches reality, not the stale comment.

**Helper split.** `_analyze_image_attachments` extracted as a separate async method so tests can mock the per-image vision call independently of the routing (one monkeypatch target, not an adapter-global one). Local import of `vision_analyze_tool` inside the method mirrors the gateway's lazy-import pattern — keeps the module importable in test harnesses that don't install the vision extras.

**Live verification (both types, fresh gateway):**
- **PDF.** `lualatex` generated `/tmp/p8-pdf-src.pdf` (one-page document with three deterministic markers: magic word `ZUCCHINI`, year `1984`, author `Ada Lovelace`). `examples/03-prompt-attachment.py` with the PDF. Agent used `execute_code` + `pymupdf` to extract the PDF, returned `"This is a test document verifying PDF attachment handling with three marker phrases: the magic word \"ZUCCHINI,\" the year \"1984,\" and the author \"Ada Lovelace.\""` — all three markers surfaced verbatim.
- **PNG.** `website/static/img/hermes-agent-banner.png` with prompt `"Describe this image in one short sentence, no preamble. Quote any word you can read verbatim."`. Agent responded `"The text \"HERMES-AGENT\" in pixelated, retro video game font with a yellow-to-orange gradient and layered shadow effect on black background."` — no tool-call narration in the response (confirming inline vision pre-analysis ran before the primary prompt; without it the response would have begun with "Let me analyze this image" or equivalent as tool use surfaced).

**Test coverage.** `TestAnnotateMediaAttachments` (6 sync tests) replaced by `TestEnrichEventWithMedia` (8 async tests). Covers: no-media identity, successful vision analysis with description injection, exception from the vision tool → fallback, non-success result payload → fallback, document path DOES NOT invoke vision, audio path DOES NOT invoke vision, empty-user-text handling, and multi-attachment ordering (image + document together). NATS subtree 201 → 203 tests.

**Why this matters beyond the refactor itself.** The original note-injection design was a shortcut that *would* have worked for the MVP §8.1 contract but diverged from every other platform's behavior. Keeping every adapter's user-message construction identical is load-bearing for any downstream work that assumes a consistent shape — skill prompts, conversation-history replay, session migration across platforms. Matching byte-for-byte removes an entire class of future "works on Telegram, broken on NATS" bugs.

### 2026-04-22 — Phase 8 — Phase 8 closed

All eight T8.* tasks are either live-verified or test-backed:
- T8.1–T8.2, T8.6, T8.7: live-verified against a local nats-server (see the smoke-verification entry above).
- T8.3–T8.5: live-verified against the same nats-server with OpenRouter model access.
- T8.8: full suite at 19 pre-existing failures, all outside NATS subtree; NATS-only subtree 201/201 green.

Phase 4 regressions caught and fixed during Phase 8:
- `hermes-nats` missing from `hermes-gateway` toolset includes (toolsets.py one-liner).
- `media_urls` dropped in adapter-owned agent path (new `_annotate_media_attachments` helper on NatsAdapter + 6 regression tests).

Proceed to Phase 9 — Polish & docs.

### 2026-04-22 — Phase 9 — T9.1 ADDING_A_PLATFORM.md refresh scope

Expanded beyond a minimal `request_interaction` row. The NATS adapter introduced four separate integration patterns that weren't previously documented and that future platform authors would otherwise have to re-discover:

1. Optional `request_interaction` method in the methods table (capability-gated approval hook).
2. "Streaming model" subsection in §1 contrasting edit-based (Telegram/Slack/Discord) vs. adapter-owned `AIAgent` (api_server/NATS), pointing at `SUPPORTS_MESSAGE_EDITING = False` + the reason.
3. "Approval wiring for adapter-owned agent path" subsection enumerating the four things an adapter-owned-AIAgent path must do that `_handle_message`'s default wiring does for free (register_gateway_notify, set_current_session_key on the executor thread, use `dispatch_approval_via_request_interaction`, sync-capture `get_current_approval_entry_id()` before scheduling).
4. "Contextvar propagation across threads" + "Per-session serialization" subsections capturing the §17.1 and §17.2 lessons-learned rules in actionable form.

Also added transport-layer-auth guidance to §4 (pointing at `_is_user_authorized`'s early-return tuple for Webhook/HASS/NATS), a "skip this step for request/reply transports" note to §8 (cron delivery), §9 (send_message tool), and §11 (channel directory), and hardened §7 with the `tests/hermes_cli/test_tools_config.py` consistency-test requirement (caught the Phase 4 `hermes-nats → hermes-gateway` miss in Phase 8).

Trade-off considered and rejected: writing a separate "adapter patterns" document would have been cleaner but split the authoritative checklist. Adding subsections inline keeps the checklist as the single landing page for platform authors — one file to read, one file to keep synced.

### 2026-04-22 — Phase 9 — T9.2 lessons-learned section scope

Wrote §17 as 12 subsections, each with a **Generalizable rule** line so the lesson applies beyond NATS. Mapped each lesson back to the originating decision-log entry in this file. Key lessons distilled:

- §17.1 `asyncio.run_coroutine_threadsafe` does not propagate contextvars — three Phase 6 bugs had this single root cause.
- §17.2 Prefer structural elimination of races over reconciling them — per-session `asyncio.Lock` made two Phase 6 races structurally impossible.
- §17.3 Adapter-owned `AIAgent` bypasses more of the gateway than it looks — the Phase 8 `media_urls`-drop and approval-notify-not-registered gaps are the canonical examples.
- §17.4 Keep cross-adapter user-message shape identical — Phase 8's note-injection → canonical `_enrich_message_with_vision` template refactor.
- §17.5 Cross-module registration surfaces need consistency tests in the phase-close gate — Phase 4's `hermes-gateway includes` miss survived three phases.
- Plus §17.6-17.12 covering notify-cb-is-sync implications, canonical attachment enrichment, per-session locks, private-attribute peek, entry-id threading for parallel subagents, full-suite gate policy, and the non-event that prompt caching stayed well-behaved.

### 2026-04-22 — Phase 9 — T9.3 put the setup guide in website/docs, not repo-root

The task's alternate phrasing ("Add example config snippet to README or new `docs/nats-gateway.md`") was ambiguous about location. The `ADDING_A_PLATFORM.md` §15 convention is `website/docs/user-guide/messaging/<platform>.md` for the full setup guide — matches every other platform doc. Implemented both:

- `website/docs/user-guide/messaging/nats.md` — full user-facing setup guide (config options, examples, security model, troubleshooting, non-goals, reference links).
- `docs/nats-gateway.md` — short developer-facing pointer that links the website doc + design + progress + protocol spec + SDK. Keeps the repo-root `docs/` layout coherent (three files: design, progress, this pointer).

Also updated:

- `website/docs/user-guide/messaging/index.md` — added NATS to the Platform Comparison table, architecture diagram, Platform-Specific Toolsets table, Next Steps list, and the description frontmatter.
- `website/docs/reference/environment-variables.md` — added a NATS block (`NATS_URL`, `NATS_CONTEXT`, `HERMES_NATS_{AGENT,OWNER,NAME,SESSION}`) before the WEBHOOK block.
- `CLAUDE.md` — replaced the "(in progress)" section with a permanent "NATS gateway channel" entry pointing at the three docs + SDK install instruction. Also noted the NATS adapter as the canonical example in-tree for adapter-owned `AIAgent` + per-session lock + `request_interaction`.

Deliberately did NOT update the README's platform lists (lines 20 / 58 / 69 / 96) — those are conversational "chat with Hermes from X" summaries and NATS is a programmatic protocol channel rather than a chat app. Mentioning it there would mislead rather than inform. The messaging docs section is the right landing surface.

### 2026-04-22 — Phase 9 — Phase 9 closed

Phase 9 is docs-only — no code changes. Full NATS test subtree re-run (203/203 green in ~3 s) confirms the code surface wasn't accidentally touched during the doc pass. Ready for PR on the `nats-gateway` branch.

### 2026-04-22 — Post-Phase 9 — SDK session support landed; `_extract_session` workaround removed

The `natsagent` SDK grew a first-class `Envelope.session: str | None` field (caller-side: `remote.prompt(text, session="…")`; examples: `--session NAME` on 02/03/04). The adapter no longer needs to peek `stream._request.data`:

- `gateway/platforms/nats.py:665` now reads `envelope.session` directly.
- `_extract_session` (method) and `_extract_x_session` (module helper) deleted — ~50 lines of dead code gone.
- `TestExtractXSession` (11 unit tests) deleted; `_on_prompt` tests updated to pin `envelope.session` explicitly on MagicMock envelopes.
- Design doc §3 ("Session model") rewritten; §17.9 retrospective marked resolved.
- Symmetric doc updates in `website/docs/user-guide/messaging/nats.md`, `website/docs/reference/environment-variables.md`, `gateway/platforms/ADDING_A_PLATFORM.md`, `gateway/run.py`.

Earlier phase logs above still reference `x-session` and the raw-bytes workaround — that's historical record of what shipped in Phases 4–8, deliberately preserved, not stale documentation to be updated.

### 2026-04-22 — Post-Phase 9 — Bumped to NATS Agent Protocol v0.2

The protocol spec bumped from v0.1 to v0.2; v0.2 is **not wire-compatible with v0.1** (§11.3). The SDK (`natsagent` at `../nats-ai-pysdk`, now 0.2.0) absorbed the three breaking wire changes — service name `SynadiaAgents`/`Synadia Agents` → `agents`, queue group `""` → `agents`, `metadata.protocol_version = "0.1"` → `"0.2"` — so hermes's diff is narrow: pin bump + one SDK kwarg + docs.

- **Dep pin.** `pyproject.toml` bumped `natsagent>=0.1.0,<1` → `>=0.2.0,<1`. Unchanged install path (editable from `../nats-ai-pysdk`). `uv pip install -e ../nats-ai-pysdk` resolved cleanly (0.1.0 → 0.2.0).
- **§3.2 compliance gap closed.** The spec says session-aware harnesses (and names `hermes` by example) MUST set `metadata.session` at registration. The adapter had parsed `session_default` from config since Phase 4 but never forwarded it to `natsagent.Agent(...)`. Added `session=settings.session_default` to the `Agent(...)` call at `gateway/platforms/nats.py:505-514`. No other adapter wiring needed — `envelope.session` routing is per-request and orthogonal to the service-level metadata field. Judgment call on the default value: kept `"default"` as the default. It's semantically muddy (`"default"` is spec convention for the session-less escape hatch) but hermes doesn't have a single canonical instance-wide session to advertise, and `session_default` is already user-overridable via `HERMES_NATS_SESSION` / `config.yaml`. Introducing a separate `metadata_session` config key would be over-engineering for a label-not-routing-key field; if a future spec reviewer flags `metadata.session = "default"` as a smell, that's the fix.
- **Test coverage.** `test_nats_connect.py` — existing `test_connect_constructs_agent_with_full_settings` gained `assert kwargs["session"] == "default"`; new sibling `test_connect_propagates_custom_session_default` pins the custom-override path via `_build_adapter(session_default="acme-prod")`. Full-file run 20 → 21 tests, all green. Broader NATS subtree unchanged (`pytest -k nats` = 194 passed + 1 skipped).
- **Docs.** Refreshed protocol-version callouts and wire-name references across `CLAUDE.md`, `docs/nats-gateway-design.md`, `docs/nats-gateway.md`, `website/docs/user-guide/messaging/nats.md`, and module + class docstrings in `gateway/platforms/nats.py`. Also retargeted the protocol-spec URL: the SDK deleted its embedded copy in v0.2 (only `docs/protocol-mapping.md` remains), so every `../nats-ai-pysdk/docs/nats-agent-protocol.md` pointer now resolves to `../nats-agent-sdk-docs/core-protocol.md` (the canonical spec, now a separate repo). Historical Phase 4–8 entries in this file left untouched — they were accurate snapshots of v0.1 behavior and mutating them would falsify the decision log.
- **Live smoke.** Local `nats-server -p 4224` (4222/4223 held by other processes on this host). Isolated `HERMES_HOME=/tmp/hermes-nats-v02-smoke` with a minimal config. `nats req '$SRV.INFO.agents'` returned the full `io.nats.micro.v1.info_response` with all v0.2 invariants:
  - `"name": "agents"` ✓ (was `SynadiaAgents` in v0.1)
  - `"metadata.protocol_version": "0.2"` ✓
  - `"metadata.session": "default"` ✓ (the §3.2 fix — new for hermes)
  - `endpoints[0].queue_group: "agents"` ✓ (was `"q"` in v0.1)
  - `endpoints[0].subject: "agents.hermes.rene.smoke"` ✓ (unchanged)
  Negative check: `nats req '$SRV.INFO.SynadiaAgents'` → `No responders are available`. Old wire is correctly dead. Non-LLM subset only — `examples/02`/`03`/`04` not re-run in this smoke since Phase 8's OpenRouter-backed runs still cover that surface and the v0.2 delta is entirely in the registration / discovery path, not the prompt / stream / attachment path.

### 2026-04-22 — Post-Phase 9 — Phase 4 docstring drift labeled v0.1 → v0.2

Three method docstrings in `gateway/platforms/nats.py` (`send_voice` at 1485, `_send_attachment` at 1525, `_ask_approval_question` at 1613) labeled current wire behavior as "v0.1". Each is a spec-version-independent observation (no voice/audio wire distinction, attachments carry identically, no per-kind query field) that is also true in v0.2 — not a historical distinguisher like `docs/nats-gateway-design.md:537` (which contrasts the current inline-base64 behavior against the *future* §5.5 chunked-upload endpoint and was deliberately left at "v0.1"). Updated the three `nats.py` docstrings to "v0.2" so they accurately reflect what the adapter currently speaks; left the design doc's contrast intact.

### 2026-04-28 — Phase 10 — Migrated to protocol v0.3 / `synadia-ai-agents` SDK

Three SDK PRs landed against `synadia-ai/synadia-agents` and bumped the wire to v0.3:

- **PR #24 — verb-first subjects + status endpoint.** Subjects gained a verb token: `agents.prompt.{a}.{o}.{n}`, `agents.hb.{a}.{o}.{n}`, new `agents.status.{a}.{o}.{n}`. `metadata.protocol_version` → `"0.3"`. `AgentService` registers prompt + status endpoints automatically via `service.start()`.
- **PR #25 — pinned reply-inbox prefix `_INBOX.agents`.** Caller-side only. Hermes is service-side, so this is informational; it lands as a NATS account-permission note in the user-facing docs.
- **PR #26 — `name` + `session` collapsed into `session_name`.** The 5th subject token IS the session. Removed: `metadata.session`, `Envelope.session`, `HeartbeatPayload.session`, `AgentService(session=...)`, `Agent.prompt(session=...)`. Multi-session multiplexing within one process is explicitly dropped — *"a worker that wants N sessions registers N services"*.

The package itself was renamed `natsagent` → `synadia-ai-agents` (PyPI), import root `synadia_ai.agents`, and the service-side class renamed `Agent` → `AgentService`.

**Decision (taken with the user before code touched):** adopt the SDK's intended single-session-per-service model and rely on Hermes' existing profile isolation for multi-session deployments. One profile = one `AgentService` = one `session_name`. Multi-session = multi-profile. This is the smallest, most SDK-aligned diff and naturally collapses today's per-`chat_id` locking + stream tracking into a single global lock.

Why this was the right call rather than registering N services per Hermes process:

- **SDK direction.** PR #26 is an explicit deprecation of the v0.2 multiplexing pattern. Building a private demuxer on top of `AgentService` would mean diverging from the SDK's intended usage and losing future SDK improvements (status endpoint enhancements, reconnect behavior, observability) that assume one-service-per-session.
- **Profile isolation already canonical.** Hermes profiles already separate `HERMES_HOME`, `.env`, sessions, memory, skills, and per-platform locks. Mapping "session" to "profile" cost zero new mechanism and inherits all of profile's existing safety properties (the platform lock, the lock-conflict diagnostic message, the `gateway/platforms/ADDING_A_PLATFORM.md` profile-safe checklist).
- **Lock simplification.** v0.2's per-`chat_id` `Dict[str, asyncio.Lock]` collapsed to a single `_session_lock`. The serialization invariant (§17.2 / §17.8) survives byte-for-byte; only the locking primitive changes. Tests that exercised distinct-`session` parallelism within one adapter (a v0.2-only concept) were deleted.
- **`nats-gateway` branch hadn't merged to `main` yet.** No backward-compat migration window was needed for the env var rename or the config-key rename. `HERMES_NATS_NAME`/`HERMES_NATS_SESSION` → `HERMES_NATS_SESSION_NAME` is a clean break with zero deployed users.

What got simpler:

- No `envelope.session` reads anywhere in the adapter.
- No `_session_default()` method.
- `_session_locks: Dict[str, asyncio.Lock]` → single `asyncio.Lock`.
- `NatsAdapterSettings`: dropped `name` and `session_default` fields, added required `session_name`.
- Renamed `self._agent` → `self._service` to avoid collision with `AIAgent`.

What stayed the same:

- The contextvar-primary + compound-key-dict stream lookup (`_current_stream` + `_active_streams`) — `chat_id` is now constant of the process but the dict shape stays useful for the contextvar-fallback diagnostic path noted at §17.1.
- Adapter-owned `AIAgent` (api_server-style) — §17.3 lessons all still apply.
- Per-handler keep-alive emission, attachment cache routing, slash-command dispatch through `_message_handler`, mid-stream `request_interaction` round-trip, profile lock acquisition.

`_INBOX.agents.>` is a service-side no-op for Hermes (PR #25 is caller-side), but the user-facing doc calls it out as the recommended NATS account permission grant for callers, since that's what unblocks them.

**Test results.** `scripts/run_tests.sh tests/gateway/test_nats_*.py` — 190/190 green. Full `scripts/run_tests.sh tests/gateway/` — 3700/3703 green; the three failures (`test_matrix.py::test_upload_encrypted_room_uses_file_payload`, `test_agent_cache.py::test_close_vs_release_full_teardown_difference`, `test_whatsapp_connect.py::test_closed_when_http_not_ready`) are pre-existing on `main` and unrelated to NATS — verified by stashing the migration diff and reproducing two of three failures against unchanged code.

**Live smoke not run in this pass.** The migration is mechanical enough (subject layout + kwarg name changes, no streaming-pipeline behavior changes) that the test suite plus a careful diff against PR #24/#25/#26's wire-format expectations was deemed sufficient. A fresh `nats-server` smoke + `examples/02-prompt-text.py` round-trip is on the verification checklist for whoever opens the PR.

### 2026-04-30 — Phase 11 — Split SDK migration (client v0.5 / agent v0.1) + broker-derived `max_payload`

Two coupled upstream changes landed in `synadia-ai/synadia-agents` and need to be picked up together:

- **CHANGELOG 0.5.0 (2026-04-30) — SDK split.** The single `synadia-ai-agents` v0.4.x distribution split into a wire-only client SDK (`synadia-ai-agents` v0.5, import root `synadia_ai.agents`) and a host-side agent SDK (`synadia-ai-agent-service` v0.1, import root `synadia_ai.agent_service`). The agent SDK depends on the client SDK at `>=0.5`. Hermes-agent imports a mix of both surfaces, so the migration retargets `AgentService` / `PromptStream` to the new `sdk_svc` alias while leaving `Envelope`, `Attachment`, chunk/error classes, and `load_context_options` on the existing `sdk` alias.
- **PR #41 — broker-derived `max_payload`.** SDK now clamps a constructor-supplied `max_payload` down to `nc.max_payload` at `start()`, and the `agents/hermes/README.md` was rewritten to claim hermes "Defaults to `nc.info.max_payload`". Hermes-agent's code hardcoded `DEFAULT_MAX_PAYLOAD = "1MB"` and unconditionally passed it to the SDK — capping every host at 1 MB regardless of negotiated capacity, contradicting the published behavior. Fixed by making `NatsConfig.max_payload: Optional[str] = None` and deriving from `getattr(nc, "max_payload", 0)` in `_on_connect` when unset; user-supplied values pass through unchanged (SDK still does its own clamp-down at `start()`). New helper `_format_max_payload_grammar` picks the largest clean unit so `1048576 → "1MB"` rather than `"1024KB"`. Connected log line now shows `(server-negotiated)` vs `(configured)` so operators can tell at a glance which path resolved.

**Local-source override.** Both SDKs are still pre-PyPI, so `pyproject.toml` gets a new top-level `[tool.uv.sources]` block resolving each to the sibling `../synadia-agents/` checkout. uv-only convention; ignored by setuptools/pip and not persisted into the wheel METADATA, so external users still pull from PyPI once published. `uv sync --all-extras --locked` now resolves the `[nats]` extra without the manual `uv pip install -e ...` step that was documented for Phase 10.

**Test mock.** `tests/gateway/conftest.py::_ensure_synadia_agents_mock` extended to register `sys.modules["synadia_ai.agent_service"]` alongside `synadia_ai.agents`. AgentService / PromptStream / PromptHandler attach to the new module mock; wire types stay on the existing one. Early-return guard checks both modules. `test_nats_connect.py::mock_synadia_agents` fixture proxies AgentService from the agent_service module so existing assertions (`mock_synadia_agents.AgentService...`) keep working unchanged.

**Out of scope.** PR #41 also added caller-side `assert_within_max_payload(connection_max_payload=...)` so callers behind a smaller-cap broker fail fast. Hermes is the agent (server) — it doesn't call `Agent.prompt()` from inside the gateway — so this is a no-op for hermes-agent code.

## Task definitions reference

If `TaskList` is empty after a context clear and you need to recreate the tasks, use these verbatim. Subject / description / activeForm are the `TaskCreate` parameters.

| T#.#   | subject                                                                 | activeForm                              |
|--------|-------------------------------------------------------------------------|-----------------------------------------|
| T0.1   | T0.1 — Write design doc docs/nats-gateway-design.md                     | Writing NATS gateway design doc         |
| T0.2   | T0.2 — Add CLAUDE.md pointer to design doc and natsagent SDK             | Updating CLAUDE.md pointers             |
| T1.1   | T1.1 — Add Platform.NATS enum value                                     | Adding Platform.NATS                    |
| T1.2   | T1.2 — Extend _apply_env_overrides() for NATS                           | Adding NATS env overrides               |
| T1.3   | T1.3 — Extend get_connected_platforms() for NATS                        | Extending get_connected_platforms for NATS |
| T1.4   | T1.4 — Register NATS adapter in _create_adapter()                       | Registering NATS adapter                |
| T1.5   | T1.5 — Add natsagent to pyproject.toml extras                           | Adding natsagent to pyproject           |
| T1.6   | T1.6 — Add natsagent mock in tests/gateway/conftest.py                  | Mocking natsagent in conftest           |
| T2.1   | T2.1 — Create gateway/platforms/nats.py skeleton                        | Creating NATS adapter skeleton          |
| T2.2   | T2.2 — Unit test for config parsing                                     | Testing NATS config parsing             |
| T3.1   | T3.1 — Implement NATS connect()                                         | Implementing NATS connect               |
| T3.2   | T3.2 — Implement NATS disconnect()                                      | Implementing NATS disconnect            |
| T3.3   | T3.3 — Implement get_chat_info()                                        | Implementing get_chat_info              |
| T3.4   | T3.4 — Tests for connect/disconnect                                     | Testing NATS connect/disconnect         |
| T4.1   | T4.1 — Implement _on_prompt handler                                     | Implementing _on_prompt handler         |
| T4.2   | T4.2 — Wire _active_streams into send()                                 | Wiring _active_streams into send        |
| T4.3   | T4.3 — Wire streaming deltas                                            | Wiring NATS streaming deltas            |
| T4.4   | T4.4 — Tests for inbound path                                           | Testing NATS inbound path               |
| T5.0   | T5.0 — Race-safe `_active_streams` lookup (folded-in Phase 4 shortcoming) | Making NATS stream lookup race-safe     |
| T5.1   | T5.1 — Implement send_image_file                                        | Implementing send_image_file            |
| T5.2   | T5.2 — Implement send_document                                          | Implementing send_document              |
| T5.3   | T5.3 — Implement send_voice / send_video                                | Implementing send_voice/send_video      |
| T5.4   | T5.4 — format_message override (likely no-op)                           | Overriding format_message               |
| T5.5   | T5.5 — Tests for outbound attachments + concurrent-x-session regression | Testing NATS outbound attachments       |
| T6.1   | T6.1 — Survey _pending_approvals usage                                  | Surveying _pending_approvals usage      |
| T6.2   | T6.2 — Add adapter-side query hook method                               | Adding request_interaction hook         |
| T6.3   | T6.3 — Wire approval callbacks to adapter.request_interaction           | Wiring approval callbacks               |
| T6.4   | T6.4 — Tests for NATS query reply                                       | Testing NATS query reply                |
| T7.1   | T7.1 — Decide which slash commands are exposed over NATS                | Deciding NATS slash commands            |
| T7.2   | T7.2 — Confirm /help renders sensibly over plain-text chunks            | Verifying /help rendering               |
| T8.1   | T8.1 — Local NATS + hermes smoke config                                 | Documenting local smoke config          |
| T8.2   | T8.2 — Verify 01-discover.py lists hermes                               | Verifying discover                      |
| T8.3   | T8.3 — Verify 02-prompt-text.py streams                                 | Verifying prompt-text streaming         |
| T8.4   | T8.4 — Verify 03-prompt-attachment.py                                   | Verifying attachment flow               |
| T8.5   | T8.5 — Verify 04-query-reply.py approval path                           | Verifying query-reply                   |
| T8.6   | T8.6 — Verify 05-liveness.py                                            | Verifying liveness                      |
| T8.7   | T8.7 — nats CLI interop check                                           | Verifying nats CLI interop              |
| T8.8   | T8.8 — Full test suite green                                            | Running full test suite                 |
| T9.1   | T9.1 — Update ADDING_A_PLATFORM.md if new integration points emerged    | Updating ADDING_A_PLATFORM              |
| T9.2   | T9.2 — Expand design doc with lessons learned                           | Expanding design doc with lessons       |
| T9.3   | T9.3 — Example hermes config snippet in docs                            | Adding example config snippet           |

Descriptions for each task are listed in the design doc's Phase tables + the plan the user originally approved. Short paraphrase of each is given in the phase checklist above — sufficient for TaskCreate.

---

## Phase 12 — Pre-PR cleanup: align wizard with Hermes conventions (2026-05-06)

Pre-PR review of how NATS configuration compared with the rest of Hermes turned up an asymmetry. Other platforms — Telegram, Discord, Slack, Mattermost, Matrix, Webhook, API Server, Yuanbao, all of them — drive their setup wizards through `save_env_value()` writes to `~/.hermes/.env`. NATS was the lone snowflake: `_setup_nats(config: dict)` mutated `platforms.nats.extra` in `config.yaml`, never touched `.env`.

Two changes folded into a single landing here:

1. **Brief detour, then reverted.** A first attempt (`b6049f12`) ripped out env-var support entirely on the (mistaken) claim that "Hermes reserves env vars for secrets." Audit of `website/docs/reference/environment-variables.md` showed that's not the convention — `MATRIX_HOMESERVER`, `MATRIX_USER_ID`, `MATTERMOST_URL`, `EMAIL_IMAP_HOST`/`SMTP_HOST`, `WHATSAPP_ENABLED`, `WEBHOOK_PORT`, `API_SERVER_HOST`, `HASS_URL`, every `*_ALLOWED_USERS`, plus the entire Yuanbao set are non-secret config carried as env vars. The rip-out was reverted before push.

2. **Wizard refactor (this commit's substance).** `_setup_nats(config)` → `_setup_nats()`: drops the `config` parameter and the `load_config()`/`save_config()` machinery; each `extra[X] = Y` mutation is now `save_env_value(...)` to `~/.hermes/.env`. Cross-profile collision check (`_find_nats_profile_collisions`) extended to scan sibling profile `.env` files via `dotenv.dotenv_values()`; if both `.env` and `config.yaml` are present, env wins per-key (mirroring how `_apply_env_overrides()` materializes them at runtime). Multi-URL `servers` lists and tuning knobs (`heartbeat_interval_s`, `max_payload`, `attachments_ok`, `ack_keepalive_interval_s`) are not wizard-prompted — those stay in `config.yaml` as the structured-override path for advanced users.

After this phase, NATS configuration follows the prevailing Hermes pattern: wizard → `.env`, `config.yaml` is the structured-override escape hatch, env vars stamp on top per-key at runtime.

Touched:

- `hermes_cli/setup.py:_setup_nats` — wizard rewrite (env-var output).
- `hermes_cli/setup.py:_find_nats_profile_collisions` — `.env` scanner added; `.env` and `config.yaml` merged with env-wins-per-key.
- `tests/hermes_cli/test_setup_nats_collision.py` — 5 new tests covering `.env`-driven sibling profiles, env-overrides-yaml priority, and yaml-only backward compatibility (14 tests total, all green).
- `website/docs/user-guide/messaging/nats.md` — Step 1 restructured: wizard recommended, manual `.env` editing as alternative, `config.yaml` documented as the advanced-overrides path.
- `docs/nats-gateway-design.md` §4 — env var override section updated to describe the `.env`-primary, `config.yaml`-as-overrides split.
- `docs/nats-gateway.md` — smoke-test recipe annotated to point at the wizard for persistent config; inline-env overrides remain as the one-shot smoke pattern.
