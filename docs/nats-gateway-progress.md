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

- **Last completed phase:** Phase 6 — Mid-stream queries / approval round-trip (T6.1 through T6.4)
- **Next phase:** Phase 7 — Slash commands (T7.1 through T7.2)
- **Branch:** `nats-gateway` (feature branch; PR target is `main`)
- **Known blockers:** none
- **Open design questions pending user input:** 4 items listed in §16 of `docs/nats-gateway-design.md`. Default answers are noted there; proceed with defaults unless the user redirects.

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

- [ ] **T7.1** — Confirm gateway-eligible commands (`/new`, `/reset`, `/model`, `/status`, `/stop`, `/help`, `/compress`, `/resume`) route as `MessageEvent(COMMAND)` with no new code. Data-only verification.
- [ ] **T7.2** — Manually verify `/help` output renders sensibly as plain-text chunks over NATS

### Phase 8 — End-to-end verification (manual; requires local nats-server)

- [ ] **T8.1** — Local nats-server + hermes smoke config (already documented in §14 of design doc; confirm it still applies)
- [ ] **T8.2** — `examples/01-discover.py` lists `agents.hermes.<owner>.<name>`
- [ ] **T8.3** — `examples/02-prompt-text.py` — simple prompt streams a response
- [ ] **T8.4** — `examples/03-prompt-attachment.py` — hermes ingests a PDF and streams a summary
- [ ] **T8.5** — `examples/04-query-reply.py` — tool call that requires approval; Query chunk; reply "yes"; stream resumes
- [ ] **T8.6** — `examples/05-liveness.py` in background; kill hermes; `is_online()` flips False after 3× interval
- [ ] **T8.7** — `nats` CLI interop — `nats req '$SRV.INFO.Synadia Agents'` and `nats sub 'agents.hermes.*.*.heartbeat'` per protocol Appendix C
- [ ] **T8.8** — `scripts/run_tests.sh` — full suite green

### Phase 9 — Polish & docs

- [ ] **T9.1** — Update `gateway/platforms/ADDING_A_PLATFORM.md` with any new integration points that emerged (e.g. `request_interaction` if it gets generalized)
- [ ] **T9.2** — Append "Lessons learned" section to `docs/nats-gateway-design.md` (especially surprises in stream_delta_callback wiring or attachments)
- [ ] **T9.3** — Add example config snippet to README or new `docs/nats-gateway.md` (user-facing)

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

### 2026-04-22 — Phase 6 — Pre-existing test failures re-confirmed

`scripts/run_tests.sh tests/gateway/` shows the same two Phase 1 failures plus a third pre-existing flake (`tests/tools/test_approval_heartbeat.py::TestApprovalHeartbeat::test_heartbeat_import_failure_does_not_break_wait`). All three fail identically with Phase 6 changes stashed, so not regressions introduced by this phase. Flagged here so a future Claude doesn't waste cycles blaming them on NATS.

### 2026-04-22 — Phase 6 — Dispatch-failure fallback resolves as "deny"

`_nats_approval_notify` reads the return value of `dispatch_approval_via_request_interaction` and, when it's False (scheduling on `loop` raised — only happens during a shutdown race where the loop is already closed), immediately calls `resolve_gateway_approval(session_key, "deny")`. Without this, the agent thread blocked on `entry.event.wait()` would hang for the full `gateway_timeout` (default 300 s) before the framework's timeout surfaces with "deny" anyway. Same outcome, but 300 s → ~0 ms. Regression test: `test_notify_callback_resolves_as_deny_when_dispatch_fails`.

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

### 2026-04-22 — Phase 6 — Pre-existing test failures re-confirmed

`scripts/run_tests.sh tests/gateway/` shows the same two Phase 1 failures plus a third pre-existing flake (`tests/tools/test_approval_heartbeat.py::TestApprovalHeartbeat::test_heartbeat_import_failure_does_not_break_wait`), plus a sporadic `tests/gateway/test_whatsapp_connect.py::TestBridgeRuntimeFailure` ordering flake under the 4-worker xdist setup. All fail identically with Phase 6 changes stashed, so not regressions introduced by this phase. Flagged here so a future Claude doesn't waste cycles blaming them on NATS.

### 2026-04-22 — Phase 6 — Dispatch-failure fallback resolves as "deny"

`_nats_approval_notify` reads the return value of `dispatch_approval_via_request_interaction` and, when it's False (scheduling on `loop` raised — only happens during a shutdown race where the loop is already closed), immediately calls `resolve_gateway_approval(session_key, "deny")`. Without this, the agent thread blocked on `entry.event.wait()` would hang for the full `gateway_timeout` (default 300 s) before the framework's timeout surfaces with "deny" anyway. Same outcome, but 300 s → ~0 ms. Regression test: `test_notify_callback_resolves_as_deny_when_dispatch_fails`.

### 2026-04-22 — Phase 6 — Known shortcomings (NOT fixed; carry forward)

1. **Entry-pop is FIFO, not reply-keyed.** `resolve_gateway_approval(session_key, choice)` pops the OLDEST entry from `_gateway_queues[session_key]`, not "the one matching this query reply". Session serialization eliminates the concurrent same-`x-session` case, but parallel subagents *inside one handler* still share a single `session_key` and can hit dangerous commands simultaneously. If the caller replies to query 2 before query 1, coroutine 2's `resolve_gateway_approval` pops entry 1 (FIFO oldest) with query 2's choice; entry 2 then resolves with query 1's choice → cross-routed approvals. Narrow (requires `delegate_tool` parallel execution + concurrent dangerous commands within one handler), framework-level root cause. Fix would require extending `tools/approval.py` with a `_current_entry` contextvar + a `resolve_approval_entry(entry_obj, choice)` function that matches by object identity — surgical but touches gateway-wide code. Defer to post-MVP.

2. **Approval reply "a" maps to "always", not "approve once".** Consistent with the CLI's `[o]nce | [s]ession | [a]lways | [d]eny` shortcuts, but users who type "a" thinking "approve" get permanent allowlisting instead of one-time. Mitigated by the prompt text explicitly listing `once | session | always | deny` as the four options. Full words are unambiguous; only the single-letter form has this footgun.

---

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
