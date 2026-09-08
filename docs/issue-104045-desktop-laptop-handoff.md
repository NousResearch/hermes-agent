# Issue #104045 — clean Desktop/laptop handoff on one gateway

## Verdict

Issue #104045 is a real architectural gap in the gateway/Desktop session contract, but the original production incident is not independently reproducible from the issue alone. The issue explicitly says that the exact error text and client versions are still missing. The failure mechanism is nevertheless confirmed in the current code and is supported by the related issue history.

The fix treats one runtime session as one state machine with:

- one durable/live session record;
- zero or more read-only viewers;
- at most one writer transport for prompt admission;
- an explicit, atomic handoff from one live writer to another viewer;
- a detached state when no live transport remains.

It does not create a second session, a second active-session lease, or a second prompt execution path.

## Remote issue and related work

Issue: [#104045](https://github.com/NousResearch/hermes-agent/issues/104045)

The requested behavior is:

1. resume the same conversation and latest task state on either device;
2. keep gateway work running through disconnect/sleep;
3. offer **Continue on this device** when another client still owns the writer;
4. prevent duplicate submissions and stale overwrites;
5. recover without restarting the gateway.

Related evidence reviewed:

- [#79064](https://github.com/NousResearch/hermes-agent/issues/79064) — a second client's `session.resume` could replace the live session transport, stop the first Desktop stream, and contribute to stale truncation state on reconnect. This is the closest symptom-level evidence.
- [#101408](https://github.com/NousResearch/hermes-agent/issues/101408) — a successfully reconnected Desktop could leave a mounted pane detached while still accepting input.
- [#92710](https://github.com/NousResearch/hermes-agent/issues/92710) — a WebSocket reconnect could lose the final event window even though the backend completed the turn and persisted the result.
- [#100970](https://github.com/NousResearch/hermes-agent/issues/100970) — remote-gateway sleep/network interruptions could leave Desktop state out of sync with the durable session.
- [#77127](https://github.com/NousResearch/hermes-agent/issues/77127) and merged [#96438](https://github.com/NousResearch/hermes-agent/pull/96438) — stale WebSocket teardown could close or detach a session after a replacement connection had already resumed it.
- Open [#95709](https://github.com/NousResearch/hermes-agent/pull/95709) — serializes WebSocket transport ownership. It addresses the teardown/rebind race but does not provide the explicit Desktop writer handoff UX implemented here.
- Open [#101531](https://github.com/NousResearch/hermes-agent/pull/101531) — reattaches mounted Desktop sessions after WebSocket reconnect. It is a client-side re-registration fix, not a writer-ownership protocol.
- Open [#103797](https://github.com/NousResearch/hermes-agent/pull/103797) — configurable per-session exclusivity/queue behavior. It changes what happens after a refusal; it does not distinguish read-only viewers from a writer on one live runtime session.

No second session registry was added. The existing active-session lease remains the process/session exclusivity mechanism.

## Graphify investigation

The repository graph was used to narrow the gateway traversal.

Commands and evidence:

- `graphify update tui_gateway --no-cluster`
  - completed successfully;
  - rebuilt `2407 nodes, 5129 edges`;
  - updated `tui_gateway/graphify-out/graph.json`.
- `graphify explain "_ensure_session_writer"`
  - resolved `tui_gateway/session_lifecycle.py:120` and its writer-lock dependency.
- `graphify explain "_drain_queued_prompt"`
  - resolved `tui_gateway/session_auto_continue.py:279`, its post-turn callers, and the new queued-writer preparation path.
- `graphify path "prompt.submit" "_lock_in_submit_turn" --undirected`
  - resolved the prompt admission path through `methods_prompt.py` and `prompt_turn.py`.
- The first full-repository graph update exceeded the tool timeout because of the repository size. The focused `tui_gateway` graph completed and was refreshed after the final gateway edits.

Graph output was used for navigation and checked against source lines; it was not treated as proof without source/test verification.

## Root cause

Before this change, `session["transport"]` served both as the event sink and as the implicit prompt owner. The following sibling paths could overwrite it:

- live `session.resume`/`session.activate` payload construction;
- unpersisted-session resume;
- `prompt.submit`;
- queued-prompt draining;
- disconnect teardown and orphan handling.

A representative race was:

1. Desktop A owns runtime session S.
2. Laptop B resumes S.
3. The resume fast path assigns B's transport to `session["transport"]`.
4. A either stops receiving events or is later affected by stale disconnect teardown.
5. B can submit on the same runtime because the existing active-session lease recognizes S as the same live session; it does not encode WebSocket writer identity.
6. A reconnects with stale UI/rewind state, creating the possibility of dropped events, duplicate actions, or stale truncation.

The active-session lease was therefore necessary but insufficient. It prevents competing runtime sessions; it does not provide a per-runtime writer gate.

The queue path had the same class of bug: an accepted queued prompt retained the old transport, and `_drain_queued_prompt` could restore that transport after disconnect. A queue cannot be allowed to resurrect a closed socket.

## Algorithm and invariants

### State

Each live session now has:

- `viewers`: transport-to-last-seen timestamps;
- `_writer_transport`: the only transport allowed to admit a prompt;
- `_writer_lock`: the per-session serialization lock;
- `transport`: the current event sink, normally the writer or the detached sentinel;
- the existing `active_session_lease`.

### Resume/attach

`session.resume` and `session.activate` register the calling transport as a viewer. They do not replace a live writer. If the prior writer is detached/closed, the new transport may reclaim it automatically; otherwise the user must explicitly hand off.

### Prompt admission

`prompt.submit` follows this order:

1. resolve the existing session;
2. acquire the session writer lock;
3. validate the calling transport against `_writer_transport`;
4. only then acquire the active-session lease if needed;
5. check busy/queue state;
6. mark the turn running while the writer lock is still held;
7. release the writer lock and execute the turn.

Keeping writer validation and the `running=True` transition in one critical section prevents this race:

- submit sees an idle session;
- handoff also sees an idle session;
- both proceed and two turns are admitted.

The active-session lease is intentionally checked after the read-only writer gate so a viewer refused for local ownership cannot claim a global lease as a side effect.

### Explicit handoff

`session.handoff` is valid only for a registered viewer and only while the current turn is idle. It atomically:

- verifies the session record is still registered;
- verifies the caller transport is live and attached;
- rejects a running turn;
- assigns `_writer_transport` and `transport` to the new writer;
- increments `_writer_generation`.

The Desktop action runs `session.handoff` first and retries `prompt.submit` once. Concurrent clicks are coalesced client-side.

### Disconnect and queued work

Disconnect teardown removes the transport from every session's viewer set. If it was the writer, the writer becomes the detached sentinel. A surviving viewer can be promoted by the next admitted action or queued-turn preparation. If no viewer survives, the queued turn keeps running against the detached sentinel so durable state is preserved and a later resume can recover it.

The queue entry's original transport is diagnostic history, not authority.

## Implementation

Backend changes:

- `tui_gateway/session_lifecycle.py`
  - `_writer_lock`, `_writer_transport_is_dead`;
  - `_bind_session_viewer_transport`;
  - `_ensure_session_writer`;
  - `_handoff_session_writer`;
  - `_prepare_queued_session_writer`;
  - viewer-aware disconnect cleanup.
- `tui_gateway/methods_session.py`
  - initializes writer/viewer state for new sessions;
  - routes live/unpersisted resume through the viewer bind helper;
  - registers `session.handoff`.
- `tui_gateway/server.py`
  - initializes writer/viewer state for eager and deferred records;
  - binds a viewer before taking the history lock in the live payload helper, preserving lock order.
- `tui_gateway/methods_prompt.py`
  - serializes writer validation through running-state admission;
  - avoids restoring a closed request transport.
- `tui_gateway/session_auto_continue.py`
  - serializes queued-turn admission with the writer lock;
  - selects a surviving writer instead of blindly restoring the queued entry's transport.

Desktop changes:

- `apps/desktop/src/app/session/hooks/use-prompt-actions/submit.ts`
  - recognizes structured `SESSION_NOT_OWNED` errors;
  - renders a warning notification with the **Continue on this device** action;
  - performs handoff and one retry without duplicating the prompt.
- `apps/desktop/src/i18n/types.ts` and the supported locale files
  - add handoff action/success copy.

Regression coverage:

- `tests/tui_gateway/test_desktop_session_handoff.py`
  - viewer cannot submit through transport rebinding;
  - idle handoff succeeds;
  - running turn cannot be stolen;
  - writer disconnect becomes reclaimable;
  - demoted-viewer disconnect does not detach the new writer;
  - queued work uses a surviving viewer after writer disconnect.

## Verification

Passed with the repository's canonical runner:

- `HERMES_PYTHON='C:/Users/Nitro/hermes-agent/.venv/Scripts/python.exe' scripts/run_tests.sh tests/test_tui_gateway_queue_on_busy.py tests/tui_gateway/test_desktop_session_handoff.py tests/tui_gateway/test_session_resume_db_ownership.py tests/tui_gateway/test_session_db_ownership_teardown.py tests/test_active_session_exclusivity.py -q`
  - `76 passed, 0 failed` in 5.8 seconds;
  - no flaky retry was reported in this final run.
- `npm ci --ignore-scripts --no-audit --no-fund --engine-strict=false`
  - installed the locked JavaScript dependencies in the isolated worktree;
  - emitted engine warnings because the host has Node `22.16.0` / npm `11.4.1`, while the repository currently requires Node `>=22.22.0`.
- `npm --prefix apps/desktop run typecheck -- --pretty false`
  - passed with exit code `0` after dependency installation.
- `python -m compileall -q tui_gateway`.
- gateway import check confirmed `session.handoff` is registered.
- `git diff --cached --check` passed.
- `graphify update tui_gateway --no-cluster` completed before the final integration review.

The full `tests/test_tui_gateway_server.py` file was not used as a merge gate: the canonical per-file runner killed it at its 300-second file limit after partial progress, making that run inconclusive rather than green or red.

The two focused server tests were then run on the current branch:

- `test_profile_scoped_agent_build_starts_mcp_discovery_in_profile_home` — passed;
- `test_model_options_preserves_canonical_custom_row_after_agent_init` — failed because the response included `anthropic` in addition to `custom:local-ollama`.

The remaining failure was reproduced in a detached worktree at the exact `origin/main` baseline with the same canonical command: `1 failed, 636 deselected`. It does not touch any handoff, transport, session lifecycle, or Desktop prompt code changed here, so it is recorded as a pre-existing baseline failure and is not folded into this PR.

An earlier run on the stale issue-investigation branch reported both server tests failing (`635 passed, 2 failed`); rebasing the handoff diff onto the current `origin/main` showed that the MCP test has since passed on baseline. Neither result changes the handoff-specific evidence above.

No authenticated two-device Desktop E2E was available in this environment.

## Scope limits

This change fixes the ownership/handoff class. It does not claim to implement the larger event-log protocol proposed in #92710:

- no persistent event sequence/ACK cursor;
- no replay-to-live atomic event stream;
- no bounded event-log snapshot fallback;
- no independent stale-truncation confirmation protocol.

Those are separate follow-up concerns. The current fix prevents the transport hijack that was a prerequisite for the reported handoff failure and prevents a demoted or closed transport from admitting another prompt.

No credentials, tokens, connection strings, or private logs were recorded. No agents/subagents or infographics were used.
