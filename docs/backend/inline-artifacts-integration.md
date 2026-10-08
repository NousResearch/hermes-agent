# Inline HTML artifacts and delegation integration

## Status and scope

This is a backend-only implementation for Grok's later frontend wiring. No renderer, GUI, styles, mobile code, live configuration or services changed. Nothing was deployed or visibly rendered. The parent must independently review and rerun the backend tests before integration.

The implementation adds `publish_html` and two authenticated JSON reads. It reuses canonical `state.db` message persistence and the existing post-flush `tool.complete` event. It does not create an artifact database, another delegation tool or another task store. `delegate_task` and its lifecycle code are unchanged.

Source base is `457a1e1cdb7fc00fe21d5b8503dae1f9e2effbb0`. Branch is `backend/inline-artifacts`. Worktree is `/home/mpotter2002/.hermes/profiles/dr-eggbot/cache/scratch/inline-artifacts`. The implementation commit is `4c974a94a797704c3eaf598fcda25dcfa2eb5fc3`. The handoff document is a subsequent documentation-only commit on the same branch; `git rev-parse backend/inline-artifacts` resolves the complete candidate for parent review. No coordinated frontend changes are included.

## Publication contract

The model-facing tool is `publish_html`, in toolset `artifacts` and the shared core bundles. Its schema accepts only these fields:

```json
{"title":"Report","html":"<h1>Hello</h1>","fallback":"Hello. This report is available as an inline HTML artifact on supported clients."}
```

- `title` is a non-empty string, at most 200 characters.
- `html` is non-empty UTF-8 HTML, at most 65536 bytes.
- `fallback` is non-empty text, at most 4000 characters. Display it as text, never HTML.
- No `session_id`, `profile`, file path, target message or alternate conversation selector is accepted. Runtime dispatch supplies the current conversation identity. Calling without a bound conversation returns a tool error.
- The format is static HTML. Scripts, forms, privileged APIs and external resource loading are not supported. Supply data-URL images and inline styles when needed.
- Validation failures use the existing `{"error":"..."}` tool envelope and publish nothing.

A successful tool result is `{"artifact":ArtifactContent,"text":fallback}`. `ArtifactContent` has `version:1`, `id`, `title`, `format:"html"`, `html` and `fallback`. The ID is `html_` followed by 64 lowercase hexadecimal characters. It is SHA-256 of the UTF-8 JSON array `[boundSessionId,title,html,fallback]`. IDs are opaque to clients. Identical retries in the same bound session return the same ID. Changed content, title or fallback produces a new ID, not an in-place revision.

IDs are scoped by connection, profile and conversation. Identical session IDs and content in separate profiles may have the same artifact ID. This does not authorize retrieval from another profile. Always retain the originating connection/profile in the client's query key.

## Canonical storage and exact association

`agent/tool_executor.py` writes the full content into the canonical tool-result message's `display_metadata.inline_artifact` before the existing append and SessionDB flush. This works without a GUI metadata callback. Completion emits only after successful persistence. A failed flush emits no successful tool completion, and the HTTP reads cannot discover an uncommitted artifact.

The artifact belongs to its tool-result occurrence, not to the newest assistant bubble. Preserve the transcript's existing order. A cold projected row contains:

```json
{
  "role":"tool",
  "name":"publish_html",
  "tool_call_id":"provider-call-id",
  "row_id":42,
  "args":{"title":"Report","fallback":"Hello"},
  "display_metadata":{
    "inline_artifact":{
      "version":1,
      "id":"html_<64 lowercase hexadecimal characters>",
      "title":"Report",
      "format":"html",
      "fallback":"Hello",
      "byte_length":14
    }
  }
}
```

The example is schematic. Row 42 and the ID placeholder are not recorded test output. Optional timestamp, context and label fields retain their existing meanings. `row_id` is durable SQLite identity; `tool_call_id` identifies the provider's call. The public metadata is `ArtifactSummary`, the full content fields minus `html`, plus UTF-8 `byte_length`.

`session.history`, resume/reopen projections and the existing history shaper carry this summary on the exact tool row. They omit `html` from projected tool args and artifact metadata. Raw authenticated session export/detail APIs remain canonical exports and can include the original tool input/result and content sidecar. Do not use raw export HTML as a rendering path.

Re-flushing the same canonical message occurrence does not append a duplicate row. A new tool call that repeats identical content remains a distinct message occurrence with the same artifact ID. Never deduplicate transcript rows by artifact ID alone. Use the existing turn/call identity live, then reconcile to canonical `row_id` on history reload. If providers reuse call IDs across turns, do not match by call ID alone.

Compression can remove a tool row from the active model history. The artifact list/get APIs include compaction display rows and compression ancestors, so the artifact remains discoverable without repopulating model context. Rewound rows are not served. Explicit sibling sessions and unrelated profiles do not acquire access through a guessed ID. Deleted sessions return 404. There is no separate artifact retention policy; content follows canonical message retention/export/deletion.

## Live event contract

Subscribe through the current connection's existing event channel. No new event name or WebSocket endpoint is introduced. The gateway emits:

```json
{
  "jsonrpc":"2.0",
  "method":"event",
  "params":{
    "type":"tool.complete",
    "session_id":"runtime-session-id",
    "payload":{
      "tool_id":"provider-call-id",
      "name":"publish_html",
      "args":{"title":"Report","fallback":"Hello"},
      "inline_artifact":{
        "version":1,"id":"html_<64 lowercase hexadecimal characters>",
        "title":"Report","format":"html","fallback":"Hello","byte_length":14
      },
      "result":{"artifact":{"version":1,"id":"html_<64 lowercase hexadecimal characters>","title":"Report","format":"html","fallback":"Hello","byte_length":14},"text":"Hello"}
    }
  }
}
```

This is a schematic payload. Duration/summary/labels may also be present. `inline_artifact` is declared in the backend `ToolCompletePayload` contract. The event includes no artifact HTML in `inline_artifact`, `args` or `result`. It still emits when tool-progress chrome is disabled. The earlier `tool.start` may carry original tool arguments if ordinary progress is enabled. Ignore HTML from that event; it is not publication and has not crossed the durability fence.

Do not introduce a separate optimistic artifact message on tool start. Add the artifact to its existing tool row on successful completion. The event's session ID is a runtime routing ID, not necessarily the stored session ID. Resolve the stored conversation and owner profile from the focused tile/session state before fetching. On reconnect, reconcile from canonical history and the artifact list instead of relying on event replay alone.

A delegated child's `publish_html` belongs to the child conversation. It must not silently publish into the parent. Cold child history and child-scoped retrieval use that child's persisted metadata. Parent subagent progress relays are not artifact completion events. Live inline rendering in a child watch window has not been verified by this backend work.

## Authenticated retrieval

Implemented routes:

```text
GET /api/sessions/{stored_session_id}/artifacts?profile={owner_profile}
GET /api/sessions/{stored_session_id}/artifacts/{artifact_id}?profile={owner_profile}&row_id={canonical_row_id}
```

`profile` is optional and defaults to the server's current store. `row_id` is optional on get. When supplied, it must match the artifact's canonical occurrence. Without it, get returns the first visible matching occurrence in canonical display order. The HTML is identical for every occurrence with the same ID. Use the list/history association for placement, not the get route's default association.

List returns `{"artifacts":[{"artifact":ArtifactSummary,"association":Association}],"count":N}` in canonical display order. It includes each message occurrence, even when IDs repeat. There is no `has_more` or pagination in this initial implementation. The existing SessionDB safety bounds still apply; very large conversation reads are not a claimed unlimited archive API.

`Association` is `{"session_id":storedSourceSessionId,"row_id":integer,"tool_call_id":string}`. For compression ancestry, the association can point to the original source segment while the query names the active tip.

Get returns:

```text
{
  artifact: ArtifactSummary,
  html: original HTML string,
  render_html: CSP-prefixed HTML string,
  render_policy: {
    sandbox: "", referrer_policy: "no-referrer", scripts: false,
    network_resources: false, min_height: 120, default_height: 320, max_height: 800
  },
  association: Association
}
```

Both are JSON, with `Cache-Control: no-store` and `X-Content-Type-Options: nosniff`. No route serves `text/html`. There is no HTTP publish/update route. POST returns 405.

These routes call the existing `_require_token` boundary. Local/header mode uses `X-Hermes-Session-Token`. OAuth-gated mode uses the existing verified session cookie/native connection authentication. Use the host's current authenticated HTTP client; do not create an independent credential transport. A `token` query parameter does not authenticate these routes.

Containment is checked against the selected profile's canonical conversation. It is not a new per-user tenant ACL. Existing Hermes server authentication determines which profiles an authenticated operator can access. An artifact ID by itself is not an authorization capability.

Errors:

- 401 means missing/invalid authentication, or missing verified OAuth session. Keep the normal reconnect/login behavior.
- 404 means an absent artifact, absent session/profile, mismatched row association, rewound row or invalid artifact ID. Keep the fallback visible. Older servers also return 404 for the new routes; treat this as unsupported, not a reason to load a filesystem path.
- 400 can come from the existing invalid-profile-name validator. 422 covers invalid typed query values such as non-integer `row_id`.
- Storage/service failures use the existing backend error behavior. Show an error and an explicit retry action. Do not substitute cached content from another profile/connection.

## Rendering rules for Grok

1. Fetch JSON through the authenticated host client. Keep credentials exclusively in that host path. Never put a token, cookie, bearer credential or gateway endpoint into the iframe URL or HTML.
2. Create an iframe with **`sandbox=""`** and `referrerpolicy="no-referrer"`. Set `srcdoc` to `render_html`. Do not use `html` directly, `innerHTML`, `dangerouslySetInnerHTML`, `file://`, the generic file preview URL, or an unsandboxed same-origin HTML endpoint.
3. Do not add `allow-same-origin`, `allow-scripts`, `allow-forms`, `allow-popups`, `allow-top-navigation` or privileged bridges. This version supports static content only. An iframe lacks host SDK, RPC, Electron preload and authentication access.
4. The prefixed CSP is `default-src 'none'; script-src 'none'; style-src 'unsafe-inline'; img-src data:; font-src data:; connect-src 'none'; frame-src 'none'; object-src 'none'; base-uri 'none'; form-action 'none'`. Apply the iframe sandbox even if the author supplies conflicting HTML/meta tags. The restrictive policy and sandbox are independent protections.
5. `network_resources:false` describes CSP resource/fetch restrictions, not a browser-wide egress firewall. HTML can still attempt navigation inside its own sandboxed frame. Do not authorize external top-level navigation or popups. The inherited sandbox must remain in force after any frame navigation. Prefer a browser integration check that redirects stay isolated.
6. Use a 320px initial height, a 120px minimum and an 800px maximum, with scrolling for overflow. No script-based resizing or `postMessage` handshake is implemented. Do not accept size, URL or bridge commands from artifact content. A trusted host container may offer bounded manual resize.
7. Render titles/fallback text through normal escaped text components. Keep title, loading skeleton, error/retry and fallback outside the frame. Never interpret the title or fallback as directives/HTML.

Client state is `loading`, `ready`, `error` or `unsupported`; these are frontend states, not new backend events. Keep the original transcript position through transitions. Cancel stale reads when the tile/connection/profile changes. Cache keys must include connection, profile, stored session, artifact ID and occurrence row ID. Clear credential-dependent content on logout. Unknown format/version stays a text fallback.

Unsupported CLI/messaging clients receive the tool's `text` and `fallback`. No existing preview directive is required for publication. The model should quote the fallback in its final reply when the client cannot render artifacts. Frontend rendering, layout, sandbox enforcement and accessibility remain unverified here.

## Existing preview/deliverable support

The desktop already supports `::preview{file="path.html"}`, `MEDIA:` and `desktop_preview`. The transcript directive and preview card are file-addressed. The current desktop artifact inventory detects files/messages; it is not the immutable publication store implemented here. No desktop source files were modified.

Official deliverable-mode docs describe `.html`/`.htm` as file uploads on messaging platforms, not executable inline browser documents. Existing `/api/fs/*` reads resolve session/profile context and guard sensitive paths. The new API does not weaken those routes or rely on a mutable workspace HTML file surviving a restart.

## Delegation contracts, preserved

Use the existing `delegate_task` tool. Top-level model dispatch can become background automatically, while nested orchestrators wait for their own workers. The tool's schema/config determine concurrency and depth; this change does not alter those defaults.

Model actions:

- Spawn uses existing `goal`, `context`, `tasks`, `background`, grouping and role/model settings. Background dispatch returns `status:"dispatched"`, a `delegation_id` and child IDs where available. Dispatch acceptance is not task completion.
- `action:"list"` returns only owned live children, including `subagent_id`, goal, status, running time and `accepting_steer`.
- `action:"steer",subagent_id,message` returns `status:"queued"` when accepted. Delivery occurs at the next safe tool boundary. Finalization can report `missed_steer`; queued is not delivered.
- `action:"stop",subagent_id` returns `status:"interrupt_requested"`. It is cooperative cancellation. In-flight calls are asked to cancel, and the partial result still arrives as a completion. Do not show it as immediately cancelled/succeeded.

Gateway RPCs:

```text
subagent.list      {session_id: runtimeOwnerId}
  -> {subagents: [...], delegations: [recent failed durable delegations]}
subagent.steer     {session_id: runtimeOwnerId, subagent_id: childId, text: correction}
  -> {status: "queued" | "rejected", subagent_id, text}
subagent.interrupt {session_id: runtimeOwnerId, subagent_id: childId}
  -> {found: boolean, subagent_id}
subagent.tail      {session_id: runtimeOwnerId, subagent_id: childId}
  -> {subagent_id, available: boolean, text: string, truncated: boolean}
```

List can recover visibility through durable parent-session lineage after a renderer reconnect. Steer/interrupt/tail still require the exact live owner session generation and transport. A visible child after a fresh owner remint is not necessarily controllable by that new generation. Respect `rejected`, `found:false` and unavailable tail rather than broadening authority. Tail is bounded to 16384 bytes and is not a durable progress subscription.

Existing gateway events use the same `event` wrapper:

| Event | Frontend meaning |
| --- | --- |
| `subagent.start` | A child started; show running state. |
| `subagent.progress` | A batched tool activity preview, not a percentage. |
| `subagent.tool` | Child tool name/preview changed. |
| `subagent.thinking` | Liveness/reasoning update. Text obeys the session's reasoning visibility setting. |
| `subagent.complete` | Child terminal outcome, with status and summary when present. |

The payload starts with `goal`, `task_count` and `task_index`. Optional fields are `subagent_id`, `parent_id`, `child_session_id`, `delegation_id`, `depth`, `model`, `tool_count`, `toolsets`, token/API counts, files read/written, `output_tail`, `tool_name`, `text`, `status`, `summary` and `duration_seconds`. Do not invent a progress ratio from tool count. `subagent.text` feeds the child watch mirror and is deliberately not broadcast on the parent's event stream.

Map `running` to active. Map `completed` to finished, but inspect the actual child result for exit reason/schema validity before claiming the goal succeeded. `failed`, `error`, `timeout` and `stalled` require error presentation. `interrupted` means the cooperative stop completed. `unknown` means the owner disappeared before a terminal result was recorded, not cancelled or successful. A batch can complete while individual children failed; keep per-child outcomes. Intermediate `stalling` and `finalizing` remain active/busy states.

Live child agent objects/progress registries are process-local. Async dispatch/result/delivery records already persist in `state.db.async_delegations`. Recovery uses owner PID plus start-time fingerprints. Abandoned running/finalizing records become `unknown`; already-recorded child results, transcript tails and git-state hints are retained when available. Pending completions replay under ownership checks, delivery claims, an existing 48-hour replay-age limit and bounded delivery retry. Delivery across crash/restart is at least once, not a universal exactly-once guarantee.

There is no automatic rerun/restart of unfinished children and no new `subagent.restart` API in this patch. Explicitly re-delegate only after reviewing the recovered outcome/workspace state. An interrupted or unknown task may have already performed side effects. A fresh backend cannot honestly restore live progress from the old process. Keep historical terminal/recovery evidence distinct from the current live roster.

The separate public plugin `ctx.subagent_lifecycle` API already exists. It is not a second GUI task store and was not changed. Its uppercase lifecycle states are not interchangeable with the lowercase `delegate_task`/gateway status strings above.

Canonical chat admission remains unchanged. The tested admission/truncation ownership and busy checks use isolated state, not a live busy-owner session. Do not bypass admission by posting directly into an unrelated conversation. Completion reinjection is an existing serialized host turn and obeys ownership/queue behavior.

## Backend verification

Run from the worktree. The PM workflow used `setup-hermes.sh --runtime-only --test-environment` through the canonical runner, with no raw pip installs. This host needs `HERMES_RUNTIME_DIR=/home/mpotter2002/.hermes/tools` for bootstrap discovery. The resulting test environment uses Python 3.14.7 and pytest 9.1.1. `scripts/run-in-hermes-env python` selects runtime Python, not the test interpreter; use `scripts/run_tests.sh`.

```bash
HERMES_RUNTIME_DIR=/home/mpotter2002/.hermes/tools scripts/run_tests.sh \
  tests/tools/test_inline_artifacts.py \
  tests/agent/test_inline_artifact_persistence.py \
  tests/hermes_cli/test_inline_artifact_routes.py -j 2
```

The focused tests exercise real sequential/concurrent tool executors, real canonical SessionDB storage, a post-flush emission readback, `session.history` RPC replay, headless callbacks, failed-flush suppression, repeated message identity, authenticated FastAPI TestClient routes, session/profile containment, repeated-ID row selection, compaction/rewind behavior, malformed metadata, header auth, OAuth-cookie middleware and fresh-process recovery. No model provider is called. The OAuth IdP is the repository's test provider; the middleware and route boundaries are real.

Verified focused result is **22 passed, 0 failed** across the three artifact files. The nine-file persistence/model-metadata/delegation/child-history/file-route regression selection passed **179 tests, 0 failed**. The async-delegation recovery selection passed **33 tests**, with **one macOS-only skip**. The admission/history/authority selection passed **21 tests, 0 failed**. These are separate test invocations, not a claim about the complete suite.

Reproduce the recovery and admission selections with the same environment prefix:

```bash
scripts/run_tests.sh tests/tools/test_async_delegation.py -j 1 \
  -k 'not gateway_formatter_renders_async_block and not gateway_cli_origin_event_left_unrouted' \
  --file-timeout 120 -r s
scripts/run_tests.sh tests/tui_gateway/test_tui_gateway_server.py -j 1 \
  -k 'turn_admission or prompt_submit_truncation_signals_busy_instead_of_queueing or prompt_submit_truncation_refuses_redirect_of_live_turn or run_prompt_submit_binds_exact_steer_authority_and_resets_contextvars or history_to_messages or session_history_ships_durable_row_ids' \
  --file-timeout 90
scripts/run_tests.sh tests/agent/test_tool_call_incremental_persistence.py \
  tests/agent/test_model_metadata.py tests/tools/test_delegate_apiserver_background.py \
  tests/tools/test_delegate_parallel_interrupt_join.py tests/tools/test_delegate_timeout_cleanup.py \
  tests/tools/test_async_delegation_failed_surface.py tests/tui_gateway/test_subagent_child_mirror.py \
  tests/tui_gateway/test_tui_gateway_server_crash_history.py tests/hermes_cli/test_web_server_files.py \
  -j 3 --file-timeout 90
```

Relevant green regressions are recorded separately. The broad 18-file attempt was interrupted by the tool timeout and encountered the repository's real-home I/O guard. Shared worktree `.git` metadata and a PM parent `manifest.json` probe fall outside the test allowlist. Three representative failing tests were reproduced on the unchanged base in a separate baseline worktree. No guard was disabled and no unrelated test/source fix was committed. This is not a claim that the full suite is green.

All logs and the decision trail are beside the worktree under `/home/mpotter2002/.hermes/profiles/dr-eggbot/cache/scratch/`, with prefix `inline-artifacts-`. `inline-artifacts-focused.log`, `inline-artifacts-regression-green.log`, `inline-artifacts-delegation-recovery.log` and `inline-artifacts-admission.log` are the successful runs. `inline-artifacts-regression.log`, `inline-artifacts-edit-regression.log` and `inline-artifacts-baseline.log` retain the blockers. The RED tracer runs are retained too.

## Frontend end-to-end acceptance checklist

These checks are for Grok/engineer after wiring, not claimed completed by this backend patch.

- Publish HTML between two assistant text segments. Confirm one card at the exact tool occurrence, with unchanged text order and no duplicate optimistic bubble.
- Publish two different artifacts, then repeat the first in a new call. Confirm IDs/content behavior and distinct row associations.
- Refresh, detach/reconnect, reopen the stored conversation and restart an isolated development backend. Fetch content again using the originating connection/profile. Preserve placement and fallback.
- Compress the conversation. Reconcile from artifact list/history without reintroducing archived model context. Rewind past publication and confirm it is no longer served.
- Observe loading, auth expiry, unsupported route/version, 404 and storage failure. Keep text fallback and expose retry without moving transcript rows.
- Try script tags, inline handlers, fetch/WebSocket, form submission, nested frames, parent/top access, popups and frame navigation. Confirm sandbox/CSP prevent privileged access; verify no host/preload bridge or credential-bearing URL is available. Check DevTools for resource/navigation behavior rather than assuming the policy is enforced.
- Test oversized and multibyte HTML/title/fallback validation. Render titles as escaped text. Check keyboard/accessibility behavior and 120/320/800px sizing with overflow.
- Open another profile and another connection with the same session/artifact ID. Confirm query caching cannot reuse the wrong host/profile's document.
- Spawn parallel delegates, watch progress, steer and stop through the existing APIs. Distinguish queued steering and requested cancellation from terminal outcomes. Reconnect with a new owner generation and honor rejected controls.
- Kill only an isolated development owner, then reopen. Show durable `unknown`/recorded partial results, do not pretend the children restarted, and reconcile at-least-once completion replay without duplicate presentation.
- Keep unsupported CLI/messaging clients readable. Do not substitute unsandboxed file previews as an inline-artifact fallback.

## References inspected

- Official feature index: https://hermes-agent.nousresearch.com/docs/llms.txt
- Existing desktop directive/plugin contract: https://hermes-agent.nousresearch.com/docs/developer-guide/desktop-plugin-sdk
- Existing file delivery: https://hermes-agent.nousresearch.com/docs/user-guide/features/deliverable-mode
- Existing delegation: https://hermes-agent.nousresearch.com/docs/user-guide/features/delegation
- Existing public plugin lifecycle: https://hermes-agent.nousresearch.com/docs/developer-guide/subagent-lifecycle-api
- Hermes source mechanisms: `agent/tool_executor.py::_commit_tool_result`, `hermes_state_messages.py`, `tui_gateway/session_history.py`, `tui_gateway/tool_progress.py`, `tui_gateway/methods_subagents.py`, `tools/delegate_tool_registry.py`, `tools/async_delegation.py` and the read-only desktop preview/directive source.
- T3 Code source reference at `805967a878e6804d58c151f29d2a3d0a06828fd1`: [architecture](https://github.com/pingdotgg/t3code/blob/805967a878e6804d58c151f29d2a3d0a06828fd1/docs/internals/overview.md), [EventSink](https://github.com/pingdotgg/t3code/blob/805967a878e6804d58c151f29d2a3d0a06828fd1/apps/server/src/orchestration-v2/EventSink.ts#L232-L260) and [command receipts](https://github.com/pingdotgg/t3code/blob/805967a878e6804d58c151f29d2a3d0a06828fd1/apps/server/src/orchestration-v2/Orchestrator.ts#L242-L253). Direct source reads verified commit-then-publish, receipt-backed idempotency and thread-scoped receipt replay. The cached extractor initially showed an older `OrchestrationEngine.ts` layout; direct GitHub retrieval corrected it to the current v2 sources. Hermes reuses its own durability fence and message identity, not T3's Effect/outbox implementation. No T3 artifact renderer or sandbox implementation is assumed or copied.
