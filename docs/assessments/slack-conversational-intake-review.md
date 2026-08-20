# Slack “Send to Hermes” conversational intake

Status: **REVIEW REQUIRED — implementation is blocked on Jerry’s go/no-go**
Assessment task: `t_7ce8f15c`
Target repository/worktree inspected: `/Users/jj/.hermes/hermes-agent/.worktrees/t_3e6da870`
Scope of this document: product and architecture decision; no production implementation

## Decision summary

**Recommendation: proceed with a private Slack DM-thread intake whose canonical promotion control is a persistent `Create card` button.** The selected Slack message must enter Hermes as the first ordinary user turn through the normal gateway prompt path. No Kanban card exists until an explicit promotion command passes completeness, authorization, privacy, and idempotency gates.

The complete product direction permits two exact standalone language conveniences—`make the card` and `queue this`—that invoke the same promotion command. `go`, casual agreement, reactions, and inferred assent never mutate Kanban. For the first private dogfood slice, implement **button-only promotion**, text-only source capture, one authorized user/workspace/profile, a direct 1:1 DM, and a fixed **HRMS → Triage, unassigned** destination. Add the exact language conveniences only after the end-to-end route, restart recovery, and exactly-once card proof pass.

This reconciles the desired conversational experience with the supported architecture:

- Slack DM threads already provide the correct conversation surface and distinct Hermes routing identity.
- Existing shortcut capture and privacy behavior are valuable and should be reused.
- The existing immediate `victorylive` Kanban mutation is the wrong endpoint and should be replaced.
- Generic CLI handoff orchestration is not a safe shortcut because its `chat_type="thread"` source does not match organic Slack DM replies (`chat_type="dm"`).
- A small gateway-owned intake operation is required to create the thread, bind the organic-equivalent Hermes session, and submit one normal `MessageEvent`.
- Exactly-once promotion requires a backend-enforced transactional uniqueness rule; an in-process lock plus preflight lookup is insufficient.

## Go/no-go recommendation

**Go**, provided Jerry approves these four product choices:

1. Private dogfood writes only to **HRMS → Triage, unassigned**.
2. The dogfood source is explicitly **text-only**; files, rich blocks, unfurls, and edited-source refresh are deferred.
3. The first slice uses **button-only promotion**; the two exact standalone phrases are the approved near-term follow-up, not part of the initial proof.
4. Re-invoking the shortcut deliberately creates a new intake, even for the same source message; Slack redelivery/retry of one invocation resumes the original intake.

A no-go is appropriate if any of those choices is unacceptable without multi-board selection, rich-source canonical fetching, or broad natural-language promotion in the first release. Those additions materially enlarge the safety and recovery surface and should not be smuggled into the minimal slice.

## Product contract

### User-visible lifecycle

1. **Invoke shortcut** — The user selects `More actions → Send to Hermes` on a Slack message. Slack is acknowledged immediately. Authorization happens before source text is rendered, cached, logged, or submitted.
2. **Reserve intake** — Hermes mints or reloads a durable intake record for this shortcut invocation. This is not a Kanban card.
3. **Start private thread** — Hermes opens the invoking user’s 1:1 DM in the correct Slack workspace and posts one thread seed. The thread is dedicated to this intake.
4. **Bind Hermes session** — The gateway builds an organic-equivalent Slack DM `SessionSource`, resolves/creates the Hermes session, and durably binds intake, Slack source, DM thread, profile, session key, and session lineage.
5. **Submit the source normally** — The captured source and optional context are submitted once as an ordinary human-originated `MessageEvent`. Hermes’s visible kickoff is the normal assistant response from that turn—not an outbound-only imitation.
6. **Refine** — Hermes states its understanding, recommendation, assumptions, `No card exists yet`, and at most one material question. The user refines the brief in the same Slack thread. No Kanban mutation occurs.
7. **Request promotion** — The current `Create card` button enters promotion evaluation. In the complete target behavior, exact standalone `make the card` and `queue this` do the same. Other plausible language only opens an explicit confirmation.
8. **Complete or clarify** — If outcome, destination, privacy boundary, or execution posture is materially unresolved, Hermes previews the known draft and asks one blocking question. No card is created.
9. **Create** — Controls move to `Creating card…`; promotion reauthorizes and submits one resolved brief under the intake’s immutable promotion key.
10. **Confirm success** — The thread shows the card ID and `Open card`. Stale actions, retries, concurrent requests, and exact-phrase/button races return the same card.
11. **Recover failure** — Definite failures preserve the intake and show `Retry creation`. Unknown outcomes reconcile by promotion key before any write is retried.

### State model

The durable state belongs to the intake record, not to one Slack message rendering or a process-local modal cache.

| State | Required durable identity | Allowed transitions | Kanban write? |
|---|---|---|---|
| `reserved` | intake, source, owner, workspace/profile | `thread_bound`, `failed_retryable` | No |
| `thread_bound` | plus DM channel/thread timestamp | `session_bound`, `failed_retryable`, `reconciling` | No |
| `session_bound` | plus session key and logical session lineage | `dispatching`, `failed_retryable` | No |
| `dispatching` | plus stable synthetic event ID | `active`, `failed_retryable`, `reconciling` | No |
| `active` | accepted first turn; resolved brief revision | `active`, `clarifying`, `promotion_pending` | No |
| `clarifying` | pending material question + draft revision | `active`, `clarifying`, `promotion_pending` | No |
| `promotion_pending` | frozen candidate revision + promotion key | `creating`, `active`, `clarifying` | No |
| `creating` | immutable promotion key and candidate hash/revision | `promoted`, `failed_retryable`, `reconciling` | At most one logical write |
| `reconciling` | last operation and external identifiers | previous safe state or `promoted` | Lookup first; no blind write |
| `failed_retryable` | safe failure code and last authoritative state | retry to the recorded step | Only if retrying the same promotion key |
| `promoted` | board + card ID + promoted timestamp | remains `promoted` | Never again |
| `closed` | close reason | optional explicit new intake, never implicit reopen | No |

State writes use a monotonic revision or equivalent compare-and-swap. Readers must never observe `active` without a session binding or `promoted` without board and card ID. Slack UI messages are projections of this record and may be stale; every action resolves current durable state before acting.

## Mutation-intent policy

### Events allowed to reach promotion evaluation

- Current `Create card` button scoped to the intake ID.
- `Confirm create` after a material field was inferred or changed during promotion.
- `Retry creation` / `Check & retry`, always with the existing promotion key.
- After dogfood: exact normalized standalone `make the card` or `queue this`, allowing terminal punctuation only.

### Events that can never directly write

- `go`, `yes`, `yeah`, `do it`, `ship it`, `send it`, `create that`, `looks good`, `that makes sense`, `sounds right`.
- Emoji, reactions, approval adjectives, or silence.
- Text containing an allowed phrase as part of a larger utterance, quotation, question, or UI-label discussion.
- Any stale Slack action whose intake owner, workspace, authorization, draft revision, or existing card state cannot be validated.

Ambiguous language receives: `Do you want me to create the HRMS → Triage card now?` with `Create card` and `Keep refining`. That confirmation UI still performs zero writes until the button is pressed.

### Completeness gate

Promotion requires a resolved outcome, board, privacy-compatible source policy, and execution posture. Dogfood fixes board/status to HRMS → Triage, unassigned. If the user contradicts that default, promotion pauses rather than silently choosing. Hermes asks no more than one material question per turn and does not ask for values it can safely derive, such as title wording or acceptance-criteria prose.

## Supported architecture

### Required gateway-owned seam

Add one supported operation, with the name illustrative rather than prescriptive:

```python
async def start_platform_intake(
    request: PlatformIntakeRequest,
) -> PlatformIntakeBinding:
    ...
```

The operation owns the smallest atomic application boundary that can be made durable:

1. Revalidate submitter, Slack workspace, served profile, and source authorization.
2. Reserve or reload an intake by a stable invocation/delivery identity.
3. Resolve the submitter’s 1:1 DM using the captured workspace’s Slack client.
4. Post or reconcile exactly one thread seed and persist its timestamp.
5. Build `SessionSource(platform=SLACK, chat_type="dm", scope_id=team_id, chat_id=dm_channel_id, thread_id=seed_ts, user_id=submitter_id, profile=resolved_profile)`.
6. Resolve/create the session through `SessionStore.get_or_create_session()` and persist peer identity through the existing peer path.
7. Persist the intake → route/session binding before dispatch.
8. Submit one ordinary, non-internal `MessageEvent` through `BasePlatformAdapter.handle_message()` / the normal gateway runner path, with a stable synthetic event ID and the real user identity.
9. Return a durable receipt containing intake ID, route/session identity, DM channel/thread, and accepted/retryable state.

The boundary belongs in the gateway because profile routing, session storage, active-turn coordination, prompt ingress, and lifecycle hooks live there. Slack adapter code alone cannot safely return a session binding or reconcile asynchronous acceptance.

### Paths explicitly prohibited

- Direct transcript row insertion.
- Direct `AIAgent.run_conversation()` invocation.
- Posting a fabricated assistant kickoff before a corresponding accepted user turn.
- Reusing generic CLI handoff’s `chat_type="thread"` for this Slack DM flow.
- Letting Slack action values or `private_metadata` carry raw source text or mutation payloads.
- Creating a Kanban card directly from modal submission.

### Normal prompt path that must remain intact

`Slack intake operation → organic-equivalent MessageEvent → BasePlatformAdapter.handle_message() → GatewayRunner._handle_message() → SessionStore/get_or-create + peer persistence → active-turn/lease/history/cache handling → agent.run_conversation() → normal transcript finalization → Slack delivery to the bound DM thread`

This preserves authorization, profile routing, active-session serialization, prompt role alternation, cached-agent consistency, streaming/delivery behavior, and session lifecycle hooks.

## Reuse versus replacement of the existing worktree

### Reuse

From `/Users/jj/.hermes/hermes-agent/.worktrees/t_3e6da870`:

- Slack message shortcut manifest identity and Bolt callback registration.
- Immediate acknowledgement before API work.
- Authorization before source disclosure and repeated submit authorization.
- Canonical Slack source identity `(team_id, channel_id, message_ts)`.
- Source author, submitter, channel/workspace metadata capture.
- Best-effort permalink lookup with honest unavailable state.
- Opaque nonce in Slack metadata and server-side source custody.
- Private modal/status rendering, bounded preview/context, and safe error copy.
- Workspace-scoped DM resolution primitive.
- Thread-seed posting primitive, but not generic handoff source construction.
- Existing `MessageEvent`/gateway/session path.
- Existing Kanban idempotency vocabulary as a starting point, after backend uniqueness is made structural.

### Replace

- `Create task` modal semantics and immediate modal-to-Kanban flow.
- Fixed `victorylive` destination.
- `build_task_payload()` raw source + freeform context mapping.
- Loopback dashboard-token `HermesIntakeClient.create_task()` as shortcut submit’s terminal action.
- Source-message-only idempotency.
- Success trapped in a one-shot modal.
- Process-local nonce/source map as the authoritative intake lifecycle.

### Keep only at explicit promotion

- Kanban creation API/tool path.
- Private success/failure messaging.
- Idempotent create behavior, upgraded to transactional uniqueness around the durable promotion key.

## Durable lineage model

Use an opaque immutable `intake_id` as the logical anchor. Suggested minimum record:

- schema version, intake kind, intake ID, monotonic revision;
- resolved profile;
- Slack team/workspace ID, source channel ID, source message timestamp;
- source permalink or explicit unavailable status, capture timestamp;
- authorized submitter/owner ID;
- Slack shortcut delivery/invocation identity used to collapse callback retries;
- destination 1:1 DM channel ID and thread timestamp;
- Hermes session key;
- initial physical session ID and current/resolved session ID or compression-tip pointer;
- stable first-turn synthetic event/message ID and dispatch status;
- current lifecycle state and safe failure code;
- current resolved-brief revision/hash, without duplicating raw transcript text;
- immutable promotion key;
- nullable card board and card ID;
- created, updated, and promoted timestamps.

### Bidirectional resolution requirements

- From `intake_id`: source, DM thread, logical Hermes session, state, and card.
- From card lineage: intake ID, source/permalink, DM thread, and Hermes session.
- From session/peer binding: Slack DM/thread and intake after restart.
- From Slack action value: opaque intake ID only, then server-side current-state lookup.

### Session rotation/compression

Treat `intake_id + session_key` as logical lineage; do not make one physical session ID the sole anchor. Store the initial session ID for provenance and resolve/update the current session through the repository’s compression/rotation tip policy. Promotion and `Open card` must follow logical lineage, not a stale physical ID.

### Card lineage block

The synthesized card contains identifiers and links, not the full Slack transcript:

- Source Slack permalink when available, plus workspace/channel/message identity and capture time.
- Intake ID.
- Hermes logical session ID/key reference suitable for the supported UI.
- Slack DM channel/thread timestamp or safe deep link.
- Promotion key or stable lineage reference.
- Resulting board/card identity is stored back on the intake record.

## Authorization and privacy rules

1. Acknowledge Slack quickly, then authorize before source text is rendered, persisted, logged, or sent elsewhere.
2. Reauthorize at intake submit/prompt ingress, promotion, and `Open card`; authorization may change during a long intake.
3. The invoking user must own the nonce/intake. Copied, stale, or cross-user controls fail closed.
4. Slack team/workspace scopes user IDs, channel IDs, DM cache, clients/tokens, source identity, event dedupe, and session identity. No primary-client fallback may cross workspaces.
5. Resolve profile before session creation. An unavailable explicit profile is an error, not a default-profile fallback.
6. Intake occurs only in the invoking user’s 1:1 DM. Never fall back to the source channel or a shared home channel.
7. Slack action metadata contains opaque IDs only. Raw source text stays in the authorized server-side source/transcript path.
8. The lineage table stores identity/status, not a second raw-content copy.
9. Permalink failure is represented honestly and does not broaden source access.
10. `Private` describes refinement, not the resulting card. Copy must state that promotion sends the resolved brief and source attribution to the selected board.
11. If source visibility and board visibility conflict materially, promotion blocks or requires a safe authorized destination.
12. Card content is a synthesized brief; raw chat transcript is excluded.

## Idempotency rules

There are two identities with different semantics:

1. **Invocation/delivery identity** collapses Slack redelivery or retry of one shortcut callback into one intake/thread/session.
2. **Intake identity** is newly minted for each deliberate shortcut invocation, even when the selected source message is the same. This preserves the user’s ability to start a fresh interpretation intentionally.

Use one immutable promotion key per intake, for example:

`slack-intake:{profile}:{team_id}:{dm_channel_id}:{thread_ts}:{intake_id}`

Required structural guarantees:

- Unique invocation delivery key for active/replayed callback handling.
- Unique intake ID.
- Unique promotion key in the intake store.
- Transactional backend uniqueness mapping `(board, promotion_key) → card_id` or an equivalent unique idempotency ledger.
- Unique non-null lineage mapping from one intake to one resulting card.

The current process lock and preflight `SELECT` are not sufficient. Two processes can both pass the lookup and insert. The implementation must prove concurrency against the real persistence boundary, not only a stateful fake.

## Failure and retry behavior

Slack and local persistence cannot form one transaction, so recovery states are product behavior, not an implementation detail.

| Failure point | Authoritative result | Retry behavior |
|---|---|---|
| Authorization/profile resolution | Nothing disclosed or created | Fail closed; user may retry after access is restored |
| DM open | Intake reserved; no thread/session | Retry same intake and workspace client |
| Seed post definite failure | No thread/session | Retry same intake |
| Seed posted, local write uncertain | Slack thread may exist | Reconcile by recorded request/response evidence before posting again |
| Session bind failure | Thread exists | Reuse thread; resolve/create same organic-equivalent route |
| Prompt dispatch rejected | Bound route exists; no accepted first turn | Retry same stable event ID and route |
| Prompt accepted, model fails | Session/transcript is authoritative | Resume normal conversation; do not replay first turn blindly |
| Assistant persisted, Slack delivery fails | Transcript is authoritative | Redeliver/recover output; do not rerun first turn |
| Promotion definite failure before commit | Intake/draft remains active | Retry same promotion key |
| Promotion commit, acknowledgment lost | Card may exist | Lookup/reconcile by promotion key before any retry |
| Slack success update fails | Card is authoritative | Re-render `Already created` + `Open card` from intake state |
| Gateway restart | Durable binding is authoritative | Reload state and continue exact recorded step |

Errors shown to Slack remain private, safe, and machine-classified internally. Never erase an authoritative earlier step to simulate rollback. Retry controls are disabled while an operation is in flight and stale actions resolve the current state.

## Safety invariants and test obligations

A future implementation is not acceptable without executable proof of these invariants:

1. Shortcut invocation creates zero Kanban cards.
2. Unauthorized input never reveals or persists selected source text.
3. The destination is the invoking user’s 1:1 DM in the correct workspace.
4. Retry/restart creates at most one thread and one first user turn per intake.
5. The synthetic intake source key exactly equals the next organic Slack DM-thread reply key, including profile, workspace, `chat_type="dm"`, DM channel, and thread timestamp.
6. The first turn goes through normal gateway authorization, session, active-turn, lease, history, cache, and finalization paths.
7. No direct transcript append, direct agent-loop call, or outbound-only kickoff occurs.
8. The second organic reply sees exactly one intake user turn and one assistant turn.
9. Casual agreement, reactions, and ambiguous phrases produce zero Kanban writes.
10. Incomplete or privacy-conflicted promotion asks one question and preserves the draft.
11. Button, stale button, retry, concurrent callers, restart, and unknown acknowledgment converge on one card ID.
12. Transactional uniqueness is tested across two independent creators/process connections.
13. Success replaces the action with `Open card`; inability to open never changes promoted state.
14. The card has outcome, rationale, scope, non-goals, acceptance criteria, decisions, constraints, unresolved questions, source, session/thread, and intake lineage—but no raw transcript dump.
15. Restart can resolve intake → source → DM thread → logical Hermes session → card in both directions.

## Smallest end-to-end private-dogfood slice

### Boundary

Include only:

- one explicitly authorized Jerry Slack identity;
- one Slack workspace and one served Hermes profile;
- text-only selected messages;
- one deliberate invocation → one intake;
- direct 1:1 DM and one dedicated thread;
- durable intake/thread/session binding;
- one source submission through the normal prompt path;
- conversational refinement using ordinary Slack replies;
- persistent **button-only** `Create card`;
- fixed **HRMS → Triage, unassigned** destination;
- a synthesized card with durable lineage;
- backend-transactional exactly-once promotion;
- `Open card`, safe retry, reconciliation, and restart recovery.

Defer exact phrase promotion, attachments/rich blocks/unfurls, multi-board selection, multiple users/workspaces/profiles, source refresh/edit semantics, explicit `start over`, broad rollout controls, analytics, and polished deep-link fallbacks.

### Setup

1. Use an isolated local/dogfood gateway connected to the intended Slack app/workspace and a non-production HRMS board or clearly designated private dogfood lane.
2. Restrict authorization to Jerry’s Slack identity.
3. Install/update the shortcut and Block Kit action manifest for the dogfood app.
4. Apply the durable intake storage and transactional promotion uniqueness migration with a documented rollback.
5. Enable the feature behind a single config/feature flag defaulting off.
6. Ensure logs expose correlation IDs only: intake ID, session key/ID, Slack team/channel/thread IDs, promotion key, card ID, state transition, and safe failure code—never raw private source text.
7. Prepare one text-only source message whose expected resolved outcome is unambiguous and safe for HRMS Triage.

### Primary live scenario

1. Invoke `Send to Hermes` on the prepared message.
2. Verify the private DM thread appears and no Kanban card exists.
3. Verify Hermes’s first response contains understanding, recommendation, assumptions, a no-card notice, source attribution, and at most one material question.
4. Reply naturally in the thread and confirm the same Hermes session continues.
5. Send `yeah, that makes sense`; confirm no card appears.
6. Click `Create card` once, then exercise a stale/double click or concurrent retry.
7. Verify exactly one HRMS Triage card appears with the resolved brief and all lineage fields.
8. Verify Slack shows the same card ID and `Open card`.
9. Restart the gateway; click the stale action or retry path again and verify `Already created` returns the same ID.
10. Open the card and follow lineage back to the intake/thread/session.

### Failure injection scenario

At minimum, inject:

- restart after thread seed but before first-turn dispatch;
- first-turn dispatch acceptance with Slack response delivery failure;
- Kanban commit with acknowledgment/Slack update failure;
- two concurrent promotion calls using separate backend connections.

Each case must converge without a second thread, first turn, session route, or card.

### Observability

Capture a timestamped trace keyed by `intake_id` with:

- each state transition and revision;
- authorization decision category without private content;
- Slack workspace, DM channel, and thread timestamp;
- computed session key and resolved logical/physical session ID;
- first-turn synthetic event ID and accepted/completed/delivered markers;
- promotion key, create/reconcile operation, and resulting card ID;
- retry count and safe failure code;
- card lookup proving one result.

Preserve the focused automated test output and a redacted live trace as review evidence. Record before/after card counts and query the promotion key directly after concurrency/failure tests.

### Rollback

- Disable the feature flag, removing the shortcut’s conversational submit path without deleting durable records.
- Leave existing promoted cards and lineage intact; never “roll back” a committed card by allowing repromotion.
- Revert the Slack manifest/action surface if needed.
- Stop new intake creation while retaining read/reconcile support for in-flight/promoted records.
- Roll back schema only if records are empty; otherwise retain the additive table/columns until a deliberate migration archives them.
- The previous immediate-card flow may remain available on its branch for comparison, but do **not** automatically restore it in production because it violates the approved product contract.

### Pass criteria

All of the following must hold:

- One invocation creates one private DM thread, one Hermes route/session, one accepted first turn, and zero cards before promotion.
- The next organic reply continues the exact same session and sees correct history once.
- Casual assent produces zero writes.
- One explicit button promotion produces exactly one HRMS Triage card with correct content and bidirectional lineage.
- Double click, concurrent promotion, unknown outcome, and restart return the same card ID.
- No raw private text appears in action payloads, lineage rows, or logs.
- Every failure injection has a deterministic, safe retry/reconcile result.
- The feature flag rollback stops new intakes without corrupting existing lineage or cards.

### Fail criteria

Any duplicate thread, session route, first user turn, or card; any source disclosure before authorization; any source-shape mismatch with an organic reply; any direct transcript/agent bypass; any casual-language mutation; any unreconciled unknown outcome; any lineage break after restart/compression; or any raw content leak is a stop-ship failure.

## Proposed implementation card — do not dispatch before approval

### Title

Implement the private Slack conversational-intake dogfood slice

### Outcome

An authorized user can invoke Slack `Send to Hermes`, refine the selected text in a dedicated private Hermes DM thread, and explicitly create exactly one context-rich HRMS Triage card through a persistent button, with durable source/session/thread/card lineage and restart-safe recovery.

### Rationale

The current shortcut safely captures source context but immediately mutates Kanban. The desired product is idea-to-execution intake: Hermes must reason with the user first, use Slack’s natural private thread surface, and promote only under unmistakable intent. The gateway already owns the ordinary prompt/session path; a supported intake seam and structural promotion idempotency are the missing pieces.

### Scope

- Reuse the existing shortcut registration, fast ack, authorization ordering, nonce/source capture, source attribution, permalink fallback, bounds, and private errors.
- Add additive durable intake storage with revisioned lifecycle, invocation dedupe, logical session lineage, promotion key, and card binding.
- Add the gateway-owned platform-intake start/reconcile operation.
- Build an organic-equivalent Slack DM source and submit one ordinary `MessageEvent` with real user identity and stable event ID.
- Render/update the dedicated DM-thread intake UI with `No card exists yet`, one-question refinement, persistent button states, and `Open card`.
- Add button-only promotion to fixed HRMS → Triage, unassigned.
- Synthesize the resolved card shape and lineage block; never dump the raw transcript.
- Enforce transactional/unique backend promotion idempotency.
- Implement private safe failure/reconcile/retry and feature-flag rollback.
- Add focused unit, integration, concurrency, restart, and live-dogfood evidence.

### Non-goals

- Natural-language promotion, including the approved exact phrases.
- `go` or broad intent classification.
- Rich blocks, files, attachments, unfurls, or canonical source refetch.
- Multi-board/status/assignee/workspace selection.
- Multi-user, multi-workspace, or broad rollout.
- Starting multiple versions from one intake; a new deliberate shortcut invocation creates a new intake.
- Copying Slack transcripts into cards or lineage storage.
- Reusing generic CLI handoff source shape.
- Production rollout or migration of existing immediate-created cards.

### Acceptance criteria

1. Shortcut ack and authorization ordering are preserved; unauthorized source text never leaves the authorized capture path.
2. Each deliberate invocation creates/resumes exactly one durable intake, private DM thread, organic-equivalent route, Hermes session binding, and first ordinary user turn; callback retries create none of those twice.
3. The initial assistant response is normal model output in the bound thread and contains understanding, recommendation, assumptions, `No card exists yet`, source attribution, and at most one material question.
4. The next organic Slack reply computes the same routing key and continues the same session with no duplicated history.
5. No Kanban call occurs before the current `Create card` button is activated and completeness/privacy gates pass.
6. Casual agreement and all text input produce zero writes in this slice.
7. Promotion creates one synthesized HRMS Triage card, unassigned, containing outcome, why, scope, non-goals, acceptance criteria, decisions, constraints, unresolved questions, source permalink/identity, intake ID, Hermes session lineage, and Slack DM/thread lineage; raw transcript is absent.
8. Backend uniqueness—not an in-process lock or preflight lookup—proves two independent concurrent promotions return one card ID.
9. Definite failure, unknown commit acknowledgment, stale action, repeated click, and gateway restart reconcile using the same promotion key and return the same card.
10. Success renders `Open card`; open failure leaves promoted state unchanged and exposes a copyable card ID.
11. Feature flag off prevents new intakes while existing records remain readable/reconcilable.
12. No raw source text appears in Slack action metadata, durable lineage rows, or logs.
13. All safety invariants in this assessment have focused automated tests, and the private live scenario plus required failure injections pass with a redacted trace.

### Dependencies

- Jerry’s explicit approval of this assessment and the four dogfood decisions.
- Access to the existing shortcut worktree as implementation input; its uncommitted changes must be preserved and reviewed rather than overwritten.
- A dogfood Slack app/workspace and authorized Jerry identity.
- An explicitly safe HRMS Triage dogfood destination.
- A reviewable additive storage migration and supported Kanban uniqueness mechanism.

### Testing

Automated:

- Existing shortcut/manifest suite remains green (`46 passed` was the audit baseline).
- Source capture/authorization/privacy unit tests.
- Routing key equality regression against a real organic Slack DM-thread event.
- Full production-seam integration from intake start through real gateway handler/session store to assistant delivery spy.
- Cached-session second-turn and restart rehydration tests.
- Crash/failure tests at each durable transition.
- Two-connection/process promotion uniqueness test against the real persistence implementation.
- Card synthesis/lineage and raw-transcript exclusion assertions.
- Feature-flag and rollback behavior tests.

Live:

- Execute the primary and failure-injection dogfood scenarios above.
- Preserve redacted state-transition evidence, exact commands/results, before/after card counts, one resulting card ID, and proof that its promotion key resolves uniquely.

### Review/delivery gate

Use an isolated worktree and normal PR/review lane. Implementation is not complete at code/test success: an independent reviewer must verify the exact head, migration/idempotency behavior, ordinary-prompt-path proof, and redacted live dogfood evidence. Jerry retains go/no-go and rollout authority.

### Follow-up hardening after the slice passes

- Add exact standalone `make the card` / `queue this` promotion through the same command and tests; keep all other language confirmation-only.
- Canonical rich-message/files/blocks ingestion with explicit privacy and size policy.
- Multiple authorized users/workspaces/profiles and routing isolation matrix.
- Board/status selection and visibility-aware destination policy.
- Explicit `start over` / multiple-intake version UX.
- Session compression/rotation stress testing and retention/deletion policy.
- Action update/deep-link polish, analytics, rate limits, abuse controls, and operational dashboards.
- Broad rollout, migration strategy, and user documentation.

## Unresolved decisions for Jerry

1. Approve or change the private dogfood destination: **HRMS → Triage, unassigned** is recommended; the existing branch’s fixed `victorylive` destination should not survive by accident.
2. Confirm text-only source is acceptable for the first live proof.
3. Confirm button-only dogfood, with exact standalone language promotion deferred until the base proof passes.
4. Confirm deliberate re-invocation creates a distinct intake while callback redelivery resumes the original.
5. Confirm whether the thread seed may be minimal system copy (`Starting a private Hermes intake…`) before model output, provided it contains no fabricated assistant analysis and the first substantive response remains normal Hermes output.
6. Confirm retention/deletion expectations for private intake lineage and abandoned drafts before broad rollout; dogfood may retain them under existing Hermes local data controls.

None of these should be silently selected by an implementation worker. Approval of this document should explicitly answer or accept the recommended defaults.

## Risk register

| Risk | Impact | Mitigation / gate |
|---|---|---|
| Generic handoff source mismatch | Next Slack reply opens a different session | Dedicated organic-equivalent DM source builder; key-equality regression test |
| Process-local intake custody | Restart loses source/thread/card reconciliation | Additive durable intake state before dispatch |
| Slack seed/local-write split brain | Duplicate or orphan thread | Reserve first; explicit reconciling state; stable Slack request evidence |
| Async prompt acceptance ambiguity | Duplicate first turn or false success | Gateway receipt + stable event ID + durable accepted/completed markers |
| Kanban preflight race | Duplicate cards under concurrent promotion | Transactional unique promotion mapping proven with independent connections |
| Casual language misclassified | Unauthorized mutation | Button-only dogfood; later exact standalone allowlist; ambiguous confirmation |
| Cross-workspace/profile fallback | Privacy breach or wrong session | Scope every identity/client/cache/key; fail closed on unavailable route |
| Raw source copied into metadata/logs | Private-content leak | Opaque action IDs, identifiers-only lineage/logging, explicit leak tests |
| Physical session ID becomes stale | Promotion/card loses conversation lineage | Logical intake/session key anchor + compression-tip resolution |
| Existing worktree has uncommitted code | Accidental overwrite or misleading provenance | Implement in isolated follow-up worktree/branch after reviewing and preserving the existing diff |
| Dogfood quietly expands to rich/multi-board | Safety surface grows before core proof | Enforce non-goals and require a new reviewed scope decision |
| Rollback deletes authoritative lineage | Repromotion or broken card traceability | Feature flag stops new writes; retain records and committed cards |

## Evidence used

- Product recommendation from `t_11db995e`: `slack-conversational-intake-recommendation.md`.
- Repository/seam audit from `t_7cfbdd42`: `slack-intake-seam-audit.md`.
- Existing implementation worktree: `/Users/jj/.hermes/hermes-agent/.worktrees/t_3e6da870`.
- Audit baseline: `uv run --with pytest --with pytest-asyncio pytest -q tests/gateway/test_slack_message_shortcut.py tests/hermes_cli/test_slack_cli.py` → **46 passed in 17.88s**.

The earlier referenced `victorylive/assessments/12-slack-to-hermes-message-shortcut.md` and root card `t_3317bf28` were not available in the inspected worktree/visible board evidence. This assessment therefore does not claim to preserve decisions that exist only there. If recovered before approval, compare it for contradictory product constraints; do not delay review merely for duplicated implementation history.

## Approval status

**Blocked on Jerry’s review. No implementation card has been created or dispatched, and no production code was changed by this assessment task.**
