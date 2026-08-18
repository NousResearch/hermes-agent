# Issue 83729: Desktop Stored Session Resume

**Issue:** [#83729](https://github.com/NousResearch/hermes-agent/issues/83729)

## Problem

Opening a persisted Desktop session can leave the thread blank even though its
REST transcript is available. Desktop starts `getLatestSessionMessages()` and
`session.resume` concurrently, but the cold path does not publish the REST
result until the RPC settles. A delayed RPC therefore holds usable persisted
history behind the loading state.

The failure is amplified by two independent mistakes:

- When `messages_omitted` is true and the REST prefetch is unavailable, the
  client still reconciles `resumed.messages`. The gateway deliberately returns
  `messages: []` in that case, so the client turns "not carried in this payload"
  into an authoritative empty transcript.
- The empty-transcript latch consults only the renderer's cached sidebar row.
  When that row is missing or stale, it ignores the resume response's
  `message_count` and accepts the fabricated empty transcript.

The stored session in #83729 is valid. Its durable ID and database history are
separate from the process-local runtime ID that `session.resume` binds.

## Existing Constraints

Commit `73c7f68456` fixed #69649 by removing eager REST publication before the
runtime projection rebuild. That change prevents duplicate inflight user rows
and keeps large transcripts from being built three times. This fix deliberately
reintroduces an early publication point, so it must preserve the constraints of
that change rather than simply revert it.

`apps/desktop/e2e/large-session-resume.spec.ts` is the performance contract:

- an unchanged cold resume builds the transcript once;
- a cold resume with a live projection, an identity correction, or a recovered
  local journal tail may build the persisted base and then one corrected result,
  for at most two paint bursts;
- a third paint is the old eager-prefetch regression.

All existing request-generation and selected-session guards remain mandatory.
An older async result must never overwrite a newer foreground selection.

## Goals

- Publish a valid REST transcript as soon as it arrives, without waiting for
  `session.resume`.
- Never treat `messages_omitted: true` plus `messages: []` as an empty
  authoritative transcript.
- Preserve same-session optimistic user rows and pending assistant rows.
- Merge rows that arrive after early REST publication instead of reverting to
  the prefetch-time snapshot.
- Reuse the published message array when the RPC adds no live projection.
- Enter the existing bounded retry and explicit error state when available
  metadata says history exists but no transcript can be painted.
- Preserve the #69649 large-session paint budget.

## Chosen Design

### Independent REST Publication

The REST prefetch and `session.resume` remain concurrent. The prefetch gets an
independent completion handler which:

1. verifies that the request generation and selected stored session are still
   current;
2. reconciles the REST rows against `$messages.get()` at completion time;
3. preserves same-session optimistic/pending rows and local assistant errors;
4. calls `setMessages` only when the result is not content-equivalent; and
5. retains the successful response as a candidate for final runtime
   reconciliation.

A stale completion is discarded. It does not paint, bind a runtime, or arm a
failure latch.

The prefetch cannot be declared usable until the RPC supplies the bound stored
identity. A **usable prefetch** is therefore exactly one which resolved
successfully and whose `session_id` matches `resumed.session_key` /
`resumed.resumed` (with the existing missing-identity compatibility allowance).
Before the RPC settles, a guarded early publication is only a candidate view.

### Omitted Messages Are Not Empty Messages

After the RPC settles, `messages_omitted` controls the source of the transcript:

- With a usable prefetch, the already-published REST transcript is the base.
- A usable prefetch remains the transcript authority when an older gateway
  honors the client's `omit_messages` request but does not echo
  `messages_omitted`. Any `inflight` or `queued` projection from that response
  is grafted onto the REST baseline instead of replacing it.
- Without a usable prefetch, the hook makes one authoritative REST fallback
  request using the stored key bound by the resume response (falling back to the
  requested key only when the response omits its identity). It never passes
  `resumed.messages` to
  `reconcileAuthoritativeMessages` when `messages_omitted` is true.
- If the fallback succeeds, it is reconciled and published through the same
  guarded path. This is a transcript recovery, not an RPC failure, so it does
  not emit a synthetic "Resume failed" notification.
- If the fallback fails, the hook evaluates the empty-transcript failure rule
  below.

An identity mismatch is the important third unusable-prefetch case, alongside a
failed request and an empty result. The hook discards the candidate REST baseline
before rebuilding from the fallback response. It may preserve only local
optimistic/pending rows through the existing pending-turn reconciler; settled
rows from the mismatched candidate cannot remain in the foreground. Publishing
the corrected identity is a legitimate second paint and remains inside the
#69649 budget.

For watch windows, the request remains `lazy: true` without `omit_messages` and
the current no-prefetch behavior is unchanged.

### Settle Against the Latest View

Final reconciliation reads `$messages.get()` again after the RPC and any REST
fallback settle. It does not reuse the prefetch-time array as the previous
state. This protects optimistic rows and stream events that arrived between the
early paint and runtime binding.

If the RPC carries `inflight` or `queued`, `appendLiveSessionProjection` grafts
that tail onto the latest view. Otherwise the exact current array is written
into the per-runtime state. The subsequent `syncSessionStateToView` therefore
sees the same reference/content and does not cause another transcript paint.

`recoverInFlightTurnJournal` remains after runtime reconciliation. A journaled
tail recovered after a renderer/app crash is another legitimate second paint;
it must not be suppressed to make the unchanged-session budget pass.

This yields one paint for an unchanged cold resume and at most two for a cold
resume that adds a real live tail, corrects a mismatched identity, or restores a
journal tail.

## Empty-Transcript Failure Rule

The hook treats history as expected when either source says it exists:

```text
sidebar row message_count > 0
OR
(
  resume response message_count > 0
  AND session was not created by this renderer run
)
```

The second clause absolutely excludes `createdThisRun` from the runtime-count
source, whether the new session is running or idle. This avoids treating an
unpersisted first turn that failed or was interrupted as proof of a missing REST
display transcript. A real persisted count on its sidebar row remains the other
side of the OR condition.

The response count is used only as a `> 0` signal; its numeric value is never
compared with the REST row count because the gateway branches expose different
projections:

- deferred cold resume counts alternation-repaired raw history;
- lazy/watch resume counts display history;
- live reuse counts in-memory history plus any ancestor prefix.

A branch can have zero raw rows while its visible transcript comes from an
ancestor prefix. Its sidebar row remains the other side of the OR condition.

When history is expected and the pre-recovery transcript is empty, the hook
clears the foreground runtime binding and arms `$resumeFailedSessionId`. The
existing four-attempt exponential backoff then retries and ultimately displays
the existing inline ErrorState with manual Retry.

## Runtime Lifecycle

The latch path must not call `session.close` on `resumed.session_id`. A successful
resume registers the runtime under the stored session key; subsequent retries
hit `_find_live_session_by_key` and reuse that runtime rather than minting four
records. More importantly, a reused runtime may be the valid auto-continue run
whose transcript the renderer is trying to recover. Closing it would destroy the
feature being repaired.

A client-side RPC timeout can still orphan a server record if the server
registers it but its response never arrives. Existing gateway session-cap
enforcement owns that cleanup; adding cooperative RPC cancellation is outside
this renderer fix.

## Residual Failure Window

If the initial REST prefetch fails while `session.resume` is still pending,
there is no transcript to paint and no resume-bound identity for the second REST
attempt yet. Although the shared client library defaults to 120 seconds,
`HermesGateway` overrides Desktop's effective request timeout to 30 seconds.
After that timeout, the current RPC-failure catch path performs its REST fallback
and arms retry/error recovery if that also fails while cached metadata says
history exists.

This change removes the unbounded blank state when REST succeeds, which is the
reported #83729 case, but it does not shorten the dual-failure window. Starting
automatic retries before the pending RPC ends would create concurrent resumes.
A follow-up should add cancellable resume requests before adopting a shorter
deadline.

When `messages_omitted` is true, both REST attempts fail, and the RPC still
provides `inflight` or `queued`, the existing degraded behavior renders only
that live projection. Because the foreground is then non-empty, the missing-
history latch does not arm. This tail-only view predates the fix and remains a
known residual case rather than broadening this change into partial-history UI.

If an early prefetch paints successfully but `session.resume` later rejects,
the existing RPC-failure path performs another REST request instead of reusing
the painted result. Reusing it would remove a redundant request, but requires
hoisting successful-prefetch state across the catch boundary and is left as a
separate optimization.

## Non-Goals

- **Genuine 404 navigation:** the current code sends a prior-run stale route to
  a fresh draft when both RPC and REST report the session gone. Distinguishing a
  boot-time stale route from a user click requires carrying an explicit resume
  reason through `useRouteResume`; changing that UX is a separate follow-up and
  is not required to fix the valid session in #83729.
- **Warm-cache policy:** the warm path's stale sidebar check is unchanged. A
  known non-empty row with an empty cached view is evicted and falls through to
  this corrected cold path. Broadening warm-cache behavior would mix activation
  recovery into this fix.
- New UI, new timeout settings, or a resume state-machine rewrite.
- Changes to watch-window lazy resume.

No follow-up issue is created as part of this change; issue creation is an
external tracking action and should be requested separately.

## Tests

Hook tests in `use-session-actions.test.tsx` must cover:

1. REST history paints while `session.resume` is still pending.
2. A stale REST completion cannot overwrite a newer selected session.
3. REST prefetch failure plus `messages_omitted` triggers one REST fallback and
   never reconciles the omitted empty array.
4. REST failure plus `message_count > 0` arms the latch when the sidebar row is
   absent.
5. REST failure plus `message_count === 0` binds a legitimate empty session and
   does not arm the latch.
6. A newly created session, both running and idle cases, is not falsely latched
   from its live count.
7. A stream/pending row arriving after early paint survives final settle.
8. Same-session reconnect early paint preserves an optimistic user message.
9. A watch window performs no REST prefetch and keeps its current lazy behavior.
10. An unchanged resume retains the already-published array through final
    per-runtime state synchronization.
11. An identity-mismatched early prefetch is replaced from the resume-bound
    stored key and cannot leave the candidate session's settled rows in the
    foreground.

The large-session Electron E2E must cover both paint budgets:

- unchanged cold resume: exactly one transcript paint burst;
- cold resume with background inference/live projection: at most two bursts,
  with no duplicate user or assistant rows.

## Acceptance Criteria

- With a pending `session.resume` and successful REST response, persisted
  messages become visible before the RPC resolves.
- `messages_omitted: true` is never interpreted as authoritative empty history.
- A response reporting expected history cannot settle into
  `messagesEmpty && !activeSessionId` without arming recovery.
- New empty sessions and watch windows retain their current behavior.
- An async result from a superseded resume attempt cannot affect the foreground.
- The focused hook suite, Desktop typecheck and lint pass.
- Both cold large-session E2E paint budgets pass, including row de-duplication.

## Risks And Rollback

The main risk is reintroducing #69649 by publishing the same large transcript
more than once or by appending an inflight user row already present in REST. The
reference/content-equivalence checks, current-view settle rule and Electron
paint-budget tests are the release guards.

The change is renderer-local. If it regresses transcript ordering or paint
counts, it can be rolled back without gateway or database migration. The prior
behavior is restored by removing independent REST publication while retaining
the `messages_omitted` source guard and response-count latch as separable fixes.
