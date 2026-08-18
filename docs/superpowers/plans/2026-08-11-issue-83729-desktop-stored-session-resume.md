# Issue 83729 Desktop Stored Session Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan.

**Goal:** Make a valid persisted Desktop transcript visible as soon as REST returns, while preserving runtime projection correctness, retry semantics, and the #69649 paint budget.

**Architecture:** Keep the existing concurrent REST prefetch and `session.resume` request. Publish the guarded REST result as a candidate view immediately, then validate its stored-session identity when the RPC settles. If messages were omitted and that candidate is unavailable or mismatched, fetch once by the resume-bound stored key; finally reconcile live projection and journal recovery against the latest foreground array.

**Tech Stack:** React 19, TypeScript, nanostores, Vitest/Testing Library, Electron Playwright.

---

## Task 1: Lock Down Candidate Prefetch Publication

**Files:**
- Modify: `apps/desktop/src/app/session/hooks/use-session-actions.test.tsx`
- Modify: `apps/desktop/src/app/session/hooks/use-session-actions/index.ts`

### Step 1: Write the failing early-publication test

Add a deferred `session.resume` test in `resumeSession failure recovery`:

```ts
it('paints REST history while session.resume is still pending', async () => {
  const resumeDeferred = deferred<SessionResumeResponse>()
  vi.mocked(getLatestSessionMessages).mockResolvedValue({
    messages: [{ content: 'persisted question', role: 'user', timestamp: 1 }],
    session_id: 'stored-1'
  } as never)

  // Start resume without awaiting it, then assert the REST row appears while
  // activeSessionId remains null. Resolve the RPC and await completion.
})
```

Assert that the persisted row is visible before `resumeDeferred.resolve(...)`, that the runtime is not bound early, and that the final resume succeeds.

### Step 2: Run the focused test and confirm RED

Run:

```bash
cd apps/desktop
npx vitest run src/app/session/hooks/use-session-actions.test.tsx -t "paints REST history while session.resume is still pending"
```

Expected: FAIL because the current hook awaits `resumePromise` before publishing `prefetchedResult`.

### Step 3: Publish a guarded candidate view from the prefetch completion

In `resumeSession` cold-path code, add a local guarded publisher that:

```ts
function publishPersistedMessages(
  result: { messages: SessionMessage[]; session_id?: string },
  mode: 'candidate' | 'authoritative'
): ChatMessage[] | null {
  if (!isCurrentResume()) return null

  const currentMessages = $messages.get()
  const previousMessages =
    mode === 'authoritative'
      ? preserveLocalPendingTurnMessages([], currentMessages)
      : resumedSameSelectedSession
        ? preserveLocalPendingTurnMessages(currentMessages, resumeStartMessages)
        : currentMessages
  const nextMessages = reconcileAuthoritativeMessages(result.messages, previousMessages)
  const messagesForView = chatMessageArraysEquivalent(currentMessages, nextMessages) ? currentMessages : nextMessages

  if (messagesForView !== currentMessages) setMessages(messagesForView)
  return messagesForView
}
```

Adapt names/types to local style. The important contracts are:

- call `isCurrentResume()` immediately before mutating the foreground;
- read `$messages.get()` at completion time;
- preserve same-session optimistic/pending rows and assistant errors through the existing reconciler;
- use content equivalence so an unchanged candidate reuses the current array;
- do not bind `activeSessionId` from the REST result.

Await the prefetch while `resumePromise` runs concurrently, publish a successful result immediately, retain its `session_id`, then await the RPC.

### Step 4: Add stale-completion and same-session tests

Add tests that:

- start resume A with deferred REST, complete resume B, then resolve A and prove A cannot replace B's foreground;
- reconnect the currently selected stored session, keep an optimistic user row visible through the early candidate paint, then resolve the RPC and prove the row remains.

### Step 5: Add latest-view settle test

After the candidate paints, append a pending/stream row with `setMessages`, resolve an omitted-message RPC without a live projection, and assert final state still contains that row. Also assert the final state uses the current message array when no correction is needed.

### Step 6: Run focused tests and confirm GREEN

Run:

```bash
cd apps/desktop
npx vitest run src/app/session/hooks/use-session-actions.test.tsx -t "REST history|stale REST|same-session reconnect|arriving after early paint"
```

Expected: PASS.

### Step 7: Commit the candidate-publication slice

```bash
git add apps/desktop/src/app/session/hooks/use-session-actions/index.ts apps/desktop/src/app/session/hooks/use-session-actions.test.tsx
git commit -m "fix(desktop): paint stored transcript before resume settles"
```

## Task 2: Make Omitted-Message Recovery Identity-Safe

**Files:**
- Modify: `apps/desktop/src/app/session/hooks/use-session-actions.test.tsx`
- Modify: `apps/desktop/src/app/session/hooks/use-session-actions/index.ts`

### Step 1: Write the failing omitted-message fallback tests

Add one test for each unusable prefetch form:

1. Initial REST request rejects; RPC succeeds with `messages_omitted: true`; the hook calls REST once more using the resume-bound key, paints that transcript, and never treats `resumed.messages: []` as authoritative.
2. Candidate REST response identifies `stored-parent`; RPC binds `stored-continuation`; the hook fetches `stored-continuation`, replaces the parent baseline, and leaves no parent settled rows in the foreground.

For the mismatch case, assert the REST calls are ordered as:

```ts
expect(getLatestSessionMessages).toHaveBeenNthCalledWith(1, 'stored-1', null)
expect(getLatestSessionMessages).toHaveBeenNthCalledWith(2, 'stored-continuation', null)
```

Use the actual profile argument expected by the hook if it differs.

### Step 2: Run the new tests and confirm RED

Run:

```bash
cd apps/desktop
npx vitest run src/app/session/hooks/use-session-actions.test.tsx -t "omitted|identity-mismatched"
```

Expected: FAIL because the current fallback is only in the RPC catch path and the success path reconciles omitted `messages: []` when prefetch is unusable.

### Step 3: Define and enforce usable prefetch

After `resumePromise` settles, compute:

```ts
const resumedStoredSessionId = resumed.session_key || resumed.resumed || storedSessionId
const prefetchMatchesResumedSession =
  !prefetchedStoredSessionId ||
  !resumedStoredSessionId ||
  prefetchedStoredSessionId === resumedStoredSessionId
const usablePrefetch = prefetchApplied && prefetchMatchesResumedSession
```

When `resumed.messages_omitted` is true and `usablePrefetch` is false:

- make one `getLatestSessionMessages(resumedStoredSessionId, sessionProfile)` request;
- guard its completion with `isCurrentResume()`;
- publish it in authoritative mode, which retains only local optimistic/pending rows from the current foreground;
- update the accepted baseline identity;
- swallow fallback failure so the existing latch rule decides recovery;
- do not send a synthetic RPC failure notification.

If the candidate identity mismatched and the fallback fails, strip its settled rows before evaluating the latch, preserving only local pending rows. Never pass `resumed.messages` to `reconcileAuthoritativeMessages` when `messages_omitted` is true.

### Step 4: Reconcile the runtime projection against the latest view

Re-read `$messages.get()` after identity validation/fallback. For omitted messages, use:

```ts
const projected = appendLiveSessionProjection(currentMessages, resumed)
const preferredMessages = chatMessageArraysEquivalent(currentMessages, projected)
  ? currentMessages
  : projected
```

For non-omitted watch/lazy responses, retain the existing authoritative resume reconciliation. Leave `recoverInFlightTurnJournal` after this step so a restored journal tail remains a legitimate second paint.

### Step 5: Run the focused tests and confirm GREEN

Run:

```bash
cd apps/desktop
npx vitest run src/app/session/hooks/use-session-actions.test.tsx -t "omitted|identity-mismatched|in-flight turn"
```

Expected: PASS, including the existing inflight/queued projection test.

### Step 6: Commit the identity-safe fallback slice

```bash
git add apps/desktop/src/app/session/hooks/use-session-actions/index.ts apps/desktop/src/app/session/hooks/use-session-actions.test.tsx
git commit -m "fix(desktop): recover omitted resume history by bound identity"
```

## Task 3: Strengthen the Empty-Transcript Failure Rule

**Files:**
- Modify: `apps/desktop/src/app/session/hooks/use-session-actions.test.tsx`
- Modify: `apps/desktop/src/app/session/hooks/use-session-actions/index.ts`

### Step 1: Add response-count failure tests

Cover both sides of the response-count predicate while the sidebar has no matching row:

- fallback unavailable/empty plus `resumed.message_count > 0` arms `$resumeFailedSessionId`;
- fallback empty plus `resumed.message_count === 0` binds a legitimate empty session and leaves the latch clear.

Use `messages_omitted: true` in both responses so the tests exercise the fixed path.

### Step 2: Add newly-created-session tests

Exercise `createBackendSessionForSend` (or the narrowest existing create harness) so the returned stored ID enters `createdThisRun`. Resume that ID with an empty REST display transcript and a positive runtime `message_count`, parameterized for `running: true` and `running: false`. Assert neither case arms the latch. Do not seed a positive sidebar `message_count`, because that remains an independent expected-history signal.

### Step 3: Run the tests and confirm RED

Run:

```bash
cd apps/desktop
npx vitest run src/app/session/hooks/use-session-actions.test.tsx -t "message_count|newly created session"
```

Expected: the response-count test fails because the current latch only reads the sidebar row.

### Step 4: Implement the combined predicate

Immediately before journal recovery can mask transcript emptiness, calculate:

```ts
const responseClaimsHistory =
  !createdThisRun.has(storedSessionId) && (resumed.message_count ?? 0) > 0
const shouldHaveTranscript = sessionShouldHaveTranscript(stored) || responseClaimsHistory
```

Arm the existing latch when `shouldHaveTranscript && preferredMessages.length === 0`. Keep the current no-`session.close` lifecycle and reuse behavior.

### Step 5: Add and verify the watch-window regression test

Mock `isWatchWindow()` per test if necessary. Assert a watch resume sends `{ lazy: true }`, omits `omit_messages`, makes no REST request, and uses the RPC-carried transcript exactly as before.

Run:

```bash
cd apps/desktop
npx vitest run src/app/session/hooks/use-session-actions.test.tsx -t "message_count|newly created session|watch window"
```

Expected: PASS.

### Step 6: Commit the latch slice

```bash
git add apps/desktop/src/app/session/hooks/use-session-actions/index.ts apps/desktop/src/app/session/hooks/use-session-actions.test.tsx
git commit -m "fix(desktop): detect missing history from resume metadata"
```

## Task 4: Enforce the #69649 Paint Budget in Electron E2E

**Files:**
- Modify: `apps/desktop/e2e/large-session-resume.spec.ts`

### Step 1: Make the unchanged cold budget exact

Parameterize `assertUnchangedResume` with an expected maximum or exact budget. The cold test must assert:

```ts
expect(paints.bursts, diagnostic).toBe(1)
```

Keep the known warm-resume `fixme` separate and capped at two if it is later re-enabled.

### Step 2: Observe the cold live-projection resume

For the `cold resume keeps background inference attached` case, attach the mutation observer after `reloadIntoColdRenderer` and before `openSeededSession`. After the held stream completes, assert:

```ts
expect(paints.bursts, diagnostic).toBeLessThanOrEqual(2)
```

Retain the existing exact one-row assertions for the running user prompt and completed assistant reply.

### Step 3: Run the focused Electron tests

Run:

```bash
cd apps/desktop
npm run build
npx playwright test e2e/large-session-resume.spec.ts --grep "cold resume"
```

Expected: the unchanged case records exactly one paint burst; the background/live case records no more than two and has no duplicate rows.

### Step 4: Commit the E2E contract

```bash
git add apps/desktop/e2e/large-session-resume.spec.ts
git commit -m "test(desktop): enforce stored resume paint budget"
```

## Task 5: Full Verification and Review

**Files:**
- Review: `apps/desktop/src/app/session/hooks/use-session-actions/index.ts`
- Review: `apps/desktop/src/app/session/hooks/use-session-actions.test.tsx`
- Review: `apps/desktop/e2e/large-session-resume.spec.ts`

### Step 1: Run the complete hook suite

```bash
cd apps/desktop
npx vitest run src/app/session/hooks/use-session-actions.test.tsx
```

Expected: PASS.

### Step 2: Run static checks

```bash
cd apps/desktop
npm run typecheck
npm run lint
```

Expected: both commands exit 0.

### Step 3: Run the scoped E2E suite again after the final build

```bash
cd apps/desktop
npm run build
npx playwright test e2e/large-session-resume.spec.ts --grep "cold resume"
```

Expected: PASS with the specified paint and de-duplication budgets.

### Step 4: Inspect the final diff for lifecycle and scope regressions

```bash
git diff origin/main --check
git diff --stat origin/main
git diff origin/main -- apps/desktop/src/app/session/hooks/use-session-actions/index.ts apps/desktop/src/app/session/hooks/use-session-actions.test.tsx apps/desktop/e2e/large-session-resume.spec.ts
```

Verify explicitly:

- no `session.close` was added to the latch path;
- no new timeout, UI state, or warm-cache policy was introduced;
- watch-window lazy resume remains intact;
- omitted messages are never reconciled as authoritative history;
- final reconciliation reads `$messages.get()` after all async work;
- unchanged cold resume stays at one paint and correction/live/journal cases stay within two.

### Step 5: Commit any verification-only adjustments

If verification required a code or test adjustment, commit it with a focused message. Otherwise leave the preceding implementation commits unchanged.
