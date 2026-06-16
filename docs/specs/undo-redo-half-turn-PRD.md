# PRD — Half-Turn `/undo` + `/redo` Stack

**Status:** v1.7 (Pass 7 = BLOCK → fixed: B-1 artifact-mismatch resolved by bringing IMPLEMENTER-SPEC.md UNDER review (pass 8 reviews the spec directly, the real build target — reviewer's recommended option a; the two new rules are mirrored into it); RC-1 general grouping invariant for non-alternating/adjacent-same-role input stated in §5.1 + Phase 1 adjacent-multimodal-user-pair fixture; RC-2 caller audit committed to AST/import form, grep fallback deleted; N-1/N-2/N-3 acknowledged. Core model converged for 4 passes; remaining work was at document seams.)
**Implementer spec (build from this; now itself reviewed):** `undo-redo-half-turn-IMPLEMENTER-SPEC.md` — self-contained, no review provenance. THIS doc is the review record; the two are semantically equivalent (pass-8 reviews the spec).
**Author:** Apollo
**Date:** 2026-06-14
**Repo:** `Kyzcreig/hermes-agent` (fork) → PR to `NousResearch/hermes-agent`
**Surfaces:** CLI (`cli.py`), Gateway (`gateway/run.py` + `gateway/session.py`), TUI (`tui_gateway/server.py`)

---

## 1. Summary & Goal

`/undo` today operates on **whole back-and-forth turns** (1 user message + its assistant reply + tool calls). This is too coarse: a user who wants to **revise their own last message** cannot do so without also discarding a good assistant reply. `/undo 1` rewinds to the user's last message (fine); but if the user instead wants to keep their last message's *reply* and only adjust something earlier, or wants finer stepping, the turn unit gets in the way. There is also **no way to reverse an `/undo`** — the `restore_rewound()` DB primitive exists but is wired to nothing.

**Goal:** Two changes, across all three surfaces:

1. **Make `/undo N` operate on half-turns.** A *half-turn* = one party's run of consecutive messages. The user's message(s) = 1 half-turn; the assistant's reply (including its tool calls + tool results) = 1 half-turn. `/undo 1` lands on the user's last message (edit & resend); `/undo 2` lands on the assistant's penultimate message with the turn handed back to the user; and so on — granular, alternating stepping.
2. **Add `/redo [N]`.** `/redo` (bare) reverses exactly the **last `/undo` operation** (whatever N it carried). `/redo N` pops **N undo operations** off a per-session stack, text-editor style. Typing any new message **clears the redo stack**.

**Why now:** User hit the wall directly — wanted to edit a message without nuking a good reply, and discovered there's no `/redo`. The redo DB primitive already exists (`SessionDB.restore_rewound()`, docstring: *"Intended for undo-of-rewind … not wired to a slash command in v1"*), so this is largely wiring + stack state, not new persistence plumbing.

---

## 2. Non-Goals

- **No change to `/rollback`** (filesystem checkpoints) or `/branch` (session forking).
- **No change to soft-delete audit semantics.** Rewound rows stay `active=0` in `state.db` for audit; redo flips them back to `active=1`. Nothing is hard-deleted by undo/redo.
- **No cross-session undo/redo.** The undo/redo stack is per-session.
- **No persistence of the in-memory redo stack across a process restart as a hard guarantee.** Restart degrades gracefully (see §5.3) — it does not corrupt, but a cold redo stack may be empty or DB-reconstructed best-effort.
- **No new UI affordance** beyond the existing prefill/echo behavior (no buttons, no visual stack browser).
- **No change to `/retry` or `/compress`** semantics.

---

## 3. Constitution / Invariants

- **Invariant — No data loss on undo.** Undo only ever flips rows to `active=0` (soft-delete); it never hard-deletes. Redo flips them back.
  - *Why:* audit/forensic retention is an existing guarantee (`rewind_to_message` docstring); we must not weaken it.
  - *Closeout proof:* `pytest` test asserting rewound rows are still present with `active=0` after undo, and `get_messages(include_inactive=True)` returns them; grep confirms no `DELETE FROM messages` introduced.

- **Invariant — Provider role-alternation preserved; the merge-gap on multimodal user content is closed at the undo layer (pass-5 RC-1).** After any undo + new user message (which can produce two consecutive user messages), the live history sent to the provider must satisfy strict alternation.
  - *Why & the real contract:* OpenAI/Anthropic/OpenRouter reject two same-role messages in a row. We rely on `repair_message_sequence` (`agent/agent_runtime_helpers.py`) Pass 2, whose **actual, code-verified contract** is asymmetric AND **conditional on content type**: it merges two consecutive *user* messages **only when BOTH `content` values are plain `str`** (`isinstance(prev_content, str) and isinstance(new_content, str)`, joined `\n\n`); **if either is list/multimodal content it is left UNMERGED** (the code comment: "collapsing image/audio blocks risks mangling the attachment structure"), and there is **no assistant-merge pass**. (Cited by *symbol + behavior*, not a line number — line numbers rot; the characterization test pins the behavior.)
  - *The conflict this surfaces (RC-1):* for a plain-text trailing user message the append+merge model (D-3) keeps alternation. But if the trailing user message landed on by undo carries **multimodal/list content**, an appended new user message is **NOT merged** → two user rows reach the provider → alternation violation. The pinned contract and "never two user-in-a-row reach the provider" otherwise disagree.
  - *Resolution (D-13):* the undo layer detects this. **When `/undo` lands on (prefills) a user message whose `content` is non-`str` (multimodal/list), the prefill path uses REPLACE semantics — it deactivates that trailing user row as part of the rewind (target = that user row, inclusive)** so the resend is the *sole* user row, never an unmergeable pair. Plain-`str` trailing user messages keep the user's append+merge model (D-3) unchanged. This closes the gap **without touching `repair_message_sequence`** (which deliberately won't merge attachments).
  - *Closeout proof:* (1) characterization test (pass-4 RC-A) pinning the real function: (a) two plain-text user msgs → merged with `\n\n`; (b) two user msgs, one multimodal/list → **NOT merged**; (c) two assistant msgs → intact. (2) **`test_multimodal_undo_no_adjacent_user` (pass-5 RC-1):** undo landing on a multimodal user message + a new user message yields, in the provider-bound payload, **no two adjacent user rows** (proving the D-13 replace-fallback fired). (3) the plain-text undo→type→repair end-to-end test asserting the merged result.

- **Invariant — Redo reactivates EXACTLY the matching undo's rows; stacked undo/redo is order-symmetric.** `undo` then `redo` (no intervening message) returns the **identical active message set** (same row ids). For *stacked* ops, **single redos invert single undos in LIFO order**: starting from a base state, `/undo` ×3 then **three separate `/redo`** (each bare = `/redo 1`) restores, after each redo, the active-id set to the post-`undo×2`, post-`undo×1`, and pre-`undo` states respectively. Equivalently, `/undo ×3` then a single `/redo 3` lands directly on the pre-`undo` state (no observable intermediates — one command pops three ops).
  - *Why:* the whole feature's correctness claim. Redo uses `restore_ids(op.rewound_ids)`, never an `id >=` range, so stacked ops of different N don't reactivate rows they don't own (B-2). The trace is stated as one unambiguous sequence to avoid the pass-2 contradiction (you cannot assert per-step intermediates *and* call it a single `redo×3`).
  - *Closeout proof:* `test_undo_redo_multi_op_identity` runs **`undo` ×3 with different N then three separate bare `/redo` calls**, asserting after each `/redo` the active-id set equals the corresponding historical snapshot (post-`undo×2`, post-`undo×1`, pre-`undo`), final = original. A separate `test_redo_n_single_command` runs `undo×3` then one `/redo 3` and asserts only the final pre-`undo` set.

- **Invariant — New user message clears the redo stack, at ONE core enforcement point.** Once the user diverges, the orphaned branch is unredoable, and the clear happens in the shared core user-append path — not per-surface (RC-2).
  - *Why:* prevents redo resurrecting a branch that no longer follows the current user message; one call site removes three-surface drift risk (R6).
  - *Closeout proof:* test does `/undo`, sends a new message, attempts `/redo`, asserts "nothing to redo" and diverged rows stay `active=0`; plus a grep asserting the clear-redo call lives in core, not in each surface's send path.

- **Invariant — Stack is per-session, never bleeds across sessions.** Undo/redo stack keyed by `session_id`.
  - *Closeout proof:* test with two sessions; undo in A; assert `/redo` in B is a no-op against B's own (empty) stack.

- **Invariant — Never two assistant messages in a row, by absence-of-production (not by a constructed-hazard test).** The half-turn model can leave two *user* messages in a row (merged by `repair_message_sequence` Pass 2); no feature operation ever appends an assistant message adjacent to another assistant message.
  - *Why:* a new assistant row only appears from a fresh model reply (which follows a user message); undo appends nothing; redo reactivates a previously-coherent contiguous group (proven by the §5.1 group-start rule). So the hazard is **structurally unreachable** — and a "construct the hazard, watch it be prevented" test would be theater, since no real code path can build the hazard (pass-3 B-2′).
  - *Closeout proof (two honest artifacts, not a strawman):* (1) `test_no_feature_path_emits_adjacent_assistant` — a grep/structural assertion that no undo/redo/append path in `hermes_undo.py` or the three surfaces emits an assistant row without an intervening user row; and (2) `test_repair_does_not_silently_merge_assistants` — feed `repair_message_sequence` a hand-built assistant-after-assistant input and assert it does **not** silently coalesce them into one (proving repair has no assistant-merge pass papering over the hazard; if such input ever arose it would surface, not hide). No `test_two_assistant_hazard_prevented`.

- **Invariant — Contract back-compat for `N`.** `/undo` bare still means "one step." (The *meaning* of a step changes from turn→half-turn — a documented, intentional behavior change, see §4 D-2.)
  - *Closeout proof:* `/undo` with no arg backs up exactly one half-turn.

---

## 4. Resolved Decisions

- **D-1 — Undo unit is the half-turn.** A half-turn = one party's run of consecutive messages. User message(s) = 1 half-turn; assistant reply + its tool calls + tool results = 1 half-turn. `/undo N` backs up N half-turns, alternating. (User confirmed via `/undo 1`→user's last msg, `/undo 2`→assistant's penultimate msg, user's turn to reply.)

- **D-2 — `N` now counts half-turns, not full turns (intentional behavior change).** Documented in CLI help, gateway help, TUI, and the slash-command registry description. This changes existing `/undo N` behavior; called out explicitly, not smuggled.

- **D-3 — Two consecutive user messages after undo+type are allowed.** Rely on existing `repair_message_sequence` Pass 2 (merges consecutive user messages). Never two assistant in a row. (User confirmed: "if we land on a user message we just add the next user message to that user message, so it's two user messages in a row… never two assistant messages in a row.") **Scope (pass-5 RC-1): this merge fires ONLY for plain-`str` user content; the multimodal exception is handled by D-13.**

- **D-13 — Multimodal trailing-user-message prefill uses REPLACE, not append (pass-5 RC-1; location & composition pinned pass-6 B-1′).** `repair_message_sequence` merges consecutive user messages **only when both contents are plain `str`**; multimodal/list content is left unmerged (would otherwise produce two adjacent user rows → provider alternation error). Closing this **does NOT change `compute_half_turn_target`** (which stays strictly role-only / N-agnostic, §5.1). Instead:
  - **Where it executes:** inside the core `hermes_undo.undo(session_id, n)` helper, as a **post-computation target adjustment** — *after* `compute_half_turn_target` returns the role-based target id `t`, the helper inspects **the message now at the new active tail** (the row immediately preceding `t`, i.e. the same row D-4 reads for the prefill decision). This is the single point that reads `content` type; the boundary primitive never does.
  - **The rule (composes with arbitrary N):** if that trailing row is a **USER row** (so D-4 would prefill) AND its `content` is **non-`str`** (list/multimodal), the helper **lowers the rewind target to that user row's id** (target becomes that user row, inclusive) before performing the soft-delete, so that row is included in `rewound_ids` and deactivated. If the trailing row is an assistant row (no prefill, D-4) or a plain-`str` user row, the target is unchanged. **This is purely a function of the new tail's trailing row — independent of N**: N picks the boundary, then this one check looks at exactly one row. (N=2 landing the tail on an assistant row → no adjustment, because the trailing row is assistant; only a multimodal *user* trailing row triggers REPLACE.)
  - **Consequence for prefill (D-4):** because the multimodal user row is now deactivated, the new tail ends on the *prior assistant row* → D-4 yields **no prefill** (you cannot edit image/audio blocks as composer text anyway). So undo onto a multimodal user message lands **armed-empty** (intentional, see N-1). Plain-`str` user rows keep append+merge (D-3) and prefill (D-4) unchanged.
  - (Apollo's judgment closing the RC-1 contract/invariant conflict; no change to `repair_message_sequence` or `compute_half_turn_target`.)

- **D-4 — Prefill rule, N-agnostic (pass-4 RC-B): prefill iff the new active tail's last message is a USER message.** After `/undo N` removes N half-turns, look at the message now at the end of the active list (the one immediately preceding the rewind target). **If it is a USER message → prefill the composer with its text** (edit-and-resend); **if it is an ASSISTANT message → do not prefill** (the session is armed for the user to type their reply / `/retry`). This single rule holds for every N — the implementation reads that one row's role; **it never computes N-parity**. Traced on tail `…U(5),A(6),U(7),A(8)`:
  - `/undo 1` removes `{A(8)}` → new tail ends **U(7)** → **prefill** (matches D-1 "lands on the user's last message, edit & resend").
  - `/undo 2` removes `{A(8),U(7)}` → new tail ends **A(6)** → **no prefill** (matches D-1 "lands on the assistant's penultimate message, your turn to reply").
  - `/undo 3` removes `{A(8),U(7),A(6)}` → new tail ends **U(5)** → **prefill**.
  - Edge: if undo clears to an empty active list (removed everything), there is no preceding message → no prefill.

- **D-5 — Redo is a stack of operations, not half-turns.** Each `/undo` pushes one operation (carrying its N and the set of row ids it deactivated). `/redo` (bare) pops and reverses the **single most-recent** undo operation (whatever N it had). `/redo N` pops and reverses N operations. (User confirmed: "by default it undoes exactly the last undo command… but if you do /redo # it would undo multiple /undo # commands, just like a stack.")

- **D-6 — Typing a new message clears the redo stack.** (User confirmed.)

- **D-7 — Stack state lives in-memory per session, in a SHARED session-state object the core can reach; no cold-restart reconstruction.** (Apollo's judgment, user delegated; tightened after review B-3/RC-2.) The undo/redo stacks live on a shared per-session state object (not duplicated in each surface's own send path). Each `UndoOp` records `{n, rewound_ids: [int]}` — the **explicit list of row ids that op deactivated** (not a range, not just a count). On a process restart the in-memory stacks are lost; a cold `/redo` with an empty stack reports **"nothing to redo"** and touches nothing. **There is NO contiguous-block / `redo_marker` reconstruction heuristic** — it was cut because reactivating "the most-recent contiguous `active=0` block" cannot prove the rows belong to a single undo op and so violates the no-corruption invariant. Cold redo = empty redo. (Resolves review B-3, Q-2.)

- **D-7a — The undo/redo stack transition table is fixed HERE, not deferred to writing-plans (resolves review B-1).** Paired stacks (`undo_stack`, `redo_stack`), text-editor semantics. The table tracks **both** stack contents **and** the resulting active-id set (review pass-2 B-1: stack-array coherence ≠ message-state coherence). Worked trace uses a concrete history whose half-turns deactivate id sets: assume `/undo 1`=op deactivating `{8}`, a second `/undo 1`=op deactivating `{7}` (each one half-turn), etc. — ids illustrative.

  | Action | `undo_stack` | `redo_stack` | active-id Δ | DB op |
  |--------|--------------|--------------|------------|-------|
  | `/undo N` | push `UndoOp{n=N, rewound_ids}` | **clear** | remove `rewound_ids` from active | `rewind_to_message` deactivates `rewound_ids` |
  | `/redo M` (bare = M=1) | pop M ops | push each popped op | add each op's `rewound_ids` back to active | `restore_ids(op.rewound_ids)` per op |
  | new user message | (unchanged) | **clear** | append the new row | append message |

  - `/undo` **clears `redo_stack`** (a fresh undo establishes a new branch; pending redo is stale). The three interleaving sequences traced with **exact ids** on a concrete history — 4 turns: `1=user,2=asst, 3=user,4=asst, 5=user,6=asst, 7=user,8=asst`, all bare commands (N=1) unless noted:
    - *undo, undo, redo, undo* (start active `{1..8}`):
      1. undo₁ → half-turn HT1 = `{8}` → **op1.rewound_ids={8}**, active `{1..7}`, `undo_stack=[op1]`, `redo_stack=[]`
      2. undo₂ → HT1 of `{1..7}` = `{7}` → **op2.rewound_ids={7}**, active `{1..6}`, `undo_stack=[op1,op2]`, `redo_stack=[]`
      3. redo → pop op2, `restore_ids({7})` → active `{1..7}`, `undo_stack=[op1]`, `redo_stack=[op2]`
      4. undo₃ → recompute on active `{1..7}`; HT1 = `{7}` → **op3.rewound_ids={7}**, active `{1..6}`, `undo_stack=[op1,op3]`, **`redo_stack=[]` (cleared)**
      **No-orphaning proven on these numbers:** the only `active=0` rows are `{7,8}`; `{8}`∈op1 (on undo_stack, redoable), `{7}`∈op3 (on undo_stack, redoable). op2 was discarded, but every id it owned (`{7}`) is now owned by op3 — **no id is `active=0` without a live owning op**. A later `/redo 2` pops op3 then op1, `restore_ids({7})` then `restore_ids({8})` → active `{1..8}`, full restore.
    - *undo, redo, undo* (start `{1..8}`): undo₁ → op1=`{8}`, active`{1..7}`; redo → `restore_ids({8})`, active`{1..8}`, `undo_stack=[]`,`redo_stack=[op1]`; undo₂ → op2=`{8}`, active`{1..7}`, `undo_stack=[op2]`, `redo_stack=[]` (cleared). All `active=0` (`{8}`)∈op2. No orphan.
    - *undo×3, redo 2, undo* (start `{1..8}`): undo₁=`{8}`, undo₂=`{7}`, undo₃=`{6}` → `undo_stack=[op1,op2,op3]`, active`{1..5}`; `/redo 2` pops op3 then op2, `restore_ids({6})` then `restore_ids({7})` → active`{1..7}`, `redo_stack=[op3,op2]`, `undo_stack=[op1]`; final undo → HT1 of `{1..7}`=`{7}` → op4=`{7}`, active`{1..6}`, `undo_stack=[op1,op4]`, `redo_stack=[]` (cleared). `active=0`=`{7,8}`; `{8}`∈op1, `{7}`∈op4. No orphan.
  - `/redo` (bare) = `/redo 1`. `/redo M` clamps to `len(undo_stack)`. `/redo` with empty `undo_stack` → "nothing to redo," no DB op, **no `redo_count` bump**.
  - **Stale-op unreachability (the real no-corruption invariant, pass-3 RC-3′):** the defense against resurrecting a wrongly-killed row is **not** `restore_ids` idempotency — it's that **`restore_ids` is only ever called with the `rewound_ids` of an op currently on a stack**. When `redo_stack` is cleared (on `/undo` or new message), discarded ops become **unreachable** — no code path retains a reference to call `restore_ids` with their ids. Idempotency (D-11a) is belt-and-suspenders only.
  - **Idempotency rule (D-11a):** `restore_ids` no-ops on any id already `active=1` (never a phantom reactivation). Note this does NOT by itself prevent reactivating a row legitimately `active=0` (killed by a later op) — that's guaranteed by stale-op unreachability above, which idempotency backs up but does not replace.

- **D-8 — `rewind_to_message` is generalized to accept any message id (not just `user` role) AND to return its rewound id list.** Half-turn stepping requires landing on assistant-message boundaries too. Add a parameter (`require_user_role: bool = True`) so existing callers keep the guard; half-turn callers pass `False`. **Also surface `rewound_ids: [int]`** in the return dict (the function already computes this list internally as `ids`) so redo can reactivate the exact set. Return-shape change is additive (new key), not a positional change — existing callers that read `rewound_count`/`target_message`/`new_head_id` are unaffected (see RC-4 caller audit, §5.2).

- **D-11 — Redo reactivates by EXPLICIT id set, never by `id >=` range (resolves review B-2).** `restore_rewound(session_id, since_id)` reactivates all `id >= since_id`, which under stacked undos of different N can reactivate rows a given op never owned (and clobber a later op's deactivation). A new DB primitive `restore_ids(session_id, ids: [int])` reactivates **exactly** the passed ids. Redo always passes `op.rewound_ids`. `restore_rewound` is left intact (unused by this feature; not removed — out of scope).

- **D-12 — Observability: add a symmetric `redo_count` bump (resolves review N-1).** `restore_ids` increments `sessions.redo_count` (additive nullable column) the way `rewind_to_message` bumps `rewind_count`. Makes the deactivate/reactivate cycle auditable. Cheap; avoids a follow-up PR.

- **D-9 — Ship all three surfaces in one PRD.** (User confirmed "all three surfaces.") Shared core logic lives in `hermes_state.py` (DB ops) + a small surface-agnostic helper for half-turn boundary computation, so each surface is thin wiring.

- **D-10 — Implement directly (no swarm).** (User confirmed.) Apollo implements after review passes.

---

## 5. Architecture / Design

### 5.1 Core: half-turn boundary computation

A new pure helper, **`compute_half_turn_target`, placed in a new shared module `hermes_undo.py`** (not `hermes_state.py` — boundary logic is not DB logic; this co-locates with the `UndoRedoState` holder and the clear-redo hook per pass-2 N-3). Given the active message list and a half-turn count `N`, it computes the **target message id** to rewind to:

```
def compute_half_turn_target(active_messages, n):
    # active_messages: ordered list of {id, role, ...} (user/assistant/tool), active=1 only
    # Walk backwards, grouping consecutive same-"party" messages into half-turns.
    #   party(user) = "user"; party(assistant|tool) = "assistant"
    #   (tool messages belong to the assistant half-turn that spawned them)
    # Count N half-turn group-starts from the end; return the id of the FIRST
    # message of the Nth group — which for an assistant half-turn is the
    # assistant(tool_calls) row, NEVER a mid-group tool row. That id (inclusive)
    # is the rewind target.
```

Key rules:
- **General grouping invariant (pass-7 RC-1, makes the rule explicit for irregular input):** **any maximal run of consecutive same-`party()` rows = exactly ONE half-turn, regardless of how the run arose** — normal alternation, a `repair_message_sequence` user-merge, a multimodal user pair left *un*-merged (D-13's carve-out can leave two adjacent user rows in the active list), or an interrupted/partial assistant turn. The walk is purely "scan backward; a `party()` change starts a new half-turn." So a tail `…A, U(multimodal), U(text)` is **one** user half-turn (both user rows), and `/undo 1` over it lands on the earliest of that run; `/undo 2` lands on the preceding assistant group-start. This is pinned because D-13 *increases* the chance of an adjacent same-role user pair in the grouping input — the rule must not silently treat that pair as two half-turns.
- Tool messages (`role="tool"`) and assistant messages coalesce into **one assistant half-turn**. The **group-start is always the earliest message of the run** — for an assistant half-turn that is the `assistant` row bearing the `tool_calls` (or the lone assistant text row), never the trailing `tool` rows.
- **Partial assistant half-turn** (assistant `tool_calls` present, the final assistant text row absent — e.g. an interrupted turn): the group still starts at the `assistant(tool_calls)` row, so the cut deactivates the *whole* group (`assistant(tool_calls)` + its `tool` rows) atomically.
- Consecutive user messages (already-merged or not, including a multimodal+text adjacent pair) coalesce into **one user half-turn** (per the general invariant above).
- `N` clamps to the oldest half-turn if it exceeds the count.
- The target id is **inclusive** (that message and everything after it gets deactivated).
- **Why this matters for redo (pass-2 B-3):** because the `rewind_to_message` cut is `id >= target` and the target is always a *group start*, a single undo can never split a tool group on the deactivate side. The real hazard is on **redo**: `restore_ids(op.rewound_ids)` must reactivate the *complete* group the matching undo removed. Since `op.rewound_ids` is exactly the contiguous set the cut removed (group-start through end), redo restores the whole group — proven by the Phase 1 fixture that asserts (a) the computed target for the partial-half-turn case **is** the `assistant(tool_calls)` row, and (b) the undo→redo round-trip on that fixture reactivates the complete group with no `assistant(tool_calls)` left without its `tool` result.

- **Input source — the grouping input MUST carry row `id` (pass-5 B-1).** `compute_half_turn_target` returns a **row id**, so its input list must include each row's `id`. The id-bearing getter is **`SessionDB.get_messages(session_id, include_inactive=False)`** (`SELECT *`, active-only, `ORDER BY id`) — verified to return the `id` column. Note **`get_messages_as_conversation` does NOT include `id`** (it returns the provider-format `{role, content, tool_calls, …}` projection only); it is the wrong input for boundary computation. All three surfaces therefore feed `compute_half_turn_target` from `get_messages` (active-only), and the Phase 7/N-B same-getter parity assertion pins **`get_messages(active-only)`** as that shared getter (not `get_messages_as_conversation`). `party()` reads `role`, and an assistant-with-tool-calls row is identified by a populated `tool_calls` field (JSON column) with `content` possibly `NULL` — the Phase 1 real-row test exercises exactly this persisted shape.

### 5.2 DB layer (`hermes_state.py`)

- **Generalize `rewind_to_message`** (D-8): add `require_user_role: bool = True`. When `False`, skip the `role != "user"` `ValueError`. Everything else (soft-delete `id >= target`, bump `rewind_count`, return `rewound_count`/`target_message`/`new_head_id`) unchanged, **plus a new additive return key `rewound_ids: [int]`** (the `ids` list it already computes). Additive — existing callers unaffected.
- **New `restore_ids(session_id, ids: [int]) -> int`** (D-11): reactivates ids that are **currently `active=0`** (`UPDATE messages SET active=1 WHERE id IN (...) AND active=0`, scoped to `session_id`), returns count reactivated. **Idempotent (D-11a):** an id already `active=1` is silently skipped — belt-and-suspenders behind the real guarantee (stale-op unreachability, D-7a). This is the redo primitive — **not** `restore_rewound`'s `id >=` range (B-2). `restore_rewound` is left in place but **gets a deprecation docstring** (pass-3 N-2′): "DEPRECATED for stacked undo/redo — use `restore_ids`; `id >=` range clobbers stacked ops of differing N." `restore_ids` does **not** bump `redo_count` (that's the core redo helper's job, below).
- **RC-4 caller audit (back-compat, AST/import-based per pass-7 RC-2 — NOT a text grep):** Phase 1 imports the modules that call `rewind_to_message` and asserts, via AST/reference resolution, that the complete caller set is exactly: the `def` site, `cli.py::undo_last`, `gateway/session.py::rewind_session`, and tests — **no third consumer** — AND that each caller **reads the return by key** (`result["..."]`/`result.get("...")`), never positionally-unpacks, splats (`**result`), iterates, aliases the function (`x = rewind_to_message; x(...)`), or passes the dict through. The AST form is required here because the failure mode (a new third caller positionally-unpacking the now-larger return) is exactly what a hand-guessed regex over consumption patterns can miss. Adding `rewound_ids` is therefore proven safe by reference resolution, not by spelling.
- **`redo_count` (D-12, ONE core site per pass-3 RC-4′):** the bump lives in the **shared core redo helper** (`hermes_undo.redo(...)` — the function that pops ops off `undo_stack` and calls `restore_ids`), **not** duplicated across the three command sites. Symmetric with the one-site clear-redo (RC-2). Counts **successful user `/redo` commands** (≥1 op reactivated), once per command regardless of M; empty-stack `/redo` does not bump. Additive nullable column on `sessions`; safe to drop on rollback.

### 5.3 Stack state & restart degradation (D-7, D-7a)

- **Shared session-state object holds `undo_stack` and `redo_stack`** (RC-2). It is **not** duplicated in each surface's send path. Concretely: a small `UndoRedoState` holder keyed by `session_id`, reachable from the core, so the clear-redo-on-new-message hook has **one enforcement point** (see below). Each surface's undo/redo command reads/writes this shared holder rather than its own list.
- **Core entry points (named, pass-3 RC-2′):** the shared module `hermes_undo.py` exposes the **only** functions that mutate stacks/state, so every surface calls one named contract rather than improvising:
  - `hermes_undo.undo(session_id, n)` — compute target, deactivate, push op, clear redo_stack.
  - `hermes_undo.redo(session_id, m)` — pop ≤m ops, `restore_ids` each, push to redo_stack, **bump `redo_count` once** if ≥1 op reactivated.
  - `hermes_undo.on_user_message_appended(session_id)` — **clears `redo_stack`** (D-6). **Every surface's user-append path MUST call this**, named individually below — TUI especially, since its JSON-RPC handlers historically persist directly.
- **Transitions** follow the D-7a table exactly (via the entry points above).
- **Restart degradation (D-7):** in-memory stacks are lost on process restart. A cold `/redo` with empty `undo_stack` reports **"nothing to redo"** and touches nothing. **No `redo_marker`, no contiguous-block reconstruction** (cut per B-3). Fail-closed to empty.
- **Restart is a TESTED degradation, not just a claim (pass-5 B-2).** Two guarantees get explicit, *falsifiable* coverage in Phase 4 (the gateway is the surface with a real persistent process):
  - **(a) Cold redo touches zero rows.** Drive a real sequence — instantiate state, `/undo` (rows go `active=0`), **drop/replace the in-memory `UndoRedoState` holder to simulate the restart**, then `/redo` — and assert (i) the **restart-specific** N-1′ message fires (`rewind_count>0` but cold stack → "redo history doesn't survive a restart"), distinct from the bare "nothing to redo" of a never-undone session, and (ii) `get_messages(include_inactive=True)` is **byte-for-byte unchanged** (zero rows reactivated). Without this, the committed N-1′ conditional is unfalsifiable.
  - **(b) No surface reconstructs a stack from orphaned `active=0` rows.** A structural assertion (grep, like the clear-redo one): no undo/redo path in `hermes_undo.py` or the three surfaces reads `active=0` rows (`include_inactive=True` / `restore_rewound` / any `active = 0` SELECT) to re-derive a redo stack. This turns "fail-closed to empty" from a claim into an enforced guarantee — the heuristic was *cut* (D-7), and this proves no surface silently re-added one.

### 5.4 Surface wiring (thin)

All three surfaces call the **same** `hermes_undo.{undo,redo,on_user_message_appended}` (§5.3) — which internally use `compute_half_turn_target`, `rewind_to_message(require_user_role=False)`, and `restore_ids`. Per-surface work is only command plumbing + **wiring the user-append path to `on_user_message_appended`**:

- **CLI (`cli.py`):** `undo_last(n)`→`hermes_undo.undo` + conditional prefill (D-4); `redo_last(n)`→`hermes_undo.redo`; register `/redo` dispatch (`elif canonical == "redo"`). **CLI user-send path calls `on_user_message_appended`.**
- **Gateway (`gateway/run.py` + `session.py`):** `rewind_session(n)`→`hermes_undo.undo`; `restore_session(m)`→`hermes_undo.redo`; `_handle_redo_command` in `run.py`; evict cached agent on redo. **Gateway message-ingest path calls `on_user_message_appended`.**
- **TUI (`tui_gateway/server.py`):** `session.undo`→`hermes_undo.undo`; add `session.redo` JSON-RPC method→`hermes_undo.redo`; busy-guard (`session busy — /interrupt before /redo`, code 4009). **TUI `session.send` user-append path MUST call `on_user_message_appended`** (the surface most likely to bypass — explicitly wired and behaviorally tested in Phase 5).
- **Registry (`hermes_cli/commands.py`):** update `/undo` description to half-turn; add `CommandDef("redo", …)` with `args_hint="[N]"`.

---

## 6. Implementation Phases

> **Structural-grep convention (pass-6 RC-1) — applies to EVERY grep/structural assertion in this section.** Six structural checks do load-bearing verification work (caller enumeration §5.2 RC-4 + its four caller-pattern greps; clear-redo-in-core §5.3; `test_no_active0_reconstruction` §5.3(b); `test_no_feature_path_emits_adjacent_assistant` Phase 2). A "no path does X" grep passes **vacuously** if its pattern doesn't match how the real code spells X. So each such test MUST either: (a) carry a **positive control** — a planted fixture string that the same pattern is asserted to MATCH (proving the pattern isn't dead) before asserting it finds zero hits in the real tree; OR (b) be implemented as an **AST/import-based** check instead of text grep where feasible (e.g. caller enumeration = import the module and assert the call sites; `active=0`-reconstruction = assert no undo/redo function calls `get_messages(include_inactive=True)` / `restore_rewound` by reference, not by spelling). The exact regex (case- and whitespace-insensitive: `active\s*=\s*0`, `active\s+IS\s+0`, etc.) must be named in the test. A grep without a passing positive control is not acceptance evidence.

### Phase 1 — Core half-turn computation + DB generalization
What ships: `compute_half_turn_target()` in new `hermes_undo.py`; `rewind_to_message(require_user_role=False)` returning `rewound_ids`; idempotent `restore_ids()` + `redo_count` column.
- *Unit/script check:* table-driven test of `compute_half_turn_target` over crafted message lists — user-only, user+assistant, **`assistant(tool_calls)+tool+tool+assistant(text)`**, **partial assistant half-turn (`assistant(tool_calls)+tool`, no final assistant text)**, multiple consecutive user msgs, **an adjacent multimodal+text user pair left un-merged (pass-7 RC-1 — assert `/undo 1` and `/undo 2` over a tail `…A,U(multimodal),U(text)` land on the run's group-start and the preceding assistant group-start respectively, proving the pair is ONE half-turn)**, N clamping. Assert exact target id per case; **specifically assert the partial-half-turn target id IS the `assistant(tool_calls)` row (group start), NOT the `tool` row** (pass-2 B-3).
- *E2E/integration check:* against a real temp `SessionDB`: **(B-1, pass-5) drive `compute_half_turn_target` over rows as actually persisted and read back** — insert the `assistant(tool_calls)+tool+tool+assistant(text)` and the partial-half-turn (`assistant(tool_calls)+tool`) fixtures via the real `add_message` path, read them back with the **id-bearing getter** (`get_messages` `SELECT *`, active-only — see §5.1 input-source note; `get_messages_as_conversation` does NOT return row `id` and so is NOT the grouping input), and assert the computed target id IS the persisted `assistant(tool_calls)` row's real id (group-start), NOT the `tool` row. This puts the highest-risk primitive under a real-row assertion, not a hand-crafted-dict one. Then `rewind_to_message(require_user_role=False)` rewinds to that group-start and returns the exact `rewound_ids`; `restore_ids(those ids)` reactivates **exactly and completely** the group (round-trip on the partial-half-turn fixture leaves no `assistant(tool_calls)` without its `tool` result, pass-2 B-3). `restore_ids` on an already-`active=1` id is a no-op (D-11a idempotency). **RC-A characterization test (pass-4):** feed the real `repair_message_sequence` (a) two adjacent plain-text user msgs → assert merged with `\n\n`; (b) two adjacent user msgs, one multimodal/list → assert NOT merged (and see RC-1 §3 Invariant 2 for the alternation consequence + its end-to-end guard); (c) two adjacent assistant msgs → assert left intact. (Real DB path = required.) **N-A:** assert `rewound_ids` per op is bounded by one half-turn's row count (well under SQLite's `SQLITE_MAX_VARIABLE_NUMBER`); the `IN (...)` splat needs no chunking at this scale — noted so a future contributor doesn't wire an unbounded splat.
- *Negative/adversarial:* `rewind_to_message` with `require_user_role=True` (default) still raises `ValueError` on a non-user target. Empty session → no-op. **RC-4 (enumeration closed, AST-based per pass-7 RC-2):** the import/AST caller audit resolves the complete caller set to {def site, `undo_last`, `rewind_session`, tests} — no third consumer — and asserts each reads the return by key (no positional unpack / splat / alias / passthrough); return carries `rewound_count`/`target_message`/`new_head_id`/`rewound_ids`. **`restore_rewound` gains its deprecation docstring** (N-2′).
- *Verify with:* `python -m pytest tests/test_half_turn_target.py tests/test_state_rewind_generalized.py tests/test_restore_ids.py -o 'addopts=' -q` → all pass; the AST caller audit resolves exactly the expected caller set with all key-reads.

### Phase 2 — Redo wiring + shared per-session stack (core)
What ships: `hermes_undo.{undo,redo,on_user_message_appended}` + `UndoRedoState` holder + `UndoOp{n, rewound_ids}` (all in `hermes_undo.py`); `redo_count` bump in `hermes_undo.redo` (one site).
- *Unit/script check:* `test_stack_transition_table` drives the three D-7a interleaving sequences and asserts, **after each step, BOTH the exact `undo_stack`/`redo_stack` contents AND the exact active-id set** matching the D-7a worked traces (the concrete-id traces, pass-3 B-1′).
- *E2E/integration check:* `test_undo_redo_multi_op_identity` against a real `SessionDB`: `undo` ×3 with different N, then **three separate bare `/redo` calls**, asserting after each the active-id set equals post-`undo×2`/post-`undo×1`/pre-`undo`; final = original. `test_redo_n_single_command`: `undo×3` then one `/redo 3` asserts only the final pre-`undo` set (pass-2 B-2). **`test_stale_op_unreachable` (pass-3 RC-3′; made structural pass-6 RC-3):** run *undo,undo,redo,undo* as the behavioral trace, then assert the **structural** invariant that actually generalizes — **the live `UndoRedoState` holder's `undo_stack`/`redo_stack` are the SOLE owners of any `UndoOp`** (no surface caches or retains an op reference outside the holder; assert via the holder being the only field that holds `UndoOp` instances), AND that `hermes_undo.redo`'s id source is **exclusively `undo_stack[-m:]`** by construction (the reactivation set is read only from the live stack, never from a discarded op or an `active=0` scan). The single trace then *exhibits* this (the discarded op2 is owned by no stack), but the **structural ownership + `undo_stack`-only-source assertions** are what prove the general no-corruption guarantee — not the un-provable "`restore_ids` is never called with discarded ids over all futures."
- *Negative/adversarial:* empty-stack `/redo` → "nothing to redo," no rows touched, **no `redo_count` bump**; `/redo 3` bumps `redo_count` **exactly once** (proven once, in `hermes_undo.redo`, not per-surface — pass-3 RC-4′). **N-2 (pass-5) clamp:** `/redo 9` against a 2-deep stack reactivates 2 ops and **still bumps `redo_count` exactly once** (the clamped-M path). **N-1 (pass-5) partial overlap:** `restore_ids` on a mixed set (some ids `active=0`, some already `active=1`) reactivates **only** the `active=0` members and leaves the rest untouched (the realistic mid-redo state), not just the all-`active=1` no-op case (D-11a). `restore_ids` on an already-`active=1` id is a safe no-op (D-11a). Two-user-after-undo merged by `repair_message_sequence`; **`test_repair_does_not_silently_merge_assistants`** asserts repair leaves assistant-after-assistant input unmerged; **`test_no_feature_path_emits_adjacent_assistant`** greps that no undo/redo/append path emits adjacent assistant rows (pass-3 B-2′ — honest artifacts, no unreachable-hazard strawman).
- *Verify with:* `python -m pytest tests/test_undo_redo_stack.py -o 'addopts=' -q` → pass; transition-table active-id assertions, both identity traces, stale-op-unreachability, and the two no-adjacent-assistant artifacts green.

### Phase 3 — CLI surface
What ships: half-turn `undo_last`, conditional prefill (D-4), `redo_last`, `/redo` dispatch, clear-redo-on-send.
- *Unit/script check:* CLI undo lands on correct half-turn boundary; prefill fires per D-4 (new active tail ends on USER). Cover **N=1 (prefill), N=2 (no prefill), and N=3 (prefill)** so the AC exercises the N-agnostic rule, not just two examples (pass-4 RC-B).
- *E2E/integration check:* drive a real CLI session object: send 2 turns, `/undo 1` (lands on user msg, prefill set), `/undo 1` again (lands on assistant msg, no prefill), `/redo` twice → original history restored. Assert via active message list. **RC-2 (per-surface clear-redo):** `/undo` → send a new user message **through the CLI's real send path** → assert `redo_stack` empty and `/redo` is a no-op (proves CLI routes through the core clear-redo hook, not bypasses it).
- *Negative/adversarial:* `/redo` before any `/undo` → no-op message; two-user-in-a-row after undo+type passes `repair_message_sequence` (no alternation error).
- *Verify with:* `python -m pytest tests/cli/test_undo_redo_half_turn.py -o 'addopts=' -q` → pass.

### Phase 4 — Gateway surface
What ships: half-turn `rewind_session`, `restore_session` (redo, bumps `redo_count` once), `_handle_redo_command`, agent-cache eviction on redo, clear-redo-on-send.
- *Unit/script check:* `rewind_session(n)` half-turn target; `restore_session` reactivates correctly and bumps `redo_count` once per command.
- *E2E/integration check:* simulate a gateway `MessageEvent` `/undo` then `/redo` against a real session store; assert active transcript restored and cached agent evicted both times. **RC-2 (clear-redo):** `/undo` → new user message **through the gateway's real message path** → assert `redo_stack` empty and `/redo` no-op. **RC-2-evict (pass-5, eviction is load-bearing — negative test):** `/undo` → `/redo` restoring history → run a **follow-up turn** and assert it sees the **restored** (not cached-stale) message list — i.e. with the eviction in place the next turn's context = the post-redo active set. (If a full follow-up-turn drive is too heavy, instead assert the cached agent entry for the session is *absent* after redo so the next turn rebuilds from the DB; state which form is used. Either way the mitigation must be proven load-bearing, not ceremonial.)
- *Restart degradation (pass-5 B-2):* **`test_cold_redo_after_restart_touches_nothing`** — instantiate gateway session state, `/undo` (rows → `active=0`), **drop/replace the in-memory `UndoRedoState` holder**, then `/redo`; assert (i) the **restart-specific** N-1′ i18n string fires (`rewind_count>0`, cold stack), distinct from the never-undone "nothing to redo", and (ii) `get_messages(include_inactive=True)` is unchanged (zero rows reactivated). **`test_no_active0_reconstruction`** — grep/structural: no undo/redo path reads `active=0` rows (`include_inactive=True`/`restore_rewound`/`active = 0` SELECT) to rebuild a redo stack.
- *Negative/adversarial:* `/redo` with no prior undo (never-undone session) → bare `gateway.redo.nothing` i18n string, no `redo_count` bump; new message between clears redo. (The restart-specific message is the §5.3(a) test above, kept distinct.)
- *Verify with:* `python -m pytest tests/gateway/test_undo_redo_half_turn.py -o 'addopts=' -q` → pass.

### Phase 5 — TUI surface
What ships: `session.redo` JSON-RPC method, half-turn `session.undo`, busy-guard parity, clear-redo-on-send.
- *Unit/script check:* `session.redo` handler returns success/`rewound` payload; busy session rejects with code 4009.
- *E2E/integration check:* drive the tui_gateway server methods: undo → redo round-trip restores active set; concurrent-write version guard intact. **RC-2:** `session.undo` → new user message **through the TUI's real `session.send` path** → assert `redo_stack` empty and `session.redo` no-op (this is the surface most likely to bypass the core hook — pass-2 RC-2 named TUI explicitly).
- *Negative/adversarial:* redo on busy session → 4009; redo with empty stack → graceful error, no DB mutation, no `redo_count` bump.
- *Verify with:* `python -m pytest tests/tui_gateway/test_undo_redo.py -o 'addopts=' -q` → pass.

### Phase 6 — Registry, help text, i18n, docs, parse-layer parity
What ships: `/redo` `CommandDef`; `/undo` description updated to half-turn; i18n strings (`gateway.redo.*`); help/docs updated.
- *Unit/script check:* command registry includes `redo`; `/help` lists it; existing `/undo` description string updated.
- *E2E/integration check:* drive `/redo` **and `/undo 2`** as **string commands through each surface's actual parser/dispatcher** (CLI `process_command`, gateway command handling, TUI method routing) and assert (a) `/redo` reaches the redo handler (RC-6: a registered command with no dispatch arm is a silent miss), and (b) **the same N reaches the helper across all three parsers** — i.e. `/undo 2` parses to N=2 on every surface, no off-by-one (pass-2 RC-3: parse-layer parity, distinct from helper-layer parity in Phase 7).
- *Negative/adversarial:* `/redo bogus` (non-int arg) → invalid-count error per surface, no DB op. **N-3 (pass-5) degenerate ints:** `/redo 0` and `/redo -1` are pinned to a single explicit behavior across surfaces — **treated as a no-op with the "nothing to redo" message** (M clamps to ≥1 only when there are ops; ≤0 never pops), asserted per surface so the `M=0` clamp boundary can't drift into an off-by-one.
- *Verify with:* `python -m pytest tests/ -k "command_registry or undo or redo" -o 'addopts=' -q` → pass; manual `/help` shows `/redo`.

### Phase 7 — Cross-surface helper parity (RC-3)
What ships: one parity test defending D-9's shared-helper promise. **Scope (pass-2 RC-3): this proves SHARED-HELPER parity, NOT command-parse parity** — the latter is Phase 6's job.
- *Unit/script check:* `Not applicable: single integration assertion.`
- *E2E/integration check:* feed a message list **persisted into a temp DB and read back via the id-bearing `get_messages` getter (the B-1 fixture shape, pass-6 N-3 — NOT a hand-crafted list, which is exactly what hid the real-row grouping bug)** through all three surfaces' undo entry points (CLI `undo_last`, gateway `rewind_session`, TUI `session.undo`) with **N already parsed** and assert **identical resulting active-id sets**; repeat for redo. **N-B (with pass-5 B-1 correction):** also assert all three surfaces obtain their active-message list via the **same `SessionDB` getter with the same args — `get_messages(session_id, include_inactive=False)` (the id-bearing getter, per §5.1 input-source note), NOT `get_messages_as_conversation`** (which omits row `id`) — so input-construction can't silently diverge before the shared helper. Combined with Phase 6's parse-layer parity, this gives true end-to-end parity without overclaiming.
- *Negative/adversarial:* `Not applicable.`
- *Verify with:* `python -m pytest tests/test_undo_redo_surface_parity.py -o 'addopts=' -q` → pass.

### Phase 8 — Full-suite regression (baseline-pinned, pass-3 N-3′)
- *Verify with:* capture `python -m pytest tests/ -o 'addopts=' -q` pass/skip counts on the **pre-PR commit** as the baseline; Phase 8 asserts pass count ≥ baseline, **0 new failures, 0 newly-skipped tests**. (`scripts/run_tests.sh` for convenience, but the baseline-delta is the gate — "full suite green" alone can hide a newly-skipped test.)

---

## 7. Security, Privacy, Ops, Observability

- **No new credentials, network calls, or external surfaces.** Pure local session-state manipulation.
- **No secret exposure:** undo/redo echo/prefill the user's own prior message text only (existing behavior); no new data surfaced.
- **Audit retention preserved:** soft-delete semantics unchanged; redo is reversible reactivation.
- **Observability:** existing `rewind_count` bump on undo; **`redo_count` bump once per successful `/redo` command** (D-12, command-layer not primitive-layer, additive nullable column) — the deactivate/reactivate cycle is fully auditable. Debug-level log line on redo with the op count. **`redo_count` consumer (pass-6 N-2):** there is **no live reader** today (no `/stats` surface, no dashboard) — the column is for **future forensic queries** and symmetry with `rewind_count`; stated explicitly so a later reviewer doesn't hunt for a missing consumer.
- **N-1 (RESOLVED → ship in Phase 4, pass-3 N-1′):** cold-restart `/redo` emits "nothing to redo (redo history doesn't survive a restart)" when `rewind_count` shows undos happened this session but the in-memory stack is cold; genuine empty-stack `/redo` emits the bare "nothing to redo." Committed (two-line conditional in the gateway redo handler), not left as "ship if cheap" (the conditional that silently doesn't ship — same discipline applied to N-2).
- **N-2 (RESOLVED → docs-only, pass-2):** the D-2 behavior-change notice is **cut to docs-only** — no first-run runtime notice. (Pass-2 flagged "ship if cheap" as the kind of conditional that silently becomes "not shipped." Decided: docs/CHANGELOG only.)
- **N-C (pass-4) — NULL-safe counters on pre-existing sessions:** `redo_count` is additive nullable; existing rows are `NULL`. The N-1′ cold-restart conditional reads `rewind_count`, which must be `COALESCE(rewind_count,0)`-safe on pre-existing sessions (confirm the existing `rewind_count` read path already handles `NULL`; `restore_ids`'s `redo_count` bump uses `COALESCE(redo_count,0)+1` like `rewind_count`).
- **Rollback:** feature is additive (`/redo` new; `/undo` behavior change is the only contract delta). Revert = git revert the PR; the only schema change is the additive nullable `redo_count` column (safe to leave or drop).

---

## 8. Risks & Mitigations

- **R1 — Half-turn boundary miscount with tool messages.** Tool results interleave; mis-grouping lands on the wrong message. *Mitigation:* table-driven unit tests with tool-message fixtures (Phase 1); `party(tool)=assistant` rule explicit.
- **R2 — `rewind_count` semantics drift.** Generalizing the target could double-count or skip the counter. *Mitigation:* keep the existing increment logic untouched; test counter increments once per undo.
- **R3 — Redo resurrects an incoherent branch.** If clear-on-new-message is missed on any surface, redo could splice a stale assistant reply after a new user message. *Mitigation:* invariant + per-surface negative test (Phase 3/4/5); single documented clear-redo call site per surface.
- **R4 — Restart loses redo silently and confuses the user.** *Mitigation:* fail-closed "nothing to redo" on cold stack (D-7); never guess.
- **R5 — Behavior change surprises existing `/undo N` users.** *Mitigation:* update help/docs/registry description (Phase 6); call out in PR description; the change is the user's explicit intent.
- **R6 — Three-surface drift** (one surface implements half-turn, another stays turn-based). *Mitigation:* shared core helper (D-9); parity tests per surface.

---

## 9. Open Questions

All blocking questions resolved during review (Pass 1). For the record:

- **Q-1 (RESOLVED → D-7a):** Stack transition table is fixed in this PRD — paired `undo_stack`/`redo_stack`, `/undo` clears redo, `/redo` moves ops between them. No longer deferred to writing-plans.
- **Q-2 (RESOLVED → D-7):** `redo_marker` / cold-redo reconstruction **dropped**. Cold stack = "nothing to redo." No schema marker.
- **Q-3 (RESOLVED → no):** `/redo` does not prefill — redo restores history as-is; prefill is an undo-only affordance (D-4).
- **N-3 placement (RESOLVED):** `compute_half_turn_target` and the `UndoRedoState` holder land together in the shared core in Phase 1/2 — not relocated later.

Remaining genuinely-open: **none blocking.** (N-2 first-run notice is a NICE-TO-HAVE, tracked in §7.)

---

## 10. Acceptance Criteria

- [ ] `/undo 1` backs up exactly **one half-turn** (lands on the user's last message). Evidence: `tests/cli/test_undo_redo_half_turn.py::test_undo_one_lands_on_user` passes.
- [ ] `/undo 2` lands on the assistant's penultimate message with the user's last message discarded (turn handed back to user). Evidence: same test file, `::test_undo_two_lands_on_assistant`.
- [ ] Prefill rule (N-agnostic, D-4): prefill iff the new active tail ends on a USER message — verified for N=1 (prefill), N=2 (no prefill), N=3 (prefill). Evidence: `::test_prefill_conditional`.
- [ ] `repair_message_sequence` asymmetric merge contract characterized against the real function: user+user plain-text→merged, user+user with multimodal→not merged, assistant+assistant→intact. Evidence: `::test_repair_merge_contract`.
- [ ] `/redo` (bare) reverses exactly the last `/undo` operation; active message set is identical to pre-undo. Evidence: `tests/test_undo_redo_stack.py::test_undo_then_redo_identity`.
- [ ] **Stacked undo/redo is order-symmetric (single redos, LIFO):** `undo` ×3 with different N then **three separate bare `/redo`** restores, after each, the post-`undo×2`/post-`undo×1`/pre-`undo` active-id set; final = original. Evidence: `tests/test_undo_redo_stack.py::test_undo_redo_multi_op_identity`.
- [ ] `/redo 3` (one command) reverses three prior `/undo` operations, landing on pre-`undo` (no intermediates), and bumps `redo_count` exactly once. Evidence: `::test_redo_n_single_command`.
- [ ] The three interleaving sequences (undo,undo,redo,undo / undo,redo,undo / undo×3,redo 2,undo) produce the exact stack states **AND the exact active-id set after each step** per D-7a. Evidence: `::test_stack_transition_table`.
- [ ] Typing a new message after `/undo` clears the redo stack (at one core call site); subsequent `/redo` is a no-op and diverged rows stay `active=0`. Evidence: `::test_new_message_clears_redo` + grep showing the clear-redo call is in core; **plus per-surface** `redo_stack`-empty-after-real-send tests (Phase 3/4/5 RC-2).
- [ ] After `/undo` to a user half-turn + a new user message, `repair_message_sequence` yields no two consecutive same-role messages. No feature path emits adjacent assistant rows, and repair does not silently merge assistant-after-assistant input. Evidence: `::test_two_user_messages_merged_no_alternation_error`, `::test_no_feature_path_emits_adjacent_assistant`, `::test_repair_does_not_silently_merge_assistants`.
- [ ] Stale-op unreachability (**structural, pass-6 RC-3**): the live `UndoRedoState` holder is the sole owner of any `UndoOp` (no surface retains an op reference), and `hermes_undo.redo`'s reactivation id source is exclusively `undo_stack[-m:]` — never a discarded op or an `active=0` scan. The *undo,undo,redo,undo* trace exhibits it. Evidence: `tests/test_undo_redo_stack.py::test_stale_op_unreachable`.
- [ ] Half-turn boundary with tool messages, **on real persisted rows (pass-5 B-1):** fixtures inserted via the real `add_message` path and read back via the **id-bearing `get_messages`** getter; the partial-half-turn target id IS the persisted `assistant(tool_calls)` group-start row's real id (not the `tool` row), and undo→redo round-trips the complete group with no dangling `tool_calls`-without-result. The grouping primitive is asserted over real rows, not hand-crafted dicts. Evidence: `tests/test_half_turn_target.py::test_tool_message_boundaries_real_rows`, `tests/test_restore_ids.py::test_partial_half_turn_roundtrip`.
- [ ] **Multimodal alternation + mechanism (pass-5 RC-1, pass-6 B-1′):** `/undo` landing on a user message with non-`str` (multimodal/list) content uses D-13 REPLACE semantics. Assert BOTH: (a) **mechanism** — the multimodal user row's id IS in the op's `rewound_ids` and is `active=0` after undo (proving REPLACE fired, not some other masking), and the new tail ends on the prior assistant row so D-4 yields **no prefill** (armed-empty); (b) **outcome** — after a new user message the provider-bound payload has **no two adjacent user rows**. A plain-`str` control case in the same test does NOT lower the target (its user row stays active, append+merge path). Evidence: `tests/test_undo_redo_stack.py::test_multimodal_undo_replace_mechanism`, `::test_multimodal_undo_no_adjacent_user`.
- [ ] **Cold redo after restart (pass-5 B-2):** undo → drop the in-memory `UndoRedoState` holder → `/redo` fires the **restart-specific** N-1′ message (distinct from never-undone "nothing to redo") AND reactivates **zero** rows (`get_messages(include_inactive=True)` unchanged); and no surface reconstructs a stack from `active=0` rows. Evidence: `tests/gateway/test_undo_redo_half_turn.py::test_cold_redo_after_restart_touches_nothing`, `::test_no_active0_reconstruction`.
- [ ] **Eviction is load-bearing (pass-5 RC-2):** after `/undo`→`/redo`, a follow-up turn sees the **restored** (not cached-stale) history — or the cached agent entry is absent post-redo so the next turn rebuilds from the DB. Evidence: `tests/gateway/test_undo_redo_half_turn.py::test_redo_evicts_stale_agent`.
- [ ] `restore_ids` is idempotent: reactivating an already-`active=1` id is a safe no-op (no phantom redo). Evidence: `tests/test_restore_ids.py::test_restore_ids_idempotent`.
- [ ] `rewind_to_message` back-compat: return carries `rewound_count`/`target_message`/`new_head_id`/`rewound_ids`; no caller positionally-unpacks, splats, or iterates the return. Evidence: `tests/test_state_rewind_generalized.py::test_return_back_compat` + four grep patterns clean.
- [ ] Helper-layer parity: all three surfaces produce the **same** half-turn boundary (and redo result) for the same history+N. Evidence: `tests/test_undo_redo_surface_parity.py` passes.
- [ ] Parse-layer parity + dispatch: `/undo 2` parses to N=2 on every surface and `/redo` reaches its handler through each real parser. Evidence: Phase 6 dispatch/parse tests pass per surface.
- [ ] No data loss: rewound rows remain `active=0` (not deleted); grep shows no new `DELETE FROM messages`. Evidence: `::test_undo_is_soft_delete` + `git grep "DELETE FROM messages"` unchanged.
- [ ] Full suite green by **baseline-delta gate** (pass-5 RC-3, supersedes the old `exits 0` checkbox which Phase 8 itself calls insufficient): on the pre-PR commit capture `python -m pytest tests/ -o 'addopts=' -q` pass/skip counts as baseline; assert post-PR **pass count ≥ baseline, 0 new failures, 0 newly-skipped tests**. Evidence: Phase 8 baseline-delta check (not bare `exits 0`).

---

→ Next: `prd-review-pipeline` (3 passes, review+fix each), then Apollo implements directly (no swarm, per D-10).
