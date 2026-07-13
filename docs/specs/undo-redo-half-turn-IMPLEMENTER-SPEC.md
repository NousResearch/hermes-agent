# Implementer Spec — Half-Turn `/undo` + `/redo` Stack

> Distilled, self-contained build instructions. No review provenance. The
> annotated PRD (`undo-redo-half-turn-PRD.md`) is the review record; THIS is
> what you implement from. Every invariant states its own "why" inline.

**Repo:** `Kyzcreig/hermes-agent` (fork) → PR to `NousResearch/hermes-agent`
**Surfaces:** CLI (`cli.py`), Gateway (`gateway/run.py` + `gateway/session.py`), TUI (`tui_gateway/server.py`)

---

## What we're building

1. **`/undo N` operates on half-turns** (was: whole turns). A *half-turn* = one
   party's run of consecutive messages. User message(s) = 1 half-turn; an
   assistant reply + its `tool_calls` + `tool` results = 1 half-turn. `/undo 1`
   lands on the user's last message (edit & resend); `/undo 2` lands on the
   assistant's penultimate message (user's turn to reply); alternating.
2. **New `/redo [N]`** — text-editor stack. Bare `/redo` reverses the single
   most-recent `/undo` (whatever N it carried). `/redo N` pops N undo operations.
   Typing any new message clears the redo stack.

Both across all three surfaces, driven by one shared core module.

---

## Core module: `hermes_undo.py` (new)

Holds the boundary logic + stack state + the three entry points. NOT in
`hermes_state.py` (that's DB only).

### `compute_half_turn_target(active_messages, n) -> int`
- Pure function. Input: ordered active-message list **including each row's `id`**
  (so its caller must use the id-bearing getter — see "DB getter" below).
- Walk backward, grouping consecutive same-*party* messages into half-turns:
  `party(user) = "user"`; `party(assistant) = party(tool) = "assistant"` (tool
  messages belong to the assistant half-turn that spawned them).
- **`party()` is total over roles (don't crash on `system`/`developer`/unknown).**
  Map any non-`user`, non-`assistant`, non-`tool` role (e.g. the `system` row at
  id=1, a provider-injected `developer` role) to a distinct party **`"other"`**.
  **One clamp rule (resolves the boundary ambiguity):** when counting the N
  group-starts AND when clamping, **`"other"` groups are NOT counted and are never
  returned as a target; `N` clamps to the oldest *non-`other`* half-turn.** The
  function must not `KeyError`. **If the active list has zero non-`other`
  half-turns** (e.g. only the system row, fresh session), **`compute_half_turn_target`
  returns `None`** (the sentinel — NOT an int target). `undo()` checks `if t is
  None:` **before** the D-13 step-3 read (so `tail = greatest id < t` never
  dereferences a `None` `t`) and short-circuits to the no-op return
  `{rewound_ids: [], prefill_text: None}` with a "nothing to undo" message — it
  does NOT call `rewind_to_message`, does NOT push an `UndoOp`, and **touches
  NEITHER stack (the `redo_stack` is NOT cleared on a no-op** — a user with a
  pending redo who fat-fingers `/undo` on a cleared session keeps it). Phase 1(d)
  exercises this zero-non-`other` branch and asserts the `None` return + the no-op
  contract. Phase 1(d) also pins the clamp: `/undo 4` over `[system,U,A,U]` (non-
  `other` half-turns = `{U,A,U}`) clamps to the **oldest non-`other` = the first
  `U` at id=2, never the id=1 system row**.
- **General grouping invariant:** **any maximal run of consecutive same-`party()`
  rows = exactly ONE half-turn, regardless of how the run arose** — normal
  alternation, a `repair_message_sequence` user-merge, a multimodal user pair left
  *un*-merged (D-13 can leave two adjacent user rows active), or an
  interrupted/partial assistant turn. The walk is purely "a `party()` change
  starts a new half-turn." So a tail `…A, U(multimodal), U(text)` is **one** user
  half-turn; `/undo 1` lands on the earliest of that run, `/undo 2` on the prior
  assistant group-start.
- Count N half-turn group-starts from the end; return the **id of the first
  (earliest) message of the Nth group**. For an assistant half-turn that is the
  `assistant` row bearing `tool_calls` (or the lone assistant text row) — **never
  a mid-group `tool` row**.
- Partial assistant half-turn (`assistant(tool_calls)+tool`, no final assistant
  text — interrupted turn): group still starts at the `assistant(tool_calls)`
  row, so the whole group is cut atomically.
- **Tool-row monotonicity precondition (required for the id-range cut to be
  correct):** `tool` rows are always persisted with `id >` their owning
  `assistant(tool_calls)` row (normal streaming-persist order). The grouping walk
  itself is order-independent (it groups by `party()`), but `rewind_to_message`'s
  cut is an **id-range** (`id >= target`), so a `tool` row with an id *below* its
  assistant would be left active as an orphan (provider rejects a `tool` not
  following its `assistant`). **Enforcement (NOT a future comment — shipped now):**
  `rewind_to_message` (or the `undo` core just before it) **raises** a clear error
  if the cut would leave an orphaned `tool` row active (an active `tool` whose
  matching `assistant(tool_calls)` is at `id >= target` and thus deactivated while
  the tool is `id < target`). Fail-loud beats silent corruption — a violating
  provider surfaces immediately instead of poisoning the next provider call. Phase 1
  fixture (e) tests BOTH: (i) the precondition holds on normal persisted data, and
  (ii) a deliberately violating `tool`-before-`assistant` fixture makes the cut
  **raise**, not silently orphan. (A future provider that legitimately reorders can
  then upgrade the raise to the union-widening fallback; until then, raising is the
  safe default.)
- `N` clamps to the oldest half-turn if it exceeds the count. **Returns the
  inclusive target id only** — the actual deactivation (`id >= target`) is performed
  by `rewind_to_message`, not this pure function (no side effects here).
- **Keys on `role` ONLY. Never reads `content` type.** (The one content-type
  read in the whole feature is D-13, applied AFTER this returns — see below.)

### `UndoOp` + `UndoRedoState`
- `UndoOp = {n: int, rewound_ids: list[int]}` — the explicit row ids this op
  deactivated (NOT a range, NOT just a count).
- `UndoRedoState`: per-`session_id` holder with `undo_stack: list[UndoOp]` and
  `redo_stack: list[UndoOp]`. **This holder is the SOLE owner of any `UndoOp`** —
  no surface caches an op reference. (This sole-ownership is what makes discarded
  ops unreachable; see no-corruption invariant.)

### Entry points (the ONLY functions that mutate stacks/state)
- **`undo(session_id, n)`**:
  1. `msgs = get_messages(session_id, include_inactive=False)` (the id-bearing
     active list, `ORDER BY id`).
  2. `t = compute_half_turn_target(msgs, n)` (role-based target id). **If `t is
     None`** (zero non-`other` half-turns — fresh/system-only session), short-circuit
     immediately: return `{rewound_ids: [], prefill_text: None}` + "nothing to undo";
     touch neither stack (do NOT clear `redo_stack`), call nothing downstream.
  3. **D-13 multimodal adjustment.** Define `tail` precisely: **`tail` = the row in
     `msgs` (the active list from step 1) with the greatest `id < t`** — i.e. the
     row that will be the new active tail after the cut. (NOT "id-adjacent in the
     DB", which could be an `active=0` hole from a prior undo. It is positional in
     the active list.) If `tail` exists AND `tail.role == "user"` AND its `content`
     is **non-`str`** (list/multimodal), set `t = tail.id` (lower the target to
     swallow that user row, inclusive). Otherwise leave `t` unchanged. *This is the
     only content-type read in the feature; `compute_half_turn_target` never reads
     content.* Function of one row, independent of N.
     - **When D-13 actually fires (the precise reachable case — there is exactly
       one, and Phase 2 MUST test it firing):** the grouping invariant already
       swallows a *trailing* user run into the cut, so D-13 never fires on the
       cut's own trailing group. D-13 fires **only when the half-turn cut exposes a
       new tail that is itself a lone multimodal user half-turn.** Concrete:
       tail `[…, A1, U(mm), A2]` (where `U(mm)` is a one-row user half-turn — a
       single multimodal message between two assistant turns). `/undo 1`: group 1 =
       `A2`, so `t = A2.id`; `tail` = greatest active `id < A2.id` = `U(mm)`, which
       is a non-`str` user row → **D-13 fires**, lowering `t = U(mm).id` so both
       `U(mm)` and `A2` are cut (`rewound_ids == {U(mm).id, A2.id}` — the **complete
       two-element set**, which redo invariant 3 depends on) and the new tail is
       `A1` (no prefill, armed-empty).
       Without D-13, the new tail would be `U(mm)` and D-4 would try to prefill an
       un-editable image/audio block. **Reachable by normal operation** (not just
       hand-construction): a plain text turn followed by an image turn produces
       active list `[…, U_prev, A1, U(mm), A2]` — user types text → assistant replies
       (`U_prev,A1`), user sends an image → assistant replies (`U(mm),A2`). `/undo 1`
       then exposes `U(mm)` as the **pre-D-13 candidate tail** (the row step-3
       inspects); **D-13 fires, lowering `t` so the actual post-cut tail is `A1`,
       NOT `U(mm)`** (the whole point of D-13 is to prevent `U(mm)` from being the
       tail). The firing assertion is `new_tail == A1` and `prefill_text=None` —
       never `new_tail == U(mm)`. Phase 2 builds the fixture
       BOTH ways — directly via `add_message` AND by replaying that two-turn op
       sequence — so the firing test proves reachability-by-operation, not just by
       construction. **For arbitrary N, `tail` is by construction the greatest active
       `id < t` while the grouping walk only ever swallows rows `id >= t`; the two
       row-sets are disjoint by that strict inequality, so D-13 never re-cuts a row
       the grouping walk already took, for any N.** **If you cannot construct a
       fixture where D-13 changes the target, the branch is unreachable and must be
       deleted, not shipped as untested defensive code — but the `[…,A1,U(mm),A2]`
       /undo-1 case above IS that fixture, so it ships with a firing test.**
  4. `result = rewind_to_message(session_id, t, require_user_role=False)`.
  5. Push `UndoOp(n, result["rewound_ids"])` to `undo_stack`; **clear `redo_stack`**.
  6. **Prefill decision (D-4) — computed in CORE, rendered by surfaces.** The
     prefill source is the **new active tail = the greatest-id active row after the
     cut** (the same row step-3 inspected, unless D-13 lowered `t` to also cut it).
     **`prefill_text` = that surviving tail row's `content` IFF it is a USER row
     with `str` content; else `None` (armed-empty).** Note: for the canonical
     `/undo 1` on a completed turn (tail ends in the assistant reply), the half-turn
     cut removes **only the assistant reply**; the user's prior message SURVIVES as
     the new tail and is the prefill source — it is NOT a row in `rewound_ids`.
     `prefill_text` is never sourced from a cut/deactivated row. *No-prefill arises
     only when the surviving tail is an assistant row — either because N≥2 cut into
     an assistant half-turn, or because step-3 D-13 swallowed a multimodal user tail
     so the survivor is the prior assistant.* (You can't edit image/audio blocks as
     composer text anyway.)
  - **`undo()` return contract:** `{rewound_ids: list[int], prefill_text: str|None}`
    — the prefill decision is computed ONCE in core (step 6) and returned; each
    surface renders it: **CLI** fills the composer with `prefill_text`; **TUI**
    emits it in the `session.undo` JSON-RPC result; **gateway** reads only
    `rewound_ids` and leaves `prefill_text` present-but-unread (don't destructure in
    a way that errors on the extra key — it has no composer). This prevents the
    three surfaces from each re-deriving the tail and drifting; the surface wiring's
    "+ D-4 prefill" means "render the returned `prefill_text`," not "recompute it."
  - **Disjoint-ids guarantee (why stacked ops never overlap):** each `undo` reads
    its input from `get_messages(include_inactive=False)`, which **excludes rows a
    prior op already set `active=0`**. So no still-active row from a prior op is in
    a later undo's input; each op's `rewound_ids` is disjoint from every other live
    op's. (Phase 2 asserts disjoint `rewound_ids` across stacked ops.) This is what
    makes the LIFO redo symmetry hold and `restore_ids` idempotency a pure
    belt-and-suspenders.
- **`redo(session_id, m)`** (bare `/redo` passes `m=1`; the default is applied at
  the **parse layer**, so the core always receives an explicit int):
  - **Step 0 (guard FIRST, before any slicing):** if `m <= 0` → return "nothing to
    redo", no DB op, no bump. (Must precede the `min`/slice — `min(-1, len)` = -1
    and `undo_stack[-(-1):]` would be a wrong non-empty slice.)
  - `k = min(m, len(undo_stack))`. If `k == 0` (empty stack) → "nothing to redo"
    (restart-aware message per §Restart degradation; the `rewind_count` read for
    that branch is a DB read of the `sessions` row **inside `redo()`**, done **only
    on the cold `k==0` path** — the warm path with a non-empty stack never reads it,
    so it's off the hot path — so the core is the single decision site and surfaces
    never query it), no bump, no DB op.
  - **Pop `k` ops from the END of `undo_stack` (LIFO, top-down).** For each popped
    op (most-recent first), `restore_ids(session_id, op.rewound_ids)` and push it to
    `redo_stack`. The reactivation id source is **exclusively the popped slice
    `undo_stack[-k:]`** — never a discarded op, never an `active=0` scan. (Top-down
    pop is what makes N separate bare `/redo`s reproduce the LIFO intermediate
    states of N stacked undos, and one `/redo k` land directly on the pre-undo set.)
    **Fail-loud:** under the disjoint-ids + LIFO guarantees every popped op's rows
    are all `active=0` at pop time, so `restore_ids` must return exactly
    `len(op.rewound_ids)`; if it returns **fewer**, the disjoint-ids invariant was
    violated — **raise** (don't silently continue). Phase 2 asserts
    `reactivated == len(op.rewound_ids)` per popped op.
  - **Bump `redo_count` once** (regardless of `k`) since ≥1 op reactivated.
  - **NULL-content tail rendering (surfaces).** When the new tail is an
    `assistant(tool_calls)` row with `content=NULL` (reachable via `/undo 2` into a
    partial assistant group, or a redo landing there), surfaces render the
    confirmation **without** the tail's content body — show a role/op summary
    ("undone to assistant tool-call turn") only. **Never `str()` a `NULL` content
    into chat** (it prints `"None"` or `KeyError`/`TypeError`s a format path).
    Phase 4/5 assert the confirmation string for a NULL-content tail contains no
    stringified `None`.
  - **`redo()` return contract:** `{reactivated_count: int, new_tail_id: int|None,
    prefill_text: None}` — **redo NEVER prefills** (it restores prior state as-is;
    text-editor redo doesn't re-arm the composer). **`new_tail_id` = the greatest-id
    active row after the `k` reactivations, read via
    `get_messages(session_id, include_inactive=False)`** (the SAME tail-recompute
    undo's step 6 uses — never `max(rewound_ids)`, which differs when an older op
    sits below an unrelated active row); `None` only if the active list is empty
    (impossible post-redo, typed for totality). CLI shows a brief confirmation +
    the restored tail; TUI returns the payload in the `session.redo` JSON-RPC
    result so the view refreshes; gateway echoes a confirmation. All three render
    the same core return; none recompute.
- **`on_user_message_appended(session_id)`**: **clears `redo_stack` ONLY; leaves
  `undo_stack` untouched** (a new message after an undo must NOT discard undo
  history — you can still undo further). Ordering is immaterial to correctness (it
  only mutates in-memory redo state, not persistence), but call it **after** the
  new row is persisted so a reader never sees a cleared redo stack with the new row
  not yet committed. Every surface's user-append path MUST call this (TUI
  especially — its JSON-RPC handlers historically persist directly).

**Worked `str` prefill path (`/undo 1`, the canonical edit-and-resend):** tail
`[…, U(str), A]`. `/undo 1` cuts group 1 = the assistant reply `{A}` (and its tool
rows) → `rewound_ids = {A.id, …}`; the user row `U(str)` **SURVIVES** (`active=1`)
as the new tail → `prefill_text = U(str).content`. The user edits and **resends**,
appending a **new** user row `U(str')` after the still-active `U(str)`. Now there
ARE two adjacent active user rows → `repair_message_sequence` Pass-2 **merges** them
(both `str`) into one before the provider call (exactly D-3, the user's stated
model). No double-user-row reaches the provider. *(Contrast the multimodal case:
D-13 cuts the multimodal user row up front so it never survives to form a pair —
because Pass-2 would NOT merge it.)*

---

## DB layer (`hermes_state.py`)

### DB getter (the grouping input source)
- Use **`get_messages(session_id, include_inactive=False)`** (`SELECT *`,
  active-only, `ORDER BY id`) — it returns the row `id` column that
  `compute_half_turn_target` needs.
- **Index rows by key, never positionally** (`row["id"]`, `row["role"]`,
  `row["content"]`) — insurance against `SELECT *` column-order drift.
- **Do NOT use `get_messages_as_conversation`** for grouping — it returns the
  provider-format projection `{role, content, tool_calls, …}` and **omits `id`**.
- An assistant-with-tool-calls row is identified by a populated `tool_calls` JSON
  column, with `content` possibly `NULL`.

### `rewind_to_message(session_id, target_id, require_user_role=True)`
- Add the `require_user_role` param. When `False`, skip the `role != "user"`
  `ValueError` (half-turn callers land on assistant boundaries too).
- Everything else unchanged (soft-delete `id >= target`, bump `rewind_count`,
  return `rewound_count`/`target_message`/`new_head_id`) **plus a new additive
  return key `rewound_ids: list[int]`**.
- **`rewound_ids` is NORMATIVELY the rows this call transitioned `active=1`→`0`,
  never rows already inactive.** The existing impl already does
  `SELECT id … WHERE id >= target AND active = 1` then updates exactly those — so
  if a prior stacked undo left `active=0` holes above the new (D-13-lowered)
  target, those holes are **excluded** from `rewound_ids`. This is what makes the
  `undo` disjoint-ids guarantee hold even though the soft-delete is an id-range:
  the *range* may span prior holes, but the *returned id set* is only the rows
  this op actually flipped. (Confirm in the impl: `ids` is built from the
  `active = 1` SELECT; `rewound_ids` returns that list, not `id >= target`.)
  Additive — existing key-reading callers unaffected.

### `restore_ids(session_id, ids: list[int]) -> int` (new — the redo primitive)
- `UPDATE messages SET active=1 WHERE id IN (...) AND active=0`, scoped to
  `session_id`. Returns count reactivated.
- **Idempotent:** an id already `active=1` is silently skipped. On a mixed set
  (some `active=0`, some `active=1`) it reactivates **only** the `active=0`
  members.
- Does **NOT** bump `redo_count` (that's `hermes_undo.redo`'s job, one site).
- `rewound_ids` per op is bounded by one half-turn's row count — well under
  SQLite's variable limit; no chunking needed (but don't wire an unbounded splat).
- Leave `restore_rewound` in place but add a deprecation docstring: "DEPRECATED
  for stacked undo/redo — use `restore_ids`; `id >=` range clobbers stacked ops
  of differing N."

### `redo_count`
- New additive nullable column on `sessions`. Bump with `COALESCE(redo_count,0)+1`
  in `hermes_undo.redo` only. No live reader today (future forensic queries /
  symmetry with `rewind_count`). Safe to drop on rollback. **Counter asymmetry
  (document inline on the column):** `rewind_count` bumps **per `rewind_to_message`
  call** (i.e. per half-turn op); `redo_count` bumps **once per `/redo` command
  regardless of M**. Intentional — note it so forensic queries comparing the two
  don't assume parity.

### Back-compat caller audit (AST, not grep)
- `rewind_to_message` gains the `rewound_ids` return key. Verify by an
  **AST/import-based** caller audit (NOT a text grep): import the calling modules,
  resolve the complete caller set to {def site, `cli.py::undo_last`,
  `gateway/session.py::rewind_session`, tests} — no third consumer — and assert
  each caller reads the return **by key** (`result["..."]`/`.get`), never
  positional-unpacks, splats (`**result`), iterates, aliases the function, or
  passes the dict through. (A regex over consumption patterns can miss aliasing /
  kwargs passthrough; AST can't.)

---

## Stack transitions (text-editor semantics)

| Action | `undo_stack` | `redo_stack` | active-id Δ | DB op |
|--------|--------------|--------------|------------|-------|
| `/undo N` | push `UndoOp{n, rewound_ids}` | **clear** | remove `rewound_ids` | `rewind_to_message` deactivates them |
| `/redo M` (bare = 1) | pop `k=min(M,len)` from top (LIFO) | push each popped | add each op's `rewound_ids` back | `restore_ids(op.rewound_ids)` per op |
| new user message | unchanged | **clear** | append new row | append |

> Table caveat: the "new user message → `redo_stack` clear" is done by
> `on_user_message_appended` **after** the row is persisted and is in-memory only —
> it is NOT part of the DB append transaction. See the entry-point ordering note.

- `/undo` clears `redo_stack` (a fresh undo establishes a new branch; pending
  redo is stale).
- **No-corruption guarantee:** `restore_ids` is only ever called with the
  `rewound_ids` of an op currently on a stack. When `redo_stack` is cleared,
  discarded ops become unreachable (the holder was their sole owner; no surface
  retained a reference). Idempotency is belt-and-suspenders behind this, not the
  primary guarantee.

## Restart degradation
- In-memory stacks are lost on process restart (no persistence). A cold `/redo`
  with empty `undo_stack`:
  - If `rewind_count > 0` (undos happened this session, stack is cold) → emit
    **"nothing to redo (redo history doesn't survive a restart)"**. *Heuristic
    caveat:* `rewind_count` is also bumped by the legacy whole-turn `/undo` path
    and any other `rewind_to_message` caller, so a session that only ever used
    legacy rewind could see this message slightly misleadingly. Acceptable (it's a
    best-effort hint, never a correctness gate); don't gate behavior on it.
  - Else (never undone) → bare **"nothing to redo"**.
  - Either way: touch zero rows. **No reconstruction from `active=0` rows.** No
    surface may read `active=0` rows to rebuild a stack.
- **`/undo` is inherently restart-safe** (it reads DB active rows live and builds a
  fresh op); only `redo`'s pre-restart history is lost. A post-restart
  `/undo`→`/redo` pair works normally (the undo repopulates `undo_stack` with one
  op, which the redo then reactivates).

---

## Surface wiring (thin — all call the core)

- **CLI (`cli.py`):** `undo_last(n)`→`hermes_undo.undo` + D-4 prefill;
  `redo_last(n)`→`hermes_undo.redo`; register `/redo` dispatch
  (`elif canonical == "redo"`). User-send path calls `on_user_message_appended`.
- **Gateway (`gateway/run.py` + `session.py`):** `rewind_session(n)`→`undo`;
  `restore_session(m)`→`redo`; `_handle_redo_command` in `run.py`. **Evict the
  cached agent on BOTH `/undo` AND `/redo`** — either operation changes the active
  message set, so a cached agent holding the pre-op context would run the next turn
  against stale history. (The spec previously named only redo; undo needs it for
  the identical reason.) **Eviction scope:** the eviction-on-active-set-change rule
  is scoped to **`/undo` and `/redo` ONLY** — they mutate the active set *without*
  going through the send path. The **normal user-send path is OUT of scope** (it
  already incorporates the appended row into the agent by its existing mechanism;
  do NOT add an eviction there or you double-evict and may drop streaming state).
  `on_user_message_appended` on that path **only clears the redo stack; it does NOT
  evict.** Message-ingest path calls `on_user_message_appended`.
- **TUI (`tui_gateway/server.py`):** `session.undo`→`undo`; add `session.redo`
  JSON-RPC method→`redo`; busy-guard (reject with code 4009 if session busy).
  `session.send` user-append path MUST call `on_user_message_appended`.
- **Registry (`hermes_cli/commands.py`):** update `/undo` description to
  half-turn; add `CommandDef("redo", …)` with `args_hint="[N]"`.

---

## Invariants (each self-contained)

1. **No data loss.** Undo only flips rows to `active=0` (soft-delete); never hard-
   deletes. Redo flips them back. *Why:* audit/forensic retention is an existing
   guarantee. *Check:* rewound rows still present with `active=0`; no new
   `DELETE FROM messages`.
2. **Provider role-alternation preserved.** Providers reject two same-role
   messages in a row. After undo+new-message: two consecutive *user* messages are
   merged by `repair_message_sequence` Pass 2 — **but only when both `content`
   values are plain `str`** (multimodal/list is left unmerged to avoid mangling
   attachment structure). The multimodal gap is closed by D-13 (replace, not
   append). Never two assistant in a row (a new assistant row only follows a user
   message; undo appends nothing; redo reactivates a previously-coherent group).
   **Redo of a partial/interrupted assistant group** (`assistant(tool_calls)+tool`,
   no final text) reactivates the **entire** group atomically — its `rewound_ids`
   was the whole group — so no `tool`-without-`assistant` orphan is ever exposed.
   (Coherence *at redo time* — not just at cut time — follows from invariant 3's
   LIFO ordering: you can only redo in the reverse order you undid, so each
   reactivated group's neighbors are exactly what they were pre-undo.)
3. **Redo reactivates EXACTLY the matching undo's rows; stacked undo/redo is
   order-symmetric (LIFO).** `undo` then `redo` (no intervening message) →
   identical active set. `undo ×3` then 3 separate `/redo` restores each prior
   state in turn; `undo ×3` then one `/redo 3` lands on pre-undo directly. Uses
   `restore_ids(op.rewound_ids)`, never an `id >=` range.
4. **New user message clears the redo stack at ONE core point**
   (`on_user_message_appended`), not per-surface.
5. **Stack is per-session**, keyed by `session_id`, never bleeds across sessions.
6. **Never two assistant in a row** — structurally unreachable (no feature path
   appends an assistant row adjacent to another). *Check (distinct from inv. 2):*
   the Phase 2 `test_no_feature_path_emits_adjacent_assistant` AST/grep (with
   positive control) over `hermes_undo.py` + the three surfaces — no undo/redo/append
   path emits an assistant row without an intervening non-assistant row.

---

## Build phases (each ends green before the next)

1. **Core + DB:** `compute_half_turn_target`; `rewind_to_message(require_user_role)`
   + `rewound_ids`; `restore_ids` + `redo_count`. Test grouping over **real
   persisted rows** (insert via `add_message`, read via `get_messages`). Named
   fixtures: (a) `assistant(tool_calls)+tool+tool+assistant(text)`; (b) partial
   assistant half-turn; (c) **adjacent multimodal+text active user pair** (the
   grouping invariant's hardest case — assert `/undo 1` & `/undo 2` over
   `…A,U(mm),U(text)` prove the pair is ONE half-turn); (d) **an active `system`
   row present** (assert `party(system)="other"`, no `KeyError`, never targeted;
   `/undo 4` over `[system,U,A,U]` clamps to earliest `U` at id=2, never the id=1
   system row; **AND (d2) a separate `[system]`-only (or empty) fixture asserts the
   zero-non-`other` branch: `compute_half_turn_target` returns `None` and `undo()`
   short-circuits to `{rewound_ids:[], prefill_text:None}` + "nothing to undo",
   touching neither stack** — the `[system,U,A,U]` fixture has non-`other` half-turns
   so it does NOT exercise the `None` path; the `None` path needs its own fixture);
   (e) **tool-row
   monotonicity** — (i) assert no active `tool` row has an id below its preceding
   `assistant` in normal persisted data, AND (ii) a deliberately violating
   `tool`-before-`assistant` fixture makes the cut **raise** (not silently orphan).
   Also (f): `rewind_to_message` `rewound_ids` returns ONLY rows it flipped `1→0`
   (insert a pre-existing `active=0` hole above target, assert it's excluded — the
   disjoint-ids guarantee's primitive-level proof).
2. **Redo wiring + shared stack:** the three entry points + `UndoRedoState` +
   `UndoOp`; `redo_count` one-site bump. Test the transition table (stack contents
   AND active-id set per step) + identity round-trips + structural sole-ownership +
   **disjoint `rewound_ids` across stacked ops** + the redo `m≤0`/`k`-clamp/LIFO-
   pop-order cases + **`restore_ids` `IN(...)` is parameterized/bounded** (AST check
   with a planted positive control, per the test convention). **D-13 FIRING test
   (pass-10 BLOCKER + pass-11 round-trip):** active tail `[…,A1,U(mm),A2]`, `/undo 1`
   → assert `t` is lowered from `A2.id` to `U(mm).id`, **`rewound_ids == {U(mm).id,
   A2.id}` (the complete two-element set, not just `U(mm)`)**, new tail = `A1`,
   `prefill_text=None`. **Then the round-trip leg:** `/redo` → active set IDENTICAL
   to pre-undo (`U(mm)` and `A2` both `active=1` again, `A2` is the tail) — this
   proves D-13 restores correctly, not just cuts correctly. A plain-`str` control at
   the same position does NOT lower `t`. If no firing fixture can be built, delete
   the D-13 branch instead of shipping it untested.
3. **CLI surface** + D-4 prefill (N=1/2/3 — **assert `prefill_text` equals the
   EXACT content of the surviving new-tail user row**, not merely "a prefill
   occurred": N=1 on `[…,U(str),A]` → `prefill_text == U(str).content`; N=2 lands on
   an assistant tail → `prefill_text is None`; N=3 → the next surviving user row's
   content) + the `str` prefill→edit→resend two-user-rows-merged path + clear-redo-
   on-send.
4. **Gateway surface** + agent-cache eviction **on undo AND redo** + cold-restart
   test (drop holder → redo → restart message + zero rows; **assert BOTH branches**
   — `rewind_count>0` emits the "doesn't survive a restart" hint, `==0` emits bare
   "nothing to redo") + **eviction-is-load-bearing negative test, run BOTH after
   `/redo` AND after `/undo`** (concrete: after the op, run one turn and assert the
   agent's provider payload == post-op `get_messages_as_conversation`, i.e. the
   restored/cut set; with eviction removed it carries the stale pre-op set —
   the undo-side assertion is required because eviction was widened to undo).
5. **TUI surface** + `session.redo` + busy-guard + clear-redo-on-send.
6. **Registry/help/i18n/docs** + parse-layer parity (`/undo 2`→N=2 every surface;
   `/redo` reaches its handler) + degenerate args (`/redo 0`, `/redo -1`, bogus).
7. **Cross-surface helper parity** — feed the **persisted-and-read-back** B-1
   fixture through all three undo entry points; assert identical active-id sets +
   same getter used. **Plus `on_user_message_appended` call-site enforcement (AST):
   resolve each of the three surfaces' user-append sites and assert each invokes
   `on_user_message_appended`** (the "ONE core point, every surface calls it"
   invariant-4 must carry a failing test — catches a future surface that persists a
   user row without clearing the redo stack; TUI is the named risk).
8. **Full-suite regression** — baseline-delta gate: capture pre-PR pass/skip
   counts; assert pass ≥ baseline, **0 new failures, 0 newly-skipped** (any
   intentional new skip must be allow-listed in the baseline-delta config with a
   one-line reason, so the gate doesn't block a legitimate provider-gated skip).

**Test convention:** every structural/grep assertion ("no path does X") must
carry a **positive control** (a planted line the pattern is asserted to match) or
be an AST/import-based check — a grep with a too-narrow pattern passes vacuously.
Use case/whitespace-insensitive regexes (`active\s*=\s*0`, `active\s+IS\s+0`).

**Acceptance:** every test exercises the REAL changed path on REAL rows. A test
that can't fail on the bug it names is not evidence.
