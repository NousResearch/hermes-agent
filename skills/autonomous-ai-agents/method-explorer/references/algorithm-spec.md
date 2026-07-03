# Pathfinder — a least-effort, repair-on-failure search

*Algorithm spec · v1.1 · provisional name "Pathfinder" (rename freely).*
*This is the canonical reasoning algorithm that the `method-explorer` skill implements. Concept-first: no implementation details here.*

---

## 1. Purpose

Given an **underspecified, high-level prompt**, reach the intent by committing to the
**cheapest happy path**, acting on it, and — *only when reality contradicts an
assumption* — repairing locally, re-questioning, or dissolving sub-goals, while
**learning from every failure**, until **success** or **genuine exhaustion**.

It optimizes for **least effort to a working result**, not for an optimal plan.

## 2. Core principle

**narrow → fail → broaden → narrow.**
Commit greedily to one cheap line; branch *only* on failure; when you branch,
re-anchor on the *desire* the dead step was serving (not the dead step itself);
learn from every death so it's never repeated; preserve the original intent
throughout; and quit only when there is truly nothing left to try.

## 3. Lineage (why this is sound, not improvised)

| Behavior | Established algorithm | Property borrowed |
|---|---|---|
| Cheapest happy path; act before full planning | **Real-time search (LRTA\*)** | commit & execute on the best-looking move |
| "Which method is cheapest/likeliest?" | **A\*** → **Tree of Thoughts** | heuristic = the **LLM's self-assessment** |
| Try a different method / look left-right | **HTN planning** / **AND-OR search (AO\*)** | swap to an alternative method for the same goal |
| Back up & repair | **D\* Lite** (incremental replanning) | repair *locally* at the parent; reuse the rest |
| Don't repeat dead ends | **LRTA\*** learning / **Reflexion** | record failure + reason; never re-expand it |
| Reopen questions; reflect on failure | **LATS** (Language Agent Tree Search) | reasoning+acting+planning+reflection in one tree |
| "Is this sub-goal even necessary?" | **Goal reasoning** (beyond classic pathfinding) | prune/relax the **sub-goal itself**, not just its method |

**Two deliberate augmentations beyond the textbooks:**
1. **LLM-as-heuristic** — no hand-coded `h()`; the model estimates remaining effort.
2. **Necessity / dissolution** — classic search holds the goal fixed and only swaps
   methods; this algorithm may *delete an instrumental sub-goal* when the parent
   desire can be met another way. (Never the root intent — see §10.)

**The cheap-by-design deviation:** unlike vanilla ToT/LATS, which explore breadth
*eagerly*, this is **greedy/depth-first** — breadth is paid for *only on failure*
(the real-time/incremental-search discipline). Tree-search robustness without the
breadth cost.

## 4. Vocabulary & data model

- **Intent** — the root desire + success criteria. **Invariant; never dissolved.**
- **Hard constraints** — inviolable; any candidate violating one is not a candidate.
- **Soft constraints** — ranked, relaxable; the release valves on failure.
- **Desire (why-ladder node)** — every node serves a parent desire. The edge reads:
  *"to satisfy [parent desire], do [method], assuming [assumption], which requires
  answering [question]."* The chain of desires up to the intent is the **why-ladder**.
- **Method** — a way to satisfy a desire; alternatives are OR-siblings.
- **Question → Answer** — the unit of work. Answers generate the next questions.
- **Progress (the pivot)** — an answer *progressed* iff it: met the postcondition,
  **or** opened ≥1 new viable branch, **or** eliminated a hypothesis that narrows
  the search. Otherwise it's a failure.
- **Tombstone** — a dead method/question, recorded **semantically** with its reason.
  Never re-expanded. (Generalizes the dead-set.)
- **Trail (journal)** — append-only log of Q→A + verdicts + reflections. It is
  simultaneously working memory, the audit trail, and the *fleshed-out prompt*.
- **Frontier** — dormant untried alternatives, kept so a later failure can grab one
  without re-deriving it.

## 4.1 Context & state — verdicts are conditional

A verdict is never absolute: "method X failed" means "method X failed *in context C*."
The same call can succeed in one context and fail in another, so the unit of record is
**`(context, tool-call) → verdict (+ delta)`**, and verdicts are **indexed by the
context they occurred in.**

Keep two structures, separate:
- **Active context = the current path stack (push / pop).** The search is depth-first,
  so the root→node path *is* a stack. Descending **pushes** a frame; backtracking
  **pops** it (its facts leave the active context). The active context is bounded by
  path depth, not by total history — that's what stops you restating the whole
  environment every step.
- **Trail = append-only memory.** Popped and dead frames stay in the trail for
  learning; they just leave the *active* context. Push/pop for "where we are";
  append-only for "what we've learned."

**Context = the relevant assumption-set, not the whole world.** Track only the facts a
verdict depended on — the method's load-bearing assumptions (§4). Each frame records
only its **delta** (what its action changed), STRIPS-style; the current context is the
fold of deltas along the path. You never repeat the full environment.

**Tombstones are context-scoped — this is D\* Lite.** A tombstone means "dead *under
assumption-set C*." When a cited killing-assumption **demonstrably changes**, the
tombstone goes **stale** and its branch may reopen — exactly how D\* Lite repairs a
path when an edge cost changes. Guardrail: reopen **only** on a demonstrated change to
a cited assumption, never on a hunch (or LRTA\* *scrubbing* returns).

**Context-as-key makes it a graph, not a tree.** If two different paths reach the
*same* context, they can **share** verdicts (a transposition table) — equivalent
states aren't re-evaluated. That's the "graph of where we are."

**Right-size it (caution).** For an LLM-driven loop you do **not** need a formal
symbolic world-model. "Context" is a compact, explicit **stack of active assumptions**
re-asserted each step; the model checks "does the killing-assumption still hold?"
in-context. Build a real state engine only if you need determinism.

**Belief vs. world.** This context model is the *belief state* you fully control
(push/pop is just how you shape the trail). The *real* environment is only partially
reversible: Hermes can snapshot/restore **files** on backtrack (`CheckpointManager`),
but external side effects (sent email, API writes) don't pop. Condition verdicts on
the context model freely, but treat real-world backtracking as best-effort — and
prefer **probes / simulation** for irreversible steps.

## 5. The heuristic (the LLM)

`cheapest_method(node)` = the LLM's estimate of **least remaining effort × likelihood
of success**, among the node's untombstoned methods. This estimate is what makes a
"happy path" selectable at all without expanding the whole tree.

## 5.1 The trail is the belief state (the update is in-context)

The trail is not just a log — it is the **belief state**. Each verdict appended shifts
the LLM's conditional distribution over the next action, because the next step is
generated *conditioned on the whole trail*. Appending the verdict **is** the update;
the model is the evaluator. (In-context learning behaves like implicit Bayesian
inference — no separate re-weighting machinery is needed; re-reading the evidence *is*
the re-weighting.)

That's the right mental model — with four caveats, because the in-context "posterior"
is **lossy, position-biased, uncalibrated, and prior-dominated when evidence is weak**:

- **Bounded + compacted.** Long trails hit the context limit and get summarized or
  dropped — the posterior silently loses terms exactly when the problem is hardest. →
  Keep the trail **curated and compact**: one tight line per cycle (outcome · reason ·
  what-it-rules-out), not the raw transcript.
- **Presence ≠ adherence; position matters.** Models over-weight the start and end and
  lose the middle ("lost in the middle"), and will re-try a dead approach sitting right
  there. → Keep a small **control header re-asserted each step** — intent, hard
  constraints, the **active context** (§4.1), the dead-set ("DO NOT retry: X because
  Y"), the open frontier — so load-bearing terms stay salient at the decision.
- **Weak verdicts → weak updates.** "That didn't quite work" barely moves the
  posterior. → A verdict must carry the **likelihood term**: outcome + *why* + what it
  *rules out* (+ confidence). Crisp verdicts = sharp conditioning.
- **Double-counting + cascades.** The model treats correlated evidence as independent
  confirmations and lets a confident-but-wrong intermediate contaminate downstream. →
  Tag verdicts with **confidence / source**, and let a later verdict **supersede** an
  earlier one with a logged reason (append-only, but revisable in effect).

Net: "the verdict is logged in context and the LLM evaluates it" is correct — *and* the
**shape** of what you log (compact, header-pinned, explicit-likelihood, revisable) is
what turns passive context into an effective update instead of a mushy one.

**Self-report < receipt.** A verdict the agent *asserts* ("SUCCESS") is the weakest
evidence — the model can write it whether or not it's true. A verdict backed by an
externally-observable **receipt** (a file read back, a row count, an exit code, a
returned artifact) is far stronger. So the loop earns trustworthy verdicts via
**predict → act → reconcile**: commit the candidates + chosen method + *expected*
outcome **before** acting (a prediction can't be a post-hoc rationalization), then
**reconcile** the actual result against a re-checked receipt. The expected-vs-actual
gap is the highest-value evidence in the system. Where a receipt can't be cited, mark
the verdict **UNVERIFIED**. (Implemented as the skill's *Decision Records*.)

## 6. The loop

```
SOLVE(prompt):
  intent = root desire of prompt            # INVARIANT — never dissolved (§10)
  tree   = { root: intent }                 # AND/OR, built LAZILY
  trail  = []                               # Q→A, verdicts, reflections
  node   = root
  loop:
    method = cheapest untombstoned method(node)       # LLM heuristic            [LRTA*/ToT]
    if method is None:                                # nothing left here
        tombstone(node); node = parent(node)          # BACK UP one rung         [HTN backtrack]
        if node is None: return EXHAUSTION            # climbed past root → no path
        continue
    answer = attempt(method)                          # ACT (real / proxy / simulated)
    trail.append(method.question, answer)             # APPEND evidence          [Reflexion]
    if progressed(answer):
        if intent satisfied: return SUCCESS
        node = next open sub-desire(answer)            # DESCEND: answers reveal next Qs
    else:
        tombstone(method); reflect(trail, why)         # LEARN — never retry      [LRTA* update]
        node = REPAIR(node)                            # see §8
```

## 7. Two modes (both produced by one step: "given prompt + trail, what next?")

- **Descend** — the answer progressed and unlocked deeper unknowns → next questions
  go *down* (flesh out the prompt).
- **Repair** — the answer was a dead end → next questions go *sideways/up* (§8).

The answer's *valence* (progress vs. failure) is the only thing that steers which
mode the same "what next?" step takes.

## 8. The failure / repair protocol (`REPAIR(node)`)

Repair **locally before climbing** (D\* Lite discipline):
1. **Local sibling** — is there another untombstoned method for the *same* desire
   (`node`)? Generate candidates if needed, broad before deep:
   - **similar-but-different** — a variant of the same approach;
   - **indirect** — satisfy the desire's effect via a different mechanism;
   then return `node` (loop will pick the cheapest).
2. **Necessity / dissolve** — if no viable sibling: is `node` actually *required* by
   `parent(node)`, or can the parent be satisfied without it? *(Goal reasoning — the
   high-leverage move; can delete a whole subtree.)*
   **Default mode = propose-and-log:** record the dispensability judgment to the trail
   ("`node` appears unnecessary for `parent` because …") and climb via back-up,
   rather than autonomously pruning. Flip to **act** to let it prune the subtree and
   commit to "`parent` is satisfiable without `node`."
3. **Upstream jump** — if **K consecutive siblings** under the same parent have
   tombstoned (default **K = 5**), stop trying siblings: the *parent decision* is the
   suspect → climb (and consider relaxing the lowest-ranked soft constraint there).
4. **Back up** — otherwise tombstone `node` and climb; `parent = None` ⇒ §10.

**On every climb (steps 2–4), run RE-QUESTION first (hindsight).** Before committing
to a method at the parent, re-derive the parent's question-set from the *full trail*.
Ask: *given everything we now know, were the questions we asked under here the right
ones — and is there a better question now available than any we originally
enumerated?* Commit to the best. Tombstoned questions stay dead (don't "rediscover"
them); bound this to **one revision pass per climb** so it doesn't re-litigate the
whole tree. This is the LRTA\*-style heuristic update applied at the *question* level:
climbing isn't just "try another method," it's "re-aim, using the evidence."

## 9. Learning & anti-repeat

- Tombstones are **semantic** (match on meaning + reason, not exact string), so
  near-duplicates of a dead approach are also skipped.
- Every failure appends a **reflection** (what broke, why, what it rules out) to the
  trail; future question-generation reads it.
- **Never re-expand a tombstone.** This is the guard against LRTA\*'s documented
  *scrubbing* (thrashing in a "heuristic depression").

## 10. Termination

- **SUCCESS** — intent's success criteria met. Stop; the trail is the solution+audit.
- **EXHAUSTION** — climbed past the root with no remaining method **and** all soft
  constraints relaxed. Emit a structured no-viable-path report (blocking hard
  constraint + unblock condition). **Never fabricate** a result.
- **GUARD-HALT** — a budget backstop fired (max depth/methods/iterations). **Distinct
  from exhaustion**, logged as such, with the smallest bump that would continue.

Never declare failure while a live alternative or an unrelaxed soft constraint remains.

## 11. Invariants

1. Never violate the **intent** or any **hard constraint**.
2. Never dissolve the **root intent** — only *instrumental* sub-goals (§8.2).
3. Never re-expand a **tombstone** whose killing-context still holds (§4.1).
4. **Repair locally before climbing** (cheap before expensive).

## 12. Tunable parameters (decisions — current defaults)

| Parameter | Default | Note |
|---|---|---|
| `progress` definition | postcondition met OR new branch opened OR hypothesis eliminated | the §4 pivot |
| question batch size | 1–3 per round | small, because each answer reshapes the next |
| `K` (siblings → upstream jump) | **5** | §8.3 |
| climb policy | one rung at a time | least commitment |
| re-question on climb (hindsight) | on · 1 pass/climb | §8 RE-QUESTION |
| dissolution mode | propose-and-log | §8.2 · flip to *act* to auto-prune |
| `attempt` mode | real · proxy · simulated | proxy = recursive delegation; simulated = test harness |
| budgets | max depth / methods / iterations | guard-halt thresholds (§10) |
| trail form | curated lines (not raw transcript) | §5.1 — survive compaction |
| control header (re-asserted each step) | intent · hard constraints · active context · dead-set · frontier | §5.1 / §4.1 |
| context encoding | deltas + push/pop along the path | §4.1 |

## 13. Worked trace (illustrative)

**Intent:** be able to start the project Friday with legal cover.
1. Happy path (assume they'll sign our paper): send standard contract → **fails**
   (clause rejected). Tombstone; reflect.
2. REPAIR §8.1 local: redline the clause; or use *their* template (indirect). Both stall.
3. REPAIR §8.3 upstream jump → climb to the desire "agreement on terms," then again
   to the **root**: "legal cover to start Friday." §8.2 necessity bites — a signed
   contract was only an *assumed method*.
4. Fan out at root: a **letter of intent** / **PO** satisfies "legal cover to start"
   without ever solving "get them to sign our paper." Commit → descend → SUCCESS.

The failed sub-question was never the point; climbing the why-ladder bypassed it.

## 14. Out of scope (for now)

- **Proxy-chain recursion depth** (the "inception" idea) — maps to Hermes
  `delegate_task` with `role="orchestrator"`; flat by default, needs `max_spawn_depth`
  raised. Add cycle + fidelity-decay guards before enabling.
- **The simulator** — an optional *test harness* for this algorithm (scenario-driven
  tombstones), not part of the algorithm itself.

---

### Lineage references
D\* Lite (Koenig & Likhachev) · Learning Real-Time A\* (LRTA\*) · A\* · Tree of
Thoughts · Language Agent Tree Search (LATS) · Reflexion · HTN / AND-OR (AO\*)
planning · goal-driven autonomy / goal reasoning.
