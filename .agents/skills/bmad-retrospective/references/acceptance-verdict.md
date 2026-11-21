# Decide: Routing and the Acceptance Verdict

Phase 4. Turn the consolidated findings into two outputs: routed action items the human can act on, and an honest verdict on whether the epic met its acceptance criteria. This skill proposes; it does not auto-apply fixes or edit the project spec. The human decides what executes.

## Route each finding

Give every finding two independent dispositions:

- **What to do about this instance** — *fix now*, *defer*, or *accept as-is*. Fix-now findings become action items. Deferred findings carry enough context to be acted on later without re-investigation. Accepted deviations are recorded so later retros stop re-flagging them.
- **What would prevent the next one** — the upstream lesson: spec wording, story sizing, a missing convention or gate, or nothing. This is where a recurring finding becomes a process change rather than a one-off fix.

Findings from sub-agents or the team discussion are unverified reports, not established facts. Before an action item relies on one, re-check it against the primary source — reopen the file, the commit, the spec. A finding whose source does not hold up is dropped, not routed.

## Action items

Compile fix-now findings and process lessons into specific, owned action items. Each names what to change and who owns it. Two kinds are *proposed, not applied* in this version:

- **Remediation** — code fixes are written up as action items (or story-shaped work) for the normal dev loop to execute later. The retrospective does not run the dev loop itself.
- **Spec reconciliation** — where the as-built diverges from the spec, propose the reconciliation as an action item with the evidence attached. The human applies it to the project contract; an uncertain interpretation is never written into the spec automatically.

## Previous-retro follow-through

When a prior retro exists, check whether the action items it committed to were completed. Read `action_items` in `{implementation_artifacts}/sprint-status.yaml` and, for every entry belonging to an earlier epic that is not already `done`, record one line in the retrospective document's Previous-retro follow-through section:

- **How to address the item** — its `id`, exactly as the file spells it. Legacy entries written before ids existed have none; for those, record the item's `epic` (the integer in the file) plus its exact `action` text, character for character. One or the other is what Phase 5 needs to name the item at all.
- **Whether it landed** — with the source that shows it: the commit, the file and line, the test. An item you cannot point at is "no evidence found", not "not done" — the reader must be able to tell a checked item from an unchecked one.
- **The status it argues for** — `done` for a landed item, `in-progress` for one demonstrably underway, or nothing. A proposal, never a write.

That record is exactly what Phase 5's `--set-action-status` offer reads: the selector becomes the JSON, the evidence is what the user is asked to confirm, and the proposed status is written only if they confirm it. A run with no prior retro, or one whose `sprint-status.yaml` is unreadable or carries no `action_items`, records that there was nothing to follow through on — and which of those it was, so a missing file is never mistaken for "no outstanding items."

## The verdict

Judge the final state against the epic's declared acceptance criteria. If the epic declared none, profile the criteria from the diff and stories and mark the verdict as **profiled** rather than declared. Weigh verification results (the Phase 2 behavior check) and unresolved findings. Render one of:

- **Accepted** — criteria demonstrably met in the evidence, no blocking findings open, and **no unfinished stories** for this epic.
- **Accepted-with-open-items** — criteria met, but named findings remain deferred and tracked — still only when every story of this epic is `done`.
- **Rejected** — criteria not met, a blocking finding stands unresolved, **or any of this epic's stories is still not `done`**.

### Unfinished stories

`pending_stories` is authoritative for this epic's incomplete work, whichever mode produced it: sprint-status story keys in file order from `detect-epic`, or `stories.yaml` ids in list order whose artifact status is not `done`. When that list is non-empty:

- The **machine** verdict is **rejected**. Name every unfinished story key in the Acceptance verdict section as the evidence. Do not soften this to accepted-with-open-items: unfinished delivery is not an open finding about a finished epic — the epic itself is incomplete.
- Record the unfinished keys in Epic summary (interactive) or Assumptions (headless) as the Inputs section already requires.
- Headless runs have no human at the console: the document's verdict is **rejected** when `pending_stories` was non-empty. Interactive runs may still let a human override (rule 1 below) after seeing the list.

If the completeness check did not run (no readable `sprint-status.yaml`), do **not** render a rejected or accepted verdict from the absence of data — say the check was unavailable and weigh only the criteria and findings you have.

Three hard rules:

1. A human decision always overrides the machine verdict.
2. An epic that fails its criteria with **no** human decision is recorded as **not accepted** — never as silently accepted.
3. A non-empty `pending_stories` list makes the machine verdict **rejected**, including in headless mode.

The verdict and its evidence carry into the Phase 5 document.
