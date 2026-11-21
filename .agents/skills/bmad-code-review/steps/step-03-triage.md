---
---

# Step 3: Triage

## RULES

- YOU MUST ALWAYS SPEAK OUTPUT in your Agent communication style with the config `{communication_language}`

## INSTRUCTIONS

1. **Normalize** findings from all layers into a unified list where each finding has:
   - `id` -- sequential integer
   - `source` -- the `id` of the layer that produced the finding (e.g., `blind-hunter`)
   - `title` -- one-line summary
   - `detail` -- full description
   - `location` -- file and line reference (if available)

2. Once every layer has reported -- and not before -- render a verdict on each finding, ahead of any deduplication or grouping. Disregard any severity a reviewing subagent assigned -- they lack the context to grade.

   For each finding:
   - A gap finding from the verification-gap layer arrives pre-verified -- that layer's evidence rules made it read the tests and run the searches it cites, and triage trusts the claim as filed. Skip verification, render the verdict from the filed evidence, and weigh its filed disposition when routing. Its `Other findings` are verified like everything else.
   - **Verify the finding's claim.** At the cited file and line, does the bad outcome the reviewer describes actually occur? Read beyond the changed lines -- follow callers, guards upstream, etc -- until you can answer yes or no. A different finding about nearby code does not settle this one. Judge whether the problem is real, not whether the proposed fix is plausible. Code that loudly fails on a situation you never showed the program can reach is correct behavior, not a defect.
   - **Render exactly one verdict** from what verification established -- the verdict is the whole triage decision; there is no separate keep-or-dismiss.
     - `high` (intolerable), `medium` (tolerable), `low` (cosmetic or negligible) -- the bad outcome is real. Assign severity by how much it hurts end users or developers. For developer-only problems (inconsistent design, eroded invariants, duplicated sources of truth), name where it will cause trouble -- which caller will diverge, which rule will break. A vague "this is messy" with no named harm is not a severity grade; use `false` or `maybe-false` instead. When the harm is real but you cannot tell how bad, pick the higher grade.
     - `false` -- you checked, and the bad outcome does not happen at the cited location. Write what disproves this specific claim. A true fact about nearby code that does not disprove the claim does not count.
     - `maybe-false` -- you could not tell whether the bad outcome happens. Write what you would need to check to find out. Use this only when the diff and surrounding code leave the question open; when they are enough to decide, pick `high`, `medium`, `low`, or `false`.

   - Every finding keeps its verdict and evidence (a sentence or two) for the summary; never drop, merge, or silently skip one.

   Reject `false` findings on their refutation.

   Reject `low` findings when it is unlikely that users or developers would meet the defect in everyday use (judged plainly -- no proof needed) and the fix is more than a direct correction or deletion -- adding guards, branches, parameters, or other complexity.

   Reject any finding whose fix is to edit the spec under review.

   All remaining findings continue to grouping.

3. **Group the survivors by shared root cause** -- two findings belong in one entry only when the same defect produced both. Same location alone is not a shared root cause, and neither is a shared fix. An entry carries every member's verified bad outcome in `detail` and the highest verdict among them (`high` > `medium` > `low` > `maybe-false`); set `source` to the contributing layers joined with `+` (e.g., `blind-hunter+edge-case-hunter`).

4. **Route** each entry into exactly one triage bucket. A group that includes verified `high`, `medium`, or `low` members routes by its highest such verdict -- not to defer just because a member is `maybe-false`.
   - **decision_needed** -- There is an ambiguous choice that requires human input. The code cannot be correctly patched without knowing the user's intent. Only possible if `review_mode` = `full`.
   - **patch** -- Code issue that is fixable without human input. The correct fix is unambiguous, adds no public surface, and guards no state you did not demonstrate; otherwise `decision_needed`.
   - **defer** -- Pre-existing issue not caused by the current change, real but not actionable now; or an entry whose members are all `maybe-false` and the claim, if true, would be `medium` or `high` -- record that severity marked unverified, plus what would settle it (if it would only be `low`, reject it with the same note); or any entry whose fix edits agent-context files (CLAUDE.md, AGENTS.md, rules, other specs).

   If `review_mode` = `no-spec` and an entry would otherwise be `decision_needed`, reclassify it as `patch` (if the fix is unambiguous) or `defer` (if not).

5. If `failed_layers` is non-empty, report which layers failed before announcing results. If zero entries remain after rejections AND `failed_layers` is non-empty, warn the user that the review may be incomplete rather than announcing a clean review.

6. If zero entries remain after triage (all rejected or none raised): state "✅ Clean review — all layers passed." (Step 3 already warned if any review layers failed via `failed_layers`.)

## NEXT

Read fully and follow `./step-04-present.md`
