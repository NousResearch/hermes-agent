# Generate Review Trail

Generate a review trail from the diff and codebase context. A generated trail is lower quality than an author-produced one, but far better than none.

## Follow Global Step Rules in SKILL.md

## INSTRUCTIONS

1. Get the full diff against the appropriate baseline (same rules as Surface Area Stats in step-01).
2. Read changed files in full — not just diff hunks. Surrounding code reveals intent that hunks alone miss. If total file content exceeds ~50k tokens, read only the files with the largest diff hunks in full and use hunks for the rest.
3. If a spec exists, use its Intent section to anchor concern identification.
4. Identify 2–5 concerns: cohesive design intents that each explain *why* behind a cluster of changes. Prefer functional groupings and architectural boundaries over file-level splits. A single-concern change is fine — don't invent groupings.
5. For each concern, select 1–4 `path:line` stops — locations where the concern is most visible. Prefer entry points, decision points, and boundary crossings over mechanical changes.
6. Lead with the entry point — the highest-leverage stop a reviewer should see first. Inside each concern, order stops so each builds on the previous. End with peripherals (tests, config, types).
7. Format each stop using `path:line` per the global step rules:

```
**{Concern name}**

- {one-line framing, ≤15 words}
  `src/path/to/file.ts:42`
```

When there is only one concern, omit the bold label — just list the stops directly.

## PRESENT

Output after the orientation:

```
I built a review trail for this {change_type} (no author-produced trail was found):

{generated trail}
```

The generated trail serves as the Suggested Review Order for subsequent steps. Set `review_mode` to `full-trail` — a trail now exists, so all downstream steps should treat it as one.

If git is unavailable or the diff cannot be retrieved, return to step-01 with: "Could not generate trail — git unavailable."
