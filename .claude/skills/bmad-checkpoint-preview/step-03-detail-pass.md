# Step 3: Detail Pass

Display: `Orientation → Walkthrough → [Detail Pass] → Testing`

## Follow Global Step Rules in SKILL.md

- The detail pass surfaces what the human should **think about**, not what the code got wrong. Machine hardening already handled correctness. This activates risk awareness.
- The LLM detects risk category by pattern. The human judges significance. Do not assign severity scores or numeric rankings — ordering by blast radius (below) is sequencing for readability, not a severity judgment.
- If no high-risk spots exist, say so explicitly. Do not invent findings.

## IDENTIFY RISK SPOTS

Scan the diff for changes touching risk-sensitive patterns. Look for 2–5 spots where a mistake would have the highest blast radius — not the most complex code, but the code where being wrong costs the most.

Risk categories to detect:

- `[auth]` — authentication, authorization, session, token, permission, access control
- `[public API]` — new/changed endpoints, exports, public methods, interface contracts
- `[schema]` — database migrations, schema changes, data model modifications, serialization
- `[billing]` — payment, pricing, subscription, metering, usage tracking
- `[infra]` — deployment, CI/CD, environment variables, config files, infrastructure
- `[security]` — input validation, sanitization, crypto, secrets, CORS, CSP
- `[config]` — feature flags, environment-dependent behavior, defaults
- `[other]` — anything risk-sensitive that doesn't fit the above (e.g., concurrency, data privacy, backwards compatibility). Use a descriptive tag.

Sequence spots so the highest blast radius comes first (how much breaks if this is wrong), not by diff order or file order. If more than 5 spots qualify, show the top 5 and note: "N additional spots omitted — ask if you want the full list."

If the change has no spots matching these patterns, state: "No high-risk spots found in this change — the diff speaks for itself." Do not force findings.

## SURFACE MACHINE HARDENING FINDINGS

Check whether the spec has a `## Spec Change Log` section with entries (populated by adversarial review loops).

- **If entries exist:** Read them. Surface findings that are instructive for the human reviewer — not bugs that were already fixed, but decisions the review loop flagged that the human should be aware of. Format: brief summary of what was flagged and what was decided.
- **If no entries or no spec:** Skip this section entirely. Do not mention it.

## PRESENT

Output as a single message:

```
Orientation → Walkthrough → [Detail Pass] → Testing
```

### Risk Spots

For each spot, one line:

```
- `path:line` — [tag] reason-phrase
```

Example:

```
- `src/auth/middleware.ts:42` — [auth] New token validation bypasses rate limiter
- `migrations/003_add_index.sql:7` — [schema] Index on high-write table, check lock behavior
- `api/routes/billing.ts:118` — [billing] Metering calculation changed, verify idempotency
```

### Machine Hardening (only if findings exist)

```
### Machine Hardening

- Finding summary — what was flagged, what was decided
- ...
```

### Closing menu

End the message with:

```
---

You've seen the design and the risk landscape. From here:
- **"dig into [area]"** — I'll deep-dive that specific area with correctness focus
- **"next"** — I'll suggest how to observe the behavior
```

## EARLY EXIT

If at any point the human signals they want to make a decision about this {change_type} (e.g., "let's ship it", "this needs a rethink", "I'm done reviewing", or anything suggesting they're ready to decide), confirm their intent:

- If they want to **approve and ship** → read fully and follow `./step-05-wrapup.md`
- If they want to **reject and rework** → read fully and follow `./step-05-wrapup.md`
- If you misread them → acknowledge and continue the current step.

## TARGETED RE-REVIEW

When the human says "dig into [area]" (e.g., "dig into the auth changes", "dig into the schema migration"):

1. If the specified area does not map to any code in the diff, say so: "I don't see [area] in this change — did you mean something else?" Return to the closing menu.
2. Identify all code locations in the diff relevant to the specified area.
3. Read each location in full context (not just the diff hunk — read surrounding code).
4. Shift to **correctness mode**: trace edge cases, check boundary conditions, verify error handling, look for off-by-one errors, race conditions, resource leaks.
5. Present findings as a compact list — each finding is `path:line` + what you found + why it matters.
6. If nothing concerning is found, say so: "Looked closely at [area] — nothing concerning. The implementation is solid."
7. After presenting, show only the closing menu (not the full risk spots list again).

The human can trigger multiple targeted re-reviews. Each time, present new findings and the closing menu only.

## NEXT

Read fully and follow `./step-04-testing.md`
