# ADR Format

ADRs live in `docs/adr/` and use sequential numbering: `0001-slug.md`, `0002-slug.md`, etc.

Create the `docs/adr/` directory lazily, only when the first ADR is needed.

## Template

```md
# {Short title of the decision}

{1-3 sentences: what's the context, what did we decide, and why.}
```

That's it. An ADR can be a single paragraph. The value is in recording that a decision was made and why, not in filling out sections.

## Optional Sections

Only include these when they add genuine value. Most ADRs will not need them.

- **Status** frontmatter: `proposed`, `accepted`, `deprecated`, or `superseded by ADR-NNNN`.
- **Considered Options**: only when the rejected alternatives are worth remembering.
- **Consequences**: only when non-obvious downstream effects need to be called out.

## Numbering

Scan `docs/adr/` for the highest existing number and increment by one.

## When to Offer an ADR

All three of these must be true:

1. **Hard to reverse** — the cost of changing your mind later is meaningful.
2. **Surprising without context** — a future reader will wonder why it was done this way.
3. **The result of a real tradeoff** — there were genuine alternatives and one was chosen for specific reasons.

If a decision is easy to reverse, skip it. If it is not surprising, nobody will wonder why. If there was no real alternative, there is nothing to record beyond "we did the obvious thing."

## What Qualifies

- Architectural shape: monorepo, event sourcing, read/write model split.
- Integration patterns between contexts: domain events vs synchronous HTTP.
- Technology choices that carry lock-in: database, message bus, auth provider, deployment target.
- Boundary and scope decisions: which context owns which data.
- Deliberate deviations from the obvious path: manual SQL instead of ORM, REST instead of GraphQL.
- Constraints not visible in code: compliance requirements, latency constraints, partner API requirements.
- Rejected alternatives when the rejection is non-obvious and likely to recur.
