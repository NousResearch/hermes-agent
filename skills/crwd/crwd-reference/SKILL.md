---
name: crwd-reference
description: "Internal CRWD fact sheet (company, gig lifecycle, Dot payments, proof formats). Not user-facing — other crwd-* skills pull these details on demand. Load a section with skill_view(\"crwd-reference\", \"references/<file>.md\")."
version: 1.0.0
metadata:
  hermes:
    tags: [crwd, reference, internal, facts, lifecycle, payments, proof]
    related_skills: [crwd-gig-discovery, crwd-gig-execution, crwd-application-expert]
---

# CRWD Reference

Shared source of truth for the deeper CRWD facts. This skill is **not** meant to be
triggered by a member directly — the other `crwd-*` skills point here when they need the
detail, so the facts live in exactly one place.

## When to Use

Not directly. Another `crwd-*` skill loads a section of this one when it needs a specific
fact (company background, lifecycle, payment/Dot rules, proof formats). If you're answering a
member and need one of those facts, load the matching reference file below rather than
recalling it from memory.

## Reference files

Load the section you need:

- `skill_view("crwd-reference", "references/company-facts.md")` — what CRWD is, who it serves
- `skill_view("crwd-reference", "references/gig-lifecycle.md")` — the full browse→paid flow
- `skill_view("crwd-reference", "references/payments-dot.md")` — Dot payouts, reimbursement rules
- `skill_view("crwd-reference", "references/proof-requirements.md")` — what proof each gig type needs

Keep answers to members short and specific — pull a fact from here, then say it in a
sentence or two. Don't paste whole reference sections into the chat.
