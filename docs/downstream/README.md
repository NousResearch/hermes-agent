# Downstream Intent Ledger

This directory records only long-lived differences that an upstream sync could
accidentally erase. It is not architecture documentation, a changelog, or a
conflict log.

The [scope ledger](scope.md) lists the approved divergence areas, their owners,
restored native areas and open decisions. Read it before changing this fork.

## Brevity is a contract

- Keep each active entry under **150 words**, excluding metadata.
- State one invariant, the upstream risk, the merge rule, and minimal checks.
- Link to authoritative docs; never copy their detail, diffs, or history here.
- Keep the active-entry index to one line per entry.

## Sync contract

Before integrating upstream changes, read every active entry and check
both path and conceptual overlap. Preserve intent, absorb compatible upstream
improvements, validate the listed checks, and report one concise outcome per
entry. Flag ambiguous or obsolete intent for review.

The sync automation is a **read-only consumer**: it may not author, expand, or
retire entries. It may only propose a separate developer change.

## Ownership

A developer or ordinary downstream change owns ledger updates. Add or change an
entry only with the long-lived divergence it governs; retire it when the intent
is removed or upstream satisfies it. Use this shape:

```markdown
# <title>

- Status: active | retired
- Scope: `<paths or surface>`
- Introduced: `<commit or pull request>`

## Downstream intent
<Invariant and upstream risk.>

## Reconciliation
<How to absorb upstream changes without losing the invariant.>

## Validation
<Concrete checks for the sync pull request.>
```

## Active entries

- [Railway container storage](divergences/railway-storage.md) — explicit persistent mounts and service deployment settings.

- [Scoped agent guidance](divergences/scoped-agent-guidance.md) — route agent guidance instead of growing one root file.

- [Employee runtime defaults](divergences/employee-runtime-defaults.md) — preserve native defaults and explicit local transcription.

- [Employee runtime](divergences/employee-runtime.md) — preserve employee behavior on native runtime owners.
- [Codex review](divergences/codex-review.md) — run read-only implementation review before landing.

- [Employee surface exclusions](divergences/employee-surface.md) — preserve the model tool filter, disabled skills and client visibility only.

- [Unified knowledge review](divergences/knowledge-review.md) — preserve source review scope and personal/shared memory tools.

- [File keeping](divergences/file-keeping.md) — profile-local documents/repos and a source-derived filing guide.

- [Connection guides](divergences/connection-guides.md) — native setup guidance and per-service operating manuals without skills.

- [Hosted settings](divergences/hosted-settings.md) — focused configuration UI over native stores and auth, with private Hindsight updates.
