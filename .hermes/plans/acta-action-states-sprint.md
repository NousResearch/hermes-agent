# Acta sprint: signed-row action states

## Product selection

Persona/scenario: P checks Acta on iPhone, sees the top signed briefing and a follow-up row, and needs to triage without opening every artifact: save the important packet, dismiss low-value noise, or mark something for later.

Feature bets considered:
1. Obvious but necessary: finish signed-row read-state parity already in progress.
2. High-leverage personal workflow: add local signed-row triage actions (Save / Dismiss / Read later) that make Acta feel like an operator inbox, not a passive feed.
3. Weird/ambitious: infer persistent P priorities from repeated action states and reorder future read order.

Rank:
- BUILD NOW: High-leverage personal workflow, MVP action buttons persisted locally only for signed/readable Today/archive rows.
- SPIKE-PROTOTYPE: priority inference from repeated actions.
- KILL FOR NOW: cosmetic-only dashboard polish.

## Scope

- Complete the existing renderer/test slice for signed-row action buttons.
- Preserve signed URL gates, ASK links, overlay click-through, read/unread state, CSP hashes, compact mobile layout.
- Do not add fake AI scoring, fake data, server-side persistence, cron changes, or production auth changes.

## Acceptance criteria

- Signed Today lead/feed rows and archive-day rows expose Save, Dismiss, and Read later controls.
- Unsafe/no-page rows do not expose action buttons or read/action state.
- Buttons persist to localStorage/cookie, toggle visible labels/pressed state/classes, and never open the row overlay.
- Browser/fixture UAT exercises lead -> action button -> row overlay and 390px overflow/console checks.
