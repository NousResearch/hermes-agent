# Acta dashboard visible read-toggle slice

## Objective
Add a compact visible read/unread toggle to signed Acta Today dashboard rows so an operator can triage briefing packets without opening them or discovering the hidden swipe affordance.

## Persona / scenario
P opens Acta on a narrow phone viewport, reviews the Situation Room dashboard, and wants to mark a signed briefing as read/unread while preserving the ability to tap the row for detail/provenance and ASK Telegram independently.

## Scope
- Main Acta dashboard lead card and signed feed rows only.
- Reuse existing `acta:read:v1` localStorage/cookie state.
- Keep signed row open overlays and ASK links intact.
- Unsafe/no-page rows remain non-readable and get no toggle.

## Out of scope
- Changing cron schedules or delivery targets.
- New source data, fake KPIs, charts, or generic assistant widgets.
- Production SSO bypass or personal login use.

## Acceptance criteria
- Signed lead/feed rows render a visible compact `read-toggle` button with accessible label.
- Clicking the toggle updates READ/UNREAD state and persistence without navigating.
- Row/overlay click still opens signed detail and marks read.
- Mobile layout remains compact with no horizontal overflow in fixture QA.
- Targeted Acta tests pass.
