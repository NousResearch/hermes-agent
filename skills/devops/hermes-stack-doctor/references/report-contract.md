# Report Contract

## Allowed classifications

- `GREEN`
- `YELLOW`
- `RED`

## Required headings

- Hermes Stack Doctor
- Overall Status
- Architecture Discovered
- Critical Findings
- Health Matrix
- Evidence Conflicts
- Required Actions
- Watchlist
- Not Verified
- Evidence Checked

Do not invent an intermediate classification. Put nuance in confidence, warnings, blockers, and Not Verified.

## Severity floor

The Overall Status must equal the worst confirmed status in the Health Matrix.

- Any confirmed `RED` row forces Overall Status `RED`.
- Confirmed gateway or message-delivery failure is `RED`, even when the gateway process is running.
- `YELLOW` is allowed only when no subsystem is confirmed `RED` and required delivery and integrity remain available.
- Missing evidence lowers confidence but does not downgrade a confirmed failure.
