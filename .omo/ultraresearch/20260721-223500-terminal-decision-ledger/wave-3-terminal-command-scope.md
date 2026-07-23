# Wave 3 — Generic terminal approval scope

## Finding

The generic terminal/tool approval path is a separate in-memory queue. Its
`post_approval_response` hook observes approve/deny/timeout, but a response is
not the same as a durable terminal execution result and the queue lacks the
proposal ID and crash-recovery contract used by write approvals
(`tools/approval.py:2115,3058-3170`).

## Closure

Do not mix generic command approvals into this write-proposal receipt table.
It requires a separate future schema and correlation design for approval
choice, actual tool result, timeout, and restart recovery. This is explicitly
out of scope for the current minimal, durable proposal ledger.

## EXPAND

none — code, hooks, and queue lifecycle establish a distinct capability
boundary rather than an unchecked implementation lead.
