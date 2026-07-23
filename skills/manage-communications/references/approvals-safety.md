# Drafts and approvals

```text
hermes communication drafts create PERSON SOURCE_ENDPOINT --text TEXT
hermes communication drafts list [--status STATUS]
hermes communication drafts show DRAFT_ID
hermes communication drafts cancel DRAFT_ID
hermes communication approvals approve DRAFT_ID [--ttl-minutes N]
hermes communication approvals reject APPROVAL_ID
```

Before creation, show the canonical person, exact source and target accounts,
exact endpoints, recipient preview, and payload. Approval binds all of those,
the route version, payload hash, recipient hash, actor, and expiry. Changing any
field invalidates it. Concurrent consumption is atomic.

There is intentionally no production-send command. Never use a legacy sender,
browser click, adapter method, or direct SQLite write to bypass this boundary.
Only deterministic tests may execute the in-memory fake sink, which requires an
observed postcondition before `sent`; expired claims become `uncertain`.
