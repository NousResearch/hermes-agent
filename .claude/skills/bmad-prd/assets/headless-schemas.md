# Headless Mode JSON Schemas

Every headless run ends with one of these payloads. Omit keys for artifacts not produced.

## Common fields

- `status` — `"complete"`, `"blocked"`, or `"partial"`
- `intent` — `"create"`, `"update"`, or `"validate"` (matches the detected intent)
- `reason` — required when `status` is `"blocked"`; one-sentence explanation
- `assumptions` — array of inferred values that were not directly confirmed by inputs
- `open_questions` — array of items that need a human decision before the artifact can be considered final

## Create

```json
{
  "status": "complete",
  "intent": "create",
  "prd": "{doc_workspace}/prd.md",
  "addendum": "{doc_workspace}/addendum.md",
  "memlog": "{doc_workspace}/.memlog.md",
  "open_questions": [],
  "assumptions": [],
  "external_handoffs": [
    {"directive": "Confluence upload", "tool": "corp:confluence_upload", "url": "https://confluence.corp/PROD/123", "status": "ok"}
  ]
}
```

## Update

```json
{
  "status": "complete",
  "intent": "update",
  "prd": "{doc_workspace}/prd.md",
  "memlog": "{doc_workspace}/.memlog.md",
  "changes_summary": "1-3 sentences describing what changed and why",
  "conflicts_with_prior_decisions": [],
  "open_questions": [],
  "external_handoffs": [
    {"directive": "Confluence upload", "tool": "corp:confluence_upload", "url": "https://confluence.corp/PROD/123", "status": "ok"}
  ]
}
```

## Validate

```json
{
  "status": "complete",
  "intent": "validate",
  "validation_report": "{doc_workspace}/validation-report.md",
  "findings_summary": {
    "critical": 0,
    "high": 0,
    "medium": 0,
    "low": 0
  },
  "offer_to_update": true
}
```

`validation_report` is always written for Validate intent — the path here is required, not optional.

## Blocked

```json
{
  "status": "blocked",
  "intent": "update",
  "reason": "Change signal ambiguous — could be a scope expansion or a clarification; no inferred direction"
}
```

Always include the intent (best-guess if not certain) and a one-sentence `reason`.
