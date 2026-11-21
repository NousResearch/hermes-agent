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
  "design": "{doc_workspace}/DESIGN.md",
  "experience": "{doc_workspace}/EXPERIENCE.md",
  "memlog": "{doc_workspace}/.memlog.md",
  "working_artifacts": ["{doc_workspace}/.working/color-themes-1.html"],
  "promoted_artifacts": {
    "mockups": ["{doc_workspace}/mockups/direction-calm-sage.html"],
    "wireframes": ["{doc_workspace}/wireframes/ia-2026-05-19.excalidraw"]
  },
  "open_questions": [],
  "assumptions": [],
  "external_handoffs": [
    {"directive": "Confluence upload", "tool": "corp:confluence_upload", "url": "https://confluence.corp/DESIGN/123", "status": "ok"}
  ]
}
```

The `working_artifacts` and `promoted_artifacts` keys are optional and omitted entirely when empty. Headless Create runs default to not enabling creative tools — both keys are typically absent in headless output unless the caller enabled them.

## Update

```json
{
  "status": "complete",
  "intent": "update",
  "design": "{doc_workspace}/DESIGN.md",
  "experience": "{doc_workspace}/EXPERIENCE.md",
  "memlog": "{doc_workspace}/.memlog.md",
  "changes_summary": "1-3 sentences describing what changed and why",
  "conflicts_with_prior_decisions": [],
  "open_questions": [],
  "external_handoffs": [
    {"directive": "Confluence upload", "tool": "corp:confluence_upload", "url": "https://confluence.corp/DESIGN/123", "status": "ok"}
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
  "reason": "Change signal ambiguous — could be a brand refresh or an accessibility audit response; no inferred direction"
}
```

Always include the intent (best-guess if not certain) and a one-sentence `reason`.
