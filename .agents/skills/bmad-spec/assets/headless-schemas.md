# Headless JSON Response

The default invocation is headless: input goes in, JSON comes out. The contract is intentionally tiny — return the outcome and the files touched. Anything else a caller needs is inside those files (SPEC.md, companions, `.memlog.md`).

## Success

```json
{
  "status": "complete",
  "files": [
    "_bmad-output/specs/spec-quarter-drop/SPEC.md",
    "_bmad-output/specs/spec-quarter-drop/glossary.md",
    "_bmad-output/specs/spec-quarter-drop/.memlog.md"
  ]
}
```

`files` lists every file written or modified in this run, in any order. The spec folder, kernel filename, memlog location, capabilities, companions, and verdict are all readable from those files; no need to re-encode them in the response.

## Blocked

```json
{
  "status": "blocked",
  "error_code": "insufficient_intent",
  "reason": "Input was a one-line idea with no surrounding context; too thin to distill. Suggest bmad-prd to draw the vision out first."
}
```

Defined `error_code` values:

- `insufficient_intent` — input too thin to distill into a kernel.
- `missing_slug` — input is sparse or multi-source and no slug was provided by the caller or derivable from a source path.
