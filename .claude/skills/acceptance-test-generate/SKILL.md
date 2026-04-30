---
name: acceptance-test-generate
description: Generate or refresh 1:1 acceptance test files from acceptance-test IR while preserving owned bodies.
---

# acceptance-test-generate

Invoke when working on the acceptance-test workflow or when specs contain scenarios with `Path Code:` lines.

Contract:
- Specs use `### Requirement`, `#### Scenario`, `- Path Code: ...`, and ordered GIVEN/WHEN/THEN/AND steps.
- Path Code format: `UT|IT|UC-DOMAIN-###[-VARIANT]`.
- The source of truth is the PathCode metadata emitted into generated tests, not filenames or method names.

Primary command:
```bash
node tools/acceptance/generate.mjs --ir tools/acceptance/.work/ir.json --map openspec/.acceptance.json --root .
```

Use this before claiming acceptance coverage is current.
