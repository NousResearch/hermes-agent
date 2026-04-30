---
name: acceptance-test-extract
description: Parse Gherkin-style OpenSpec markdown into acceptance-test JSON IR and report Path Code issues.
---

# acceptance-test-extract

Invoke when working on the acceptance-test workflow or when specs contain scenarios with `Path Code:` lines.

Contract:
- Specs use `### Requirement`, `#### Scenario`, `- Path Code: ...`, and ordered GIVEN/WHEN/THEN/AND steps.
- Path Code format: `UT|IT|UC-DOMAIN-###[-VARIANT]`.
- The source of truth is the PathCode metadata emitted into generated tests, not filenames or method names.

Primary command:
```bash
node tools/acceptance/extract.mjs openspec/specs --out tools/acceptance/.work/ir.json --strict
```

Use this before claiming acceptance coverage is current.
