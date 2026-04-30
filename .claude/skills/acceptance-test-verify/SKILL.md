---
name: acceptance-test-verify
description: Run the acceptance-test gate: extract, generate, scan metadata coverage, and execute scoped tests.
---

# acceptance-test-verify

Invoke when working on the acceptance-test workflow or when specs contain scenarios with `Path Code:` lines.

Contract:
- Specs use `### Requirement`, `#### Scenario`, `- Path Code: ...`, and ordered GIVEN/WHEN/THEN/AND steps.
- Path Code format: `UT|IT|UC-DOMAIN-###[-VARIANT]`.
- The source of truth is the PathCode metadata emitted into generated tests, not filenames or method names.

Primary command:
```bash
node tools/acceptance/verify.mjs --all
```

Use this before claiming acceptance coverage is current.
