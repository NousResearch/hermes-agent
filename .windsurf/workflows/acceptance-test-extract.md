---
description: Parse Gherkin-style OpenSpec markdown into acceptance-test JSON IR and report Path Code issues.
auto_execution_mode: 3
---

# acceptance-test-extract

Run this workflow when acceptance-test specs or generated acceptance tests change.

```bash
node tools/acceptance/extract.mjs openspec/specs --out tools/acceptance/.work/ir.json --strict
```
