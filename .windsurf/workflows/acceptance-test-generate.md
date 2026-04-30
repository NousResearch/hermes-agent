---
description: Generate or refresh 1:1 acceptance test files from acceptance-test IR while preserving owned bodies.
auto_execution_mode: 3
---

# acceptance-test-generate

Run this workflow when acceptance-test specs or generated acceptance tests change.

```bash
node tools/acceptance/generate.mjs --ir tools/acceptance/.work/ir.json --map openspec/.acceptance.json --root .
```
