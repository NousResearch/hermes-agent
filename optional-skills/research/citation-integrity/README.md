# citation-integrity

A standalone Hermes optional skill for citation-ledger quality control. It
covers four concerns that are easy to miss in ordinary citation checks:

- preserving source and evidence identity;
- detecting duplicate sources across URL variants and repeated retrievals;
- proving that verbatim quotes came from the archived evidence; and
- failing strict verification when the draft, ledger, or evidence chain is
  incomplete.

The skill is intentionally tool-agnostic. It can be used with a custom ledger,
local scripts, or a hosted research workflow. It does not require proprietary
commands, private paths, or a companion package.

Install or copy `SKILL.md` into the optional skills directory used by Hermes.
The included test is a dependency-free publication smoke test:

```bash
python3 tests/test_skill.py
```
