# Skill installation security blocks

Use `hermes skills inspect <identifier>` to establish source identity and `hermes skills install --help` for flags. Marketplace listings include third-party publishers; preserve the complete source-qualified identifier.

`--force` can override non-dangerous policy blocks, not a DANGEROUS verdict. Do not copy around quarantine or disable global scanning to install a blocked bundle. Read the actual scan provenance and distinguish scanner rules from the unrelated security-guidance plugin.

For false-positive scanner fixes, reproduce pinned upstream content in isolated scratch, add adversarial controls, and bump SCANNER_VERSION so cached verdicts cannot survive semantic changes. Keep official-publisher identity, code fences, eval filenames, localhost endpoints and purported denial metadata from becoming blanket trust exceptions.

Treat environment subscript accesses as possible writes, not just reads. Assignment, augmented assignment, del, tuple/loop/context-manager targets, backslash or bracket continuation, comments and escaped quotes defeat line-only heuristics. Fail closed on ambiguous context. Prefer a narrow proven complete read shape over broad environment-read demotion; preserve secret-bearing reads at their original severity.

When refining placeholder credentials, evaluate every match from the original scanner pattern, including unterminated literals; a second parser requiring closing quotes can miss a real secret next to a placeholder and incorrectly demote the whole line.
