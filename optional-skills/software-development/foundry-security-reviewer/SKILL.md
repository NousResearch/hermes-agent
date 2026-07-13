---
name: foundry-security-reviewer
description: Run Foundry test/coverage/snapshot and surface security issues in Solidity projects
version: 1.0.0
metadata:
  hermes:
    tags: [blockchain, security, solidity, foundry, smart-contracts]
    category: software-development
---

# Foundry Security Reviewer

Run a repeatable security-review baseline for a Foundry Solidity project. Use the bundled script to collect build, test, coverage, gas, and optional static-analysis evidence into one Markdown report.

## When to Use

- Review a Solidity project that uses Foundry.
- Perform a Foundry security review or smart-contract audit triage.
- Prepare a security-focused PR review comment for a smart-contract change.

## Prerequisites

- Require the `forge` CLI on `PATH`.
- Require a Foundry project root containing `foundry.toml`.
- Optionally install `slither` for detector findings and `aderyn` for Solidity static analysis.
- Run from the target project root, or pass the root as the first argument.

## Procedure

1. Confirm the project root contains `foundry.toml`.
2. Run the bundled reviewer:

   ```bash
   bash ${HERMES_SKILL_DIR}/scripts/run_review.sh .
   ```

3. Inspect `review_output.md` in the target project root.
4. Report compiler errors from `forge build`; do not treat an uncompiled project as audited.
5. List failing cases from `forge test -vv` and identify the contract or invariant they exercise.
6. Review the `forge coverage --report summary` table and investigate every module below 80% coverage.
7. Keep the `forge snapshot` result as the gas baseline; compare later changes against it.
8. When installed, review Slither HIGH/MEDIUM findings and Aderyn critical findings. Confirm exploitability in code before escalating a finding.
9. Compare code against [common vulnerabilities](references/common-vulnerabilities.md) and add concrete remediation recommendations.
10. Paste the report's checklist into the PR review, replacing placeholders with file, function, severity, impact, and test evidence.

## Output Format

Return `review_output.md` as a PR-review-ready Markdown checklist with these sections:

```markdown
# Foundry Security Review

## Test Results
- [ ] Build succeeds
- [ ] Tests pass

## Coverage
| Module | Coverage | Status |
| --- | ---: | --- |
| `src/Example.sol` | 72% | Needs tests |

## Gas Snapshot
- [ ] Baseline captured

## Security Findings
- [ ] HIGH — `src/Example.sol:42` — impact and exploit path

## Recommendations
- [ ] Add a regression test for the confirmed finding.
```

## Pitfalls

- `forge coverage` can be slow in large projects; narrow it with `--match-contract <ContractName>` when triaging a focused change.
- Slither can report false positives; note lines suppressed with `# slither-disable` and verify each remaining detector result manually.
- Gas snapshots can vary between environments; compare them only under a consistent toolchain, compiler, RPC, and test configuration.
