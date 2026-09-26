---
name: foundry-security-reviewer
description: "Audit Foundry projects with tests and security tools."
version: 1.0.0
author: Ahmet Osrak (Osraka), Hermes Agent
license: MIT
platforms: [linux, macos]
metadata:
  hermes:
    tags: [blockchain, security, solidity, foundry, smart-contracts]
    category: software-development
---

# Foundry Security Reviewer Skill

Run a repeatable security-review baseline for a Solidity project that uses
Foundry. The bundled script records build, test, coverage, gas, and optional
static-analysis evidence; it does not claim that automated output alone is a
complete smart-contract audit.

## When to Use

- The user asks to review a Solidity project that uses Foundry.
- The user requests smart-contract security triage or audit evidence.
- The user wants a security-focused PR review checklist.

## Prerequisites

- The `forge` CLI on `PATH`.
- A project root containing `foundry.toml`.
- Optional `slither` and `aderyn` installations for static analysis.
- The `terminal` and `read_file` tools.
- Linux or macOS, because the bundled helper uses Bash and POSIX temporary-file
  primitives.

## How to Run

Invoke the bundled reviewer through the `terminal` tool from the target project
root, or pass the project root as its first argument:

```text
bash ${HERMES_SKILL_DIR}/scripts/run_review.sh .
```

Read the generated `review_output.md` with `read_file` before writing the
review comment.

## Quick Reference

| Check | Command | Evidence |
| --- | --- | --- |
| Build | `forge build` | Compiler status and diagnostics |
| Tests | `forge test -vv` | Failing cases and traces |
| Coverage | `forge coverage --report summary` | Modules below 80% |
| Gas | `forge snapshot` | Baseline for later comparisons |
| Optional | `slither .`, `aderyn .` | Static-analysis leads |

## Procedure

1. Confirm the project root contains `foundry.toml`.
2. Run `scripts/run_review.sh <project-root>` through `terminal`.
3. Read `review_output.md`; compiler errors mean the project is not ready for
   a complete review.
4. List failing cases from `forge test -vv` and identify the contract or
   invariant each case exercises.
5. Review the coverage summary and investigate every module below 80%.
6. Keep the `forge snapshot` result as a gas baseline. Compare later changes
   only under a consistent toolchain, compiler, RPC, and test configuration.
7. When installed, inspect Slither HIGH/MEDIUM and Aderyn critical findings.
   Confirm exploitability in source before escalating a finding.
8. Compare the code with
   [common vulnerabilities](references/common-vulnerabilities.md).
9. Paste the checklist into the PR review and replace placeholders with the
   affected file, function, severity, impact, and test evidence.

## Pitfalls

- `forge coverage` can be slow in large projects; narrow it with
  `--match-contract <ContractName>` when triaging a focused change.
- If coverage fails or produces no parseable Solidity coverage rows, the
  script reports the threshold result as unavailable rather than claiming that
  no module is below 80%.
- Slither can report false positives. Check suppressed lines and reachable
  paths before dismissing or escalating a detector.
- Gas snapshots vary between environments. Compare them only under consistent
  toolchain, compiler, RPC, and test settings.

## Verification

Verify that `review_output.md` exists, every required command is marked with
its actual exit status, and coverage is marked unavailable when its command or
summary cannot be parsed. Automated findings still require source-level
validation and a regression test before they are treated as vulnerabilities.

Use this report shape:

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
- [ ] Add a Forge regression or fuzz test for every confirmed vulnerability.
```
