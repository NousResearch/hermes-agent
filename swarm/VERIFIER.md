# Swarm Verifier

Framework for scoring and validating swarm task results.

## Overview

The verifier runs a configurable pipeline of checks against a `(SwarmTask, SwarmResult)` pair, producing a weighted score and a pass/fail verdict.

## Quick Start

```python
from swarm.types import SwarmTask, SwarmResult
from swarm.verifier import Verifier, format_report

task = SwarmTask(name="build widget", prompt="...")
result = SwarmResult(task_id=task.id, success=True, output="def hello(): ...")

v = Verifier()  # uses DEFAULT_CHECKS, threshold=0.7
vr = v.verify(task, result)
print(format_report(vr))
```

## Built-in Checks

| Check | What it does |
|---|---|
| `NonEmptyOutputCheck` | Fails if output is None, empty string, or empty collection |
| `NoErrorCheck` | Scans output for error patterns (Traceback, FATAL, panic, etc.) |
| `FilesExistCheck` | Verifies referenced artifact paths exist on disk |
| `SyntaxCheck` | If output looks like Python, runs `compile()` to check syntax |
| `DiffSizeCheck(max_lines)` | Fails if output exceeds a line-count limit (default 2000) |

## Custom Checks

Subclass `VerificationCheck` and implement `check()`:

```python
from swarm.verifier import VerificationCheck, CheckResult

class ContainsKeywordCheck(VerificationCheck):
    def __init__(self, keyword: str):
        self.keyword = keyword

    def check(self, task, result):
        found = self.keyword in str(result.output)
        return CheckResult(
            name="contains_keyword",
            passed=found,
            detail=f"{'Found' if found else 'Missing'} keyword {self.keyword!r}",
        )

v = Verifier(checks=[ContainsKeywordCheck("TODO")], threshold=1.0)
```

## Scoring

Each `CheckResult` carries a `weight` (default 1.0). The final score is the sum of weights for passing checks divided by total weight. The result passes when `score >= threshold`.

## API

- `Verifier(checks=None, threshold=0.7, name="Verifier")` -- create a verifier
- `Verifier.verify(task, result) -> VerificationResult` -- run all checks
- `format_report(vr) -> str` -- human-readable report
- `DEFAULT_CHECKS` -- the default check list
