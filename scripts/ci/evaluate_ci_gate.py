"""Evaluate CI job results and fail closed for non-success outcomes."""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Mapping
from typing import Any


_ACCEPTED_RESULTS = frozenset({"success", "skipped"})


def evaluate(needs: Mapping[str, Any]) -> tuple[dict[str, str], list[str]]:
    """Return compact results and the jobs that did not complete acceptably."""
    compact: dict[str, str] = {}
    blocked: list[str] = []
    for name, info in needs.items():
        result = info.get("result", "unknown") if isinstance(info, Mapping) else "unknown"
        result = str(result)
        compact[name] = result
        if result not in _ACCEPTED_RESULTS:
            blocked.append(name)
    return compact, blocked


def main() -> int:
    try:
        needs = json.loads(os.environ["NEEDS"])
    except (KeyError, json.JSONDecodeError) as exc:
        print(f"::error::Unable to read CI job results: {exc}")
        return 1
    if not isinstance(needs, Mapping):
        print("::error::CI job results must be a JSON object")
        return 1

    compact, blocked = evaluate(needs)
    needs_json = json.dumps(compact)
    print(f"needs-json={needs_json}")

    output_path = os.environ.get("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output:
            output.write(f"needs-json={needs_json}\n")

    for name, result in sorted(compact.items()):
        status = "[ok]" if result in _ACCEPTED_RESULTS else "[blocked]"
        print(f"{status} {name}: {result}")
    if blocked:
        print(f"::error::{len(blocked)} job(s) did not complete: {', '.join(blocked)}")
        return 1

    print("All checks passed (or were skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
