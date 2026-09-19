"""Print a compact comparison from an edit-tool audit JSON scorecard."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    data = json.loads(args.result.read_text(encoding="utf-8"))
    print(f"Edit-tool shape audit: {data['label']} ({data['task_count']} tasks)")
    for arm, records in data["arms"].items():
        outcomes = Counter(record["outcome"] for record in records)
        passed = sum(record["passed"] for record in records)
        print(
            f"{arm}: score={passed}/{len(records)}; "
            + ", ".join(f"{key}={outcomes[key]}" for key in sorted(outcomes))
        )
        for record in records:
            verdict = "PASS" if record["passed"] else "DRIFT"
            print(
                f"  {record['task_id']}: {record['outcome']} "
                f"(expected {record['expected']}; {verdict}) — {record['reason']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
