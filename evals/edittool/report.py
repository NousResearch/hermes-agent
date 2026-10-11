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
    metrics = data.get("metrics", {})
    status_metric = metrics.get("status", {"name": "passed", "version": 1})
    artifact_metric = metrics.get("artifact", {"name": "artifact_correct", "version": 1})
    provenance = data.get("provenance")
    if provenance:
        print(
            "Provenance: "
            f"measurement={provenance['measurement_commit']}; "
            f"fixtures={provenance['fixture_commit']}; "
            f"matcher={provenance['matcher_commit']}"
        )
    for arm, records in data["arms"].items():
        outcomes = Counter(record["outcome"] for record in records)
        passed = sum(record[status_metric["name"]] for record in records)
        artifact_available = all(artifact_metric["name"] in record for record in records)
        artifact_score = (
            str(sum(record[artifact_metric["name"]] for record in records))
            if artifact_available
            else "N/A (pre-v2)"
        )
        print(
            f"{arm}: status_score[{status_metric['name']}/v{status_metric['version']}"
            f"]={passed}/{len(records)}; artifact_score["
            f"{artifact_metric['name']}/v{artifact_metric['version']}]={artifact_score}"
            f"{f'/{len(records)}' if artifact_available else ''}; "
            + ", ".join(f"{key}={outcomes[key]}" for key in sorted(outcomes))
        )
        for record in records:
            verdict = "PASS" if record["passed"] else "DRIFT"
            artifact = (
                "NOT_MEASURED"
                if artifact_metric["name"] not in record
                else "CORRECT" if record[artifact_metric["name"]] else "WRONG_ARTIFACT"
            )
            partial = "; partial write" if record.get("partial_write") else ""
            print(
                f"  {record['task_id']}: {record['outcome']} "
                f"(expected {record['expected']}; {verdict}; {artifact}{partial}) — "
                f"{record['reason']}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
