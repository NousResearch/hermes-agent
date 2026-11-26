"""Child-process entry for one machine-acceptance evaluation (``kanban_acceptance.evaluate_in_child``).

argv[1] is the JSON job; the JSON result list is the last stdout line.
"""
from __future__ import annotations

import json
import sys


def main(argv: list[str]) -> int:
    from hermes_cli.kanban_acceptance import evaluate_job
    sys.stdout.write(json.dumps(evaluate_job(json.loads(argv[1])), ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
