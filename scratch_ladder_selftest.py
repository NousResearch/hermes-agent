"""Scratch self-test for the pre-review review-gate ladder.

Deliberately contains a lint-only violation: ``open()`` without an explicit
``encoding=`` argument (PLW1514, one of the lints this project's ruff config
actually selects).  It is syntactically valid and importable, so ONLY the lint
rung can object — not a syntax error, not an import error.
"""


def _read_cfg() -> str:
    with open("some_config.txt") as fh:  # PLW1514: missing encoding= — lint only
        return fh.read()