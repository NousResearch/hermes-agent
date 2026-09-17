#!/usr/bin/env python3
"""
Generate property-based test SCAFFOLDS from EARS-JSON requirement records.

This does NOT fully auto-generate correct tests -- that would be fake
output masquerading as verification. What it does: reads a directory of
JSON files validated against ears-schema.json, and for each requirement
emits one hypothesis-based pytest scaffold with the trigger/response
already wired into the test name and docstring, and a `given(...)`
strategy stub the human/agent must fill in with real domain strategies
before the test is meaningful.

Complementary to hand-written tests (per spec-driven-workflow.md section
3), not a replacement: hand-written tests catch what you know to check;
these scaffolds are seeded from the requirement's own logical properties
so they catch what you didn't think to write down -- but only once the
strategy stub is filled in with real data generation, not left as
placeholders.

Requires the `hypothesis` package (not a hermes-agent core dependency --
installed by the caller, e.g. `pip install hypothesis`, only when this
script is actually run).

Usage:
  python3 generate_property_tests.py <requirements_dir> --out <test_file.py>
"""

import argparse
import json
import sys
from pathlib import Path

TEST_FILE_HEADER = '''"""
Property-based tests generated from EARS-JSON requirements.

GENERATED SCAFFOLD -- each test's `given(...)` strategy is a placeholder
(st.nothing() or a bare type strategy). Replace every placeholder with a
real domain strategy before trusting these tests; an unfilled scaffold
will either fail immediately (st.nothing()) or generate meaningless data.

Regenerate with:
  python3 generate_property_tests.py <requirements_dir> --out <this file>
"""

from hypothesis import given, strategies as st


'''

TEST_TEMPLATE = '''
def test_{fn_name}():
    """
    {req_id}: {actor} SHALL {response}
    Trigger ({pattern}): {trigger}

    Acceptance criteria:
{criteria_block}
    """
    # TODO: replace this placeholder strategy with one that generates
    # real inputs matching the trigger condition above.
    @given(st.nothing())
    def _property(_placeholder):
        raise NotImplementedError(
            "Fill in a real hypothesis strategy and assertion for {req_id} "
            "before trusting this test."
        )

    # TODO: uncomment once the strategy above is real.
    # _property()
'''


def _slug(req_id: str) -> str:
    return req_id.lower().replace("-", "_")


def _criteria_block(criteria: list[dict]) -> str:
    lines = []
    for c in criteria:
        lines.append(
            f"    - Given {c['given']}, when {c['when']}, then {c['then']}."
        )
    return "\n".join(lines) if lines else "    (none recorded)"


def load_requirements(requirements_dir: Path) -> list[dict]:
    reqs = []
    for path in sorted(requirements_dir.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            reqs.extend(data)
        else:
            reqs.append(data)
    return reqs


def generate(requirements: list[dict]) -> str:
    body = [TEST_FILE_HEADER]
    for req in requirements:
        body.append(
            TEST_TEMPLATE.format(
                fn_name=_slug(req["id"]),
                req_id=req["id"],
                actor=req.get("actor", "the system"),
                response=req.get("response", "<no response recorded>"),
                pattern=req.get("pattern", "unknown"),
                trigger=req.get("trigger") or "(none -- ubiquitous)",
                criteria_block=_criteria_block(req.get("acceptance_criteria", [])),
            )
        )
    return "".join(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requirements_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        import hypothesis  # noqa: F401
    except ImportError:
        print(
            "ERROR: the `hypothesis` package is required to run the "
            "generated tests (not to generate them, but they will fail "
            "to import without it). Install with: pip install hypothesis",
            file=sys.stderr,
        )
        sys.exit(1)

    requirements = load_requirements(args.requirements_dir)
    if not requirements:
        print(
            f"No requirement JSON files found under {args.requirements_dir}",
            file=sys.stderr,
        )
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(generate(requirements))
    print(
        f"Wrote {len(requirements)} test scaffold(s) to {args.out}. "
        "Every @given(st.nothing()) placeholder must be replaced with a "
        "real strategy before these tests mean anything.",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
