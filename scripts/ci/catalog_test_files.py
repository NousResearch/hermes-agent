#!/usr/bin/env python3
"""Print the test files the ``catalog`` CI lane runs (catalog-tests.yml).

Every ``tests/**/test_*.py`` outside ``tests/e2e/`` whose source names the
plugin catalog or the contributor email maps. Discovery instead of a list, so a
new test that reads ``plugin-catalog/`` joins the lane without anyone editing
CI. Over-selection only costs seconds; a missed file is a test a catalog-only
PR never runs.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PATTERN = re.compile(
    r"plugin[-_]catalog|load_catalog|get_catalog_dir|contributors[/\"' ,]+emails"
    r"|extract-plugins|fetch-plugin-stars|shared_metrics_catalog"
)


def catalog_test_files(repo: Path = REPO) -> list[str]:
    out = []
    for path in sorted((repo / "tests").rglob("test_*.py")):
        rel = path.relative_to(repo).as_posix()
        if rel.startswith("tests/e2e/"):
            continue
        if PATTERN.search(path.read_text(encoding="utf-8-sig", errors="replace")):
            out.append(rel)
    return out


if __name__ == "__main__":
    print("\n".join(catalog_test_files()))
