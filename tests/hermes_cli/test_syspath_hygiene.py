"""Regression guard for t_d40ea7e5: no ``tests/`` directory ever on sys.path.

If a ``<root>/tests`` entry lands on ``sys.path`` during a run, the repo's own
``tests/acp/`` package is importable as the bare top-level name ``acp`` and
shadows the real ACP SDK. Any optional-extra gate written as
``pytest.importorskip("acp")`` (or any future same-colliding name) then
false-positives for every test that runs after the pollution in the same
pytest process — the symptom this guard exists to catch.

The concrete failure mode this guards against (observed in the multi-file
``tests/hermes_cli/ -k kanban`` shard): a test module at ``tests/<dir>/``
does ``sys.path.insert(0, str(Path(__file__).resolve().parents[1]))`` —
an off-by-one; the repo root is ``parents[2]``. Because pytest imports
every module in a directory during collection even when ``-k`` deselects
its tests, one such top-level statement pollutes the whole shard before
any deselected file's hooks run. ``parents[1]`` from a file at
``tests/<dir>/`` is almost always wrong; assert accordingly.
"""

import sys
from pathlib import Path


def test_no_tests_directory_on_sys_path() -> None:
    offenders = sorted(
        {
            str(Path(p).resolve())
            for p in sys.path
            if p and Path(p).resolve().name == "tests"
        }
    )
    assert not offenders, (
        "tests/ directories on sys.path (repo tests/acp/ would shadow any "
        f"installed acp SDK, breaking importorskip gates): {offenders}"
    )
