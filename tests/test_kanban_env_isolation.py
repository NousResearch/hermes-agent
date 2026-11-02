"""Kanban worker-shell env vars must not leak into the test process.

``tests/conftest.py::_hermetic_environment`` blanks
``tests/_fixtures/env_filter._HERMES_BEHAVIORAL_VARS`` before every test, but that
list is hand-maintained while the family it has to cover is minted by the
dispatcher: a dispatched worker exports ``HERMES_KANBAN_*`` pins into the
environment of everything it runs, and the suite is routinely started from
inside such a shell. A member the list does not know about changes what the
tests under it assert — an ambient ``GOAL_MODE`` sends the quiet/single-query
path down the goal branch, an ambient ``CRASH_GRACE_SECONDS`` /
``RATE_LIMIT_COOLDOWN_SECONDS`` / ``CLAIM_TTL_SECONDS`` silently retimes the
dispatcher, ``SPECIFY_MAX_TOKENS`` is folded into a module constant at import.

Two halves of the family are pinned here — the names production code reads from
the ambient environment, and the identity keys ``agent.delegation_context``
already declares — plus one test that drives the fixture itself and proves an
ambient value is actually cleared.
"""

from __future__ import annotations

import inspect
import os
import re
from pathlib import Path

from tests._fixtures.env_filter import _HERMES_BEHAVIORAL_VARS

_REPO_ROOT = Path(__file__).resolve().parents[1]
# Production surfaces that can read an ambient kanban pin. ``tests/``, ``scripts/``
# and ``evals/`` are excluded on purpose: a harness that sets a pin for a child
# process is not the leak this guards.
_SCAN_DIRS = ("agent", "gateway", "hermes_cli", "tools", "plugins", "cron")

# Environment READS only. A name that is merely written into a child's env block
# is covered by the ``KANBAN_ENV_KEYS`` test below when it is identity-bearing,
# and needs no entry otherwise.
_AMBIENT_READ_RE = re.compile(
    r"(?:os\.environ\.get|os\.getenv|environ\.get|_env_int|env_int|os\.environ\s*\[)"
    r"\s*[\(\[]?\s*[\"'](HERMES_KANBAN_[A-Z0-9_]+)[\"']"
)


def _ambient_kanban_reads() -> dict[str, str]:
    """``{name: first file that reads it}`` for every ambient kanban pin read by production code."""
    reads: dict[str, str] = {}
    paths = [p for d in _SCAN_DIRS for p in sorted((_REPO_ROOT / d).rglob("*.py"))]
    paths += sorted(_REPO_ROOT.glob("*.py"))
    for path in paths:
        if "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for match in _AMBIENT_READ_RE.finditer(text):
            reads.setdefault(match.group(1), path.relative_to(_REPO_ROOT).as_posix())
    return reads


def test_every_kanban_var_read_from_the_environment_is_cleared():
    reads = _ambient_kanban_reads()
    assert reads, "the scan found no HERMES_KANBAN_* environment read; the pattern or the tree moved"

    missing = {name: where for name, where in reads.items() if name not in _HERMES_BEHAVIORAL_VARS}
    assert not missing, (
        "these HERMES_KANBAN_* names are read from the ambient environment but are missing from "
        "tests/_fixtures/env_filter._HERMES_BEHAVIORAL_VARS, so a worker-shell value leaks into every "
        "test that reads them: "
        + ", ".join(f"{name} ({where})" for name, where in sorted(missing.items()))
    )


def test_kanban_identity_keys_are_cleared():
    """``KANBAN_ENV_KEYS`` is the tree's own declaration of dispatcher-owned identity."""
    from agent.delegation_context import KANBAN_ENV_KEYS

    missing = [name for name in KANBAN_ENV_KEYS if name not in _HERMES_BEHAVIORAL_VARS]
    assert not missing, (
        "agent.delegation_context.KANBAN_ENV_KEYS declares these dispatcher-owned pins, but the hermetic "
        f"fixture does not clear them: {', '.join(missing)}"
    )


def _hermetic_fixture_function():
    from tests import conftest

    # pytest wraps fixtures; ``inspect.unwrap`` reaches the body behind the marker.
    return inspect.unwrap(conftest._hermetic_environment)


def test_hermetic_fixture_clears_an_ambient_worker_shell_var(tmp_path, tmp_path_factory, monkeypatch):
    """Drive the fixture twice: values set after the first pass must not survive the second."""
    monkeypatch.setenv("HERMES_KANBAN_GOAL_MODE", "1")
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    assert os.environ["HERMES_KANBAN_GOAL_MODE"] == "1"

    second_home = tmp_path / "second_pass"
    second_home.mkdir()
    _hermetic_fixture_function()(second_home, tmp_path_factory, monkeypatch)

    assert "HERMES_KANBAN_GOAL_MODE" not in os.environ
    assert "HERMES_KANBAN_CRASH_GRACE_SECONDS" not in os.environ
