from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path

import pytest

from scripts.refactor_equiv.coverage_gate import CoverageGateError, require_full_branch_coverage
from scripts.refactor_equiv.determinism import Determinism, FROZEN_MONOTONIC, preflight_scan
from scripts.refactor_equiv.equiv_normalize import AllowlistError, lint_allowlist
from scripts.refactor_equiv.mutate import Mutation, MutationMissed, replace_once, run_mutations
from scripts.refactor_equiv.runner import capture


def test_determinism_seams_make_two_captures_byte_identical(tmp_path):
    def run(_case):
        db = tmp_path / "state.db"
        conn = sqlite3.connect(db)
        try:
            return {
                "time": time.time(),
                "monotonic": time.monotonic(),
                "uuid": str(uuid.uuid4()),
                "sql_now": conn.execute("select datetime('now')").fetchone()[0],
                "home": str(Path(__import__("os").environ["HERMES_HOME"]).name),
            }
        finally:
            conn.close()

    cases = [{"name": "one"}]
    first = json.dumps(capture(cases, run), sort_keys=True)
    second = json.dumps(capture(cases, run), sort_keys=True)
    assert first == second
    assert str(FROZEN_MONOTONIC) in first


def test_allowlist_self_lint_rejects_time_named_fields():
    with pytest.raises(AllowlistError):
        lint_allowlist({"created_at": "absolute temp path cannot be stable"})


def test_mutation_harness_exits_nonzero_when_mutation_is_not_detected(tmp_path):
    module = tmp_path / "target.py"
    module.write_text("VALUE = 'old'\n", encoding="utf-8")
    mutation = Mutation("ignored output class", lambda p: replace_once(p, "'old'", "'new'"))

    with pytest.raises(MutationMissed):
        run_mutations(
            module,
            [mutation],
            [sys.executable, "-c", "raise SystemExit(0)"],
        )

    assert module.read_text(encoding="utf-8") == "VALUE = 'old'\n"


def test_mutation_cli_exits_nonzero_when_mutation_is_not_detected(tmp_path):
    module = tmp_path / "relay_headers.py"
    module.write_text(
        textwrap.dedent(
            '''
            _POOL_AFFINITY_PROVIDERS = frozenset({"claude-apr"})
            def f():
                out = {}
                sid = "x"
                out["x-hermes-session"] = sid
                return "background" if (delegated or noninteractive) else "interactive"
            '''
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.refactor_equiv.mutate",
            "--module",
            str(module),
            "--verify-cmd",
            sys.executable,
            "-c",
            "raise SystemExit(0)",
        ],
        cwd=Path.cwd(),
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0


def test_branch_coverage_gate_red_on_uncovered_branch(tmp_path):
    module = tmp_path / "branchy.py"
    module.write_text(
        "def choose(flag):\n"
        "    if flag:\n"
        "        return 'yes'\n"
        "    else:\n"
        "        return 'no'\n",
        encoding="utf-8",
    )
    sys.path.insert(0, str(tmp_path))
    try:
        import branchy

        with pytest.raises(CoverageGateError):
            require_full_branch_coverage(module, lambda: branchy.choose(True))
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("branchy", None)


def test_preflight_scan_requires_named_seams(tmp_path):
    target = tmp_path / "target.py"
    target.write_text("import time\nx = time.monotonic()\n", encoding="utf-8")
    with pytest.raises(Exception):
        preflight_scan([target], named_seams=[])
    preflight_scan([target], named_seams=["monotonic"])


def test_sandbox_guard_refuses_real_home(monkeypatch, tmp_path):
    """The 2026-07-16 incident class: a runner invoked outside Determinism()
    with HERMES_HOME pointing at the operator's real ~/.hermes must refuse."""
    import pytest

    from scripts.refactor_equiv.sandbox_guard import UnsafeHomeError, require_sandboxed_home

    real = Path.home() / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(real))
    with pytest.raises(UnsafeHomeError):
        require_sandboxed_home()
    # ...and any path INSIDE the real home
    monkeypatch.setenv("HERMES_HOME", str(real / "sub"))
    with pytest.raises(UnsafeHomeError):
        require_sandboxed_home()
    # unset is also a refusal
    monkeypatch.delenv("HERMES_HOME")
    with pytest.raises(UnsafeHomeError):
        require_sandboxed_home()


def test_sandbox_guard_allows_temp_and_marker(monkeypatch, tmp_path):
    from scripts.refactor_equiv.sandbox_guard import require_sandboxed_home

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    assert require_sandboxed_home() == (tmp_path / "home").resolve()

    # a non-temp path is allowed only with the explicit marker
    import pytest

    from scripts.refactor_equiv.sandbox_guard import UnsafeHomeError

    # simulate a non-temp location WITHOUT touching the real home: fabricate
    # a fake user home under tmp_path and monkeypatch both Path.home and
    # tempfile.gettempdir so tmp_path itself no longer counts as temp.
    import tempfile

    fake_home = tmp_path / "fakeuser"
    fake_tmp = tmp_path / "faketmp"
    fake_home.mkdir()
    fake_tmp.mkdir()
    import scripts.refactor_equiv.sandbox_guard as sg

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(fake_tmp))
    monkeypatch.setattr(sg, "_EXTRA_TMP_ROOTS", ())

    non_temp = tmp_path / "not-temp-shaped"
    non_temp.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(non_temp))
    with pytest.raises(UnsafeHomeError):
        require_sandboxed_home()
    (non_temp / ".refactor-equiv-sandbox").touch()
    assert require_sandboxed_home() == non_temp.resolve()


def test_state_ext_runner_writes_are_guarded(monkeypatch):
    """The runner itself must refuse before its first write when pointed at
    the real home — the exact incident invocation shape."""
    import pytest

    from scripts.refactor_equiv.sandbox_guard import UnsafeHomeError
    from tests.golden.state_ext import runner as state_runner

    monkeypatch.setenv("HERMES_HOME", str(Path.home() / ".hermes"))
    with pytest.raises(UnsafeHomeError):
        state_runner.run_case({"kind": "denorm_flag", "configs": [True]})
    with pytest.raises(UnsafeHomeError):
        state_runner.run_case(
            {
                "kind": "title_search",
                "name": "guard probe",
                "query": "x",
                "limit": 1,
                "include_archived": False,
            }
        )
