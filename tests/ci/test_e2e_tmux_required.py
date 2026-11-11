"""The release/dispatch E2E lane fails, not skips, when the classic-CLI tmux suites lose tmux.

The tmux suites (``tests/e2e/core/cli_tmux`` and the Vertex ``/bg`` CLI test) skip without tmux,
which is right on a developer box. In the CI ``e2e`` job a skip would turn the lane green over zero
coverage if the runner image ever dropped tmux. That job sets ``HERMES_E2E_REQUIRE_TUI=1`` (and
GitHub sets ``CI``); under both, a missing tmux is a failure.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from tests.e2e.core.cli_tmux import _classic

_REPO = Path(__file__).resolve().parents[2]
_TMUX_SUITES = sorted(
    [*(_REPO / "tests" / "e2e" / "core" / "cli_tmux").glob("test_*.py"),
     _REPO / "tests" / "e2e" / "core" / "providers" / "test_native_vertex_cli_background.py"]
)
_BARE_SKIP = re.compile(r"skipif\(\s*shutil\.which\(\s*[\"']tmux[\"']\s*\)")


def _no_tmux(monkeypatch, *, ci: str | None, require: str | None) -> None:
    monkeypatch.setattr(_classic.shutil, "which", lambda name: None if name == "tmux" else "/bin/x")
    for name, value in (("CI", ci), ("HERMES_E2E_REQUIRE_TUI", require)):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)


def test_missing_tmux_fails_in_the_ci_e2e_lane(monkeypatch):
    _no_tmux(monkeypatch, ci="true", require="1")
    with pytest.raises(pytest.fail.Exception, match="tmux"):
        _classic.require_tmux()


@pytest.mark.parametrize(("ci", "require"), [(None, None), ("true", None), (None, "1")])
def test_missing_tmux_skips_outside_the_ci_e2e_lane(monkeypatch, ci, require):
    _no_tmux(monkeypatch, ci=ci, require=require)
    with pytest.raises(pytest.skip.Exception):
        _classic.require_tmux()


def test_every_tmux_suite_uses_the_gate_instead_of_a_bare_skip():
    assert len(_TMUX_SUITES) >= 4, _TMUX_SUITES
    bare = [str(p.relative_to(_REPO)) for p in _TMUX_SUITES if _BARE_SKIP.search(p.read_text(encoding="utf-8"))]
    assert not bare, f"skip over zero coverage in the e2e lane: {bare}"
    ungated = [str(p.relative_to(_REPO)) for p in _TMUX_SUITES if "require_tmux()" not in p.read_text(encoding="utf-8")]
    assert not ungated, ungated


def test_the_e2e_job_requires_tmux():
    yaml = pytest.importorskip("hermes_yaml")
    job = yaml.safe_load((_REPO / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8"))["jobs"]["e2e"]
    run_step = next(s for s in job["steps"] if s.get("name") == "Run e2e tests")
    assert run_step["env"]["HERMES_E2E_REQUIRE_TUI"] == "1"
    assert any("tmux" in s.get("run", "") and "tmux -V" in s.get("run", "") for s in job["steps"]), \
        "the e2e job must install/verify tmux before the tmux suites run"
