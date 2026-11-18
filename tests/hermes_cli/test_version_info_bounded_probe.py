"""Version probes must be bounded on both ends.

Regression context: version resolution runs a ``git`` probe every time
``install-stamp.json`` reports ``source: "git"`` (a source checkout). The old
implementation used ``subprocess.run(..., timeout=3)``, whose post-timeout cleanup
awaits an *unbounded* ``communicate()``; when the killed launcher leaves a descendant
``git.exe`` holding the captured pipes, that join never returns and the calling
worker wedges instead of degrading the version label. ``_run_git`` now routes through
``_subprocess_compat.bounded_probe_run`` (job-object containment + tree-kill + 1 s
drain + abandon).

These tests pin the routing (cheap) and the bound itself (one real hung child), so a
future refactor back to ``subprocess.run`` fails here rather than in a gateway worker.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from hermes_cli import version_info


def test_run_git_routes_through_bounded_probe(monkeypatch, tmp_path: Path):
    """``_run_git`` must call the deadlock-safe helper with the repo as cwd."""
    seen: dict = {}

    def fake_probe(argv, **kwargs):
        seen["argv"] = list(argv)
        seen.update(kwargs)
        return subprocess.CompletedProcess(list(argv), 0, "abc1234\n", "")

    monkeypatch.setattr(version_info, "bounded_probe_run", fake_probe)

    assert version_info._run_git(tmp_path, "rev-parse", "HEAD") == "abc1234"
    assert seen["argv"] == ["git", "rev-parse", "HEAD"]
    assert seen["cwd"] == str(tmp_path)
    assert seen["timeout"] == version_info._GIT_PROBE_TIMEOUT_SECONDS
    assert isinstance(seen["env"], dict)


def test_run_git_degrades_instead_of_raising(monkeypatch, tmp_path: Path):
    """A timeout (``None``), a non-zero exit and blank output all read as 'no value'."""
    monkeypatch.setattr(version_info, "bounded_probe_run", lambda argv, **kwargs: None)
    assert version_info._run_git(tmp_path, "rev-parse", "HEAD") is None

    monkeypatch.setattr(
        version_info,
        "bounded_probe_run",
        lambda argv, **kwargs: subprocess.CompletedProcess(list(argv), 128, "", "boom"),
    )
    assert version_info._run_git(tmp_path, "rev-parse", "HEAD") is None

    monkeypatch.setattr(
        version_info,
        "bounded_probe_run",
        lambda argv, **kwargs: subprocess.CompletedProcess(list(argv), 0, "  \n", ""),
    )
    assert version_info._run_git(tmp_path, "rev-parse", "HEAD") is None


def test_bounded_probe_abandons_a_hung_child():
    """The bound is real: a child that outlives its timeout returns ``None`` promptly."""
    started = time.monotonic()
    result = version_info.bounded_probe_run(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        timeout=1,
    )
    elapsed = time.monotonic() - started
    assert result is None
    # 1 s timeout + 1 s drain, with slack for process spawn/kill on a loaded box.
    assert elapsed < 8, f"bounded probe took {elapsed:.1f}s — the timeout is not bounding"
