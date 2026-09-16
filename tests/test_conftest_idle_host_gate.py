"""The ``requires_idle_host`` gate must key off a MEASURED live runtime.

``tests/conftest.py`` skips tests marked ``requires_idle_host`` when this
machine already runs a Hermes gateway/dashboard, because the product then
correctly refuses to act (``SystemExit(75)`` / ``HTTP 409``) and the test can
only fail. The danger of any such gate is the silent direction: if the probe
returned True everywhere, real coverage would vanish with a green summary.
These tests pin both directions of the probe and of the skip it drives.
"""

import pytest

from tests.conftest import (
    _REQUIRES_IDLE_HOST_MARK,
    _live_hermes_runtime_present,
    pytest_collection_modifyitems,
)


def _fake_proc(tmp_path, pid_to_argv):
    root = tmp_path / "proc"
    root.mkdir()
    for pid, argv in pid_to_argv.items():
        entry = root / str(pid)
        entry.mkdir()
        entry.joinpath("cmdline").write_bytes(b"\x00".join(argv) + b"\x00")
    # Non-numeric entries are real in /proc (self, sys, meminfo...) and must be
    # walked past rather than crashing the scan.
    root.joinpath("self").mkdir()
    root.joinpath("meminfo").write_text("MemTotal: 1 kB\n")
    return root


@pytest.mark.parametrize(
    "argv",
    [
        [b"/venv/bin/python", b"-m", b"hermes_cli.main", b"gateway", b"run"],
        [b"/venv/bin/python", b"-m", b"hermes_cli.main", b"--profile",
         b"jarvis", b"gateway", b"run"],
        [b"/venv/bin/python3", b"/venv/bin/hermes", b"dashboard"],
    ],
)
def test_probe_sees_a_live_runtime(tmp_path, argv):
    root = _fake_proc(tmp_path, {4242: argv})
    assert _live_hermes_runtime_present(root) is True


@pytest.mark.parametrize(
    "argv",
    [
        [b"/venv/bin/python", b"-m", b"pytest", b"tests/"],
        [b"/usr/bin/sshd"],
        # Named like Hermes but not a running runtime: must NOT gate.
        [b"/venv/bin/python", b"-m", b"hermes_cli.main", b"config", b"get"],
        [b"/bin/grep", b"hermes_cli.main gateway run"],
    ],
)
def test_probe_ignores_everything_else(tmp_path, argv):
    root = _fake_proc(tmp_path, {4242: argv})
    assert _live_hermes_runtime_present(root) is False


def test_probe_is_false_without_proc(tmp_path):
    """No /proc (macOS, Windows) means no gating - the safe direction."""
    assert _live_hermes_runtime_present(tmp_path / "does-not-exist") is False


class _Item:
    def __init__(self, marked):
        self._marked = marked
        self.added = []

    def get_closest_marker(self, name):
        if name == _REQUIRES_IDLE_HOST_MARK and self._marked:
            return object()
        return None

    def iter_markers(self):
        return iter(())

    def add_marker(self, marker):
        self.added.append(marker)


def _run_gate(monkeypatch, live):
    import tests.conftest as cf

    monkeypatch.setattr(cf, "_live_hermes_runtime_present", lambda *a, **k: live)
    marked, plain = _Item(True), _Item(False)
    pytest_collection_modifyitems(None, [marked, plain])
    return marked, plain


def test_marked_test_is_skipped_when_a_runtime_is_live(monkeypatch):
    marked, plain = _run_gate(monkeypatch, live=True)
    assert len(marked.added) == 1
    reason = marked.added[0].kwargs["reason"]
    assert "local" in reason and ("75" in reason or "409" in reason)
    assert plain.added == [], "an unmarked test must never be gated"


def test_marked_test_runs_on_an_idle_host(monkeypatch):
    """The whole point: a pristine CI runner still executes these tests."""
    marked, plain = _run_gate(monkeypatch, live=False)
    assert marked.added == []
    assert plain.added == []


def test_marker_is_registered_for_strict_markers():
    import tomllib
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads(root.joinpath("pyproject.toml").read_text())
    markers = data["tool"]["pytest"]["ini_options"]["markers"]
    assert any(m.startswith(f"{_REQUIRES_IDLE_HOST_MARK}:") for m in markers), (
        "an unregistered marker is silently ignored under --strict-markers"
    )
