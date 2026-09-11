"""Tests for uv.lock alignment after the update dependency reinstall.

``hermes update`` reinstalls dependencies with ``uv pip install -e .[all]``,
which resolves from pyproject.toml constraints and never reads ``uv.lock``.
A lockfile-only version bump (the ``fix(sec)`` pattern) therefore leaves an
existing venv on the old release even after a successful update. These tests
cover the pin collection, the alignment plan, and the non-fatal contract of
the install step itself.
"""

from __future__ import annotations

import subprocess

import pytest

import hermes_cli.main as main
import hermes_cli.main_install_repair as m

LOCK = """
version = 1

[[package]]
name = "httplib2"
version = "0.32.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "PyNaCl"
version = "1.6.2"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "hermes-agent"
version = "0.19.1"
source = { editable = "." }

[[package]]
name = "forked-dep"
version = "1.0.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "forked-dep"
version = "2.0.0"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "git-dep"
version = "0.1.0"
source = { git = "https://example.invalid/repo?rev=abc" }
"""


def _write_lock(tmp_path):
    (tmp_path / "uv.lock").write_text(LOCK, encoding="utf-8")


def test_collect_lockfile_pins_filters_and_canonicalizes(tmp_path):
    _write_lock(tmp_path)
    pins = m._collect_lockfile_pins(tmp_path)
    # Registry packages survive with canonical names; the editable project,
    # the git source, and the platform-forked double entry are all skipped.
    assert pins == {"httplib2": "0.32.0", "pynacl": "1.6.2"}


def test_collect_lockfile_pins_missing_lock(tmp_path):
    assert m._collect_lockfile_pins(tmp_path) == {}


def test_collect_lockfile_pins_unparseable_lock(tmp_path):
    (tmp_path / "uv.lock").write_text("not = [valid", encoding="utf-8")
    assert m._collect_lockfile_pins(tmp_path) == {}


def test_plan_only_targets_drifted_installed_packages():
    pins = {"httplib2": "0.32.0", "pynacl": "1.6.2", "pygments": "2.20.0"}
    installed = {"httplib2": "0.31.2", "pynacl": "1.6.2", "requests": "2.32.0"}
    # Drifted installed package is realigned; matching stays untouched;
    # a locked package that is not installed is never added.
    assert m._plan_lockfile_alignment(pins, installed) == ["httplib2==0.32.0"]


def test_plan_is_empty_when_everything_matches():
    pins = {"httplib2": "0.32.0"}
    installed = {"httplib2": "0.32.0"}
    assert m._plan_lockfile_alignment(pins, installed) == []


def test_align_runs_single_batched_install(tmp_path, monkeypatch):
    _write_lock(tmp_path)
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    calls = []

    monkeypatch.setattr(
        m,
        "_list_installed_package_versions",
        lambda prefix, *, env=None: {"httplib2": "0.31.2", "pynacl": "1.6.2"},
    )

    def fake_install(cmd, *, env=None, scripts_dir=None, **_kw):
        calls.append(cmd)

    monkeypatch.setattr(m, "_run_quarantined_install", fake_install)

    m._align_installed_packages_with_lockfile(["uv", "pip"])
    assert calls == [["uv", "pip", "install", "httplib2==0.32.0"]]


def test_align_no_op_without_lockfile(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    def explode(*args, **kwargs):  # pragma: no cover - guards the no-op path
        raise AssertionError("should not be called without a lockfile")

    monkeypatch.setattr(m, "_list_installed_package_versions", explode)
    monkeypatch.setattr(m, "_run_quarantined_install", explode)
    m._align_installed_packages_with_lockfile(["uv", "pip"])


def _neutralize_install_machinery(monkeypatch, align_calls, install_calls=None):
    """Stub the subprocess-heavy pieces under the shared reinstall boundary.

    The install doubles absorb unknown keywords. They stand in for
    ``_run_quarantined_install``, whose signature grows over time (it gained
    ``strict_quarantine`` in #87331), and pinning it here would fail the suite
    for a change that has nothing to do with lockfile alignment.
    """

    def fake_install(cmd, *, env=None, scripts_dir=None, **_kw):
        if install_calls is not None:
            install_calls.append(cmd)

    monkeypatch.setattr(m, "_run_quarantined_install", fake_install)
    monkeypatch.setattr(
        m, "_verify_core_dependencies_installed", lambda *a, **k: None
    )
    monkeypatch.setattr(
        m, "_verify_console_scripts_installed", lambda *a, **k: None
    )
    monkeypatch.setattr(
        m,
        "_align_installed_packages_with_lockfile",
        lambda prefix, *, env=None: align_calls.append(list(prefix)),
        raising=False,
    )


def test_boundary_aligns_after_happy_path_install(monkeypatch):
    align_calls: list[list[str]] = []
    _neutralize_install_machinery(monkeypatch, align_calls)

    m._install_python_dependencies_with_optional_fallback(["uv", "pip"])
    assert align_calls == [["uv", "pip"]]


def test_boundary_aligns_after_extras_fallback_path(monkeypatch):
    align_calls: list[list[str]] = []
    install_calls: list[list[str]] = []
    _neutralize_install_machinery(monkeypatch, align_calls, install_calls)
    monkeypatch.setattr(
        m, "_load_installable_optional_extras", lambda group="all": ["voice"]
    )

    state = {"first": True}

    def flaky_install(cmd, *, env=None, scripts_dir=None, **_kw):
        install_calls.append(cmd)
        if state["first"]:
            state["first"] = False
            raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(m, "_run_quarantined_install", flaky_install)

    m._install_python_dependencies_with_optional_fallback(["uv", "pip"])
    # The .[all] attempt failed, base and extras were reinstalled, and the
    # alignment boundary still ran exactly once at the end.
    assert install_calls[0] == ["uv", "pip", "install", "-e", ".[all]"]
    assert ["uv", "pip", "install", "-e", "."] in install_calls
    assert align_calls == [["uv", "pip"]]


def test_recovery_route_reaches_alignment_and_clears_marker(
    tmp_path, monkeypatch
):
    """Core-marker recovery runs the real shared installer, so alignment
    happens before the ``.update-incomplete`` breadcrumb is cleared."""
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    from hermes_cli.update_cmd import _write_update_incomplete_marker
    _write_update_incomplete_marker()
    marker = tmp_path / ".update-incomplete"
    assert marker.exists()

    align_calls: list[list[str]] = []
    _neutralize_install_machinery(monkeypatch, align_calls)
    monkeypatch.setattr(
        m, "_windows_running_hermes_launcher_locked", lambda: False
    )

    import hermes_cli.managed_uv as managed_uv

    monkeypatch.setattr(managed_uv, "ensure_uv", lambda: None)

    class _Done:
        returncode = 0

    monkeypatch.setattr(
        m.subprocess, "run", lambda *a, **k: _Done(), raising=True
    )

    m._recover_core_update_marker_locked()

    assert len(align_calls) == 1
    assert not marker.exists()


def test_align_is_non_fatal_on_install_failure(tmp_path, monkeypatch, capsys):
    _write_lock(tmp_path)
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        m,
        "_list_installed_package_versions",
        lambda prefix, *, env=None: {"httplib2": "0.31.2"},
    )

    def boom(cmd, *, env=None, scripts_dir=None, **_kw):
        raise subprocess.CalledProcessError(2, cmd)

    monkeypatch.setattr(m, "_run_quarantined_install", boom)

    m._align_installed_packages_with_lockfile(["uv", "pip"])
    out = capsys.readouterr().out
    assert "Could not align packages with uv.lock" in out
    assert "exit 2" in out


# ---------------------------------------------------------------------------
# Alignment inherits the update sync's quarantine contract (#87331).
#
# Alignment runs inside _install_python_dependencies_with_optional_fallback,
# which is the update dependency sync. A shim that cannot be renamed aside
# proves a hard venv hold, so installing pinned versions anyway would strand
# the venv between resolutions: exactly the half-updated state strict mode
# exists to prevent.
# ---------------------------------------------------------------------------

def test_alignment_install_is_strictly_quarantined(tmp_path, monkeypatch):
    _write_lock(tmp_path)
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        m,
        "_list_installed_package_versions",
        lambda prefix, *, env=None: {"httplib2": "0.31.2"},
    )

    seen = {}

    def spy(cmd, *, env=None, scripts_dir=None, strict_quarantine=False):
        seen["strict"] = strict_quarantine

    monkeypatch.setattr(m, "_run_quarantined_install", spy)

    m._align_installed_packages_with_lockfile(["uv", "pip"])

    assert seen["strict"] is True, (
        "a contended venv must not be mutated by the alignment install either"
    )


def test_shim_quarantine_refusal_is_not_softened_into_a_warning(
    tmp_path, monkeypatch
):
    """A held venv defers the update; it does not 'keep current versions'.

    Every other install failure is advisory here, because the venv is already
    consistent and only the lockfile pins are missing. A quarantine refusal is
    different: nothing was installed and the venv is held, so it has to reach
    the sync boundary, which defers via the update-incomplete marker.
    """
    _write_lock(tmp_path)
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        m,
        "_list_installed_package_versions",
        lambda prefix, *, env=None: {"httplib2": "0.31.2"},
    )

    def refuse(cmd, *, env=None, scripts_dir=None, **_kw):
        raise m.ShimQuarantineError(["hermes.exe"])

    monkeypatch.setattr(m, "_run_quarantined_install", refuse)

    with pytest.raises(m.ShimQuarantineError):
        m._align_installed_packages_with_lockfile(["uv", "pip"])
