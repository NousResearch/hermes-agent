"""The test suite must never mutate the LIVE checkout or its venv.

Regression tests for the pytest-guards on the checkout-root mutation paths.
Before the guards, tests that drove ``cmd_update``/recovery without
sandboxing ``PROJECT_ROOT`` left ``.lazy-refresh-incomplete`` at the real
repo root (observed after full-suite runs), and — on a venv with genuinely
broken packages — test-spawned subprocesses importing ``hermes_cli.main``
ran a REAL ``ensurepip`` + ``pip install --force-reinstall`` against the
developer's executing environment mid-suite.

The guard predicate requires BOTH conditions (under pytest AND the target is
this checkout itself), so every tmp_path-sandboxed test keeps exercising the
real code paths unchanged.
"""

from __future__ import annotations

from pathlib import Path

import hermes_cli.main as main_mod
from hermes_cli import _early_recovery as er

CHECKOUT_ROOT = Path(er.__file__).resolve().parent.parent


class TestPredicate:
    def test_true_for_live_checkout_under_pytest(self):
        # PYTEST_CURRENT_TEST is set by pytest itself right now.
        assert er._pytest_owns_live_checkout(CHECKOUT_ROOT) is True
        assert main_mod._pytest_owns_live_checkout(CHECKOUT_ROOT) is True

    def test_false_for_sandboxed_root(self, tmp_path):
        assert er._pytest_owns_live_checkout(tmp_path) is False
        assert main_mod._pytest_owns_live_checkout(tmp_path) is False

    def test_false_outside_pytest(self, monkeypatch):
        monkeypatch.delenv("PYTEST_CURRENT_TEST", raising=False)
        assert er._pytest_owns_live_checkout(CHECKOUT_ROOT) is False
        assert main_mod._pytest_owns_live_checkout(CHECKOUT_ROOT) is False


class TestMarkerWrites:
    def test_refuses_breadcrumb_at_live_repo_root(self):
        target = CHECKOUT_ROOT / ".lazy-refresh-incomplete"
        # The marker may legitimately pre-exist: upstream currently TRACKS a
        # littered copy in git (the exact pollution this guard prevents), so
        # the contract is content-unchanged, not never-exists.
        before = target.read_text(encoding="utf-8") if target.exists() else None
        try:
            main_mod._write_marker_file(target, label="lazy-refresh-incomplete")
            after = (
                target.read_text(encoding="utf-8") if target.exists() else None
            )
            assert after == before, (
                "marker breadcrumb written into the LIVE checkout from a test"
            )
        finally:
            # If the guard is broken (RED state), restore the pre-test state —
            # leaving pollution behind is exactly the bug being pinned.
            if before is None:
                target.unlink(missing_ok=True)
            else:
                target.write_text(before, encoding="utf-8")

    def test_still_writes_sandboxed(self, tmp_path):
        target = tmp_path / ".lazy-refresh-incomplete"
        main_mod._write_marker_file(target, label="lazy-refresh-incomplete")
        assert target.exists()
        assert "pid=" in target.read_text(encoding="utf-8")


class TestEarlyRecovery:
    def test_skips_live_checkout_before_any_probe_or_lock(self, monkeypatch):
        # A probe call would mean recovery is proceeding against the live
        # checkout; the guard must return before ANY side-effectful step.
        def _boom():
            raise AssertionError("probe ran against the live checkout")

        monkeypatch.setattr(er, "_probe_broken_packages", _boom)
        monkeypatch.setattr(er, "_run_repair_install", lambda *a, **k: _boom())
        er.recover_if_needed(project_root=CHECKOUT_ROOT, argv=[])

    def test_sandboxed_root_still_recovers(self, tmp_path, monkeypatch):
        # The guard must not disable recovery for sandboxed roots: with a
        # marker present and a broken probe, the repair path still runs.
        (tmp_path / ".lazy-refresh-incomplete").write_text("started=1\npid=1\n")
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        monkeypatch.setattr(er, "_probe_broken_packages", lambda: ["PyYAML"])
        monkeypatch.setattr(er, "_pinned_specs", lambda broken, root: broken)
        installs = []
        monkeypatch.setattr(
            er, "_run_repair_install", lambda specs, root: installs.append(specs) or True
        )
        er.recover_if_needed(project_root=tmp_path, argv=[])
        assert installs, "sandboxed recovery was wrongly disabled by the guard"


class TestLaunchRecovery:
    def test_recover_from_interrupted_install_noops_on_live_checkout(
        self, monkeypatch
    ):
        # PROJECT_ROOT is the live checkout in-suite; the launch-time
        # recovery must return before touching markers or spawning installs.
        def _boom(*a, **k):
            raise AssertionError("launch recovery ran against the live checkout")

        monkeypatch.setattr(main_mod, "_update_marker_path", _boom)
        main_mod._recover_from_interrupted_install()
