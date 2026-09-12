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


class TestEarlyRecovery:
    def test_skips_live_checkout_before_any_probe_or_lock(self, monkeypatch):
        # A probe call would mean recovery is proceeding against the live
        # checkout; the guard must return before ANY side-effectful step.
        # PM generation: recovery's repair action is pm.recovery.
        # repair_dependencies, reached through er.recover_if_needed.
        def _boom(*a, **k):
            raise AssertionError("repair ran against the live checkout")

        import pm.recovery as pm_recovery

        monkeypatch.setattr(pm_recovery, "repair_dependencies", _boom)
        assert er.recover_if_needed(project_root=CHECKOUT_ROOT, argv=[]) is False

    def test_sandboxed_root_still_recovers(self, tmp_path, monkeypatch):
        # The guard must not disable recovery for sandboxed roots: with a
        # marker present, PM repair still runs and clears the marker.
        import pm.recovery as pm_recovery

        (tmp_path / ".lazy-refresh-incomplete").write_text("started=1\npid=0\n")
        (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
        repairs = []
        monkeypatch.setattr(
            pm_recovery, "repair_dependencies", lambda root: repairs.append(root)
        )
        assert er.recover_if_needed(project_root=tmp_path, argv=[]) is True
        assert repairs == [tmp_path], "sandboxed recovery was wrongly disabled by the guard"
        assert not (tmp_path / ".lazy-refresh-incomplete").exists()


class TestLaunchRecovery:
    def test_recover_if_needed_noops_on_live_checkout(self, monkeypatch):
        # PROJECT_ROOT is the live checkout in-suite. Startup recovery must
        # return before touching markers, locks, or repair dependencies.
        import pm.recovery as pm_recovery

        def _boom(*a, **k):
            raise AssertionError("startup recovery ran against the live checkout")

        monkeypatch.setattr(er, "_project_root", lambda: CHECKOUT_ROOT)
        monkeypatch.setattr(pm_recovery, "repair_dependencies", _boom)
        assert er.recover_if_needed(argv=[]) is False
