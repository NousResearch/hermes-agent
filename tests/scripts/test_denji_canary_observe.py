"""Driver test for scripts/denji-canary-observe.py (P2-3 close-the-loop).

Verifies the orchestration: eval -> observe -> revert-on-regression -> fleet
health refresh. The blast-radius primitives themselves are covered in
test_blast_radius.py; here we stub them and assert the wiring.
"""
from __future__ import annotations

import importlib.util
import types
from pathlib import Path

import pytest

_DRIVER_PATH = (
    Path(__file__).resolve().parents[2] / "scripts" / "denji-canary-observe.py"
)


def _load_driver():
    spec = importlib.util.spec_from_file_location("denji_canary_observe", _DRIVER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Canary:
    def __init__(self, edit_id, profile, stage, commit=""):
        self.edit_id = edit_id
        self.profile = profile
        self.stage = stage
        self.commit = commit


class _EvalRun:
    def __init__(self, pass_rate):
        self.pass_rate = pass_rate


def test_regressed_canary_is_reverted(monkeypatch):
    driver = _load_driver()
    import hermes_cli.blast_radius as br
    import hermes_cli.eval_harness as eh

    canary = _Canary("e1", "coder", br.CanaryStage.APPLIED, commit="deadbeef0000")
    reverted = _Canary("e1", "coder", br.CanaryStage.REVERTED, commit="deadbeef0000")

    fake_guard = types.SimpleNamespace(
        active_canaries=lambda: [canary],
        observe_canary=lambda eid, run: reverted,
        check_fleet_health=lambda **kw: types.SimpleNamespace(
            tripwire=types.SimpleNamespace(value="normal")
        ),
    )
    monkeypatch.setattr(br, "EditGuard", lambda *a, **k: fake_guard)
    monkeypatch.setattr(eh, "run_eval", lambda tasks, **kw: _EvalRun(0.2))
    monkeypatch.setattr(eh.GoldenTask, "load_set", staticmethod(lambda p: ["t"]))
    monkeypatch.setattr(driver, "_load_eval_domains", lambda: {"coder": "code"})
    monkeypatch.setattr(driver, "_golden_set_for_domain", lambda d: "golden.yaml")

    reverts = []
    monkeypatch.setattr(driver, "_revert_commit", lambda c: reverts.append(c) or True)

    assert driver.main() == 0
    assert reverts == ["deadbeef0000"]


def test_no_canaries_is_noop(monkeypatch):
    driver = _load_driver()
    import hermes_cli.blast_radius as br

    fake_guard = types.SimpleNamespace(active_canaries=lambda: [])
    monkeypatch.setattr(br, "EditGuard", lambda *a, **k: fake_guard)
    assert driver.main() == 0


def test_empty_golden_set_skipped_no_false_revert(monkeypatch):
    driver = _load_driver()
    import hermes_cli.blast_radius as br
    import hermes_cli.eval_harness as eh

    canary = _Canary("e3", "coder", br.CanaryStage.APPLIED, commit="abc")
    fake_guard = types.SimpleNamespace(
        active_canaries=lambda: [canary],
        observe_canary=lambda *a, **k: pytest.fail("must not observe on empty set"),
        check_fleet_health=lambda **kw: None,
    )
    monkeypatch.setattr(br, "EditGuard", lambda *a, **k: fake_guard)
    monkeypatch.setattr(driver, "_load_eval_domains", lambda: {"coder": "code"})
    monkeypatch.setattr(driver, "_golden_set_for_domain", lambda d: "golden.yaml")
    monkeypatch.setattr(eh.GoldenTask, "load_set", staticmethod(lambda p: []))
    ran = []
    monkeypatch.setattr(eh, "run_eval", lambda *a, **k: ran.append(1))
    reverts = []
    monkeypatch.setattr(driver, "_revert_commit", lambda c: reverts.append(c))
    assert driver.main() == 0
    assert ran == [] and reverts == []


def test_unmapped_profile_skipped_not_reverted(monkeypatch):
    driver = _load_driver()
    import hermes_cli.blast_radius as br
    import hermes_cli.eval_harness as eh

    canary = _Canary("e2", "mystery", br.CanaryStage.APPLIED, commit="c0ffee")
    fake_guard = types.SimpleNamespace(
        active_canaries=lambda: [canary],
        observe_canary=lambda *a, **k: pytest.fail("should not observe unmapped"),
        check_fleet_health=lambda **kw: None,
    )
    monkeypatch.setattr(br, "EditGuard", lambda *a, **k: fake_guard)
    monkeypatch.setattr(driver, "_load_eval_domains", lambda: {})  # no mapping
    reverts = []
    monkeypatch.setattr(driver, "_revert_commit", lambda c: reverts.append(c))
    assert driver.main() == 0
    assert reverts == []
