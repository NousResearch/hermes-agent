"""A completed update must not exit 1 over an unverified gateway respawn (#123971).

``_resume_windows_gateways_and_merge_outcome`` folds Windows gateway resume failures into the
restart outcome. A verify miss AFTER the update completed (the watchers ran, the poll saw no
stable gateway) is a manual follow-up, not a failed update:

* ``outcome.incomplete`` must stay False — that flag is what makes
  ``_verify_fleet_after_update`` exit 1 and leaves ``fleet_restart_pending`` behind, turning
  one healthy update into a repeating failure loop.
* the gateway update exit code must NOT be written False (the desktop reads it as failure).
* the receipt still records the miss through ``outcome.phase_errors`` (surfaced by
  ``record_gateway_restart``).
* ``token["resume_needed"]`` is cleared before the raise, so no atexit replay re-runs
  ``gateway run --replace`` against the fresh gateway the watcher did manage to start.
"""

from __future__ import annotations

import inspect

from hermes_cli import update_cmd_windows


class _Outcome:
    def __init__(self):
        self.incomplete = False
        self.phase_errors: list[str] = []
        self.restarted_services: list[str] = []
        self.relaunched_profiles: list[str] = []
        self.externally_supervised_profiles: list[str] = []
        self.killed_pids: list[int] = []
        self.failed_or_stale_units: list[str] = []


def _run_merge(monkeypatch, token, resume_exc):
    """Call _resume_windows_gateways_and_merge_outcome with stubbed collaborators."""
    calls = {}

    import hermes_cli.update_cmd as _uc

    monkeypatch.setattr(
        _uc, "_resume_windows_gateways_after_update",
        lambda _token: (_ for _ in ()).throw(resume_exc) if resume_exc else None,
    )
    monkeypatch.setattr(_uc, "_m", lambda: _uc, raising=False)
    monkeypatch.setattr(
        _uc, "_write_gateway_update_exit_code",
        lambda ok: calls.setdefault("exit_code_write", ok),
    )
    # record_gateway_restart imports must not blow up; stub at module level.
    import hermes_cli.update_receipt as _ur

    monkeypatch.setattr(_ur, "record_gateway_restart", lambda **_kw: None, raising=False)

    outcome = _Outcome()
    update_cmd_windows._resume_windows_gateways_and_merge_outcome(outcome, token, gateway_mode=True)
    return outcome, calls


def test_unverified_relaunch_is_a_manual_followup_not_an_incomplete_update(monkeypatch):
    exc = update_cmd_windows._GatewayRelaunchUnverified(
        "Windows gateway relaunch after update was not verified alive"
    )
    token = {"profiles": {"default": 14980}, "unmapped": []}
    outcome, calls = _run_merge(monkeypatch, token, exc)

    assert outcome.incomplete is False
    assert calls.get("exit_code_write") is None  # never written False
    assert any("not verified alive" in e for e in outcome.phase_errors)


def test_genuine_resume_failures_still_mark_the_update_incomplete(monkeypatch):
    exc = RuntimeError("Could not restart every paused Windows gateway")
    token = {"profiles": {"default": 14980}, "unmapped": []}
    outcome, calls = _run_merge(monkeypatch, token, exc)

    assert outcome.incomplete is True
    assert calls.get("exit_code_write") is False
    assert outcome.phase_errors


def test_verify_miss_clears_resume_needed_before_raising(monkeypatch):
    """The atexit replay must not re-run `gateway run --replace` over the fresh gateway.

    ``_verify_relaunched_gateways_alive`` runs inside the resume path; assert from its
    source contract what the merge path cannot fake: the token's resume obligation is
    cleared in the same code block that raises, so no replay is left armed (#123971).
    """
    source = inspect.getsource(update_cmd_windows._verify_relaunched_gateways_alive)
    assert 'token["resume_needed"] = False' in source
    raise_marker = source.index("_GatewayRelaunchUnverified")
    clear_marker = source.index('token["resume_needed"] = False')
    assert clear_marker < raise_marker
