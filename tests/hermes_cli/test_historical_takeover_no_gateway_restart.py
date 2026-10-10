"""Historical takeover of `--no-gateway-restart` must defer the gateway restart (#135863).

RED test: `_historical_context` reads the flag from the historical frame's argv
and the schema seams propagate it all the way to `finish_update`, which must
skip the fleet restart exactly like the current completion path does
(`update_completion.py:427`).
"""
from __future__ import annotations

import sys
import types


def _load_old_updater():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "hermes_cli/_old_updater.py"
    spec = importlib.util.spec_from_file_location("_old_updater_isolated", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_takeover_request_propagates_no_gateway_restart_from_argv_tail():
    from hermes_cli.update_handoff import _takeover_request
    payload = {"assume_yes": True, "receipt": {"update_id": "x"}}
    request = _takeover_request(payload, ["update", "--yes", "--no-gateway-restart"])
    assert request.get("no_gateway_restart") is True


def test_takeover_request_defaults_no_gateway_restart_off():
    from hermes_cli.update_handoff import _takeover_request
    request = _takeover_request({"assume_yes": True}, ["update", "--yes"])
    assert request.get("no_gateway_restart", False) is False


def test_finish_update_defers_fleet_restart_when_no_gateway_restart(tmp_path, monkeypatch):
    from hermes_cli import update_finish
    from hermes_cli import update_cmd, update_receipt

    events = []
    monkeypatch.setattr(update_cmd, "_run_post_update_maintenance", lambda **kwargs: True)
    monkeypatch.setattr(update_receipt, "record_skip",
                        lambda name, reason: events.append(("skip", name, reason)))
    monkeypatch.setattr(update_receipt, "record_stage",
                        lambda name, outcome, **facts: events.append(("stage", name, outcome)))
    monkeypatch.setattr(
        update_cmd, "_restart_gateway_fleet_after_update",
        lambda plan, gateway_mode: events.append(("restart",)) or object(),
    )
    monkeypatch.setattr(
        update_cmd, "_resume_windows_gateways_and_merge_outcome", lambda *args: None)
    monkeypatch.setattr(update_cmd, "_verify_fleet_after_update", lambda *args, **kwargs: None)

    update_finish.finish_update(
        root=tmp_path,
        assume_yes=True,
        gateway_mode=False,
        pre_update_snapshot_id=None,
        had_desktop_app_before_update=False,
        pre_update_version=None,
        plan=None,
        windows_resume=None,
        no_gateway_restart=True,
    )

    assert ("skip", "gateway_restart", "--no-gateway-restart: deferred, marker kept") in events
    assert [e for e in events if e[0] == "restart"] == []


def test_old_updater_context_reads_no_gateway_restart_from_sys_argv():
    # A real historical frame: co_name and f_globals.__name__ decide what the
    # frame walk recognizes, so compile one with those identities instead of
    # mutating code objects (readonly).
    module = _load_old_updater()
    namespace = {"__name__": "hermes_cli.update_cmd", "mod": module, "sys": sys}
    exec(compile(
        "def _cmd_update_impl(pre_update_version=None, mod=mod, sys=sys):\n"
        "    sys.argv = ['hermes', 'update', '--no-gateway-restart']\n"
        "    return mod._historical_context()\n",
        "<historical-frame>", "exec",
    ), namespace)
    old_argv = sys.argv
    try:
        context, _, _ = namespace["_cmd_update_impl"]()
    finally:
        sys.argv = old_argv
    assert context.get("no_gateway_restart") is True
