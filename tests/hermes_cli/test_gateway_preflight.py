"""``hermes gateway preflight``: the boot-mode verdict for a host that is NOT booting yet.

The service layer is faked through ``gateway_migrate``'s seams exactly the way
``test_gateway_multiplex_mode.py`` fakes it, so the preflight verdict is exercised against the same
shape of host the boot guard sees.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest

import hermes_constants
from gateway.run_profile_reconcile import GatewayProfileReconcileMixin
from hermes_cli import gateway_migrate as gm
from hermes_cli import gateway_multiplex_mode as mode
from hermes_cli import gateway_preflight as pf
from hermes_cli.gateway_preflight import cmd_preflight, preflight_report


@pytest.fixture
def fleet(tmp_path, monkeypatch):
    """default + coder + ops with distinct bot tokens; nothing runs a gateway unless a test says so."""
    root = tmp_path / "hermes"
    for sub in ("profiles/coder", "profiles/ops"):
        (root / sub).mkdir(parents=True)
    (root / "config.yaml").write_text("model:\n  default: x\n", encoding="utf-8")
    (root / ".env").write_text("TELEGRAM_BOT_TOKEN=111111:default-token\n", encoding="utf-8")
    (root / "profiles/coder/.env").write_text("TELEGRAM_BOT_TOKEN=222222:coder-token\n", encoding="utf-8")
    (root / "profiles/ops/.env").write_text("DISCORD_BOT_TOKEN=ops-discord-333333\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    for name in ("TELEGRAM_BOT_TOKEN", "DISCORD_BOT_TOKEN", "API_SERVER_KEY", "WEBHOOK_ENABLED"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    services: dict[str, list] = {}
    pids: dict[str, int] = {}
    monkeypatch.setattr(gm, "_installed_services", lambda home: services.get(_name(home), []))
    monkeypatch.setattr(gm, "_live_gateway_pid", lambda home: pids.get(_name(home)))
    monkeypatch.setattr(gm, "_host_supports_migration", lambda: None)
    return root, services, pids


def _name(home: Path) -> str:
    return hermes_constants.profile_name_for_home(home) or "default"


def test_compute_multiplex_decision_matches_resolve_and_writes_no_runtime_status(fleet):
    """T1: the factored decision is the boot verdict verbatim — and PURE: no runtime-status record
    appears, so a preflight on a not-yet-booted box changes nothing."""
    root, _services, pids = fleet
    pids["coder"] = 4101  # coder still runs its own gateway -> the boot guard refuses
    status_path = root / "gateway_state.json"
    assert not status_path.exists()

    computed = mode.compute_multiplex_decision(cfg := load_gateway_config_helper())
    resolved = mode.resolve_multiplex_mode(load_gateway_config_helper())
    assert computed == resolved
    assert not computed.enabled and computed.source == "guard"
    assert cfg.multiplex_profiles is None  # compute did NOT settle the config in place...
    assert resolved.enabled is False  # ...resolve did (unchanged boot behaviour)

    pids.clear()
    assert mode.compute_multiplex_decision(load_gateway_config_helper()).enabled
    assert not status_path.exists()


def load_gateway_config_helper():
    from gateway.config import load_gateway_config
    return load_gateway_config()


def test_cmd_preflight_exit_codes_and_json_shape(fleet, monkeypatch, capsys):
    """T2: exit 0 on a clean fleet, 1 under an injected blocker, 0 for a single-profile box; the
    --json shape carries mode/reason/blockers(reason, fix)/profiles."""
    root, _services, _pids = fleet

    with pytest.raises(SystemExit) as exc:
        cmd_preflight(SimpleNamespace(json=False))
    assert exc.value.code == 0  # a clean fleet multiplexes
    assert capsys.readouterr().out.startswith("mode: multiplex\n")

    _real_blocker = mode.implicit_multiplex_blocker
    monkeypatch.setattr(mode, "implicit_multiplex_blocker", lambda: "injected blocker")
    with pytest.raises(SystemExit) as exc:
        cmd_preflight(SimpleNamespace(json=False))
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert out.startswith("mode: standalone (default only)\n")
    assert "reason: injected blocker" in out
    assert out.rstrip().endswith("profiles: default")

    report = json.loads(_json_report(monkeypatch))
    assert set(report) == {"mode", "reason", "blockers", "profiles"}
    assert report["mode"] == "standalone" and report["reason"] == "injected blocker"
    assert report["blockers"] == [{"reason": "injected blocker",
                                   "fix": "run `hermes gateway migrate --multiplex` once fixed"}]
    assert report["profiles"] == ["default"]

    import shutil
    for sub in ("profiles/coder", "profiles/ops"):
        shutil.rmtree(root / sub)
    monkeypatch.setattr(mode, "implicit_multiplex_blocker", _real_blocker)
    with pytest.raises(SystemExit) as exc:
        cmd_preflight(SimpleNamespace(json=False))
    assert exc.value.code in (None, 0)  # single-profile: nothing to multiplex, not a failure
    out = capsys.readouterr().out
    assert out.startswith("mode: single\n")
    assert "run `hermes gateway migrate" not in out  # nothing to multiplex is not a blocker


def _json_report(monkeypatch) -> str:
    import io
    from contextlib import redirect_stdout
    buf = io.StringIO()
    with redirect_stdout(buf):
        with pytest.raises(SystemExit):
            cmd_preflight(SimpleNamespace(json=True))
    return buf.getvalue()


def test_reconcile_warns_about_a_degraded_host_once_per_window(monkeypatch, caplog, tmp_path):
    """T3: while the host is degraded, every reconcile logs the standalone WARN — but rate-limited
    to once per 10 minutes, so the 30 s rescan does not drown the log."""
    monkeypatch.setattr("gateway.run_profile_reconcile._standalone_warn_last", {})
    fake_time = [1000.0]
    monkeypatch.setattr("gateway.run_profile_reconcile.time.monotonic", lambda: fake_time[0])
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda: tmp_path)
    monkeypatch.setattr(
        "gateway.status.read_runtime_status",
        lambda *a, **k: {"multiplex_standalone_reason": "profile(s) 'coder' still run their own gateway"})

    def _degraded():
        return SimpleNamespace(_multiplex_on=lambda: False, served_profile_names=lambda: ["default"])

    def _rescan():
        asyncio.run(GatewayProfileReconcileMixin.reconcile_served_profiles(_degraded()))

    async def _two_rescans():
        await GatewayProfileReconcileMixin.reconcile_served_profiles(_degraded())
        await GatewayProfileReconcileMixin.reconcile_served_profiles(_degraded())

    def _warns():
        return [r for r in caplog.records if "Host is standalone (default only)" in r.getMessage()]

    with caplog.at_level(logging.WARNING, logger="gateway.run_profile_reconcile"):
        asyncio.run(_two_rescans())
    warns = _warns()
    assert len(warns) == 1
    assert "hermes gateway preflight" in warns[0].getMessage()
    assert "'coder'" in warns[0].getMessage()

    fake_time[0] += 60.0  # inside the 10-minute window: still suppressed
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="gateway.run_profile_reconcile"):
        _rescan()
    assert not _warns()

    fake_time[0] += 601.0  # past the window: it warns again
    with caplog.at_level(logging.WARNING, logger="gateway.run_profile_reconcile"):
        _rescan()
    assert len(_warns()) == 1


def test_preflight_report_is_pure_and_reuses_the_migrate_blocker_list(fleet):
    """preflight_report records no boot decision (the gateway's resolve_multiplex_mode does) and
    lifts its blockers from the migration plan."""
    root, _services, _pids = fleet
    from gateway.status import read_runtime_status
    assert not (read_runtime_status() or {}).get("multiplex_standalone_reason")
    report = preflight_report()
    assert not (read_runtime_status() or {}), "preflight must not write the gateway's runtime status"
    assert report["mode"] == "multiplex" and report["blockers"] == []
    assert set(report["profiles"]) == {"default", "coder", "ops"}
