"""A served secondary's api_server/webhook are MIRRORS of the default's listener (``/p/<profile>/...``),
never adapters of their own, so the multiplexer record has no ``<profile>:api_server`` entry. Every
status reader used to fall through to ``pending_restart``: the Desktop Messaging card and Command
Center read "Restart needed" forever for a platform that was answering. The mirror must project as the
default's live state plus the URL the client has to call; ``/api/status?profile=`` names the profiles a
restart of the shared gateway would blip.
"""

from __future__ import annotations

import json
import os

import pytest


@pytest.fixture
def served_root(tmp_path, monkeypatch):
    root = tmp_path / "hermes"
    (root / "profiles" / "alpha").mkdir(parents=True)
    (root / "config.yaml").write_text("gateway: {multiplex_profiles: true}\n", encoding="utf-8")
    (root / "gateway.pid").write_text(json.dumps({"pid": os.getpid(), "hermes_home": str(root)}), encoding="utf-8")
    (root / "gateway_state.json").write_text(json.dumps({
        "pid": os.getpid(), "hermes_home": str(root), "gateway_state": "running",
        "served_profiles": ["default", "alpha", "beta"],
        "platforms": {
            "api_server": {"state": "connected", "listener_base": "http://127.0.0.1:45719"},
            "webhook": {"state": "fatal", "error_code": "port_in_use"},
            "alpha:telegram": {"state": "connected"},
        }}), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.delenv("GATEWAY_MULTIPLEX_PROFILES", raising=False)
    import hermes_constants
    import gateway.status as status
    # Liveness is a verified identity; this pytest process passes as the default gateway only by
    # wearing a gateway command line.
    monkeypatch.setattr(status, "_read_process_cmdline", lambda pid: "hermes gateway run")
    monkeypatch.setattr(hermes_constants, "_default_hermes_root_memo", None)
    return root


def test_served_profile_projects_the_default_listener_mirrors_with_their_url(served_root):
    from gateway.status import profile_platforms_from_multiplexer, resolve_gateway_liveness
    alpha = served_root / "profiles" / "alpha"
    live = resolve_gateway_liveness(profile_dir=alpha, health_probe=None, use_cache=False)
    plats = profile_platforms_from_multiplexer(live.runtime, "alpha")
    # The mirror inherits the default's live state and points at the profile's own prefix.
    assert plats["api_server"]["state"] == "connected"
    assert plats["api_server"]["ingress_url"] == "http://127.0.0.1:45719/p/alpha/v1"
    # A dead default listener is dead for the profile too — never "connected" by fiat.
    assert "webhook" not in plats
    assert plats["telegram"] == {"state": "connected"}
    # The default profile keeps its own un-prefixed entries; nothing is mirrored onto it.
    assert "ingress_url" not in profile_platforms_from_multiplexer(live.runtime, "default").get("api_server", {})


def test_messaging_card_for_a_served_profile_reads_connected_not_restart_needed(served_root, monkeypatch):
    from hermes_cli.web_routers import messaging
    monkeypatch.setattr(messaging, "_platform_enablement", lambda *a, **k: (True, True, None))
    entry = {"id": "api_server", "name": "API server", "description": "", "docs_url": "", "env_vars": [],
             "required_env": []}
    alpha = served_root / "profiles" / "alpha"
    [payload] = messaging._platform_payloads(alpha, [entry])
    assert payload["gateway_running"] is True
    assert payload["state"] == "connected", payload
    assert payload["ingress_url"] == "http://127.0.0.1:45719/p/alpha/v1"


def test_mirror_lazy_import_survives_a_stale_pre_update_module_cache(served_root, monkeypatch):
    """KENSEI CUSTOM regression: cron delivery / status flows can run inside long-lived processes
    that hold a ``gateway.config`` module cached from before an editable-install update introduced
    the multiplex-era names. The lazy import must reload the module once instead of failing the
    caller (observed: manual cron runs failing with ``cannot import name
    'SHARED_LISTENER_MIRROR_PLATFORMS'`` while scheduled gateway deliveries passed)."""
    import gateway.config as gw_config

    from gateway.status import shared_listener_mirror_platforms

    for name in ("SHARED_LISTENER_MIRROR_PATHS", "SHARED_LISTENER_MIRROR_PLATFORMS"):
        monkeypatch.delattr(gw_config, name)  # exactly what a pre-update cache looks like

    runtime = {"platforms": {"api_server": {"state": "connected",
                                            "listener_base": "http://127.0.0.1:45719"}}}
    mirrored = shared_listener_mirror_platforms(runtime, "alpha")
    assert mirrored["api_server"]["ingress_url"] == "http://127.0.0.1:45719/p/alpha/v1"
def test_messaging_card_ignores_a_served_profiles_stale_own_runtime_record(served_root, monkeypatch):
    """A served profile writes no live ``gateway_state.json`` of its own, but one left behind by a
    pre-multiplex or standalone run (``stopped``, empty platforms) used to shadow the multiplexer's
    record: the fallback only ran when the file was missing, the bare-key lookup found nothing, and
    the card read "Restart needed" forever for a platform that was connected (#112765)."""
    from hermes_cli.web_routers import messaging
    monkeypatch.setattr(messaging, "_platform_enablement", lambda *a, **k: (True, True, None))
    entry = {"id": "telegram", "name": "Telegram", "description": "", "docs_url": "", "env_vars": [],
             "required_env": []}
    alpha = served_root / "profiles" / "alpha"
    (alpha / "gateway_state.json").write_text(json.dumps(
        {"gateway_state": "stopped", "platforms": {}}), encoding="utf-8")
    [payload] = messaging._platform_payloads(alpha, [entry])
    assert payload["gateway_running"] is True
    assert payload["state"] == "connected", payload
    # The shared record is scoped per profile: beta has no ``beta:telegram`` entry, so alpha's
    # connected verdict must not bleed into beta's card.
    beta = served_root / "profiles" / "beta"
    beta.mkdir()
    (beta / "gateway_state.json").write_text(json.dumps({"gateway_state": "stopped", "platforms": {}}), encoding="utf-8")
    [beta_payload] = messaging._platform_payloads(beta, [entry])
    assert beta_payload["state"] == "pending_restart", beta_payload


def test_messaging_card_keeps_a_live_own_gateway_record_over_the_multiplexer(served_root, monkeypatch):
    """Transitional dual-live case: a profile running its own standalone gateway while the live
    multiplexer still lists it in ``served_profiles``. ``resolve_gateway_liveness`` answers from the
    own record (rung 3) before the multiplexer (rung 4); the card must read the same record, or
    liveness and platform state come from two different gateways."""
    import gateway.status as status
    from hermes_cli.web_routers import messaging
    monkeypatch.setattr(messaging, "_platform_enablement", lambda *a, **k: (True, True, None))
    alpha = served_root / "profiles" / "alpha"
    # Two live gateways: this process is the default multiplexer; a second (fake, never signalled)
    # PID wears alpha's argv. Only the live guard sees a foreign PID, so existence is stubbed.
    own_pid = 2 ** 22 - 1
    (alpha / "gateway_state.json").write_text(json.dumps({
        "pid": own_pid, "hermes_home": str(alpha), "gateway_state": "running",
        "platforms": {"telegram": {"state": "retrying", "error_code": "network"}}}), encoding="utf-8")
    real_pid_exists = status._pid_exists
    monkeypatch.setattr(status, "_pid_exists", lambda pid: pid == own_pid or real_pid_exists(pid))
    monkeypatch.setattr(status, "_read_process_cmdline",
                        lambda pid: "hermes -p alpha gateway run" if pid == own_pid else "hermes gateway run")
    assert status.multiplexer_liveness_for_profile(alpha) is not None
    entry = {"id": "telegram", "name": "Telegram", "description": "", "docs_url": "", "env_vars": [],
             "required_env": []}
    [payload] = messaging._platform_payloads(alpha, [entry])
    assert payload["gateway_running"] is True
    assert payload["state"] == "retrying", payload
