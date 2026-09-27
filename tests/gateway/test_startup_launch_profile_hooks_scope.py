"""Multiplex invariant: the launch profile's ``hooks:`` block registers at gateway startup.

Startup runs before any turn scope exists. With ``multiplex_profiles`` on, ``get_secret`` fails
closed outside a scope, so the unscoped launch-profile registration raised on the first
``hooks.outbound[].secret_env`` target and ``_register_config_hooks`` swallowed it at DEBUG level —
every launch-profile outbound webhook was silently dropped. Regression for the #108319 follow-up.
"""
from __future__ import annotations

import logging

from agent import secret_scope
from gateway.run_startup import GatewayStartupMixin


def _capture_registration(monkeypatch, hooks_cfg):
    seen: dict = {}
    import agent.outbound_webhooks as ow
    import agent.shell_hooks as sh
    import hermes_cli.config as cfgmod

    def fake_register_outbound(cfg):
        seen["targets"] = ow.iter_configured_targets(cfg)
        seen["scope"] = secret_scope.current_secret_scope()

    monkeypatch.setattr(cfgmod, "load_config", lambda: hooks_cfg)
    monkeypatch.setattr(sh, "register_from_config", lambda cfg, accept_hooks=False: None)
    monkeypatch.setattr(ow, "register_from_config", fake_register_outbound)
    return seen


def test_launch_profile_webhooks_register_signed_under_multiplex(tmp_path, monkeypatch, caplog):
    launch_home = tmp_path / ".hermes"
    launch_home.mkdir()
    (launch_home / ".env").write_text("MY_HOOK_SECRET=s3cret-of-launch\n")
    monkeypatch.setenv("HERMES_HOME", str(launch_home))
    monkeypatch.delenv("MY_HOOK_SECRET", raising=False)
    cfg = {"hooks": {"outbound": [
        {"url": "https://hooks.example/signed", "events": ["on_session_end"], "secret_env": "MY_HOOK_SECRET"},
        {"url": "https://hooks.example/plain", "events": ["on_session_end"]},
    ]}}
    seen = _capture_registration(monkeypatch, cfg)

    secret_scope.set_multiplex_active(True)
    try:
        assert secret_scope.current_secret_scope() is None  # startup: no turn scope yet
        with caplog.at_level(logging.WARNING, logger="gateway.run_startup"):
            GatewayStartupMixin._register_launch_profile_config_hooks()
    finally:
        secret_scope.set_multiplex_active(False)

    assert seen.get("scope") is not None, "registration must run inside the launch profile's scope"
    assert [t.url for t in seen["targets"]] == ["https://hooks.example/signed", "https://hooks.example/plain"]
    assert seen["targets"][0].secret == "s3cret-of-launch"
    assert "registration failed" not in caplog.text


def test_registration_failure_is_a_warning_not_debug(monkeypatch, caplog):
    """A dropped hook block must be visible: the swallow used to log at DEBUG."""
    import hermes_cli.config as cfgmod

    def boom():
        raise RuntimeError("config exploded")

    monkeypatch.setattr(cfgmod, "load_config", boom)
    with caplog.at_level(logging.WARNING, logger="gateway.run_startup"):
        GatewayStartupMixin._register_launch_profile_config_hooks()
    assert any(r.levelno == logging.WARNING and "registration failed" in r.getMessage() for r in caplog.records)
