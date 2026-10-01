from __future__ import annotations

from types import SimpleNamespace

from hermes_cli import profile_cmd


def test_profile_gateway_scope_prefers_user_systemd_unit(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["systemctl", "--user"]:
            return SimpleNamespace(stdout="hermes-gateway-demo.service loaded active running\n")
        return SimpleNamespace(stdout="")

    monkeypatch.setattr(profile_cmd.sys, "platform", "linux")
    monkeypatch.setattr(profile_cmd.subprocess, "run", fake_run)

    scope = profile_cmd._profile_gateway_scope(SimpleNamespace(is_default=False, name="demo"))

    assert scope == "user"
    assert calls == [[
        "systemctl", "--user", "list-units", "--all", "hermes-gateway-demo.service",
        "--no-legend", "--plain", "--no-pager",
    ]]


def test_profile_gateway_scope_falls_back_to_system_unit(monkeypatch):
    def fake_run(argv, **kwargs):
        if argv[:2] == ["systemctl", "--user"]:
            return SimpleNamespace(stdout="")
        return SimpleNamespace(stdout="hermes-gateway.service loaded inactive dead\n")

    monkeypatch.setattr(profile_cmd.sys, "platform", "linux")
    monkeypatch.setattr(profile_cmd.subprocess, "run", fake_run)

    assert profile_cmd._profile_gateway_scope(SimpleNamespace(is_default=True, name="default")) == "system"


def test_profile_gateway_scope_is_not_applicable_off_linux(monkeypatch):
    monkeypatch.setattr(profile_cmd.sys, "platform", "darwin")
    assert profile_cmd._profile_gateway_scope(SimpleNamespace(is_default=True, name="default")) == "—"
