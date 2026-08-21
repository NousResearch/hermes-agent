from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import cron.scheduler as sched


def _gateway_config(connected_values):
    config = MagicMock()
    config.get_connected_platforms.return_value = [
        MagicMock(value=value) for value in connected_values
    ]
    return config


def test_preflight_accepts_authorized_default_broker(monkeypatch):
    monkeypatch.setattr(
        sched, "_default_cron_broker_connected_platforms", lambda: {"telegram"}
    )
    with patch("gateway.config.load_gateway_config", return_value=_gateway_config(set())):
        assert sched._preflight_check_delivery({"deliver": "telegram"}) is None


def test_preflight_rejects_platform_outside_broker(monkeypatch):
    monkeypatch.setattr(
        sched, "_default_cron_broker_connected_platforms", lambda: {"telegram"}
    )
    with patch("gateway.config.load_gateway_config", return_value=_gateway_config(set())):
        reason = sched._preflight_check_delivery({"deliver": "discord:123"})
    assert reason is not None
    assert "no authorized default-profile outbound broker" in reason


def test_bare_platform_can_resolve_default_broker_home(monkeypatch):
    monkeypatch.setattr(sched, "_get_home_target_chat_id", lambda _name: "")
    monkeypatch.setattr(sched, "_get_home_target_thread_id", lambda _name: None)
    monkeypatch.setattr(
        sched,
        "_default_cron_broker_home_target",
        lambda name: ("1234567890", None) if name == "telegram" else ("", None),
    )
    target = sched._resolve_single_delivery_target({}, "telegram")
    assert target == {
        "platform": "telegram",
        "chat_id": "1234567890",
        "thread_id": None,
    }


def test_broker_send_scopes_child_to_default_home(monkeypatch, tmp_path):
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "specialist-must-not-leak")
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "specialist-must-not-leak")
    monkeypatch.setattr(
        sched,
        "_default_cron_broker_policy",
        lambda: (True, {"telegram", "discord"}, default_home),
    )
    completed = MagicMock(returncode=0, stdout="", stderr="")
    monkeypatch.setattr(sched.subprocess, "run", MagicMock(return_value=completed))

    error = sched._send_via_default_cron_broker(
        "telegram",
        "1234567890",
        "hello",
        media_files=["/tmp/chart.png"],
    )

    assert error is None
    call = sched.subprocess.run.call_args
    assert call.kwargs["env"]["HERMES_HOME"] == str(default_home)
    assert call.kwargs["creationflags"] == sched.windows_hide_flags()
    assert "TELEGRAM_BOT_TOKEN" not in call.kwargs["env"]
    assert "DISCORD_BOT_TOKEN" not in call.kwargs["env"]
    assert call.kwargs["input"] == "hello\nMEDIA:/tmp/chart.png"
    assert call.args[0][1:4] == ["-p", "default", "send"]
    assert call.args[0][-6:] == [
        "send",
        "--to",
        "telegram:1234567890",
        "--file",
        "-",
        "--quiet",
    ]


def _set_profile_homes(monkeypatch, tmp_path, profile="growth"):
    default_home = tmp_path / "fleet-root"
    specialist_home = default_home / "profiles" / profile
    specialist_home.mkdir(parents=True)
    monkeypatch.setattr(sched, "get_hermes_home", lambda: specialist_home)
    monkeypatch.setattr(sched, "get_default_hermes_root", lambda: default_home)
    return default_home, specialist_home


def test_policy_reads_authority_from_default_profile(monkeypatch, tmp_path):
    default_home, _specialist_home = _set_profile_homes(monkeypatch, tmp_path)
    observed = {}

    def fake_load_config():
        from hermes_constants import get_hermes_home

        observed["home"] = get_hermes_home()
        return {
            "gateway": {"multiplex_profiles": True},
            "cron": {
                "broker_outbound_via_default": True,
                "broker_outbound_platforms": '["telegram", "discord"]',
            },
        }

    monkeypatch.setattr(sched, "load_config", fake_load_config)
    enabled, allowed, resolved_default = sched._default_cron_broker_policy()

    assert enabled is True
    assert allowed == {"telegram", "discord"}
    assert resolved_default == default_home
    assert observed["home"] == default_home


def test_default_profile_never_brokers_through_itself(monkeypatch, tmp_path):
    default_home = tmp_path / "fleet-root"
    default_home.mkdir()
    monkeypatch.setattr(sched, "get_hermes_home", lambda: default_home)
    monkeypatch.setattr(sched, "get_default_hermes_root", lambda: default_home)
    load = MagicMock()
    monkeypatch.setattr(sched, "load_config", load)

    assert sched._default_cron_broker_policy() == (False, set(), default_home)
    load.assert_not_called()


def test_policy_real_config_load_from_named_profile_context(monkeypatch, tmp_path):
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    default_home = tmp_path / "fleet-root"
    specialist_home = default_home / "profiles" / "growth"
    specialist_home.mkdir(parents=True)
    (default_home / "config.yaml").write_text(
        "gateway:\n"
        "  multiplex_profiles: true\n"
        "cron:\n"
        "  broker_outbound_via_default: true\n"
        "  broker_outbound_platforms:\n"
        "    - telegram\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(default_home))

    token = set_hermes_home_override(specialist_home)
    try:
        enabled, allowed, resolved_default = sched._default_cron_broker_policy()
    finally:
        reset_hermes_home_override(token)

    assert enabled is True
    assert allowed == {"telegram"}
    assert resolved_default == default_home


@pytest.mark.parametrize(
    "gateway_cfg,cron_cfg",
    [
        ({"multiplex_profiles": False}, {"broker_outbound_via_default": True, "broker_outbound_platforms": ["telegram"]}),
        ({"multiplex_profiles": True}, {"broker_outbound_via_default": False, "broker_outbound_platforms": ["telegram"]}),
        ({"multiplex_profiles": True}, {"broker_outbound_via_default": True, "broker_outbound_platforms": []}),
        ({"multiplex_profiles": True}, {"broker_outbound_via_default": True, "broker_outbound_platforms": {"telegram": True}}),
    ],
)
def test_policy_fails_closed_without_all_required_authority(
    monkeypatch, tmp_path, gateway_cfg, cron_cfg
):
    default_home, _specialist_home = _set_profile_homes(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sched,
        "load_config",
        lambda: {"gateway": gateway_cfg, "cron": cron_cfg},
    )

    enabled, allowed, resolved_default = sched._default_cron_broker_policy()
    assert enabled is False
    assert allowed == set()
    assert resolved_default == default_home


def test_policy_normalizes_comma_separated_allowlist(monkeypatch, tmp_path):
    _default_home, _specialist_home = _set_profile_homes(monkeypatch, tmp_path)
    monkeypatch.setattr(
        sched,
        "load_config",
        lambda: {
            "gateway": {"multiplex_profiles": True},
            "cron": {
                "broker_outbound_via_default": True,
                "broker_outbound_platforms": " Telegram, DISCORD , ",
            },
        },
    )

    enabled, allowed, _ = sched._default_cron_broker_policy()
    assert enabled is True
    assert allowed == {"telegram", "discord"}


def test_connected_platforms_are_intersection_of_default_and_allowlist(monkeypatch, tmp_path):
    default_home = tmp_path / "fleet-root"
    default_home.mkdir()
    monkeypatch.setattr(
        sched,
        "_default_cron_broker_policy",
        lambda: (True, {"telegram", "discord"}, default_home),
    )
    with patch(
        "gateway.config.load_gateway_config",
        return_value=_gateway_config({"telegram", "slack"}),
    ), patch(
        "tools.send_message_tool.supports_standalone_send",
        side_effect=lambda name: name == "telegram",
    ):
        assert sched._default_cron_broker_connected_platforms() == {"telegram"}


def test_connected_live_only_platform_is_not_broker_authorized(monkeypatch, tmp_path):
    default_home = tmp_path / "fleet-root"
    default_home.mkdir()
    monkeypatch.setattr(
        sched,
        "_default_cron_broker_policy",
        lambda: (True, {"yuanbao"}, default_home),
    )
    with patch(
        "gateway.config.load_gateway_config",
        return_value=_gateway_config({"yuanbao"}),
    ), patch(
        "tools.send_message_tool.supports_standalone_send",
        return_value=False,
    ):
        assert sched._default_cron_broker_connected_platforms() == set()


def test_home_target_is_read_under_default_profile(monkeypatch, tmp_path):
    default_home = tmp_path / "fleet-root"
    default_home.mkdir()
    observed = {}
    monkeypatch.setattr(
        sched,
        "_default_cron_broker_policy",
        lambda: (True, {"telegram"}, default_home),
    )

    def fake_home(_platform):
        from hermes_constants import get_hermes_home

        observed["home"] = get_hermes_home()
        return SimpleNamespace(chat_id="123", thread_id="17")

    monkeypatch.setattr(sched, "_get_config_home_channel", fake_home)
    assert sched._default_cron_broker_home_target("telegram") == ("123", "17")
    assert observed["home"] == default_home


def test_broker_refuses_platform_not_in_allowlist(monkeypatch, tmp_path):
    default_home = tmp_path / "fleet-root"
    default_home.mkdir()
    monkeypatch.setattr(
        sched,
        "_default_cron_broker_policy",
        lambda: (True, {"telegram"}, default_home),
    )
    run = MagicMock()
    monkeypatch.setattr(sched.subprocess, "run", run)

    error = sched._send_via_default_cron_broker("discord", "123", "hello")

    assert "not enabled" in error
    run.assert_not_called()


def test_broker_fails_closed_if_sanitized_env_builder_fails(monkeypatch, tmp_path):
    default_home = tmp_path / "fleet-root"
    default_home.mkdir()
    monkeypatch.setattr(
        sched,
        "_default_cron_broker_policy",
        lambda: (True, {"telegram"}, default_home),
    )
    run = MagicMock()
    monkeypatch.setattr(sched.subprocess, "run", run)

    with patch(
        "tools.environments.local.build_subprocess_env",
        side_effect=RuntimeError("sanitizer unavailable"),
    ):
        error = sched._send_via_default_cron_broker("telegram", "123", "hello")

    assert "sanitized child environment" in error
    assert "RuntimeError" in error
    run.assert_not_called()


def test_preflight_accepts_explicit_target_on_authorized_broker(monkeypatch):
    monkeypatch.setattr(
        sched, "_default_cron_broker_connected_platforms", lambda: {"telegram"}
    )
    with patch("gateway.config.load_gateway_config", return_value=_gateway_config(set())):
        assert sched._preflight_check_delivery({"deliver": "telegram:123"}) is None


def test_delivery_targets_include_brokered_platform_and_default_home(monkeypatch):
    monkeypatch.setattr(sched, "_iter_home_target_platforms", lambda: ["telegram", "discord"])
    monkeypatch.setattr(sched, "_is_known_delivery_platform", lambda name: True)
    monkeypatch.setattr(sched, "_resolve_home_env_var", lambda name: f"{name.upper()}_HOME")
    monkeypatch.setattr(sched, "_get_home_target_chat_id", lambda _name: "")
    monkeypatch.setattr(
        sched, "_default_cron_broker_connected_platforms", lambda: {"telegram"}
    )
    monkeypatch.setattr(
        sched,
        "_default_cron_broker_home_target",
        lambda name: ("123", None) if name == "telegram" else ("", None),
    )

    with patch("gateway.config.load_gateway_config", return_value=_gateway_config(set())):
        platform_targets = [
            target
            for target in sched.cron_delivery_targets()
            if not target["id"].startswith("bot-chat:")
        ]
        assert platform_targets == [
            {
                "id": "telegram",
                "name": "Telegram",
                "home_target_set": True,
                "home_env_var": "TELEGRAM_HOME",
            }
        ]


def _delivery_config(local_platforms):
    from gateway.config import Platform, PlatformConfig

    config = MagicMock()
    config.get_connected_platforms.return_value = [
        MagicMock(value=name) for name in local_platforms
    ]
    config.platforms = {
        Platform(name): PlatformConfig(enabled=True)
        for name in local_platforms
    }
    return config


def test_local_specialist_transport_wins_over_broker(monkeypatch):
    config = _delivery_config({"telegram"})
    broker_send = MagicMock(return_value=None)
    standalone_send = AsyncMock(return_value={"success": True})
    monkeypatch.setattr(
        sched,
        "_resolve_delivery_targets",
        lambda _job: [{"platform": "telegram", "chat_id": "123", "thread_id": None}],
    )
    monkeypatch.setattr(
        sched, "_default_cron_broker_connected_platforms", lambda: {"telegram"}
    )
    monkeypatch.setattr(sched, "_send_via_default_cron_broker", broker_send)

    with patch("gateway.config.load_gateway_config", return_value=config), patch(
        "cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}
    ), patch("tools.send_message_tool._send_to_platform", new=standalone_send):
        error = sched._deliver_result({"id": "job", "deliver": "telegram"}, "hello")

    assert error is None
    standalone_send.assert_awaited_once()
    broker_send.assert_not_called()


def test_broker_failure_propagates_as_delivery_error(monkeypatch):
    config = _delivery_config(set())
    monkeypatch.setattr(
        sched,
        "_resolve_delivery_targets",
        lambda _job: [{"platform": "telegram", "chat_id": "123", "thread_id": None}],
    )
    monkeypatch.setattr(
        sched, "_default_cron_broker_connected_platforms", lambda: {"telegram"}
    )
    monkeypatch.setattr(
        sched,
        "_send_via_default_cron_broker",
        lambda *_args, **_kwargs: "simulated broker outage",
    )

    with patch("gateway.config.load_gateway_config", return_value=config), patch(
        "cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}
    ):
        error = sched._deliver_result({"id": "job", "deliver": "telegram"}, "hello")

    assert error is not None
    assert "brokered delivery to telegram:123 failed" in error
    assert "simulated broker outage" in error


def test_mixed_local_and_brokered_targets_each_send_once(monkeypatch):
    config = _delivery_config({"discord"})
    broker_send = MagicMock(return_value=None)
    standalone_send = AsyncMock(return_value={"success": True})
    monkeypatch.setattr(
        sched,
        "_resolve_delivery_targets",
        lambda _job: [
            {"platform": "telegram", "chat_id": "111", "thread_id": None},
            {"platform": "discord", "chat_id": "222", "thread_id": None},
        ],
    )
    monkeypatch.setattr(
        sched, "_default_cron_broker_connected_platforms", lambda: {"telegram"}
    )
    monkeypatch.setattr(sched, "_send_via_default_cron_broker", broker_send)

    with patch("gateway.config.load_gateway_config", return_value=config), patch(
        "cron.scheduler.load_config", return_value={"cron": {"wrap_response": False}}
    ), patch("tools.send_message_tool._send_to_platform", new=standalone_send):
        error = sched._deliver_result(
            {"id": "job", "deliver": "telegram:111,discord:222"}, "hello"
        )

    assert error is None
    broker_send.assert_called_once()
    assert broker_send.call_args.args[:2] == ("telegram", "111")
    standalone_send.assert_awaited_once()
