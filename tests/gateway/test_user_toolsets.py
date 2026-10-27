"""Per-user gateway toolset overrides from top-level ``user_toolsets``."""

from types import SimpleNamespace
from typing import Any

from gateway.run import GatewayRunner
from hermes_cli.tools_config import _get_platform_tools


BASE_CONFIG = {
    "platform_toolsets": {
        "telegram": ["web"],
        "whatsapp": ["web"],
    },
}


def _runner(adapter=None):
    runner = object.__new__(GatewayRunner)
    runner._delivery_adapter_for = lambda source: adapter
    return runner


def _source(user_id, user_id_alt=None) -> Any:
    return SimpleNamespace(user_id=user_id, user_id_alt=user_id_alt)


def _expected(config, platform, override):
    effective = dict(config)
    platform_toolsets = dict(config.get("platform_toolsets") or {})
    platform_toolsets[platform] = override
    effective["platform_toolsets"] = platform_toolsets
    return sorted(_get_platform_tools(effective, platform))


def _resolve(config, source, platform):
    return GatewayRunner._resolve_enabled_toolsets_for_source(
        _runner(), config, source, platform
    )


def test_telegram_user_toolset_replaces_platform_toolset_with_numeric_yaml_key():
    config = {
        **BASE_CONFIG,
        "user_toolsets": {"telegram": {123456789: ["terminal"]}},
    }

    result = _resolve(config, _source("123456789"), "telegram")

    assert result == _expected(config, "telegram", ["terminal"])
    assert result != sorted(_get_platform_tools(config, "telegram"))


def test_missing_user_falls_back_to_platform_toolset():
    config = {
        **BASE_CONFIG,
        "user_toolsets": {"telegram": {"123456789": ["terminal"]}},
    }

    result = _resolve(config, _source("987654321"), "telegram")

    assert result == sorted(_get_platform_tools(config, "telegram"))


def test_user_toolsets_apply_to_any_platform_with_a_user_id():
    config = {
        "platform_toolsets": {"signal": ["web"]},
        "user_toolsets": {"signal": {"user-42": ["terminal"]}},
    }

    result = _resolve(config, _source("user-42"), "signal")

    assert result == _expected(config, "signal", ["terminal"])


def test_user_id_alt_matches_on_any_platform():
    config = {
        "platform_toolsets": {"signal": ["web"]},
        "user_toolsets": {"signal": {"uuid-42": ["terminal"]}},
    }

    result = _resolve(config, _source(None, "uuid-42"), "signal")

    assert result == _expected(config, "signal", ["terminal"])


def test_user_id_takes_priority_over_user_id_alt():
    config = {
        "platform_toolsets": {"signal": ["web"]},
        "user_toolsets": {
            "signal": {
                "uuid-42": ["file"],
                "+49123": ["terminal"],
            },
        },
    }

    result = _resolve(config, _source("+49123", "uuid-42"), "signal")

    assert result == _expected(config, "signal", ["terminal"])


def test_source_without_user_id_falls_back_to_platform_toolset():
    config = {
        "platform_toolsets": {"signal": ["web"]},
        "user_toolsets": {"signal": {"None": ["terminal"]}},
    }

    result = _resolve(config, _source(None), "signal")

    assert result == sorted(_get_platform_tools(config, "signal"))


def test_explicit_empty_user_toolset_is_authoritative():
    config = {
        **BASE_CONFIG,
        "user_toolsets": {"telegram": {"123456789": []}},
    }

    result = _resolve(config, _source("123456789"), "telegram")

    assert result == _expected(config, "telegram", [])
    assert result != sorted(_get_platform_tools(config, "telegram"))


def test_whatsapp_phone_key_matches_device_jid():
    config = {
        **BASE_CONFIG,
        "user_toolsets": {"whatsapp": {"491707127232": ["terminal"]}},
    }

    result = _resolve(
        config, _source("491707127232:12@s.whatsapp.net"), "whatsapp"
    )

    assert result == _expected(config, "whatsapp", ["terminal"])


def test_whatsapp_known_lid_alias_matches_phone_key(monkeypatch):
    config = {
        **BASE_CONFIG,
        "user_toolsets": {"whatsapp": {"491707127232": ["terminal"]}},
    }

    def canonical(value):
        return "491707127232" if str(value).split("@", 1)[0] in {
            "491707127232", "987654321"
        } else str(value)

    monkeypatch.setattr(
        "gateway.user_toolsets.canonical_whatsapp_identifier", canonical
    )

    result = _resolve(config, _source("987654321@lid"), "whatsapp")

    assert result == _expected(config, "whatsapp", ["terminal"])


def test_whatsapp_unknown_lid_does_not_match_privileged_phone():
    config = {
        **BASE_CONFIG,
        "user_toolsets": {"whatsapp": {"491707127232": ["terminal"]}},
    }

    result = _resolve(config, _source("987654321@lid"), "whatsapp")

    assert result == sorted(_get_platform_tools(config, "whatsapp"))


def test_whatsapp_configured_ids_are_not_platform_normalized():
    config = {
        **BASE_CONFIG,
        "user_toolsets": {
            "whatsapp": {"491707127232@lid": ["terminal"]}
        },
    }

    result = _resolve(
        config, _source("491707127232@s.whatsapp.net"), "whatsapp"
    )

    assert result == sorted(_get_platform_tools(config, "whatsapp"))


def test_global_disabled_toolsets_still_apply_after_user_override():
    config = {
        **BASE_CONFIG,
        "agent": {"disabled_toolsets": ["terminal"]},
        "user_toolsets": {
            "telegram": {"123456789": ["terminal", "web"]}
        },
    }

    result = _resolve(config, _source("123456789"), "telegram")

    assert result == _expected(config, "telegram", ["terminal", "web"])
    assert "terminal" not in result


def test_adapter_override_keeps_priority_over_user_toolset():
    config = {
        **BASE_CONFIG,
        "user_toolsets": {"telegram": {"123456789": ["terminal"]}},
    }
    adapter = SimpleNamespace(toolsets_for_source=lambda source: ["file"])

    result = GatewayRunner._resolve_enabled_toolsets_for_source(
        _runner(adapter), config, _source("123456789"), "telegram"
    )

    assert result == _expected(config, "telegram", ["file"])


def test_resolution_does_not_mutate_config():
    config = {
        "platform_toolsets": {"telegram": ["web"]},
        "user_toolsets": {"telegram": {"123456789": ["terminal"]}},
    }

    _resolve(config, _source("123456789"), "telegram")

    assert config == {
        "platform_toolsets": {"telegram": ["web"]},
        "user_toolsets": {"telegram": {"123456789": ["terminal"]}},
    }
