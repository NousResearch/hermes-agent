from __future__ import annotations

import yaml

from gateway.config import GatewayConfig, Platform, PlatformConfig, load_gateway_config
from gateway.run import (
    _gateway_effective_allow_silent_response,
    _gateway_effective_suppress_provider_diagnostics,
    _normalize_empty_agent_response,
    _prepare_gateway_status_message,
    _sanitize_gateway_final_response,
)


def _codex_incomplete_result() -> dict:
    return {
        "final_response": None,
        "api_calls": 3,
        "completed": False,
        "partial": True,
        "error": "Codex response remained incomplete after 3 continuation attempts",
    }


def test_default_partial_empty_response_still_surfaces_warning():
    response = _normalize_empty_agent_response(
        _codex_incomplete_result(),
        "",
        history_len=5,
    )

    assert "Processing stopped" in response
    assert "Codex response remained incomplete" in response


def test_allow_silent_response_keeps_intentional_no_reply_empty():
    response = _normalize_empty_agent_response(
        {"final_response": None, "api_calls": 1, "completed": True},
        "",
        history_len=5,
        allow_silent_response=True,
    )

    assert response == ""


def test_allow_silent_response_does_not_hide_partial_without_diagnostic_suppression():
    response = _normalize_empty_agent_response(
        _codex_incomplete_result(),
        "",
        history_len=5,
        allow_silent_response=True,
        suppress_provider_diagnostics=False,
    )

    assert "Processing stopped" in response
    assert "Codex response remained incomplete" in response


def test_suppress_provider_diagnostics_hides_codex_incomplete_warning():
    response = _normalize_empty_agent_response(
        _codex_incomplete_result(),
        "",
        history_len=5,
        allow_silent_response=True,
        suppress_provider_diagnostics=True,
    )

    assert response == ""


def test_suppress_provider_diagnostics_filters_final_and_status_messages():
    diagnostic = "⚠️ Processing stopped: Codex response remained incomplete after 3 continuation attempts. Try again."

    assert (
        _sanitize_gateway_final_response(
            Platform.WHATSAPP,
            diagnostic,
            suppress_provider_diagnostics=True,
        )
        == ""
    )
    assert (
        _prepare_gateway_status_message(
            Platform.WHATSAPP,
            "model.retry",
            "⚠️ Empty/malformed response — switching to fallback...",
            suppress_provider_diagnostics=True,
        )
        is None
    )


def test_effective_policy_reads_global_and_platform_specific_flags():
    global_config = GatewayConfig(allow_silent_response=True)
    assert _gateway_effective_allow_silent_response(global_config, Platform.WHATSAPP)

    platform_config = GatewayConfig(
        platforms={
            Platform.WHATSAPP: PlatformConfig(
                extra={
                    "allow_silent_response": True,
                    "suppress_diagnostics": True,
                }
            )
        }
    )
    assert _gateway_effective_allow_silent_response(platform_config, Platform.WHATSAPP)
    assert _gateway_effective_suppress_provider_diagnostics(platform_config, Platform.WHATSAPP)


def test_whatsapp_defaults_to_silent_no_diagnostics_without_config_flags():
    config = GatewayConfig()

    assert _gateway_effective_allow_silent_response(config, Platform.WHATSAPP)
    assert _gateway_effective_suppress_provider_diagnostics(config, Platform.WHATSAPP)
    assert not _gateway_effective_allow_silent_response(config, Platform.TELEGRAM)
    assert not _gateway_effective_suppress_provider_diagnostics(config, Platform.TELEGRAM)


def test_whatsapp_can_explicitly_opt_out_of_silent_no_diagnostics_defaults():
    config = GatewayConfig(
        platforms={
            Platform.WHATSAPP: PlatformConfig(
                extra={
                    "allow_silent_response": False,
                    "suppress_diagnostics": False,
                }
            )
        }
    )

    assert not _gateway_effective_allow_silent_response(config, Platform.WHATSAPP)
    assert not _gateway_effective_suppress_provider_diagnostics(config, Platform.WHATSAPP)


def test_config_yaml_loads_gateway_and_whatsapp_silent_diagnostics(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    for name in (
        "WHATSAPP_REQUIRE_MENTION",
        "WHATSAPP_MENTION_PATTERNS",
        "WHATSAPP_FREE_RESPONSE_CHATS",
        "WHATSAPP_DM_POLICY",
        "WHATSAPP_ALLOWED_USERS",
        "WHATSAPP_GROUP_POLICY",
        "WHATSAPP_GROUP_ALLOWED_USERS",
    ):
        monkeypatch.delenv(name, raising=False)

    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "gateway": {
                    "allow_silent_response": True,
                    "suppress_provider_diagnostics_in_chat": True,
                },
                "whatsapp": {
                    "allow_silent_response": True,
                    "suppress_diagnostics": True,
                },
            }
        ),
        encoding="utf-8",
    )

    config = load_gateway_config()

    assert config.allow_silent_response is True
    assert config.suppress_provider_diagnostics_in_chat is True
    assert config.platforms[Platform.WHATSAPP].extra["allow_silent_response"] is True
    assert config.platforms[Platform.WHATSAPP].extra["suppress_diagnostics"] is True
