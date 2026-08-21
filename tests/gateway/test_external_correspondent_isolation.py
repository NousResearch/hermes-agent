"""Regression coverage for per-correspondent gateway isolation."""

from gateway.config import Platform
from gateway.run import (
    GatewayRunner,
    _external_agent_isolation_kwargs,
    _is_external_correspondent,
)
from gateway.session import SessionSource


def _email_source(user_id: str) -> SessionSource:
    return SessionSource(
        platform=Platform.EMAIL,
        user_id=user_id,
        chat_id=user_id,
        chat_type="dm",
    )


def test_external_correspondent_match_is_case_insensitive():
    config = {
        "platforms": {
            "email": {
                "extra": {"external_correspondents": ["Boardy@Boardy.AI"]}
            }
        }
    }

    assert _is_external_correspondent(config, _email_source("boardy@boardy.ai"))


def test_external_correspondent_string_config_is_supported():
    config = {
        "platforms": {
            "email": {
                "extra": {
                    "external_correspondents": "first@example.com, second@example.com"
                }
            }
        }
    }

    assert _is_external_correspondent(config, _email_source("second@example.com"))


def test_isolation_kwargs_disable_private_context_only_for_correspondents():
    config = {
        "platforms": {
            "email": {"extra": {"external_correspondents": ["outside@example.com"]}}
        }
    }

    assert _external_agent_isolation_kwargs(
        config, _email_source("outside@example.com")
    ) == {
        "skip_memory": True,
        "skip_context_files": True,
        "load_soul_identity": False,
    }
    assert _external_agent_isolation_kwargs(
        config, _email_source("operator@example.com")
    ) == {
        "skip_memory": False,
        "skip_context_files": False,
        "load_soul_identity": True,
    }


def test_external_correspondent_gets_no_model_toolsets(monkeypatch):
    config = {
        "platforms": {
            "email": {"extra": {"external_correspondents": ["outside@example.com"]}}
        }
    }
    monkeypatch.setattr(
        "hermes_cli.tools_config._get_platform_tools",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("toolset resolution must be bypassed")
        ),
    )

    runner = GatewayRunner.__new__(GatewayRunner)
    assert runner._resolve_enabled_toolsets_for_source(
        config,
        _email_source("outside@example.com"),
        "email",
    ) == []
