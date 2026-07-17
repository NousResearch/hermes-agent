"""Non-runtime API/webhook port-binding config parity tests.

These tests use temporary HERMES_HOME directories, dummy config values, and
monkeypatched environment lookups only. They must not start listeners, make
network/webhook/API calls, read a real profile config, or touch credentials.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import yaml

from gateway.config import Platform, PlatformConfig
from hermes_constants import reset_hermes_home_override, set_hermes_home_override


_API_SERVER_SHAPES = pytest.mark.parametrize(
    ("shape_name", "yaml_config"),
    [
        (
            "top_level_api_server",
            {
                "api_server": {
                    "enabled": True,
                    "extra": {
                        "host": "127.0.0.1",
                        "port": 18642,
                        "key": "dummy-api-key-for-test-only",
                        "cors_origins": ["https://client.example.invalid"],
                    },
                }
            },
        ),
        (
            "platforms_api_server",
            {
                "platforms": {
                    "api_server": {
                        "enabled": True,
                        "extra": {
                            "host": "127.0.0.1",
                            "port": 28642,
                            "key": "dummy-api-key-for-test-only",
                            "cors_origins": ["https://client.example.invalid"],
                        },
                    }
                }
            },
        ),
        (
            "gateway_platforms_api_server",
            {
                "gateway": {
                    "platforms": {
                        "api_server": {
                            "enabled": True,
                            "extra": {
                                "host": "127.0.0.1",
                                "port": 38642,
                                "key": "dummy-api-key-for-test-only",
                                "cors_origins": ["https://client.example.invalid"],
                            },
                        }
                    }
                }
            },
        ),
    ],
)

_WEBHOOK_SHAPES = pytest.mark.parametrize(
    ("shape_name", "yaml_config"),
    [
        (
            "top_level_webhook",
            {
                "webhook": {
                    "enabled": True,
                    "extra": {
                        "host": "127.0.0.1",
                        "port": 18644,
                        "secret": "dummy-webhook-secret-for-test-only",
                        "routes": {
                            "ci": {
                                "secret": "dummy-route-secret-for-test-only",
                                "prompt": "summarize {event}",
                                "deliver": "log",
                            }
                        },
                    },
                }
            },
        ),
        (
            "platforms_webhook",
            {
                "platforms": {
                    "webhook": {
                        "enabled": True,
                        "extra": {
                            "host": "127.0.0.1",
                            "port": 28644,
                            "secret": "dummy-webhook-secret-for-test-only",
                            "routes": {
                                "ci": {
                                    "secret": "dummy-route-secret-for-test-only",
                                    "prompt": "summarize {event}",
                                    "deliver": "log",
                                }
                            },
                        },
                    }
                }
            },
        ),
        (
            "gateway_platforms_webhook",
            {
                "gateway": {
                    "platforms": {
                        "webhook": {
                            "enabled": True,
                            "extra": {
                                "host": "127.0.0.1",
                                "port": 38644,
                                "secret": "dummy-webhook-secret-for-test-only",
                                "routes": {
                                    "ci": {
                                        "secret": "dummy-route-secret-for-test-only",
                                        "prompt": "summarize {event}",
                                        "deliver": "log",
                                    }
                                },
                            },
                        }
                    }
                }
            },
        ),
    ],
)


def _load_config_from_temp_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    yaml_config: Mapping[str, Any],
    *,
    dummy_env: Mapping[str, str] | None = None,
):
    """Load gateway config from a temp home with an isolated dummy env view."""
    from gateway import config as gateway_config
    from hermes_cli import managed_scope

    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    token = set_hermes_home_override(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(managed_scope, "apply_managed_overlay", lambda cfg: cfg)

    env_values = dict(dummy_env or {})

    def fake_getenv(name: str, default: str | None = None) -> str | None:
        return env_values.get(name, default)

    monkeypatch.setattr(gateway_config.os, "getenv", fake_getenv)
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump(dict(yaml_config), sort_keys=True),
        encoding="utf-8",
    )

    try:
        return gateway_config.load_gateway_config()
    finally:
        reset_hermes_home_override(token)


@_API_SERVER_SHAPES
def test_api_server_port_binding_config_shapes_are_loaded_without_listener_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shape_name: str,
    yaml_config: Mapping[str, Any],
) -> None:
    config = _load_config_from_temp_home(tmp_path, monkeypatch, yaml_config)

    platform_config = config.platforms[Platform.API_SERVER]

    assert shape_name
    assert platform_config.enabled is True
    assert platform_config.extra["host"] == "127.0.0.1"
    assert platform_config.extra["port"] in {18642, 28642, 38642}
    assert platform_config.extra["key"] == "dummy-api-key-for-test-only"
    assert platform_config.extra["cors_origins"] == [
        "https://client.example.invalid"
    ]


@_WEBHOOK_SHAPES
def test_webhook_port_binding_config_shapes_are_loaded_without_listener_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    shape_name: str,
    yaml_config: Mapping[str, Any],
) -> None:
    config = _load_config_from_temp_home(tmp_path, monkeypatch, yaml_config)

    platform_config = config.platforms[Platform.WEBHOOK]

    assert shape_name
    assert platform_config.enabled is True
    assert platform_config.extra["host"] == "127.0.0.1"
    assert platform_config.extra["port"] in {18644, 28644, 38644}
    assert platform_config.extra["secret"] == "dummy-webhook-secret-for-test-only"
    assert platform_config.extra["routes"]["ci"]["secret"] == (
        "dummy-route-secret-for-test-only"
    )


def test_top_level_platform_blocks_overlay_nested_api_and_webhook_port_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load_config_from_temp_home(
        tmp_path,
        monkeypatch,
        {
            "gateway": {
                "platforms": {
                    "api_server": {"enabled": True, "extra": {"port": 18642}},
                    "webhook": {"enabled": True, "extra": {"port": 18644}},
                }
            },
            "platforms": {
                "api_server": {"extra": {"port": 28642}},
                "webhook": {"extra": {"port": 28644}},
            },
            "api_server": {"extra": {"port": 38642}},
            "webhook": {"extra": {"port": 38644}},
        },
    )

    assert config.platforms[Platform.API_SERVER].extra["port"] == 38642
    assert config.platforms[Platform.WEBHOOK].extra["port"] == 38644


def test_dummy_env_overrides_api_server_and_webhook_ports_without_real_env_reads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _load_config_from_temp_home(
        tmp_path,
        monkeypatch,
        {
            "platforms": {
                "api_server": {"enabled": False, "extra": {"port": 18642}},
                "webhook": {"enabled": False, "extra": {"port": 18644}},
            }
        },
        dummy_env={
            "API_SERVER_ENABLED": "true",
            "API_SERVER_PORT": "28642",
            "API_SERVER_HOST": "127.0.0.1",
            "API_SERVER_KEY": "dummy-api-key-for-test-only",
            "WEBHOOK_ENABLED": "true",
            "WEBHOOK_PORT": "28644",
            "WEBHOOK_SECRET": "dummy-webhook-secret-for-test-only",
        },
    )

    api_config = config.platforms[Platform.API_SERVER]
    webhook_config = config.platforms[Platform.WEBHOOK]

    assert api_config.enabled is True
    assert api_config.extra["port"] == 28642
    assert api_config.extra["host"] == "127.0.0.1"
    assert api_config.extra["key"] == "dummy-api-key-for-test-only"
    assert webhook_config.enabled is True
    assert webhook_config.extra["port"] == 28644
    assert webhook_config.extra["secret"] == "dummy-webhook-secret-for-test-only"


def test_api_server_adapter_reads_port_binding_config_without_starting_site(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from gateway.platforms import api_server

    monkeypatch.setattr(api_server, "ResponseStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        api_server.APIServerAdapter,
        "_resolve_model_name",
        staticmethod(lambda explicit: explicit or "dummy-model"),
    )
    monkeypatch.setattr(
        api_server.APIServerAdapter,
        "_resolve_max_concurrent_runs",
        staticmethod(lambda: 0),
    )
    monkeypatch.setattr(api_server.os, "getenv", lambda name, default=None: default)

    adapter = api_server.APIServerAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "host": "127.0.0.1",
                "port": "28642",
                "key": "dummy-api-key-for-test-only",
                "cors_origins": "https://client.example.invalid, https://admin.example.invalid",
                "model_name": "dummy-api-model",
            },
        )
    )

    assert adapter._host == "127.0.0.1"
    assert adapter._port == 28642
    assert adapter._api_key == "dummy-api-key-for-test-only"
    assert adapter._cors_origins == (
        "https://client.example.invalid",
        "https://admin.example.invalid",
    )
    assert adapter._model_name == "dummy-api-model"
    assert adapter._app is None
    assert adapter._runner is None
    assert adapter._site is None


def test_webhook_adapter_reads_port_binding_config_without_starting_site() -> None:
    from gateway.platforms.webhook import WebhookAdapter

    adapter = WebhookAdapter(
        PlatformConfig(
            enabled=True,
            extra={
                "host": "127.0.0.1",
                "port": "28644",
                "secret": "dummy-webhook-secret-for-test-only",
                "routes": {
                    "ci": {
                        "secret": "dummy-route-secret-for-test-only",
                        "prompt": "summarize {event}",
                        "deliver": "log",
                    }
                },
                "rate_limit": "7",
                "max_body_bytes": "4096",
            },
        )
    )

    assert adapter._host == "127.0.0.1"
    assert adapter._port == 28644
    assert adapter._global_secret == "dummy-webhook-secret-for-test-only"
    assert adapter._static_routes["ci"]["prompt"] == "summarize {event}"
    assert adapter._rate_limit == 7
    assert adapter._max_body_bytes == 4096
    assert adapter._runner is None


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", True),
        ("localhost", True),
        ("::1", True),
        ("[::1]", False),
        ("0.0.0.0", False),
        ("", False),
    ],
)
def test_webhook_loopback_host_classifier_is_static_and_conservative(
    host: str,
    expected: bool,
) -> None:
    from gateway.platforms.webhook import _is_loopback_host

    assert _is_loopback_host(host) is expected
