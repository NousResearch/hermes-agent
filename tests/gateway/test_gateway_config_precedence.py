"""Gateway config precedence tests for nested gateway.* fallbacks.

These tests exercise only ``load_gateway_config()`` against a temporary
``HERMES_HOME`` containing a synthetic config.yaml. They must not read a real
profile config or launch gateway/runtime code.
"""

from __future__ import annotations

from textwrap import dedent

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override


@pytest.fixture
def load_config_from_temp_home(tmp_path):
    def _load(yaml_text: str):
        hermes_home = tmp_path / "hermes-home"
        hermes_home.mkdir()
        (hermes_home / "config.yaml").write_text(dedent(yaml_text), encoding="utf-8")

        token = set_hermes_home_override(hermes_home)
        try:
            from gateway.config import load_gateway_config

            return load_gateway_config()
        finally:
            reset_hermes_home_override(token)

    return _load


class TestGatewayConfigPrecedence:
    def test_top_level_multiplex_profiles_precedes_nested_gateway_value(
        self, load_config_from_temp_home
    ):
        cfg = load_config_from_temp_home(
            """
            multiplex_profiles: true
            gateway:
              multiplex_profiles: false
            """
        )

        assert cfg.multiplex_profiles is True

    def test_top_level_max_concurrent_sessions_precedes_nested_gateway_value(
        self, load_config_from_temp_home
    ):
        cfg = load_config_from_temp_home(
            """
            max_concurrent_sessions: 7
            gateway:
              max_concurrent_sessions: 3
            """
        )

        assert cfg.max_concurrent_sessions == 7

    def test_top_level_streaming_precedes_nested_gateway_value(
        self, load_config_from_temp_home
    ):
        cfg = load_config_from_temp_home(
            """
            streaming:
              enabled: true
              transport: edit
            gateway:
              streaming:
                enabled: false
                transport: draft
            """
        )

        assert cfg.streaming.enabled is True
        assert cfg.streaming.transport == "edit"

    @pytest.mark.parametrize(
        ("yaml_text", "assertion"),
        [
            (
                """
                gateway:
                  multiplex_profiles: true
                """,
                lambda cfg: cfg.multiplex_profiles is True,
            ),
            (
                """
                gateway:
                  max_concurrent_sessions: 4
                """,
                lambda cfg: cfg.max_concurrent_sessions == 4,
            ),
            (
                """
                gateway:
                  streaming:
                    enabled: true
                    transport: draft
                """,
                lambda cfg: cfg.streaming.enabled is True
                and cfg.streaming.transport == "draft",
            ),
        ],
    )
    def test_nested_gateway_values_used_when_top_level_absent(
        self, load_config_from_temp_home, yaml_text, assertion
    ):
        cfg = load_config_from_temp_home(yaml_text)

        assert assertion(cfg)

    def test_temp_home_sessions_dir_confirms_no_real_profile_config_read(
        self, load_config_from_temp_home, tmp_path
    ):
        cfg = load_config_from_temp_home(
            """
            gateway:
              max_concurrent_sessions: 2
            """
        )

        assert cfg.sessions_dir == tmp_path / "hermes-home" / "sessions"
        assert cfg.max_concurrent_sessions == 2
