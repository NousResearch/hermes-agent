"""Configurable credential resolution order for Anthropic.

``resolve_anthropic_token()`` consults credential sources in a fixed order, with
the Claude Code credentials file ranked above the Hermes credential pool. For a
multi-account pool that is the wrong precedence: the pool rotates, but every
call site that falls back to a bare resolve bills whichever account the static
file happens to hold.

These tests pin the default order (unchanged) and the config override that lets
a user promote ``credential_pool``.
"""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from agent.anthropic_adapter import (
    DEFAULT_ANTHROPIC_RESOLVE_ORDER,
    SOURCE_CLAUDE_CODE_CREDENTIALS,
    SOURCE_CREDENTIAL_POOL,
    SOURCE_ENV_ANTHROPIC_TOKEN,
    get_anthropic_resolve_order,
    resolve_anthropic_token,
)

CLAUDE_TOKEN = "sk-ant-oat01-from-claude-code-file"
POOL_TOKEN = "sk-ant-oat01-from-credential-pool"


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """A temp HERMES_HOME with a real config.yaml and auth.json pool entry."""
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    (home / "auth.json").write_text(
        json.dumps(
            {
                "credential_pool": {
                    "anthropic": [
                        {
                            "id": "pool1",
                            "label": "anthropic-pooled",
                            "auth_type": "oauth",
                            "priority": 0,
                            "source": "manual",
                            "access_token": POOL_TOKEN,
                            "base_url": "https://api.anthropic.com",
                        }
                    ]
                }
            }
        )
    )
    # No Anthropic env vars should leak in from the developer's shell.
    for var in ("ANTHROPIC_TOKEN", "CLAUDE_CODE_OAUTH_TOKEN", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    return home


def write_config(home: Path, config: dict) -> None:
    (home / "config.yaml").write_text(yaml.safe_dump(config))


class TestResolveOrderConfig:
    def test_default_order_when_unset(self, hermes_home):
        write_config(hermes_home, {})
        assert get_anthropic_resolve_order() == DEFAULT_ANTHROPIC_RESOLVE_ORDER

    def test_default_order_when_config_missing(self, hermes_home):
        assert get_anthropic_resolve_order() == DEFAULT_ANTHROPIC_RESOLVE_ORDER

    def test_pool_can_be_promoted(self, hermes_home):
        write_config(
            hermes_home,
            {"credential_resolve_order": {"anthropic": [SOURCE_CREDENTIAL_POOL]}},
        )
        order = get_anthropic_resolve_order()
        assert order[0] == SOURCE_CREDENTIAL_POOL
        # Unmentioned sources are retained, in default relative order.
        assert set(order) == set(DEFAULT_ANTHROPIC_RESOLVE_ORDER)
        assert order.index(SOURCE_CREDENTIAL_POOL) < order.index(
            SOURCE_CLAUDE_CODE_CREDENTIALS
        )

    def test_unknown_sources_ignored(self, hermes_home):
        write_config(
            hermes_home,
            {
                "credential_resolve_order": {
                    "anthropic": ["not_a_source", SOURCE_CREDENTIAL_POOL]
                }
            },
        )
        order = get_anthropic_resolve_order()
        assert "not_a_source" not in order
        assert order[0] == SOURCE_CREDENTIAL_POOL

    def test_all_invalid_falls_back_to_default(self, hermes_home):
        write_config(
            hermes_home, {"credential_resolve_order": {"anthropic": ["nonsense"]}}
        )
        assert get_anthropic_resolve_order() == DEFAULT_ANTHROPIC_RESOLVE_ORDER

    def test_non_list_value_falls_back_to_default(self, hermes_home):
        write_config(
            hermes_home,
            {"credential_resolve_order": {"anthropic": SOURCE_CREDENTIAL_POOL}},
        )
        assert get_anthropic_resolve_order() == DEFAULT_ANTHROPIC_RESOLVE_ORDER

    def test_duplicates_collapse(self, hermes_home):
        write_config(
            hermes_home,
            {
                "credential_resolve_order": {
                    "anthropic": [SOURCE_CREDENTIAL_POOL, SOURCE_CREDENTIAL_POOL]
                }
            },
        )
        order = get_anthropic_resolve_order()
        assert len(order) == len(set(order)) == len(DEFAULT_ANTHROPIC_RESOLVE_ORDER)


class TestResolveTokenHonoursOrder:
    """End-to-end: the resolved token follows the configured order."""

    def _patch_claude_file(self):
        return patch(
            "agent.anthropic_adapter.read_claude_code_credentials",
            return_value={"accessToken": CLAUDE_TOKEN},
        )

    def test_default_prefers_claude_code_file(self, hermes_home):
        write_config(hermes_home, {})
        with self._patch_claude_file():
            assert resolve_anthropic_token() == CLAUDE_TOKEN

    def test_promoted_pool_wins_over_claude_code_file(self, hermes_home):
        write_config(
            hermes_home,
            {"credential_resolve_order": {"anthropic": [SOURCE_CREDENTIAL_POOL]}},
        )
        with self._patch_claude_file():
            assert resolve_anthropic_token() == POOL_TOKEN

    def test_promoted_pool_still_falls_back_when_pool_empty(self, hermes_home):
        (hermes_home / "auth.json").write_text(
            json.dumps({"credential_pool": {"anthropic": []}})
        )
        write_config(
            hermes_home,
            {"credential_resolve_order": {"anthropic": [SOURCE_CREDENTIAL_POOL]}},
        )
        with self._patch_claude_file():
            # Pool yields nothing, so the next source in order still answers.
            assert resolve_anthropic_token() == CLAUDE_TOKEN

    def test_explicit_env_token_still_wins_when_listed_first(
        self, hermes_home, monkeypatch
    ):
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-explicit-env")
        write_config(
            hermes_home,
            {
                "credential_resolve_order": {
                    "anthropic": [SOURCE_ENV_ANTHROPIC_TOKEN, SOURCE_CREDENTIAL_POOL]
                }
            },
        )
        with patch(
            "agent.anthropic_adapter.read_claude_code_credentials", return_value=None
        ):
            assert resolve_anthropic_token() == "sk-ant-oat01-explicit-env"

    def test_env_token_can_be_deprioritised_below_pool(self, hermes_home, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-explicit-env")
        write_config(
            hermes_home,
            {"credential_resolve_order": {"anthropic": [SOURCE_CREDENTIAL_POOL]}},
        )
        with patch(
            "agent.anthropic_adapter.read_claude_code_credentials", return_value=None
        ):
            assert resolve_anthropic_token() == POOL_TOKEN

    def test_returns_none_when_no_source_answers(self, hermes_home):
        (hermes_home / "auth.json").write_text(
            json.dumps({"credential_pool": {"anthropic": []}})
        )
        write_config(hermes_home, {})
        with patch(
            "agent.anthropic_adapter.read_claude_code_credentials", return_value=None
        ):
            assert resolve_anthropic_token() is None


class TestConfigSchemaAcceptsKey:
    def test_key_is_open_dict_so_arbitrary_providers_validate(self):
        from hermes_cli.config import _OPEN_DICT_TOP_LEVEL_KEYS

        assert "credential_resolve_order" in _OPEN_DICT_TOP_LEVEL_KEYS

    def test_default_config_declares_the_key(self):
        from hermes_cli.config_defaults import DEFAULT_CONFIG

        assert DEFAULT_CONFIG["credential_resolve_order"] == {}
