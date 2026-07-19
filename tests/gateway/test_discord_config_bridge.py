"""Unit coverage for the Discord config.yaml -> extra / env bridge.

``load_gateway_config`` bridges a ``discord:`` config block into
``PlatformConfig.extra`` and, for the two keys that have a runtime env-var
reader, also into ``os.environ``. These tests pin that contract:

* all bridged Discord keys land in ``extra``;
* a YAML bool serializes to a *lowercase* env string ("true", not "True"),
  matching the adapter's ``str(...).lower()`` reader;
* config.yaml is authoritative for the two env-backed keys (``allow_bots``,
  ``bots_require_inline_mention``): a present config value force-overwrites a
  stale/conflicting env var so every env-only consumer (authz_mixin, relay,
  history) resolves the same value as the config-first admission path; when
  config omits the key the existing env var stays as the intended fallback;
* ``group_mention_role_ids`` / ``group_mention_channel_ids`` are bridged to
  ``extra`` only and are intentionally NOT exported to ``os.environ`` (they
  have no env reader; the adapter consumes them from ``extra``).

The bridge writes directly to ``os.environ``; the ``clean_discord_env``
fixture ``delenv``s the affected vars so monkeypatch teardown restores the
environment even for vars the loader sets mid-test.
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from gateway.config import Platform, load_gateway_config

DISCORD_ENV_VARS = (
    "DISCORD_ALLOW_BOTS",
    "DISCORD_BOTS_REQUIRE_INLINE_MENTION",
    "DISCORD_THREAD_REQUIRE_MENTION",
)
# Group role/channel lists have no env reader and must never be exported.
NON_EXPORTED_ENV_VARS = (
    "DISCORD_GROUP_MENTION_ROLE_IDS",
    "DISCORD_GROUP_MENTION_CHANNEL_IDS",
)


def _load_with_yaml_dict(yaml_dict: dict):
    """Patch the filesystem so load_gateway_config() sees *yaml_dict*."""
    fake_home = Path("/tmp/fake_hermes_home_discord_bridge")

    def fake_exists(self):
        return str(self).endswith("config.yaml")

    with (
        patch("gateway.config.get_hermes_home", return_value=fake_home),
        patch.object(Path, "exists", fake_exists),
        patch("builtins.open", create=True) as mock_file,
    ):
        mock_file.return_value.__enter__ = lambda s: s
        mock_file.return_value.__exit__ = MagicMock(return_value=False)
        with patch("yaml.safe_load", return_value=yaml_dict):
            return load_gateway_config()


@pytest.fixture
def clean_discord_env(monkeypatch):
    """Isolate every Discord env var this file touches.

    ``load_gateway_config`` writes directly to ``os.environ``, so remove any
    variables it creates before monkeypatch restores the original environment.
    """
    vars_to_clear = DISCORD_ENV_VARS + NON_EXPORTED_ENV_VARS
    for var in vars_to_clear:
        monkeypatch.delenv(var, raising=False)
    yield monkeypatch
    # Do not use monkeypatch here: its finalizer would restore the loader's
    # direct write after this fixture exits. These keys were intentionally
    # absent at setup, so remove every value the loader created.
    import os

    for var in vars_to_clear:
        os.environ.pop(var, None)


def _discord_extra(cfg):
    return cfg.platforms[Platform.DISCORD].extra


def test_all_discord_keys_bridged_to_extra(clean_discord_env):
    """Every bridged Discord key is surfaced on PlatformConfig.extra."""
    cfg = _load_with_yaml_dict({
        "discord": {
            "allow_bots": "all",
            "bots_require_inline_mention": True,
            "thread_require_mention": False,
            "group_mention_role_ids": ["111", "222"],
            "group_mention_channel_ids": ["333"],
        }
    })
    extra = _discord_extra(cfg)
    assert extra["allow_bots"] == "all"
    assert extra["bots_require_inline_mention"] is True
    assert extra["thread_require_mention"] is False
    assert extra["group_mention_role_ids"] == ["111", "222"]
    assert extra["group_mention_channel_ids"] == ["333"]


def test_yaml_bool_serializes_to_lowercase_env(clean_discord_env):
    """A YAML bool True bridges to the lowercase env string "true"."""
    import os

    _load_with_yaml_dict({"discord": {"bots_require_inline_mention": True}})
    assert os.environ["DISCORD_BOTS_REQUIRE_INLINE_MENTION"] == "true"


def test_config_force_syncs_allow_bots_over_stale_env(clean_discord_env):
    """config.yaml is authoritative: a stale/conflicting DISCORD_ALLOW_BOTS is
    force-overwritten, not preserved.

    The env-only consumers (authz_mixin, relay, history) read the env var
    directly; force-syncing it to the config value is what keeps them from
    splitting from the config-first admission path.
    """
    import os

    clean_discord_env.setenv("DISCORD_ALLOW_BOTS", "mentions")
    cfg = _load_with_yaml_dict({"discord": {"allow_bots": "all"}})
    # Config wins: env force-synced to the config value...
    assert os.environ["DISCORD_ALLOW_BOTS"] == "all"
    # ...and extra carries the same value for the config-first reader.
    assert _discord_extra(cfg)["allow_bots"] == "all"


def test_config_force_syncs_inline_mention_over_stale_env(clean_discord_env):
    """The same force-sync holds for bots_require_inline_mention: a YAML bool
    overwrites a stale env string and serializes lowercase."""
    import os

    clean_discord_env.setenv("DISCORD_BOTS_REQUIRE_INLINE_MENTION", "false")
    _load_with_yaml_dict({"discord": {"bots_require_inline_mention": True}})
    assert os.environ["DISCORD_BOTS_REQUIRE_INLINE_MENTION"] == "true"


def test_config_absent_leaves_env_fallback_intact(clean_discord_env):
    """When config omits allow_bots, the existing env var is the intended
    fallback and must be left untouched (no force-sync without a config value)."""
    import os

    clean_discord_env.setenv("DISCORD_ALLOW_BOTS", "mentions")
    # discord block present (require_mention) but no allow_bots key -> fallback.
    _load_with_yaml_dict({"discord": {"require_mention": True}})
    assert os.environ["DISCORD_ALLOW_BOTS"] == "mentions"


def test_group_role_and_channel_lists_not_exported_to_env(clean_discord_env):
    """group_mention_* lists bridge to extra only, never to os.environ."""
    import os

    cfg = _load_with_yaml_dict({
        "discord": {
            "group_mention_role_ids": ["111"],
            "group_mention_channel_ids": ["333"],
            "thread_require_mention": True,
        }
    })
    for var in NON_EXPORTED_ENV_VARS:
        assert var not in os.environ, f"{var} must not be exported"
    # Still present where the adapter actually reads them.
    extra = _discord_extra(cfg)
    assert extra["group_mention_role_ids"] == ["111"]
    assert extra["group_mention_channel_ids"] == ["333"]
    assert extra["thread_require_mention"] is True
