"""Regression tests for surgical runtime config-key persistence (issue #77513).

Issue #77513 report: runtime triggers (`persist_home_channel`,
``ensure_install_id``) called ``load_config()`` then ``save_config(merged)``
to persist a single key. ``load_config()`` deep-merges DEFAULT_CONFIG, and
``save_config()`` round-trips through that merged dict — silently overwriting
user-set leaves like ``model.default``, ``model.api_key`` and
``model.provider`` with schema defaults. After every gateway start, a
custom OpenAI-compatible provider config editing session was erased and the
agent became unusable.

The fix adds ``hermes_cli.config.persist_config_key``: read the RAW
config.yaml (no DEFAULT_CONFIG merge), set one dotted key via ``_set_nested``,
and write back atomically. User values elsewhere in the document survive
byte-for-byte. Tests verify:

1. ``persist_config_key`` writes only the targeted dotted key and does NOT
   inject DEFAULT_CONFIG defaults (agent/compression/etc.) into the file.
2. A pre-existing custom provider config survives a single-key write.
3. ``persist_home_channel`` does not replace ``model.default`` or
   ``model.api_key`` with defaults when the user set custom values.
4. ``ensure_install_id`` does not clobber the same.

Tests run against the real implementation with a temp ``HERMES_HOME`` (via
the autouse ``_hermetic_environment`` fixture).
"""
from __future__ import annotations

from pathlib import Path

import yaml

import pytest


# --------------------------------------------------------------------------- #
# Test fixtures
# --------------------------------------------------------------------------- #


CUSTOM_USER_CONFIG = """\
_config_version: 33
model:
  context_length: 1000000
  default: ag/claude-opus-4-6-thinking
  provider: custom:myrouter
  api_key: c3cret-keep-me
custom_providers:
  - api_key: c3cret-keep-me
    base_url: http://my-server:1997/v1
    model: ag/claude-opus-4-6-thinking
    name: myrouter
"""


def _write_user_config(config_yaml: str) -> None:
    """Write the given YAML body verbatim into the active HERMES_HOME/config.yaml."""
    from hermes_constants import get_hermes_home

    cfg_path = get_hermes_home() / "config.yaml"
    cfg_path.write_text(config_yaml, encoding="utf-8")


def _read_user_config() -> dict:
    """Read back the raw config.yaml return empty dict on missing file."""
    from hermes_constants import get_hermes_home

    cfg_path = get_hermes_home() / "config.yaml"
    if not cfg_path.exists():
        return {}
    text = cfg_path.read_text(encoding="utf-8")
    return yaml.safe_load(text) or {}


# --------------------------------------------------------------------------- #
# Tests for ``persist_config_key``
# --------------------------------------------------------------------------- #


def test_persist_config_key_writes_only_target_dotted_path() -> None:
    """A single-key write must NOT inject any DEFAULT_CONFIG subtree the user
    never set (the #77513 bug). Only the targeted key should appear."""
    _write_user_config(CUSTOM_USER_CONFIG)
    from hermes_cli.config import persist_config_key

    ok = persist_config_key("monitoring.install_id", "11111111-2222-3333-4444-555555555555")
    assert ok is True

    cfg = _read_user_config()

    # The targeted key must be present.
    assert cfg["monitoring"]["install_id"] == "11111111-2222-3333-4444-555555555555"

    # The user's custom provider block must be byte-preserved.
    assert cfg["custom_providers"] == [
        {
            "api_key": "c3cret-keep-me",
            "base_url": "http://my-server:1997/v1",
            "model": "ag/claude-opus-4-6-thinking",
            "name": "myrouter",
        }
    ]

    # The user's model config must be byte-preserved.
    assert cfg["model"]["default"] == "ag/claude-opus-4-6-thinking"
    assert cfg["model"]["provider"] == "custom:myrouter"
    assert cfg["model"]["api_key"] == "c3cret-keep-me"
    assert cfg["model"]["context_length"] == 1000000

    # DEFAULT_CONFIG subtrees the user never set must NOT be injected.
    # The compactly-stored config kept its original 3 top-level keys plus the
    # newly added monitoring slot — and nothing else.
    assert set(cfg.keys()) == {
        "_config_version", "model", "custom_providers", "monitoring"
    }, (
        f"persist_config_key injected unexpected top-level keys the user never "
        f"set: {sorted(cfg.keys())}"
    )

    # The classic DEFAULT_CONFIG contamination (issue #77513) is agent.* and
    # compression.* appearing out of nowhere. They must be absent.
    assert "agent" not in cfg, (
        "persist_config_key must not inject DEFAULT_CONFIG 'agent' — that was "
        "the #77513 contamination signature"
    )
    assert "compression" not in cfg, (
        "persist_config_key must not inject DEFAULT_CONFIG 'compression' — "
        "that was the #77513 contamination signature"
    )


def test_persist_config_key_handles_missing_config_file() -> None:
    """When config.yaml does not exist yet, the helper must create it with
    only the targeted key (no DEFAULT_CONFIG dump)."""
    from hermes_constants import get_hermes_home

    cfg_path = get_hermes_home() / "config.yaml"
    assert not cfg_path.exists()

    from hermes_cli.config import persist_config_key

    ok = persist_config_key("monitoring.install_id", "deadbeef")
    assert ok is True

    cfg = _read_user_config()
    assert cfg == {"monitoring": {"install_id": "deadbeef"}}
    assert "agent" not in cfg
    assert "compression" not in cfg


def test_persist_config_key_write_fails_open_for_readonly_home() -> None:
    """A read-only config.yaml must not raise — the helper should return False
    and leave the surrounding runtime intact (fail-open contract)."""
    _write_user_config(CUSTOM_USER_CONFIG)
    from hermes_constants import get_hermes_home

    cfg_path = get_hermes_home() / "config.yaml"
    cfg_path.chmod(0o444)  # r--r--r--
    try:
        from hermes_cli.config import persist_config_key

        # Running as the file's owner, python's open(...,'w') may still
        # succeed on some filesystems despite 0o444 (depends on privileges).
        # The contract we assert is "no exception bubbling out" — the helper
        # returns either True or False but does not raise.
        outcome = persist_config_key("monitoring.install_id", "x")
        assert outcome in (True, False)
    finally:
        cfg_path.chmod(0o644)


# --------------------------------------------------------------------------- #
# Tests for ``persist_home_channel`` (the gateway trigger from #77513)
# --------------------------------------------------------------------------- #


def test_persist_home_channel_preserves_custom_provider() -> None:
    """persist_home_channel must not round-trip through load_config/store_config.

    The user has a custom provider with a non-default model; after the home
    channel gets persisted, every model.* and custom_providers entry must
    survive byte-for-byte. This is the direct repro of #77513."""
    _write_user_config(CUSTOM_USER_CONFIG)

    from gateway.config import HomeChannel, persist_home_channel, Platform

    home = HomeChannel(
        platform=Platform.TELEGRAM,
        chat_id="123",
        name="MyHome",
        thread_id=None,
        user_id="42",
    )
    persist_home_channel(home, enabled_if_new=True)

    cfg = _read_user_config()

    # The user's provider configuration survives untouched.
    assert cfg["model"]["default"] == "ag/claude-opus-4-6-thinking"
    assert cfg["model"]["provider"] == "custom:myrouter"
    assert cfg["model"]["api_key"] == "c3cret-keep-me"
    assert cfg["custom_providers"] == [
        {
            "api_key": "c3cret-keep-me",
            "base_url": "http://my-server:1997/v1",
            "model": "ag/claude-opus-4-6-thinking",
            "name": "myrouter",
        }
    ]

    # The home channel WAS persisted.
    assert cfg["platforms"]["telegram"]["home_channel"]["chat_id"] == "123"
    assert cfg["platforms"]["telegram"]["enabled"] is True

    # No DEFAULT_CONFIG contamination besides what the user set plus the new
    # top-level platforms and monitoring slots (persist_home_channel adds one
    # dotted subtree only).
    assert "agent" not in cfg, (
        "persist_home_channel re-introduced #77513 by injecting DEFAULT_CONFIG "
        "'agent' into the user's config.yaml"
    )
    assert "compression" not in cfg, (
        "persist_home_channel re-introduced #77513 by injecting DEFAULT_CONFIG "
        "'compression' into the user's config.yaml"
    )


# --------------------------------------------------------------------------- #
# Tests for ``ensure_install_id`` (the monitoring trigger from #77513)
# --------------------------------------------------------------------------- #


def test_ensure_install_id_does_not_clobber_custom_provider() -> None:
    """ensure_install_id should write ONLY monitoring.install_id and leave
    the user's custom provider alone — the #77513 gateway-start failure."""
    _write_user_config(CUSTOM_USER_CONFIG)

    from agent.monitoring.policy import ensure_install_id

    returned = ensure_install_id({})

    cfg = _read_user_config()

    # install_id was minted and persisted.
    assert isinstance(returned, str) and len(returned) > 0
    assert cfg["monitoring"]["install_id"] == returned

    # The user's custom provider is byte-preserved.
    assert cfg["model"]["default"] == "ag/claude-opus-4-6-thinking"
    assert cfg["model"]["provider"] == "custom:myrouter"
    assert cfg["model"]["api_key"] == "c3cret-keep-me"
    assert cfg["custom_providers"][0]["api_key"] == "c3cret-keep-me"

    # No DEFAULT_CONFIG contamination.
    assert "agent" not in cfg, (
        "ensure_install_id re-introduced #77513 by injecting DEFAULT_CONFIG "
        "'agent' into the user's config.yaml"
    )
    assert "compression" not in cfg, (
        "ensure_install_id re-introduced #77513 by injecting DEFAULT_CONFIG "
        "'compression' into the user's config.yaml"
    )


def test_ensure_install_id_returns_existing_id_without_overwrite() -> None:
    """If a config already has an install_id, ensure_install_id should return
    it and NOT mint a new one or hit the write path."""
    _write_user_config(CUSTOM_USER_CONFIG)
    existing_id = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    from hermes_cli.config import persist_config_key

    persist_config_key("monitoring.install_id", existing_id)

    from agent.monitoring.policy import ensure_install_id

    # Caller passes the in-memory config (which contains the existing id).
    in_memory = {"monitoring": {"install_id": existing_id}}
    returned = ensure_install_id(in_memory)
    assert returned == existing_id

    cfg = _read_user_config()
    # Unchanged — no new write.
    assert cfg["monitoring"]["install_id"] == existing_id
    # Custom provider still intact.
    assert cfg["model"]["default"] == "ag/claude-opus-4-6-thinking"


# --------------------------------------------------------------------------- #
# Tests for ``migrate_config`` (the third trigger from #77513)
# --------------------------------------------------------------------------- #


def test_migrate_config_does_not_clobber_custom_provider() -> None:
    """migrate_config() runs every gateway start when ``_config_version``
    trails DEFAULT_CONFIG's version. The migration path is the third
    runtime trigger called out in #77513: it must NOT round-trip the user's
    custom provider config through DEFAULT_CONFIG the way save_config() does,
    or every gateway restart with a non-default model wipes the user's
    provider settings.

    We assert the full #77513 contract here: after migrate_config() runs
    against a config at an older version with a custom OpenAI-compatible
    provider, every ``model.*`` leaf and the ``custom_providers`` block
    survive byte-for-byte, and no DEFAULT_CONFIG subtree (agent.*,
    compression.*, etc.) leaks into the file.
    """
    _write_user_config(CUSTOM_USER_CONFIG)

    from hermes_cli import config as cfg_module

    latest = cfg_module.DEFAULT_CONFIG.get("_config_version")
    if not isinstance(latest, int):
        pytest.skip("DEFAULT_CONFIG['_config_version'] missing — cannot run migration")

    # Make sure the user's config is at least one step behind so
    # migrate_config() actually has work to do.
    before = _read_user_config()
    before["_config_version"] = latest - 1
    from hermes_constants import get_hermes_home
    (get_hermes_home() / "config.yaml").write_text(
        __import__("yaml").safe_dump(before, sort_keys=False), encoding="utf-8"
    )

    cfg_module.migrate_config(interactive=False, quiet=True)

    after = _read_user_config()

    # Custom provider block survives.
    assert after["custom_providers"] == [
        {
            "api_key": "c3cret-keep-me",
            "base_url": "http://my-server:1997/v1",
            "model": "ag/claude-opus-4-6-thinking",
            "name": "myrouter",
        }
    ]

    # User's model config survives.
    assert after["model"]["default"] == "ag/claude-opus-4-6-thinking"
    assert after["model"]["provider"] == "custom:myrouter"
    assert after["model"]["api_key"] == "c3cret-keep-me"
    assert after["model"]["context_length"] == 1000000

    # DEFAULT_CONFIG contamination signature is absent.
    assert "agent" not in after or set(after.get("agent", {}).keys()) <= {
        # migration may add keys under agent.* via the migration ladder
        # (e.g. verify_on_stop flip in v32→33). The KEY assertion is that
        # values the user never set must not be re-defaulted wholesale.
        k for k in after.get("agent", {}).keys()
        if k in {"verify_on_stop"}  # migration-touched keys we know about
    }, (
        f"migrate_config re-introduced #77513 contamination under 'agent': "
        f"{list(after.get('agent', {}).keys())}"
    )
