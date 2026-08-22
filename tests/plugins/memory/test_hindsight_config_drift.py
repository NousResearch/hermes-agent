"""Regression tests for the embedded Hindsight config-drift check (#82943).

The hindsight-embed profile ``.env`` has two writers. The plugin owns the LLM
settings; ``hindsight_embed.ProfileManager.create_profile`` re-renders the same
file from its own bundled template on every ``ensure_running`` — including the
"daemon already healthy, reuse it" path — and writes ``HINDSIGHT_API_PORT``,
which the plugin never emits.

A whole-dict ``saved != expected`` compare therefore never matched, so the
"config changed" branch fired on every single session start and stopped a
daemon that was already healthy (SIGTERM reads as a clean exit, so an
externally supervised unit never restarts). These tests pin the comparison to
the keys the plugin actually manages, in both directions: foreign keys must be
ignored, and real drift — including *removal* of a conditionally-emitted key —
must still be caught.
"""

from pathlib import Path

import pytest

from plugins.memory.hindsight import (
    _MANAGED_PROFILE_ENV_KEYS,
    _build_embedded_profile_env,
    _embedded_profile_env_changed,
    _embedded_profile_env_path,
    _load_simple_env,
    _materialize_embedded_profile_env,
)


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    """Keep every profile-env write inside the temp dir, never ``~/.hindsight``."""
    isolated_home = tmp_path / "user-home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: isolated_home))
    return isolated_home


@pytest.fixture(autouse=True)
def _no_ambient_env(monkeypatch):
    """``_build_embedded_profile_env`` falls back to these when config omits them."""
    monkeypatch.delenv("HINDSIGHT_API_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("HINDSIGHT_IDLE_TIMEOUT", raising=False)


_CONFIG = {
    "profile": "hermes",
    "llm_provider": "openai",
    "llm_model": "gpt-4o-mini",
}

# Exercises both conditionally-emitted keys.
_CONFIG_FULL = {
    **_CONFIG,
    "llm_base_url": "https://example.invalid/v1",
    "idle_timeout": 900,
}


def _expected(config=_CONFIG):
    return _build_embedded_profile_env(config, llm_api_key="sk-test")


# ---------------------------------------------------------------------------
# The bug: a key owned by hindsight_embed must not read as local config drift
# ---------------------------------------------------------------------------


def test_foreign_daemon_key_does_not_trigger_restart():
    """#82943: ``HINDSIGHT_API_PORT`` is written by hindsight_embed, not by the
    plugin. Its presence alone must not restart a healthy daemon."""
    expected = _expected()
    saved = {**expected, "HINDSIGHT_API_PORT": "9177"}

    assert _embedded_profile_env_changed(saved, expected) is False


def test_unknown_future_daemon_keys_are_also_ignored():
    """The guarantee is "keys we don't manage", not a HINDSIGHT_API_PORT
    special case — the template is free to grow new keys."""
    expected = _expected()
    saved = {
        **expected,
        "HINDSIGHT_API_PORT": "9177",
        "HINDSIGHT_API_HOST": "127.0.0.1",
        "HINDSIGHT_DB_PATH": "/var/lib/hindsight/db",
    }

    assert _embedded_profile_env_changed(saved, expected) is False


def test_identical_managed_env_is_not_a_change():
    expected = _expected()

    assert _embedded_profile_env_changed(dict(expected), expected) is False


# ---------------------------------------------------------------------------
# Real drift must still be caught — the check has to stay useful
# ---------------------------------------------------------------------------


def test_changed_managed_value_is_detected():
    expected = _expected()
    saved = {**expected, "HINDSIGHT_API_LLM_MODEL": "some-older-model"}

    assert _embedded_profile_env_changed(saved, expected) is True


def test_rotated_api_key_is_detected():
    expected = _expected()
    saved = {**expected, "HINDSIGHT_API_LLM_API_KEY": "sk-previous"}

    assert _embedded_profile_env_changed(saved, expected) is True


def test_missing_managed_key_is_detected():
    expected = _expected()
    saved = {k: v for k, v in expected.items() if k != "HINDSIGHT_API_LLM_PROVIDER"}
    saved["HINDSIGHT_API_LLM_PROVIDER"] = ""

    assert _embedded_profile_env_changed(saved, expected) is True


def test_removed_conditional_key_is_detected():
    """Dropping ``llm_base_url`` from the config removes the key from *expected*
    while the stale value lives on in the saved file. A subset-of-expected
    compare would miss this; the fixed managed-key set catches it."""
    saved = _expected(_CONFIG_FULL)
    expected = _expected(_CONFIG)  # no llm_base_url, no idle_timeout

    assert "HINDSIGHT_API_LLM_BASE_URL" in saved
    assert "HINDSIGHT_API_LLM_BASE_URL" not in expected
    assert _embedded_profile_env_changed(saved, expected) is True


def test_added_conditional_key_is_detected():
    saved = _expected(_CONFIG)
    expected = _expected(_CONFIG_FULL)

    assert _embedded_profile_env_changed(saved, expected) is True


def test_missing_and_empty_managed_value_are_equivalent():
    """``_build_embedded_profile_env`` renders an unset value as ``""`` for the
    keys it always emits and omits the conditional ones, so ``KEY=`` and a
    missing KEY describe the same state. Treating them as different would
    reintroduce the permanent-restart loop via an empty template placeholder."""
    expected = _expected()
    saved = {**expected, "HINDSIGHT_API_LLM_BASE_URL": ""}

    assert "HINDSIGHT_API_LLM_BASE_URL" not in expected
    assert _embedded_profile_env_changed(saved, expected) is False


# ---------------------------------------------------------------------------
# The managed-key set must stay pinned to the builder
# ---------------------------------------------------------------------------


def test_managed_keys_cover_every_built_key():
    """Invariant: every key ``_build_embedded_profile_env`` can emit is declared
    managed. Without this, a new key added to the builder would be compared
    against nothing and silently stop triggering a restart."""
    for config in (_CONFIG, _CONFIG_FULL):
        built = set(_build_embedded_profile_env(config, llm_api_key="sk-test"))
        assert built <= _MANAGED_PROFILE_ENV_KEYS, (
            f"builder emits keys not declared in _MANAGED_PROFILE_ENV_KEYS: "
            f"{sorted(built - _MANAGED_PROFILE_ENV_KEYS)}"
        )

    # Guard against the assertion above passing vacuously: the maximal config
    # must actually exercise the conditionally-emitted keys.
    full = set(_build_embedded_profile_env(_CONFIG_FULL, llm_api_key="sk-test"))
    assert {"HINDSIGHT_API_LLM_BASE_URL", "HINDSIGHT_EMBED_DAEMON_IDLE_TIMEOUT"} <= full


# ---------------------------------------------------------------------------
# End-to-end over the real file, the way the daemon path actually runs
# ---------------------------------------------------------------------------


def test_materialized_env_plus_daemon_port_key_is_stable_on_reload():
    """The real loop: the plugin writes the profile env, hindsight_embed appends
    its port key, and the next session start re-reads the file. That second read
    must not report a config change."""
    profile_env = _materialize_embedded_profile_env(_CONFIG, llm_api_key="sk-test")

    # hindsight_embed re-renders the file and adds the key the plugin never emits.
    with profile_env.open("a", encoding="utf-8") as fh:
        fh.write("HINDSIGHT_API_PORT=9177\n")

    saved = _load_simple_env(profile_env)
    expected = _build_embedded_profile_env(_CONFIG, llm_api_key="sk-test")

    assert saved["HINDSIGHT_API_PORT"] == "9177"
    assert _embedded_profile_env_changed(saved, expected) is False, (
        "a healthy daemon would be SIGTERMed on every session start"
    )


def test_real_config_edit_still_restarts_after_daemon_wrote_its_port_key():
    """The complement: a genuine config change is still caught even once the
    foreign port key is present in the file."""
    profile_env = _materialize_embedded_profile_env(_CONFIG, llm_api_key="sk-test")
    with profile_env.open("a", encoding="utf-8") as fh:
        fh.write("HINDSIGHT_API_PORT=9177\n")

    saved = _load_simple_env(profile_env)
    expected = _build_embedded_profile_env(
        {**_CONFIG, "llm_model": "gpt-4o"}, llm_api_key="sk-test"
    )

    assert _embedded_profile_env_changed(saved, expected) is True
    assert _embedded_profile_env_path(_CONFIG) == profile_env
