"""Tests for the Telegram rich-reply index store (``gateway.rich_sent_store``).

The store must resolve its on-disk path through the canonical
``get_hermes_home()`` so the index follows the active profile and lands in the
platform-native Hermes home — not a hand-rolled ``os.environ`` read that
ignores the context-local override and hardcodes ``~/.hermes`` on Windows.
"""

from hermes_constants import (
    get_hermes_home,
    reset_hermes_home_override,
    set_hermes_home_override,
)

from gateway import rich_sent_store


def _expected(home) -> str:
    return str(home / "state" / "rich_sent_index.json")


def test_store_path_tracks_canonical_home_from_env(monkeypatch, tmp_path):
    """With HERMES_HOME set, the store path matches the canonical resolver."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert rich_sent_store._store_path() == _expected(get_hermes_home())


def test_store_path_uses_canonical_home_when_env_unset(monkeypatch):
    """No HERMES_HOME: the store must still defer to the platform-native home
    (``%LOCALAPPDATA%\\hermes`` on Windows), not a hardcoded ``~/.hermes``."""
    monkeypatch.delenv("HERMES_HOME", raising=False)
    assert rich_sent_store._store_path() == _expected(get_hermes_home())


def test_store_path_honors_context_override(monkeypatch, tmp_path):
    """A context-local profile override must redirect the index path.

    ``set_hermes_home_override`` scopes the home per task WITHOUT touching
    ``os.environ`` (env is shared across threads). The old hand-rolled
    ``os.environ.get("HERMES_HOME")`` read could never see this, so it wrote
    the index into the wrong profile. This is the cross-platform regression
    guard for that bug.
    """
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "env_home"))
    override_home = tmp_path / "profile_home"
    token = set_hermes_home_override(override_home)
    try:
        assert rich_sent_store._store_path() == _expected(override_home)
    finally:
        reset_hermes_home_override(token)


def test_record_and_lookup_roundtrip_under_override(monkeypatch, tmp_path):
    """record() then lookup() round-trips, and the file lands under the
    override home — not the env home."""
    env_home = tmp_path / "env_home"
    override_home = tmp_path / "profile_home"
    monkeypatch.setenv("HERMES_HOME", str(env_home))
    token = set_hermes_home_override(override_home)
    try:
        rich_sent_store.record("12345", "678", "Your morning briefing.")
        assert rich_sent_store.lookup("12345", "678") == "Your morning briefing."
        # The index landed in the active profile, not the stale env home.
        assert (override_home / "state" / "rich_sent_index.json").exists()
        assert not (env_home / "state" / "rich_sent_index.json").exists()
    finally:
        reset_hermes_home_override(token)
