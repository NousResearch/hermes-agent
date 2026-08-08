"""Every credential store the read guard hides must also be write-protected.

``agent/file_safety`` kept two hand-maintained lists and they had drifted. The
read guard blocked ``auth.json``, ``auth.lock``, ``webhook_subscriptions.json``,
``auth/google_oauth.json`` and the plaintext ``cache/bws_cache.json``; the write
denylist knew none of them. The agent therefore could not READ its own OAuth
store but could freely OVERWRITE it — one ``write_file`` and every provider
login is gone, with no backup and no undo.

The drift ran the other way too: the write list knew only the ENCRYPTED
``cache/bws_cache.enc.json`` while the read list knew only the plaintext
spelling, so each guard was missing the other's file. Both are real
(``agent/secret_sources/bitwarden.py`` writes both).

Both guards now read one list, so the pairing is asserted rather than assumed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.file_safety import (
    HERMES_CREDENTIAL_FILE_NAMES,
    get_read_block_error,
    is_write_denied,
)


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_WRITE_SAFE_ROOT", raising=False)
    return home


def _touch(home: Path, rel: str) -> Path:
    p = home / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("secret", encoding="utf-8")
    return p


@pytest.mark.parametrize("rel", HERMES_CREDENTIAL_FILE_NAMES)
class TestCredentialStoresAreBlockedBothWays:
    def test_read_is_blocked(self, hermes_home, rel):
        assert get_read_block_error(str(_touch(hermes_home, rel))) is not None

    def test_write_is_blocked(self, hermes_home, rel):
        assert is_write_denied(str(_touch(hermes_home, rel))) is True


class TestTheSpecificFilesThatWereWritable:
    """Named explicitly so a future edit to the shared list can't silently
    drop one of the five that were reachable."""

    @pytest.mark.parametrize(
        "rel",
        [
            "auth.json",
            "auth.lock",
            "webhook_subscriptions.json",
            "auth/google_oauth.json",
            "cache/bws_cache.json",
        ],
    )
    def test_no_longer_overwritable(self, hermes_home, rel):
        assert is_write_denied(str(_touch(hermes_home, rel))) is True

    def test_both_bitwarden_cache_spellings_are_covered(self, hermes_home):
        for rel in ("cache/bws_cache.json", "cache/bws_cache.enc.json"):
            p = _touch(hermes_home, rel)
            assert is_write_denied(str(p)) is True
            assert get_read_block_error(str(p)) is not None


class TestOrdinaryFilesUnaffected:
    def test_skill_file_stays_writable(self, hermes_home):
        assert is_write_denied(str(_touch(hermes_home, "skills/mine/SKILL.md"))) is False

    def test_project_file_stays_writable(self, tmp_path, hermes_home):
        p = tmp_path / "project" / "main.py"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")
        assert is_write_denied(str(p)) is False

    def test_env_example_stays_writable(self, tmp_path, hermes_home):
        p = tmp_path / "project" / ".env.example"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x", encoding="utf-8")
        assert is_write_denied(str(p)) is False
