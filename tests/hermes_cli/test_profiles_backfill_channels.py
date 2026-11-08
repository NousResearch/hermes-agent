"""``hermes update``'s .env backfill must never copy the default's secrets into a profile.

Secrets are profile-scoped, and one bot token can't be consumed twice: a copied TELEGRAM_BOT_TOKEN only
ever surfaces as a ``duplicate_credential`` fatal next to the default's live adapter.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli.profiles import backfill_profile_envs, create_profile


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


def test_backfill_never_copies_default_credentials(profile_env):
    (profile_env / ".hermes" / ".env").write_text(
        "OPENROUTER_API_KEY=root-key\nTELEGRAM_BOT_TOKEN=123:abc\nDISCORD_BOT_TOKEN=d1\n", encoding="utf-8"
    )
    p = create_profile("legacy", no_alias=True, no_skills=True)
    (p / ".env").unlink()

    assert backfill_profile_envs(quiet=True) == ["legacy"]
    assert all(line.startswith("#") or not line.strip()
               for line in (p / ".env").read_text(encoding="utf-8-sig").splitlines())
