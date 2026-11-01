"""``hermes update``'s .env backfill must not hand a profile the default's bot credentials.

One bot token can't be consumed twice: a copied TELEGRAM_BOT_TOKEN only ever surfaces as a
``duplicate_credential`` fatal next to the default's live adapter (Desktop: "Backend telegram down").
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


def test_backfill_keeps_model_keys_and_drops_messaging_channels(profile_env):
    (profile_env / ".hermes" / ".env").write_text(
        "OPENROUTER_API_KEY=root-key\nTELEGRAM_BOT_TOKEN=123:abc\nTELEGRAM_ALLOWED_USERS=42\n"
        "DISCORD_BOT_TOKEN=d1\nDISCORD_HOME_CHANNEL=9\nFAL_KEY=f\n",
        encoding="utf-8",
    )
    p = create_profile("legacy", no_alias=True, no_skills=True)
    (p / ".env").unlink()

    assert backfill_profile_envs(quiet=True) == ["legacy"]
    assert (p / ".env").read_text(encoding="utf-8-sig") == "OPENROUTER_API_KEY=root-key\nFAL_KEY=f\n"
