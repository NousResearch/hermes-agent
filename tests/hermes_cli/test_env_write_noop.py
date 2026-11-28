"""A ``.env`` write that changes nothing leaves the file alone.

The gateway's served-profile reconcile signs ``.env`` by stat (inode, ctime) plus digest; an
atomic replace with identical bytes still swaps the inode, reads as a credential change, and
reconnects that profile's adapters.
"""

from __future__ import annotations

import os

from hermes_cli import config as config_mod


def test_identical_env_save_keeps_the_file_and_a_real_change_still_lands(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    env_path = tmp_path / ".env"
    config_mod.save_env_value("TELEGRAM_HOME_CHANNEL", "12345")
    before = os.stat(env_path)

    config_mod.save_env_value("TELEGRAM_HOME_CHANNEL", "12345")
    after = os.stat(env_path)
    assert (after.st_ino, after.st_mtime_ns, after.st_ctime_ns) == (before.st_ino, before.st_mtime_ns, before.st_ctime_ns)

    config_mod.save_env_value("TELEGRAM_HOME_CHANNEL", "67890")
    assert "TELEGRAM_HOME_CHANNEL=67890" in env_path.read_text(encoding="utf-8")
    assert os.stat(env_path).st_ino != before.st_ino
