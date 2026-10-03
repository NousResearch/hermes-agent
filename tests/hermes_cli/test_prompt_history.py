"""Credential-bearing prompts never reach the on-disk CLI prompt history."""

from __future__ import annotations

import os
import stat

import pytest

from hermes_cli.prompt_history import CredentialSafeFileHistory

SECRET_PROMPT = "set the key to sk-abcdefghijklmnopqrstuvwxyz0123456789ABCD please"
URL_PROMPT = "fetch https://user:hunter2@example.com/report"


def test_credential_prompts_are_neither_persisted_nor_recalled(tmp_path):
    path = tmp_path / "history"
    history = CredentialSafeFileHistory(str(path))
    history.load()
    history.append_string("fix the bug in main.py")
    history.append_string(SECRET_PROMPT)
    history.append_string(URL_PROMPT)
    history.append_string("now run the tests")

    on_disk = path.read_text(encoding="utf-8")
    assert "fix the bug in main.py" in on_disk and "now run the tests" in on_disk
    assert "sk-abcdef" not in on_disk and "hunter2" not in on_disk
    # In-process recall (Up arrow / auto-suggest) skips them too, not just the file.
    assert SECRET_PROMPT not in history.get_strings() and URL_PROMPT not in history.get_strings()
    # A fresh process sees only the safe prompts.
    assert list(CredentialSafeFileHistory(str(path)).load_history_strings()) == [
        "now run the tests", "fix the bug in main.py",
    ]


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_history_file_is_owner_only(tmp_path):
    path = tmp_path / "history"
    path.write_text("+older prompt\n", encoding="utf-8")
    path.chmod(0o644)  # what an older Hermes left behind under the umask default

    CredentialSafeFileHistory(str(path)).store_string("hello")

    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert "hello" in path.read_text(encoding="utf-8")
