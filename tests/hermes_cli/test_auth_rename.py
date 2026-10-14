"""`hermes auth rename`: relabel one pooled credential without touching its secrets (#115558)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent.credential_pool import PooledCredential, load_pool, write_credential_pool
from hermes_cli import auth_commands
from hermes_cli.auth import _auth_file_path, _load_auth_store, _save_auth_store, read_credential_pool

PROVIDER = "openai-codex"


def _row(id_: str, label: str, priority: int = 0, source: str = "manual:device_code") -> dict:
    return {"id": id_, "label": label, "auth_type": "oauth", "priority": priority, "source": source,
            "access_token": f"access-{id_}", "refresh_token": f"refresh-{id_}"}


def _seed(*rows: dict) -> None:
    write_credential_pool(PROVIDER, [PooledCredential.from_dict(PROVIDER, r).to_dict() for r in rows])


def _rename(target: str, new_label) -> None:
    auth_commands.auth_rename_command(SimpleNamespace(provider=PROVIDER, target=target, new_label=new_label))


def _disk_rows() -> dict[str, dict]:
    return {row["id"]: row for row in read_credential_pool(PROVIDER)}


@pytest.mark.parametrize("target", ["2", "bbb222", "codex-2", "CODEX-2"])
def test_rename_changes_only_the_label_of_the_targeted_entry(target):
    _seed(_row("aaa111", "codex-1"), _row("bbb222", "codex-2", priority=1))
    before = _disk_rows()

    _rename(target, "  work seat  ")

    after = _disk_rows()
    assert after["bbb222"] == {**before["bbb222"], "label": "work seat"}
    assert after["aaa111"] == before["aaa111"]
    assert [e.label for e in load_pool(PROVIDER).entries()] == ["codex-1", "work seat"]


def test_rename_keeps_a_token_another_process_rotated_meanwhile():
    _seed(_row("aaa111", "codex-1"))
    pool = load_pool(PROVIDER)
    store = _load_auth_store()
    store["credential_pool"][PROVIDER][0].update(access_token="access-rotated", refresh_token="refresh-rotated")
    _save_auth_store(store)

    pool.rename_entry("aaa111", "work")

    row = _disk_rows()["aaa111"]
    assert (row["label"], row["access_token"], row["refresh_token"]) == (
        "work", "access-rotated", "refresh-rotated")


@pytest.mark.parametrize("bad_label", [
    None, "", "   ", "line\nbreak", "nul\x00", "ansi\x1b[31m", "bidi\u202eflip", "zero\u200bwidth",
    "2", " 7 ", "x" * 129,
    "codex-2", "Codex-2", "bbb222",  # would make another entry's label or id ambiguous
])
def test_rejected_label_leaves_the_store_byte_for_byte(bad_label):
    _seed(_row("aaa111", "codex-1"), _row("bbb222", "codex-2", priority=1))
    before = _auth_file_path().read_bytes()

    with pytest.raises(SystemExit):
        _rename("1", bad_label)

    assert _auth_file_path().read_bytes() == before


def test_same_label_is_a_no_op_without_a_write(capsys):
    _seed(_row("aaa111", "codex-1"))
    before = _auth_file_path().read_bytes()

    _rename("1", "codex-1")

    assert _auth_file_path().read_bytes() == before
    assert "nothing to change" in capsys.readouterr().out


def test_unknown_target_exits_without_writing():
    _seed(_row("aaa111", "codex-1"))
    before = _auth_file_path().read_bytes()

    with pytest.raises(SystemExit, match="No credential"):
        _rename("nope", "work")

    assert _auth_file_path().read_bytes() == before


def test_old_label_with_control_characters_is_echoed_escaped(capsys):
    _seed(_row("aaa111", "evil\x1b]0;pwned\x07"))

    _rename("1", "clean")

    out = capsys.readouterr().out
    assert "\x1b" not in out and "\\x1b" in out and '"clean"' in out


def _write_codex_login(label: str | None) -> None:
    store = _load_auth_store()
    state = {"tokens": {"access_token": "access-login", "refresh_token": "refresh-login"},
             "auth_mode": "chatgpt"}
    if label is not None:
        state["label"] = label
    store.setdefault("providers", {})[PROVIDER] = state
    _save_auth_store(store)


def _login_state() -> dict:
    return _load_auth_store()["providers"][PROVIDER]


@pytest.mark.parametrize(("stored", "expected"), [
    ("my-login", "renamed"),       # still the old label -> follows the rename
    (None, "renamed"),             # token-derived label -> the rename becomes the stored one
    ("hand-edited", "hand-edited"),  # diverged by the user -> left alone
])
def test_login_singleton_label_follows_a_device_code_rename(stored, expected):
    _write_codex_login(stored)
    pool = load_pool(PROVIDER)
    entry = next(e for e in pool.entries() if e.source == "device_code")
    if stored == "hand-edited":
        pool.rename_entry(entry.id, "my-login")  # pool row and login now disagree
    tokens_before = _login_state()["tokens"]

    _rename(entry.id, "renamed")

    assert _login_state()["label"] == expected
    assert _login_state()["tokens"] == tokens_before


def test_independent_account_rename_never_touches_the_login_singleton():
    _write_codex_login("shared-name")
    _seed(_row("aaa111", "other-account"))
    before = _login_state()

    _rename("aaa111", "renamed")

    assert _login_state() == before
