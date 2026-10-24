"""Invariants for the KeePass (KeePassXC) vault backend (``keepassxc-cli``).

Same contracts as the 1Password / Bitwarden backends (``test_vault_backends.py``):
1. A locked database never prompts where nobody can answer (cron/headless) and never runs the CLI:
   browser_vault_list reports it under ``locked``, browser_vault_fill refuses.
2. The master password reaches ``keepassxc-cli`` on stdin only (its documented non-interactive
   channel) — never argv, never any environment. Listing keeps metadata only and binds the origin
   from the entry's URL field; a fill reads exactly one entry's password.
3. Without a configured ``database`` the backend is not a login source.
"""

from __future__ import annotations

import json
import os
import stat
from unittest.mock import patch

import pytest

from agent.vault_backends import unlock as unlock_mod
from agent.vault_backends.keepass import KeePassLoginBackend

_MASTER = "correct horse battery"
_SECRET = "plain sentence nobody would flag 7"

# A stand-in `keepassxc-cli` with the real CLI's shape: `<cmd> -q [-k key] <db> [args]`, master password
# read from stdin. It records argv, stdin and whether the master password appears anywhere in its env.
_FAKE_KPXC = r'''#!/usr/bin/env python3
import json, os, sys
MASTER = "correct horse battery"
argv = sys.argv[1:]
stdin = sys.stdin.read() if not sys.stdin.isatty() else ""
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kpxc.log"), "a") as log:
    log.write(json.dumps({"argv": argv, "stdin": stdin,
                          "env_has_master": any(MASTER in v for v in os.environ.values())}) + "\n")
if stdin.rstrip("\n") != MASTER:
    sys.stderr.write("Error while reading the database: Invalid credentials were provided\n"); sys.exit(1)
cmd, rest = argv[0], argv[3:]
if cmd == "db-info":
    print("Name: Test"); sys.exit(0)
if cmd == "export" and rest == ["-f", "csv"]:
    print('"Group","Title","Username","Password","URL","Notes","TOTP","Icon","Last Modified","Created"')
    print('"Root/Web","Example","jane@example.com","plain sentence nobody would flag 7","https://example.com/login","secret note","","0","",""')
    print('"Root","Bare host","bob","pw-2","shop.example.org","","","0","",""')
    print('"Root","No URL","carol","pw-3","","","","0","",""')
    print('"Root/Recycle Bin","Deleted","dave","pw-4","https://deleted.example.com","","","0","",""')
    sys.exit(0)
if cmd == "show" and rest[:3] == ["-s", "-a", "Password"]:
    print({"Web/Example": "plain sentence nobody would flag 7", "Bare host": "pw-2"}[rest[3]]); sys.exit(0)
sys.exit(2)
'''


pytestmark = pytest.mark.platforms("posix")  # fake keepassxc-cli is a shebang script; the backend is host-agnostic


@pytest.fixture
def fake_kpxc(tmp_path, monkeypatch):
    exe = tmp_path / "keepassxc-cli"
    exe.write_text(_FAKE_KPXC, encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    db = tmp_path / "test.kdbx"
    db.write_bytes(b"")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    unlock_mod.lock()
    backend = KeePassLoginBackend({"binary_path": str(exe), "database": str(db)})
    yield backend, tmp_path / "kpxc.log"
    unlock_mod.lock()


def _calls(log):
    return [json.loads(line) for line in log.read_text(encoding="utf-8-sig").splitlines()] if log.exists() else []


def _only(backend):
    return patch("agent.vault_backends.base.enabled_backends", return_value=[backend]), \
        patch("agent.vault_backends.enabled_backends", return_value=[backend])


def test_locked_database_is_reported_not_prompted_when_headless(fake_kpxc, monkeypatch):
    backend, log = fake_kpxc
    from tools.browser_vault_tool import browser_vault_fill, browser_vault_list

    monkeypatch.setenv("HERMES_CRON_SESSION", "1")  # headless: nobody can answer a prompt
    unlock_mod.set_unlock_prompt_callback(lambda *_: _MASTER)  # even a wired prompt must not fire
    a, b = _only(backend)
    try:
        with a, b:
            listed = json.loads(browser_vault_list())
            assert listed["items"] == []
            assert listed["locked"] == [{"backend": "keepass", "display_name": "KeePass",
                                         "unlock": "unavailable_in_this_session"}]
            filled = json.loads(browser_vault_fill("kp:Web/Example", task_id="t"))
            assert filled["success"] is False and filled["error_type"] == "unlock_unavailable"
    finally:
        unlock_mod.set_unlock_prompt_callback(None)
    assert _calls(log) == [], "keepassxc-cli must not run at all while locked in a headless session"
    assert not unlock_mod.is_unlocked("keepass")


def test_unlock_feeds_master_on_stdin_only_and_rejects_a_wrong_one(fake_kpxc):
    backend, log = fake_kpxc
    with pytest.raises(RuntimeError, match="wrong master password"):
        backend.unlock("wrong")
    assert not backend.is_unlocked()

    backend.unlock(_MASTER)
    assert backend.is_unlocked()
    backend.list_items()
    backend.resolve_password("kp:Web/Example")

    calls = _calls(log)
    assert [c["argv"][0] for c in calls] == ["db-info", "db-info", "export", "show"]
    assert all(c["stdin"] == _MASTER + "\n" for c in calls[1:])
    assert all(_MASTER not in " ".join(c["argv"]) for c in calls), "master password must never be argv"
    assert not any(c["env_has_master"] for c in calls), "master password must never be in the child env"
    assert all(_MASTER not in v for v in os.environ.values()), "master password must never touch our env"

    unlock_mod.lock("keepass")
    assert not backend.is_unlocked()


def test_listing_is_metadata_only_and_binds_origins_from_the_url_field(fake_kpxc):
    backend, _log = fake_kpxc
    assert backend.list_items() == []  # locked: nothing listed, nothing run
    backend.unlock(_MASTER)
    metas = {m.id: m for m in backend.list_items()}
    # No URL → not fillable anywhere; Recycle Bin entries are never offered.
    assert set(metas) == {"kp:Web/Example", "kp:Bare host"}
    ex = metas["kp:Web/Example"]
    assert (ex.label, ex.identifier, ex.origin, tuple(ex.allowed_origins)) == \
        ("Example", "jane@example.com", "https://example.com", ("https://example.com",))
    assert metas["kp:Bare host"].origin == "https://shop.example.org"  # scheme-less URL → https
    dumped = repr(list(metas.values()))
    assert _SECRET not in dumped and "secret note" not in dumped and "pw-2" not in dumped


def test_fill_reads_exactly_one_entry_password(fake_kpxc):
    backend, log = fake_kpxc
    from tools.browser_vault_tool import browser_vault_fill, browser_vault_list

    prompts = []

    def prompt(name, display):
        prompts.append((name, display))
        return _MASTER

    unlock_mod.set_unlock_prompt_callback(prompt)
    a, b = _only(backend)
    try:
        with a, b, \
             patch("tools.browser_vault_tool._current_page_origin", return_value="https://example.com"), \
             patch("tools.browser_vault_tool._eval_js", return_value={"success": True, "result": json.dumps([
                 {"tag": "input", "type": "password", "name": "password", "id": "pw", "autocomplete": "current-password",
                  "visible": True}])}), \
             patch("tools.browser_vault_tool._eval_js_secret", return_value={"success": True, "result": json.dumps(
                 {"filled": 1})}) as secret_eval:
            out = json.loads(browser_vault_fill("kp:Web/Example", task_id="t"))
            out.pop("next", None)
            assert out == {"success": True, "filled_fields": 1, "backend": "keepass", "kind": "login",
                           "origin": "https://example.com"}
            assert prompts == [("keepass", "KeePass")]
            assert _SECRET in secret_eval.call_args.args[1]
            listed = json.loads(browser_vault_list())
            assert [i["handle"] for i in listed["items"]] == ["kp:Web/Example", "kp:Bare host"]
            assert _SECRET not in json.dumps(listed)
    finally:
        unlock_mod.set_unlock_prompt_callback(None)

    shows = [c["argv"] for c in _calls(log) if c["argv"][0] == "show"]
    assert len(shows) == 1 and shows[0][-4:] == ["-s", "-a", "Password", "Web/Example"]


def test_not_a_login_source_without_a_database(tmp_path, monkeypatch):
    from agent.vault_backends import base

    exe = tmp_path / "keepassxc-cli"
    exe.write_text(_FAKE_KPXC, encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    with patch.object(base, "_cfg", return_value={"keepass": {"binary_path": str(exe)}}):
        assert not base.is_installed("keepass")
        assert not base.is_enabled("keepass")
        assert "keepass" not in {b.name for b in base.enabled_backends()}
    with patch.object(base, "_cfg", return_value={"keepass": {"binary_path": str(exe),
                                                              "database": str(tmp_path / "x.kdbx")}}):
        assert base.is_enabled("keepass")
        assert "keepass" in {b.name for b in base.enabled_backends()}
