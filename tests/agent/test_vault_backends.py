"""Invariants for external password-manager vault backends (1Password / Bitwarden).

Two contracts that must never regress:
1. A locked manager never prompts where nobody can answer (cron/headless) and never leaks a
   value: browser_vault_list reports it under ``locked``, browser_vault_fill refuses.
2. The unlock path hands the master password to the manager CLI through its documented
   non-interactive channel (bw: ``--passwordenv`` on the CHILD env only — never argv, never our
   process env), keeps just the session token in memory scoped to the profile, and a fill then
   routes by handle prefix through the real subprocess path. Locking forgets the token.
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from unittest.mock import patch

import pytest

from agent.vault_backends import unlock as unlock_mod
from agent.vault_backends.bitwarden import BitwardenLoginBackend

# A stand-in `bw` that mimics the three commands the backend uses and the real CLI's password contract
# (bw 2026.x rejects a piped password: "Master password is required"; it reads --passwordenv <VAR>).
# It records argv + stdin + the named env var so the test can prove where the master password travelled.
# (No env passthrough: the backend's allowlisted child env is part of what is under test.)
_FAKE_BW = r'''#!/usr/bin/env python3
import json, os, sys
log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bw.log"), "a")
argv = sys.argv[1:]
stdin = sys.stdin.read() if not sys.stdin.isatty() else ""
pw_env = argv[argv.index("--passwordenv") + 1] if "--passwordenv" in argv else None
log.write(json.dumps({"argv": argv, "stdin": stdin, "BW_SESSION": os.environ.get("BW_SESSION"),
                      "account": os.environ.get("BITWARDENCLI_APPDATA_DIR"),
                      "home": os.environ.get("HOME"), "xdg": os.environ.get("XDG_CONFIG_HOME"),
                      "pw": os.environ.get(pw_env) if pw_env else None}) + "\n")
if argv[:2] == ["unlock", "--raw"]:
    if pw_env is None:
        sys.stderr.write("Master password is required. Try again in interactive mode or provide a password file or environment variable.\n"); sys.exit(1)
    if os.environ.get(pw_env) != "correct horse":
        sys.stderr.write("Invalid master password.\n"); sys.exit(1)
    print("SESSION-TOKEN-123"); sys.exit(0)
account = os.environ.get("BITWARDENCLI_APPDATA_DIR")
marker = os.path.join(account, "session") if account else ""
expected = open(marker).read() if marker and os.path.isfile(marker) else "SESSION-TOKEN-123"
if os.environ.get("BW_SESSION") != expected:
    sys.stderr.write("Vault is locked.\n"); sys.exit(1)
if argv[:2] == ["list", "items"]:
    print(json.dumps([{"id": "abc-" + os.path.basename(os.path.dirname(account)) if marker and os.path.isfile(marker) else "abc", "type": 1, "name": "Example", "creationDate": "2026-01-01T00:00:00Z",
                       "login": {"username": "jane@example.com", "uris": [{"uri": "https://example.com/login"}]}},
                      {"id": "note", "type": 2, "name": "Secure note"}])); sys.exit(0)
if argv[:2] == ["get", "password"]:
    print("plain sentence nobody would flag 7"); sys.exit(0)
sys.exit(2)
'''


pytestmark = pytest.mark.platforms("posix")  # fake bw is a shebang script; the backend under test is host-agnostic


@pytest.fixture
def fake_bw(tmp_path, monkeypatch):
    exe = tmp_path / "bw"
    exe.write_text(_FAKE_BW, encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "bw.log"  # the backend runs bw with an allowlisted env, so the fake logs beside itself
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    unlock_mod.lock()
    yield exe, log
    unlock_mod.lock()


def _enabled(exe):
    """The tool late-imports ``enabled_backends`` from the package facade; ``backend_for_handle`` reads
    the sibling's own binding — patch both so the fake is the only backend anywhere."""
    backend = BitwardenLoginBackend({"enabled": True, "binary_path": str(exe)})
    return patch("agent.vault_backends.base.enabled_backends", return_value=[backend]), backend


def test_locked_manager_is_reported_not_prompted_when_headless(fake_bw, monkeypatch):
    exe, log = fake_bw
    from tools.browser_vault_tool import browser_vault_fill, browser_vault_list

    patcher, backend = _enabled(exe)
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")  # headless: nobody can answer a prompt
    unlock_mod.set_unlock_prompt_callback(lambda *_: "correct horse")  # even a wired prompt must not fire
    try:
        with patcher, patch("agent.vault_backends.enabled_backends", return_value=[backend]):
            listed = json.loads(browser_vault_list())
            assert listed["items"] == []
            assert listed["locked"] == [{"backend": "bitwarden", "display_name": "Bitwarden",
                                         "unlock": "unavailable_in_this_session"}]
            filled = json.loads(browser_vault_fill("bw:abc", task_id="t"))
            assert filled["success"] is False and filled["error_type"] == "unlock_unavailable"
    finally:
        unlock_mod.set_unlock_prompt_callback(None)
    assert not log.exists(), "bw must not be invoked at all while locked in a headless session"
    assert not unlock_mod.is_unlocked("bitwarden")


def test_opted_in_bitwarden_helper_unlocks_headless_with_profile_isolation(fake_bw, tmp_path, monkeypatch):
    """An explicit unattended helper opens only its own profile's vault; no prompt or token reaches tools."""
    exe, log = fake_bw
    helper = tmp_path / "unlock-helper"
    helper.write_text("#!/usr/bin/env python3\nimport os\nprint('SESSION-TOKEN-123' if os.environ['HERMES_HOME'].endswith('profile-a') else '')\n")
    helper.chmod(0o700)
    from tools.browser_vault_tool import browser_vault_list

    backend = BitwardenLoginBackend({"enabled": True, "binary_path": str(exe),
                                    "unattended_helper": str(helper)})
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    unlock_mod.set_unlock_prompt_callback(None)
    with patch("agent.vault_backends.enabled_backends", return_value=[backend]):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))
        listed = json.loads(browser_vault_list())
        assert listed["items"][0]["handle"] == "bw:abc"
        assert "SESSION-TOKEN-123" not in json.dumps(listed)
        assert backend.is_unlocked()
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-b"))
        listed_b = json.loads(browser_vault_list())
        assert not listed_b["items"] and listed_b["errors"]
        assert not backend.is_unlocked()
        monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile-a"))
        assert backend.is_unlocked()
        unlock_mod.lock("bitwarden")
    assert not backend.is_unlocked()


def test_helper_and_bw_use_distinct_account_data_across_profiles(fake_bw, tmp_path, monkeypatch):
    exe, _ = fake_bw
    helper = tmp_path / "account-helper"
    helper.write_text("#!/usr/bin/env python3\nimport os\n"
                      "from pathlib import Path\n"
                      "print((Path(os.environ['BITWARDENCLI_APPDATA_DIR']) / 'session').read_text())\n")
    helper.chmod(0o700)
    from tools.browser_vault_tool import browser_vault_list
    backend = BitwardenLoginBackend({"binary_path": str(exe), "unattended_helper": str(helper)})
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    monkeypatch.setenv("BITWARDENCLI_APPDATA_DIR", str(tmp_path / "launch-account"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "launch-config"))
    monkeypatch.setenv("HOME", str(tmp_path / "launch-home"))
    for name in ("a", "b"):
        account = tmp_path / "profiles" / name / "bitwarden-cli"
        account.mkdir(parents=True)
        (account / "session").write_text("SESSION-A" if name == "a" else "SESSION-B")
    with patch("agent.vault_backends.enabled_backends", return_value=[backend]):
        for name in ("a", "b", "a"):
            monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profiles" / name))
            result = json.loads(browser_vault_list())
            assert result["items"][0]["handle"] == "bw:abc-" + name
            assert backend.is_unlocked()
            assert "SESSION-" not in json.dumps(result)
    calls = [json.loads(line) for line in (tmp_path / "bw.log").read_text().splitlines()]
    assert [c["BW_SESSION"] for c in calls] == ["SESSION-A", "SESSION-B", "SESSION-A"]
    assert [c["account"] for c in calls] == [str(tmp_path / "profiles" / name / "bitwarden-cli") for name in ("a", "b", "a")]
    assert [c["home"] for c in calls] == [str(tmp_path / "profiles" / name) for name in ("a", "b", "a")]
    assert [c["xdg"] for c in calls] == [str(tmp_path / "profiles" / name / ".config") for name in ("a", "b", "a")]


def test_default_profile_does_not_inherit_named_launch_bitwarden_account(fake_bw, tmp_path, monkeypatch):
    exe, log = fake_bw
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override
    native_home = Path.home()
    default_home = native_home / ".hermes"
    named_home = default_home / "profiles" / "other"
    monkeypatch.setenv("HERMES_HOME", str(named_home))
    monkeypatch.setenv("HOME", str(named_home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(named_home / ".config"))
    monkeypatch.setenv("BITWARDENCLI_APPDATA_DIR", str(named_home / "bitwarden-cli"))
    token = set_hermes_home_override(default_home)
    try:
        backend = BitwardenLoginBackend({"binary_path": str(exe)})
        # A default-profile prompt must consult the default account, never the launch account.
        backend.unlock("correct horse")
        backend.list_items()
        calls = [json.loads(line) for line in log.read_text().splitlines()]
        assert all(c["home"] == str(native_home) for c in calls)
        assert all(c["account"] is None for c in calls)
        assert all(c["xdg"] == str(native_home / ".config") for c in calls)
    finally:
        unlock_mod.lock("bitwarden")
        reset_hermes_home_override(token)


def test_untrusted_helper_paths_are_rejected(fake_bw, tmp_path, monkeypatch):
    exe, _ = fake_bw
    helper = tmp_path / "helper"
    helper.write_text("#!/bin/sh\nprintf token\n")
    helper.chmod(0o700)
    backend = BitwardenLoginBackend({"binary_path": str(exe), "unattended_helper": str(helper)})
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o777)
    unsafe.chmod(0o777)
    (unsafe / "helper").write_text(helper.read_text())
    (unsafe / "helper").chmod(0o700)
    backend.cfg["unattended_helper"] = str(unsafe / "helper")
    with pytest.raises(RuntimeError, match="helper"):
        backend.unlock_unattended()
    link = tmp_path / "link"
    link.symlink_to(helper)
    backend.cfg["unattended_helper"] = str(link)
    with pytest.raises(RuntimeError, match="helper"):
        backend.unlock_unattended()


def test_helper_path_swap_cannot_replace_checked_executable(fake_bw, tmp_path, monkeypatch):
    exe, _ = fake_bw
    helper = tmp_path / "helper"
    replacement = tmp_path / "replacement"
    helper.write_text("#!/bin/sh\nprintf SESSION-TOKEN-123\n")
    replacement.write_text("#!/bin/sh\nprintf WRONG-TOKEN\n")
    helper.chmod(0o700)
    replacement.chmod(0o700)
    backend = BitwardenLoginBackend({"binary_path": str(exe), "unattended_helper": str(helper)})
    real_run = __import__("subprocess").run

    def swapped_run(*args, **kwargs):
        helper.rename(tmp_path / "old-helper")
        replacement.rename(helper)
        return real_run(*args, **kwargs)

    with patch("agent.vault_backends.bitwarden.subprocess.run", side_effect=swapped_run):
        assert backend.unlock_unattended()
    assert backend.is_unlocked()


def test_helper_owned_by_another_user_is_rejected(fake_bw, tmp_path):
    exe, _ = fake_bw
    helper = tmp_path / "helper"
    helper.write_text("#!/bin/sh\nprintf SESSION-TOKEN-123\n")
    helper.chmod(0o700)
    backend = BitwardenLoginBackend({"binary_path": str(exe), "unattended_helper": str(helper)})
    real_fstat = os.fstat

    def other_owner(fd):
        info = real_fstat(fd)
        if info.st_ino == helper.stat().st_ino:
            values = list(info)
            values[4] = os.geteuid() + 1
            return os.stat_result(values)
        return info

    with patch("agent.vault_backends.bitwarden.os.fstat", side_effect=other_owner):
        with pytest.raises(RuntimeError, match="helper"):
            backend.unlock_unattended()


def test_broken_helper_reports_lock_and_masked_prompt_fallback(fake_bw, tmp_path, monkeypatch):
    exe, _ = fake_bw
    helper = tmp_path / "broken-helper"
    helper.write_text("#!/bin/sh\nexit 1\n")
    helper.chmod(0o700)
    backend = BitwardenLoginBackend({"binary_path": str(exe), "unattended_helper": str(helper)})
    from tools.browser_vault_tool import browser_vault_list, browser_vault_unlock
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    prompts = []
    unlock_mod.set_unlock_prompt_callback(lambda *_: prompts.append(True) or "correct horse")
    try:
        with patch("agent.vault_backends.enabled_backends", return_value=[backend]):
            result = json.loads(browser_vault_list())
            assert result["locked"][0]["unlock"] == "browser_vault_unlock"
            assert json.loads(browser_vault_unlock("bitwarden"))["success"] is True
            assert prompts == [True]
    finally:
        unlock_mod.set_unlock_prompt_callback(None)


def test_broken_helper_headless_still_reports_locked(fake_bw, tmp_path, monkeypatch):
    exe, _ = fake_bw
    helper = tmp_path / "broken-helper"
    helper.write_text("#!/bin/sh\nexit 1\n")
    helper.chmod(0o700)
    backend = BitwardenLoginBackend({"binary_path": str(exe), "unattended_helper": str(helper)})
    from tools.browser_vault_tool import browser_vault_list, browser_vault_unlock
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    with patch("agent.vault_backends.enabled_backends", return_value=[backend]):
        listed = json.loads(browser_vault_list())
        assert listed["locked"][0]["unlock"] == "unavailable_in_this_session"
        assert listed["errors"]
        assert json.loads(browser_vault_unlock("bitwarden"))["error_type"] == "unlock_unavailable"


def test_unlock_uses_vendor_passwordenv_contract_then_fill_routes_by_prefix(fake_bw):
    exe, log = fake_bw
    from tools.browser_vault_tool import browser_vault_fill, browser_vault_list

    patcher, backend = _enabled(exe)
    prompts = []

    def prompt(name, display):
        prompts.append((name, display))
        return "correct horse"

    unlock_mod.set_unlock_prompt_callback(prompt)
    try:
        with patcher, patch("agent.vault_backends.enabled_backends", return_value=[backend]), \
             patch("tools.browser_vault_tool._current_page_origin", return_value="https://example.com"), \
             patch("tools.browser_vault_tool._eval_js", return_value={"success": True, "result": json.dumps([
                 {"tag": "input", "type": "password", "name": "password", "id": "pw", "autocomplete": "current-password",
                  "visible": True}])}), \
             patch("tools.browser_vault_tool._eval_js_secret", return_value={"success": True, "result": json.dumps(
                 {"filled": 1})}) as secret_eval:
            out = json.loads(browser_vault_fill("bw:abc", task_id="t"))
            out.pop("next")
            assert out == {"success": True, "filled_fields": 1, "backend": "bitwarden", "kind": "login",
                           "origin": "https://example.com"}
            assert prompts == [("bitwarden", "Bitwarden")]
            # Now unlocked: listing exposes metadata only, never the password.
            listed = json.loads(browser_vault_list())
            assert listed["items"][0]["handle"] == "bw:abc"
            assert listed["items"][0]["identifier"] == "jane@example.com"
            assert "plain sentence nobody would flag 7" not in json.dumps(listed)
            # The password reached the fill script, and only there.
            assert "plain sentence nobody would flag 7" in secret_eval.call_args.args[1]
    finally:
        unlock_mod.set_unlock_prompt_callback(None)

    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    unlock_calls = [c for c in calls if c["argv"][:2] == ["unlock", "--raw"]]
    assert len(unlock_calls) == 1 and unlock_calls[0]["pw"] == "correct horse" and unlock_calls[0]["stdin"] == ""
    assert all("correct horse" not in " ".join(c["argv"]) for c in calls), "master password must never be argv"
    assert all(c["BW_SESSION"] == "SESSION-TOKEN-123" for c in calls if c["argv"][0] != "unlock")
    assert "HERMES_BW_MASTER" not in os.environ, "master password env var is child-only"

    # Tokens are profile-scoped: another HERMES_HOME sees the manager locked and cannot lock ours.
    other = str(exe.parent / "other-profile")
    with patch.dict(os.environ, {"HERMES_HOME": other}):
        assert not backend.is_unlocked()
        unlock_mod.lock("bitwarden")
    assert backend.is_unlocked()

    unlock_mod.lock("bitwarden")
    assert not backend.is_unlocked()
    assert os.environ.get("BW_SESSION") is None, "session token must never touch the process env"


def test_lock_during_unlock_wins_and_only_the_owning_session_release_drops_a_token(fake_bw, monkeypatch):
    """A Lock acknowledged while `bw unlock` is still running must not be undone when the child returns;
    a session teardown releases only the tokens that session unlocked."""
    exe, _log = fake_bw
    patcher, backend = _enabled(exe)
    with patcher:
        # Lock races the in-flight unlock: the generation moved, so the late token is discarded.
        gen = unlock_mod.begin_unlock("bitwarden")
        unlock_mod.lock("bitwarden")
        assert unlock_mod.store_session_token("bitwarden", "LATE-TOKEN", gen) is False
        assert not backend.is_unlocked()

        unlock_mod.set_current_session_id("sess-A")
        backend.unlock("correct horse")
        assert backend.is_unlocked()
        unlock_mod.release_session("sess-B")  # an unrelated sibling session ends
        assert backend.is_unlocked()
        unlock_mod.release_session("sess-A")
        assert not backend.is_unlocked()
        unlock_mod.set_current_session_id(None)


def test_bitwarden_multi_uri_item_binds_every_saved_web_origin():
    """A Bitwarden login with several URIs binds all of them (deduped, first stays
    primary); non-web URIs and URIs marked match=Never (5) never widen the fill set."""
    backend = BitwardenLoginBackend({"enabled": True})
    items_json = json.dumps([{
        "id": "multi", "type": 1, "name": "Amazon", "creationDate": "2026-01-01T00:00:00Z",
        "login": {"username": "jane@example.com", "uris": [
            {"uri": "https://amazon.co.uk/signin"},
            {"uri": "https://www.amazon.co.uk"},
            {"uri": "https://eu.account.amazon.com"},
            {"uri": "androidapp://com.amazon.shopping"},
            {"uri": "not a url"},
            {"uri": "https://never.amazon.co.uk", "match": 5},
        ]},
    }])
    with patch.object(BitwardenLoginBackend, "is_unlocked", return_value=True), \
         patch.object(backend, "_run", return_value=items_json):
        metas = backend.list_items()
    assert len(metas) == 1
    assert metas[0].origin == "https://amazon.co.uk"
    assert list(metas[0].allowed_origins) == ["https://amazon.co.uk", "https://www.amazon.co.uk",
                                              "https://eu.account.amazon.com"]


def test_onepassword_multi_url_item_binds_every_saved_web_origin():
    """A 1Password login with several websites binds all of them; the app URI is kept
    out of the fill set and a single-URL item is unchanged."""
    from agent.vault_backends.onepassword import OnePasswordLoginBackend, _all_origins, _web_origins

    backend = OnePasswordLoginBackend({"enabled": True})
    items_json = json.dumps([{
        "id": "multi", "title": "Amazon", "created_at": "2026-01-01T00:00:00Z",
        "additional_information": "jane@example.com",
        "urls": [{"href": "https://amazon.co.uk"}, {"href": "https://www.amazon.co.uk"},
                 {"href": "https://eu.account.amazon.com"}, {"href": "androidapp://com.amazon.shopping"},
                 {"href": "::not parseable::"}],
    }, {
        "id": "single", "title": "Shop", "created_at": "2026-01-01T00:00:00Z",
        "urls": [{"href": "https://shop.example.com"}],
    }])
    with patch.object(OnePasswordLoginBackend, "is_unlocked", return_value=True), \
         patch.object(backend, "_run", return_value=items_json):
        metas = {m.id: m for m in backend.list_items()}
    assert metas["op:multi"].origin == "https://amazon.co.uk"
    assert list(metas["op:multi"].allowed_origins) == ["https://amazon.co.uk", "https://www.amazon.co.uk",
                                                       "https://eu.account.amazon.com"]
    assert metas["op:single"].origin == "https://shop.example.com"
    assert list(metas["op:single"].allowed_origins) == ["https://shop.example.com"]
    # helpers: dedupe keeps first occurrence; app-only items keep their single origin
    assert _all_origins(["https://a.com/x", "https://a.com/y"]) == ["https://a.com"]
    assert _web_origins(["androidapp://com.x"]) == ("androidapp://com.x",)


def test_onepassword_backend_env_forwards_config_directory(monkeypatch):
    """Vault reads use the same explicit 1Password CLI config location."""
    from agent.vault_backends.onepassword import OnePasswordLoginBackend

    monkeypatch.setenv("OP_CONFIG_DIR", "/tmp/op-config")
    backend = OnePasswordLoginBackend({"enabled": True})

    assert backend._env(None)["OP_CONFIG_DIR"] == "/tmp/op-config"
