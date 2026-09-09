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
                      "pw": os.environ.get(pw_env) if pw_env else None}) + "\n")
if argv[:2] == ["unlock", "--raw"]:
    if pw_env is None:
        sys.stderr.write("Master password is required. Try again in interactive mode or provide a password file or environment variable.\n"); sys.exit(1)
    if os.environ.get(pw_env) != "correct horse":
        sys.stderr.write("Invalid master password.\n"); sys.exit(1)
    print("SESSION-TOKEN-123"); sys.exit(0)
if os.environ.get("BW_SESSION") != "SESSION-TOKEN-123":
    sys.stderr.write("Vault is locked.\n"); sys.exit(1)
if argv[:2] == ["list", "items"]:
    print(json.dumps([{"id": "abc", "type": 1, "name": "Example", "creationDate": "2026-01-01T00:00:00Z",
                       "login": {"username": "jane@example.com", "uris": [{"uri": "https://example.com/login"}]}},
                      {"id": "note", "type": 2, "name": "Secure note"}])); sys.exit(0)
if argv[:2] == ["get", "password"]:
    print("plain sentence nobody would flag 7"); sys.exit(0)
sys.exit(2)
'''


pytestmark = pytest.mark.skipif(os.name == "nt", reason="fake bw is a shebang script; the backend under test is host-agnostic")


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
