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
from agent.vault_backends.base import UnlockRequired
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


# --- desktop/biometric (ambient) auth: `op signin --raw` exits 0 with empty stdout because the
# account is already authenticated through the 1Password desktop app / system-auth integration.
# `item list` (the liveness probe) succeeds without any OP_SESSION_* or service-account token.
_FAKE_OP_AMBIENT = r'''#!/usr/bin/env python3
import json, os, sys
log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "op.log"), "a")
argv = sys.argv[1:]
log.write(json.dumps({"argv": argv, "OP_SESSION": {k: v for k, v in os.environ.items() if k.startswith("OP_SESSION")},
                      "OP_SERVICE_ACCOUNT_TOKEN": os.environ.get("OP_SERVICE_ACCOUNT_TOKEN")}) + "\n")
if argv[:2] == ["signin", "--raw"]:
    sys.exit(0)  # ambient auth: success, no token printed
if argv[:2] == ["item", "list"]:
    if "--cache" in argv:
        sys.exit(0)  # liveness probe: rc is all that matters
    print(json.dumps([{"id": "abc", "title": "Example", "created_at": "2026-01-01T00:00:00Z",
                       "additional_information": "jane@example.com",
                       "urls": [{"href": "https://example.com/login"}]}])); sys.exit(0)
if argv[:2] == ["item", "get"]:
    print("plain sentence nobody would flag 7"); sys.exit(0)
sys.exit(2)
'''

# Same auth mode but the account is genuinely locked: `item list` also fails.
_FAKE_OP_LOCKED = r'''#!/usr/bin/env python3
import sys
argv = sys.argv[1:]
if argv[:2] == ["signin", "--raw"]:
    sys.exit(0)  # ambient signin reports success even though nothing is actually usable yet
if argv[:2] == ["item", "list"]:
    sys.stderr.write("[ERROR] 2026/01/01 00:00:00 you are not signed in\n"); sys.exit(1)
sys.exit(2)
'''


@pytest.fixture
def fake_op_ambient(tmp_path, monkeypatch):
    exe = tmp_path / "op"
    exe.write_text(_FAKE_OP_AMBIENT, encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "op.log"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
    unlock_mod.lock()
    yield exe, log
    unlock_mod.lock()


@pytest.fixture
def fake_op_locked(tmp_path, monkeypatch):
    exe = tmp_path / "op"
    exe.write_text(_FAKE_OP_LOCKED, encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("OP_SERVICE_ACCOUNT_TOKEN", raising=False)
    unlock_mod.lock()
    yield exe
    unlock_mod.lock()


def test_onepassword_ambient_desktop_auth_unlocks_and_lists_without_a_session_token(fake_op_ambient):
    """Biometric/desktop-app auth: signin --raw exits 0 with no token, but a read probe succeeds —
    that must count as unlocked, not `no session token`."""
    from agent.vault_backends.onepassword import OnePasswordLoginBackend

    exe, log = fake_op_ambient
    backend = OnePasswordLoginBackend({"enabled": True, "binary_path": str(exe)})

    assert backend.is_unlocked() is True
    backend.unlock("")  # must not raise
    items = backend.list_items()
    assert len(items) == 1
    assert items[0].label == "Example" and items[0].origin == "https://example.com"
    assert backend.resolve_password(items[0].id) == "plain sentence nobody would flag 7"

    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert all(c["OP_SERVICE_ACCOUNT_TOKEN"] is None for c in calls)
    assert all(not c["OP_SESSION"] for c in calls), "ambient auth must not need any OP_SESSION_* var"


def test_onepassword_ambient_probe_is_cached_not_respawned_per_call(fake_op_ambient):
    from agent.vault_backends.onepassword import OnePasswordLoginBackend

    exe, log = fake_op_ambient
    backend = OnePasswordLoginBackend({"enabled": True, "binary_path": str(exe)})

    assert backend.is_unlocked() is True
    backend.list_items()
    backend.list_items()

    calls = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    probe_calls = [c for c in calls if c["argv"][:2] == ["item", "list"] and "--cache" in c["argv"]]
    assert len(probe_calls) == 1, "the ambient liveness probe must be cached, not re-spawned every call"


def test_onepassword_ambient_auth_locked_raises_and_reports_locked(fake_op_locked):
    """Ambient signin exits 0 too, but the account is not actually usable: the probe fails and the
    genuine lock must still surface (no false 'unlocked')."""
    from agent.vault_backends.onepassword import OnePasswordLoginBackend

    exe = fake_op_locked
    backend = OnePasswordLoginBackend({"enabled": True, "binary_path": str(exe)})

    assert backend.is_unlocked() is False
    with pytest.raises(RuntimeError):
        backend.unlock("")
    with pytest.raises(UnlockRequired):
        backend._run("item", "list", "--categories", "Login", "--format", "json")


def test_onepassword_service_account_and_session_token_paths_still_work(monkeypatch, tmp_path):
    """Regression: the two pre-existing auth modes are untouched by the ambient-probe addition."""
    from agent.vault_backends.onepassword import OnePasswordLoginBackend

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    unlock_mod.lock()

    # Service account: is_unlocked() is True from the token alone, no `op` invocation needed.
    backend = OnePasswordLoginBackend({"enabled": True})
    backend._service_token = "svc-token"
    with patch("agent.vault_backends.onepassword.run_cli") as run_cli_mock:
        assert backend.is_unlocked() is True
        run_cli_mock.assert_not_called()

    # Session token: unlock() stores a real token and is_unlocked() reflects it, independent of probing.
    unlock_mod.lock()
    backend2 = OnePasswordLoginBackend({"enabled": True})
    with patch.object(backend2, "_op", return_value=Path("/usr/bin/op")), \
         patch("agent.vault_backends.onepassword.run_with_stdin_secret") as signin_mock:
        signin_mock.return_value = type("P", (), {"returncode": 0, "stdout": "SESSION-TOKEN-XYZ\n", "stderr": ""})()
        backend2.unlock("correct horse")
    assert backend2.is_unlocked() is True
    unlock_mod.lock()
