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
import time
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

    calls = [json.loads(line) for line in log.read_text(encoding="utf-8-sig").splitlines()]
    unlock_calls = [c for c in calls if c["argv"][:2] == ["unlock", "--raw"]]
    # Two unlock attempts: the secretless desktop-approval probe first (bare argv, no password env, no
    # --nointeraction), then the documented --passwordenv contract once the CLI cannot approve it.
    assert [c["pw"] for c in unlock_calls] == [None, "correct horse"]
    assert unlock_calls[0]["argv"] == ["unlock", "--raw"]
    assert unlock_calls[1]["stdin"] == ""
    assert prompts == [("bitwarden", "Bitwarden")], "the prompt is the fallback, and it fires once"
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


# A `bw` with Bitwarden's desktop-biometric path (Desktop 2026.9.0+ / CLI with the
# `biometrics-sdk-ipc` release): a bare `unlock --raw`, with no password anywhere, is approved by the
# running desktop app and returns the session key. Every later call must carry that key as BW_SESSION.
_FAKE_BW_BIOMETRIC = r'''#!/usr/bin/env python3
import json, os, sys
log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bw.log"), "a")
argv = sys.argv[1:]
log.write(json.dumps({"argv": argv, "stdin": sys.stdin.read() if not sys.stdin.isatty() else "",
                      "BW_NOINTERACTION": os.environ.get("BW_NOINTERACTION"),
                      "BW_SESSION": os.environ.get("BW_SESSION")}) + "\n")
if argv == ["unlock", "--raw"]:
    print("BIOMETRIC-TOKEN-1"); sys.exit(0)
if os.environ.get("BW_SESSION") != "BIOMETRIC-TOKEN-1":
    sys.stderr.write("Vault is locked.\n"); sys.exit(1)
if argv[:2] == ["list", "items"]:
    print(json.dumps([{"id": "abc", "type": 1, "name": "Example", "creationDate": "2026-01-01T00:00:00Z",
                       "login": {"username": "jane@example.com", "uris": [{"uri": "https://example.com/login"}]}}]))
    sys.exit(0)
if argv[:2] == ["get", "password"]:
    print("plain sentence nobody would flag 7"); sys.exit(0)
sys.exit(2)
'''


@pytest.fixture
def fake_bw_biometric(tmp_path, monkeypatch):
    exe = tmp_path / "bw"
    exe.write_text(_FAKE_BW_BIOMETRIC, encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "bw.log"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    unlock_mod.lock()
    yield exe, log
    unlock_mod.lock()


def test_desktop_biometric_unlock_needs_no_prompt(fake_bw_biometric):
    """A CLI that can approve through the desktop app unlocks with `bw unlock --raw` alone: the
    master-password prompt never fires and the session key still arrives as BW_SESSION."""
    exe, log = fake_bw_biometric
    from tools.browser_vault_tool import browser_vault_fill

    patcher, backend = _enabled(exe)
    prompts = []
    unlock_mod.set_unlock_prompt_callback(lambda name, display: prompts.append(name) or "correct horse")
    try:
        with patcher, patch("agent.vault_backends.enabled_backends", return_value=[backend]), \
             patch("tools.browser_vault_tool._current_page_origin", return_value="https://example.com"), \
             patch("tools.browser_vault_tool._eval_js", return_value={"success": True, "result": json.dumps([
                 {"tag": "input", "type": "password", "name": "password", "id": "pw",
                  "autocomplete": "current-password", "visible": True}])}), \
             patch("tools.browser_vault_tool._eval_js_secret", return_value={"success": True, "result": json.dumps(
                 {"filled": 1})}) as secret_eval:
            out = json.loads(browser_vault_fill("bw:abc", task_id="t"))
            assert out["success"] is True and out["filled_fields"] == 1
            assert "plain sentence nobody would flag 7" in secret_eval.call_args.args[1]
    finally:
        unlock_mod.set_unlock_prompt_callback(None)

    assert prompts == [], "an approved desktop unlock must not ask for the master password"
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8-sig").splitlines()]
    unlock_calls = [c for c in calls if c["argv"][:2] == ["unlock", "--raw"]]
    assert len(unlock_calls) == 1, "one attempt, and no password fallback once it succeeded"
    # The two conditions that disable the CLI's biometric path must never be on the probe.
    assert unlock_calls[0]["argv"] == ["unlock", "--raw"]
    assert unlock_calls[0]["BW_NOINTERACTION"] is None
    assert all(c["BW_SESSION"] == "BIOMETRIC-TOKEN-1" for c in calls if c["argv"][0] != "unlock")


@pytest.mark.parametrize("env", [
    {"HERMES_CRON_SESSION": "1"},
    {"HERMES_SINGLE_QUERY_SESSION": "1"},
    {"HERMES_SESSION_PLATFORM": "webhook"},
    {"HERMES_SESSION_PLATFORM": "msgraph_webhook"},
    {"HERMES_SESSION_PLATFORM": "api_server"},
])
def test_desktop_biometric_is_never_attempted_where_nobody_can_approve(fake_bw_biometric, monkeypatch, env):
    """Every unattended context refuses, at the public backend entry point as well as at the tool: no
    `bw` runs, so no desktop approval prompt can be raised for nobody. Cron, single-query and the three
    programmatic platforms are the contexts `can_prompt_here()` treats as unattended; a registered
    prompt callback must not be enough on its own."""
    exe, log = fake_bw_biometric
    from agent.vault_backends.unlock import secretless_unlock
    from tools.browser_vault_tool import browser_vault_unlock

    patcher, backend = _enabled(exe)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    unlock_mod.set_unlock_prompt_callback(lambda *_: "correct horse")  # wired, but nobody is there
    try:
        with patcher, patch("agent.vault_backends.enabled_backends", return_value=[backend]):
            # The gate is enforced inside the backend, so a direct caller cannot bypass it either.
            assert backend.try_secretless_unlock() is False
            assert secretless_unlock(backend) is False
            out = json.loads(browser_vault_unlock("bitwarden"))
            assert out["success"] is False and out["error_type"] == "unlock_unavailable"
    finally:
        unlock_mod.set_unlock_prompt_callback(None)
    assert not log.exists(), "no approval prompt may be raised headless"
    assert not unlock_mod.is_unlocked("bitwarden")


def test_desktop_biometric_can_be_turned_off(fake_bw_biometric):
    """`vault.bitwarden.desktop_biometric: false` restores the prompt-only path exactly."""
    exe, log = fake_bw_biometric
    backend = BitwardenLoginBackend({"enabled": True, "binary_path": str(exe), "desktop_biometric": False})
    unlock_mod.set_unlock_prompt_callback(lambda *_: "correct horse")
    try:
        assert backend.try_secretless_unlock() is False
    finally:
        unlock_mod.set_unlock_prompt_callback(None)
    assert not log.exists(), "the opt-out must not spawn bw at all"


# A `bw` that hangs on the secretless probe, the shape of a wedged desktop app: it spawns the helper it
# would talk to and waits. The documented --passwordenv contract still works, so the timeout can be
# shown to fall back into a real unlock. The helper pid is written out so the cleanup can be checked.
_FAKE_BW_HANG = r'''#!/usr/bin/env python3
import json, os, sys, time
here = os.path.dirname(os.path.abspath(__file__))
log = open(os.path.join(here, "bw.log"), "a")
argv = sys.argv[1:]
pw_env = argv[argv.index("--passwordenv") + 1] if "--passwordenv" in argv else None
log.write(json.dumps({"argv": argv, "pw": os.environ.get(pw_env) if pw_env else None}) + "\n")
if pw_env is None:
    import subprocess
    child = subprocess.Popen(["sleep", "120"])
    with open(os.path.join(here, "child.pid"), "a") as fh:
        fh.write(str(child.pid) + "\n")
    time.sleep(120)
print("SESSION-TOKEN-123"); sys.exit(0)
'''


@pytest.fixture
def fake_bw_hang(tmp_path, monkeypatch):
    exe = tmp_path / "bw"
    exe.write_text(_FAKE_BW_HANG, encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "bw.log"
    child_pid = tmp_path / "child.pid"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    unlock_mod.lock()
    yield exe, child_pid, log
    unlock_mod.lock()
    # A failing assertion must not leave the fixture's own children behind.
    if child_pid.exists():
        try:
            import psutil

            for pid in child_pid.read_text(encoding="utf-8-sig").split():
                psutil.Process(int(pid)).kill()
        except Exception:  # noqa: BLE001 - best-effort fixture cleanup
            pass


def _running(pids) -> list:
    """Zombie-aware liveness: a reaped-but-unwaited child still answers os.kill(pid, 0)."""
    import psutil

    out = []
    for pid in pids:
        try:
            proc = psutil.Process(int(pid))
        except psutil.Error:
            continue
        if proc.status() not in (psutil.STATUS_ZOMBIE, psutil.STATUS_DEAD):
            out.append(int(pid))
    return out


def test_desktop_biometric_probe_times_out_then_prompts_exactly_once(fake_bw_hang, monkeypatch):
    """A probe that hangs (wedged desktop app) is killed with the helper it spawned, then the caller
    falls back to exactly one master-password prompt, whose password still unlocks the vault."""
    exe, child_pid, log = fake_bw_hang
    from agent.vault_backends import bitwarden as bitwarden_mod
    from tools.browser_vault_tool import browser_vault_unlock

    monkeypatch.setattr(bitwarden_mod, "_DESKTOP_TIMEOUT", 0.5)
    backend = BitwardenLoginBackend({"enabled": True, "binary_path": str(exe)})
    prompts: list = []
    unlock_mod.set_unlock_prompt_callback(lambda name, display: prompts.append(name) or "correct horse")
    try:
        started = time.monotonic()
        assert backend.try_secretless_unlock() is False
        elapsed = time.monotonic() - started
        assert elapsed < 15, "the probe must not block the unlock indefinitely"
        assert prompts == [], "the probe itself never prompts; the caller does"
        assert not unlock_mod.is_unlocked("bitwarden")

        with patch("agent.vault_backends.base.enabled_backends", return_value=[backend]), \
             patch("agent.vault_backends.enabled_backends", return_value=[backend]):
            assert json.loads(browser_vault_unlock("bitwarden"))["success"] is True
    finally:
        unlock_mod.set_unlock_prompt_callback(None)

    assert prompts == ["bitwarden"], "exactly one fallback prompt, never two"
    assert unlock_mod.is_unlocked("bitwarden")
    calls = [json.loads(line) for line in log.read_text(encoding="utf-8-sig").splitlines()]
    # Every probe comes first and carries no password; the password travels only on the final call.
    assert calls and calls[-1]["pw"] == "correct horse"
    assert all(c["pw"] is None for c in calls[:-1]), "probes must not carry a password"
    assert all("correct horse" not in " ".join(c["argv"]) for c in calls)

    # Every killed probe took its own helper with it: one still running after the fact is a leak.
    helper_pids = child_pid.read_text(encoding="utf-8-sig").split()
    assert helper_pids, "the hanging fake should have recorded at least one helper"
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and _running(helper_pids):
        time.sleep(0.1)
    assert _running(helper_pids) == [], "the timed-out probe left the CLI's helper process running"
