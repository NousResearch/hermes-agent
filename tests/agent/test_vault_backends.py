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
import sys
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


# bw/op fakes are shebang scripts → POSIX only. Proton Pass tests use a
# Windows-compatible `.cmd` shim and run on posix + windows.
_POSIX_ONLY = pytest.mark.platforms("posix")
_PP_PLATFORMS = pytest.mark.platforms("posix", "windows")


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


@_POSIX_ONLY
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


@_POSIX_ONLY
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


@_POSIX_ONLY
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


@_POSIX_ONLY
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


@_POSIX_ONLY
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


@_POSIX_ONLY
def test_onepassword_backend_env_forwards_config_directory(monkeypatch):
    """Vault reads use the same explicit 1Password CLI config location."""
    from agent.vault_backends.onepassword import OnePasswordLoginBackend

    monkeypatch.setenv("OP_CONFIG_DIR", "/tmp/op-config")
    backend = OnePasswordLoginBackend({"enabled": True})

    assert backend._env(None)["OP_CONFIG_DIR"] == "/tmp/op-config"


# ── Proton Pass (pass-cli) ─────────────────────────────────────────────────

# A stand-in `pass-cli` that mimics the four commands the backend uses and the CLI's
# agent-session contract: `login --pat`, `vault list --output json`,
# `item list --vault-name … --filter-type login --filter-state active --output json`,
# and `item view --vault-name … --item-id … --field …`. It records argv + the child env
# so the test can prove the PAT travelled via the documented env var and session dir/reason
# were set. `--pat` is pass-cli's documented argv interface for the PAT (it is not a
# master password), so the PAT legitimately appears in argv for `login` only.
_FAKE_PASS = r'''#!/usr/bin/env python3
import json, os, sys
log = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pass.log"), "a")
argv = sys.argv[1:]
log.write(json.dumps({"argv": argv,
                      "PAT": os.environ.get("PROTON_PASS_PERSONAL_ACCESS_TOKEN"),
                      "SESSION_DIR": os.environ.get("PROTON_PASS_SESSION_DIR"),
                      "REASON": os.environ.get("PROTON_PASS_AGENT_REASON")}) + "\n")
if argv[:2] == ["login", "--pat"]:
    if os.environ.get("PROTON_PASS_PERSONAL_ACCESS_TOKEN") != argv[2]:
        sys.stderr.write("PAT mismatch: argv %r vs env %r" % (argv[2], os.environ.get("PROTON_PASS_PERSONAL_ACCESS_TOKEN")))
        sys.exit(1)
    sys.exit(0)
if argv[:2] == ["vault", "list"]:
    print(json.dumps({"vaults": [{"name": "Personal", "vault_id": "v1", "share_id": "s1"}]})); sys.exit(0)
if argv[:2] == ["item", "list"]:
    print(json.dumps({"items": [
        {"id": "item-1", "title": "Google", "create_time": "2026-01-01T00:00:00Z", "item_type": "login"},
        {"id": "item-2", "title": "Shop", "create_time": "2026-01-02T00:00:00Z", "item_type": "login"},
    ]})); sys.exit(0)
if argv[:2] == ["item", "view"]:
    # item view --field <name>
    field = argv[argv.index("--field") + 1] if "--field" in argv else ""
    item_id = argv[argv.index("--item-id") + 1] if "--item-id" in argv else ""
    if field == "urls":
        print("https://google.com, https://accounts.google.com, androidapp://com.google"); sys.exit(0)
    if field == "email":
        print("jane@example.com" if item_id == "item-1" else "shop@example.com"); sys.exit(0)
    if field == "username":
        sys.exit(1)
    if field == "password":
        print("hunter2-proton"); sys.exit(0)
    if field == "totp_uri":
        print("otpauth://totp/Google:jane@example.com?secret=JBSWY3DPEHPK3PXP"); sys.exit(0)
    sys.exit(1)
if argv[:2] == ["totp", "generate"]:
    print("123456"); sys.exit(0)
sys.exit(2)
'''


@pytest.fixture
def fake_pass(tmp_path, monkeypatch):
    """A pass-cli shim that runs on both POSIX and Windows.

    POSIX: the shebang script above. Windows: a `.cmd` sibling that the backend's
    ``binary_path`` points at, which invokes the same python script via `python -c`.
    """
    exe = tmp_path / "pass-cli"
    exe.write_text(_FAKE_PASS, encoding="utf-8")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    log = tmp_path / "pass.log"
    if sys.platform == "win32":
        # Native python cannot exec a shebang script; wrap it in a .cmd shim.
        shim = tmp_path / "pass-cli.cmd"
        shim.write_text(
            "@echo off\r\n"
            f'"{sys.executable}" "{exe}" %*\r\n',
            encoding="utf-8",
        )
        exe = shim
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    yield exe, log


def _pass_enabled(exe, **cfg):
    from agent.vault_backends.protonpass import ProtonPassLoginBackend

    backend = ProtonPassLoginBackend({"enabled": True, "binary_path": str(exe),
                                      "pat_env": "PROTON_PASS_PERSONAL_ACCESS_TOKEN", **cfg})
    return patch("agent.vault_backends.base.enabled_backends", return_value=[backend]), backend


@_PP_PLATFORMS
def test_protonpass_locked_without_pat_is_reported_not_prompted(fake_pass, monkeypatch):
    """No PAT → nothing prompts (there is no master-password path) and nothing runs."""
    exe, log = fake_pass
    from tools.browser_vault_tool import browser_vault_fill, browser_vault_list

    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    patcher, backend = _pass_enabled(exe)
    # The backend's own is_unlocked gate is what actually prevents the call; also ensure
    # a wired prompt (there is none for pass-cli) would never fire.
    try:
        with patcher, patch("agent.vault_backends.enabled_backends", return_value=[backend]):
            listed = json.loads(browser_vault_list())
            assert listed["items"] == []
            assert listed["locked"] == [{"backend": "protonpass", "display_name": "Proton Pass",
                                         "unlock": "unavailable_in_this_session"}]
            filled = json.loads(browser_vault_fill("pp:item-1", task_id="t"))
            assert filled["success"] is False and filled["error_type"] == "unlock_unavailable"
    finally:
        pass
    assert not log.exists(), "pass-cli must not be invoked at all while locked without a PAT"


@_PP_PLATFORMS
def test_protonpass_list_maps_metadata_via_field_reads(fake_pass, monkeypatch):
    """PAT present → `item list` gives instant titles; `get_meta` resolves the one
    item's URLs+identifier via field reads. The metadata path must never call a full
    item view (totp_uri/passkeys would leak), and list_items must not do per-item reads."""
    exe, log = fake_pass
    patcher, backend = _pass_enabled(exe)

    _require_pat(monkeypatch)
    with patcher:
        metas = backend.list_items()
        assert len(metas) == 2
        assert all(m.origin is None for m in metas), "list_items must not resolve per-item detail"

        meta = backend.get_meta("pp:item-1")
        assert meta is not None
        assert meta.label == "Google"
        assert meta.origin == "https://google.com"
        assert meta.identifier == "jane@example.com"
        assert meta.identifier_type == "email"
        assert list(meta.allowed_origins) == ["https://google.com", "https://accounts.google.com"]

    calls = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    views = [c for c in calls if c["argv"][:2] == ["item", "view"]]
    assert views, "expected at least one item view"
    assert all("--output" not in c["argv"] or "json" not in c["argv"] for c in views), \
        "metadata reads must use --field only, never the full JSON item view"
    assert all(c["REASON"] for c in calls), "PROTON_PASS_AGENT_REASON is mandatory for agent reads"
    assert all(c["SESSION_DIR"] for c in calls), "agent sessions must be isolated via PROTON_PASS_SESSION_DIR"


def _require_pat(monkeypatch, value="test-token-value"):
    monkeypatch.setenv("PROTON_PASS_PERSONAL_ACCESS_TOKEN", value)


@_PP_PLATFORMS
def test_protonpass_login_pat_uses_documented_argv_channel_and_child_env(fake_pass, monkeypatch):
    """The PAT reaches pass-cli through its documented `login --pat` argv slot, and the
    child env also carries it (the CLI's env contract). It must never appear on ANY
    non-login argv (metadata/password reads), and never leak into our process env."""
    exe, log = fake_pass
    patcher, backend = _pass_enabled(exe)

    _require_pat(monkeypatch)
    with patcher:
        assert backend.resolve_password("pp:item-1") == "hunter2-proton"

    calls = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines()]
    login_calls = [c for c in calls if c["argv"][:2] == ["login", "--pat"]]
    assert login_calls and login_calls[0]["argv"][2] == "test-token-value"
    non_login = [c for c in calls if c["argv"][:2] != ["login", "--pat"]]
    assert all("test-token-value" not in " ".join(c["argv"]) for c in non_login), \
        "the PAT must never appear on item/vault argv (those are the audited reads)"
    assert all(c["PAT"] == "test-token-value" for c in calls), "PAT must reach the child via env"


@_PP_PLATFORMS
def test_protonpass_resolve_otp_uses_totp_generate(fake_pass, monkeypatch):
    exe, log = fake_pass
    patcher, backend = _pass_enabled(exe)

    _require_pat(monkeypatch)
    with patcher:
        assert backend.resolve_otp("pp:item-1") == "123456"
