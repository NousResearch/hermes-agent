"""macOS Keychain vault backend: contract tests + real-binary E2E.

Two layers, mirroring the suite's fake-CLI pattern for `bw`/`op`:

1. A fake ``security`` shim (JSON state file) pins the backend's contract —
   dump parsing (metadata, no passwords), password extraction, the locked-rc
   ladder (152 → auto-unlock with sidecar / UnlockRequired without), the
   delete-then-add idempotency, and origin mapping — on every host.
2. A ``macos_only`` E2E exercises the REAL ``security`` binary on a tmp keychain:
   provision → add → list → resolve → lock → self-healing unlock → resolve → rm.

The shim emulates the exact ``security`` internet-password surface verified on
macOS 27 (see the module docstring in agent/vault_backends/keychain.py).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from agent.vault_backends.base import UnlockRequired
from agent.vault_backends.keychain import MacOSKeychainLoginBackend, provision
from agent.vault_store import VaultError

_FAKE_SEC = r'''#!/usr/bin/env python3
# Fake `security` for internet-password ops. State: {"unlocked": bool, "password": str,
# "items": [{server, account, label, cdat, password}]}. Mirrors the real rc codes:
#   locked reads → 152; duplicate add → 45; not found → 44; bad -A flag → 48.
import json, sys

state_path, args = sys.argv[1], sys.argv[2:]
state = json.load(open(state_path))
item = None

def find(server, account):
    return next((i for i in state["items"] if i["server"] == server and i["account"] == account), None)

def check_locked():
    if not state.get("unlocked", True):
        sys.exit(152)

def write():
    json.dump(state, open(state_path, "w"))

if args[0] == "create-keychain":
    state["password"] = args[args.index("-p") + 1]
    state["unlocked"] = True
    write(); sys.exit(0)
if args[0] == "unlock-keychain":
    pw = args[args.index("-p") + 1]
    if pw == state.get("password"):
        state["unlocked"] = True; write(); sys.exit(0)
    sys.exit(1)
if args[0] == "lock-keychain":
    state["unlocked"] = False; write(); sys.exit(0)
if args[0] == "set-keychain-settings":
    sys.exit(0)

def parse(opts, flags_with_value):
    values, rest = {}, []
    i = 0
    while i < len(opts):
        if opts[i] in flags_with_value:
            values[opts[i]] = opts[i + 1]; i += 2
        else:
            rest.append(opts[i]); i += 1
    return values, rest

if args[0] == "add-internet-password":
    values, rest = parse(args[1:], ("-a", "-s", "-l", "-w", "-D", "-j"))
    if any(a.startswith("-A") for a in args[1:]):
        sys.exit(48)
    server, account = values.get("-s", ""), values.get("-a", "")
    if find(server, account):
        sys.exit(45)
    state["items"].append({"server": server, "account": account,
                           "label": values.get("-l"), "cdat": "20260922144615Z",
                           "password": values.get("-w", "")})
    write(); sys.exit(0)
if args[0] == "delete-internet-password":
    values, rest = parse(args[1:], ("-a", "-s"))
    before = len(state["items"])
    state["items"] = [i for i in state["items"]
                      if not (i["server"] == values.get("-s") and i["account"] == values.get("-a"))]
    write()
    sys.exit(0 if len(state["items"]) != before else 44)
if args[0] == "find-internet-password":
    values, rest = parse(args[1:], ("-a", "-s"))
    server, account = values.get("-s", ""), values.get("-a", "")
    check_locked()
    hit = find(server, account)
    if hit is None:
        sys.exit(44)
    print(f'keychain: "PATH"')
    print("version: 512")
    print('class: "inet"')
    # Real `security` writes the password line to STDERR, not stdout (macOS 27, verified),
    # and escapes quotes/backslashes with backslashes inside the quoted value.
    escaped = hit["password"].replace("\\", "\\\\").replace('"', '\\"')
    sys.stderr.write(f'password: "{escaped}"\n')
    print("attributes:")
    print(f'    "srvr"<blob>="{hit["server"]}"')
    print(f'    "acct"<blob>="{hit["account"]}"')
    print(f'    "cdat"<timedate>=0x30  "{hit["cdat"]}Z\\000"')
    if hit.get("label"):
        print(f'    0x00000007 <blob>="{hit["label"]}"')
    sys.exit(0)
if args[0] == "dump-keychain":
    for hit in state["items"]:
        print('keychain: "PATH"')
        print("version: 512")
        print('class: "inet"')
        print("attributes:")
        print(f'    "srvr"<blob>="{hit["server"]}"')
        print(f'    "acct"<blob>="{hit["account"]}"')
        print(f'    "cdat"<timedate>=0x30  "{hit["cdat"]}Z\\000"')
        if hit.get("label"):
            print(f'    0x00000007 <blob>="{hit["label"]}"')
    sys.exit(0)
sys.exit(2)
'''


def _state(tmp_path):
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"unlocked": True, "password": "", "items": []}))
    return p


@pytest.fixture
def fake_security(tmp_path, monkeypatch):
    """Patch the backend's ``_sec`` to a fake ``security`` script backed by JSON state."""
    shim = tmp_path / "security"
    shim.write_text(_FAKE_SEC, encoding="utf-8")
    shim.chmod(0o755)
    state_path = _state(tmp_path)
    calls = []

    def fake_sec(self, *args, cwd=None):
        proc = subprocess.run([sys.executable, str(shim), str(state_path), *args],
                              cwd=str(cwd) if cwd else None, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=30)
        calls.append(list(args))
        return proc

    monkeypatch.setattr(MacOSKeychainLoginBackend, "_sec", fake_sec)
    return SimpleNamespace(state_path=state_path, calls=calls, tmp=tmp_path)


def _backend(tmp_path, *, with_sidecar=False, **cfg):
    base = {"file": str(tmp_path / "vault.keychain-db")}
    # A real `security add` would have created the file already (the backend gates on
    # existence); the shim never touches disk, so the test provides it.
    (tmp_path / "vault.keychain-db").touch()
    if with_sidecar:
        base["password_file"] = str(tmp_path / "vault.pw")
        (tmp_path / "vault.pw").write_text("masterpw\n")
    base.update(cfg)
    return MacOSKeychainLoginBackend(base)


def _seed_state(fake_security, **kwargs):
    state = json.loads(fake_security.state_path.read_text())
    state.update(kwargs)
    fake_security.state_path.write_text(json.dumps(state))


# ---------------------------------------------------------------------------
# Contract tests (shim)
# ---------------------------------------------------------------------------

def test_list_items_parses_dump_metadata_only(fake_security):
    _seed_state(fake_security, items=[
        {"server": "example.org", "account": "jane@example.com", "label": "Example",
         "cdat": "20260922144615Z", "password": "pwOne"},
        {"server": "example.net:8443", "account": "admin", "label": None,
         "cdat": "20260922144615Z", "password": "pwTwo"},
    ])
    backend = _backend(fake_security.tmp)
    metas = backend.list_items()

    assert len(metas) == 2
    first, second = metas
    assert first.id == "kc:example.org|jane@example.com"
    assert first.kind == "login"
    assert first.label == "Example"
    assert first.origin == "https://example.org"
    assert first.allowed_origins == ("https://example.org", "http://example.org")
    assert first.identifier == "jane@example.com"
    assert first.identifier_type == "email"
    assert first.created_at == datetime(2026, 9, 22, 14, 46, 15, tzinfo=timezone.utc).isoformat()
    assert second.origin == "https://example.net:8443"
    # list_items never touches passwords: the dump has none to expose.
    assert all(getattr(m, "has_otp", False) is False for m in metas)


def test_list_items_missing_file_is_empty(tmp_path):
    backend = MacOSKeychainLoginBackend({"file": str(tmp_path / "nope.keychain-db")})
    assert backend.list_items() == []


def test_resolve_password_extracts_and_unescapes(fake_security):
    _seed_state(fake_security, items=[
        {"server": "example.org", "account": "jane@example.com", "label": None,
         "cdat": "20260922144615Z", "password": 'a"b\\c'},
    ])
    backend = _backend(fake_security.tmp)
    assert backend.resolve_password("kc:example.org|jane@example.com") == 'a"b\\c'


def test_resolve_password_missing_item_raises_vault_error(fake_security):
    backend = _backend(fake_security.tmp)
    with pytest.raises(VaultError):
        backend.resolve_password("kc:example.org|nobody")


def test_resolve_password_malformed_handle_raises(fake_security):
    backend = _backend(fake_security.tmp)
    with pytest.raises(VaultError):
        backend.resolve_password("kc:novalue")


def test_locked_without_sidecar_raises_unlock_required(fake_security):
    _seed_state(fake_security, unlocked=False, password="masterpw",
                items=[{"server": "example.org", "account": "jane@example.com", "label": None,
                        "cdat": "20260922144615Z", "password": "pwOne"}])
    backend = _backend(fake_security.tmp)  # no sidecar → prompt mode
    assert backend.needs_unlock is True
    assert backend.is_unlocked() is False
    # list_items keeps working while locked (metadata is not gated by the OS).
    assert len(backend.list_items()) == 1
    with pytest.raises(UnlockRequired):
        backend.resolve_password("kc:example.org|jane@example.com")


def test_locked_with_sidecar_self_heals(fake_security):
    _seed_state(fake_security, unlocked=False, password="masterpw",
                items=[{"server": "example.org", "account": "jane@example.com", "label": None,
                        "cdat": "20260922144615Z", "password": "pwOne"}])
    backend = _backend(fake_security.tmp, with_sidecar=True)
    assert backend.needs_unlock is False
    assert backend.resolve_password("kc:example.org|jane@example.com") == "pwOne"
    # The unlock happened automatically with the sidecar password.
    assert any(c[0] == "unlock-keychain" for c in fake_security.calls)
    assert json.loads(fake_security.state_path.read_text())["unlocked"] is True


def test_attended_unlock_then_resolve(fake_security):
    _seed_state(fake_security, unlocked=False, password="masterpw",
                items=[{"server": "example.org", "account": "jane@example.com", "label": None,
                        "cdat": "20260922144615Z", "password": "pwOne"}])
    backend = _backend(fake_security.tmp)
    backend.unlock("masterpw")
    assert backend.is_unlocked() is True
    assert backend.resolve_password("kc:example.org|jane@example.com") == "pwOne"
    with pytest.raises(RuntimeError):
        backend.unlock("wrongpw")


def test_get_meta_derives_from_handle_without_subprocess(fake_security, monkeypatch):
    backend = _backend(fake_security.tmp)

    def boom(self, *a, **k):
        raise AssertionError("get_meta must not invoke security")

    monkeypatch.setattr(MacOSKeychainLoginBackend, "_sec", boom)
    meta = backend.get_meta("kc:example.org|jane@example.com")
    assert meta.id == "kc:example.org|jane@example.com"
    assert meta.origin == "https://example.org"
    assert meta.identifier == "jane@example.com"
    assert meta.identifier_type == "email"
    # A separator-less handle is a server with no account — still derivable (safe: fills
    # resolve against the real item and fail cleanly when nothing matches).
    assert backend.get_meta("kc:example.org").origin == "https://example.org"
    assert backend.get_meta("kc:") is None


def test_add_item_creates_then_duplicate_replaces(fake_security):
    backend = _backend(fake_security.tmp, with_sidecar=True)
    handle = backend.add_item("example.org", "jane@example.com", "pwOne", label="Example")
    assert handle == "kc:example.org|jane@example.com"
    # Idempotent: a second add of the same server|account deletes then re-adds.
    backend.add_item("example.org", "jane@example.com", "pwTwo")
    state = json.loads(fake_security.state_path.read_text())
    same = [i for i in state["items"] if i["server"] == "example.org"]
    assert len(same) == 1 and same[0]["password"] == "pwTwo"
    assert any(c[0] == "delete-internet-password" for c in fake_security.calls)


def test_add_item_requires_server_and_account(fake_security):
    backend = _backend(fake_security.tmp, with_sidecar=True)
    with pytest.raises(VaultError):
        backend.add_item("example.org", "", "pw")
    with pytest.raises(VaultError):
        backend.add_item("", "jane@example.com", "pw")


def test_remove_item(fake_security):
    _seed_state(fake_security, items=[
        {"server": "example.org", "account": "jane@example.com", "label": None,
         "cdat": "20260922144615Z", "password": "pwOne"}])
    backend = _backend(fake_security.tmp, with_sidecar=True)
    assert backend.remove_item("kc:example.org|jane@example.com") is True
    assert backend.remove_item("kc:example.org|jane@example.com") is False
    assert backend.remove_item("kc:malformed") is False


def test_status_shape(fake_security):
    _seed_state(fake_security, items=[
        {"server": "example.org", "account": "jane@example.com", "label": None,
         "cdat": "20260922144615Z", "password": "pwOne"}])
    backend = _backend(fake_security.tmp, with_sidecar=True)
    info = backend.status()
    assert info["mode"] == "unattended"
    assert info["file_exists"] is True
    assert info["item_count"] == 1
    assert info["unlocked"] is True


# ---------------------------------------------------------------------------
# Real-binary E2E (macOS hosts with /usr/bin/security)
# ---------------------------------------------------------------------------

@pytest.mark.macos_only
@pytest.mark.skipif(shutil.which("security") is None or not os.path.exists("/usr/bin/security"),
                    reason="requires the macOS security binary")
def test_keychain_e2e_real_binary(tmp_path):
    kc_path = tmp_path / "e2e.keychain-db"
    pw_path = tmp_path / "e2e.pw"
    provision(file=kc_path, password_file=pw_path)
    assert kc_path.exists()
    assert pw_path.exists()
    assert pw_path.stat().st_mode & 0o777 == 0o600

    backend = MacOSKeychainLoginBackend({"file": str(kc_path), "password_file": str(pw_path)})
    assert backend.needs_unlock is False

    handle = backend.add_item("example.com", "alice@example.com", "hunter2", label="Example")
    assert handle == "kc:example.com|alice@example.com"

    metas = backend.list_items()
    assert len(metas) == 1
    assert metas[0].origin == "https://example.com"
    assert metas[0].allowed_origins == ("https://example.com", "http://example.com")
    assert metas[0].identifier == "alice@example.com"
    assert metas[0].created_at  # real cdat came through

    assert backend.resolve_password(handle) == "hunter2"

    # Lock the keychain for real; the sidecar mode must self-heal on the next read.
    backend._sec("lock-keychain", str(kc_path))
    assert backend.resolve_password(handle) == "hunter2"

    assert backend.remove_item(handle) is True
    assert backend.list_items() == []


@pytest.mark.macos_only
@pytest.mark.skipif(shutil.which("security") is None or not os.path.exists("/usr/bin/security"),
                    reason="requires the macOS security binary")
def test_keychain_prompt_mode_real_binary(tmp_path):
    kc_path = tmp_path / "e2e-prompt.keychain-db"
    pw_path = tmp_path / "e2e-prompt.pw"
    provision(file=kc_path, password_file=pw_path)
    master = pw_path.read_text().strip()
    pw_path.unlink()  # drop the sidecar → prompt mode

    backend = MacOSKeychainLoginBackend({"file": str(kc_path)})
    assert backend.needs_unlock is True
    handle = backend.add_item("example.org", "bob@example.org", "s3cret")
    backend._sec("lock-keychain", str(kc_path))
    assert backend.is_unlocked() is False
    # No sidecar → a locked read refuses instead of prompting on its own.
    with pytest.raises(UnlockRequired):
        backend.resolve_password(handle)
    backend.unlock(master)
    assert backend.is_unlocked() is True
    assert backend.resolve_password(handle) == "s3cret"
    backend.remove_item(handle)