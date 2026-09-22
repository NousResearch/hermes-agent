"""macOS Keychain vault backend: contract tests + real-binary E2E.

Two layers, mirroring the suite's fake-CLI pattern for `bw`/`op`:

1. A fake ``security`` shim (JSON state file) pins the backend's contract — dump
   parsing (metadata, no passwords), password extraction, the locked-rc ladder
   (152 → auto-unlock with sidecar / UnlockRequired without), the atomic ``-U``
   create-or-update replace, exact-origin persistence (``-D``), the pty prompt
   channel (no secrets in argv/env), the process-level unlock lease, and the
   idle-TTL relock — on every host.
2. A ``macos_only`` E2E exercises the REAL ``security`` binary on a tmp keychain:
   provision (pty-fed create) → add → replace in place → list → resolve → lock →
   self-healing unlock → resolve → attended unlock across FRESH instances →
   release → re-lock → rm.

The shim emulates the exact ``security`` internet-password surface verified on
macOS 27 (see the module docstring in agent/vault_backends/keychain.py).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from agent.vault_backends.base import UnlockRequired
from agent.vault_backends import keychain as kc_mod
from agent.vault_backends.keychain import (
    MacOSKeychainLoginBackend,
    _LEASE_TTL,
    provision,
)
from agent.vault_store import VaultError

_FAKE_SEC = r'''#!/usr/bin/env python3
# Fake `security` for internet-password ops. State: {"unlocked": bool, "password": str,
# "items": [{server, account, label, desc, cdat, password}]}. Mirrors the real rc codes:
#   locked reads → 152; duplicate add w/o -U → 45; not found → 44; bad -A flag → 48.
# Prompt-mode ops (create-/unlock-keychain WITHOUT -p, add WITHOUT -w) read replies
# from stdin, one line per prompt, emulating the real TTY prompt channel.
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

def read_replies(n):
    lines = []
    for _ in range(n):
        try:
            lines.append(sys.stdin.readline().rstrip("\r\n"))
        except Exception:
            lines.append("")
    return lines

if args[0] == "create-keychain":
    if "-p" in args:
        state["password"] = args[args.index("-p") + 1]
    else:
        replies = read_replies(2)
        state["password"] = replies[-1] if replies[-1] else replies[0]
    state["unlocked"] = True
    write(); sys.exit(0)
if args[0] == "unlock-keychain":
    if "-p" in args:
        pw = args[args.index("-p") + 1]
    else:
        pw = read_replies(1)[0]
    if pw == state.get("password"):
        state["unlocked"] = True; write(); sys.exit(0)
    sys.exit(1)
if args[0] == "lock-keychain":
    state["unlocked"] = False; write(); sys.exit(0)
if args[0] == "set-keychain-settings":
    sys.exit(0)

def parse(opts, flags_with_value):
    values, rest, flags = {}, [], []
    i = 0
    while i < len(opts):
        if opts[i] in flags_with_value:
            values[opts[i]] = opts[i + 1]; i += 2
        elif opts[i].startswith("-"):
            flags.append(opts[i]); i += 1
        else:
            rest.append(opts[i]); i += 1
    return values, rest, flags

if args[0] == "add-internet-password":
    values, rest, flags = parse(args[1:], ("-a", "-s", "-l", "-w", "-D", "-j"))
    if any(a.startswith("-A") for a in args[1:]):
        sys.exit(48)
    server, account = values.get("-s", ""), values.get("-a", "")
    existing = find(server, account)
    if existing and "-U" not in flags:
        sys.exit(45)
    entry = {"server": server, "account": account,
             "label": values.get("-l"), "desc": values.get("-D"),
             "cdat": "20260922144615Z", "password": values.get("-w", "")}
    if existing:
        existing.update({k: v for k, v in entry.items() if v is not None})
        existing["password"] = entry["password"]
    else:
        state["items"].append(entry)
    write(); sys.exit(0)
if args[0] == "delete-internet-password":
    values, rest, flags = parse(args[1:], ("-a", "-s"))
    before = len(state["items"])
    state["items"] = [i for i in state["items"]
                      if not (i["server"] == values.get("-s") and i["account"] == values.get("-a"))]
    write()
    sys.exit(0 if len(state["items"]) != before else 44)
if args[0] == "find-internet-password":
    values, rest, flags = parse(args[1:], ("-a", "-s"))
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
    if hit.get("desc"):
        print(f'    "desc"<blob>="{hit["desc"]}"')
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
        if hit.get("desc"):
            print(f'    "desc"<blob>="{hit["desc"]}"')
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
    """Patch ``_sec``/``_sec_prompt``/``_env`` so the backend talks to a fake
    ``security`` script backed by JSON state, recording every argv and env used."""
    shim = tmp_path / "security"
    shim.write_text(_FAKE_SEC, encoding="utf-8")
    shim.chmod(0o755)
    state_path = _state(tmp_path)
    calls, envs = [], []

    def fake_env(self):
        envs.append(dict(os.environ))
        return dict(os.environ)

    def fake_sec(self, *args, cwd=None):
        proc = subprocess.run([sys.executable, str(shim), str(state_path), *args],
                              cwd=str(cwd) if cwd else None, env=self._env(),
                              capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=30)
        calls.append(list(args))
        return proc

    def fake_prompt(self, argv, replies):
        # Emulates the pty channel: the fake security reads its "prompts" from stdin.
        proc = subprocess.run([sys.executable, str(shim), str(state_path), *argv],
                              input="\n".join(replies) + "\n", env=self._env(),
                              capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=30)
        calls.append(list(argv))
        return proc

    monkeypatch.setattr(MacOSKeychainLoginBackend, "_env", fake_env)
    monkeypatch.setattr(MacOSKeychainLoginBackend, "_sec", fake_sec)
    monkeypatch.setattr(MacOSKeychainLoginBackend, "_sec_prompt", fake_prompt)
    kc_mod._LEASES.clear()  # per-test lease isolation
    return SimpleNamespace(state_path=state_path, calls=calls, envs=envs, tmp=tmp_path)


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


_ITEM = {"server": "example.org", "account": "jane@example.com", "label": None,
         "desc": "https://example.org", "cdat": "20260922144615Z", "password": "pwOne"}


# ---------------------------------------------------------------------------
# Contract tests (shim)
# ---------------------------------------------------------------------------

def test_list_items_parses_dump_metadata_only(fake_security):
    _seed_state(fake_security, items=[
        dict(_ITEM, label="Example"),
        {"server": "example.net:8443", "account": "admin", "label": None, "desc": None,
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
    # Exact-origin: only the persisted desc origin is fillable — no https/http widening.
    assert first.allowed_origins == ("https://example.org",)
    assert first.identifier == "jane@example.com"
    assert first.identifier_type == "email"
    assert first.created_at == datetime(2026, 9, 22, 14, 46, 15, tzinfo=timezone.utc).isoformat()
    # No desc on record -> not fillable anywhere (never invented).
    assert second.origin is None
    assert second.allowed_origins == ()
    # list_items never touches passwords: the dump has none to expose.
    assert all(getattr(m, "has_otp", False) is False for m in metas)


def test_list_items_missing_file_is_empty(tmp_path):
    backend = MacOSKeychainLoginBackend({"file": str(tmp_path / "nope.keychain-db")})
    assert backend.list_items() == []


def test_resolve_password_extracts_and_unescapes(fake_security):
    _seed_state(fake_security, items=[dict(_ITEM, password='a"b\\c')])
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
    _seed_state(fake_security, unlocked=False, password="masterpw", items=[dict(_ITEM)])
    backend = _backend(fake_security.tmp)  # no sidecar → prompt mode
    assert backend.needs_unlock is True
    assert backend.is_unlocked() is False
    # list_items keeps working while locked (metadata is not gated by the OS).
    assert len(backend.list_items()) == 1
    with pytest.raises(UnlockRequired):
        backend.resolve_password("kc:example.org|jane@example.com")


def test_locked_with_sidecar_self_heals(fake_security):
    _seed_state(fake_security, unlocked=False, password="masterpw", items=[dict(_ITEM)])
    backend = _backend(fake_security.tmp, with_sidecar=True)
    assert backend.needs_unlock is False
    assert backend.resolve_password("kc:example.org|jane@example.com") == "pwOne"
    # The unlock happened automatically with the sidecar password — via the pty
    # channel: argv carries the keychain path only, never the secret.
    unlock_calls = [c for c in fake_security.calls if c[0] == "unlock-keychain"]
    assert unlock_calls
    assert all("masterpw" not in c for c in unlock_calls)
    assert json.loads(fake_security.state_path.read_text())["unlocked"] is True


def test_attended_unlock_then_resolve(fake_security):
    _seed_state(fake_security, unlocked=False, password="masterpw", items=[dict(_ITEM)])
    backend = _backend(fake_security.tmp)
    backend.unlock("masterpw")
    assert backend.is_unlocked() is True
    assert backend.resolve_password("kc:example.org|jane@example.com") == "pwOne"
    with pytest.raises(RuntimeError):
        backend.unlock("wrongpw")


def test_unlock_lease_carries_across_fresh_instances(fake_security):
    """P1 review: unlock authority must survive the per-call backend construction."""
    _seed_state(fake_security, unlocked=False, password="masterpw", items=[dict(_ITEM)])
    backend1 = _backend(fake_security.tmp)
    backend1.unlock("masterpw")

    # Every later tool call builds a FRESH instance — it must still see the unlock.
    backend2 = _backend(fake_security.tmp)
    assert backend2.is_unlocked() is True
    assert backend2.resolve_password("kc:example.org|jane@example.com") == "pwOne"

    # Session release: physical re-lock + lease dropped; the next fresh instance
    # must be locked again and refuse secret reads.
    backend2.release()
    backend3 = _backend(fake_security.tmp)
    assert backend3.is_unlocked() is False
    with pytest.raises(UnlockRequired):
        backend3.resolve_password("kc:example.org|jane@example.com")
    assert json.loads(fake_security.state_path.read_text())["unlocked"] is False


def test_lease_expiry_relocks(fake_security, monkeypatch):
    """Idle-TTL expiry physically re-locks and drops the lease (P1 review)."""
    _seed_state(fake_security, unlocked=False, password="masterpw", items=[dict(_ITEM)])
    shim = fake_security.tmp / "security"
    state_path = fake_security.state_path

    def fake_physical_lock(path):
        subprocess.run([sys.executable, str(shim), str(state_path), "lock-keychain", path],
                       capture_output=True, timeout=30)

    monkeypatch.setattr(kc_mod, "_physical_lock", fake_physical_lock)
    backend1 = _backend(fake_security.tmp)
    backend1.unlock("masterpw")
    assert backend1.is_unlocked() is True

    monkeypatch.setattr(kc_mod, "_LEASE_TTL", 0.2)
    time.sleep(0.35)
    backend2 = _backend(fake_security.tmp)  # construction purges expired leases
    assert backend2.is_unlocked() is False
    # The physical lock really happened: the shim keychain is locked again.
    assert json.loads(state_path.read_text())["unlocked"] is False
    with pytest.raises(UnlockRequired):
        backend2.resolve_password("kc:example.org|jane@example.com")


def test_get_meta_uses_persisted_exact_origin(fake_security):
    _seed_state(fake_security, items=[dict(_ITEM), dict(_ITEM, server="legacy.net",
                                                       account="old", desc=None,
                                                       password="pwOld")])
    backend = _backend(fake_security.tmp)
    meta = backend.get_meta("kc:example.org|jane@example.com")
    assert meta is not None
    assert meta.id == "kc:example.org|jane@example.com"
    assert meta.origin == "https://example.org"
    assert meta.allowed_origins == ("https://example.org",)  # NOT http:// widened
    assert meta.identifier == "jane@example.com"
    assert meta.identifier_type == "email"
    # Items predating origin recording are visible but NOT fillable anywhere.
    legacy = backend.get_meta("kc:legacy.net|old")
    assert legacy is not None
    assert legacy.origin is None
    assert legacy.allowed_origins == ()
    assert backend.get_meta("kc:") is None
    assert backend.get_meta("kc:example.org|nobody") is None
    # get_meta now performs a real lookup (exactness needs the persisted desc).
    assert any(c[0] == "find-internet-password" for c in fake_security.calls)


def test_add_item_creates_then_replaces_atomically(fake_security):
    """P1 review: replace must not delete-then-add (a failed add would lose the
    credential). The backend uses `-U` — update in place, single call."""
    backend = _backend(fake_security.tmp, with_sidecar=True)
    handle = backend.add_item("example.org", "jane@example.com", "pwOne",
                              label="Example", origin="https://example.org")
    assert handle == "kc:example.org|jane@example.com"
    backend.add_item("example.org", "jane@example.com", "pwTwo",
                     origin="https://example.org")
    state = json.loads(fake_security.state_path.read_text())
    same = [i for i in state["items"] if i["server"] == "example.org"]
    assert len(same) == 1 and same[0]["password"] == "pwTwo"
    assert same[0]["desc"] == "https://example.org"
    # -U means the old value was never deleted: not a single delete call, ever.
    assert all(c[0] != "delete-internet-password" for c in fake_security.calls)
    assert any("-U" in c for c in fake_security.calls)


def test_concurrent_writers_no_loss(fake_security):
    """P1 review: same-item writer contention — the item must always exist with one
    of the written values (atomic -U; never delete-then-add)."""
    backend_a = _backend(fake_security.tmp, with_sidecar=True)
    backend_b = _backend(fake_security.tmp, with_sidecar=True)
    for _ in range(3):
        backend_a.add_item("example.org", "jane@example.com", "val-A",
                           origin="https://example.org")
        backend_b.add_item("example.org", "jane@example.com", "val-B",
                           origin="https://example.org")
    state = json.loads(fake_security.state_path.read_text())
    same = [i for i in state["items"] if i["server"] == "example.org"]
    assert len(same) == 1
    assert same[0]["password"] in ("val-A", "val-B")
    assert all(c[0] != "delete-internet-password" for c in fake_security.calls)


def test_origin_is_not_widened_across_schemes(fake_security):
    """P1 review (#96970/#111480): an https-saved credential must not be fillable
    on the plaintext http sibling."""
    _seed_state(fake_security, items=[dict(_ITEM, desc="https://example.org")])
    backend = _backend(fake_security.tmp)
    meta = backend.get_meta("kc:example.org|jane@example.com")
    assert meta is not None
    assert meta.allowed_origins == ("https://example.org",)
    assert "http://example.org" not in meta.allowed_origins


def test_subprocess_boundary_no_secrets_in_argv_or_env(fake_security):
    """P1 review: fixture secrets must never leak into argv/env/output/errors —
    except the single documented `-w <value>` slot `security` requires for item
    writes (verified: its prompt flavor does not commit headlessly)."""
    _seed_state(fake_security, unlocked=False, password="masterpw", items=[dict(_ITEM)])
    backend = _backend(fake_security.tmp, with_sidecar=True)
    backend.add_item("example.org", "jane@example.com", "secret-PW",
                     origin="https://example.org")
    backend.add_item("example.org", "jane@example.com", "secret-PW-2",
                     origin="https://example.org")
    backend.resolve_password("kc:example.org|jane@example.com")  # triggers pty unlock

    for argv in fake_security.calls:
        # Everything OUTSIDE the single documented -w slot must be secret-free.
        if "-w" in argv:
            wpos = argv.index("-w")
            if wpos + 1 >= len(argv) or argv[wpos + 1] not in ("secret-PW", "secret-PW-2"):
                raise AssertionError(f"unexpected -w slot in argv: {argv!r}")
            exposed = argv[:wpos] + argv[wpos + 2:]
        else:
            exposed = argv
        blob = " ".join(exposed).lower()
        for secret in ("masterpw", "secret-pw"):
            assert secret not in blob, f"secret leaked into argv: {argv!r}"
    for env in fake_security.envs:
        blob = json.dumps(env)
        assert "masterpw" not in blob
        assert "secret-PW" not in blob


def test_add_item_requires_server_and_account(fake_security):
    backend = _backend(fake_security.tmp, with_sidecar=True)
    with pytest.raises(VaultError):
        backend.add_item("example.org", "", "pw")
    with pytest.raises(VaultError):
        backend.add_item("", "jane@example.com", "pw")


def test_remove_item(fake_security):
    _seed_state(fake_security, items=[dict(_ITEM)])
    backend = _backend(fake_security.tmp, with_sidecar=True)
    assert backend.remove_item("kc:example.org|jane@example.com") is True
    assert backend.remove_item("kc:example.org|jane@example.com") is False
    assert backend.remove_item("kc:malformed") is False


def test_status_shape(fake_security):
    _seed_state(fake_security, items=[dict(_ITEM)])
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
    provision(file=kc_path, password_file=pw_path)  # pty-fed create; no argv secret
    assert kc_path.exists()
    assert pw_path.exists()
    assert pw_path.stat().st_mode & 0o777 == 0o600

    backend = MacOSKeychainLoginBackend({"file": str(kc_path), "password_file": str(pw_path)})
    assert backend.needs_unlock is False

    handle = backend.add_item("example.com", "alice@example.com", "hunter2",
                              label="Example", origin="https://example.com")
    assert handle == "kc:example.com|alice@example.com"

    metas = backend.list_items()
    assert len(metas) == 1
    assert metas[0].origin == "https://example.com"
    assert metas[0].allowed_origins == ("https://example.com",)
    assert metas[0].identifier == "alice@example.com"
    assert metas[0].created_at  # real cdat came through

    assert backend.resolve_password(handle) == "hunter2"

    # Atomic in-place replace (real `-U`): same item, new password, still one item.
    backend.add_item("example.com", "alice@example.com", "hunter3",
                     origin="https://example.com")
    assert backend.resolve_password(handle) == "hunter3"
    assert len(backend.list_items()) == 1

    # Lock the keychain for real; the sidecar mode must self-heal on the next read.
    backend._sec("lock-keychain", str(kc_path))
    assert backend.resolve_password(handle) == "hunter3"

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
    handle = backend.add_item("example.org", "bob@example.org", "s3cret",
                              origin="https://example.org")
    backend._sec("lock-keychain", str(kc_path))
    assert backend.is_unlocked() is False
    # No sidecar → a locked read refuses instead of prompting on its own.
    with pytest.raises(UnlockRequired):
        backend.resolve_password(handle)
    backend.unlock(master)  # real pty prompt channel; no argv secret
    assert backend.is_unlocked() is True
    assert backend.resolve_password(handle) == "s3cret"

    # P1 review: a FRESH instance (as every tool call builds) sees the unlock...
    fresh = MacOSKeychainLoginBackend({"file": str(kc_path)})
    assert fresh.is_unlocked() is True
    assert fresh.resolve_password(handle) == "s3cret"
    # ...and session release physically re-locks for the process.
    fresh.release()
    later = MacOSKeychainLoginBackend({"file": str(kc_path)})
    assert later.is_unlocked() is False
    with pytest.raises(UnlockRequired):
        later.resolve_password(handle)

    later.unlock(master)
    backend.remove_item(handle)