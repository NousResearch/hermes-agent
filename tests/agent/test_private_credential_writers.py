"""Cross-writer invariant: every private-credential file is 0600 from the moment its temp file exists.

The credential writers (auth.json, MCP OAuth tokens, secret-source cache, iron-proxy state, the
exchanged-JWT store, the Photon sidecar record, pairing data, the vault blob, the third-party
credential file, the spawn ledger, the meet-node token) all funnel through ``utils.atomic_json_write`` / ``atomic_write_text`` /
``atomic_write_bytes`` with ``mode=0o600``. The contract under test: the *temp* file is created
with mode 0600 (``O_EXCL``) BEFORE any byte lands and the final file carries 0600 — never
"open at umask, then chmod" (the #19673 window). POSIX-only: mode bits are not enforced on Windows.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(sys.platform.startswith("win"), reason="POSIX mode bits not enforced on Windows")

_PRIVATE = stat.S_IRUSR | stat.S_IWUSR


@pytest.fixture
def opens_spy(monkeypatch):
    """Record every ``os.open`` create (path, flags, mode) issued through ``utils``."""
    import utils

    observed: list[tuple[str, int, int]] = []
    real_open = os.open

    def spying(path, flags, mode=0o777, *args, **kwargs):
        if flags & os.O_CREAT:
            observed.append((os.fspath(path), flags, mode))
        return real_open(path, flags, mode, *args, **kwargs)

    # mkstemp resolves ``os.open`` at call time from the ``tempfile`` module namespace.
    monkeypatch.setattr(utils.tempfile._os, "open", spying)
    return observed


def _writers(home: Path, monkeypatch):
    """``(label, callable, target_path)`` for every private-credential writer."""
    from agent import anthropic_credentials
    from agent.secret_sources._cache import CachedFetch, DiskCache
    from agent.proxy_sources import iron_proxy
    from agent.vault_store import VaultStore
    from gateway import pairing
    from hermes_cli import auth as auth_mod, copilot_auth
    from hermes_cli import process_identity
    from tools import mcp_oauth
    from plugins.google_meet.node.server import NodeServer
    from plugins.platforms.photon import adapter as photon_adapter

    photon_record = home / "runtime" / "photon.json"
    monkeypatch.setattr(photon_adapter, "_runtime_record_path", lambda: photon_record)
    ledger = home / "spawn-ledger.json"
    monkeypatch.setattr(process_identity, "_ledger_path", lambda: ledger)
    cache = DiskCache("probe.json", key_serializer=str)
    vault = VaultStore(home / "vault")
    meet_node = NodeServer(token_path=home / "meetings" / "node_token.json")
    return [
        ("auth.json", lambda: auth_mod._save_auth_store({"version": auth_mod.AUTH_STORE_VERSION, "providers": {}}),
         auth_mod._auth_file_path()),
        ("third-party credentials", lambda: anthropic_credentials._atomic_write_private_json(
            home / "cc" / ".credentials.json", {"tok": 1}), home / "cc" / ".credentials.json"),
        ("mcp oauth tokens", lambda: mcp_oauth._write_json(home / "mcp" / "probe.tokens.json", {"access_token": "x"}),
         home / "mcp" / "probe.tokens.json"),
        ("secret-source cache", lambda: cache.write("k", CachedFetch(secrets={"A": "b"}, fetched_at=1.0), 60, home),
         cache.path(home)),
        ("iron-proxy mappings", lambda: iron_proxy.write_mappings([]), home / "proxy" / "mappings.json"),
        ("exchanged JWT store", lambda: copilot_auth._save_jwt_to_disk("fp", "jwt", 9e12, None),
         copilot_auth._jwt_disk_path()),
        ("photon sidecar record", lambda: photon_adapter._write_runtime_record(1, "tok", 2), photon_record),
        ("pairing", lambda: pairing._save_json_file(home / "pairing" / "p.json", {"a": 1}), home / "pairing" / "p.json"),
        ("vault blob", lambda: vault._write_all([]), vault._vault_path),
        ("spawn ledger", lambda: process_identity.register_self("probe"), ledger),
        ("meet node token", meet_node.ensure_token, meet_node.token_path),
    ]


def test_every_credential_writer_creates_its_temp_file_at_0600(tmp_path, monkeypatch, opens_spy):
    pytest.importorskip("cryptography")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    old_umask = os.umask(0o022)  # a "write then chmod" regression would surface as 0o644
    try:
        for label, write, target in _writers(home, monkeypatch):
            del opens_spy[:]
            write()
            assert target.exists(), f"{label}: nothing written at {target}"
            assert stat.S_IMODE(target.stat().st_mode) == _PRIVATE, f"{label}: final file not 0600"
            creates = [(p, m) for p, fl, m in opens_spy if Path(p).parent == target.parent]
            assert creates, f"{label}: no temp file created in {target.parent}; opens={opens_spy!r}"
            for path, mode in creates:
                assert mode == _PRIVATE, f"{label}: temp file {path} created 0o{mode:o}, not 0600"
    finally:
        os.umask(old_umask)


def test_canonical_private_writers_round_trip_json_text_and_bytes(tmp_path):
    from utils import atomic_json_write, atomic_write_bytes, atomic_write_text

    target = tmp_path / "nested" / "creds.json"
    payload = {"token": "sk-\u00e9\u2603", "n": [1, 2]}
    atomic_json_write(target, payload, mode=0o600, fsync_dir=True)
    assert json.loads(target.read_text(encoding="utf-8")) == payload
    assert stat.S_IMODE(target.stat().st_mode) == 0o600

    atomic_write_text(target, "plain\n", mode=0o600)
    assert target.read_text(encoding="utf-8") == "plain\n"

    blob = bytes(range(256)) * 3
    atomic_write_bytes(target, blob, mode=0o600, fsync_dir=True)
    assert target.read_bytes() == blob
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert [p.name for p in target.parent.iterdir()] == ["creds.json"], "temp files must not survive"
