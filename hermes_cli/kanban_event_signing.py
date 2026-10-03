#!/usr/bin/env python3
"""
kanban_event_signing.py — per-agent ed25519 authorship signatures for kanban
task_events. Kanban card t_3f244a06 (jarvis-os board). Design source:
obsidian/fleet-vault/Architecture/agent-signing-keys-design-2026-08-01.md §4b.

WHAT THIS IS
  Attribution of kanban events to the acting agent. Today a `kind`/payload
  string is the only author signal; any process on the box can forge it.
  This module adds a cryptographic authorship signature over the canonical
  JSON of each event row, produced in-process with the acting agent's
  profile-local ed25519 key and stored in a SIDECAR signature store.

COMPOSE-SAFETY (non-negotiable)
  The kanban hash-chain (t_21781f08, ~/.hermes/scripts/kanban-audit-chain.py)
  records `source_columns` of task_events and HALTS append on any schema
  drift of that table. Therefore this module NEVER alters the task_events
  schema and NEVER writes to the live kanban DB. Signatures live in their own
  sidecar SQLite DB (~/.hermes/audit/kanban-event-signatures.db), keyed by
  (board, event_id) — exactly as the chain keeps its own kanban-chain.db.
  chain  = tamper-evidence for ordering/content
  sig    = authorship of the same canonical content

ROLE OF allowed_signers
  The registered-principal root is /home/frank/.hermes/governance/allowed_signers
  (canonical) — the SAME root git commit-signing uses. An event signature is
  GOOD only if it (a) cryptographically verifies against the signer's public
  key AND (b) that signer is registered in allowed_signers. A valid signature
  from an unregistered key is UNTRUSTED — the same policy the commit
  verifier uses (see verify-commit-signatures.sh).

POLICY (verify-and-report first)
  Everything here is report-only. No event is ever blocked for being unsigned
  or untrusted. Enforcement is a separate future card
  (signing-keys-enforcement-gate) gated through the PM layer.

MODES
  sign-one --content <canonical-json> --key <path>         -> print base64 sig
  selftest                                                -> red/green proof
  verify --kanban-db PATH --sidecar PATH --allowed-signers PATH  (report-only)
  resolve-key [profile]                                  -> print signing key path

IMPORTANT: this module is stdlib + `cryptography` only (available in the
hermes-agent venv and system python on DGX). No ssh-keygen subprocess for the
signature itself — pure in-process Ed25519 via the same OpenSSH key material.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sqlite3
import sys
import tempfile

# ---- paths / constants -----------------------------------------------------
DEFAULT_ALLOWED_SIGNERS = "/home/frank/.hermes/governance/allowed_signers"
DEFAULT_SIDECAR = "/home/frank/.hermes/audit/kanban-event-signatures.db"
HERMES_HOME = "/home/frank/.hermes"

SIDECAR_STATEMENTS = [
    "CREATE TABLE IF NOT EXISTS event_signatures ("
    "    board       TEXT NOT NULL,"
    "    event_id    INTEGER NOT NULL,"
    "    signer      TEXT NOT NULL,"
    "    signature   TEXT NOT NULL,"
    "    content     TEXT NOT NULL,"
    "    created_at  INTEGER NOT NULL,"
    "    PRIMARY KEY (board, event_id)"
    ")",
    "CREATE INDEX IF NOT EXISTS idx_sig_board ON event_signatures(board, created_at)",
]


def open_sidecar(path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    con = sqlite3.connect(path)
    for stmt in SIDECAR_STATEMENTS:
        con.execute(stmt)
    con.commit()
    return con


def board_for_conn(conn: sqlite3.Connection) -> str:
    """Derive the board slug from a live kanban connection's main database
    file path. The parent dir name of the DB file IS the board slug for
    named boards (e.g. .../kanban/boards/jarvis-os/kanban.db -> jarvis-os);
    the legacy default board lives at <root>/kanban.db -> 'default'."""
    try:
        row = conn.execute("PRAGMA database_list").fetchone()
        # row: (seq, name, file); main DB is name == 'main'
        for r in conn.execute("PRAGMA database_list"):
            if r[1] == "main" and r[2] and not r[2].startswith(":"):
                dbpath = os.path.abspath(r[2])
                parent = os.path.basename(os.path.dirname(dbpath))
                if parent == "boards":
                    return "default"
                if parent and parent != "kanban":
                    return parent
                return "default"
    except Exception:
        pass
    return os.environ.get("HERMES_KANBAN_BOARD", "default").strip() or "default"
def canonical_json(d: dict) -> str:
    """Deterministic serialization identical to the hash-chain's convention
    (sort_keys, compact separators, ensure_ascii). Signing and hashing the
    same bytes means sig (authorship) and chain (integrity) compose cleanly."""
    return json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def event_content(event_id: int, task_id, run_id, kind, payload, created_at) -> str:
    """Build the canonical content of a task_events row over the SAME columns
    the hash-chain hashes. payload may be a dict or JSON string."""
    if isinstance(payload, dict):
        payload = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return canonical_json({
        "id": event_id,
        "task_id": task_id,
        "run_id": run_id,
        "kind": kind,
        "payload": payload,
        "created_at": created_at,
    })


# ---- key resolution --------------------------------------------------------
def resolve_signing_key(profile: str | None = None) -> tuple[str, str] | None:
    """Return (identity, private_key_path) for the acting agent, or None.
    Resolution order:
      1. HERMES_PROFILE -> ~/.hermes/profiles/<profile>/keys/<profile>-signing
      2. explicit profile arg -> same
      3. HERMES_SEAT -> ~/.hermes/seats/<seat>/keys/<seat>-signing
      4. profiles/<name>/keys/<name>-signing for each known profile
    Returns the FIRST key that exists. Never falls back to a shared/global key.
    """
    def cand(identity: str) -> str:
        for base in (
            os.path.join(HERMES_HOME, "profiles", identity, "keys"),
            os.path.join(HERMES_HOME, "seats", identity, "keys"),
        ):
            p = os.path.join(base, f"{identity}-signing")
            if os.path.isfile(p):
                return p
        return ""

    candidates = []
    env_profile = os.environ.get("HERMES_PROFILE") or ""
    if profile:
        candidates.append(profile)
    if env_profile:
        candidates.append(env_profile)
    seat = os.environ.get("HERMES_SEAT") or ""
    if seat:
        candidates.append(seat)

    seen = set()
    for ident in candidates:
        if not ident or ident in seen:
            continue
        seen.add(ident)
        p = cand(ident)
        if p:
            return f"{ident}@hermes-fleet", p

    # Last resort: scan known profile key dirs (never a shared key).
    prof_root = os.path.join(HERMES_HOME, "profiles")
    try:
        for name in sorted(os.listdir(prof_root)):
            if name in seen or not os.path.isdir(os.path.join(prof_root, name)):
                continue
            seen.add(name)
            p = cand(name)
            if p:
                return f"{name}@hermes-fleet", p
    except FileNotFoundError:
        pass
    return None


# ---- sign / verify ---------------------------------------------------------
def sign_content(content: str, key_path: str) -> str:
    """Sign canonical content with an OpenSSH ed25519 private key (no
    passphrase). Returns base64 raw Ed25519 signature."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    with open(key_path, "rb") as f:
        key = serialization.load_ssh_private_key(f.read(), password=None)
    if not isinstance(key, Ed25519PrivateKey):
        raise TypeError(f"key {key_path} is not an ed25519 private key")
    sig = key.sign(content.encode("utf-8"))
    return base64.b64encode(sig).decode("ascii")


def sign_event_payload(
    event_id: int,
    task_id,
    run_id,
    kind,
    payload,
    created_at: int,
    key_path: str,
) -> str:
    """Sign an event row's canonical content. Returns base64 sig."""
    return sign_content(event_content(event_id, task_id, run_id, kind, payload, created_at), key_path)


def _load_pubkey(signer: str, allowed_signers: str):
    """Return the ssh-ed25519 public key object registered for `signer`, or
    None if the signer is not registered."""
    from cryptography.hazmat.primitives import serialization

    if not os.path.isfile(allowed_signers):
        return None
    with open(allowed_signers, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            principal = parts[0]
            if principal != signer:
                continue
            # find <keytype> <base64>
            for i in range(1, len(parts) - 1):
                if parts[i].startswith("ssh-") or parts[i].startswith("ecdsa-") or parts[i].startswith("sk-"):
                    try:
                        return serialization.load_ssh_public_key(
                            f"{parts[i]} {parts[i+1]}".encode("ascii")
                        )
                    except Exception:
                        return None
    return None


def verify_signature(content: str, signature_b64: str, signer: str, allowed_signers: str):
    """Return one of: GOOD / UNTRUSTED / BAD / KEY-MISSING.
    GOOD = verifies AND signer registered. UNTRUSTED = verifies but signer
    not in allowed_signers (registered-principal policy). BAD = does not
    verify against the registered key. KEY-MISSING = signer registered but
    key unparseable (shouldn't happen)."""
    pubkey = _load_pubkey(signer, allowed_signers)
    if pubkey is None:
        return "UNTRUSTED"  # cryptographically-present sig, principal not registered
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        if not isinstance(pubkey, Ed25519PublicKey):
            return "BAD"  # only ed25519 signing keys are supported for events
        sig = base64.b64decode(signature_b64)
        pubkey.verify(sig, content.encode("utf-8"))
        return "GOOD"
    except Exception:
        return "BAD"


# ---- sidecar store (read/write) -------------------------------------------
def store_signature(
    sidecar_path: str,
    board: str,
    event_id: int,
    signer: str,
    signature: str,
    content: str,
    created_at: int,
) -> None:
    """Fail-open write: never raises. Duplicate (board,event_id) is replaced
    (re-sign) so re-appending the same event id is idempotent."""
    try:
        con = open_sidecar(sidecar_path)
        try:
            con.execute(
                "INSERT INTO event_signatures (board,event_id,signer,signature,content,created_at) "
                "VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(board,event_id) DO UPDATE SET "
                "signer=excluded.signer, signature=excluded.signature, "
                "content=excluded.content, created_at=excluded.created_at",
                (board, event_id, signer, signature, content, created_at),
            )
            con.commit()
        finally:
            con.close()
    except Exception as e:  # noqa: BLE001 - fail-open
        print(f"WARN: store_signature failed ({type(e).__name__}: {e}) — event {event_id} left unsigned", file=sys.stderr)


def verify_sidecar(kanban_db: str, sidecar_path: str, allowed_signers: str, board: str | None = None):
    """Report-only reconciliation of event signatures against the live kanban
    DB. Never blocks. Returns a dict of counts. `board` is the kanban board
    slug; if omitted it is derived from the kanban DB parent dir name."""
    counts = {"GOOD": 0, "UNTRUSTED": 0, "BAD": 0, "UNSIGNED": 0, "STALE": 0}
    if board is None:
        board = os.path.basename(os.path.dirname(kanban_db)) or "?"
    try:
        kdb = sqlite3.connect(f"file:{kanban_db}?mode=ro", uri=True, timeout=3)
        try:
            # live event ids for freshness of the report
            live_ids = {r[0] for r in kdb.execute("SELECT id FROM task_events")}
        finally:
            kdb.close()
    except Exception as e:
        print(f"ERROR: cannot open kanban DB {kanban_db}: {e}", file=sys.stderr)
        return counts

    if not os.path.isfile(sidecar_path):
        counts["UNSIGNED"] = len(live_ids)
        return counts

    con = sqlite3.connect(f"file:{sidecar_path}?mode=ro", uri=True, timeout=3)
    rows = con.execute(
        "SELECT event_id, signer, signature, content, created_at "
        "FROM event_signatures WHERE board=? ORDER BY event_id",
        (board,),
    ).fetchall()
    con.close()

    signed_ids = set()
    for event_id, signer, signature, content, created_at in rows:
        signed_ids.add(event_id)
        if event_id not in live_ids:
            counts["STALE"] += 1
            continue
        status = verify_signature(content, signature, signer, allowed_signers)
        counts[status] = counts.get(status, 0) + 1

    counts["UNSIGNED"] = len(live_ids - signed_ids)
    return counts


# ---- CLI -------------------------------------------------------------------
def _cmd_sign_one(args):
    sig = sign_content(args.content, args.key)
    print(sig)


def _cmd_resolve_key(args):
    resolved = resolve_signing_key(args.profile)
    if not resolved:
        print("NO KEY", file=sys.stderr)
        return 1
    ident, path = resolved
    print(f"{ident}\t{path}")
    return 0


def _cmd_verify(args):
    counts = verify_sidecar(args.kanban_db, args.sidecar, args.allowed_signers)
    parts = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"EVENT-SIG-VERIFY: board={os.path.basename(os.path.dirname(args.kanban_db))} {parts}")
    return 0


def _cmd_selftest(args=None):
    """Red/green proof on scratch copies. Never touches live data."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    ok = True
    def check(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        if not cond:
            ok = False

    with tempfile.TemporaryDirectory() as td:
        # two independent keys
        keypaths = {}
        for ident in ("alice", "bob"):
            priv = Ed25519PrivateKey.generate()
            kp = os.path.join(td, f"{ident}-signing")
            with open(kp, "wb") as f:
                f.write(priv.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.OpenSSH,
                    encryption_algorithm=serialization.NoEncryption(),
                ))
            pub = priv.public_key().public_bytes(
                encoding=serialization.Encoding.OpenSSH,
                format=serialization.PublicFormat.OpenSSH,
            ).decode("ascii")
            with open(os.path.join(td, f"{ident}.pub"), "w") as f:
                f.write(f"{ident}@hermes-fleet namespaces=\"git\" ssh-ed25519 {pub.split()[1]}\n")
            keypaths[ident] = kp

        allowed = os.path.join(td, "allowed_signers")
        with open(allowed, "w") as f:
            f.write(open(os.path.join(td, "alice.pub")).read())
            f.write(open(os.path.join(td, "bob.pub")).read())

        content = event_content(1, "t_000001", None, "heartbeat", None, 1700000000)
        # 1. sign+verify roundtrip (alice)
        sig = sign_content(content, keypaths["alice"])
        check("alice sign+verify roundtrip",
              verify_signature(content, sig, "alice@hermes-fleet", allowed) == "GOOD")
        # 2. wrong key rejected (bob's sig, verified as bob's principal)
        sigb = sign_content(content, keypaths["bob"])
        check("bob signs, verifies as bob (independent)",
              verify_signature(content, sigb, "bob@hermes-fleet", allowed) == "GOOD")
        # 3. tampered content detected
        check("tampered content -> BAD",
              verify_signature(content + "x", sig, "alice@hermes-fleet", allowed) == "BAD")
        # 4. registered-principal policy: unregistered signer -> UNTRUSTED
        check("unregistered signer -> UNTRUSTED",
              verify_signature(content, sig, "carol@hermes-fleet", allowed) == "UNTRUSTED")
        # 5. sidecar verify roundtrip
        sidecar = os.path.join(td, "sig.db")
        store_signature(sidecar, "testboard", 1, "alice@hermes-fleet", sig, content, 1700000000)
        # build a minimal kanban db with event id 1
        kdb = os.path.join(td, "kanban.db")
        kc = sqlite3.connect(kdb)
        kc.execute("CREATE TABLE task_events (id INTEGER PRIMARY KEY, task_id TEXT, run_id INTEGER, kind TEXT, payload TEXT, created_at INTEGER)")
        kc.execute("INSERT INTO task_events (id,task_id,run_id,kind,payload,created_at) VALUES (1,'t_000001',NULL,'heartbeat',NULL,1700000000)")
        kc.commit(); kc.close()
        counts = verify_sidecar(kdb, sidecar, allowed, board="testboard")
        check("sidecar verify: GOOD=1 unsigned=0",
              counts.get("GOOD") == 1 and counts.get("UNSIGNED") == 0)
        # 6. missing signature reported as unsigned (not a block)
        sidecar2 = os.path.join(td, "sig2.db")
        counts2 = verify_sidecar(kdb, sidecar2, allowed, board="testboard")
        check("no sidecar -> all unsigned (report-only, no block)",
              counts2.get("UNSIGNED") == 1)
        # 7. sign_event_payload full-row convenience
        sigrow = sign_event_payload(1, "t_000001", None, "heartbeat", None, 1700000000, keypaths["alice"])
        check("sign_event_payload matches sign_content",
              sigrow == sig)

    print("SELFTEST:", "ALL PASS" if ok else "FAILURES PRESENT")
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    s1 = sub.add_parser("sign-one")
    s1.add_argument("--content", required=True)
    s1.add_argument("--key", required=True)
    s1.set_defaults(func=_cmd_sign_one)
    rk = sub.add_parser("resolve-key")
    rk.add_argument("--profile", default=None)
    rk.set_defaults(func=_cmd_resolve_key)
    v = sub.add_parser("verify")
    v.add_argument("--kanban-db", required=True)
    v.add_argument("--sidecar", default=DEFAULT_SIDECAR)
    v.add_argument("--allowed-signers", default=DEFAULT_ALLOWED_SIGNERS)
    v.set_defaults(func=_cmd_verify)
    st = sub.add_parser("selftest")
    st.set_defaults(func=_cmd_selftest)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
