"""At-rest encryption for sensitive files via the `age` binary (optional).

Engaged ONLY when config `encryption: age` AND an age identity key exists AND the
`age`/`age-keygen` binaries are available. When engaged, JSON docs under
`subjects/` (dossier, ledger) are written as `<file>.age` ciphertext; the audit
log (field NAMES + states only, no raw PII values), `config.json`, and the broker
cache stay plaintext so the engine can read them.

Threat model (be honest): this protects against casual disk inspection, accidental
`git add`/commits, screen-shares, and backup/cloud-sync leakage. The identity key
defaults to living beside the data at `$PDD_DATA_DIR/age-identity.txt` (0600); set
`PDD_AGE_IDENTITY` to a separate volume/token for true key separation. It does NOT
protect against an attacker who can already read your whole HERMES_HOME (they get
key + data together).
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from shutil import which

import paths


def age_available() -> bool:
    return which("age") is not None and which("age-keygen") is not None


def encryption_setting() -> str:
    """Read `encryption` straight from config.json (no config/storage import => no cycle)."""
    cfg = paths.config_path()
    if not cfg.exists():
        return "none"
    try:
        return (json.loads(cfg.read_text(encoding="utf-8")) or {}).get("encryption", "none")
    except (ValueError, OSError):
        return "none"


def identity_path() -> Path:
    return paths.age_identity_path()


def ensure_identity() -> Path:
    """Generate an age identity (X25519 keypair) if missing; return its path."""
    if not age_available():
        raise RuntimeError("`age`/`age-keygen` not found; cannot enable encryption")
    p = identity_path()
    if not p.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            p.parent.chmod(0o700)
        except OSError:
            pass
        subprocess.run(["age-keygen", "-o", str(p)], check=True, capture_output=True)
        try:
            p.chmod(0o600)
        except OSError:
            pass
    return p


def recipient() -> str:
    """The age public key (recipient) for the identity, parsed from its header."""
    p = ensure_identity()
    for line in p.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.lower().startswith("# public key:"):
            return s.split(":", 1)[1].strip()
        if s.startswith("age1"):
            return s
    raise RuntimeError(f"no public key found in {p}")


def is_engaged() -> bool:
    """True only when encryption is actually active (configured + available + key present)."""
    return encryption_setting() == "age" and age_available() and identity_path().exists()


def encrypt(data: bytes) -> bytes:
    out = subprocess.run(["age", "-r", recipient()], input=data, capture_output=True, check=True)
    return out.stdout


def decrypt(data: bytes) -> bytes:
    out = subprocess.run(["age", "-d", "-i", str(identity_path())], input=data, capture_output=True, check=True)
    return out.stdout
