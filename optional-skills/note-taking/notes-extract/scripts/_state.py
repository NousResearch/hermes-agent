"""Shared helpers for the notes-extract skill: HERMES_HOME state, identity, IO.

Stdlib-only and cross-platform. State lives under HERMES_HOME (never inside the
user's vault), mirroring the convention used by other Hermes skill scripts. All
text IO is UTF-8 and atomic (write temp + os.replace) so a crash mid-run never
leaves a half-written file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

STATE_VERSION = 1

# Section keys allowed per entity kind. Order is the canonical heading order.
PERSON_SECTIONS = ("facts", "commitments", "topics")
PROJECT_SECTIONS = ("ideas", "decisions", "blockers", "todos")


def get_hermes_home() -> Path:
    """Return the Hermes home directory (default: ~/.hermes).

    Mirrors hermes_constants.get_hermes_home() without requiring it on the path.
    """
    val = os.environ.get("HERMES_HOME", "").strip()
    return Path(val) if val else Path.home() / ".hermes"


def now_iso(clock=None) -> str:
    """UTC date stamp (YYYY-MM-DD). `clock` is injectable for deterministic tests."""
    dt = (clock or (lambda: datetime.now(timezone.utc)))()
    return dt.strftime("%Y-%m-%d")


def nfc(text: str) -> str:
    """Normalize to NFC (macOS stores filenames NFD; normalize everywhere)."""
    return unicodedata.normalize("NFC", text)


def norm_key(text: str) -> str:
    """Lowercased, whitespace-collapsed NFC key for alias/claim matching."""
    return re.sub(r"\s+", " ", nfc(text).strip().lower())


def slugify(name: str) -> str:
    """Filesystem-safe slug. Transliterate to ASCII; fall back to a hash."""
    decomposed = unicodedata.normalize("NFKD", nfc(name))
    ascii_only = "".join(c for c in decomposed if not unicodedata.combining(c))
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_only.lower()).strip("-")
    if not slug:
        slug = "n-" + hashlib.sha256(nfc(name).encode("utf-8")).hexdigest()[:8]
    return slug


def entity_id(kind: str, name: str) -> str:
    """Deterministic id derived from the canonical (first-seen) name."""
    digest = hashlib.sha256(norm_key(name).encode("utf-8")).hexdigest()[:10]
    return f"{kind}-{digest}"


def source_id(vault: Path, source_path: Path) -> str:
    """Stable id for a source note, keyed on its path relative to the vault."""
    try:
        rel = source_path.resolve().relative_to(vault.resolve()).as_posix()
    except ValueError:
        rel = source_path.resolve().as_posix()
    return "src-" + hashlib.sha256(rel.encode("utf-8")).hexdigest()[:12]


def entry_id(eid: str, section: str, claim: dict, src_id: str) -> str:
    """Stable entry id keyed on the CLAIM + source, not on prose.

    Rewording the same fact maps to the same id (no duplicate); a second source
    asserting the same claim yields a distinct id (provenance preserved).
    """
    parts = [
        eid,
        section,
        norm_key(claim.get("subject", "")),
        norm_key(claim.get("predicate", "")),
        norm_key(claim.get("object", "")),
        src_id,
    ]
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:8]
    return "nx-" + digest


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    """Write UTF-8 text atomically (temp file in same dir + os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def state_path(vault: Path) -> Path:
    """Per-vault state cache under HERMES_HOME, keyed by vault path hash."""
    key = hashlib.sha256(str(vault.resolve()).encode("utf-8")).hexdigest()[:16]
    return get_hermes_home() / "notes-extract" / f"{key}.json"


def empty_state(vault: Path) -> dict:
    return {
        "version": STATE_VERSION,
        "vault": str(vault.resolve()),
        "sources": {},      # source_id -> {path, link, sha, entries: [{id, entity_id, section}]}
        "entities": {},     # entity_id -> {kind, slug, name, aliases}
        "alias_index": {},  # norm_key(name/alias) -> entity_id
    }


def load_state(vault: Path) -> dict:
    p = state_path(vault)
    if not p.exists():
        return empty_state(vault)
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return empty_state(vault)
    if data.get("version") != STATE_VERSION:
        return empty_state(vault)
    return data


def save_state(vault: Path, state: dict) -> None:
    atomic_write_text(state_path(vault), json.dumps(state, indent=2, ensure_ascii=False))
