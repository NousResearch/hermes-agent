"""Filesystem paths for the unbroker skill (stdlib only).

All per-subject data lives under PDD_DATA_DIR (default: $HERMES_HOME/unbroker),
which is the same trust boundary Hermes uses for .env and OAuth tokens.
"""
from __future__ import annotations

import os
from pathlib import Path


def hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


def data_dir() -> Path:
    override = os.environ.get("PDD_DATA_DIR")
    return Path(override) if override else hermes_home() / "unbroker"


def config_path() -> Path:
    return data_dir() / "config.json"


def subjects_dir() -> Path:
    return data_dir() / "subjects"


def subject_dir(subject_id: str) -> Path:
    return subjects_dir() / subject_id


def dossier_path(subject_id: str) -> Path:
    return subject_dir(subject_id) / "dossier.json"


def ledger_path(subject_id: str) -> Path:
    return subject_dir(subject_id) / "ledger.json"


def audit_path(subject_id: str) -> Path:
    return subject_dir(subject_id) / "audit.jsonl"


def evidence_dir(subject_id: str) -> Path:
    return subject_dir(subject_id) / "evidence"


def skill_root() -> Path:
    """The skill directory (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def brokers_dir() -> Path:
    return skill_root() / "references" / "brokers"


def brokers_cache_path() -> Path:
    """Live broker snapshot pulled from BADBOOL (merged under the curated DB)."""
    return data_dir() / "brokers-cache" / "badbool.json"


def registry_cache_path() -> Path:
    """CA Data Broker Registry snapshot (separate coverage lane; DROP/email, not scanned)."""
    return data_dir() / "brokers-cache" / "ca-registry.json"


def age_identity_path() -> Path:
    """age identity (private key) used for at-rest encryption when enabled.

    Defaults beside the data; point PDD_AGE_IDENTITY at a separate volume/token
    for real key separation from the encrypted data.
    """
    override = os.environ.get("PDD_AGE_IDENTITY")
    return Path(override) if override else data_dir() / "age-identity.txt"


def templates_dir() -> Path:
    return skill_root() / "templates"
