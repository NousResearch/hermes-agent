"""Binding hash and human attestation file.

The attestation is a tamper-EVIDENT record for honest operators, not a
security boundary: anyone who can write the repo or the attestation file can
forge it. It exists so `run` refuses after any runner/probe/image/policy/scope
change until a human re-proves containment.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path

from isolated_tests.policy import policy_hash, sha256_json
from isolated_tests.probe import PROBE_SHA256, all_passed

SCHEMA = "hermes-isolated-attestation/1"
RUNNER_DIR = Path(__file__).resolve().parent


class AttestationError(Exception):
    pass


def runner_hash() -> str:
    h = hashlib.sha256()
    for path in sorted(RUNNER_DIR.glob("*.py")):
        h.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return h.hexdigest()


def binding(*, image: str, python: str, socket: str, docker: str,
            manifest_sha256: str, suite: str, image_meta: dict, server: dict) -> dict:
    """Everything a re-proof must cover. Time is recorded, never bound."""
    return {
        "image_metadata": image_meta,
        "docker_server": server,
        "runner_sha256": runner_hash(),
        "probe_sha256": PROBE_SHA256,
        "policy_sha256": policy_hash(),
        "image": image,
        "container_python": python,
        "docker_socket": socket,
        "docker_binary": docker,
        "scope": {"manifest_sha256": manifest_sha256, "suite": suite},
    }


CONFIRM_MARKER = "human /dev/tty code x2"


def write(path: Path, bind: dict, probe: dict, probe_digest: str, wall_canary: dict,
          recorded_at: str) -> None:
    doc = {"schema": SCHEMA, "binding": bind, "binding_sha256": sha256_json(bind),
           "probe_results": probe, "probe_output_sha256": probe_digest,
           "wall_canary": wall_canary, "confirmed_by": CONFIRM_MARKER,
           "recorded_at": recorded_at}
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
    with os.fdopen(fd, "w") as fh:
        json.dump(doc, fh, sort_keys=True, indent=2)


def verify(path_raw: str, bind: dict) -> None:
    if not path_raw or not os.path.isabs(path_raw):
        raise AttestationError("attestation: absolute path required")
    if os.path.islink(path_raw) or not os.path.isfile(path_raw):
        raise AttestationError(f"attestation: missing or not a regular file: {path_raw}")
    try:
        doc = json.loads(Path(path_raw).read_bytes())
    except ValueError:
        raise AttestationError("attestation: not JSON") from None
    if not isinstance(doc, dict) or doc.get("schema") != SCHEMA:
        raise AttestationError("attestation: wrong schema")
    want = sha256_json(bind)
    if doc.get("binding_sha256") != want or doc.get("binding") != bind:
        changed = sorted(k for k in bind if (doc.get("binding") or {}).get(k) != bind[k])
        raise AttestationError(f"attestation: stale, re-run prove-human (changed: {changed})")
    check_structure(doc)


def check_structure(doc: dict) -> None:
    """Honest-corruption detection; a repo writer can still forge it."""
    if not all_passed(doc.get("probe_results") or {}):
        raise AttestationError("attestation: probe_results missing or not all true")
    digest = doc.get("probe_output_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise AttestationError("attestation: probe_output_sha256 missing")
    wc = doc.get("wall_canary") or {}
    if wc.get("timed_out") is not True or wc.get("cleanup_status") != "removed":
        raise AttestationError("attestation: wall-clock kill canary not proven")
    if doc.get("confirmed_by") != CONFIRM_MARKER or not isinstance(doc.get("recorded_at"), str):
        raise AttestationError("attestation: confirmation marker/time missing")


def new_code() -> str:
    return secrets.token_hex(4)
