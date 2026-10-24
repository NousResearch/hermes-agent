"""Narrow trusted validation receipts for Phase 1 mission completion.

Checks are fixed in this module. Callers choose artifact paths only; they
cannot provide a shell command, exit code, contract, or reviewer identity.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from hermes_cli import mission_ownership as missions


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=root, text=True, capture_output=True,
        timeout=15, check=False,
    )


def _repo_root() -> tuple[Path, str]:
    cwd = Path.cwd().resolve()
    root_result = _git(cwd, "rev-parse", "--show-toplevel")
    if root_result.returncode:
        raise missions.ContractError("trusted validation requires a Git checkout")
    root = Path(root_result.stdout.strip()).resolve()
    head = _git(root, "rev-parse", "HEAD")
    if head.returncode:
        raise missions.ContractError("trusted validation could not read the checkout revision")
    return root, head.stdout.strip()


def _artifact(root: Path, value: str) -> dict[str, Any]:
    path = (root / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise missions.ContractError("validation artifacts must be inside the active Git checkout") from exc
    if not path.is_file():
        raise missions.ContractError(f"validation artifact is missing or not a file: {relative}")
    current = path.read_bytes()
    tracked = _git(root, "ls-files", "--error-unmatch", "--", relative).returncode == 0
    before = _git(root, "show", f"HEAD:{relative}") if tracked else None
    before_hash = hashlib.sha256(before.stdout.encode()).hexdigest() if before and before.returncode == 0 else None
    current_hash = hashlib.sha256(current).hexdigest()
    if tracked and before_hash == current_hash:
        raise missions.ContractError(f"validation artifact has no change from HEAD: {relative}")
    return {"path": relative, "sha256": current_hash, "before_sha256": before_hash,
            "created": not tracked, "size": len(current)}


def _run_checks(root: Path, artifacts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    python_paths = [a["path"] for a in artifacts if Path(a["path"]).suffix.lower() == ".py"]
    if not python_paths:
        raise missions.ContractError("trusted validation requires at least one changed Python artifact")

    checks: list[dict[str, Any]] = []
    syntax_ok = True
    syntax_error = ""
    for relative in python_paths:
        try:
            source = (root / relative).read_text(encoding="utf-8")
            compile(source, relative, "exec", dont_inherit=True)
        except Exception as exc:
            syntax_ok = False
            syntax_error = f"{relative}: {type(exc).__name__}: {exc}"
            break
    checks.append({"check_id": "python-syntax", "exit_code": 0 if syntax_ok else 1,
                   "output": "Python artifacts parsed" if syntax_ok else syntax_error})

    result = subprocess.run(
        [sys.executable, "-m", "ruff", "check", *python_paths],
        cwd=root, text=True, capture_output=True, timeout=45, check=False,
    )
    checks.append({"check_id": "ruff", "exit_code": result.returncode,
                   "output": (result.stdout + result.stderr)[-12000:]})
    if any(check["exit_code"] != 0 for check in checks):
        raise missions.ContractError("trusted checks failed: " + ", ".join(
            f"{check['check_id']}={check['exit_code']}" for check in checks
        ))
    return checks


def validate(
    conn, *, mission_id: str, owner_kind: str, owner_id: str, generation: int,
    contract_version: int, artifact_paths: list[str], reviewer_identity: str,
) -> dict[str, Any]:
    """Run the fixed syntax/Ruff checks and persist their current-source receipt."""
    missions.ensure_schema(conn)
    mission = missions.get(conn, mission_id)
    if mission is None:
        raise missions.StaleOwnerError("mission does not exist")
    missions._assert_owner(
        conn.execute("SELECT * FROM missions WHERE id=?", (mission_id,)).fetchone(),
        owner_kind, owner_id, generation,
    )
    if mission["contract_version"] != int(contract_version):
        raise missions.ContractError("validation request targets a stale contract version")
    if not isinstance(artifact_paths, list) or not artifact_paths:
        raise missions.ContractError("validation requires one or more artifact paths")
    root, head_sha = _repo_root()
    artifacts = [_artifact(root, str(value)) for value in artifact_paths]
    checks = _run_checks(root, artifacts)
    # Detect concurrent edits while checks ran; never certify a moving tree.
    for artifact in artifacts:
        actual = hashlib.sha256((root / artifact["path"]).read_bytes()).hexdigest()
        if actual != artifact["sha256"]:
            raise missions.ContractError(f"artifact changed while trusted checks ran: {artifact['path']}")

    verifier_id = "hermes-phase1-python-validator:" + hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    receipt_id = "v_" + uuid.uuid4().hex
    contract_sha = hashlib.sha256(missions._json(mission["contract"]).encode()).hexdigest()
    now = int(time.time())
    with missions._write_txn(conn):
        fresh = conn.execute("SELECT * FROM missions WHERE id=?", (mission_id,)).fetchone()
        missions._assert_owner(fresh, owner_kind, owner_id, generation)
        if int(fresh["contract_version"]) != int(contract_version):
            raise missions.ContractError("mission contract changed while trusted checks ran")
        fresh_contract = json.loads(fresh["contract_json"])
        fresh_hash = hashlib.sha256(missions._json(fresh_contract).encode()).hexdigest()
        if fresh_hash != contract_sha:
            raise missions.ContractError("mission contract changed while trusted checks ran")
        for artifact in artifacts:
            if hashlib.sha256((root / artifact["path"]).read_bytes()).hexdigest() != artifact["sha256"]:
                raise missions.ContractError(f"artifact changed while receipt was being recorded: {artifact['path']}")
        conn.execute(
            "INSERT INTO mission_validations VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (receipt_id, mission_id, generation, contract_version, contract_sha,
             str(owner_id), str(reviewer_identity), verifier_id, str(root), head_sha,
             missions._json(artifacts), missions._json(checks), now),
        )
    return {"validation_id": receipt_id, "mission_id": mission_id,
            "contract_version": contract_version, "contract_sha256": contract_sha,
            "reviewer_identity": str(reviewer_identity), "verifier_id": verifier_id,
            "repository_root": str(root), "head_sha": head_sha,
            "artifacts": artifacts, "checks": checks}
