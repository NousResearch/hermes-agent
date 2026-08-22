"""Activate reviewed builder proposals into immutable per-cycle governance."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path

from .attestation import GovernanceSnapshot
from .canonical import canonical_sha256
from .errors import AdapterError
from .preparation import inspect_repository
from .runtime import RuntimeSettings, _read_owner_json


def _run_git(repository: Path, *args: str, env_extra: dict[str, str] | None = None) -> str:
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_ASKPASS": "/bin/false",
        "SSH_ASKPASS": "/bin/false",
    }
    if env_extra:
        env.update(env_extra)
    try:
        result = subprocess.run(
            ["/usr/bin/git", "--no-pager", "-C", str(repository), *args],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise AdapterError("INTERNAL_ERROR", "activation Git operation failed") from exc
    return result.stdout.strip()


def load_proposal(path: str | Path) -> dict:
    proposal = _read_owner_json(Path(path).expanduser(), exact_mode=0o600)
    claimed = proposal.get("bundle_sha256")
    material = dict(proposal)
    material.pop("bundle_sha256", None)
    if not isinstance(claimed, str) or canonical_sha256(material) != claimed:
        raise AdapterError("CONTRACT_MISMATCH", "proposal hash does not match its contents")
    if (
        proposal.get("schema_version") != "1.0.0"
        or proposal.get("bundle_kind") != "hermes.builder_job_proposal"
    ):
        raise AdapterError("INVALID_REQUEST", "unsupported proposal format")
    return proposal


def _write_new_json(root: Path, relative: str, value: dict) -> bytes:
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        raise AdapterError("CONTRACT_MISMATCH", "governance artifact already exists")
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    descriptor = os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return raw


def _atomic_runtime_update(config_path: Path, value: dict) -> None:
    parent = config_path.parent
    descriptor, temporary = tempfile.mkstemp(prefix=".runtime-activate-", dir=parent)
    try:
        os.fchmod(descriptor, 0o600)
        raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
        os.write(descriptor, raw)
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = -1
        os.replace(temporary, config_path)
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def activate_proposal(config_path: str | Path, proposal_path: str | Path) -> dict:
    config_path = Path(config_path).expanduser()
    settings = RuntimeSettings.from_file(config_path)
    runtime = _read_owner_json(config_path, exact_mode=None)
    proposal = load_proposal(proposal_path)
    cycle_id = proposal["cycle_id"]
    contract_id = proposal["contract_id"]
    repository_value = proposal["repository"]

    if cycle_id in settings.cycle_registry:
        raise AdapterError("CONTRACT_MISMATCH", "cycle is already registered")
    registered_remote = settings.repository_allowlist.get(repository_value["repository_id"])
    if registered_remote != repository_value["canonical_remote"]:
        raise AdapterError("REPOSITORY_MISMATCH", "repository is not allowlisted")
    source = inspect_repository(repository_value["source_root"])
    if (
        source.remote != repository_value["canonical_remote"]
        or source.head != repository_value["base_sha"]
    ):
        raise AdapterError("HEAD_MISMATCH", "proposal source repository has changed")

    governance = settings.governance_repo.resolve(strict=True)
    if _run_git(governance, "status", "--porcelain=v1"):
        raise AdapterError("CONTRACT_MISMATCH", "governance repository is not clean")
    if _run_git(governance, "rev-parse", "HEAD") != settings.governance_commit:
        raise AdapterError("CONTRACT_MISMATCH", "governance HEAD is not the approved snapshot")

    safe_name = re.sub(r"[^A-Z0-9_-]", "_", contract_id)
    manifest_registered = f"contracts/manifests/{safe_name}.allowed-paths.json"
    contract_registered = f"contracts/active/{safe_name}.json"
    manifest = copy.deepcopy(proposal["allowed_path_manifest"])
    manifest.update(
        {
            "schema_version": "1.0.0",
            "manifest_id": f"{safe_name}-ALLOWED-PATHS",
            "repository_id": repository_value["repository_id"],
            "base_sha": repository_value["base_sha"],
            "live_execution_affected": False,
        }
    )
    manifest["rules"] = [
        {**rule, "reason": "Explicitly approved in the job proposal"}
        for rule in manifest["rules"]
    ]

    template = GovernanceSnapshot(governance, settings.governance_commit)
    try:
        from jsonschema import Draft202012Validator

        errors = list(
            Draft202012Validator(template.value("allowed_path_manifest_schema")).iter_errors(
                manifest
            )
        )
    except ImportError as exc:
        raise AdapterError("CONTRACT_MISMATCH", "JSON Schema validator unavailable") from exc
    if errors:
        raise AdapterError("MANIFEST_MISMATCH", "generated allowed-path manifest is invalid")

    prefix = GovernanceSnapshot.PREFIX
    manifest_raw = _write_new_json(governance, prefix + manifest_registered, manifest)
    contract = copy.deepcopy(template.contract)
    contract["contract_id"] = contract_id
    contract["title"] = proposal["objective"]["summary"]
    contract["objective"] = {
        "summary": proposal["objective"]["summary"],
        "motivation": "Prepared through the governed Hermes operator workflow.",
        "success_criteria": proposal["objective"]["success_criteria"],
    }
    contract["acceptance_criteria"] = proposal["objective"]["success_criteria"]
    contract["artifact_bindings"]["allowed_path_manifest"] = {
        "path": manifest_registered,
        "sha256": hashlib.sha256(manifest_raw).hexdigest(),
    }
    _write_new_json(governance, prefix + contract_registered, contract)
    _run_git(
        governance,
        "add",
        "--",
        prefix + manifest_registered,
        prefix + contract_registered,
    )
    identity = {
        "GIT_AUTHOR_NAME": "Hermes Builder Orchestrator",
        "GIT_AUTHOR_EMAIL": "hermes-builder@localhost",
        "GIT_COMMITTER_NAME": "Hermes Builder Orchestrator",
        "GIT_COMMITTER_EMAIL": "hermes-builder@localhost",
    }
    _run_git(
        governance,
        "commit",
        "-m",
        f"governance: activate {cycle_id}",
        env_extra=identity,
    )
    governance_commit = _run_git(governance, "rev-parse", "HEAD")
    GovernanceSnapshot(
        governance,
        governance_commit,
        registered_contract_path=contract_registered,
    )

    worktree = Path(repository_value["planned_worktree"])
    worktree.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _run_git(
        Path(source.root),
        "worktree",
        "add",
        "-b",
        repository_value["planned_branch"],
        str(worktree),
        repository_value["base_sha"],
    )

    cycle = {
        "revision": proposal["cycle_revision"],
        "contract_id": contract_id,
        "repository_id": repository_value["repository_id"],
        "governance_repository_id": "ai-engineering-orchestrator",
        "governance_commit": governance_commit,
        "contract_path": contract_registered,
        "canonical_remote": repository_value["canonical_remote"],
        "worktree_path": str(worktree),
        "branch": repository_value["planned_branch"],
        "expected_head_sha": repository_value["base_sha"],
        "validation_profile_id": proposal["validation_profile_id"],
        "timeout_policy": proposal["timeout_policy"],
        "retry_policy": proposal["retry_policy"],
        "proposal_sha256": proposal["bundle_sha256"],
    }
    runtime["governance_commit"] = governance_commit
    runtime["cycle_registry"][cycle_id] = cycle
    _atomic_runtime_update(config_path, runtime)
    return {
        "cycle_id": cycle_id,
        "contract_id": contract_id,
        "governance_commit": governance_commit,
        "contract_path": contract_registered,
        "worktree_path": str(worktree),
        "branch": repository_value["planned_branch"],
        "proposal_sha256": proposal["bundle_sha256"],
        "state": "ACTIVATED_RESTART_REQUIRED",
    }
