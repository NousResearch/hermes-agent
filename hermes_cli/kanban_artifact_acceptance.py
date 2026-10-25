"""Trusted, content-addressed acceptance for ``artifact-review-v1`` tasks.

The worker-supplied receipt is only a candidate.  A verifier command selected
in operator-owned ``config.yaml`` receives that candidate plus board-owned
task, run, reviewer and artifact facts.  Only its normalized, bound verdict is
eligible for the terminal transaction.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import yaml

from hermes_constants import get_hermes_home


CONTRACT = "artifact-review-v1"
SCHEMA = "hermes.artifact-review-acceptance/v1"
_TRUSTED_MANAGED_DIR = Path("/etc/hermes")
_TRUSTED_OWNER_UID = 0


def _failure(classification: str, detail: str) -> dict[str, Any]:
    return {
        "ok": False,
        "classification": classification,
        "detail": detail,
        "recovery": (
            "Preserve the deliverable as a review attachment, obtain an independent "
            "review using the configured verifier, and retry completion."
        ),
    }


def _json_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _configured_command() -> tuple[list[str], float, dict[str, list[str]]] | None:
    # Acceptance authority is deliberately stricter than the general managed
    # overlay.  HERMES_MANAGED_DIR is useful for ordinary configuration and
    # tests, but it is inherited by workers and therefore cannot select a
    # terminal verifier.  The bootstrap-pinned directory is the trust root.
    managed_dir = _TRUSTED_MANAGED_DIR
    config_path = managed_dir / "config.yaml"
    try:
        managed_resolved = managed_dir.resolve(strict=True)
        config_resolved = config_path.resolve(strict=True)
        dir_stat, cfg_stat = managed_resolved.stat(), config_resolved.stat()
    except OSError:
        return None
    if (
        managed_dir.is_symlink()
        or config_path.is_symlink()
        or managed_resolved != managed_dir.resolve()
        or config_resolved.parent != managed_resolved
        or not managed_resolved.is_dir()
        or not config_resolved.is_file()
        or dir_stat.st_uid != _TRUSTED_OWNER_UID
        or cfg_stat.st_uid != _TRUSTED_OWNER_UID
        or (dir_stat.st_mode & 0o022)
        or (cfg_stat.st_mode & 0o022)
    ):
        return None
    try:
        loaded = yaml.safe_load(config_resolved.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError):
        return None
    cfg = (loaded.get("kanban") or {}) if isinstance(loaded, dict) else {}
    command = cfg.get("artifact_review_verifier_command")
    # One immutable executable only.  Allowing an interpreter plus script (or
    # -c) would make argv[1] another caller-replaceable code input.
    if not isinstance(command, list) or len(command) != 1 or not all(
        isinstance(item, str) and item.strip() for item in command
    ):
        return None
    argv = [item.strip() for item in command]
    executable = Path(argv[0])
    if not executable.is_absolute() or executable.is_symlink():
        return None
    try:
        executable_resolved = executable.resolve(strict=True)
        executable_stat = executable_resolved.stat()
    except OSError:
        return None
    try:
        executable_resolved.relative_to(managed_resolved)
    except ValueError:
        return None
    trusted_ancestor = executable_resolved.parent
    while True:
        try:
            ancestor_stat = trusted_ancestor.stat()
        except OSError:
            return None
        if (
            not trusted_ancestor.is_dir()
            or ancestor_stat.st_uid != _TRUSTED_OWNER_UID
            or (ancestor_stat.st_mode & 0o022)
        ):
            return None
        if trusted_ancestor == managed_resolved:
            break
        if managed_resolved not in trusted_ancestor.parents:
            return None
        trusted_ancestor = trusted_ancestor.parent
    if (
        not executable_resolved.is_file()
        or executable_stat.st_uid != _TRUSTED_OWNER_UID
        or (executable_stat.st_mode & 0o022)
    ):
        return None
    argv[0] = str(executable_resolved)
    timeout = cfg.get("artifact_review_verifier_timeout_seconds", 30)
    try:
        timeout = min(120.0, max(1.0, float(timeout)))
    except (TypeError, ValueError):
        timeout = 30.0
    raw_policy = cfg.get("artifact_review_evidence_policy") or {}
    policy: dict[str, list[str]] = {}
    for key in ("test_scopes", "integration_scopes"):
        values = raw_policy.get(key) if isinstance(raw_policy, dict) else None
        if not isinstance(values, list) or not values or not all(
            isinstance(item, str) and item.strip() for item in values
        ):
            return None
        policy[key] = [item.strip() for item in values]
    return argv, timeout, policy


def _cas_copy(path: Path, expected_size: int) -> dict[str, Any]:
    """Copy one durable attachment to a digest path while hashing its bytes."""
    if not path.is_file() or path.is_symlink():
        raise ValueError("review artifact is missing, not regular, or is a symlink")
    accepted_dir = Path(get_hermes_home()) / "kanban" / "acceptance-cas" / "sha256"
    accepted_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    fd, tmp_name = tempfile.mkstemp(prefix=".accept-", dir=accepted_dir)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as dst, path.open("rb") as src:
            while chunk := src.read(1024 * 1024):
                digest.update(chunk)
                dst.write(chunk)
                size += len(chunk)
            dst.flush()
            os.fsync(dst.fileno())
        if size != int(expected_size):
            raise ValueError("review artifact size changed after its attachment row was recorded")
        sha = digest.hexdigest()
        final = accepted_dir / sha
        if final.exists():
            if final.is_symlink() or not final.is_file():
                raise ValueError("content-addressed artifact path is not a regular file")
            tmp.unlink()
            existing = hashlib.sha256()
            with final.open("rb") as existing_file:
                while chunk := existing_file.read(1024 * 1024):
                    existing.update(chunk)
            if final.stat().st_size != size or existing.hexdigest() != sha:
                raise ValueError("content-addressed artifact path has conflicting content")
        else:
            os.replace(tmp, final)
            final.chmod(0o444)
            dir_fd = os.open(accepted_dir, os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        return {"sha256": sha, "size": size, "cas_path": str(final.resolve())}
    finally:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _authoritative_context(conn, task_id: str, snapshot: tuple[Any, ...]) -> dict[str, Any]:
    run_id, status, contract = snapshot
    event = conn.execute(
        "SELECT payload FROM task_events WHERE task_id=? AND kind='review_requested' "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone()
    handoff = _json_dict(event["payload"] if event else None)
    owner, reviewer = handoff.get("implementer"), handoff.get("reviewer")
    if not isinstance(owner, str) or not owner.strip():
        raise ValueError("latest review handoff has no implementer provenance")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("latest review handoff has no reviewer provenance")
    if owner.strip() == reviewer.strip():
        raise ValueError("artifact review must be independent from the implementer")
    declared = handoff.get("artifacts")
    if not isinstance(declared, list) or not declared:
        raise ValueError("latest review handoff declares no durable artifacts")
    wanted = {str(item) for item in declared if isinstance(item, str) and item.strip()}
    rows = conn.execute(
        "SELECT id, filename, stored_path, size FROM task_attachments WHERE task_id=? "
        "ORDER BY id", (task_id,),
    ).fetchall()
    matched = [row for row in rows if row["stored_path"] in wanted]
    if len(matched) != len(wanted):
        raise ValueError("every reviewed artifact must resolve to a durable task attachment")
    artifacts = []
    for row in matched:
        item = _cas_copy(Path(row["stored_path"]), int(row["size"] or 0))
        item.update({"attachment_id": int(row["id"]), "filename": row["filename"]})
        artifacts.append(item)
    return {
        "task_id": task_id,
        "run_id": run_id,
        "status": status,
        "completion_contract": contract,
        "owner": owner.strip(),
        "reviewer": reviewer.strip(),
        "artifacts": artifacts,
    }


def _evidence_items(value: Any, artifact_hashes: set[str], label: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"accepted receipt requires {label} evidence")
    normalized = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError(f"{label} evidence entries must be objects")
        required = ("scope", "source_revision", "environment", "artifact_sha256")
        if any(not isinstance(item.get(key), str) or not item[key].strip() for key in required):
            raise ValueError(f"{label} evidence is missing scope/source/environment/artifact binding")
        if item.get("status") not in {"passed", "success"}:
            raise ValueError(f"{label} evidence did not pass")
        if item["artifact_sha256"] not in artifact_hashes:
            raise ValueError(f"{label} evidence is bound to a different artifact")
        normalized.append({key: item[key] for key in (*required, "status")})
    return normalized


def _normalize_verified(value: Any, authority: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("accepted") is not True:
        raise ValueError("verifier did not accept the review")
    bindings = value.get("bindings")
    if not isinstance(bindings, dict):
        raise ValueError("verifier omitted authoritative bindings")
    for key in ("task_id", "run_id", "completion_contract", "owner", "reviewer"):
        if bindings.get(key) != authority[key]:
            raise ValueError(f"verifier receipt has a stale or mismatched {key}")
    if value.get("schema") != SCHEMA or value.get("verdict") != "accepted":
        raise ValueError("verifier returned an unsupported receipt schema or verdict")
    independent = value.get("independent_review")
    if not isinstance(independent, dict) or independent.get("independent") is not True:
        raise ValueError("verifier did not attest an independent review")
    if independent.get("reviewer") != authority["reviewer"]:
        raise ValueError("independent reviewer does not match the review handoff")
    expected = {item["sha256"] for item in authority["artifacts"]}
    got = value.get("artifact_sha256")
    if not isinstance(got, list) or set(got) != expected or len(got) != len(expected):
        raise ValueError("verifier receipt is not bound to the complete artifact set")
    evidence = value.get("evidence")
    if not isinstance(evidence, dict):
        raise ValueError("verifier omitted test and integration evidence")
    tests = _evidence_items(evidence.get("tests"), expected, "test")
    integration = _evidence_items(evidence.get("integration"), expected, "integration")
    if {item["artifact_sha256"] for item in tests} != expected:
        raise ValueError("test evidence does not cover the complete artifact set")
    if {item["artifact_sha256"] for item in integration} != expected:
        raise ValueError("integration evidence does not cover the complete artifact set")
    policy = authority["evidence_policy"]
    if not set(policy["test_scopes"]).issubset({item["scope"] for item in tests}):
        raise ValueError("test evidence does not satisfy the configured scope policy")
    if not set(policy["integration_scopes"]).issubset({item["scope"] for item in integration}):
        raise ValueError("integration evidence does not satisfy the configured scope policy")
    usage = value.get("usage")
    if not isinstance(usage, dict) or not isinstance(usage.get("applicable"), bool):
        raise ValueError("verifier omitted usage applicability")
    if usage["applicable"]:
        for key in ("provider", "model", "policy", "budget"):
            if key not in usage or usage[key] in (None, "", {}):
                raise ValueError(f"applicable usage receipt omitted {key}")
        normalized_usage = {key: usage[key] for key in ("applicable", "provider", "model", "policy", "budget")}
    else:
        if not isinstance(usage.get("reason"), str) or not usage["reason"].strip():
            raise ValueError("non-applicable usage receipt requires a reason")
        normalized_usage = {"applicable": False, "reason": usage["reason"].strip()}
    receipt = {
        "ok": True,
        "classification": "success",
        "schema": SCHEMA,
        "verdict": "accepted",
        "bindings": {key: authority[key] for key in ("task_id", "run_id", "completion_contract", "owner", "reviewer")},
        "artifacts": authority["artifacts"],
        "independent_review": {"independent": True, "reviewer": authority["reviewer"]},
        "evidence": {"tests": tests, "integration": integration},
        "evidence_policy": policy,
        "usage": normalized_usage,
    }
    receipt["receipt_sha256"] = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return receipt


def collect_acceptance(conn, task_id: str, snapshot: tuple[Any, ...], metadata: Any) -> dict[str, Any]:
    """Run the configured verifier outside the terminal write transaction."""
    configured = _configured_command()
    if configured is None:
        return _failure("missing", "Configure kanban.artifact_review_verifier_command as an argv list.")
    candidate = metadata.get("artifact_review_receipt") if isinstance(metadata, dict) else None
    if not isinstance(candidate, dict):
        return _failure("missing", "Supply metadata.artifact_review_receipt for the configured verifier.")
    try:
        authority = _authoritative_context(conn, task_id, snapshot)
        command, timeout, policy = configured
        authority["evidence_policy"] = policy
        request = {"schema": SCHEMA, "authority": authority, "candidate": candidate}
        proc = subprocess.run(
            command, input=json.dumps(request), text=True, capture_output=True,
            timeout=timeout, check=False, cwd="/", env={}, close_fds=True,
            start_new_session=True,
        )
        if proc.returncode != 0:
            return _failure("rejected", "Configured artifact-review verifier rejected the candidate.")
        try:
            verified = json.loads(proc.stdout)
        except (TypeError, ValueError):
            return _failure("infra", "Configured artifact-review verifier returned invalid JSON.")
        if not isinstance(verified, dict) or verified.get("accepted") is not True:
            return _failure("rejected", "Configured artifact-review verifier did not accept the review.")
        return _normalize_verified(verified, authority)
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
        return _failure("infra", str(exc))


def collect_non_acceptance_disposition(
    task_id: str, snapshot: tuple[Any, ...], reason: Any,
) -> dict[str, Any]:
    """Ask the configured trust boundary to authorize a protected archive."""
    configured = _configured_command()
    text = str(reason or "").strip()
    if configured is None:
        return _failure("missing", "Configure a trusted absolute verifier argv before disposing protected work.")
    if not text:
        return _failure("missing", "A non-acceptance disposition reason is required.")
    command, timeout, _policy = configured
    run_id, status, contract = snapshot
    request = {
        "schema": "hermes.non-acceptance-disposition/v1",
        "action": "verify_non_acceptance_disposition",
        "authority": {
            "task_id": task_id,
            "run_id": run_id,
            "status": status,
            "completion_contract": contract,
        },
        "candidate": {"reason": text},
    }
    try:
        proc = subprocess.run(
            command, input=json.dumps(request), text=True, capture_output=True,
            timeout=timeout, check=False, cwd="/", env={}, close_fds=True,
            start_new_session=True,
        )
        if proc.returncode != 0:
            return _failure("rejected", "Configured verifier rejected the non-acceptance disposition.")
        value = json.loads(proc.stdout)
        bindings = value.get("bindings") if isinstance(value, dict) else None
        if (
            not isinstance(value, dict)
            or value.get("schema") != "hermes.non-acceptance-disposition/v1"
            or value.get("accepted") is not True
            or not isinstance(bindings, dict)
            or any(bindings.get(key) != expected for key, expected in {
                "task_id": task_id,
                "run_id": run_id,
                "completion_contract": contract,
            }.items())
            or not isinstance(value.get("actor"), str)
            or not value["actor"].strip()
        ):
            return _failure("rejected", "Verifier returned an unbound non-acceptance disposition.")
        receipt = {
            "ok": True,
            "classification": "success",
            "schema": "hermes.non-acceptance-disposition/v1",
            "bindings": bindings,
            "reason": text,
            "actor": value["actor"].strip(),
        }
        receipt["receipt_sha256"] = hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return receipt
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
        return _failure("infra", str(exc))


def revalidate_receipt_artifacts(receipt: Any) -> bool:
    """Final short CAS check under the terminal transaction."""
    if not isinstance(receipt, dict):
        return False
    artifacts = receipt.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        return False
    for item in artifacts:
        if not isinstance(item, dict):
            return False
        path = Path(str(item.get("cas_path") or ""))
        expected = item.get("sha256")
        if not isinstance(expected, str) or path.is_symlink() or not path.is_file():
            return False
        digest = hashlib.sha256()
        try:
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
        except OSError:
            return False
        if digest.hexdigest() != expected:
            return False
    return True
