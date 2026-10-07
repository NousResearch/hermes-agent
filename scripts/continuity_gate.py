#!/usr/bin/env python3
"""Create, verify, and promote exact-state continuity receipts."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import stat
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows release is not supported by this pilot
    fcntl = None  # type: ignore[assignment]

from continuity_bridge import ALLOWED_ACTIVE_STATES, build_preflight
from continuity_common import (
    ContinuityError,
    confined_atomic_write_json,
    confined_atomic_write_text,
    confined_ensure_dir,
    confined_load_json,
    confined_parent,
    confined_read_text,
    external_volume_available,
    external_input_digests,
    git_state,
    load_json,
    minimal_child_env,
    parse_markdown_frontmatter,
    read_markdown_frontmatter,
    receipt_errors,
    receipt_signing_key,
    receipt_signature_errors,
    render_markdown_frontmatter,
    run_command,
    sha256_file,
    sign_receipt,
    validate_state_storage,
    verify_pinned_executable,
)


COMMAND_SPEC_KEYS = {"name", "argv", "timeout_seconds"}
RUNTIME_SPEC_KEYS = {
    "name",
    "argv",
    "surface",
    "event",
    "adapter_path",
    "bound_paths",
    "native",
    "timeout_seconds",
}
ROLLBACK_SPEC_KEYS = {"name", "argv", "mode", "timeout_seconds"}
TEST_PATTERNS = (
    re.compile(r"Summary:\s+\d+ files?,\s+(\d+) tests? passed,\s+(\d+) failed"),
    re.compile(r"(?<![\w])(?P<passed>\d+) passed(?:,\s+(?P<skipped>\d+) skipped)?"),
)
SENSITIVE_ARG = re.compile(
    r"(?i)(authorization|api[_-]?key|cookie|credential|password|passwd|secret|token)"
)
CANONICAL_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / ".continuity/config.json"
)


def _native_adapter_path(
    config: dict[str, Any], repo_root: Path, adapter_token: str, *, require_exists: bool
) -> Path:
    candidate = Path(adapter_token)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        allowed_root = Path(config["state_root"]).resolve()
    else:
        resolved = (repo_root / candidate).resolve()
        allowed_root = repo_root.resolve()
    if not resolved.is_relative_to(allowed_root):
        raise ContinuityError("native observation adapter escapes its allowed root")
    if require_exists and not resolved.is_file():
        raise ContinuityError("native observation adapter is missing")
    return resolved


def _native_bound_artifacts(
    config: dict[str, Any],
    repo_root: Path,
    spec: dict[str, Any],
    *,
    require_exists: bool,
) -> dict[str, Path]:
    adapter_token = spec.get("adapter_path")
    tokens = spec.get("bound_paths", [adapter_token])
    if (
        not isinstance(adapter_token, str)
        or not adapter_token
        or not isinstance(tokens, list)
        or not tokens
        or not all(isinstance(token, str) and token for token in tokens)
        or len(set(tokens)) != len(tokens)
        or adapter_token not in tokens
    ):
        raise ContinuityError("native observation bound paths are invalid")
    return {
        token: _native_adapter_path(
            config,
            repo_root,
            token,
            require_exists=require_exists and not Path(token).is_absolute(),
        )
        for token in tokens
    }


def _native_host_identity(
    config: dict[str, Any], surface: str
) -> dict[str, str] | None:
    policy = config.get("native_observation_policy")
    hosts = policy.get("hosts") if isinstance(policy, dict) else None
    configured = hosts.get(surface) if isinstance(hosts, dict) else None
    if configured is None:
        return None
    if not isinstance(configured, dict) or set(configured) != {"path", "sha256"}:
        raise ContinuityError(f"{surface} native host identity is malformed")
    path = Path(str(configured["path"])).resolve()
    actual = sha256_file(path)
    if not hmac.compare_digest(actual, str(configured["sha256"])):
        raise ContinuityError(f"{surface} native host identity is stale")
    return {"path": str(path), "sha256": actual}


def _task_from_value(value: Any, task_id: str) -> dict[str, Any] | None:
    candidates = value if isinstance(value, list) else [value]
    return next(
        (
            item
            for item in candidates
            if isinstance(item, dict) and item.get("id") == task_id
        ),
        None,
    )


def _require_committed_config(
    config_path: Path, config: dict[str, Any], *, cwd: Path
) -> Path:
    expected = CANONICAL_CONFIG_PATH.resolve()
    if config_path.resolve() != expected:
        raise ContinuityError(
            f"continuity authority requires committed config: {expected}"
        )
    repo_root = Path(git_state(cwd.resolve()).root)
    if expected.parent.parent != repo_root:
        raise ContinuityError(
            "continuity gate is not running from its canonical repository"
        )
    committed = run_command(
        ["git", "show", "HEAD:.continuity/config.json"],
        cwd=repo_root,
        timeout=30,
        env=minimal_child_env(
            state_root=config["state_root"],
            external_volume=config["external_volume"],
        ),
        required_mount=(config["external_volume"], config["state_root"]),
    )
    try:
        committed_config = (
            json.loads(committed.stdout) if committed.returncode == 0 else None
        )
    except json.JSONDecodeError as exc:
        raise ContinuityError("committed continuity config is invalid") from exc
    if config != committed_config:
        raise ContinuityError("continuity config does not match committed HEAD")
    return repo_root


def strict_authority_check(
    config_path: Path,
    *,
    cwd: Path,
    require_mounted_volume: bool,
    allow_closed_task: bool = False,
    allow_recovery_journal: bool = False,
) -> dict[str, Any]:
    """Re-check all authorities and require a live Beads CLI response."""
    config = load_json(config_path)
    preflight = build_preflight(
        config_path,
        cwd=cwd,
        require_mounted_volume=require_mounted_volume,
        allow_recovery_journal=allow_recovery_journal,
    )
    errors = list(preflight.get("errors") or [])
    beads = config["beads"]
    verify_pinned_executable(config, cwd.resolve(), "beads", Path(beads["binary"]))
    env = minimal_child_env(
        {"BEADS_DIR": beads["data_dir"]},
        state_root=config["state_root"],
        external_volume=config["external_volume"],
    )
    try:
        result = run_command(
            [beads["binary"], "show", beads["active_task"], "--json"],
            cwd=cwd,
            timeout=float(beads.get("completion_timeout_seconds", 60)),
            env=env,
            required_mount=(config["external_volume"], config["state_root"]),
        )
        if result.returncode != 0:
            raise ContinuityError(f"bd show exited with status {result.returncode}")
        task = _task_from_value(json.loads(result.stdout), beads["active_task"])
        if task is None:
            errors.append("strict Beads query did not return the active task")
        elif task.get("status") not in ALLOWED_ACTIVE_STATES and not (
            allow_closed_task and task.get("status") == "closed"
        ):
            errors.append(
                f"strict Beads query returned terminal status {task.get('status')!r}"
            )
    except (ContinuityError, json.JSONDecodeError) as exc:
        errors.append(f"strict Beads query failed: {exc}")
        task = None
    return {
        "passed": not errors,
        "preflight_status": preflight.get("status"),
        "task_id": task.get("id") if task else None,
        "task_status": task.get("status") if task else None,
        "errors": errors,
    }


def _load_list(path: Path | None, label: str) -> list[dict[str, Any]]:
    if path is None:
        return []
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContinuityError(f"cannot read {label}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ContinuityError(f"{label} must be a JSON list of objects")
    return value


def _load_object(path: Path | None, label: str) -> dict[str, Any]:
    if path is None:
        return {}
    return load_json(path)


def _validate_closed_spec(value: dict[str, Any], allowed: set[str], label: str) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ContinuityError(f"{label} contains unknown keys: {', '.join(unknown)}")
    name = value.get("name")
    argv = value.get("argv")
    if not isinstance(name, str) or not name or len(name) > 120:
        raise ContinuityError(f"{label} has an invalid name")
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(item, str) and item for item in argv)
    ):
        raise ContinuityError(f"{label} argv must be a non-empty string list")


def _policy_evidence_specs(policy: dict[str, Any]) -> list[dict[str, Any]]:
    runtime = policy.get("runtime_checks")
    values = [
        policy.get("focused_suite"),
        policy.get("full_suite"),
        *(runtime if isinstance(runtime, list) else []),
        policy.get("rollback_check"),
    ]
    if not all(isinstance(value, dict) for value in values):
        raise ContinuityError("evidence policy contains an invalid command spec")
    return values  # type: ignore[return-value]


def _evidence_pin_map(config: dict[str, Any]) -> dict[str, str]:
    pins: dict[str, str] = {}
    for item in config.get("evidence_executables", []):
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ContinuityError("evidence executable pin is malformed")
        token = item.get("path")
        digest = item.get("sha256")
        if (
            not isinstance(token, str)
            or not token
            or token in pins
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ContinuityError("evidence executable pin is malformed or duplicated")
        pins[token] = digest
    return pins


def _verify_evidence_entrypoint(
    config: dict[str, Any], repo_root: Path, spec: dict[str, Any]
) -> str:
    token = str(spec["argv"][0])
    pins = _evidence_pin_map(config)
    expected = pins.get(token)
    if expected is None:
        raise ContinuityError(f"evidence executable is not pinned: {token}")
    path = Path(token)
    if not path.is_absolute():
        path = repo_root / path
    actual = sha256_file(path.resolve())
    if not hmac.compare_digest(actual, expected):
        raise ContinuityError(
            f"evidence executable digest mismatch: expected {expected}, got {actual}"
        )
    return actual


def _resolved_argv(spec: dict[str, Any], repo_root: Path) -> list[str]:
    argv = list(spec["argv"])
    if any(SENSITIVE_ARG.search(value) for value in argv):
        raise ContinuityError("evidence argv contains a secret-bearing argument name")
    first = Path(argv[0])
    if not first.is_absolute():
        invocation = (repo_root / first).absolute()
    else:
        invocation = first.absolute()
    resolved = invocation.resolve()
    if resolved == Path(sys.executable).resolve():
        if len(argv) < 2 or argv[1].startswith("-"):
            raise ContinuityError(
                "Python evidence commands require a repository script"
            )
        script = (repo_root / argv[1]).resolve()
        if not script.is_relative_to(repo_root) or script.suffix != ".py":
            raise ContinuityError(
                "Python evidence script must be a repository .py file"
            )
        argv[0] = str(resolved)
        argv[1] = str(script)
        return argv
    if resolved == Path("/bin/bash").resolve():
        if len(argv) < 2 or argv[1].startswith("-"):
            raise ContinuityError("Bash evidence commands require a repository script")
        script = (repo_root / argv[1]).resolve()
        if not script.is_relative_to(repo_root) or script.suffix != ".sh":
            raise ContinuityError("Bash evidence script must be a repository .sh file")
        argv[0] = str(invocation)
        argv[1] = str(script)
        return argv
    if not resolved.is_relative_to(repo_root):
        raise ContinuityError(f"evidence executable escapes repository: {resolved}")
    if not resolved.is_file():
        raise ContinuityError(f"evidence executable is missing: {resolved}")
    argv[0] = str(invocation)
    return argv


def _python_venv_context(spec: dict[str, Any], repo_root: Path) -> dict[str, str]:
    """Preserve venv imports without executing through a mutable launcher link."""
    token = Path(str(spec["argv"][0]))
    invocation = (
        token.absolute() if token.is_absolute() else (repo_root / token).absolute()
    )
    resolved = invocation.resolve()
    if resolved != Path(sys.executable).resolve() or invocation == resolved:
        return {}
    venv_root = invocation.parent.parent
    candidates = [
        venv_root / "Lib/site-packages",
        venv_root
        / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages",
    ]
    site_packages = [str(path) for path in candidates if path.is_dir()]
    context = {"__PYVENV_LAUNCHER__": str(invocation)}
    if site_packages:
        context["PYTHONPATH"] = os.pathsep.join(site_packages)
    return context


def _dependency_identity(config: dict[str, Any], repo_root: Path) -> dict[str, Any]:
    """Fingerprint the lock policy and resolved Python distribution environment."""
    policy = config.get("dependency_identity")
    if not isinstance(policy, dict) or set(policy) != {
        "python",
        "requirements_lock",
    }:
        raise ContinuityError("dependency identity policy is absent or malformed")
    python_token = policy.get("python")
    requirements_token = policy.get("requirements_lock")
    if not isinstance(python_token, str) or not python_token:
        raise ContinuityError("dependency identity Python is invalid")
    if not isinstance(requirements_token, str) or not requirements_token:
        raise ContinuityError("dependency identity requirements lock is invalid")

    python_path = Path(python_token)
    if not python_path.is_absolute():
        python_path = (repo_root / python_path).absolute()
    requirements_path = (repo_root / requirements_token).resolve()
    if (
        not requirements_path.is_relative_to(repo_root)
        or not requirements_path.is_file()
    ):
        raise ContinuityError("dependency identity requirements lock is missing")
    if not python_path.exists():
        raise ContinuityError("dependency identity Python is missing")
    resolved_python = python_path.resolve()
    launcher_material = str(python_path)
    if python_path.is_symlink():
        launcher_material += "\0" + os.readlink(python_path)
    launcher_material += "\0" + sha256_file(resolved_python)
    try:
        requirements_text = requirements_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ContinuityError(
            "dependency identity requirements lock is unreadable"
        ) from exc
    logical_requirements = requirements_text.replace("\\\n", " ")
    locked: dict[str, str] = {}
    requirement_pattern = re.compile(
        r"^([A-Za-z0-9_.-]+)==([^\s;]+).*--hash=sha256:[0-9a-fA-F]{64}(?:\s|$)"
    )
    for raw_line in logical_requirements.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = requirement_pattern.match(line)
        if match is None:
            raise ContinuityError(
                "dependency identity requirements must be exact and SHA-256 hashed"
            )
        name = match.group(1).lower().replace("_", "-")
        if name in locked:
            raise ContinuityError("dependency identity requirements are duplicated")
        locked[name] = match.group(2)
    if not locked:
        raise ContinuityError("dependency identity requirements lock is empty")

    probe = r"""
import hashlib
import importlib.metadata
import json
import sys

records = []
for distribution in importlib.metadata.distributions():
    name = distribution.metadata.get("Name") or ""
    record = {
        "name": name.lower().replace("_", "-") if name else "unknown",
        "version": distribution.version,
    }
    for metadata_name in ("METADATA", "RECORD", "direct_url.json"):
        value = distribution.read_text(metadata_name)
        record[metadata_name] = hashlib.sha256(
            (value or "").encode("utf-8")
        ).hexdigest()
    records.append(record)
records.sort(key=lambda item: (item["name"], item["version"], item["METADATA"]))
print(json.dumps({"python_version": sys.version, "packages": records}, separators=(",", ":")))
"""
    result = run_command(
        [str(python_path), "-I", "-c", probe],
        cwd=repo_root,
        timeout=60,
        env=minimal_child_env(
            state_root=config["state_root"],
            external_volume=config["external_volume"],
        ),
        required_mount=(config["external_volume"], config["state_root"]),
    )
    if result.returncode != 0:
        raise ContinuityError("dependency identity probe failed")
    try:
        resolved = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ContinuityError(
            "dependency identity probe returned invalid JSON"
        ) from exc
    packages = resolved.get("packages") if isinstance(resolved, dict) else None
    version = resolved.get("python_version") if isinstance(resolved, dict) else None
    if not isinstance(packages, list) or not packages or not isinstance(version, str):
        raise ContinuityError("dependency identity probe returned invalid data")
    installed = {
        item.get("name"): item.get("version")
        for item in packages
        if isinstance(item, dict)
    }
    mismatched = sorted(
        name for name, wanted in locked.items() if installed.get(name) != wanted
    )
    if mismatched:
        raise ContinuityError(
            "resolved dependency environment does not match requirements lock: "
            + ", ".join(mismatched)
        )
    packages_json = json.dumps(packages, sort_keys=True, separators=(",", ":"))
    return {
        "schema_version": 1,
        "requirements_sha256": sha256_file(requirements_path),
        "python_launcher_sha256": hashlib.sha256(
            launcher_material.encode("utf-8")
        ).hexdigest(),
        "python_executable_sha256": sha256_file(resolved_python),
        "python_version": version,
        "packages_sha256": hashlib.sha256(packages_json.encode("utf-8")).hexdigest(),
        "package_count": len(packages),
    }


def _native_observation_errors(
    config: dict[str, Any], receipt: dict[str, Any], current: Any
) -> list[str]:
    if receipt.get("lifecycle_target") != "ENFORCED":
        return []
    policy = config.get("native_observation_policy")
    if not isinstance(policy, dict) or set(policy) != {
        "required_surfaces",
        "max_age_seconds",
        "hosts",
    }:
        return ["native observation policy is absent or malformed"]
    required = policy.get("required_surfaces")
    max_age = policy.get("max_age_seconds")
    if (
        not isinstance(required, list)
        or not required
        or not all(isinstance(item, str) and item for item in required)
        or len(set(required)) != len(required)
        or not isinstance(max_age, int)
        or max_age < 1
        or max_age > 3600
    ):
        return ["native observation policy is invalid"]
    errors: list[str] = []
    observations: dict[str, dict[str, Any]] = {}
    for check in receipt.get("runtime_checks") or []:
        observation = (
            check.get("native_observation") if isinstance(check, dict) else None
        )
        if isinstance(observation, dict) and isinstance(
            observation.get("surface"), str
        ):
            surface = observation["surface"]
            if surface in observations:
                errors.append(
                    f"native observation is duplicated for surface: {surface}"
                )
            observations[surface] = observation
    repo_root = Path(current.root).resolve()
    runtime_specs = {
        item.get("surface"): item
        for item in (config.get("evidence_policy") or {}).get("runtime_checks", [])
        if isinstance(item, dict) and isinstance(item.get("surface"), str)
    }

    def last_authenticated_success(surface: str) -> str:
        receipt_dir = Path(str(config.get("receipt_dir", "")))
        if not receipt_dir.is_dir():
            return "none"
        latest: tuple[datetime, str] | None = None
        for path in receipt_dir.glob("*.json"):
            try:
                candidate = load_json(path)
                if (
                    candidate.get("result") != "PASS"
                    or candidate.get("project") != config.get("project")
                    or candidate.get("repo_id") != config.get("repo_id")
                    or receipt_signature_errors(config, candidate)
                ):
                    continue
                for check in candidate.get("runtime_checks") or []:
                    prior = (
                        check.get("native_observation")
                        if isinstance(check, dict)
                        else None
                    )
                    if not isinstance(prior, dict) or prior.get("surface") != surface:
                        continue
                    token = str(prior.get("observed_at"))
                    observed = datetime.fromisoformat(token)
                    if observed.tzinfo is None:
                        continue
                    if latest is None or observed > latest[0]:
                        latest = (observed, token)
            except (ContinuityError, OSError, TypeError, ValueError):
                continue
        return latest[1] if latest else "none"

    def expectation(surface: str, observation: dict[str, Any] | None) -> str:
        spec = runtime_specs.get(surface) or {}
        event = spec.get("event") or "unknown"
        digest = "missing"
        adapter_token = spec.get("adapter_path")
        if isinstance(adapter_token, str):
            try:
                adapter = _native_adapter_path(
                    config, repo_root, adapter_token, require_exists=True
                )
                digest = sha256_file(adapter)
            except ContinuityError:
                pass
        last_success = (
            observation.get("observed_at")
            if isinstance(observation, dict) and observation.get("observed_at")
            else last_authenticated_success(surface)
        )
        return (
            f"surface={surface}; expected_event={event}; "
            f"expected_adapter_sha256={digest}; last_success={last_success}"
        )

    for surface in required:
        observation = observations.get(surface)
        if observation is None:
            errors.append(
                "native observation is missing: " + expectation(surface, observation)
            )
            continue
        if observation.get("commit") != current.commit:
            errors.append(
                "native observation commit is stale: "
                + expectation(surface, observation)
            )
        spec = runtime_specs.get(surface) or {}
        if observation.get("event") != spec.get("event"):
            errors.append(
                "native observation event does not match policy: "
                + expectation(surface, observation)
            )
        adapter_path = observation.get("adapter_path")
        if not isinstance(adapter_path, str) or adapter_path != spec.get(
            "adapter_path"
        ):
            errors.append(
                "native observation adapter does not match policy: "
                + expectation(surface, observation)
            )
        else:
            try:
                resolved_adapter = _native_adapter_path(
                    config, repo_root, adapter_path, require_exists=True
                )
                digest_matches = observation.get("adapter_sha256") == sha256_file(
                    resolved_adapter
                )
            except ContinuityError:
                digest_matches = False
            if not digest_matches:
                errors.append(
                    "native observation adapter digest is stale: "
                    + expectation(surface, observation)
                )
        try:
            current_artifacts = {
                token: sha256_file(path)
                for token, path in _native_bound_artifacts(
                    config, repo_root, spec, require_exists=True
                ).items()
            }
        except ContinuityError:
            current_artifacts = {}
        if observation.get("bound_artifacts") != current_artifacts:
            errors.append(
                "native observation bound artifacts are stale: "
                + expectation(surface, observation)
            )
        try:
            current_host = _native_host_identity(config, surface)
        except ContinuityError:
            current_host = {"stale": "true"}
        if (
            current_host is not None
            and observation.get("host_identity") != current_host
        ):
            errors.append(
                "native observation host identity is stale: "
                + expectation(surface, observation)
            )
        try:
            observed_at = datetime.fromisoformat(str(observation.get("observed_at")))
            if observed_at.tzinfo is None:
                raise ValueError("timestamp is not timezone-aware")
            age = (datetime.now(timezone.utc) - observed_at).total_seconds()
            if age < -60 or age > max_age:
                errors.append(
                    "native observation is stale: " + expectation(surface, observation)
                )
        except (TypeError, ValueError):
            errors.append(
                "native observation timestamp is invalid: "
                + expectation(surface, observation)
            )
    return errors


def _test_summary(output: str) -> tuple[int, int, int]:
    for pattern in TEST_PATTERNS:
        match = pattern.search(output)
        if not match:
            continue
        groups = match.groupdict()
        if groups:
            return int(groups["passed"]), 0, int(groups.get("skipped") or 0)
        return int(match.group(1)), int(match.group(2)), 0
    return 0, 0, 0


def _execute_evidence(
    spec: dict[str, Any],
    repo_root: Path,
    *,
    kind: str,
    evidence_home: Path,
    config: dict[str, Any],
    index: int,
) -> dict[str, Any]:
    allowed = {
        "command": COMMAND_SPEC_KEYS,
        "runtime": RUNTIME_SPEC_KEYS,
        "rollback": ROLLBACK_SPEC_KEYS,
    }[kind]
    _validate_closed_spec(spec, allowed, f"{kind} evidence")
    entrypoint_digest = _verify_evidence_entrypoint(config, repo_root, spec)
    argv = _resolved_argv(spec, repo_root)
    policy = config.get("evidence_policy")
    if not isinstance(policy, dict):
        raise ContinuityError("continuity config has no evidence_policy")
    if kind == "runtime":
        configured_runtime = policy.get("runtime_checks")
        allowed_surfaces = {
            item.get("surface")
            for item in configured_runtime or []
            if isinstance(item, dict)
        }
        if spec.get("surface") not in allowed_surfaces:
            raise ContinuityError("runtime evidence surface is not allowlisted")
        native = spec.get("native")
        if native is not True:
            raise ContinuityError("runtime evidence must be a native observation")
        if not isinstance(spec.get("event"), str) or not spec["event"]:
            raise ContinuityError("native runtime evidence event is invalid")
        adapter_token = spec.get("adapter_path")
        if not isinstance(adapter_token, str) or not adapter_token:
            raise ContinuityError("native runtime evidence adapter is invalid")
        bound_paths = _native_bound_artifacts(
            config, repo_root, spec, require_exists=True
        )
        adapter_path = bound_paths[adapter_token]
        adapter_digest = sha256_file(adapter_path)
        bound_artifacts = {
            token: sha256_file(path) for token, path in bound_paths.items()
        }
        host_identity = _native_host_identity(config, str(spec.get("surface")))
    if kind == "rollback" and spec.get("mode") != "dry-run":
        raise ContinuityError("receipt rollback evidence must use dry-run mode")
    timeout = float(spec.get("timeout_seconds", 900))
    if timeout <= 0 or timeout > 7200:
        raise ContinuityError(f"{kind} evidence timeout is outside 1..7200 seconds")
    started = datetime.now(timezone.utc)
    monotonic_start = time.monotonic()
    child_env = {
        "HOME": str(evidence_home),
        "PYTHONHASHSEED": "0",
        "TZ": "UTC",
    }
    child_env.update(_python_venv_context(spec, repo_root))
    result = run_command(
        argv,
        cwd=repo_root,
        timeout=timeout,
        env=minimal_child_env(
            child_env,
            state_root=config["state_root"],
            external_volume=config["external_volume"],
        ),
        required_mount=(config["external_volume"], config["state_root"]),
        require_native_containment=True,
    )
    if _verify_evidence_entrypoint(config, repo_root, spec) != entrypoint_digest:
        raise ContinuityError("evidence executable changed during execution")
    if kind == "runtime":
        if {
            token: sha256_file(path) for token, path in bound_paths.items()
        } != bound_artifacts:
            raise ContinuityError(
                "native observation bound artifact changed during execution"
            )
        if _native_host_identity(config, str(spec.get("surface"))) != host_identity:
            raise ContinuityError("native observation host changed during execution")
    completed = datetime.now(timezone.utc)
    output = (result.stdout or "") + (result.stderr or "")
    test_count, failed_count, skipped = _test_summary(output)
    record: dict[str, Any] = {
        "name": f"{kind}-{index + 1}",
        "kind": kind,
        "command_sha256": hashlib.sha256(
            json.dumps(argv, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "cwd": str(repo_root),
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
        "duration_ms": round((time.monotonic() - monotonic_start) * 1000),
        "exit_code": result.returncode,
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
    }
    if kind == "command":
        full_spec = policy.get("full_suite")
        if not isinstance(full_spec, dict):
            raise ContinuityError("configured full suite is missing")
        _validate_closed_spec(full_spec, COMMAND_SPEC_KEYS, "configured full suite")
        full_argv = _resolved_argv(full_spec, repo_root)
        record.update({
            "scope": "full" if argv == full_argv else "focused",
            "test_count": test_count,
            "failed_count": failed_count,
            "skipped": skipped,
            "flaky": False,
        })
    elif kind == "runtime":
        record.update({
            "surface": spec.get("surface"),
            "passed": result.returncode == 0,
            "native_observation": {
                "surface": spec.get("surface"),
                "event": spec.get("event"),
                "commit": git_state(
                    repo_root, integration_ref=config.get("integration_ref")
                ).commit,
                "adapter_path": spec.get("adapter_path"),
                "adapter_sha256": adapter_digest,
                "bound_artifacts": bound_artifacts,
                "host_identity": host_identity,
                "observed_at": completed.isoformat(),
            },
        })
    else:
        record.update({
            "mode": spec.get("mode", "dry-run"),
            "passed": result.returncode == 0,
        })
    return record


def create_receipt(
    config_path: Path,
    *,
    cwd: Path,
    risk_tier: str,
    lifecycle_target: str,
    commands: list[dict[str, Any]],
    runtime_checks: list[dict[str, Any]],
    rollback: dict[str, Any],
    require_mounted_volume: bool = True,
) -> dict[str, Any]:
    config = load_json(config_path)
    validate_state_storage(config["external_volume"], config["state_root"])
    repo_root = _require_committed_config(config_path, config, cwd=cwd)
    configured_tier = config.get("risk_tier")
    if risk_tier != configured_tier:
        raise ContinuityError(
            f"risk tier is committed as {configured_tier}; caller requested {risk_tier}"
        )
    state = git_state(cwd.resolve(), integration_ref=config.get("integration_ref"))
    expected_root = str(Path(config["expected_repo_root"]).resolve())
    if state.root != expected_root:
        raise ContinuityError(
            f"wrong repository folder: expected {expected_root}, got {state.root}"
        )
    if state.dirty:
        raise ContinuityError("receipt creation requires a clean repository")
    if state.integration_ref and state.merge_base != state.integration_sha:
        raise ContinuityError(
            "receipt creation requires the branch to include the integration SHA"
        )
    policy = config.get("evidence_policy")
    policy_entrypoints: set[str] | None = None
    if not isinstance(policy, dict):
        raise ContinuityError("continuity config has no evidence_policy")
    suite_key = "full_suite" if lifecycle_target == "ENFORCED" else "focused_suite"
    expected_command = policy.get(suite_key)
    expected_runtime = policy.get("runtime_checks")
    expected_rollback = policy.get("rollback_check")
    if commands != [expected_command]:
        raise ContinuityError("command evidence does not match committed policy")
    if runtime_checks != expected_runtime:
        raise ContinuityError("runtime evidence does not match committed policy")
    if rollback != expected_rollback:
        raise ContinuityError("rollback evidence does not match committed policy")
    policy_entrypoints = {
        str(spec["argv"][0]) for spec in _policy_evidence_specs(policy)
    }
    if set(_evidence_pin_map(config)) != policy_entrypoints:
        raise ContinuityError(
            "evidence executable pins do not exactly match policy entrypoints"
        )
    if require_mounted_volume and not external_volume_available(
        Path(config["external_volume"])
    ):
        raise ContinuityError("external volume is not mounted")
    authority_check = strict_authority_check(
        config_path,
        cwd=cwd,
        require_mounted_volume=require_mounted_volume,
    )
    if not authority_check.get("passed"):
        details = "; ".join(str(item) for item in authority_check.get("errors", []))
        suffix = f": {details}" if details else ""
        raise ContinuityError(f"strict authority check failed{suffix}")
    dependency_identity = _dependency_identity(config, repo_root)
    signing_key = receipt_signing_key(config)
    evidence_home = confined_ensure_dir(config, "evidence-home")
    receipt = {
        "schema_version": 1,
        "gate": "hermes-continuity-gate/2",
        "project": config["project"],
        "repo_id": config["repo_id"],
        "lifecycle_target": lifecycle_target,
        "risk_tier": configured_tier,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": state.as_dict(),
        "authority_check": authority_check,
        "external_inputs": external_input_digests(config, repo_root),
        "dependency_identity": dependency_identity,
        "commands": [
            _execute_evidence(
                spec,
                repo_root,
                kind="command",
                evidence_home=evidence_home,
                config=config,
                index=index,
            )
            for index, spec in enumerate(commands)
        ],
        "runtime_checks": [
            _execute_evidence(
                spec,
                repo_root,
                kind="runtime",
                evidence_home=evidence_home,
                config=config,
                index=index,
            )
            for index, spec in enumerate(runtime_checks)
        ],
        "rollback": _execute_evidence(
            rollback,
            repo_root,
            kind="rollback",
            evidence_home=evidence_home,
            config=config,
            index=0,
        )
        if rollback
        else {},
    }
    if receipt["external_inputs"] != external_input_digests(config, repo_root):
        raise ContinuityError("external inputs changed during evidence execution")
    if receipt["dependency_identity"] != _dependency_identity(config, repo_root):
        raise ContinuityError("resolved dependency identity changed during evidence")
    final_state = git_state(
        cwd.resolve(), integration_ref=config.get("integration_ref")
    )
    receipt["auth"] = {
        "algorithm": "hmac-sha256",
        "key_id": "pending",
        "digest": "0" * 64,
    }
    validation_errors = receipt_errors(receipt, final_state)
    validation_errors.extend(_native_observation_errors(config, receipt, final_state))
    receipt["result"] = "PASS" if not validation_errors else "FAIL"
    receipt["auth"] = sign_receipt(config, receipt, key=signing_key)
    return receipt


def verify_receipt(
    config_path: Path,
    receipt_path: Path,
    *,
    cwd: Path,
    allow_recovery_journal: bool = False,
) -> list[str]:
    config = load_json(config_path)
    validate_state_storage(config["external_volume"], config["state_root"])
    receipt_path = _confined_receipt_output(config, receipt_path)
    repo_root = _require_committed_config(config_path, config, cwd=cwd)
    current = git_state(cwd.resolve(), integration_ref=config.get("integration_ref"))
    errors: list[str] = []
    if current.root != str(Path(config["expected_repo_root"]).resolve()):
        errors.append("current repository root does not match continuity config")
    receipt = confined_load_json(config, receipt_path)
    errors.extend(receipt_signature_errors(config, receipt))
    if receipt.get("project") != config.get("project"):
        errors.append("receipt project does not match continuity config")
    if receipt.get("repo_id") != config.get("repo_id"):
        errors.append("receipt repo_id does not match continuity config")
    try:
        if receipt.get("external_inputs") != external_input_digests(config, repo_root):
            errors.append("external instruction or executable identity is stale")
    except ContinuityError as exc:
        errors.append(str(exc))
    try:
        if receipt.get("dependency_identity") != _dependency_identity(
            config, repo_root
        ):
            errors.append("resolved dependency identity is stale")
    except ContinuityError as exc:
        errors.append(str(exc))
    authority = strict_authority_check(
        config_path,
        cwd=cwd,
        require_mounted_volume=True,
        allow_closed_task=receipt.get("lifecycle_target") == "ENFORCED",
        allow_recovery_journal=allow_recovery_journal,
    )
    if not authority.get("passed"):
        errors.append("current strict authority check failed")
        errors.extend(authority.get("errors") or [])
    errors.extend(receipt_errors(receipt, current))
    errors.extend(_native_observation_errors(config, receipt, current))
    if receipt.get("result") != "PASS":
        errors.append("receipt result is not PASS")
    return errors


def _run_beads(config: dict[str, Any], arguments: list[str], repo_root: Path) -> None:
    beads = config["beads"]
    verify_pinned_executable(config, repo_root, "beads", Path(beads["binary"]))
    env = minimal_child_env(
        {"BEADS_DIR": beads["data_dir"]},
        state_root=config["state_root"],
        external_volume=config["external_volume"],
    )
    result = run_command(
        [beads["binary"], *arguments, "--json"],
        cwd=repo_root,
        timeout=float(beads.get("completion_timeout_seconds", 60)),
        env=env,
        required_mount=(config["external_volume"], config["state_root"]),
    )
    if result.returncode != 0:
        raise ContinuityError(f"Beads update failed with status {result.returncode}")


def _beads_status(config: dict[str, Any], repo_root: Path) -> str | None:
    beads = config["beads"]
    verify_pinned_executable(config, repo_root, "beads", Path(beads["binary"]))
    result = run_command(
        [beads["binary"], "show", beads["active_task"], "--json"],
        cwd=repo_root,
        timeout=float(beads.get("completion_timeout_seconds", 60)),
        env=minimal_child_env(
            {"BEADS_DIR": beads["data_dir"]},
            state_root=config["state_root"],
            external_volume=config["external_volume"],
        ),
        required_mount=(config["external_volume"], config["state_root"]),
    )
    if result.returncode != 0:
        raise ContinuityError("Beads verification query failed")
    try:
        task = _task_from_value(json.loads(result.stdout), beads["active_task"])
    except json.JSONDecodeError as exc:
        raise ContinuityError("Beads verification returned invalid JSON") from exc
    if task is None:
        raise ContinuityError("Beads verification did not return the active task")
    return task.get("status")


@contextmanager
def _promotion_lock(config: dict[str, Any], timeout: float = 15):
    if fcntl is None:
        raise ContinuityError("promotion locking is unavailable on this platform")
    path = Path(config["state_root"]) / "promotion.lock"
    with confined_parent(config, path) as (parent_fd, target):
        try:
            descriptor = (
                os.open(
                    target,
                    os.O_RDWR
                    | os.O_CREAT
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0),
                    0o600,
                )
                if parent_fd is None
                else os.open(
                    target,
                    os.O_RDWR
                    | os.O_CREAT
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0),
                    0o600,
                    dir_fd=parent_fd,
                )
            )
        except OSError as exc:
            raise ContinuityError(f"cannot open promotion lock safely: {exc}") from exc
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            os.close(descriptor)
            raise ContinuityError("promotion lock is not a regular file")
        deadline = time.monotonic() + timeout
        try:
            while True:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise ContinuityError(
                            "timed out waiting for the promotion lock"
                        )
                    time.sleep(0.05)
            yield
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def promote(
    config_path: Path,
    receipt_path: Path,
    *,
    cwd: Path,
    target: str,
) -> None:
    if target not in {"TESTED", "ENFORCED"}:
        raise ContinuityError("only TESTED or ENFORCED may be promoted by the gate")
    if fcntl is None:
        raise ContinuityError("promotion locking is unavailable on this platform")
    config = load_json(config_path)
    validate_state_storage(config["external_volume"], config["state_root"])
    receipt_path = _confined_receipt_output(config, receipt_path)
    _require_committed_config(config_path, config, cwd=cwd)
    if not external_volume_available(Path(config["external_volume"])):
        raise ContinuityError("external volume is not mounted")
    admission = strict_authority_check(
        config_path,
        cwd=cwd,
        require_mounted_volume=True,
        allow_closed_task=target == "ENFORCED",
        allow_recovery_journal=True,
    )
    if not admission.get("passed"):
        raise ContinuityError("promotion strict authority check failed")
    journal_path = Path(config["state_root"]) / "promotion.json"
    with _promotion_lock(config):
        recovery = False
        try:
            existing_journal = confined_load_json(config, journal_path)
        except FileNotFoundError:
            existing_journal = None
        if existing_journal is not None:
            recovery = (
                existing_journal.get("status")
                in {"PREPARED", "CARD_WRITTEN", "RECOVERY_REQUIRED"}
                and existing_journal.get("target") == target
                and existing_journal.get("receipt") == str(receipt_path.resolve())
            )
        errors = verify_receipt(
            config_path,
            receipt_path,
            cwd=cwd,
            allow_recovery_journal=recovery,
        )
        receipt = confined_load_json(config, receipt_path)
        if recovery and existing_journal.get("commit") != receipt.get("git", {}).get(
            "commit"
        ):
            errors.append("recovery journal commit does not match receipt")
        if receipt.get("lifecycle_target") != target:
            errors.append("requested target does not match receipt lifecycle target")
        if errors:
            raise ContinuityError("promotion refused: " + "; ".join(errors))

        card_path = Path(config["basic_memory"]["card_path"])
        card_data, card_body = parse_markdown_frontmatter(
            confined_read_text(config, card_path), card_path
        )
        try:
            prior_journal = confined_load_json(config, journal_path)
        except FileNotFoundError:
            prior_journal = None
        if prior_journal is not None:
            if (
                prior_journal.get("status") == "COMMITTED"
                and prior_journal.get("target") == target
                and prior_journal.get("receipt") == str(receipt_path.resolve())
                and prior_journal.get("commit") == receipt["git"]["commit"]
                and card_data.get("state") == target
            ):
                return
        now = datetime.now(timezone.utc).isoformat()
        evidence = (
            card_data.get("evidence")
            if isinstance(card_data.get("evidence"), dict)
            else {}
        )
        evidence.update({
            "commit": receipt["git"]["commit"],
            "tested_at": now,
            "receipt": str(receipt_path.resolve()),
        })
        journal = {
            "schema_version": 1,
            "status": "PREPARED",
            "target": target,
            "receipt": str(receipt_path.resolve()),
            "commit": receipt["git"]["commit"],
            "updated_at": now,
        }
        confined_atomic_write_json(config, journal_path, journal)
        card_data["state"] = target
        card_data["evidence"] = evidence
        card_data["next_action"] = (
            "Review the exact-state receipt and decide whether to expand the pilot."
            if target == "TESTED"
            else "Monitor enforced continuity before expanding to another project."
        )
        confined_atomic_write_text(
            config, card_path, render_markdown_frontmatter(card_data, card_body)
        )
        journal["status"] = "CARD_WRITTEN"
        journal["updated_at"] = datetime.now(timezone.utc).isoformat()
        confined_atomic_write_json(config, journal_path, journal)
        if target == "ENFORCED":
            try:
                repo_root = Path(config["expected_repo_root"])
                if _beads_status(config, repo_root) != "closed":
                    _run_beads(
                        config,
                        [
                            "close",
                            config["beads"]["active_task"],
                            "--reason",
                            f"ENFORCED by exact-state receipt {receipt_path.name}",
                        ],
                        repo_root,
                    )
                if _beads_status(config, repo_root) != "closed":
                    raise ContinuityError("Beads task did not reach closed state")
            except BaseException as exc:
                journal["status"] = "RECOVERY_REQUIRED"
                journal["error_type"] = type(exc).__name__
                journal["updated_at"] = datetime.now(timezone.utc).isoformat()
                confined_atomic_write_json(config, journal_path, journal)
                raise
        journal["status"] = "COMMITTED"
        journal["updated_at"] = datetime.now(timezone.utc).isoformat()
        confined_atomic_write_json(config, journal_path, journal)


def manifest_membership_errors(
    installed: list[str], identities: list[str]
) -> list[str]:
    required = set(installed) | {"speckit"}
    actual = set(identities)
    errors: list[str] = []
    if actual != required:
        errors.append(
            "integration manifest membership mismatch: expected "
            + ", ".join(sorted(required))
            + "; got "
            + ", ".join(sorted(actual))
        )
    if len(identities) != len(actual):
        errors.append("integration manifest identities are duplicated")
    return errors


def _confined_receipt_output(config: dict[str, Any], output: Path) -> Path:
    state = validate_state_storage(config["external_volume"], config["state_root"])
    try:
        receipt_dir = Path(config["receipt_dir"]).resolve(strict=True)
    except OSError as exc:
        raise ContinuityError(f"receipt directory is unavailable: {exc}") from exc
    if not receipt_dir.is_dir() or not receipt_dir.is_relative_to(state):
        raise ContinuityError("receipt directory escapes pilot state root")
    normalized = Path(os.path.abspath(output))
    try:
        parent = normalized.parent.resolve(strict=True)
    except OSError as exc:
        raise ContinuityError(f"receipt output parent is unavailable: {exc}") from exc
    if parent != receipt_dir:
        raise ContinuityError("receipt output escapes configured receipt directory")
    with confined_parent(config, normalized):
        pass
    return normalized


def managed_manifest_file_errors(
    declared_files: set[str], managed_files: set[str]
) -> list[str]:
    missing = sorted(managed_files - declared_files)
    extra = sorted(declared_files - managed_files)
    errors: list[str] = []
    if missing:
        errors.append(
            "managed integration files missing from manifests: " + ", ".join(missing)
        )
    if extra:
        errors.append(
            "manifest files outside managed integration set: " + ", ".join(extra)
        )
    return errors


def _managed_integration_files(repo_root: Path) -> set[str]:
    candidates: set[Path] = set()
    for relative in (".specify/scripts", ".specify/templates", ".specify/workflows"):
        root = repo_root / relative
        if root.is_dir():
            candidates.update(path for path in root.rglob("*") if path.is_file())
    for relative in (
        ".specify/.gitignore",
        ".specify/events.py",
        ".specify/init-options.json",
        ".specify/integration.json",
    ):
        path = repo_root / relative
        if path.is_file():
            candidates.add(path)
    for root_name in (".agents/skills", ".claude/skills"):
        root = repo_root / root_name
        if root.is_dir():
            candidates.update(
                path
                for directory in root.glob("speckit-*")
                for path in directory.rglob("*")
                if path.is_file()
            )
    return {
        str(path.relative_to(repo_root))
        for path in candidates
        if "__pycache__" not in path.parts and path.suffix != ".pyc"
    }


def static_validate(config_path: Path) -> list[str]:
    """Validate committed structure without requiring external pilot state (CI-safe)."""
    errors: list[str] = []
    try:
        config = load_json(config_path)
    except ContinuityError as exc:
        return [str(exc)]
    repo_root = config_path.resolve().parent.parent
    for key in (
        "project",
        "repo_id",
        "risk_tier",
        "goal",
        "expected_repo_root",
        "external_volume",
        "integration_ref",
        "toolchain_lock",
    ):
        if not config.get(key):
            errors.append(f"config missing {key}")
    if config.get("risk_tier") != "T3":
        errors.append("continuity pilot risk_tier must remain T3")
    native_policy = config.get("native_observation_policy")
    if not isinstance(native_policy, dict) or set(native_policy) != {
        "required_surfaces",
        "max_age_seconds",
        "hosts",
    }:
        errors.append("native observation policy has an invalid closed schema")
        required_native_surfaces: set[str] = set()
    else:
        surfaces = native_policy.get("required_surfaces")
        max_age = native_policy.get("max_age_seconds")
        hosts = native_policy.get("hosts")
        if (
            not isinstance(surfaces, list)
            or not surfaces
            or not all(isinstance(item, str) and item for item in surfaces)
            or len(set(surfaces)) != len(surfaces)
            or not isinstance(max_age, int)
            or max_age < 1
            or max_age > 3600
            or not isinstance(hosts, dict)
            or not set(surfaces).issubset(hosts)
            or any(
                not isinstance(surface, str)
                or not isinstance(identity, dict)
                or set(identity) != {"path", "sha256"}
                or not isinstance(identity.get("path"), str)
                or not identity["path"]
                or not isinstance(identity.get("sha256"), str)
                or re.fullmatch(r"[0-9a-f]{64}", identity["sha256"]) is None
                for surface, identity in (hosts or {}).items()
            )
        ):
            errors.append("native observation policy is invalid")
            required_native_surfaces = set()
        else:
            required_native_surfaces = set(surfaces)
    dependency_policy = config.get("dependency_identity")
    if not isinstance(dependency_policy, dict) or set(dependency_policy) != {
        "python",
        "requirements_lock",
    }:
        errors.append("dependency identity policy has an invalid closed schema")
    else:
        requirements = dependency_policy.get("requirements_lock")
        requirements_path = (
            (repo_root / requirements).resolve()
            if isinstance(requirements, str)
            else repo_root.parent
        )
        if (
            not requirements_path.is_relative_to(repo_root)
            or not requirements_path.is_file()
        ):
            errors.append("dependency identity requirements lock is missing")
        if (
            not isinstance(dependency_policy.get("python"), str)
            or not dependency_policy["python"]
        ):
            errors.append("dependency identity Python is invalid")
    for relative in config.get("instructions", []):
        if not (repo_root / relative).is_file():
            errors.append(f"instruction missing: {relative}")
    policy = config.get("evidence_policy")
    if not isinstance(policy, dict) or set(policy) != {
        "focused_suite",
        "full_suite",
        "runtime_checks",
        "rollback_check",
    }:
        errors.append("evidence_policy has an invalid closed schema")
    elif not isinstance(policy["runtime_checks"], list):
        errors.append("evidence_policy runtime_checks must be a list")
    else:
        policy_specs = [
            (policy["focused_suite"], COMMAND_SPEC_KEYS, "focused suite"),
            (policy["full_suite"], COMMAND_SPEC_KEYS, "full suite"),
            *(
                (item, RUNTIME_SPEC_KEYS, "runtime check")
                for item in policy["runtime_checks"]
            ),
            (policy["rollback_check"], ROLLBACK_SPEC_KEYS, "rollback check"),
        ]
        for value, allowed, label in policy_specs:
            if not isinstance(value, dict):
                errors.append(f"evidence_policy {label} must be an object")
                continue
            try:
                _validate_closed_spec(value, allowed, f"evidence_policy {label}")
            except ContinuityError as exc:
                errors.append(str(exc))
        native_runtime_surfaces: set[str] = set()
        for runtime in policy["runtime_checks"]:
            if not isinstance(runtime, dict):
                continue
            surface = runtime.get("surface")
            adapter_token = runtime.get("adapter_path")
            if (
                runtime.get("native") is not True
                or not isinstance(surface, str)
                or not surface
                or not isinstance(runtime.get("event"), str)
                or not runtime["event"]
                or not isinstance(adapter_token, str)
                or not adapter_token
            ):
                errors.append("runtime check is not a valid native observation")
                continue
            try:
                artifacts = _native_bound_artifacts(
                    config, repo_root, runtime, require_exists=True
                )
            except ContinuityError:
                artifacts = None
            if artifacts is None:
                errors.append(
                    f"native observation bound artifacts are invalid: {surface}"
                )
            native_runtime_surfaces.add(surface)
        if native_runtime_surfaces != required_native_surfaces:
            errors.append(
                "native observation runtime surfaces do not match required surfaces"
            )
        if len(native_runtime_surfaces) != len(policy["runtime_checks"]):
            errors.append("native observation runtime surfaces are duplicated")
        try:
            policy_entrypoints = {
                str(spec["argv"][0]) for spec in _policy_evidence_specs(policy)
            }
        except (ContinuityError, KeyError, IndexError, TypeError) as exc:
            errors.append(str(exc))
    try:
        evidence_pins = _evidence_pin_map(config)
        if policy_entrypoints is not None and set(evidence_pins) != policy_entrypoints:
            errors.append(
                "evidence executable pins do not exactly match policy entrypoints"
            )
    except ContinuityError as exc:
        errors.append(str(exc))
    for item in config.get("external_instructions", []):
        if (
            not isinstance(item, dict)
            or set(item) != {"path", "sha256"}
            or not isinstance(item.get("sha256"), str)
            or re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) is None
        ):
            errors.append("external instruction pin is malformed")
    spec_kit_version: str | None = None
    try:
        spec, _ = read_markdown_frontmatter(repo_root / config["spec"]["path"])
        if spec.get("change_id") != config["spec"].get("change_id"):
            errors.append("spec change_id does not match config")
        if spec.get("active_task") != config["beads"].get("active_task"):
            errors.append("spec active_task does not match config")
    except (ContinuityError, KeyError) as exc:
        errors.append(str(exc))
    for relative in (
        "scripts/continuity_bridge.py",
        "scripts/continuity_gate.py",
        "scripts/continuity_event.py",
        ".specify/events.py",
        ".claude/settings.json",
        ".codex/hooks.json",
    ):
        if not (repo_root / relative).is_file():
            errors.append(f"continuity surface missing: {relative}")
    try:
        lock = load_json(repo_root / config["toolchain_lock"])
        tools = lock.get("tools")
        if lock.get("schema_version") != 1 or not isinstance(tools, dict):
            errors.append("toolchain lock has an invalid schema")
        else:
            for name, tool in tools.items():
                if not isinstance(tool, dict) or not tool.get("version"):
                    errors.append(f"toolchain lock entry is invalid: {name}")
                source = tool.get("source") if isinstance(tool, dict) else None
                if not isinstance(source, str) or not source.startswith("https://"):
                    errors.append(f"toolchain lock source is not HTTPS: {name}")
            beads = tools.get("beads")
            spec_kit = tools.get("spec-kit")
            if isinstance(spec_kit, dict) and isinstance(spec_kit.get("version"), str):
                spec_kit_version = spec_kit["version"]
            hashes = [
                value for key, value in (beads or {}).items() if key.endswith("-sha256")
            ]
            if not hashes or not all(
                isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
                for value in hashes
            ):
                errors.append("toolchain lock has no valid Beads executable digest")
    except (ContinuityError, KeyError) as exc:
        errors.append(str(exc))

    manifests = sorted((repo_root / ".specify/integrations").glob("*.manifest.json"))
    if not manifests:
        errors.append("no integration supply-chain manifests are committed")
    identities: list[str] = []
    manifest_versions: list[str] = []
    declared_files: set[str] = set()
    for manifest_path in manifests:
        try:
            manifest = load_json(manifest_path)
            identities.append(str(manifest.get("integration") or ""))
            manifest_versions.append(str(manifest.get("version") or ""))
            files = manifest.get("files")
            if not manifest.get("integration") or not manifest.get("version"):
                errors.append(
                    f"integration manifest identity missing: {manifest_path.name}"
                )
            if not isinstance(files, dict):
                errors.append(
                    f"integration manifest files invalid: {manifest_path.name}"
                )
                continue
            for relative, expected in files.items():
                declared_files.add(relative)
                candidate = (repo_root / relative).resolve()
                if not candidate.is_relative_to(repo_root):
                    errors.append(f"integration manifest path escapes repo: {relative}")
                elif not candidate.is_file():
                    errors.append(f"integration manifest file missing: {relative}")
                elif sha256_file(candidate) != expected:
                    errors.append(f"integration manifest digest mismatch: {relative}")
        except ContinuityError as exc:
            errors.append(str(exc))
    try:
        integration_state = load_json(repo_root / ".specify/integration.json")
        installed = integration_state.get("installed_integrations")
        if not isinstance(installed, list) or not all(
            isinstance(item, str) and item for item in installed
        ):
            errors.append("Spec Kit installed integration list is invalid")
        else:
            errors.extend(manifest_membership_errors(installed, identities))
        integration_version = integration_state.get("version")
        if not isinstance(integration_version, str) or any(
            version != integration_version for version in manifest_versions
        ):
            errors.append(
                "integration manifest versions do not match integration state"
            )
        if spec_kit_version != integration_version:
            errors.append("integration state version does not match Spec Kit lock")
    except ContinuityError as exc:
        errors.append(str(exc))
    errors.extend(
        managed_manifest_file_errors(
            declared_files, _managed_integration_files(repo_root)
        )
    )
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("create-receipt")
    create.add_argument("--config", type=Path, default=Path(".continuity/config.json"))
    create.add_argument("--cwd", type=Path, default=Path.cwd())
    create.add_argument("--risk-tier", choices=("T0", "T1", "T2", "T3"), required=True)
    create.add_argument("--target", choices=("TESTED", "ENFORCED"), required=True)
    create.add_argument("--commands-json", type=Path)
    create.add_argument("--runtime-json", type=Path)
    create.add_argument("--rollback-json", type=Path)
    create.add_argument("--output", type=Path, required=True)

    verify = subparsers.add_parser("verify-receipt")
    verify.add_argument("--config", type=Path, default=Path(".continuity/config.json"))
    verify.add_argument("--cwd", type=Path, default=Path.cwd())
    verify.add_argument("--receipt", type=Path, required=True)

    promote_parser = subparsers.add_parser("promote")
    promote_parser.add_argument(
        "--config", type=Path, default=Path(".continuity/config.json")
    )
    promote_parser.add_argument("--cwd", type=Path, default=Path.cwd())
    promote_parser.add_argument("--receipt", type=Path, required=True)
    promote_parser.add_argument(
        "--target", choices=("TESTED", "ENFORCED"), required=True
    )

    static = subparsers.add_parser("static")
    static.add_argument("--config", type=Path, default=Path(".continuity/config.json"))

    args = parser.parse_args(argv)
    try:
        if args.command == "create-receipt":
            output_config = load_json(args.config)
            output_path = _confined_receipt_output(output_config, args.output)
            receipt = create_receipt(
                args.config,
                cwd=args.cwd,
                risk_tier=args.risk_tier,
                lifecycle_target=args.target,
                commands=_load_list(args.commands_json, "command evidence"),
                runtime_checks=_load_list(args.runtime_json, "runtime evidence"),
                rollback=_load_object(args.rollback_json, "rollback evidence"),
                require_mounted_volume=True,
            )
            confined_atomic_write_json(output_config, output_path, receipt)
            print(
                json.dumps({"result": receipt["result"], "receipt": str(output_path)})
            )
            return 0 if receipt["result"] == "PASS" else 2
        if args.command == "verify-receipt":
            errors = verify_receipt(args.config, args.receipt, cwd=args.cwd)
            print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True))
            return 0 if not errors else 2
        if args.command == "promote":
            promote(args.config, args.receipt, cwd=args.cwd, target=args.target)
            print(json.dumps({"promoted": args.target, "receipt": str(args.receipt)}))
            return 0
        errors = static_validate(args.config)
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True))
        return 0 if not errors else 2
    except (ContinuityError, KeyError, TypeError, ValueError) as exc:
        print(json.dumps({"valid": False, "errors": [str(exc)]}, sort_keys=True))
        return 2


if __name__ == "__main__":
    sys.exit(main())
