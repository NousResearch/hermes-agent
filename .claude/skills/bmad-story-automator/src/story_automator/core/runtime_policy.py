from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .frontmatter import parse_simple_frontmatter
from .runtime_layout import active_marker_path, bundled_story_skill_root, resolve_portable_path, resolve_skill_dir
from .utils import ensure_dir, get_project_root, iso_now, md5_hex8, read_text, write_atomic

VALID_TOP_LEVEL_KEYS = {"version", "snapshot", "runtime", "workflow", "steps"}
VALID_STEP_NAMES = {"create", "dev", "auto", "review", "retro"}
VALID_VERIFIERS = {"create_story_artifact", "session_exit", "review_completion", "epic_complete"}
VALID_ASSET_NAMES = {"skill", "workflow", "instructions", "checklist", "template"}
VALID_PARSER_PROVIDERS = {"claude"}


def load_bundled_policy(project_root: str | None = None, *, resolve_assets: bool = True) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    bundle_root = bundled_skill_root(root)
    policy = _read_json(bundle_root / "data" / "orchestration-policy.json")
    _validate_policy_shape(policy)
    if resolve_assets:
        _resolve_policy_paths(policy, project_root=root, bundle_root=bundle_root)
    else:
        _resolve_success_paths(policy, project_root=root, bundle_root=bundle_root)
    return policy


class PolicyError(ValueError):
    pass


def load_effective_policy(project_root: str | None = None, *, resolve_assets: bool = True) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    bundled = load_bundled_policy(str(root), resolve_assets=False)
    override_path = root / "_bmad" / "bmm" / "story-automator.policy.json"
    override = _read_json(override_path) if override_path.is_file() else {}
    policy = _deep_merge(bundled, override)
    _apply_legacy_env(policy)
    _validate_policy_shape(policy)
    _clear_resolved_fields(policy)
    if resolve_assets:
        _resolve_policy_paths(policy, project_root=root, bundle_root=bundled_skill_root(root))
    else:
        _resolve_success_paths(policy, project_root=root, bundle_root=bundled_skill_root(root))
    return policy


def load_runtime_policy(
    project_root: str | None = None,
    state_file: str | Path | None = None,
    *,
    resolve_assets: bool = True,
) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    resolved_state, source = resolve_policy_state_file(root, state_file)
    if resolved_state:
        state_path = Path(resolved_state)
        if source in {"env", "marker"} and not state_path.is_file():
            raise PolicyError(f"{source} state file missing: {state_path}")
        if source != "explicit" and not state_path.is_file():
            return load_effective_policy(str(root), resolve_assets=resolve_assets)
        return load_policy_for_state(str(state_path), project_root=str(root), resolve_assets=resolve_assets)
    return load_effective_policy(str(root), resolve_assets=resolve_assets)


def snapshot_effective_policy(project_root: str | None = None) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    policy = load_effective_policy(str(root))
    snapshot_dir = _resolve_snapshot_dir(policy, root)
    ensure_dir(snapshot_dir)
    stable_json = _stable_policy_json(policy)
    snapshot_hash = md5_hex8(stable_json)
    stamp = iso_now().replace("-", "").replace(":", "").replace("T", "-").replace("Z", "")
    snapshot_path = snapshot_dir / f"{stamp}-{snapshot_hash}.json"
    write_atomic(snapshot_path, stable_json)
    return {
        "policy": policy,
        "policyVersion": policy.get("version", 1),
        "policySnapshotHash": snapshot_hash,
        "policySnapshotFile": _display_path(snapshot_path, root),
    }


def load_policy_snapshot(
    snapshot_file: str,
    *,
    project_root: str | None = None,
    expected_hash: str = "",
    resolve_assets: bool = True,
) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    path = Path(snapshot_file)
    if not path.is_absolute():
        path = root / path
    path = _ensure_within(path, root, "policy snapshot")
    if not path.is_file():
        raise PolicyError(f"policy snapshot missing: {path}")
    try:
        raw = read_text(path)
    except OSError as exc:
        raise PolicyError(f"policy snapshot unreadable: {path}") from exc
    actual_hash = md5_hex8(raw)
    if expected_hash and actual_hash != expected_hash:
        raise PolicyError(f"policy snapshot hash mismatch: expected {expected_hash}, got {actual_hash}")
    try:
        policy = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise PolicyError(f"policy json invalid: {path}") from exc
    _validate_policy_shape(policy)
    if resolve_assets:
        _resolve_policy_paths(policy, project_root=root, bundle_root=bundled_skill_root(root))
    else:
        _resolve_success_paths(policy, project_root=root, bundle_root=bundled_skill_root(root))
    return policy


def load_policy_for_state(
    state_file: str | Path,
    project_root: str | None = None,
    *,
    resolve_assets: bool = True,
) -> dict[str, Any]:
    root = Path(project_root or get_project_root()).resolve()
    try:
        fields = parse_simple_frontmatter(read_text(state_file))
    except OSError as exc:
        raise PolicyError(f"state file unreadable: {state_file}") from exc
    snapshot_file, snapshot_hash, legacy_mode = _state_policy_mode(fields)
    if not legacy_mode:
        return load_policy_snapshot(
            snapshot_file,
            project_root=str(root),
            expected_hash=snapshot_hash,
            resolve_assets=resolve_assets,
        )
    return load_bundled_policy(str(root), resolve_assets=resolve_assets)


def summarize_state_policy_fields(fields: dict[str, Any], *, project_root: str | Path | None = None) -> tuple[str, str, str, str, str]:
    policy_version = str(fields.get("policyVersion") or "").strip()
    try:
        snapshot_file, snapshot_hash, legacy_mode = _state_policy_mode(fields)
        if snapshot_file and snapshot_hash:
            load_policy_snapshot(
                snapshot_file,
                project_root=str(Path(project_root or get_project_root()).resolve()),
                expected_hash=snapshot_hash,
                resolve_assets=False,
            )
    except PolicyError as exc:
        return "", "", policy_version, "false", str(exc)
    return snapshot_file, snapshot_hash, policy_version, "true" if legacy_mode else "false", ""


def resolve_policy_state_file(project_root: str | Path | None = None, state_file: str | Path | None = None) -> tuple[str, str]:
    root = Path(project_root or get_project_root()).resolve()
    explicit = Path(state_file).expanduser() if state_file else None
    if explicit:
        return str(_resolve_state_path(root, explicit)), "explicit"
    env_state = os.environ.get("STORY_AUTOMATOR_STATE_FILE", "").strip()
    if env_state:
        return str(_resolve_state_path(root, Path(env_state).expanduser(), allow_outside=False, label="env state file")), "env"
    marker = active_marker_path(root)
    if marker.is_file():
        try:
            payload = _read_json(marker)
        except PolicyError as exc:
            raise PolicyError(f"active-run marker invalid: {exc}") from exc
        marker_state = str(payload.get("stateFile") or "").strip()
        if not marker_state:
            raise PolicyError("active-run marker missing stateFile")
        return str(_resolve_state_path(root, Path(marker_state).expanduser(), allow_outside=False, label="marker state file")), "marker"
    return "", ""


def step_contract(policy: dict[str, Any], step: str) -> dict[str, Any]:
    contract = (policy.get("steps") or {}).get(step)
    if not isinstance(contract, dict):
        raise PolicyError(f"unknown step: {step}")
    return contract


def review_max_cycles(policy: dict[str, Any]) -> int:
    repeat = ((policy.get("workflow") or {}).get("repeat") or {}).get("review") or {}
    return int(repeat.get("maxCycles", 5))


def crash_max_retries(policy: dict[str, Any]) -> int:
    crash = ((policy.get("workflow") or {}).get("crash")) or {}
    return int(crash.get("maxRetries", 2))


def parser_runtime_config(policy: dict[str, Any]) -> dict[str, object]:
    runtime = _expect_optional_dict(policy, "runtime")
    parser = _expect_optional_nested_dict(runtime, "parser", "runtime")
    provider = str(parser.get("provider") or "").strip()
    model = str(parser.get("model") or "").strip()
    timeout = parser.get("timeoutSeconds")
    if provider not in VALID_PARSER_PROVIDERS:
        raise PolicyError(f"runtime.parser.provider must be one of: {', '.join(sorted(VALID_PARSER_PROVIDERS))}")
    if not model:
        raise PolicyError("runtime.parser.model must be a string")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise PolicyError("runtime.parser.timeoutSeconds must be a positive integer")
    return {"provider": provider, "model": model, "timeoutSeconds": timeout}


def bundled_skill_root(project_root: str | Path | None = None) -> Path:
    root = Path(project_root or get_project_root()).resolve()
    try:
        return bundled_story_skill_root(root)
    except FileNotFoundError as exc:
        raise PolicyError("bundled policy not found") from exc


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        payload = json.loads(read_text(path))
    except json.JSONDecodeError as exc:
        raise PolicyError(f"policy json invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise PolicyError(f"policy json must be an object: {path}")
    return payload


def _deep_merge(base: Any, override: Any) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        merged = dict(base)
        for key, value in override.items():
            merged[key] = _deep_merge(merged[key], value) if key in merged else value
        return merged
    if isinstance(override, list):
        return list(override)
    return override


def _clear_resolved_fields(policy: dict[str, Any]) -> None:
    for contract in (policy.get("steps") or {}).values():
        if not isinstance(contract, dict):
            continue
        assets = contract.get("assets")
        if isinstance(assets, dict):
            assets.pop("files", None)
        prompt = contract.get("prompt")
        if isinstance(prompt, dict):
            prompt.pop("templatePath", None)
            prompt.pop("templateHash", None)
        parse = contract.get("parse")
        if isinstance(parse, dict):
            parse.pop("schemaPath", None)
            parse.pop("schemaHash", None)
        success = contract.get("success")
        if isinstance(success, dict):
            success.pop("contractPath", None)
            success.pop("contractHash", None)


def _apply_legacy_env(policy: dict[str, Any]) -> None:
    review_cycles = os.environ.get("MAX_REVIEW_CYCLES")
    crash_retries = os.environ.get("MAX_CRASH_RETRIES")
    if review_cycles:
        policy.setdefault("workflow", {}).setdefault("repeat", {}).setdefault("review", {})["maxCycles"] = _legacy_env_int(
            "MAX_REVIEW_CYCLES",
            review_cycles,
        )
    if crash_retries:
        policy.setdefault("workflow", {}).setdefault("crash", {})["maxRetries"] = _legacy_env_int(
            "MAX_CRASH_RETRIES",
            crash_retries,
        )


def _legacy_env_int(name: str, raw: str) -> int:
    try:
        return int(raw)
    except ValueError as exc:
        raise PolicyError(f"{name} must be an integer") from exc


def _validate_policy_shape(policy: dict[str, Any]) -> None:
    unknown_keys = sorted(set(policy) - VALID_TOP_LEVEL_KEYS)
    if unknown_keys:
        raise PolicyError(f"unknown top-level policy keys: {', '.join(unknown_keys)}")
    snapshot = _expect_optional_dict(policy, "snapshot")
    if "snapshot" in policy and "relativeDir" in snapshot and not isinstance(snapshot.get("relativeDir"), str):
        raise PolicyError("snapshot.relativeDir must be a string")
    runtime = _expect_optional_dict(policy, "runtime")
    _expect_optional_nested_dict(runtime, "merge", "runtime")
    parser_runtime_config(policy)
    workflow = _expect_optional_dict(policy, "workflow")
    repeat = _expect_optional_nested_dict(workflow, "repeat", "workflow")
    review = _expect_optional_nested_dict(repeat, "review", "workflow.repeat")
    crash = _expect_optional_nested_dict(workflow, "crash", "workflow")
    steps = policy.get("steps")
    if not isinstance(steps, dict):
        raise PolicyError("steps must be an object")
    unknown_steps = sorted(set(steps) - VALID_STEP_NAMES)
    if unknown_steps:
        raise PolicyError(f"unknown step names: {', '.join(unknown_steps)}")
    sequence = (workflow.get("sequence")) or []
    if not isinstance(sequence, list) or not all(isinstance(item, str) for item in sequence):
        raise PolicyError("workflow.sequence must be a string array")
    if "maxCycles" in review and not isinstance(review.get("maxCycles"), int):
        raise PolicyError("workflow.repeat.review.maxCycles must be an integer")
    if "maxRetries" in crash and not isinstance(crash.get("maxRetries"), int):
        raise PolicyError("workflow.crash.maxRetries must be an integer")
    for step in sequence:
        if step not in steps:
            raise PolicyError(f"workflow.sequence references missing step: {step}")
    for name, contract in steps.items():
        if not isinstance(contract, dict):
            raise PolicyError(f"step contract must be an object: {name}")
        assets = _expect_step_dict(contract, "assets", name)
        _expect_step_dict(contract, "prompt", name)
        _expect_step_dict(contract, "parse", name)
        _expect_step_dict(contract, "success", name)
        verifier = str(((contract.get("success") or {}).get("verifier")) or "")
        if verifier not in VALID_VERIFIERS:
            raise PolicyError(f"invalid verifier for {name}: {verifier}")
        required = (assets.get("required")) or []
        if not isinstance(required, list) or any(item not in VALID_ASSET_NAMES for item in required):
            raise PolicyError(f"invalid required assets for {name}")


def _resolve_policy_paths(policy: dict[str, Any], *, project_root: Path, bundle_root: Path) -> None:
    for name, contract in (policy.get("steps") or {}).items():
        assets = contract.setdefault("assets", {})
        assets["files"] = _resolve_step_assets(name, assets, project_root)
        prompt = contract.setdefault("prompt", {})
        template_file = str(prompt.get("templateFile") or "").strip()
        if not template_file:
            raise PolicyError(f"missing prompt template for {name}")
        prompt["templatePath"] = _resolve_data_path(template_file, project_root=project_root, bundle_root=bundle_root)
        _set_or_verify_hash(prompt, path_key="templatePath", hash_key="templateHash", label="policy template")
        parse = contract.setdefault("parse", {})
        schema_file = str(parse.get("schemaFile") or "").strip()
        if not schema_file:
            raise PolicyError(f"missing parse schema for {name}")
        parse["schemaPath"] = _resolve_data_path(schema_file, project_root=project_root, bundle_root=bundle_root)
        _set_or_verify_hash(parse, path_key="schemaPath", hash_key="schemaHash", label="policy parse schema")
        success = contract.setdefault("success", {})
        contract_file = str(success.get("contractFile") or "").strip()
        if contract_file:
            success["contractPath"] = _resolve_data_path(contract_file, project_root=project_root, bundle_root=bundle_root)
            _set_or_verify_hash(success, path_key="contractPath", hash_key="contractHash", label="policy success contract")


def _resolve_success_paths(policy: dict[str, Any], *, project_root: Path, bundle_root: Path) -> None:
    for contract in (policy.get("steps") or {}).values():
        success = contract.setdefault("success", {})
        contract_file = str(success.get("contractFile") or "").strip()
        if contract_file:
            success["contractPath"] = _resolve_data_path(contract_file, project_root=project_root, bundle_root=bundle_root)
            _set_or_verify_hash(success, path_key="contractPath", hash_key="contractHash", label="policy success contract")


def _resolve_step_assets(step: str, assets: dict[str, Any], project_root: Path) -> dict[str, str]:
    skill_name = str(assets.get("skillName") or "").strip()
    if not skill_name:
        raise PolicyError(f"missing skillName for {step}")
    try:
        skill_dir = resolve_skill_dir(project_root, skill_name)
    except ValueError as exc:
        raise PolicyError(str(exc)) from exc
    skills_root = skill_dir.parent
    required = set(assets.get("required") or [])
    files = {
        "skill": _resolve_required_file(skill_dir / "SKILL.md", project_root, required, "skill", step),
        "workflow": _resolve_candidate_file(skill_dir, assets.get("workflowCandidates"), project_root, required, "workflow", step),
        "instructions": _resolve_candidate_file(skill_dir, assets.get("instructionsCandidates"), project_root, required, "instructions", step),
        "checklist": _resolve_candidate_file(skill_dir, assets.get("checklistCandidates"), project_root, required, "checklist", step),
        "template": _resolve_candidate_file(skill_dir, assets.get("templateCandidates"), project_root, required, "template", step),
    }
    if not files["skill"]:
        files["workflow"] = ""
        files["instructions"] = ""
        files["checklist"] = ""
        files["template"] = ""
    return files


def _resolve_required_file(path: Path, project_root: Path, required: set[str], asset: str, step: str) -> str:
    if path.is_file():
        return _display_path(path, project_root)
    if asset in required:
        raise PolicyError(f"missing required {asset} asset for {step}: {path}")
    return ""


def _resolve_candidate_file(
    skill_dir: Path,
    candidates: Any,
    project_root: Path,
    required: set[str],
    asset: str,
    step: str,
) -> str:
    if not isinstance(candidates, list):
        candidates = []
    for name in candidates:
        if not isinstance(name, str) or not name:
            continue
        path = _ensure_within(skill_dir / name, skill_dir, f"{asset} candidate for {step}")
        if path.is_file():
            return _display_path(path, project_root)
    if asset == "workflow" and asset in required:
        skill_file = skill_dir / "SKILL.md"
        if skill_file.is_file():
            return _display_path(skill_file, project_root)
    if asset in required:
        searched = ", ".join(str(skill_dir / str(name)) for name in candidates if isinstance(name, str) and name)
        raise PolicyError(f"missing required {asset} asset for {step}: {searched}")
    return ""


def _resolve_data_path(path_value: str, *, project_root: Path, bundle_root: Path) -> str:
    portable = resolve_portable_path(path_value, project_root)
    if portable:
        if not portable.is_file():
            raise PolicyError(f"policy data file missing: {path_value}")
        return str(portable)
    raw = Path(path_value)
    allowed_roots = (bundle_root.resolve(), project_root.resolve())
    if raw.is_absolute():
        resolved = raw.resolve()
        if not _is_within_any(resolved, allowed_roots):
            raise PolicyError(f"policy data path escapes allowed roots: {path_value}")
        if not resolved.is_file():
            raise PolicyError(f"policy data file missing: {raw}")
        return str(resolved)
    escaped_all = True
    for base in allowed_roots:
        candidate = (base / raw).resolve()
        if not _is_within(candidate, base):
            continue
        escaped_all = False
        if candidate.is_file():
            return str(candidate)
    if escaped_all:
        raise PolicyError(f"policy data path escapes allowed roots: {path_value}")
    raise PolicyError(f"policy data file missing: {path_value}")


def _snapshot_relative_dir(policy: dict[str, Any]) -> str:
    snapshot = _expect_optional_dict(policy, "snapshot")
    relative_dir = str(snapshot.get("relativeDir") or "").strip()
    if not relative_dir:
        raise PolicyError("snapshot.relativeDir missing")
    return relative_dir


def _resolve_snapshot_dir(policy: dict[str, Any], project_root: Path) -> Path:
    raw = Path(_snapshot_relative_dir(policy))
    candidate = raw if raw.is_absolute() else project_root / raw
    return _ensure_within(candidate, project_root.resolve(), "snapshot.relativeDir")


def _stable_policy_json(policy: dict[str, Any]) -> str:
    return json.dumps(policy, indent=2, sort_keys=True) + "\n"


def _display_path(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _resolve_state_path(project_root: Path, path: Path, *, allow_outside: bool = True, label: str = "state file") -> Path:
    candidate = path if path.is_absolute() else project_root / path
    if allow_outside:
        return candidate.resolve()
    return _ensure_within(candidate, project_root.resolve(), label)


def _set_or_verify_hash(payload: dict[str, Any], *, path_key: str, hash_key: str, label: str) -> None:
    path = str(payload.get(path_key) or "").strip()
    if not path:
        return
    actual = md5_hex8(read_text(path))
    expected = str(payload.get(hash_key) or "").strip()
    if expected and expected != actual:
        raise PolicyError(f"{label} hash mismatch: {path}")
    payload[hash_key] = actual


def _ensure_within(path: Path, root: Path, label: str) -> Path:
    resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise PolicyError(f"{label} escapes allowed root: {path}") from exc
    return resolved


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _is_within_any(path: Path, roots: tuple[Path, ...]) -> bool:
    return any(_is_within(path, root) for root in roots)


def _state_policy_mode(fields: dict[str, Any]) -> tuple[str, str, bool]:
    snapshot_file = str(fields.get("policySnapshotFile") or "").strip()
    snapshot_hash = str(fields.get("policySnapshotHash") or "").strip()
    policy_version = str(fields.get("policyVersion") or "").strip()
    legacy_policy = str(fields.get("legacyPolicy") or "").strip().lower()
    if snapshot_file or snapshot_hash:
        if not snapshot_file or not snapshot_hash:
            raise PolicyError("state policy metadata incomplete")
        if legacy_policy == "true":
            raise PolicyError("state policy metadata contradictory")
        return snapshot_file, snapshot_hash, False
    if legacy_policy == "false" or policy_version:
        raise PolicyError("state policy snapshot missing")
    if legacy_policy == "true":
        return "", "", True
    return "", "", True


def _expect_optional_dict(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PolicyError(f"{key} must be an object")
    return value


def _expect_step_dict(contract: dict[str, Any], key: str, step: str) -> dict[str, Any]:
    value = contract.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PolicyError(f"{step}.{key} must be an object")
    return value


def _expect_optional_nested_dict(payload: dict[str, Any], key: str, label: str) -> dict[str, Any]:
    value = payload.get(key)
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PolicyError(f"{label}.{key} must be an object")
    return value
