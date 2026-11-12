"""Read-only local GPU dispatch checks separated from the Kanban facade."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Optional
from urllib.parse import urlsplit
from urllib.request import urlopen


def _capability_route_error(capability, capability_id, profile_home, model, provider, model_cfg, provider_cfg, base_url):
    trial_binding = capability.get("trial_binding") or {}
    route_evidence = capability.get("profile_route_evidence") or {}
    mapping_review = trial_binding.get("mapping_review") or {}
    if (trial_binding.get("admission_capability_id") != capability_id
            or not trial_binding.get("trial_task_class")
            or trial_binding.get("trial_task_class") != route_evidence.get("task_class")
            or mapping_review.get("status") != "approved"
            or not str(mapping_review.get("reviewer") or "").strip()
            or not str(mapping_review.get("reviewed_at_utc") or "").strip()):
        return "trial task-class mapping is missing or not independently reviewed"
    if (not str(capability.get("status") or "").startswith("narrowly_supported")
            or capability.get("request_schema_policy") != "json_schema_from_fixture_types_v1"):
        return "capability status or response-schema policy is outside the supported envelope"

    runtime = capability.get("observed_runtime") or {}
    expected_profile = str(runtime.get("profile") or "")
    if Path(profile_home).name != expected_profile:
        return "assigned profile differs from the experimentally measured profile"
    if model != capability.get("model_tag") or provider != str(runtime.get("backend") or "").lower():
        return "task model/provider differs from the pinned capability route"
    if (int(model_cfg.get("ollama_num_ctx") or 0) != int(runtime.get("ollama_num_ctx") or 0)
            or int(model_cfg.get("context_length") or 0) != int(runtime.get("context_length") or 0)):
        return "configured request context is absent or differs from the measured context"
    route = {
        "profile": expected_profile,
        "profile_default_model": model_cfg.get("default"),
        "provider": model_cfg.get("provider"),
        "ollama_num_ctx": model_cfg.get("ollama_num_ctx"),
        "api_base": provider_cfg.get("base_url"),
    }
    route_hash = hashlib.sha256(json.dumps(
        route, sort_keys=True, separators=(",", ":"),
    ).encode()).hexdigest()
    if route_hash != runtime.get("profile_route_sha256"):
        return "profile route settings differ from the measured route hash"
    expected_url = urlsplit(str(runtime.get("base_url") or ""))
    current_url = urlsplit(base_url)
    if ((current_url.scheme, current_url.hostname, current_url.port, current_url.path.rstrip("/"))
            != (expected_url.scheme, expected_url.hostname, expected_url.port, expected_url.path.rstrip("/"))):
        return "effective local endpoint differs from the measured route"

    return None


def _verify_artifacts_and_model(row, capability, registry_path, provider, parsed, model, urlopen_fn):
    runtime = capability.get("observed_runtime") or {}
    verifier = capability.get("prepared_external_verifier") or {}
    if verifier.get("task_id") != row["id"]:
        return "pinned capability has no verifier for this exact task"
    if verifier.get("state") in ("prepared_not_executed", "missing", "rejected"):
        return "task-specific independent verifier has not passed its trial"
    root = registry_path.parent.resolve(strict=True)
    for path_key, hash_key in (("verifier_path", "verifier_sha256"),
                               ("expected_fixture_path", "expected_fixture_sha256")):
        asset = (root / str(verifier.get(path_key) or "")).resolve(strict=True)
        if not asset.is_relative_to(root) or hashlib.sha256(asset.read_bytes()).hexdigest() != verifier.get(hash_key):
            return "pinned verifier or expected fixture digest mismatch"

    if provider != "ollama" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        return "only the measured loopback Ollama route is supported by this preflight"
    tags_url = f"{parsed.scheme}://{parsed.netloc}/api/tags"
    with urlopen_fn(tags_url, timeout=2) as response:
        tags = json.loads(response.read(1024 * 1024 + 1).decode("utf-8"))
    matches = [item for item in tags.get("models", []) if item.get("name") == model]
    if len(matches) != 1 or matches[0].get("digest") != capability.get("model_digest"):
        return "installed model digest differs from the proven capability digest"
    details = matches[0].get("details") or {}
    if str(details.get("quantization_level") or "").upper() != str(runtime.get("quantization") or "").upper():
        return "installed model quantization differs from the proven capability"
    with urlopen_fn(f"{parsed.scheme}://{parsed.netloc}/api/version", timeout=2) as response:
        version = json.loads(response.read(64 * 1024 + 1).decode("utf-8"))
    if version.get("version") != runtime.get("runtime_version"):
        return "Ollama runtime version differs from the proven capability"

    return None


def local_gpu_dispatch_preflight(
    row: Mapping[str, Any], assignee: str, *, worker_profile_scope: Callable,
    urlopen_fn: Callable = urlopen, now: Callable[[], float] = time.time,
) -> Optional[str]:
    """Refuse dispatch unless the task, profile, runtime, and lease state match."""
    from hermes_cli.kanban_local_model_scope import capability_id_from_body

    capability_id = capability_id_from_body(row["body"])
    try:
        from hermes_cli.profiles import resolve_profile_env
        profile_home = resolve_profile_env(assignee)
        if not profile_home:
            return "assigned profile configuration is unavailable"
        with worker_profile_scope(profile_home):
            from hermes_cli.config import load_config_readonly
            config = load_config_readonly() or {}
        model_cfg = config.get("model") or {}
        provider = str(row["provider_override"] or model_cfg.get("provider") or "").strip().lower()
        model = str(row["model_override"] or model_cfg.get("default") or "").strip()
        provider_cfg = (config.get("providers") or {}).get(provider) or {}
        base_url = str(provider_cfg.get("base_url") or model_cfg.get("base_url") or "").strip()
        from hermes_cli.local_model_routes import is_local_llm_route

        parsed = urlsplit(base_url)
        is_local = is_local_llm_route(provider, base_url)
        if not is_local and not capability_id:
            return None
        if not capability_id:
            return "effective profile route is local but task has no Local-GPU-Capability marker"
        if not is_local:
            return "capability-marked task resolves to a non-local provider route"

        plugins_cfg = config.get("plugins") or {}
        entry = (plugins_cfg.get("entries") or {}).get("local_gpu_admission") or {}
        settings = entry.get("settings") or {}
        if entry.get("enabled") is not True:
            return "profile local_gpu_admission gate is not enabled"
        registry_path = Path(str(settings.get("capability_registry_path") or "")).expanduser().resolve(strict=True)
        registry_bytes = registry_path.read_bytes()
        registry_hash = hashlib.sha256(registry_bytes).hexdigest()
        if registry_hash != str(settings.get("capability_registry_sha256") or "").lower():
            return "capability registry digest differs from profile pin"
        registry = json.loads(registry_bytes.decode("utf-8"))
        capabilities = [item for item in registry.get("capabilities", [])
                        if isinstance(item, dict) and item.get("id") == capability_id]
        if len(capabilities) != 1:
            return "capability ID is missing or duplicated in the pinned registry"
        capability = capabilities[0]
        if capability.get("allow") is not True:
            return "pinned capability is not authorized (allow is false)"
        route_error = _capability_route_error(
            capability, capability_id, profile_home, model, provider, model_cfg, provider_cfg, base_url,
        )
        if route_error:
            return route_error
        parsed = urlsplit(base_url)
        runtime_error = _verify_artifacts_and_model(
            row, capability, registry_path, provider, parsed, model, urlopen_fn,
        )
        if runtime_error:
            return runtime_error

        state_dir = Path(str(settings.get("resource_balancer_state_dir") or "")).expanduser().resolve(strict=True)
        conn = sqlite3.connect((state_dir / "queue.sqlite3").as_uri() + "?mode=ro", uri=True, timeout=1)
        conn.row_factory = sqlite3.Row
        try:
            state_row = conn.execute("SELECT data FROM controller WHERE id=1").fetchone()
            if state_row is None:
                return "Resource Balancer has no current controller status"
            state = json.loads(state_row["data"])
            active = conn.execute(
                "SELECT 1 FROM jobs WHERE class IN ('gpu','mixed') "
                "AND state IN ('starting','running') LIMIT 1",
            ).fetchone()
        finally:
            conn.close()
        if (state.get("mode") != "manage" or state.get("blocked") is not False
                or now() - float(state.get("updated") or 0) > 30
                or (state.get("resource_gates") or {}).get("gpu", {}).get("blocked") is not False):
            return "Resource Balancer is stale, globally blocked, or GPU-gated"
        if active is not None:
            return "another cooperating GPU reservation is already active"
        return None
    # health: allow BLE001 -- any evidence parsing/runtime failure must refuse before claim.
    except Exception as exc:
        if not capability_id:
            # Legacy/non-capability rows may use profiles in control-plane
            # test and remote-dispatch environments where no profile config
            # is available to this dispatcher. Their model middleware still
            # denies local provider I/O without an admitted task/run lease.
            return None
        detail = " ".join(str(exc).split())[:180]
        return f"local GPU dispatcher preflight failed closed: {detail or type(exc).__name__}"
