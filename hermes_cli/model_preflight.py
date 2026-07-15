"""Read-only provider/model preflight bound to the active Hermes profile.

The preflight deliberately performs no network request and never emits
credential values.  It verifies that the selected profile, persisted config,
live gateway process, active source tree, and provider credential structure
refer to the same runtime before an operator attempts a model change.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

from hermes_constants import get_config_path, get_hermes_home


_OAUTH_PROVIDERS = {
    "minimax-oauth",
    "nous",
    "openai-codex",
    "qwen-oauth",
    "xai-oauth",
}


@dataclass(frozen=True)
class ModelPreflightResult:
    profile_home: str
    config_path: str
    configured_provider: str
    configured_model: str
    gateway_pid: int | None
    gateway_command: str
    gateway_interpreter: str
    gateway_source_root: str
    profile_match: bool
    source_match: bool
    auth_present: bool
    status: str
    blockers: tuple[str, ...]


def _model_config(config: Mapping[str, Any]) -> tuple[str, str]:
    model = config.get("model")
    if isinstance(model, Mapping):
        provider = str(model.get("provider") or "").strip()
        selected = str(
            model.get("default") or model.get("model") or model.get("name") or ""
        ).strip()
        return provider, selected
    if isinstance(model, str):
        return "", model.strip()
    return "", ""


def _profile_from_command(command: str) -> Optional[str]:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    for index, token in enumerate(tokens):
        if token in {"--profile", "-p"} and index + 1 < len(tokens):
            return tokens[index + 1].strip()
        if token.startswith("--profile="):
            return token.split("=", 1)[1].strip()
    return None


def _interpreter_from_command(command: str) -> str:
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    return tokens[0] if tokens else ""


def _redact_command(command: str) -> str:
    """Redact credential-shaped CLI arguments before rendering diagnostics."""
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    secret_flags = {
        "--api-key",
        "--password",
        "--refresh-token",
        "--secret",
        "--token",
    }
    redacted: list[str] = []
    hide_next = False
    for token in tokens:
        if hide_next:
            redacted.append("[REDACTED]")
            hide_next = False
            continue
        lowered = token.lower()
        if lowered in secret_flags:
            redacted.append(token)
            hide_next = True
            continue
        if any(lowered.startswith(flag + "=") for flag in secret_flags):
            redacted.append(token.split("=", 1)[0] + "=[REDACTED]")
            continue
        redacted.append(token)
    return shlex.join(redacted)


def _source_from_runtime(
    runtime: Mapping[str, Any] | None,
    current_source_root: Path,
) -> str:
    if not isinstance(runtime, Mapping):
        return ""
    argv = runtime.get("argv")
    if not isinstance(argv, list) or not argv:
        return ""
    first = Path(str(argv[0])).expanduser()
    try:
        resolved = first.resolve(strict=False)
    except OSError:
        resolved = first.absolute()
    if resolved.name == "main.py" and resolved.parent.name == "hermes_cli":
        return str(resolved.parent.parent)
    current_source = current_source_root.expanduser().resolve(strict=False)
    if resolved in {
        current_source / ".venv" / "bin" / "hermes",
        current_source / "venv" / "bin" / "hermes",
    }:
        return str(current_source)
    return ""


def _provider_auth_present(provider: str, auth: Mapping[str, Any]) -> bool:
    if provider not in _OAUTH_PROVIDERS:
        # API-key providers may resolve credentials from profile-scoped env or
        # config references.  This preflight does not inspect secret values.
        return True
    providers = auth.get("providers")
    pool = auth.get("credential_pool")
    provider_entry = isinstance(providers, Mapping) and bool(providers.get(provider))
    pool_entry = isinstance(pool, Mapping) and bool(pool.get(provider))
    return bool(provider_entry or pool_entry)


def evaluate_model_preflight(
    *,
    profile_home: Path,
    config: Mapping[str, Any],
    runtime: Mapping[str, Any] | None,
    process_command: str,
    auth: Mapping[str, Any],
    current_source_root: Path,
) -> ModelPreflightResult:
    """Evaluate already-collected, non-secret runtime facts."""

    profile_home = profile_home.expanduser().resolve(strict=False)
    config_path = profile_home / "config.yaml"
    provider, model = _model_config(config)
    expected_profile = profile_home.name
    actual_profile = _profile_from_command(process_command)
    gateway_source = _source_from_runtime(runtime, current_source_root)
    current_source = str(current_source_root.expanduser().resolve(strict=False))

    runtime_running = bool(
        isinstance(runtime, Mapping)
        and runtime.get("gateway_state") == "running"
        and runtime.get("pid")
    )
    raw_pid = runtime.get("pid") if isinstance(runtime, Mapping) else None
    gateway_pid = int(raw_pid) if runtime_running and raw_pid is not None else None

    profile_match = actual_profile == expected_profile
    source_match = bool(gateway_source and gateway_source == current_source)
    auth_present = _provider_auth_present(provider, auth)

    failures: list[str] = []
    inconclusive: list[str] = []
    if not provider:
        failures.append("configured_provider_missing")
    if not model:
        failures.append("configured_model_missing")
    if not runtime_running:
        inconclusive.append("gateway_not_running")
    else:
        if actual_profile is None:
            inconclusive.append("gateway_profile_unverified")
        elif not profile_match:
            failures.append("gateway_profile_mismatch")
        if not gateway_source:
            inconclusive.append("gateway_source_unverified")
        elif not source_match:
            failures.append("gateway_source_mismatch")
    if provider and not auth_present:
        failures.append("configured_provider_auth_missing")

    blockers = tuple(failures + inconclusive)
    if failures:
        status = "FAIL"
    elif inconclusive:
        status = "INCONCLUSIVE"
    else:
        status = "PASS"

    return ModelPreflightResult(
        profile_home=str(profile_home),
        config_path=str(config_path),
        configured_provider=provider,
        configured_model=model,
        gateway_pid=gateway_pid,
        gateway_command=_redact_command(process_command),
        gateway_interpreter=_interpreter_from_command(process_command),
        gateway_source_root=gateway_source,
        profile_match=profile_match,
        source_match=source_match,
        auth_present=auth_present,
        status=status,
        blockers=blockers,
    )


def collect_model_preflight() -> ModelPreflightResult:
    """Collect preflight facts from the active profile without persistent writes."""

    from gateway.status import (
        _read_process_cmdline,
        get_runtime_status_running_pid,
        read_runtime_status,
    )
    from hermes_cli.config import load_config

    home = get_hermes_home().expanduser().resolve(strict=False)
    config = load_config()
    persisted_runtime = read_runtime_status()
    verified_pid = get_runtime_status_running_pid(
        persisted_runtime,
        expected_home=home,
    )
    runtime: Mapping[str, Any] | None
    if verified_pid is None:
        runtime = None
        process_command = ""
    else:
        runtime = dict(persisted_runtime or {})
        runtime["pid"] = verified_pid
        process_command = _read_process_cmdline(verified_pid) or ""

    auth_path = home / "auth.json"
    auth: Mapping[str, Any] = {}
    try:
        parsed = json.loads(auth_path.read_text(encoding="utf-8"))
        if isinstance(parsed, Mapping):
            auth = parsed
    except (OSError, ValueError):
        auth = {}

    result = evaluate_model_preflight(
        profile_home=home,
        config=config if isinstance(config, Mapping) else {},
        runtime=runtime,
        process_command=process_command,
        auth=auth,
        current_source_root=Path(__file__).resolve().parents[1],
    )
    # Keep the canonical config helper authoritative in case a platform uses a
    # non-default config filename in future.
    return ModelPreflightResult(
        **{**asdict(result), "config_path": str(get_config_path())}
    )


def format_model_preflight(result: ModelPreflightResult, *, as_json: bool = False) -> str:
    payload = asdict(result)
    if as_json:
        # The full live command line is intentionally excluded even after
        # redaction: wrappers may carry arbitrary secret-shaped env assignments.
        payload.pop("gateway_command", None)
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)

    blockers = ",".join(result.blockers) if result.blockers else "none"
    return "\n".join(
        (
            f"MODEL_PREFLIGHT_RESULT: {result.status}",
            f"profile_home: {result.profile_home}",
            f"config: {result.config_path}",
            f"configured: {result.configured_provider}/{result.configured_model}",
            f"gateway_pid: {result.gateway_pid or 'none'}",
            f"gateway_interpreter: {result.gateway_interpreter or 'unknown'}",
            f"gateway_source: {result.gateway_source_root or 'unknown'}",
            f"profile_match: {str(result.profile_match).lower()}",
            f"source_match: {str(result.source_match).lower()}",
            f"auth_present: {str(result.auth_present).lower()}",
            f"blockers: {blockers}",
        )
    )
