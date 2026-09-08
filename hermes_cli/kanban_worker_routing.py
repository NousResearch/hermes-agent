"""Restore opt-in worker routing across the profile process boundary."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Optional
from urllib.parse import urlparse

if TYPE_CHECKING:
    from hermes_cli.kanban_db import Task

_LOCAL_KANBAN_PROVIDERS = frozenset({"ollama", "ollama-launch", "local"})


def prepare_worker_route(task, profile_home, env):
    import yaml

    env.pop("HERMES_KANBAN_LOCAL_ONLY", None)
    route = _resolve_explicit_local_task_route(task)
    if route is None and not task.model_override and _kanban_local_first_enabled():
        path = Path(profile_home) / "config.yaml" if profile_home else None
        config = yaml.safe_load(path.read_text(encoding="utf-8")) if path and path.is_file() else None
        route = _resolve_local_first_route(config)
    if route is None:
        return task
    provider, model = route
    endpoint = _resolve_process_local_provider_endpoint(provider)
    if endpoint:
        env["CUSTOM_BASE_URL"] = endpoint
        provider = "ollama"
    env["HERMES_KANBAN_LOCAL_ONLY"] = "1"
    return replace(task, provider_override=provider, model_override=model,
                   reasoning_effort=task.reasoning_effort or "none")


def _local_route_candidates(raw: Any) -> list[Mapping[str, Any]]:
    """Normalize a profile's configured model-route list for local selection."""
    if isinstance(raw, Mapping):
        return [raw]
    if isinstance(raw, (list, tuple)):
        return [item for item in raw if isinstance(item, Mapping)]
    return []


def _resolve_local_first_route(
    profile_config: Optional[Mapping[str, Any]],
) -> Optional[tuple[str, str]]:
    """Return the first usable local route from a profile configuration.

    Kanban workers receive task-owned paths, attachments, and tool results. A
    remote primary therefore often cannot be authorized by the egress policy;
    waiting for that rejection and then falling back wastes a full model turn.
    When the operator enables ``kanban.local_first``, choose the profile's
    already-configured local route before spawning. This function is pure so
    route precedence is testable without starting a worker or contacting a
    provider. Unknown providers are deliberately ignored.
    """
    if not isinstance(profile_config, Mapping):
        return None

    candidates: list[Mapping[str, Any]] = []
    primary = profile_config.get("model")
    candidates.extend(_local_route_candidates(primary))
    # ``fallback_model`` is the effective profile fallback schema in current
    # installs. ``fallback_providers`` remains supported for older profiles.
    candidates.extend(_local_route_candidates(profile_config.get("fallback_model")))
    candidates.extend(
        _local_route_candidates(profile_config.get("fallback_providers"))
    )
    for entry in candidates:
        provider = str(entry.get("provider") or "").strip().lower()
        model = str(entry.get("model") or entry.get("default") or "").strip()
        if provider in _LOCAL_KANBAN_PROVIDERS and model:
            return provider, model

    providers = profile_config.get("providers")
    if isinstance(providers, Mapping):
        for provider, raw_provider in providers.items():
            provider_name = str(provider or "").strip().lower()
            if provider_name not in _LOCAL_KANBAN_PROVIDERS:
                continue
            if not isinstance(raw_provider, Mapping):
                continue
            model = str(raw_provider.get("default_model") or "").strip()
            if model:
                return provider_name, model
    return None


def _kanban_local_first_enabled() -> bool:
    """Read the explicit operator opt-in for local-first Kanban spawning."""
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
    except Exception:
        return False
    kanban = cfg.get("kanban") if isinstance(cfg, Mapping) else None
    return bool(isinstance(kanban, Mapping) and kanban.get("local_first"))


def _resolve_explicit_local_task_route(
    task: Task,
) -> Optional[tuple[str, str]]:
    """Return an explicitly pinned local task route, if one was supplied.

    ``kanban.local_first`` is a policy for avoiding doomed remote attempts;
    it must not erase a deliberate per-card local model selection.  The
    provider is required here so a model name cannot be guessed to be local
    from its spelling alone.
    """
    provider = str(task.provider_override or "").strip().lower()
    model = str(task.model_override or "").strip()
    if provider in _LOCAL_KANBAN_PROVIDERS and model:
        return provider, model
    return None


def _resolve_process_local_provider_endpoint(provider: str) -> Optional[str]:
    """Return a configured loopback endpoint for a local provider name.

    Kanban workers run with ``HERMES_HOME`` set to the assignee profile so
    profile-local state and credentials stay isolated.  Provider definitions,
    however, are commonly owned by the process-level Hermes config.  A local
    task pin must not become an unknown provider in that profile and then fall
    through to a remote fallback.  Resolve only the named provider's endpoint
    from the process config, and only when it is loopback; never copy or
    expose a remote provider definition across the profile boundary.
    """
    provider_name = str(provider or "").strip().lower()
    if provider_name not in _LOCAL_KANBAN_PROVIDERS:
        return None
    try:
        from hermes_constants import get_process_hermes_home
        from hermes_cli.config import read_user_config_raw

        config = read_user_config_raw(
            get_process_hermes_home() / "config.yaml"
        )
        providers = config.get("providers")
        entry = providers.get(provider_name) if isinstance(providers, Mapping) else None
        if not isinstance(entry, Mapping):
            return None
        endpoint = str(
            entry.get("api") or entry.get("url") or entry.get("base_url") or ""
        ).strip()
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"}:
            return None
        if (parsed.hostname or "").lower().rstrip(".") not in {
            "localhost", "127.0.0.1", "::1"
        }:
            return None
        return endpoint.rstrip("/") or None
    except (OSError, TypeError, ValueError):
        return None


def recover_generated_assignee(conn, row, default_assignee, *, dry_run, result):
    from hermes_cli import kanban_db as kb
    from hermes_cli.profiles import profile_exists
    assignee = row["assignee"]
    if not assignee or profile_exists(assignee) or not default_assignee:
        return assignee
    if (row["created_by"] or "").strip().lower() not in {
        "auto-decomposer", "decomposer", "specialist-routing",
    } or not profile_exists(default_assignee):
        return assignee
    if not dry_run:
        with kb.write_txn(conn):
            changed = conn.execute(
                "UPDATE tasks SET assignee = ?, consecutive_failures = 0, last_failure_error = NULL "
                "WHERE id = ? AND assignee = ? AND status = 'ready' AND claim_lock IS NULL",
                (default_assignee, row["id"], assignee))
            if changed.rowcount != 1:
                return assignee
            kb._append_event(conn, row["id"], "assigned", {
                "assignee": default_assignee, "previous_assignee": assignee,
                "source": "kanban.invalid_assignee_fallback",
            })
    result.auto_reassigned_invalid.append(row["id"])
    return default_assignee
