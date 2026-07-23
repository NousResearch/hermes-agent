"""Exact configured-route diagnostics for :mod:`hermes_cli.doctor`.

The regular doctor connectivity table proves that a credential can reach a
provider-level endpoint (usually ``/models``).  It does not prove that the
configured provider/model/API-mode tuple can execute inference.  This module
keeps the structural validation and opt-in live probes isolated from the
already-large doctor command.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from agent.auxiliary_client import (
    _build_call_kwargs,
    _get_task_extra_body,
    resolve_provider_client,
)
from agent.redact import redact_sensitive_text
from hermes_cli.fallback_config import get_fallback_chain


@dataclass(frozen=True)
class ConfigIssue:
    path: str
    message: str
    fix: str


@dataclass(frozen=True)
class RouteSpec:
    label: str
    provider: str
    model: str
    base_url: str = ""
    api_key: str = field(default="", repr=False, compare=False)
    api_mode: str = ""
    task: str = ""


@dataclass(frozen=True)
class ProbeResult:
    route: RouteSpec
    ok: bool
    detail: str = ""
    skipped: bool = False


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_fallback_entries(
    raw: Any,
    *,
    path: str,
    require_list: bool,
) -> list[ConfigIssue]:
    issues: list[ConfigIssue] = []
    if raw is None or raw == {} or raw == []:
        return issues
    if require_list and not isinstance(raw, list):
        return [
            ConfigIssue(
                path,
                f"{path} must be a YAML list of provider/model mappings; got {type(raw).__name__}",
                f"Rewrite {path} as a YAML list, for example: "
                "[{provider: openrouter, model: openai/gpt-4.1-mini}]",
            )
        ]
    if not isinstance(raw, (dict, list)):
        return [
            ConfigIssue(
                path,
                f"{path} must be a provider/model mapping or list of mappings; got {type(raw).__name__}",
                f"Remove the malformed {path} value or replace it with valid YAML mappings",
            )
        ]

    entries = raw if isinstance(raw, list) else [raw]
    for index, entry in enumerate(entries):
        entry_path = f"{path}[{index}]"
        if not isinstance(entry, dict):
            issues.append(
                ConfigIssue(
                    entry_path,
                    "fallback entry must be a YAML mapping",
                    f"Set {entry_path} to a mapping with string provider and model fields",
                )
            )
            continue
        provider = entry.get("provider")
        model = entry.get("model")
        if not _nonempty_string(provider):
            issues.append(
                ConfigIssue(
                    f"{entry_path}.provider",
                    "fallback provider must be a non-empty string",
                    f"Set {entry_path}.provider to a valid provider ID",
                )
            )
        if not _nonempty_string(model):
            issues.append(
                ConfigIssue(
                    f"{entry_path}.model",
                    "fallback model must be a non-empty string",
                    f"Set {entry_path}.model to a model available on that provider",
                )
            )
    return issues


def validate_route_config(raw_config: dict[str, Any] | None) -> list[ConfigIssue]:
    """Return route/config defects that runtime loaders would otherwise ignore.

    This deliberately validates only routing fields with clear runtime
    contracts.  It is not a speculative whole-config schema and therefore does
    not reject plugin-owned or future keys.
    """

    config = raw_config if isinstance(raw_config, dict) else {}
    issues: list[ConfigIssue] = []

    model = config.get("model")
    if isinstance(model, dict):
        for key in ("fallback_providers", "fallback_model"):
            if key in model:
                issues.append(
                    ConfigIssue(
                        f"model.{key}",
                        f"model.{key} is ignored by the runtime because fallback chains are top-level",
                        f"Move model.{key} to top-level {key}",
                    )
                )

    issues.extend(
        _validate_fallback_entries(
            config.get("fallback_providers"),
            path="fallback_providers",
            require_list=True,
        )
    )
    issues.extend(
        _validate_fallback_entries(
            config.get("fallback_model"),
            path="fallback_model",
            require_list=False,
        )
    )

    auxiliary = config.get("auxiliary")
    if auxiliary is not None and not isinstance(auxiliary, dict):
        issues.append(
            ConfigIssue(
                "auxiliary",
                "auxiliary must be a YAML mapping of task names to route settings",
                "Replace auxiliary with a YAML mapping or remove the malformed value",
            )
        )
    elif isinstance(auxiliary, dict):
        for task, task_config in auxiliary.items():
            if not isinstance(task_config, dict):
                issues.append(
                    ConfigIssue(
                        f"auxiliary.{task}",
                        "auxiliary task configuration must be a YAML mapping",
                        f"Replace auxiliary.{task} with a mapping or remove it",
                    )
                )

    delegation = config.get("delegation")
    if delegation is not None and not isinstance(delegation, dict):
        issues.append(
            ConfigIssue(
                "delegation",
                "delegation must be a YAML mapping",
                "Replace delegation with a YAML mapping or remove the malformed value",
            )
        )
    elif isinstance(delegation, dict):
        limit_rules = {
            "child_timeout_seconds": 0,
            "max_spawn_depth": 1,
            "max_concurrent_children": 1,
        }
        for key, minimum in limit_rules.items():
            if key not in delegation:
                continue
            value = delegation.get(key)
            # bool is an int subclass but is never a useful numeric limit here.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                issues.append(
                    ConfigIssue(
                        f"delegation.{key}",
                        f"delegation.{key} must be numeric, not {type(value).__name__}",
                        f"Set delegation.{key} to a number greater than or equal to {minimum}",
                    )
                )
            elif value < minimum:
                issues.append(
                    ConfigIssue(
                        f"delegation.{key}",
                        f"delegation.{key} must be greater than or equal to {minimum}",
                        f"Set delegation.{key} to {minimum} or higher",
                    )
                )

    return issues


def _route_api_key(route_config: dict[str, Any]) -> str:
    inline = route_config.get("api_key")
    if _nonempty_string(inline):
        return inline.strip()
    key_env = route_config.get("key_env") or route_config.get("api_key_env")
    if _nonempty_string(key_env):
        return os.getenv(key_env.strip(), "").strip()
    return ""


def _route_from_mapping(
    label: str,
    mapping: dict[str, Any],
    *,
    inherited_provider: str = "auto",
    inherited_model: str = "",
    task: str = "",
) -> RouteSpec | None:
    provider = mapping.get("provider")
    model = mapping.get("model")
    if not _nonempty_string(model) and label == "model.primary":
        model = mapping.get("default")
    explicit_model = _nonempty_string(model)
    provider_text = (
        str(provider).strip() if _nonempty_string(provider) else inherited_provider
    )
    model_text = str(model).strip() if explicit_model else inherited_model
    base_url = mapping.get("base_url")
    api_mode = mapping.get("api_mode")
    base_url_text = str(base_url).strip() if _nonempty_string(base_url) else ""
    api_mode_text = str(api_mode).strip() if _nonempty_string(api_mode) else ""

    # Inherited/auto auxiliary defaults do not create a distinct route. Their
    # actual backend is already covered by the primary route.
    if (
        task
        and provider_text in {"", "auto"}
        and not any((
            explicit_model,
            base_url_text,
            api_mode_text,
            _route_api_key(mapping),
        ))
    ):
        return None
    if not model_text:
        return None
    return RouteSpec(
        label=label,
        provider=provider_text or "auto",
        model=model_text,
        base_url=base_url_text,
        api_key=_route_api_key(mapping),
        api_mode=api_mode_text,
        task=task,
    )


def collect_configured_routes(raw_config: dict[str, Any] | None) -> list[RouteSpec]:
    """Enumerate every explicit inference route selected by the configuration."""

    config = raw_config if isinstance(raw_config, dict) else {}
    routes: list[RouteSpec] = []
    model = config.get("model")
    if isinstance(model, str):
        model = {"default": model}
    if not isinstance(model, dict):
        model = {}

    primary_provider = (
        model.get("provider").strip()
        if _nonempty_string(model.get("provider"))
        else "auto"
    )
    primary_model = ""
    for key in ("default", "model"):
        if _nonempty_string(model.get(key)):
            primary_model = model[key].strip()
            break
    primary = _route_from_mapping(
        "model.primary",
        model,
        inherited_provider=primary_provider,
        inherited_model=primary_model,
    )
    if primary is not None:
        routes.append(primary)

    for index, entry in enumerate(get_fallback_chain(config)):
        route = _route_from_mapping(f"fallback[{index}]", entry)
        if route is not None:
            routes.append(route)

    delegation = config.get("delegation")
    if isinstance(delegation, dict) and any(
        _nonempty_string(delegation.get(key))
        for key in ("provider", "model", "base_url", "api_mode", "api_key")
    ):
        route = _route_from_mapping(
            "delegation",
            delegation,
            inherited_provider=primary_provider,
            inherited_model=primary_model,
        )
        if route is not None:
            routes.append(route)

    auxiliary = config.get("auxiliary")
    if isinstance(auxiliary, dict):
        for task, task_config in sorted(auxiliary.items()):
            if not isinstance(task_config, dict):
                continue
            if not any(
                _nonempty_string(task_config.get(key))
                for key in ("provider", "model", "base_url", "api_mode", "api_key")
            ):
                continue
            route = _route_from_mapping(
                f"auxiliary.{task}",
                task_config,
                inherited_provider=primary_provider,
                inherited_model=primary_model,
                task=str(task),
            )
            if route is not None:
                routes.append(route)

    return routes


def _safe_error_text(exc: BaseException, *, api_key: str = "") -> str:
    text = redact_sensitive_text(str(exc), force=True)
    if api_key:
        text = text.replace(api_key, "[REDACTED]")
    # Avoid producing multi-page SDK payloads in the doctor summary.
    text = " ".join(text.split())
    return text[:500] + ("…" if len(text) > 500 else "")


def probe_route(route: RouteSpec, *, timeout: float = 15.0) -> ProbeResult:
    """Execute one minimal, exact inference call without provider fallback."""

    try:
        client, resolved_model = resolve_provider_client(
            route.provider,
            model=route.model,
            explicit_base_url=route.base_url or None,
            explicit_api_key=route.api_key or None,
            api_mode=route.api_mode or None,
        )
        if client is None:
            return ProbeResult(route, False, "runtime could not resolve a client")
        if not hasattr(getattr(client, "chat", None), "completions"):
            return ProbeResult(
                route,
                False,
                "resolved transport does not expose a probeable chat-completions interface",
                skipped=True,
            )

        final_model = resolved_model or route.model
        client_base_url = str(getattr(client, "base_url", route.base_url) or "")
        kwargs = _build_call_kwargs(
            route.provider,
            final_model,
            [{"role": "user", "content": "Reply with exactly: OK"}],
            temperature=0,
            max_tokens=8,
            timeout=timeout,
            extra_body=_get_task_extra_body(route.task) if route.task else None,
            base_url=client_base_url or route.base_url,
        )
        response = client.chat.completions.create(**kwargs)
        choices = getattr(response, "choices", None)
        if not choices or not hasattr(choices[0], "message"):
            return ProbeResult(
                route, False, "provider returned an invalid response shape"
            )
        return ProbeResult(route, True, f"{route.provider}/{final_model}")
    except Exception as exc:  # noqa: BLE001 - doctor must report, never crash
        return ProbeResult(
            route,
            False,
            _safe_error_text(exc, api_key=route.api_key),
        )
