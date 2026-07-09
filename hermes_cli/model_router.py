"""Skeleton for the agent-level model router (#61371).

The actual routing logic is intentionally NOT implemented here — this
module reads ``model_router`` from config.yaml, validates the schema,
and exposes a single ``resolve_router_config()`` / ``is_router_enabled()``
so the agent loop can probe whether the user opted into router-based
selection. The actual per-task classifier and switch-model integration
are tracked in #61371; until they land, callers receive a logger.info()
"router enabled but no dispatch yet" message and fall back to the
existing single-provider path.

The router schema mirrors the example in #61371:

    model_router:
      enabled: true
      local:
        provider: ollama
        model: llama3.2-3b
        for_tasks: [read_only, shell_commands]
      cloud:
        provider: nous
        model: anthropic/claude-sonnet-4.6
        for_tasks: [debugging, code_review, log_analysis]

Validation enforces the schema strictly so a malformed config fails at
load time rather than producing silent fallbacks later. The
for_tasks-list-per-bucket structure is not yet mapped to real prompt
classifiers — that decision lands in the follow-up PR.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping

logger = logging.getLogger(__name__)

# Known top-level task-type labels the router advertises as a vocabulary.
# Buckets that name labels outside this set are rejected at config-load
# time so users find out about typos immediately rather than silently
# losing the routing rule at dispatch.
KNOWN_TASK_LABELS: frozenset[str] = frozenset(
    {
        "read_only",
        "shell_commands",
        "debugging",
        "code_review",
        "log_analysis",
        # Future buckets land here as the per-task classifier lands.
    }
)


class ModelRouterConfigError(ValueError):
    """Raised when the ``model_router`` block in config.yaml is malformed.

    Mirrors the fail-fast parse-error contract the rest of config.py
    uses — never let a malformed router config pass through to dispatch.
    """


def is_router_enabled(config: Mapping[str, Any] | None) -> bool:
    """Return True iff the user opted into the model router.

    Defensive against missing/None/empty config dicts. Default is False
    so this skeleton is a no-op for everyone who hasn't added the
    ``model_router`` block to their config.yaml yet — preserving the
    existing single-provider behavior (#61371 acceptance criterion).
    """
    if not config:
        return False
    block = config.get("model_router")
    if not isinstance(block, Mapping):
        return False
    return bool(block.get("enabled", False))


def _validate_bucket(name: str, bucket: Any) -> dict[str, Any]:
    """Validate one bucket (e.g. ``local`` / ``cloud``) of the router config.

    Returns a normalized dict of (provider, model, for_tasks) so the
    downstream dispatcher and tests can rely on a stable shape.
    """
    if not isinstance(bucket, Mapping):
        raise ModelRouterConfigError(
            f"model_router.{name} must be a mapping with provider/model/"
            "for_tasks; got {type(bucket).__name__}"
        )

    for required in ("provider", "model"):
        value = bucket.get(required)
        if not isinstance(value, str) or not value.strip():
            raise ModelRouterConfigError(
                f"model_router.{name}.{required} must be a non-empty string"
            )

    for_tasks = bucket.get("for_tasks", [])
    if isinstance(for_tasks, str):
        # Allow the comma-joined shorthand the issue body uses — split it.
        for_tasks = [t.strip() for t in for_tasks.split(",") if t.strip()]
    if not isinstance(for_tasks, list) or not all(
        isinstance(t, str) and t.strip() for t in for_tasks
    ):
        raise ModelRouterConfigError(
            f"model_router.{name}.for_tasks must be a list of non-empty "
            "strings (or a comma-joined string)"
        )

    unknown = sorted(set(for_tasks) - KNOWN_TASK_LABELS)
    if unknown:
        raise ModelRouterConfigError(
            f"model_router.{name}.for_tasks contains labels that aren't in "
            f"the known vocabulary yet: {unknown}. Add them to "
            "KNOWN_TASK_LABELS in model_router.py once the per-task "
            "classifier lands (#61371)."
        )

    return {
        "provider": bucket["provider"].strip(),
        "model": bucket["model"].strip(),
        "for_tasks": list(for_tasks),
    }


def resolve_router_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Parse + validate the model's router block.

    Returns a normalized dict, or the disabled-shape ``{"enabled": False}``
    when the user hasn't opted in. Raises :class:`ModelRouterConfigError`
    on malformed schema.

    Calling this in the agent loop is a no-op for routing *today* — it
    only validates + logs. Real dispatch lands in the follow-up PR.
    """
    if not is_router_enabled(config):
        return {"enabled": False, "buckets": {}, "default_task_label": None}

    block = config["model_router"]  # type: ignore[index]  # already validated
    buckets: dict[str, dict[str, Any]] = {}
    for name in ("local", "cloud"):
        if name in block:
            buckets[name] = _validate_bucket(name, block[name])

    if not buckets:
        raise ModelRouterConfigError(
            "model_router.enabled is true but neither a 'local' nor a "
            "'cloud' bucket is defined — at least one routing target "
            "must be configured (#61371)"
        )

    logger.info(
        "model_router: enabled with buckets=%s — routing dispatch is "
        "not implemented yet (skeleton PR for #61371). Falling back to "
        "the legacy single-provider path until the per-task classifier "
        "ships in the follow-up.",
        sorted(buckets.keys()),
    )

    return {
        "enabled": True,
        "buckets": buckets,
        "default_task_label": None,  # set by classifier; tracked in #61371
    }
