"""Compression-route ``local_override`` scoping + fast-lane config fields.

``auxiliary.compression.local_override`` re-routes compaction to its own
provider/model/base_url only while the live main runtime is a listed local
endpoint: main provider ``custom``/``ollama``/``lmstudio``/``vllm`` (or
``custom:*``) AND its base_url is an exact member of ``base_urls`` (trailing
slash ignored; list each spelling you use, e.g. ``.../v1`` and bare
``host:port``). Every other main model keeps the base compression route.
Resolution reads the live runtime per call, so ``/model`` switches and
concurrent sessions route correctly. Empty ``local_override`` = disabled.
"""

from typing import Any, Optional


def scope_compression_override(
    task: str, task_config: dict[str, Any], runtime: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Apply auxiliary.compression.local_override only on a listed local endpoint."""
    if task != "compression":
        return task_config
    local_override = task_config.get("local_override")
    if not isinstance(local_override, dict):
        return task_config
    main_provider = str((runtime or {}).get("provider") or "").lower()
    main_url = str((runtime or {}).get("base_url") or "").rstrip("/")
    local_urls = local_override.get("base_urls", [])
    if (
        main_provider in {"custom", "ollama", "lmstudio", "vllm"}
        or main_provider.startswith("custom:")
    ) and main_url and isinstance(local_urls, list) and main_url in {
        str(url).rstrip("/") for url in local_urls if isinstance(url, str)
    }:
        scoped = dict(task_config)
        for field in ("provider", "model", "base_url", "api_key", "api_mode",
                      "key_env", "api_key_env"):
            if field in local_override:
                scoped[field] = local_override[field]
        return scoped
    return task_config


def _fast_lane_config_fields(config: dict[str, Any]) -> tuple[str, str, bool]:
    """Only explicit reasoning disablement certifies a non-reasoning route."""
    from hermes_constants import parse_reasoning_effort
    provider = str(config.get("provider") or "").strip().lower()
    model = str(config.get("model") or "").strip()
    parsed_effort = parse_reasoning_effort(config.get("reasoning_effort"))
    non_reasoning = parsed_effort is not None and parsed_effort.get("enabled") is False
    return provider, model, non_reasoning
