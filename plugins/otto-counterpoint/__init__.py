"""Hermes plugin entrypoint for the bounded Otto counterpoint controller."""
from __future__ import annotations

from typing import Any

from .controller import CounterpointController


def _config(ctx: Any) -> dict[str, Any]:
    keys = (
        "mode",
        "project_id",
        "generator_provider",
        "generator_vendor",
        "generator_family",
        "generator_reasoning_effort",
        "counterpoint_route",
        "adjudicator_route",
        "criteria",
        "sample_percent",
        "canary_percent",
        "max_corrections",
        "timeout_seconds",
        "run_budget_seconds",
        "supervisor_approved",
    )
    result: dict[str, Any] = {}
    for key in keys:
        value = ctx.get_config(key, None)
        if value is not None:
            result[key] = value
    return result


def register(ctx: Any) -> None:
    controller = CounterpointController(config=_config(ctx), state=ctx.state)
    ctx.register_hook("pre_llm_call", controller.on_pre_llm_call)
    ctx.register_hook("pre_tool_call", controller.on_pre_tool_call)
    ctx.register_hook("transform_llm_output", controller.on_transform_llm_output)
