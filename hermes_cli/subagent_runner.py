#!/usr/bin/env python3
"""Subagent runner entrypoint.

This module is invoked by the delegate tool in a separate process.

It reads a JSON task envelope from a temp file, initializes Langfuse tracing
from local Hermes config (without passing trace context via environment
variables), runs a focused AIAgent, and prints a single JSON result to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from contextlib import nullcontext
from typing import Any, Dict, Optional


def _safe_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


def _read_task_envelope(task_file: str) -> Dict[str, Any]:
    with open(task_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    _safe_unlink(task_file)
    if not isinstance(data, dict):
        raise ValueError("Task envelope must be a JSON object")
    return data


def _init_langfuse_from_local_config() -> Dict[str, Any]:
    """Initialize Langfuse client from Hermes config.

    Returns a dict {enabled, client}.
    """
    try:
        from hermes_cli.config import get_langfuse_config
        cfg = get_langfuse_config()
    except Exception:
        cfg = {"enabled": False}

    enabled = bool(cfg.get("enabled"))
    public_key = str(cfg.get("public_key") or "").strip()
    secret_key = str(cfg.get("secret_key") or "").strip()
    host = str(cfg.get("host") or "").strip() or None
    sample_rate = cfg.get("sample_rate", 1.0)
    try:
        sample_rate = float(sample_rate)
    except Exception:
        sample_rate = 1.0

    # Best-effort: if Langfuse SDK is not installed, just disable.
    try:
        from langfuse import Langfuse
    except Exception:
        return {"enabled": False, "client": None}

    if not enabled or not (public_key and secret_key):
        return {"enabled": False, "client": None}

    kwargs: Dict[str, Any] = {
        "public_key": public_key,
        "secret_key": secret_key,
        "sample_rate": sample_rate,
    }
    if host:
        # Langfuse Python SDK uses host/base_url naming across versions.
        kwargs["host"] = host
        kwargs["base_url"] = host

    try:
        client = Langfuse(**kwargs)
    except TypeError:
        # Older SDKs may not support sample_rate/host kwarg names.
        minimal: Dict[str, Any] = {
            "public_key": public_key,
            "secret_key": secret_key,
        }
        if host:
            minimal["host"] = host
            minimal["base_url"] = host
        client = Langfuse(**minimal)

    # Wire Hermes' singleton (used by model_tools tool spans) to this client.
    try:
        from agent.observability import configure_langfuse_client
        configure_langfuse_client(client)
    except Exception:
        pass

    return {"enabled": True, "client": client}


def _start_subagent_observation(
    lf_client,
    *,
    trace_id: Optional[str],
    parent_observation_id: Optional[str],
    name: str,
    input_obj: Any,
):
    if lf_client is None or not trace_id:
        return nullcontext(None)

    # Prefer v4 API: start_as_current_observation + trace_context.
    if hasattr(lf_client, "start_as_current_observation"):
        trace_context = None
        if parent_observation_id:
            trace_context = {"trace_id": trace_id, "parent_span_id": parent_observation_id}
        else:
            trace_context = {"trace_id": trace_id}
        try:
            return lf_client.start_as_current_observation(
                as_type="span",
                name=name,
                input=input_obj,
                trace_context=trace_context,
            )
        except TypeError:
            # Older signature fallback.
            return lf_client.start_as_current_observation(
                as_type="span",
                name=name,
                trace_context=trace_context,
            )

    # v3-ish API compatibility.
    if hasattr(lf_client, "start_as_current_span"):
        return lf_client.start_as_current_span(
            name=name,
            trace_id=trace_id,
            parent_observation_id=parent_observation_id,
            metadata={"input": input_obj} if input_obj is not None else None,
        )

    return nullcontext(None)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Hermes subagent runner")
    parser.add_argument("--task-file", required=True, help="Path to JSON task envelope")
    args = parser.parse_args(argv)

    envelope = _read_task_envelope(args.task_file)

    goal = str(envelope.get("goal") or "").strip()
    context = envelope.get("context")
    toolsets = envelope.get("toolsets")
    max_iterations = envelope.get("max_iterations")
    task_id = str((envelope.get("trace_context") or {}).get("task_id") or uuid.uuid4())

    agent_cfg = envelope.get("agent_config") if isinstance(envelope.get("agent_config"), dict) else {}
    model = agent_cfg.get("model")
    provider = agent_cfg.get("provider")
    base_url = agent_cfg.get("base_url")
    api_mode = agent_cfg.get("api_mode")
    platform = agent_cfg.get("platform")

    trace_ctx = envelope.get("trace_context") if isinstance(envelope.get("trace_context"), dict) else {}
    parent_trace_id = trace_ctx.get("parent_trace_id")
    parent_observation_id = trace_ctx.get("parent_observation_id")
    trace_name = str(trace_ctx.get("trace_name") or "subagent").strip() or "subagent"
    session_id = trace_ctx.get("session_id")
    user_id = trace_ctx.get("user_id")

    lf_info = _init_langfuse_from_local_config()
    lf_enabled = bool(lf_info.get("enabled"))
    lf_client = lf_info.get("client")

    # Attach trace attributes (incl traceName) to all nested observations.
    propagate_ctx = nullcontext()
    if lf_enabled and lf_client is not None:
        try:
            from langfuse import propagate_attributes
            propagate_ctx = propagate_attributes(
                session_id=session_id,
                user_id=user_id,
                tags=["subagent"],
                metadata={
                    "task_id": task_id,
                    "parent_trace_id": parent_trace_id,
                    "parent_observation_id": parent_observation_id,
                },
                trace_name=trace_name,
            )
        except Exception:
            propagate_ctx = nullcontext()

    # Start a subagent root span under the parent's delegate span.
    obs_ctx = _start_subagent_observation(
        lf_client,
        trace_id=parent_trace_id if isinstance(parent_trace_id, str) else None,
        parent_observation_id=parent_observation_id if isinstance(parent_observation_id, str) else None,
        name=trace_name,
        input_obj={"goal": goal, "context": context},
    )

    try:
        from run_agent import AIAgent
    except Exception as exc:
        print(json.dumps({"error": f"Failed to import AIAgent: {exc}"}, ensure_ascii=False))
        return 2

    result: Dict[str, Any]
    subagent_obs_id = None
    current_trace_id = None

    try:
        with propagate_ctx:
            with obs_ctx as obs:
                subagent_obs_id = getattr(obs, "id", None) if obs is not None else None
                current_trace_id = getattr(obs, "trace_id", None) if obs is not None else None

                child = AIAgent(
                    model=model,
                    provider=provider,
                    base_url=base_url,
                    api_mode=api_mode,
                    max_iterations=int(max_iterations) if isinstance(max_iterations, int) else 50,
                    enabled_toolsets=toolsets if isinstance(toolsets, list) else None,
                    quiet_mode=True,
                    ephemeral_system_prompt=None,
                    platform=platform,
                    skip_context_files=True,
                    skip_memory=True,
                    langfuse_enabled=lf_enabled,
                    langfuse_sampling=True,
                    parent_trace_id=parent_trace_id if isinstance(parent_trace_id, str) else None,
                    parent_observation_id=subagent_obs_id if isinstance(subagent_obs_id, str) else None,
                )
                result = child.run_conversation(user_message=goal, task_id=task_id)

                # Update the subagent span output if possible.
                try:
                    if obs is not None and hasattr(obs, "update"):
                        obs.update(output={"status": result.get("completed"), "final": result.get("final_response")})
                except Exception:
                    pass

    except Exception as exc:
        result = {"completed": False, "error": str(exc), "final_response": None, "api_calls": 0}

    # Attach trace IDs for the parent to link/debug.
    if isinstance(result, dict):
        result["subagent_observation_id"] = subagent_obs_id
        result["subagent_trace_id"] = current_trace_id or parent_trace_id

    # Flush in short-lived process.
    if lf_client is not None:
        for meth in ("flush", "shutdown"):
            fn = getattr(lf_client, meth, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass

    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
