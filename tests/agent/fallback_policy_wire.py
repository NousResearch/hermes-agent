"""Exercise a surface's resolved constructor route through the real first request builder."""
from unittest.mock import patch

from run_agent import AIAgent


def first_wire_from_runtime(kwargs):
    runtime_keys = ("model", "provider", "api_mode", "api_key", "base_url",
                    "fallback_service_tier_override", "fallback_model", "request_overrides")
    options = {key: kwargs[key] for key in runtime_keys if key in kwargs}
    # Premium can arrive from either a session setting or provider request overrides.
    options["service_tier"] = "priority"
    overrides = dict(options.get("request_overrides") or {})
    overrides.update(service_tier="priority", speed="fast")
    overrides["extra_body"] = {**(overrides.get("extra_body") or {}),
                               "service_tier": "priority", "speed": "fast", "marker": "keep"}
    options["request_overrides"] = overrides
    options.update(quiet_mode=True, skip_context_files=True, skip_memory=True)
    with (patch("agent.process_bootstrap.OpenAI"),
          patch("model_tools.get_tool_definitions", return_value=[]),
          patch("model_tools.check_toolset_requirements", return_value={}),
          patch("agent.context_compressor.get_model_context_length", return_value=200_000),
          patch("agent.model_metadata.get_model_context_length", return_value=200_000)):
        agent = AIAgent(**options)
        try:
            return agent._build_api_kwargs([{"role": "user", "content": "first"}]), agent._primary_runtime
        finally:
            agent.shutdown_memory_provider()
            agent.close()
