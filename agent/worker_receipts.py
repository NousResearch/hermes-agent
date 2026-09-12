"""Allowlisted worker route evidence at request dispatch and response intake.

No request body, prompt, headers, endpoint, or credentials enter these records.
The SDK dispatch observation is distinct from the provider's reported model.
"""

from __future__ import annotations


def _field(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _identifier(value):
    return value if isinstance(value, str) and 0 < len(value) <= 512 else None


def observe_worker_request(agent, api_kwargs):
    receipt = getattr(agent, "_worker_route_receipt", None)
    if not isinstance(receipt, dict):
        return
    # SDK extra_body overrides ordinary request fields when building its JSON.
    payload = dict(api_kwargs)
    extra = payload.pop("extra_body", None)
    if isinstance(extra, dict):
        payload.update(extra)
    reasoning = payload.get("reasoning")
    output_config = payload.get("output_config")
    effort = (_field(reasoning, "effort") or payload.get("reasoning_effort")
              or _field(output_config, "effort"))
    thinking = payload.get("thinking")
    budget = _field(thinking, "budget_tokens")
    observed = {
        "transmitted_provider": _identifier(getattr(agent, "provider", None)),
        "transmitted_model": _identifier(payload.get("model") or payload.get("modelId")),
        "transmitted_reasoning_effort": _identifier(effort),
        "transmitted_thinking_type": _identifier(_field(thinking, "type")),
        "transmitted_thinking_budget": budget if isinstance(budget, int) and not isinstance(budget, bool) else None,
        "provider_reported_model": None,
        "request_evidence_source": "post_middleware_sdk_dispatch",
        "response_evidence_source": None,
    }
    receipt.update(observed)
    # Keep route changes and missing/observed responses distinct without one
    # receipt row per token or streaming event.
    attempts = receipt.setdefault("execution_attempts", [])
    attempts.append(dict(observed))


def observe_worker_response(agent, response):
    receipt = getattr(agent, "_worker_route_receipt", None)
    if not isinstance(receipt, dict):
        return
    marker = _field(response, "_provider_reported_model")
    mode = getattr(agent, "api_mode", None)
    if mode in {"codex_responses", "bedrock_converse", "codex_app_server"}:
        # These adapters can synthesize response.model from the requested ID.
        model = _identifier(marker)
        source = "provider_response_marker" if model else None
    else:
        from agent.chat_completion_helpers import PARTIAL_STREAM_STUB_ID
        model = None if _field(response, "id") == PARTIAL_STREAM_STUB_ID else _identifier(_field(response, "model"))
        source = "provider_response.model" if model else None
    receipt["provider_reported_model"] = model
    receipt["response_evidence_source"] = source
    attempts = receipt.get("execution_attempts")
    if attempts:
        attempts[-1].update(provider_reported_model=model, response_evidence_source=source)
