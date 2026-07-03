"""Governed OpenAI Agents SDK bridge tools for Hermes.

Hermes/GPT-5.5 remains the orchestrator. These handlers expose bounded
OpenAI Agents SDK worker lanes with structured proof outputs so external SDK
workers cannot satisfy Scott's governance standard with unsupported prose.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from hermes_cli.config import get_env_value, load_config
from hermes_constants import get_hermes_home
from pydantic import BaseModel, Field, ValidationError
from tools.registry import tool_error, tool_result

DEFAULT_MODEL = "gpt-5.5"

_MAX_TASK_CHARS = 12000
_MAX_CONTEXT_CHARS = 20000
_MAX_LIST_ITEMS = 25
_MAX_ITEM_CHARS = 1000
_DEFAULT_MAX_TURNS = 4
_DEFAULT_MAX_TOKENS = 1600
_DEFAULT_OUTPUT_CHARS = 12000
_ARCHITECTURE_MIN_TOKENS = 1400

# Scott's trust policy: never route SDK workers through Chinese-origin models or
# provider/model aliases, even if a caller passes one explicitly.
_BLOCKED_MODEL_FRAGMENTS = {
    "alibaba",
    "baichuan",
    "deepseek",
    "doubao",
    "kimi",
    "minimax",
    "moonshot",
    "qwen",
    "stepfun",
    "tencent",
    "volcengine",
    "wenxin",
    "yi-",
    "zhipu",
}

_GOVERNANCE_CONTRACT = [
    "No claim without proof.",
    "No mutation without explicit scope.",
    "No success without verification evidence.",
    "No learning or preference update without provenance and approval.",
    "No autonomy outside bounded max_turns/max_tokens/model allowlist.",
]

_HIGH_RISK_PATTERN = re.compile(
    r"\b(delete|remove|wipe|reset|install|uninstall|restart|stop\s+service|"
    r"credential|secret|token|firewall|registry|admin|sudo|chmod|chown|rm\s+-rf)\b",
    re.IGNORECASE,
)
_APPROVAL_SCOPE_PATTERN = re.compile(
    r"\b(approved|authorized|explicit\s+scope|read-only|no\s+mutation|no\s+side\s+effects|analysis\s+only)\b",
    re.IGNORECASE,
)
_SECRET_KEY_PATTERN = re.compile(
    r"^(api[_-]?key|secret|password|credential|access[_-]?token|refresh[_-]?token|auth[_-]?token|bearer[_-]?token)$",
    re.IGNORECASE,
)

_LANE_CONFIG: dict[str, dict[str, str]] = {
    "review": {
        "tool": "openai_agents_review",
        "name": "HermesOpenAIAgentsReviewWorker",
        "description": "Advisory OpenAI Agents SDK worker for skeptical plan/code/claim review. No mutation authority.",
        "instructions": """You are an OpenAI Agents SDK REVIEW worker under Hermes/GPT-5.5 supervision.
Your job is skeptical review only. Do not execute or claim mutation. Evaluate the task against the provided acceptance criteria, constraints, and governance contract.
Return a structured proof object. Use status='verified' only when you can cite concrete evidence from the prompt/context. Otherwise use 'partial' or 'blocked'.""",
    },
    "execute": {
        "tool": "openai_agents_execute",
        "name": "HermesOpenAIAgentsExecuteWorker",
        "description": "Bounded OpenAI Agents SDK worker for narrow execution/drafting tasks. Does not inherit Hermes tools or filesystem access.",
        "instructions": """You are an OpenAI Agents SDK EXECUTE worker under Hermes/GPT-5.5 supervision.
Execute only the bounded task described in the prompt. You do not have Hermes tools, memory, filesystem, terminal, browser, or external side-effect authority unless explicitly included in the prompt as data.
Return a structured proof object. Never say work is done/fixed/verified unless your proof field explains exactly why. If execution requires tools you do not have, return status='blocked'.""",
    },
    "verify": {
        "tool": "openai_agents_verify",
        "name": "HermesOpenAIAgentsVerifyWorker",
        "description": "Independent OpenAI Agents SDK verification lane. High skepticism, no mutation authority.",
        "instructions": """You are an OpenAI Agents SDK VERIFY worker under Hermes/GPT-5.5 supervision.
Act as an independent verifier. Try to falsify the claim/task. Separate evidence from assumptions. You have no mutation authority.
Return status='verified' only when the provided evidence is sufficient under the acceptance criteria. Return 'partial' for incomplete proof and 'blocked' when verification requires missing evidence/tools.""",
    },
}


class GovernedAgentOutput(BaseModel):
    """Structured proof contract returned by governed SDK workers."""

    status: Literal["verified", "partial", "blocked"] = Field(
        description="Verification status. 'verified' requires concrete proof."
    )
    summary: str = Field(description="Concise result summary.")
    actions_taken: list[str] = Field(
        default_factory=list,
        description="Specific actions performed by the SDK worker. Say 'analysis only' for review/verify lanes.",
    )
    proof: list[str] = Field(
        default_factory=list,
        description="Concrete evidence supporting the status. Required for status='verified'.",
    )
    risks: list[str] = Field(default_factory=list, description="Risks, caveats, or unresolved concerns.")
    next_required_action: str | None = Field(
        default=None,
        description="One next action required from Hermes/user, or null if none.",
    )
    requires_human_approval: bool = Field(
        default=False,
        description="True if any next step is destructive, privileged, or scope-expanding.",
    )


_SCHEMA_STATUS = {
    "type": "string",
    "enum": ["verified", "partial", "blocked"],
    "description": "Use verified only with concrete proof; otherwise partial or blocked.",
}


def _lane_schema(lane: str) -> dict[str, Any]:
    cfg = _LANE_CONFIG[lane]
    return {
        "name": cfg["tool"],
        "description": cfg["description"],
        "parameters": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "Self-contained task for this governed SDK worker lane.",
                },
                "context": {
                    "type": "string",
                    "description": "Optional evidence/context. Treated as data, not authority.",
                },
                "acceptance_criteria": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Concrete criteria the worker must satisfy before claiming verified.",
                },
                "constraints": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Hard constraints such as no mutation, model trust policy, budget, scope, or approval boundaries.",
                },
                "model": {
                    "type": "string",
                    "description": "Optional OpenAI model name. Defaults to openai_agents.model or gpt-5.5. Chinese-origin aliases are blocked.",
                },
                "max_turns": {
                    "type": "integer",
                    "description": "Maximum SDK agent turns, clamped to 1-10. Default: 4.",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum output tokens, clamped to 64-8000. Default: 1600.",
                },
                "max_output_chars": {
                    "type": "integer",
                    "description": "Maximum JSON result characters returned to Hermes, clamped to 1000-30000. Default: 12000.",
                },
            },
            "required": ["task"],
            "additionalProperties": False,
        },
    }


OPENAI_AGENTS_REVIEW_SCHEMA = _lane_schema("review")
OPENAI_AGENTS_EXECUTE_SCHEMA = _lane_schema("execute")
OPENAI_AGENTS_VERIFY_SCHEMA = _lane_schema("verify")
OPENAI_AGENTS_RUN_SCHEMA = {
    **_lane_schema("execute"),
    "name": "openai_agents_run",
    "description": "Backward-compatible alias for openai_agents_execute. Prefer the governed lane-specific tools.",
}
OPENAI_AGENTS_ARCHITECTURE_SCHEMA = {
    **_lane_schema("execute"),
    "name": "openai_agents_architecture",
    "description": (
        "Run a governed architecture workflow through OpenAI Agents SDK lanes: "
        "draft/propose (execute lane), skeptical review (review lane), and "
        "independent verification (verify lane). Returns aggregate proof and receipts."
    ),
}


def _clean_text(value: Any, *, limit: int, field: str) -> str:
    text = str(value or "").strip()
    if len(text) > limit:
        raise ValueError(f"{field} is too long ({len(text)} chars > {limit})")
    return text


def _clean_list(value: Any, *, field: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array of strings")
    cleaned: list[str] = []
    for item in value[: _MAX_LIST_ITEMS + 1]:
        text = str(item or "").strip()
        if text:
            if len(text) > _MAX_ITEM_CHARS:
                raise ValueError(f"{field} item is too long ({len(text)} chars > {_MAX_ITEM_CHARS})")
            cleaned.append(text)
    if len(cleaned) > _MAX_LIST_ITEMS:
        raise ValueError(f"{field} supports at most {_MAX_LIST_ITEMS} items")
    return cleaned


def _clamp_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(minimum, min(maximum, parsed))


def _load_openai_agents_config() -> dict[str, Any]:
    try:
        cfg = load_config().get("openai_agents", {})
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _resolve_openai_api_key() -> str:
    api_key = str(get_env_value("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    if api_key:
        # The Agents SDK reads the standard env var in a few internal paths.
        os.environ.setdefault("OPENAI_API_KEY", api_key)
    return api_key


def _resolve_model(raw_model: Any = None) -> str:
    cfg = _load_openai_agents_config()
    model = str(raw_model or cfg.get("model") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    lowered = model.lower()
    if any(fragment in lowered for fragment in _BLOCKED_MODEL_FRAGMENTS):
        raise ValueError(f"Model/provider name is blocked by local trust policy: {model}")
    if not re.match(r"^[A-Za-z0-9._:/+\-]+$", model):
        raise ValueError(f"Model name contains unsupported characters: {model!r}")
    return model


def _resolve_base_url() -> str | None:
    cfg = _load_openai_agents_config()
    raw = cfg.get("base_url") or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
    base_url = str(raw or "").strip().rstrip("/")
    return base_url or None


def _format_worker_input(
    *,
    lane: str,
    task: str,
    context: str,
    acceptance_criteria: list[str],
    constraints: list[str],
) -> str:
    return json.dumps(
        {
            "lane": lane,
            "task": task,
            "context": context,
            "acceptance_criteria": acceptance_criteria,
            "constraints": constraints,
            "governance_contract": _GOVERNANCE_CONTRACT,
            "output_contract": {
                "status": "verified | partial | blocked",
                "summary": "concise result",
                "actions_taken": ["specific actions; review/verify should say analysis only"],
                "proof": ["concrete evidence; required for verified"],
                "risks": ["caveats"],
                "next_required_action": "string or null",
                "requires_human_approval": "boolean",
            },
        },
        ensure_ascii=False,
    )


def _preflight_request(lane: str, task: str, constraints: list[str]) -> None:
    """Fail before SDK/model spend when local governance scope is insufficient."""
    combined_constraints = "\n".join(constraints or [])
    if _HIGH_RISK_PATTERN.search(task) and not _APPROVAL_SCOPE_PATTERN.search(combined_constraints):
        raise ValueError(
            "High-risk SDK worker task requires explicit scope/approval constraint "
            "(for example: 'read-only analysis; no mutation' or 'authorized within explicit scope')."
        )
    if lane in {"review", "verify"} and not _APPROVAL_SCOPE_PATTERN.search(combined_constraints):
        # No-mutation lanes default to advisory, but adding the constraint keeps
        # the worker prompt and receipt unambiguous.
        constraints.append("analysis only; no mutation")


def _sanitize_for_receipt(value: Any) -> Any:
    """Remove obvious secret-bearing fields before persisting receipts."""
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if _SECRET_KEY_PATTERN.search(str(key)):
                continue
            result[str(key)] = _sanitize_for_receipt(item)
        return result
    if isinstance(value, list):
        return [_sanitize_for_receipt(item) for item in value]
    return value


def _receipt_dir() -> Path:
    return Path(get_hermes_home()) / "receipts" / "openai_agents"


def _write_receipt(payload: dict[str, Any]) -> str:
    """Persist a sanitized SDK-run receipt and return its absolute path."""
    sanitized = _sanitize_for_receipt(payload)
    lane = re.sub(r"[^a-z0-9_-]+", "-", str(sanitized.get("lane") or "unknown").lower()).strip("-") or "unknown"
    result = sanitized.get("result") if isinstance(sanitized.get("result"), dict) else {}
    status = re.sub(r"[^a-z0-9_-]+", "-", str(result.get("status") or "unknown").lower()).strip("-") or "unknown"
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out_dir = _receipt_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ts}-{lane}-{status}.json"
    path.write_text(json.dumps(sanitized, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return str(path)


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _failure_receipt_payload(
    *,
    lane: str,
    error: BaseException,
    task: str = "",
    context: str = "",
    acceptance_criteria: list[str] | None = None,
    constraints: list[str] | None = None,
    worker: str | None = None,
    model: str | None = None,
    max_turns: int | None = None,
    max_tokens: int | None = None,
    preflight_enforced: bool = True,
    sdk_guardrails_attached: bool = False,
) -> dict[str, Any]:
    """Build a sanitized receipt for failed SDK lane attempts.

    Failure receipts are intentionally task/context fingerprinted, not content
    dumping. They preserve traceability without writing user prompts, secrets,
    or long private context into durable receipt files.
    """
    error_type = type(error).__name__
    error_message = str(error)
    return {
        "success": False,
        "lane": lane,
        "model": model or DEFAULT_MODEL,
        "max_turns": max_turns or _DEFAULT_MAX_TURNS,
        "max_tokens": max_tokens or _DEFAULT_MAX_TOKENS,
        "usage": {"available": False, "reason": "sdk_lane_failed_before_usage_capture"},
        "governance_contract": _GOVERNANCE_CONTRACT,
        "receipt": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "worker": worker or _LANE_CONFIG.get(lane, {}).get("name") or "HermesOpenAIAgentsUnknownWorker",
            "structured_output": True,
            "preflight_enforced": preflight_enforced,
            "sdk_guardrails_attached": sdk_guardrails_attached,
            "postconditions_enforced": True,
            "trace_sensitive_data": False,
        },
        "result": {
            "status": "blocked",
            "summary": f"OpenAI Agents SDK {lane} lane failed before verified output: {error_type}",
            "actions_taken": [],
            "proof": [],
            "risks": ["No verified SDK worker output was produced for this attempt."],
            "next_required_action": "Inspect the sanitized failure receipt and retry with adjusted bounds or prompt if appropriate.",
            "requires_human_approval": False,
        },
        "truncated": False,
        "error": {"type": error_type, "message": error_message[:1000]},
        "input_fingerprints": {
            "task_sha256": _sha256_text(task) if task else None,
            "task_chars": len(task or ""),
            "context_sha256": _sha256_text(context) if context else None,
            "context_chars": len(context or ""),
            "acceptance_criteria_count": len(acceptance_criteria or []),
            "constraints_count": len(constraints or []),
        },
    }


def _write_failure_receipt(**kwargs: Any) -> tuple[str, str]:
    path = _write_receipt(_failure_receipt_payload(**kwargs))
    return path, _sha256_file(path)


def _usage_to_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    if hasattr(usage, "model_dump"):
        try:
            usage = usage.model_dump()
        except Exception:
            pass
    if not isinstance(usage, dict):
        usage = {
            name: getattr(usage, name, None)
            for name in (
                "input_tokens", "output_tokens", "total_tokens",
                "prompt_tokens", "completion_tokens",
                "cached_tokens", "reasoning_tokens",
            )
            if getattr(usage, name, None) is not None
        }
    result: dict[str, int] = {}
    aliases = {
        "input_tokens": ["input_tokens", "prompt_tokens"],
        "output_tokens": ["output_tokens", "completion_tokens"],
        "total_tokens": ["total_tokens"],
        "cached_tokens": ["cached_tokens"],
        "reasoning_tokens": ["reasoning_tokens"],
    }
    for canonical, names in aliases.items():
        for name in names:
            value = usage.get(name)
            if value is not None:
                try:
                    result[canonical] = int(value)
                    break
                except Exception:
                    pass
    return result


def _pricing_for_model(model: str) -> dict[str, Any]:
    cfg = _load_openai_agents_config().get("pricing", {})
    if not isinstance(cfg, dict):
        return {}
    models = cfg.get("models", cfg)
    if not isinstance(models, dict):
        return {}
    pricing = models.get(model) or models.get(model.lower()) or {}
    return pricing if isinstance(pricing, dict) else {}


def _apply_cost_estimate(usage: dict[str, Any], *, model: str) -> dict[str, Any]:
    if not usage.get("available"):
        return usage
    pricing = _pricing_for_model(model)
    try:
        input_per_1m = float(pricing["input_per_1m"])
        output_per_1m = float(pricing["output_per_1m"])
    except Exception:
        usage["cost_estimate_usd"] = None
        usage["cost_estimate_status"] = "not_estimated_without_verified_pricing_config"
        return usage
    cached_per_1m = pricing.get("cached_input_per_1m")
    try:
        cached_per_1m = float(cached_per_1m) if cached_per_1m is not None else input_per_1m
    except Exception:
        cached_per_1m = input_per_1m
    cached = int(usage.get("cached_tokens") or 0)
    input_tokens = max(0, int(usage.get("input_tokens") or 0) - cached)
    output_tokens = int(usage.get("output_tokens") or 0)
    estimate = (input_tokens * input_per_1m + cached * cached_per_1m + output_tokens * output_per_1m) / 1_000_000
    usage["cost_estimate_usd"] = round(estimate, 8)
    usage["cost_estimate_status"] = "estimated_from_openai_agents_pricing_config"
    return usage


def _extract_usage(result: Any, *, model: str | None = None) -> dict[str, Any]:
    """Extract best-effort token usage without assuming one SDK response shape."""
    totals = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0}
    found = False
    raw_responses = getattr(result, "raw_responses", None) or []
    for response in raw_responses:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        item = _usage_to_dict(usage)
        if item:
            found = True
            for key in totals:
                totals[key] += int(item.get(key) or 0)
    if not found:
        return {"available": False, "reason": "SDK result did not expose token usage in raw_responses"}
    if totals["total_tokens"] == 0:
        totals["total_tokens"] = totals["input_tokens"] + totals["output_tokens"]
    usage = {"available": True, **totals}
    return _apply_cost_estimate(usage, model=model or DEFAULT_MODEL)


def _guardrail_input_text(input_value: Any) -> str:
    if isinstance(input_value, str):
        return input_value
    try:
        return json.dumps(input_value, ensure_ascii=False)
    except Exception:
        return str(input_value)


def _build_agent_guardrail_kwargs(lane: str) -> dict[str, list[Any]]:
    """Return SDK guardrails supported by openai-agents 0.17.x.

    Local deterministic gates remain authoritative; these SDK guardrails add
    native tripwires around the expensive model run and final output.
    """
    try:
        from agents import GuardrailFunctionOutput, input_guardrail, output_guardrail
    except Exception:
        return {"input_guardrails": [], "output_guardrails": []}

    @input_guardrail(name=f"hermes_{lane}_input_scope", run_in_parallel=False)
    def _input_scope_guardrail(ctx, agent, input_value):
        text = _guardrail_input_text(input_value)
        if _HIGH_RISK_PATTERN.search(text) and not _APPROVAL_SCOPE_PATTERN.search(text):
            return GuardrailFunctionOutput(
                output_info={"reason": "high-risk request lacks explicit scope/approval"},
                tripwire_triggered=True,
            )
        return GuardrailFunctionOutput(output_info={"ok": True}, tripwire_triggered=False)

    @output_guardrail(name=f"hermes_{lane}_proof_output")
    def _proof_output_guardrail(ctx, agent, output_value):
        try:
            output = output_value if isinstance(output_value, GovernedAgentOutput) else GovernedAgentOutput.model_validate(output_value)
        except Exception:
            return GuardrailFunctionOutput(
                output_info={"reason": "output did not satisfy GovernedAgentOutput schema"},
                tripwire_triggered=True,
            )
        missing_verified_proof = output.status == "verified" and not output.proof
        mutation_in_no_mutation_lane = lane in {"review", "verify"} and any(
            re.search(r"\b(wrote|edited|deleted|ran|executed|changed|mutated)\b", action, re.I)
            for action in output.actions_taken
        )
        if missing_verified_proof or mutation_in_no_mutation_lane:
            return GuardrailFunctionOutput(
                output_info={
                    "missing_verified_proof": missing_verified_proof,
                    "mutation_in_no_mutation_lane": mutation_in_no_mutation_lane,
                },
                tripwire_triggered=True,
            )
        return GuardrailFunctionOutput(output_info={"ok": True}, tripwire_triggered=False)

    return {"input_guardrails": [_input_scope_guardrail], "output_guardrails": [_proof_output_guardrail]}


def _coerce_structured_output(result: Any) -> GovernedAgentOutput:
    final = getattr(result, "final_output", None)
    if isinstance(final, GovernedAgentOutput):
        return final
    if isinstance(final, BaseModel):
        return GovernedAgentOutput.model_validate(final.model_dump())
    if isinstance(final, dict):
        return GovernedAgentOutput.model_validate(final)
    if isinstance(final, str):
        try:
            return GovernedAgentOutput.model_validate_json(final)
        except ValidationError:
            pass
        try:
            return GovernedAgentOutput.model_validate(json.loads(final))
        except Exception:
            return GovernedAgentOutput(status="partial", summary=final, proof=[], risks=["Worker returned unstructured text."])
    return GovernedAgentOutput(status="partial", summary=str(final), proof=[], risks=["Worker returned non-JSON output."])


def _enforce_postconditions(output: GovernedAgentOutput, *, lane: str) -> GovernedAgentOutput:
    if output.status == "verified" and not output.proof:
        output.status = "partial"
        output.risks.append("Downgraded from verified: no proof items were supplied.")
        output.next_required_action = output.next_required_action or "Provide concrete proof before claiming verified."
    if lane in {"review", "verify"}:
        output.actions_taken = output.actions_taken or ["analysis only"]
        mutation_claims = [a for a in output.actions_taken if re.search(r"\b(wrote|edited|deleted|ran|executed|changed|mutated)\b", a, re.I)]
        if mutation_claims:
            output.status = "partial" if output.status == "verified" else output.status
            output.risks.append("Review/verify lane reported mutation-like actions; lane authority is advisory/no-mutation only.")
    return output


def _check_openai_agents_available() -> bool:
    try:
        import agents  # noqa: F401
    except Exception:
        return False
    return bool(_resolve_openai_api_key())


def _run_governed_lane(lane: Literal["review", "execute", "verify"], args: dict) -> str:
    try:
        import agents
        from agents import Agent, ModelSettings, RunConfig, Runner
        from agents.models.openai_provider import OpenAIProvider
    except Exception as exc:
        return tool_error(
            "OpenAI Agents SDK is not importable. Install it with `uv pip install openai-agents`.",
            detail=f"{type(exc).__name__}: {exc}",
        )

    task = ""
    context = ""
    acceptance_criteria: list[str] = []
    constraints: list[str] = []
    model: str | None = None
    max_turns = _DEFAULT_MAX_TURNS
    max_tokens = _DEFAULT_MAX_TOKENS
    lane_cfg = _LANE_CONFIG[lane]
    guardrail_kwargs: dict[str, list[Any]] = {"input_guardrails": [], "output_guardrails": []}

    try:
        task = _clean_text(args.get("task") or args.get("prompt"), limit=_MAX_TASK_CHARS, field="task")
        if not task:
            return tool_error("task is required")
        context = _clean_text(args.get("context"), limit=_MAX_CONTEXT_CHARS, field="context")
        acceptance_criteria = _clean_list(args.get("acceptance_criteria"), field="acceptance_criteria")
        constraints = _clean_list(args.get("constraints"), field="constraints")
        _preflight_request(lane, task, constraints)
        model = _resolve_model(args.get("model"))
        max_turns = _clamp_int(args.get("max_turns"), default=_DEFAULT_MAX_TURNS, minimum=1, maximum=10)
        max_tokens = _clamp_int(args.get("max_tokens"), default=_DEFAULT_MAX_TOKENS, minimum=64, maximum=8000)
        max_output_chars = _clamp_int(
            args.get("max_output_chars"), default=_DEFAULT_OUTPUT_CHARS, minimum=1000, maximum=30000
        )
        api_key = _resolve_openai_api_key()
        if not api_key:
            return tool_error("OPENAI_API_KEY is not configured for the OpenAI Agents SDK bridge")

        provider_kwargs: dict[str, Any] = {"api_key": api_key}
        base_url = _resolve_base_url()
        if base_url:
            provider_kwargs["base_url"] = base_url

        lane_cfg = _LANE_CONFIG[lane]
        guardrail_kwargs = _build_agent_guardrail_kwargs(lane)
        sdk_agent = Agent(
            name=lane_cfg["name"],
            instructions=lane_cfg["instructions"],
            model=model,
            model_settings=ModelSettings(max_tokens=max_tokens),
            output_type=GovernedAgentOutput,
            **guardrail_kwargs,
        )
        run_config = RunConfig(
            model_provider=OpenAIProvider(**provider_kwargs),
            tracing_disabled=True,
            trace_include_sensitive_data=False,
            workflow_name=f"Hermes OpenAI Agents SDK {lane} lane",
        )
        result = Runner.run_sync(
            sdk_agent,
            _format_worker_input(
                lane=lane,
                task=task,
                context=context,
                acceptance_criteria=acceptance_criteria,
                constraints=constraints,
            ),
            max_turns=max_turns,
            run_config=run_config,
        )
        output = _enforce_postconditions(_coerce_structured_output(result), lane=lane)
        output_payload = output.model_dump()
        output_json = json.dumps(output_payload, ensure_ascii=False)
        truncated = len(output_json) > max_output_chars
        if truncated:
            output_payload["summary"] = output_payload["summary"][: max(0, max_output_chars - 200)] + "…[truncated]"
            output_payload["risks"] = list(output_payload.get("risks") or []) + ["Result was truncated by Hermes bridge."]
        receipt = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "worker": lane_cfg["name"],
            "structured_output": True,
            "preflight_enforced": True,
            "sdk_guardrails_attached": bool(guardrail_kwargs.get("input_guardrails") or guardrail_kwargs.get("output_guardrails")),
            "postconditions_enforced": True,
            "trace_sensitive_data": False,
        }
        response_payload = {
            "success": True,
            "lane": lane,
            "sdk_version": getattr(agents, "__version__", None),
            "model": model,
            "max_turns": max_turns,
            "max_tokens": max_tokens,
            "usage": _extract_usage(result, model=model),
            "governance_contract": _GOVERNANCE_CONTRACT,
            "receipt": receipt,
            "result": output_payload,
            "truncated": truncated,
        }
        receipt_path = _write_receipt(response_payload)
        response_payload["receipt_path"] = receipt_path
        response_payload["receipt_sha256"] = _sha256_file(receipt_path)
        return tool_result(response_payload)
    except Exception as exc:
        try:
            receipt_path, receipt_sha256 = _write_failure_receipt(
                lane=lane,
                error=exc,
                task=task,
                context=context,
                acceptance_criteria=acceptance_criteria,
                constraints=constraints,
                worker=lane_cfg["name"],
                model=model,
                max_turns=max_turns,
                max_tokens=max_tokens,
                preflight_enforced=True,
                sdk_guardrails_attached=bool(
                    guardrail_kwargs.get("input_guardrails") or guardrail_kwargs.get("output_guardrails")
                ),
            )
        except Exception as receipt_exc:
            return tool_error(
                f"OpenAI Agents SDK {lane} lane failed: {type(exc).__name__}: {exc}",
                receipt_error=f"{type(receipt_exc).__name__}: {receipt_exc}",
            )
        return tool_error(
            f"OpenAI Agents SDK {lane} lane failed: {type(exc).__name__}: {exc}",
            receipt_path=receipt_path,
            receipt_sha256=receipt_sha256,
        )


def _parse_tool_json(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except Exception as exc:
        return {"error": f"invalid JSON tool result: {type(exc).__name__}: {exc}", "raw": raw}
    return payload if isinstance(payload, dict) else {"error": "tool result was not a JSON object", "raw": payload}


def _blocked_architecture_stage(stage: str, parsed: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "blocked",
        "summary": f"Architecture {stage} stage failed before verified output.",
        "actions_taken": ["analysis workflow stopped before downstream stages"],
        "proof": [],
        "risks": [str(parsed.get("error") or "unknown stage failure")[:1000]],
        "next_required_action": "Inspect the stage receipt/error and retry with adjusted bounds or narrower task.",
        "requires_human_approval": False,
    }


def _skipped_architecture_stage(reason: str) -> dict[str, Any]:
    return {
        "status": "blocked",
        "summary": "Architecture stage skipped because an upstream stage failed.",
        "actions_taken": [],
        "proof": [],
        "risks": [reason],
        "next_required_action": "Resolve the upstream stage failure first.",
        "requires_human_approval": False,
    }


def _architecture_aggregate_result(
    *,
    status: str,
    stages: dict[str, dict[str, Any]],
    stage_receipts: dict[str, str | None],
    stage_errors: dict[str, str] | None = None,
) -> str:
    aggregate = {
        "success": True,
        "workflow": "architecture",
        "status": status,
        "governance_contract": _GOVERNANCE_CONTRACT,
        "stages": stages,
        "stage_receipts": stage_receipts,
        "stage_errors": stage_errors or {},
        "receipt": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "worker": "HermesOpenAIAgentsArchitectureWorkflow",
            "structured_output": True,
            "preflight_enforced": True,
            "sdk_guardrails_attached": True,
            "postconditions_enforced": True,
            "trace_sensitive_data": False,
        },
    }
    aggregate["receipt_path"] = _write_receipt({"lane": "architecture", "result": {"status": status}, **aggregate})
    aggregate["receipt_sha256"] = _sha256_file(aggregate["receipt_path"])
    return tool_result(aggregate)


def _handle_openai_agents_architecture(args: dict, **kw) -> str:
    """Run a deterministic architecture workflow over the governed SDK lanes."""
    try:
        task = _clean_text(args.get("task") or args.get("prompt"), limit=_MAX_TASK_CHARS, field="task")
        if not task:
            return tool_error("task is required")
        context = _clean_text(args.get("context"), limit=_MAX_CONTEXT_CHARS, field="context")
        acceptance_criteria = _clean_list(args.get("acceptance_criteria"), field="acceptance_criteria")
        constraints = _clean_list(args.get("constraints"), field="constraints")
        workflow_constraints = constraints + [
            "architecture workflow; analysis/drafting only unless explicitly authorized",
            "no mutation; no filesystem, terminal, browser, memory, or network side effects",
            "separate evidence from assumptions and identify approval boundaries",
            "keep each structured field concise; do not quote full prompt/context/constraints",
        ]
        _preflight_request("review", task, workflow_constraints)
        shared = {
            "model": args.get("model"),
            "max_turns": _clamp_int(args.get("max_turns"), default=_DEFAULT_MAX_TURNS, minimum=1, maximum=10),
            "max_tokens": max(
                _ARCHITECTURE_MIN_TOKENS,
                _clamp_int(args.get("max_tokens"), default=_DEFAULT_MAX_TOKENS, minimum=64, maximum=8000),
            ),
            "max_output_chars": _clamp_int(args.get("max_output_chars"), default=_DEFAULT_OUTPUT_CHARS, minimum=1000, maximum=30000),
        }
        proposal_raw = _run_governed_lane("execute", {
            "task": "Draft a best-practice architecture proposal for: " + task,
            "context": context,
            "acceptance_criteria": acceptance_criteria + [
                "Return a concrete architecture proposal with assumptions, risks, and proof-grounded rationale."
            ],
            "constraints": workflow_constraints,
            **shared,
        })
        proposal = _parse_tool_json(proposal_raw)
        if proposal.get("error"):
            reason = "proposal stage failed; downstream architecture stages were not run"
            return _architecture_aggregate_result(
                status="blocked",
                stages={
                    "proposal": _blocked_architecture_stage("proposal", proposal),
                    "review": _skipped_architecture_stage(reason),
                    "verification": _skipped_architecture_stage(reason),
                },
                stage_receipts={
                    "proposal": proposal.get("receipt_path"),
                    "review": None,
                    "verification": None,
                },
                stage_errors={"proposal": str(proposal.get("error"))[:1000]},
            )

        review_raw = _run_governed_lane("review", {
            "task": "Skeptically review the architecture proposal for gaps, unsafe assumptions, and missing proof.",
            "context": json.dumps({"original_task": task, "proposal": proposal}, ensure_ascii=False),
            "acceptance_criteria": acceptance_criteria + [
                "Identify blockers, weak claims, and required revisions before implementation."
            ],
            "constraints": workflow_constraints,
            **shared,
        })
        review = _parse_tool_json(review_raw)
        if review.get("error"):
            reason = "review stage failed; verification stage was not run"
            return _architecture_aggregate_result(
                status="blocked",
                stages={
                    "proposal": (proposal.get("result") or {}) if isinstance(proposal.get("result"), dict) else {},
                    "review": _blocked_architecture_stage("review", review),
                    "verification": _skipped_architecture_stage(reason),
                },
                stage_receipts={
                    "proposal": proposal.get("receipt_path"),
                    "review": review.get("receipt_path"),
                    "verification": None,
                },
                stage_errors={"review": str(review.get("error"))[:1000]},
            )

        verify_raw = _run_governed_lane("verify", {
            "task": "Verify whether the architecture workflow output is ready for Hermes orchestration planning.",
            "context": json.dumps({"original_task": task, "proposal": proposal, "review": review}, ensure_ascii=False),
            "acceptance_criteria": acceptance_criteria + [
                "Verified requires concrete proof and no unresolved high-risk blockers."
            ],
            "constraints": workflow_constraints,
            **shared,
        })
        verification = _parse_tool_json(verify_raw)
        if verification.get("error"):
            return _architecture_aggregate_result(
                status="blocked",
                stages={
                    "proposal": (proposal.get("result") or {}) if isinstance(proposal.get("result"), dict) else {},
                    "review": (review.get("result") or {}) if isinstance(review.get("result"), dict) else {},
                    "verification": _blocked_architecture_stage("verification", verification),
                },
                stage_receipts={
                    "proposal": proposal.get("receipt_path"),
                    "review": review.get("receipt_path"),
                    "verification": verification.get("receipt_path"),
                },
                stage_errors={"verification": str(verification.get("error"))[:1000]},
            )

        verification_result = (verification.get("result") or {}) if isinstance(verification.get("result"), dict) else {}
        review_result = (review.get("result") or {}) if isinstance(review.get("result"), dict) else {}
        proposal_result = (proposal.get("result") or {}) if isinstance(proposal.get("result"), dict) else {}
        status = verification_result.get("status") or "partial"
        if review_result.get("status") == "blocked" or proposal_result.get("status") == "blocked":
            status = "blocked"
        aggregate = {
            "success": True,
            "workflow": "architecture",
            "status": status,
            "governance_contract": _GOVERNANCE_CONTRACT,
            "stages": {
                "proposal": proposal_result,
                "review": review_result,
                "verification": verification_result,
            },
            "stage_receipts": {
                "proposal": proposal.get("receipt_path"),
                "review": review.get("receipt_path"),
                "verification": verification.get("receipt_path"),
            },
            "receipt": {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "worker": "HermesOpenAIAgentsArchitectureWorkflow",
                "structured_output": True,
                "preflight_enforced": True,
                "sdk_guardrails_attached": True,
                "postconditions_enforced": True,
                "trace_sensitive_data": False,
            },
        }
        aggregate["receipt_path"] = _write_receipt({"lane": "architecture", "result": {"status": status}, **aggregate})
        aggregate["receipt_sha256"] = _sha256_file(aggregate["receipt_path"])
        return tool_result(aggregate)
    except Exception as exc:
        return tool_error(f"OpenAI Agents SDK architecture workflow failed: {type(exc).__name__}: {exc}")


def _handle_openai_agents_review(args: dict, **kw) -> str:
    return _run_governed_lane("review", args)


def _handle_openai_agents_execute(args: dict, **kw) -> str:
    return _run_governed_lane("execute", args)


def _handle_openai_agents_verify(args: dict, **kw) -> str:
    return _run_governed_lane("verify", args)


def _handle_openai_agents_run(args: dict, **kw) -> str:
    """Backward-compatible alias for the governed execute lane."""
    if "task" not in args and "prompt" in args:
        args = {**args, "task": args.get("prompt")}
    return _run_governed_lane("execute", args)
