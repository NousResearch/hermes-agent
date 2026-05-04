"""CaMeL-style trust separation for Hermes tool execution.

This module separates trusted control inputs from untrusted data inputs:
- trusted control comes from the system prompt, approved skills, and user turns
- untrusted data comes from tool outputs and retrieved context
- sensitive tools are authorized against a trusted action plan, not against
  instructions embedded in untrusted content
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import re
from typing import Any, Dict, Iterable, List, Sequence


CAMEL_UNTRUSTED_PREFIX = "[CaMeL: UNTRUSTED TOOL DATA]"
CAMEL_GUARD_RUNTIME_CHOICES = ("on", "off", "monitor", "enforce", "legacy")

_CAMEL_GUARD_MODE_ALIASES = {
    "on": "monitor",
    "off": "off",
    "monitor": "monitor",
    "enforce": "enforce",
    "legacy": "off",
}

_TRUSTED_CONTROL_TOOLS = {
    "clarify",
    "skill_view",
    "skills_list",
    "todo",
}

_SENSITIVE_TOOL_CAPABILITIES = {
    "browser_click": "browser_interaction",
    "browser_press": "browser_interaction",
    "browser_type": "browser_interaction",
    "cronjob": "scheduled_action",
    "delegate_task": "delegation",
    "execute_code": "command_execution",
    "ha_call_service": "external_side_effect",
    "memory": "persistent_memory",
    "mixture_of_agents": "delegation",
    "patch": "file_mutation",
    "rl_edit_config": "file_mutation",
    "rl_start_training": "external_side_effect",
    "rl_stop_training": "external_side_effect",
    "send_message": "external_messaging",
    "skill_manage": "skill_mutation",
    "terminal": "command_execution",
    "write_file": "file_mutation",
}

_CAPABILITY_LABELS = {
    "browser_interaction": "browser interaction",
    "command_execution": "command execution",
    "delegation": "delegation / subagents",
    "external_messaging": "external messaging",
    "external_side_effect": "external system side effects",
    "file_mutation": "file mutation",
    "persistent_memory": "persistent memory writes",
    "scheduled_action": "scheduled actions",
    "skill_mutation": "skill mutation",
}

_CAPABILITY_ORDER = tuple(sorted(_CAPABILITY_LABELS.keys()))

_CAPABILITY_CLASSIFIER_PROMPT = """You classify trusted operator intent for Hermes' CaMeL guard.

Input contains only trusted user instructions. You do not have tools, memory, or external context.
Return strict JSON with this exact shape:
{
  "goal_summary": "short summary",
  "allowed_capabilities": ["command_execution"],
  "denied_capabilities": ["external_messaging"],
  "rationale": "short reason"
}

Rules:
- Only allow a capability when the trusted user explicitly asks for it or it is clearly necessary.
- If intent is ambiguous, do not allow the capability.
- Put a capability in denied_capabilities only when the user explicitly forbids it.
- Do not invent capabilities outside the allowed list.
- Be conservative.

Allowed capability ids:
""" + "\n".join(
    f"- {cap}: {_CAPABILITY_LABELS[cap]}" for cap in sorted(_CAPABILITY_LABELS.keys())
)

_SUSPICIOUS_INSTRUCTION_PATTERNS = [
    (re.compile(r"ignore\s+(previous|all|above|prior)\s+instructions", re.IGNORECASE), "ignore_previous_instructions"),
    (re.compile(r"do\s+not\s+tell\s+the\s+user", re.IGNORECASE), "hide_from_user"),
    (re.compile(r"(reveal|show|print|dump).*(system prompt|api key|token|secret|credential)", re.IGNORECASE), "secret_exfiltration"),
    (re.compile(r"system\s+prompt\s+override", re.IGNORECASE), "system_prompt_override"),
    (re.compile(r"send_message|tweet|email|dm|post this", re.IGNORECASE), "embedded_side_effect_instruction"),
]

_OUTPUT_INSTRUCTION_PATTERNS = [
    re.compile(r"\b(?:begin|start)\s+your\s+reply\s+with:\s*(.+)$", re.IGNORECASE),
    re.compile(r"\b(?:prefix|start)\s+your\s+output\s+with:\s*(.+)$", re.IGNORECASE),
    re.compile(r"\brespond\s+with:\s*(.+)$", re.IGNORECASE),
    re.compile(r"\boutput\s+exactly:\s*(.+)$", re.IGNORECASE),
    re.compile(r"\bthen\s+write:\s*(.+)$", re.IGNORECASE),
    re.compile(r"\bwrite:\s*(.+)$", re.IGNORECASE),
]

_OUTPUT_ANALYSIS_CONTEXT_RE = re.compile(
    r"\b(quote|repeat|show\s+the\s+hidden|extract\s+the\s+hidden|what\s+does\s+the\s+hidden|"
    r"analyze\s+the\s+hidden|explain\s+the\s+hidden|classify\s+the\s+hidden|prompt injection)\b",
    re.IGNORECASE,
)

_SYSTEM_ANNOTATION_RE = re.compile(r"\[System:.*?\]", re.IGNORECASE | re.DOTALL)
_URL_RE = re.compile(r"https?://\S+", re.IGNORECASE)
_PATH_RE = re.compile(
    r"(?:~?/[\w.\-~/]+|"
    r"(?:\./|\.\./)[\w.\-~/]+|"
    r"\b[\w.\-]+\.(?:py|js|ts|tsx|jsx|json|ya?ml|md|txt|sh|toml|ini|cfg|go|rs|java|c|cc|cpp|h|hpp|sql)\b)"
)


def _truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _strip_system_annotations(text: str) -> str:
    return _SYSTEM_ANNOTATION_RE.sub("", text or "").strip()


def _extract_urls(text: str) -> List[str]:
    seen: List[str] = []
    for match in _URL_RE.findall(text or ""):
        if match not in seen:
            seen.append(match)
    return seen[:4]


def _extract_paths(text: str) -> List[str]:
    seen: List[str] = []
    for match in _PATH_RE.findall(text or ""):
        if match not in seen:
            seen.append(match)
    return seen[:6]


def _extract_suspicious_flags(text: str) -> List[str]:
    flags: List[str] = []
    haystack = text or ""
    for pattern, label in _SUSPICIOUS_INSTRUCTION_PATTERNS:
        if pattern.search(haystack):
            flags.append(label)
    return flags


def _extract_first_json_object(text: str) -> Dict[str, Any] | None:
    haystack = (text or "").strip()
    if not haystack:
        return None

    try:
        parsed = json.loads(haystack)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = haystack.find("{")
    end = haystack.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    snippet = haystack[start : end + 1]
    try:
        parsed = json.loads(snippet)
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _call_trusted_capability_classifier(messages: List[Dict[str, str]]) -> Dict[str, Any]:
    from agent.auxiliary_client import call_llm

    response = call_llm(
        task="camel_guard",
        messages=messages,
        temperature=0,
        max_tokens=220,
        timeout=12.0,
    )
    content = ""
    if getattr(response, "choices", None):
        choice = response.choices[0]
        message = getattr(choice, "message", None)
        content = getattr(message, "content", "") or ""
    parsed = _extract_first_json_object(content)
    if not parsed:
        raise ValueError("CaMeL classifier returned invalid JSON")
    return parsed


def _normalize_for_match(text: str) -> str:
    return " ".join((text or "").split()).strip().casefold()


def _extract_output_markers(text: str) -> List[str]:
    markers: List[str] = []
    for raw_line in (text or "").splitlines():
        line = raw_line.strip().lstrip("-*").strip()
        if not line:
            continue
        for pattern in _OUTPUT_INSTRUCTION_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            marker = match.group(1).strip().strip("`\"'")
            marker = re.sub(r"\s+", " ", marker).strip()
            if marker and marker not in markers:
                markers.append(marker[:120])
    return markers


def _response_starts_with_marker(response_text: str, marker: str) -> bool:
    normalized_marker = _normalize_for_match(marker)
    if not normalized_marker:
        return False

    lines = [line.strip() for line in (response_text or "").splitlines() if line.strip()]
    for line in lines[:4]:
        if _normalize_for_match(line).startswith(normalized_marker):
            return True
    return False


def _strip_marker_from_response(response_text: str, marker: str) -> str:
    if not response_text:
        return response_text

    escaped = re.escape(marker)
    leading_line_pattern = re.compile(rf"^\s*{escaped}\s*$\n?", re.IGNORECASE | re.MULTILINE)
    updated = leading_line_pattern.sub("", response_text, count=1)

    inline_prefix_pattern = re.compile(rf"^\s*{escaped}(?:\s*[:\-]\s*)?", re.IGNORECASE)
    updated = inline_prefix_pattern.sub("", updated, count=1)
    return updated.lstrip()


def _format_capabilities(capabilities: Sequence[str]) -> str:
    if not capabilities:
        return "none"
    return ", ".join(_CAPABILITY_LABELS.get(cap, cap.replace("_", " ")) for cap in capabilities)


def _extract_source_label(tool_name: str) -> str:
    if tool_name.startswith("mcp_"):
        return "mcp"
    if tool_name.startswith("browser_"):
        return "browser"
    return tool_name


def _tool_capability(tool_name: str, tool_args: Dict[str, Any] | None = None) -> str:
    tool_args = tool_args or {}
    if tool_name == "send_message" and str(tool_args.get("action", "send")).lower() == "list":
        return ""
    if tool_name == "cronjob" and str(tool_args.get("action", "")).lower() == "list":
        return ""
    return _SENSITIVE_TOOL_CAPABILITIES.get(tool_name, "")


def normalize_camel_guard_mode(value: Any, *, default: str = "monitor") -> str:
    raw = "" if value is None else str(value).strip().lower()
    if not raw:
        raw = default
    normalized = _CAMEL_GUARD_MODE_ALIASES.get(raw, raw)
    if normalized not in {"off", "monitor", "enforce"}:
        normalized = default
    return normalized


def is_untrusted_tool(tool_name: str) -> bool:
    # In the full CaMeL model, nearly all tool outputs are data, not control.
    return tool_name not in _TRUSTED_CONTROL_TOOLS


def is_sensitive_tool(tool_name: str, tool_args: Dict[str, Any] | None = None) -> bool:
    return bool(_tool_capability(tool_name, tool_args))


def _message_contains_untrusted_marker(message: Dict[str, Any]) -> bool:
    if message.get("_camel_untrusted"):
        return True
    content = message.get("content", "")
    if isinstance(content, str) and CAMEL_UNTRUSTED_PREFIX in content:
        return True
    if not isinstance(content, str):
        return False
    try:
        parsed = json.loads(content)
    except Exception:
        return False
    if isinstance(parsed, dict):
        meta = parsed.get("_camel_guard")
        return isinstance(meta, dict) and meta.get("trust") == "untrusted_data"
    return False


def _tool_call_source_index(history: Sequence[Dict[str, Any]]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for message in history:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            call_id = str(tool_call.get("id") or "").strip()
            function = tool_call.get("function") or {}
            tool_name = str(function.get("name") or "").strip()
            if call_id and tool_name:
                index[call_id] = tool_name
    return index


def _extract_untrusted_record(message: Dict[str, Any]) -> tuple[str, List[str], List[str]] | None:
    if not _message_contains_untrusted_marker(message):
        return None

    source = message.get("_camel_source") or "history"
    flags: List[str] = []
    markers: List[str] = []
    content = message.get("content", "")

    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except Exception:
            parsed = None
        if isinstance(parsed, dict):
            meta = parsed.get("_camel_guard")
            if isinstance(meta, dict):
                source = str(meta.get("source") or source)
                raw_flags = meta.get("flags") or []
                flags = [str(flag) for flag in raw_flags if str(flag).strip()]
                raw_markers = meta.get("output_markers") or []
                markers = [str(marker) for marker in raw_markers if str(marker).strip()]
        else:
            flags = _extract_suspicious_flags(content)
            markers = _extract_output_markers(content)

    return source, flags, markers


def sanitize_message_for_api(message: Dict[str, Any]) -> Dict[str, Any]:
    """Drop internal guard bookkeeping before sending messages to providers."""
    sanitized = {}
    for key, value in message.items():
        if key.startswith("_camel_"):
            continue
        sanitized[key] = value
    return sanitized


@dataclass
class CamelGuardConfig:
    enabled: bool = False
    mode: str = "monitor"
    wrap_untrusted_tool_results: bool = False
    trace_enabled: bool = True
    trace_preview_chars: int = 220

    @classmethod
    def from_dict(cls, raw: Dict[str, Any] | None) -> "CamelGuardConfig":
        raw = raw or {}
        mode = normalize_camel_guard_mode(raw.get("mode"), default="monitor")
        preview_chars = raw.get("trace_preview_chars", 220)
        try:
            preview_chars = int(preview_chars)
        except Exception:
            preview_chars = 220
        preview_chars = max(80, min(preview_chars, 1000))
        return cls(
            enabled=bool(raw.get("enabled", False)),
            mode=mode,
            wrap_untrusted_tool_results=bool(raw.get("wrap_untrusted_tool_results", False)),
            trace_enabled=bool(raw.get("trace_enabled", True)),
            trace_preview_chars=preview_chars,
        )


@dataclass
class CamelPlan:
    operator_request: str = ""
    goal_summary: str = ""
    trusted_context_excerpt: List[str] = field(default_factory=list)
    allowed_capabilities: List[str] = field(default_factory=list)
    denied_capabilities: List[str] = field(default_factory=list)
    read_only: bool = True
    mentioned_urls: List[str] = field(default_factory=list)
    mentioned_paths: List[str] = field(default_factory=list)
    planner: str = "none"
    planner_status: str = "disabled"
    planner_notes: str = ""

    @classmethod
    def from_trusted_history(
        cls, current_user_message: str, trusted_user_history: Sequence[str]
    ) -> "CamelPlan":
        cleaned_current = _strip_system_annotations(current_user_message)
        history = [_strip_system_annotations(msg) for msg in trusted_user_history if _strip_system_annotations(msg)]
        if cleaned_current and (not history or history[-1] != cleaned_current):
            history = [*history, cleaned_current]

        recent = history[-3:]
        policy_source = "\n".join(recent)
        goal_source = cleaned_current or (recent[-1] if recent else "")
        goal_summary = _truncate(goal_source or "No explicit operator goal available.", 220)

        urls = _extract_urls(policy_source)
        paths = _extract_paths(policy_source)
        trusted_excerpt = [_truncate(item, 160) for item in recent[-3:]]

        return cls(
            operator_request=cleaned_current or "",
            goal_summary=goal_summary,
            trusted_context_excerpt=trusted_excerpt,
            allowed_capabilities=[],
            denied_capabilities=[],
            read_only=True,
            mentioned_urls=urls,
            mentioned_paths=paths,
            planner="none",
            planner_status="disabled",
            planner_notes="CaMeL disabled or classifier not run",
        )


@dataclass
class CamelDecision:
    allowed: bool
    reason: str
    sources: List[str] = field(default_factory=list)
    capability: str = ""


@dataclass
class CamelResponseDecision:
    allowed: bool
    reason: str
    content: str
    matched_markers: List[str] = field(default_factory=list)


@dataclass
class CamelToolDecisionTrace:
    tool_name: str
    capability: str
    allowed: bool
    reason: str
    sources: List[str] = field(default_factory=list)
    tool_args_preview: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CamelResponseTrace:
    allowed: bool
    reason: str
    matched_markers: List[str] = field(default_factory=list)
    original_preview: str = ""
    final_preview: str = ""


@dataclass
class CamelTurnTrace:
    turn_index: int
    started_at: str
    runtime_mode: str
    operator_request: str = ""
    goal_summary: str = ""
    trusted_context_excerpt: List[str] = field(default_factory=list)
    planner: str = ""
    planner_status: str = ""
    planner_notes: str = ""
    untrusted_sources: List[str] = field(default_factory=list)
    untrusted_source_counts: Dict[str, int] = field(default_factory=dict)
    suspicious_flags: Dict[str, int] = field(default_factory=dict)
    output_markers: List[str] = field(default_factory=list)
    tool_decisions: List[CamelToolDecisionTrace] = field(default_factory=list)
    response_decision: CamelResponseTrace | None = None


@dataclass
class CamelGuard:
    config: CamelGuardConfig
    latest_trusted_user_message: str = ""
    trusted_user_history: List[str] = field(default_factory=list)
    current_plan: CamelPlan = field(default_factory=CamelPlan)
    untrusted_sources: List[str] = field(default_factory=list)
    untrusted_source_counts: Dict[str, int] = field(default_factory=dict)
    untrusted_flag_counts: Dict[str, int] = field(default_factory=dict)
    untrusted_output_markers: List[str] = field(default_factory=list)
    session_id: str = ""
    trace_turns: List[CamelTurnTrace] = field(default_factory=list)
    current_turn_trace: CamelTurnTrace | None = None
    _plan_cache: Dict[str, CamelPlan] = field(default_factory=dict)

    def set_session_id(self, session_id: str) -> None:
        self.session_id = session_id or ""

    def _trace_mode(self) -> str:
        if not self.config.enabled or self.config.mode == "off":
            return "off"
        return self.config.mode

    def _tool_args_preview(self, tool_args: Dict[str, Any] | None = None) -> Dict[str, Any]:
        preview: Dict[str, Any] = {}
        tool_args = tool_args or {}
        for index, (key, value) in enumerate(tool_args.items()):
            if index >= 6:
                preview["..."] = f"+{len(tool_args) - 6} more fields"
                break
            if isinstance(value, str):
                preview[key] = _truncate(value, self.config.trace_preview_chars // 2)
            elif isinstance(value, (int, float, bool)) or value is None:
                preview[key] = value
            else:
                try:
                    serialized = json.dumps(value, ensure_ascii=False, default=str)
                except Exception:
                    serialized = str(value)
                preview[key] = _truncate(serialized, self.config.trace_preview_chars // 2)
        return preview

    def _sync_current_turn_trace(self) -> None:
        if not self.config.trace_enabled or not self.current_turn_trace:
            return

        self.current_turn_trace.runtime_mode = self._trace_mode()
        self.current_turn_trace.operator_request = self.current_plan.operator_request
        self.current_turn_trace.goal_summary = self.current_plan.goal_summary
        self.current_turn_trace.trusted_context_excerpt = list(self.current_plan.trusted_context_excerpt)
        self.current_turn_trace.planner = self.current_plan.planner
        self.current_turn_trace.planner_status = self.current_plan.planner_status
        self.current_turn_trace.planner_notes = self.current_plan.planner_notes
        self.current_turn_trace.untrusted_sources = list(self.untrusted_sources)
        self.current_turn_trace.untrusted_source_counts = dict(self.untrusted_source_counts)
        self.current_turn_trace.suspicious_flags = dict(self.untrusted_flag_counts)
        self.current_turn_trace.output_markers = list(self.untrusted_output_markers)

    def _plan_cache_key(self, current_user_message: str, trusted_user_history: Sequence[str]) -> str:
        cleaned_current = _strip_system_annotations(current_user_message)
        history = [_strip_system_annotations(msg) for msg in trusted_user_history if _strip_system_annotations(msg)]
        if cleaned_current and (not history or history[-1] != cleaned_current):
            history = [*history, cleaned_current]
        recent = history[-3:]
        return "\n".join(recent).strip().casefold()

    def _base_trusted_plan(self) -> CamelPlan:
        plan = CamelPlan.from_trusted_history(
            self.latest_trusted_user_message,
            self.trusted_user_history,
        )
        if self.config.enabled and self.config.mode != "off":
            plan.planner = "auxiliary_llm"
            plan.planner_status = "deferred"
            plan.planner_notes = (
                "Classifier deferred until a sensitive tool decision requires "
                "trusted-capability authorization under untrusted context"
            )
        return plan

    def _classify_trusted_plan(self, current_user_message: str, trusted_user_history: Sequence[str]) -> CamelPlan:
        base_plan = CamelPlan.from_trusted_history(current_user_message, trusted_user_history)
        if not self.config.enabled or self.config.mode == "off":
            return base_plan

        cache_key = self._plan_cache_key(current_user_message, trusted_user_history)
        if cache_key and cache_key in self._plan_cache:
            return self._plan_cache[cache_key]

        classifier_messages = [
            {"role": "system", "content": _CAPABILITY_CLASSIFIER_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "current_request": base_plan.operator_request,
                        "recent_trusted_user_turns": base_plan.trusted_context_excerpt,
                        "mentioned_urls": base_plan.mentioned_urls,
                        "mentioned_paths": base_plan.mentioned_paths,
                    },
                    ensure_ascii=False,
                ),
            },
        ]

        try:
            payload = _call_trusted_capability_classifier(classifier_messages)
            raw_allowed = payload.get("allowed_capabilities") or []
            raw_denied = payload.get("denied_capabilities") or []
            allowed = sorted(
                cap for cap in {str(item).strip() for item in raw_allowed}
                if cap in _CAPABILITY_ORDER
            )
            denied = sorted(
                cap for cap in {str(item).strip() for item in raw_denied}
                if cap in _CAPABILITY_ORDER
            )
            allowed = [cap for cap in allowed if cap not in denied]
            goal_summary = _truncate(str(payload.get("goal_summary") or base_plan.goal_summary), 220)
            rationale = _truncate(str(payload.get("rationale") or "Classifier-derived policy plan"), 260)
            plan = CamelPlan(
                operator_request=base_plan.operator_request,
                goal_summary=goal_summary,
                trusted_context_excerpt=list(base_plan.trusted_context_excerpt),
                allowed_capabilities=allowed,
                denied_capabilities=denied,
                read_only=not bool(allowed),
                mentioned_urls=list(base_plan.mentioned_urls),
                mentioned_paths=list(base_plan.mentioned_paths),
                planner="auxiliary_llm",
                planner_status="ok",
                planner_notes=rationale,
            )
        except Exception as exc:
            plan = CamelPlan(
                operator_request=base_plan.operator_request,
                goal_summary=base_plan.goal_summary,
                trusted_context_excerpt=list(base_plan.trusted_context_excerpt),
                allowed_capabilities=[],
                denied_capabilities=[],
                read_only=True,
                mentioned_urls=list(base_plan.mentioned_urls),
                mentioned_paths=list(base_plan.mentioned_paths),
                planner="auxiliary_llm",
                planner_status="fallback_read_only",
                planner_notes=_truncate(f"Classifier unavailable: {exc}", 260),
            )

        if cache_key:
            self._plan_cache[cache_key] = plan
        return plan

    def _ensure_classified_plan(self) -> CamelPlan:
        if not self.config.enabled or self.config.mode == "off":
            self.current_plan = self._base_trusted_plan()
            return self.current_plan

        if self.current_plan.planner_status in {"ok", "fallback_read_only"}:
            return self.current_plan

        self.current_plan = self._classify_trusted_plan(
            self.latest_trusted_user_message,
            self.trusted_user_history,
        )
        self._sync_current_turn_trace()
        return self.current_plan

    def _record_tool_decision(
        self,
        tool_name: str,
        tool_args: Dict[str, Any] | None,
        decision: CamelDecision,
    ) -> None:
        if not self.config.trace_enabled or not self.current_turn_trace:
            return

        self.current_turn_trace.tool_decisions.append(
            CamelToolDecisionTrace(
                tool_name=tool_name,
                capability=decision.capability,
                allowed=decision.allowed,
                reason=decision.reason,
                sources=list(decision.sources),
                tool_args_preview=self._tool_args_preview(tool_args),
            )
        )
        self._sync_current_turn_trace()

    def _record_response_decision(
        self,
        original_content: str,
        decision: CamelResponseDecision,
    ) -> None:
        if not self.config.trace_enabled or not self.current_turn_trace:
            return

        self.current_turn_trace.response_decision = CamelResponseTrace(
            allowed=decision.allowed,
            reason=decision.reason,
            matched_markers=list(decision.matched_markers),
            original_preview=_truncate(original_content or "", self.config.trace_preview_chars),
            final_preview=_truncate(decision.content or "", self.config.trace_preview_chars),
        )
        self._sync_current_turn_trace()

    def trace_summary(self) -> Dict[str, Any]:
        policy_alerts = 0
        response_alerts = 0
        sources: set[str] = set()
        for turn in self.trace_turns:
            sources.update(turn.untrusted_sources)
            policy_alerts += sum(1 for item in turn.tool_decisions if not item.allowed)
            if turn.response_decision and not turn.response_decision.allowed:
                response_alerts += 1

        return {
            "session_id": self.session_id,
            "runtime_mode": self._trace_mode(),
            "turn_count": len(self.trace_turns),
            "policy_alert_count": policy_alerts,
            "response_alert_count": response_alerts,
            "unique_untrusted_sources": sorted(sources),
        }

    def trace_payload(self) -> Dict[str, Any]:
        return {
            "trace_version": 1,
            "session_id": self.session_id,
            "runtime_mode": self._trace_mode(),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": self.trace_summary(),
            "turns": [asdict(turn) for turn in self.trace_turns],
        }

    def mark_user_message(
        self,
        message: Dict[str, Any],
        *,
        operator_request: str | None = None,
    ) -> Dict[str, Any]:
        marked = dict(message)
        marked["_camel_trust"] = "trusted_user"
        if operator_request is not None:
            marked["_camel_operator_request"] = operator_request
        return marked

    def mark_assistant_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        marked = dict(message)
        marked["_camel_trust"] = "model_generated"
        return marked

    def mark_system_control_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        marked = dict(message)
        marked["_camel_trust"] = "system_control"
        return marked

    def _remember_untrusted_observation(
        self,
        source: str,
        flags: Sequence[str],
        markers: Sequence[str] | None = None,
    ) -> None:
        if source not in self.untrusted_sources:
            self.untrusted_sources.append(source)
        counts = Counter(self.untrusted_source_counts)
        counts[source] += 1
        self.untrusted_source_counts = dict(counts)

        flag_counts = Counter(self.untrusted_flag_counts)
        for flag in flags:
            flag_counts[flag] += 1
        self.untrusted_flag_counts = dict(flag_counts)

        for marker in markers or []:
            if marker not in self.untrusted_output_markers:
                self.untrusted_output_markers.append(marker)
        self._sync_current_turn_trace()

    def begin_turn(self, user_message: str, history: Iterable[Dict[str, Any]] | None = None) -> None:
        self.latest_trusted_user_message = _strip_system_annotations(user_message or "")
        self.trusted_user_history = []
        self.current_plan = CamelPlan()
        self.untrusted_sources = []
        self.untrusted_source_counts = {}
        self.untrusted_flag_counts = {}
        self.untrusted_output_markers = []

        history_list = list(history or [])
        tool_sources = _tool_call_source_index(history_list)

        for message in history_list:
            if (
                message.get("role") == "user"
                and not message.get("_flush_sentinel")
                and message.get("_camel_trust") != "system_control"
            ):
                trusted_text = message.get("_camel_operator_request") or message.get("content", "")
                trusted_text = _strip_system_annotations(str(trusted_text))
                if trusted_text:
                    self.trusted_user_history.append(trusted_text)

            record = _extract_untrusted_record(message)
            if record:
                source, flags, markers = record
                self._remember_untrusted_observation(source, flags, markers)
                continue

            if message.get("role") == "tool":
                tool_call_id = str(message.get("tool_call_id") or "").strip()
                source = tool_sources.get(tool_call_id, "")
                if source and is_untrusted_tool(source):
                    content = message.get("content", "")
                    text = content if isinstance(content, str) else ""
                    flags = _extract_suspicious_flags(text)
                    markers = _extract_output_markers(text)
                    self._remember_untrusted_observation(source, flags, markers)

        self.current_plan = self._base_trusted_plan()
        if self.config.trace_enabled:
            self.current_turn_trace = CamelTurnTrace(
                turn_index=len(self.trace_turns) + 1,
                started_at=datetime.now(timezone.utc).isoformat(),
                runtime_mode=self._trace_mode(),
            )
            self.trace_turns.append(self.current_turn_trace)
            self._sync_current_turn_trace()
        else:
            self.current_turn_trace = None

    def system_prompt_guidance(self) -> str:
        return ""

    def render_security_envelope(self) -> str:
        return ""

    def evaluate_tool_call(self, tool_name: str, tool_args: Dict[str, Any] | None = None) -> CamelDecision:
        tool_args = tool_args or {}

        if not self.config.enabled or self.config.mode == "off":
            decision = CamelDecision(True, "CaMeL disabled")
            self._record_tool_decision(tool_name, tool_args, decision)
            return decision

        capability = _tool_capability(tool_name, tool_args)
        if not capability:
            decision = CamelDecision(True, "Tool is not policy-gated")
            self._record_tool_decision(tool_name, tool_args, decision)
            return decision

        if not self.untrusted_sources:
            decision = CamelDecision(
                True,
                "No untrusted data in current context",
                capability=capability,
            )
            self._record_tool_decision(tool_name, tool_args, decision)
            return decision

        plan = self._ensure_classified_plan()

        if capability in plan.denied_capabilities:
            decision = CamelDecision(
                False,
                f"Blocked by CaMeL guard: the trusted operator request explicitly denied {_CAPABILITY_LABELS.get(capability, capability)}",
                sources=list(self.untrusted_sources),
                capability=capability,
            )
            self._record_tool_decision(tool_name, tool_args, decision)
            return decision

        if capability in plan.allowed_capabilities:
            decision = CamelDecision(
                True,
                f"Trusted operator plan authorizes {_CAPABILITY_LABELS.get(capability, capability)}",
                sources=list(self.untrusted_sources),
                capability=capability,
            )
            self._record_tool_decision(tool_name, tool_args, decision)
            return decision

        decision = CamelDecision(
            False,
            (
                f"Blocked by CaMeL guard: {tool_name} requires {_CAPABILITY_LABELS.get(capability, capability)} "
                f"but current context includes untrusted data from {', '.join(self.untrusted_sources)} "
                "and the trusted operator plan did not authorize that capability"
            ),
            sources=list(self.untrusted_sources),
            capability=capability,
        )
        self._record_tool_decision(tool_name, tool_args, decision)
        return decision

    def evaluate_assistant_response(self, content: str) -> CamelResponseDecision:
        text = content or ""
        if not self.config.enabled or self.config.mode == "off" or not text.strip():
            decision = CamelResponseDecision(True, "CaMeL disabled or empty response", text)
            self._record_response_decision(text, decision)
            return decision

        if not self.untrusted_output_markers:
            decision = CamelResponseDecision(True, "No embedded output directives observed", text)
            self._record_response_decision(text, decision)
            return decision

        operator_request = self.current_plan.operator_request or ""
        if _OUTPUT_ANALYSIS_CONTEXT_RE.search(operator_request):
            decision = CamelResponseDecision(
                True,
                "Operator explicitly requested analysis or quotation of hidden content",
                text,
            )
            self._record_response_decision(text, decision)
            return decision

        normalized_request = _normalize_for_match(operator_request)
        matched: List[str] = []
        sanitized = text

        for marker in self.untrusted_output_markers:
            normalized_marker = _normalize_for_match(marker)
            if not normalized_marker or normalized_marker in normalized_request:
                continue
            if _response_starts_with_marker(sanitized, marker):
                matched.append(marker)
                sanitized = _strip_marker_from_response(sanitized, marker)

        sanitized = sanitized.strip()
        if not matched:
            decision = CamelResponseDecision(True, "No response hijack markers detected", text)
            self._record_response_decision(text, decision)
            return decision

        if not sanitized:
            sanitized = (
                "I detected hidden instructions embedded in untrusted content and ignored them. "
                "Use only the visible application information."
            )

        decision = CamelResponseDecision(
            False,
            "Blocked by CaMeL guard: assistant response echoed output directives from untrusted content",
            sanitized,
            matched_markers=matched,
        )
        self._record_response_decision(text, decision)
        return decision

    def wrap_tool_result(self, tool_name: str, content: str) -> tuple[str, bool]:
        if not self.config.enabled or self.config.mode == "off":
            return content, False
        if not is_untrusted_tool(tool_name):
            return content, False

        try:
            parsed = json.loads(content)
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            existing_guard = parsed.get("_camel_guard")
            if isinstance(existing_guard, dict) and existing_guard.get("blocked"):
                return content, False

        source = _extract_source_label(tool_name)
        flags = _extract_suspicious_flags(content)
        self._remember_untrusted_observation(source, flags, [])
        return content, False
