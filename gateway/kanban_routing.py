"""Slack/Kanban routing helpers.

Pure helpers for the gateway-side L2 routing contract:
- normalize inbound platform origins into stable routing keys;
- resolve a key against a read-only routing map;
- build Kanban creation metadata that preserves origin/report-to fields;
- block protected-scope requests before worker dispatch.

The module intentionally performs no network calls, service control, or live DB
writes. Callers pass the returned request into the existing Kanban APIs.
"""

from __future__ import annotations

import json
import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from hermes_constants import get_hermes_home

DEFAULT_ROUTING_MAP_RELATIVE_PATH = Path("kanban") / "routing_map.json"
PROTECTED_ROUTE_STATUS = "blocked"
DEFAULT_ROUTE_STATUS = "running"
STAGE0_RECORD_ONLY_STATUS = "blocked"

_BOUNDARY_MARKERS = (
    "금지",
    "제외",
    "하지 마",
    "하지말",
    "하지 않",
    "미수행",
    "안 함",
    "not_performed",
    "forbidden",
    "excluded",
    "do not",
    "don't",
    "no ",
    "read-only",
    "dry-run",
)

_DISPATCH_REQUEST_PATTERN = re.compile(
    r"\b(?:dispatch|spawn|worker|queue|ready|promote|unblock)\b|"
    r"디스패치|워커|큐|대기열|작업자|작업자\s*실행|실행\s*대상|ready|"
    r"준비\s*상태|승격|언블록",
    re.IGNORECASE,
)

_SECRET_VALUE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:sk|sk-proj|xox[baprs]?|gh[pousr]|glpat|hf)[_-][A-Za-z0-9_\-]{12,}\b"),
    re.compile(r"\b[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{20,}\b"),
)

_CLAUSE_SPLIT_PATTERN = re.compile(
    r"[\n\r.;；。!?！？]+|(?:\s+그리고\s+)|(?:\s+단,?\s+)|(?:\s+but\s+)",
    re.IGNORECASE,
)
_BOUNDARY_ZONE_PATTERN = re.compile(
    r"(?:금지\s*(?:범위|대상|항목)|제외\s*(?:범위|대상|항목)|"
    r"forbidden\s*(?:scope|items?|boundary)|excluded\s*(?:scope|items?|boundary))\s*[:：]",
    re.IGNORECASE,
)
_CONTRAST_EXECUTION_PATTERN = re.compile(
    r"말고|대신|but|however|except|except\s+for|run\s+|execute\s+|해줘|진행|수행|접속|"
    r"write|migration|restart|start|stop|dispatch|worker|promote|ready",
    re.IGNORECASE,
)

_PROTECTED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("m1", re.compile(r"\b(?:M1|macmini|macmini-ts|yk-start)\b", re.IGNORECASE)),
    ("kcc_source", re.compile(r"\bKCC source\b|dashboard/src", re.IGNORECASE)),
    ("gateway_service", re.compile(r"\bgateway\s+(?:restart|stop|start)\b|LaunchAgent|launchctl|plist", re.IGNORECASE)),
    ("git_push", re.compile(r"\bgit\s+push\b|force\s+push|reset\s+--hard", re.IGNORECASE)),
    ("live_config", re.compile(r"\blive\s+config\b|config\s+(?:change|apply|write|edit|반영|변경)", re.IGNORECASE)),
    ("slack_permission", re.compile(r"Slack\s+(?:permission|scope|권한)|channels:manage|groups:read", re.IGNORECASE)),
    ("db_runtime", re.compile(r"\b(?:db|database)\s+(?:write|migration|schema|runtime|수정|쓰기|마이그레이션)\b|DB\s*(?:write|수정|삭제)", re.IGNORECASE)),
    ("trading_runtime", re.compile(r"\b(?:order|account|backtest|collector|sync|trading)\b|주문|계좌|백테스트", re.IGNORECASE)),
    ("secret_plaintext", re.compile(r"\b(?:secret|token|password)\b|비밀번호|토큰|\.env", re.IGNORECASE)),
)


@dataclass(frozen=True)
class RoutingOrigin:
    platform: str
    chat_id: str
    thread_id: Optional[str] = None
    chat_name: Optional[str] = None
    chat_type: Optional[str] = None
    user_id: Optional[str] = None
    user_name: Optional[str] = None

    def to_metadata(self) -> dict[str, str]:
        data = {
            "platform": self.platform,
            "chat_id": self.chat_id,
        }
        optional = {
            "thread_id": self.thread_id,
            "chat_name": self.chat_name,
            "chat_type": self.chat_type,
            "user_id": self.user_id,
            "user_name": self.user_name,
        }
        for key, value in optional.items():
            if value is not None and str(value) != "":
                data[key] = str(value)
        return data

    def report_to_metadata(self) -> dict[str, str]:
        data = {
            "platform": self.platform,
            "chat_id": self.chat_id,
        }
        if self.thread_id:
            data["thread_id"] = self.thread_id
        return data


@dataclass(frozen=True)
class RouteResolution:
    matched_key: Optional[str]
    board: Optional[str]
    anchor_task_id: Optional[str] = None
    route: Mapping[str, Any] = field(default_factory=dict)

    @property
    def found(self) -> bool:
        return bool(self.matched_key and self.board)


@dataclass(frozen=True)
class RoutedKanbanRequest:
    title: str
    body: str
    board: Optional[str]
    parents: tuple[str, ...]
    initial_status: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class Stage0RecordOnlyDecision:
    """Pure record-only/fail-closed decision for AI Staff OS stage0.

    This object is intentionally side-effect-free: no card creation, no
    dispatcher calls, no queue promotion, no network, and no secret-bearing raw
    text in metadata. Gateway callers may log or persist the sanitized metadata
    through an already-approved record-only channel.
    """

    decision: str
    route_key: str
    matched_key: Optional[str]
    board: Optional[str]
    anchor_task_id: Optional[str]
    initial_status: str
    dispatch_allowed: bool
    ready_status_allowed: bool
    protected_terms: tuple[str, ...]
    redactions_applied: bool
    idempotency_key: str
    reason: str
    sanitized_preview: str

    def to_metadata(self) -> dict[str, Any]:
        return {
            "version": "ai-staff-os-stage0-record-only.v1",
            "decision": self.decision,
            "route_key": self.route_key,
            "matched_key": self.matched_key,
            "board": self.board,
            "anchor_task_id": self.anchor_task_id,
            "initial_status": self.initial_status,
            "dispatch_allowed": self.dispatch_allowed,
            "ready_status_allowed": self.ready_status_allowed,
            "protected_terms": list(self.protected_terms),
            "redactions_applied": self.redactions_applied,
            "idempotency_key": self.idempotency_key,
            "reason": self.reason,
            "sanitized_preview": self.sanitized_preview,
        }


def _platform_value(value: Any) -> str:
    raw = getattr(value, "value", value)
    return str(raw or "").strip().lower()


def origin_from_source(source: Any) -> RoutingOrigin:
    """Coerce a SessionSource-like object or dict into a routing origin."""
    if isinstance(source, Mapping):
        getter = source.get
    else:
        getter = lambda key, default=None: getattr(source, key, default)

    platform = _platform_value(getter("platform"))
    chat_id = str(getter("chat_id") or "").strip()
    if not platform:
        raise ValueError("source platform is required")
    if not chat_id:
        raise ValueError("source chat_id is required")

    def opt(name: str) -> Optional[str]:
        value = getter(name)
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    return RoutingOrigin(
        platform=platform,
        chat_id=chat_id,
        thread_id=opt("thread_id"),
        chat_name=opt("chat_name"),
        chat_type=opt("chat_type"),
        user_id=opt("user_id"),
        user_name=opt("user_name"),
    )


def normalize_routing_key(source: Any) -> str:
    """Return `platform:chat_id[:thread_id]` for a SessionSource-like origin."""
    origin = origin_from_source(source)
    if origin.thread_id:
        return f"{origin.platform}:{origin.chat_id}:{origin.thread_id}"
    return f"{origin.platform}:{origin.chat_id}"


def _candidate_keys(origin: RoutingOrigin) -> tuple[str, ...]:
    channel_key = f"{origin.platform}:{origin.chat_id}"
    if origin.thread_id:
        return (f"{channel_key}:{origin.thread_id}", channel_key)
    return (channel_key,)


def load_routing_map(path: str | Path | None = None) -> dict[str, Any]:
    """Load a routing map JSON file without creating or modifying files."""
    routing_path = Path(path) if path is not None else get_hermes_home() / DEFAULT_ROUTING_MAP_RELATIVE_PATH
    if not routing_path.exists():
        return {"routes": {}}
    with routing_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("routing map must be a JSON object")
    routes = data.get("routes")
    if routes is None:
        data["routes"] = {}
    elif not isinstance(routes, dict):
        raise ValueError("routing map routes must be a JSON object")
    return data


def resolve_route(source: Any, routing_map: Mapping[str, Any]) -> RouteResolution:
    """Resolve the most specific route for a source.

    Thread-specific keys win over channel-level fallback keys.
    """
    origin = origin_from_source(source)
    routes = routing_map.get("routes", {}) if isinstance(routing_map, Mapping) else {}
    if not isinstance(routes, Mapping):
        raise ValueError("routing map routes must be a mapping")
    for key in _candidate_keys(origin):
        route = routes.get(key)
        if not isinstance(route, Mapping):
            continue
        board = str(route.get("board") or "").strip() or None
        anchor = str(route.get("anchor_task_id") or "").strip() or None
        if board:
            return RouteResolution(
                matched_key=key,
                board=board,
                anchor_task_id=anchor,
                route=route,
            )
    return RouteResolution(matched_key=None, board=None)


def protected_scope_matches(*texts: str) -> tuple[str, ...]:
    haystack = "\n".join(t for t in texts if t)
    if not haystack:
        return ()
    matches: list[str] = []
    for label, pattern in _PROTECTED_PATTERNS:
        if pattern.search(haystack):
            matches.append(label)
    return tuple(matches)


def _mentions_boundary(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _BOUNDARY_MARKERS)


def _clauses(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in _CLAUSE_SPLIT_PATTERN.split(text or "") if part.strip())


def _match_in_boundary_zone(clause: str, match_start: int) -> bool:
    """Return True when a match is inside an explicit boundary-list zone."""
    for zone in _BOUNDARY_ZONE_PATTERN.finditer(clause):
        if zone.start() <= match_start:
            return True
    return False


def _match_has_local_boundary(clause: str, match: re.Match[str]) -> bool:
    """Return True only when the boundary marker locally scopes the match.

    Explicit list headers such as "금지 범위:" cover the rest of that sentence.
    Otherwise a denial marker must be close to the protected/dispatch term, and
    contrast/execution language between the marker and the match fails closed.
    This prevents inputs like "M1 하지 말고 DB write 해" or
    "do not use M1, run DB migration" from being globally allowed.
    """
    if _match_in_boundary_zone(clause, match.start()):
        return True

    lowered = clause.lower()
    marker_spans: list[tuple[int, int]] = []
    for marker in _BOUNDARY_MARKERS:
        start = 0
        needle = marker.lower()
        while needle and (idx := lowered.find(needle, start)) != -1:
            marker_spans.append((idx, idx + len(needle)))
            start = idx + len(needle)

    for start, end in marker_spans:
        distance = min(abs(match.start() - end), abs(start - match.end()))
        if distance > 32:
            continue
        bridge = clause[min(end, match.end()) : max(start, match.start())]
        if _CONTRAST_EXECUTION_PATTERN.search(bridge):
            continue
        return True
    return False


def _pattern_is_boundary_scoped(text: str, pattern: re.Pattern[str]) -> bool:
    """Return True only when every match is locally boundary-scoped."""
    found = False
    for clause in _clauses(text):
        for match in pattern.finditer(clause):
            found = True
            if not _match_has_local_boundary(clause, match):
                return False
    return found


def _protected_terms_boundary_scoped(text: str, terms: Sequence[str]) -> bool:
    if not terms:
        return False
    patterns = {label: pattern for label, pattern in _PROTECTED_PATTERNS}
    return all(_pattern_is_boundary_scoped(text, patterns[label]) for label in terms if label in patterns)


def redact_stage0_record_text(text: str, *, max_preview_chars: int = 240) -> tuple[str, bool]:
    """Return a short redacted preview safe for record-only metadata."""
    preview = (text or "").replace("\x00", " ").strip()
    redacted = False
    for pattern in _SECRET_VALUE_PATTERNS:
        preview, count = pattern.subn("[REDACTED]", preview)
        redacted = redacted or bool(count)
    if len(preview) > max_preview_chars:
        preview = preview[: max_preview_chars - 1].rstrip() + "…"
    return preview, redacted


def build_stage0_record_only_decision(
    *,
    source: Any,
    text: str,
    routing_map: Mapping[str, Any],
    existing_idempotency_keys: Sequence[str] | None = None,
) -> Stage0RecordOnlyDecision:
    """Classify an inbound instruction for stage0 record-only handling.

    The helper is pure and fail-closed. It never creates cards, never returns a
    `ready` status, and never permits dispatch. Protected words that appear in
    explicit forbidden/read-only/dry-run boundary text are recorded as safe
    boundary mentions instead of being treated as execution intent.
    """
    origin = origin_from_source(source)
    route_key = normalize_routing_key(origin)
    route = resolve_route(origin, routing_map)
    sanitized_preview, redactions_applied = redact_stage0_record_text(text)
    idempotency_key = "sha256:" + hashlib.sha256(
        "\x1f".join(
            [
                "ai-staff-os-stage0-record-only.v1",
                route_key,
                route.matched_key or "",
                route.board or "",
                sanitized_preview,
            ]
        ).encode("utf-8")
    ).hexdigest()
    existing = set(existing_idempotency_keys or [])
    protected_terms = protected_scope_matches(text)
    protected_boundary_scoped = _protected_terms_boundary_scoped(text, protected_terms)
    dispatch_requested = bool(_DISPATCH_REQUEST_PATTERN.search(text or ""))
    dispatch_boundary_scoped = (
        _pattern_is_boundary_scoped(text, _DISPATCH_REQUEST_PATTERN)
        if dispatch_requested
        else False
    )

    decision = "PASS_RECORD_ONLY"
    reason = "record_only_no_dispatch"
    if not route.found or not route.anchor_task_id:
        decision = "BLOCKED_REFERENCE_CANDIDATE"
        reason = "missing_route_or_anchor"
    elif redactions_applied:
        decision = "BLOCK_SECRET"
        reason = "secret_like_value_redacted"
    elif "secret_plaintext" in protected_terms and not protected_boundary_scoped:
        decision = "BLOCK_SECRET"
        reason = "secret_plaintext_request"
    elif dispatch_requested and not dispatch_boundary_scoped:
        decision = "BLOCK_DISPATCH"
        reason = "dispatch_or_worker_activation_request"
    elif protected_terms and not protected_boundary_scoped:
        decision = "BLOCK_PROTECTED"
        reason = "protected_scope_execution_request"
    elif idempotency_key in existing:
        decision = "NO_DUPLICATE"
        reason = "duplicate_idempotency_key"
    elif protected_terms and protected_boundary_scoped:
        reason = "protected_terms_only_in_boundary_text"

    return Stage0RecordOnlyDecision(
        decision=decision,
        route_key=route_key,
        matched_key=route.matched_key,
        board=route.board,
        anchor_task_id=route.anchor_task_id,
        initial_status=STAGE0_RECORD_ONLY_STATUS,
        dispatch_allowed=False,
        ready_status_allowed=False,
        protected_terms=protected_terms,
        redactions_applied=redactions_applied,
        idempotency_key=idempotency_key,
        reason=reason,
        sanitized_preview=sanitized_preview,
    )


def build_routed_kanban_request(
    *,
    title: str,
    body: str,
    source: Any,
    routing_map: Mapping[str, Any],
) -> RoutedKanbanRequest:
    """Build the Kanban creation contract for a routed Slack/gateway request."""
    if not title or not title.strip():
        raise ValueError("title is required")
    origin = origin_from_source(source)
    route = resolve_route(origin, routing_map)
    matched_terms = protected_scope_matches(title, body)
    protected = bool(matched_terms)
    initial_status = PROTECTED_ROUTE_STATUS if protected else DEFAULT_ROUTE_STATUS
    parents = (route.anchor_task_id,) if route.anchor_task_id else ()

    metadata: dict[str, Any] = {
        "origin": origin.to_metadata(),
        "report_to": origin.report_to_metadata(),
        "routing": {
            "version": "slack-kanban-routing.v1",
            "matched_key": route.matched_key,
            "board": route.board,
            "anchor_task_id": route.anchor_task_id,
            "protected_scope": protected,
        },
    }
    if matched_terms:
        metadata["routing"]["matched_terms"] = list(matched_terms)

    return RoutedKanbanRequest(
        title=title.strip(),
        body=body or "",
        board=route.board,
        parents=parents,
        initial_status=initial_status,
        metadata=metadata,
    )


def _has_option(tokens: Sequence[str], name: str) -> bool:
    return name in tokens or any(token.startswith(f"{name}=") for token in tokens)


def _set_option_value(tokens: list[str], name: str, value: str) -> list[str]:
    """Set an existing option value or append it when absent.

    This is used for protected-scope fail-closed handling: if the user passed
    ``--initial-status running`` but the text mentions a protected operation,
    the gateway create path must force the single status flag to ``blocked``
    instead of appending a second flag that relies on argparse ordering.
    """
    updated = list(tokens)
    for idx, token in enumerate(updated):
        if token == name:
            if idx + 1 < len(updated):
                updated[idx + 1] = value
            else:
                updated.append(value)
            return updated
        if token.startswith(f"{name}="):
            updated[idx] = f"{name}={value}"
            return updated
    return [*updated, name, value]


def _merge_metadata_option(tokens: list[str], metadata: Mapping[str, Any]) -> list[str]:
    merged_json = json.dumps(dict(metadata), ensure_ascii=False)
    for idx, token in enumerate(tokens):
        existing_raw: str | None = None
        if token == "--metadata" and idx + 1 < len(tokens):
            existing_raw = tokens[idx + 1]
            target_idx = idx + 1
            prefix = None
        elif token.startswith("--metadata="):
            existing_raw = token.split("=", 1)[1]
            target_idx = idx
            prefix = "--metadata="
        else:
            continue
        try:
            existing = json.loads(existing_raw)
        except json.JSONDecodeError:
            # Preserve CLI validation: do not hide malformed user metadata by
            # appending a later valid value.
            return tokens
        if not isinstance(existing, dict):
            return tokens
        merged = {**existing, **dict(metadata)}
        merged_json = json.dumps(merged, ensure_ascii=False)
        updated = list(tokens)
        updated[target_idx] = f"{prefix}{merged_json}" if prefix else merged_json
        return updated
    return [*tokens, "--metadata", merged_json]


def route_kanban_create_tokens(
    tokens: Sequence[str],
    *,
    source: Any,
    routing_map: Mapping[str, Any],
) -> tuple[list[str], RoutedKanbanRequest | None]:
    """Inject routing options into a `/kanban create` token vector.

    Returns the original tokens when the command is not a create command, a
    caller already specified `--board`, or no route matches. The returned token
    list is suitable for `hermes_cli.kanban.run_slash` via `shlex.join`.
    """
    routed = list(tokens)
    if not routed:
        return routed, None
    explicit_board = "--board" in routed or any(t.startswith("--board=") for t in routed)
    if routed[0] != "create":
        return routed, None

    request = build_routed_kanban_request(
        title=" ".join(routed[1:]) or "kanban create",
        body=" ".join(routed),
        source=source,
        routing_map=routing_map,
    )
    protected = bool(request.metadata.get("routing", {}).get("protected_scope"))
    if explicit_board and not protected:
        return routed, None
    if not request.board and not protected:
        return routed, None

    injected = ([] if explicit_board else (["--board", request.board] if request.board else [])) + [*routed]
    if not explicit_board:
        for parent in request.parents:
            injected.extend(["--parent", parent])
    if protected:
        injected = _set_option_value(injected, "--initial-status", PROTECTED_ROUTE_STATUS)
    elif not _has_option(injected, "--initial-status"):
        injected.extend(["--initial-status", request.initial_status])
    injected = _merge_metadata_option(injected, request.metadata)
    return injected, request
