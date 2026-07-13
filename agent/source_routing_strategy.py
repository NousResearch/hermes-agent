"""Evidence-backed source-first routing strategy.

The strategy is selected at the turn boundary when the user's message contains
a URL for a domain with a native Hermes path.  Selection never changes the
model, tool schema, or system prompt during an active turn.  The first executed
tool supplies production evidence for the strategy registry; once promoted,
its guidance is injected when a new agent session is constructed.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Optional
from urllib.parse import parse_qs, urlparse

logger = logging.getLogger(__name__)

STRATEGY_NAME = "source-first-routing"
STRATEGY_DESCRIPTION = (
    "Use a domain-native tool before generic search when the user provides "
    "an exact URL."
)
STRATEGY_GUIDANCE = (
    "When the user provides an exact URL, use the domain-native integration "
    "first. Prefer native Git/GitHub tools for github.com, transcript or media "
    "extraction for youtube.com and youtu.be, and xurl for x.com and twitter.com. "
    "Do not search for a target "
    "whose exact URL is already available."
)

_ROUTABLE_URL = re.compile(
    r"(?P<url>https?://(?:www\.)?"
    r"(?P<domain>x\.com|twitter\.com|youtube\.com|youtu\.be|github\.com)/"
    r"[^\s<>\[\]\"']*)",
    re.IGNORECASE,
)
_GENERIC_SEARCH_TOOLS = frozenset({"web_search", "browser_navigate"})
_GUIDANCE_BLOCK = re.compile(
    r"\n*<strategy-guidance>.*?</strategy-guidance>\s*",
    re.DOTALL,
)


def _target_for_message(user_message: Any) -> tuple[Optional[str], Optional[str]]:
    if not isinstance(user_message, str):
        return None, None
    match = _ROUTABLE_URL.search(user_message)
    if not match:
        return None, None
    return match.group("domain").lower(), match.group("url").rstrip(".,);]")


def _target_identifiers(domain: str, target_url: str) -> tuple[str, ...]:
    parsed = urlparse(target_url)
    parts = [part for part in parsed.path.split("/") if part]
    identifiers = {target_url.lower()}
    if domain == "github.com" and len(parts) >= 2:
        identifiers.add(f"{parts[0]}/{parts[1]}".lower())
    elif domain in {"x.com", "twitter.com"} and "status" in parts:
        status_index = parts.index("status")
        if status_index + 1 < len(parts):
            identifiers.add(parts[status_index + 1].lower())
    elif domain == "youtu.be" and parts:
        identifiers.add(parts[0].lower())
    elif domain == "youtube.com":
        video_id = (parse_qs(parsed.query).get("v") or [None])[0]
        if video_id:
            identifiers.add(video_id.lower())
    return tuple(identifier for identifier in identifiers if identifier)


def _github_structured_target(target_url: str, args: dict[str, Any]) -> bool:
    """Match conventional GitHub MCP owner/repo and issue/PR arguments."""
    parts = [part for part in urlparse(target_url).path.split("/") if part]
    if len(parts) < 2:
        return False
    if str(args.get("owner") or "").lower() != parts[0].lower():
        return False
    repo = str(args.get("repo") or args.get("repository") or "").lower()
    if repo != parts[1].lower():
        return False
    if len(parts) >= 4 and parts[2] in {"pull", "pulls"}:
        return str(args.get("pull_number") or args.get("number") or "") == parts[3]
    if len(parts) >= 4 and parts[2] in {"issue", "issues"}:
        return str(args.get("issue_number") or args.get("number") or "") == parts[3]
    return True


def _uses_native_route(
    domain: str,
    target_url: str,
    tool_name: str,
    tool_args: Any,
) -> bool:
    """Return whether the first tool concretely matches the domain route."""
    name = str(tool_name or "").lower()
    args = tool_args if isinstance(tool_args, dict) else {}
    command = str(args.get("command") or args.get("cmd") or "").lower()
    args_text = json.dumps(args, sort_keys=True).lower()
    targets = _target_identifiers(domain, target_url)
    targets_targeted = any(target in args_text for target in targets)
    if domain == "github.com":
        github_tool = name.startswith("mcp_github_") or name.startswith("github_")
        if github_tool:
            return targets_targeted or _github_structured_target(target_url, args)
        return targets_targeted and name == "terminal" and (
            re.search(r"(?:^|[;&|]\s*)(?:gh|git)(?:\s|$)", command) is not None
        )
    if domain in {"x.com", "twitter.com"}:
        return targets_targeted and (
            name in {"xurl", "xurl_read"}
            or (
                name == "terminal"
                and re.search(r"(?:^|[;&|]\s*)xurl\s+read(?:\s|$)", command)
                is not None
            )
        )
    if domain in {"youtube.com", "youtu.be"}:
        return targets_targeted and (
            name in {"youtube_transcript", "youtube_content", "web_extract"}
            or (
                name == "terminal"
                and re.search(
                    r"(?:^|[;&|]\s*)(?:yt-dlp|youtube-transcript(?:-api)?|"
                    r"python(?:3)?\s+\S*fetch_transcript\.py)(?:\s|$)",
                    command,
                )
                is not None
            )
        )
    return False


def routable_domain(user_message: Any) -> Optional[str]:
    """Return the native-routing domain in *user_message*, if any."""
    domain, _target_url = _target_for_message(user_message)
    return domain


def register_source_routing_strategy(session_db: Any) -> None:
    """Idempotently register the concrete production strategy."""
    if session_db is None:
        return
    try:
        session_db.register_strategy(
            STRATEGY_NAME,
            description=STRATEGY_DESCRIPTION,
            strategy_type="routing",
            task_class="direct_url",
            config={"guidance": STRATEGY_GUIDANCE},
        )
    except Exception:
        logger.debug("source-first strategy registration failed", exc_info=True)


def evaluate_source_routing_promotion(session_db: Any) -> None:
    """Apply promotion gates at the cache-safe new-session boundary."""
    if session_db is None:
        return
    try:
        session_db.promote_strategy(STRATEGY_NAME)
    except Exception:
        logger.debug("source-first promotion evaluation failed", exc_info=True)


def compose_strategy_prompt(existing: Optional[str], guidance: Optional[str]) -> Optional[str]:
    """Replace the generated block so inherited prompts cannot duplicate it."""
    base = _GUIDANCE_BLOCK.sub("\n", existing or "").strip()
    if not guidance:
        return base or None
    block = f"<strategy-guidance>\n{guidance}\n</strategy-guidance>"
    return "\n\n".join(part for part in (base, block) if part)


def strategy_startup_guidance(session_db: Any) -> Optional[str]:
    """Return stable startup guidance for registered runtime strategies.

    Candidate guidance is what makes the strategy genuinely active while its
    evidence is collected. Promotion keeps the same behavior but changes its
    status from experiment to evidence-backed default for future sessions.
    """
    if session_db is None:
        return None
    try:
        source_strategy = session_db.get_strategy(STRATEGY_NAME)
        promoted = session_db.get_promoted_strategies()
    except Exception:
        logger.debug("strategy guidance lookup failed", exc_info=True)
        return None

    lines = []
    strategies = []
    if source_strategy:
        strategies.append(source_strategy)
    strategies.extend(
        strategy for strategy in promoted
        if strategy.get("name") != STRATEGY_NAME
    )
    for strategy in strategies:
        config = strategy.get("config_json")
        try:
            parsed = json.loads(config) if isinstance(config, str) else (config or {})
        except (TypeError, ValueError):
            parsed = {}
        guidance = str(parsed.get("guidance") or "").strip()
        if guidance:
            lines.append(f"- {strategy['name']}: {guidance}")
    if not lines:
        return None
    header = (
        "# Evidence-backed strategies"
        if source_strategy and source_strategy.get("state") == "promoted"
        else "# Candidate strategy under evaluation"
    )
    return header + "\n" + "\n".join(lines)


def select_for_turn(agent: Any, user_message: Any) -> None:
    """Select source-first routing once, before the turn's first API call."""
    domain, target_url = _target_for_message(user_message)
    if domain:
        try:
            session_db = getattr(agent, "_session_db", None)
            if session_db is None or session_db.get_strategy(STRATEGY_NAME) is None:
                domain = None
        except Exception:
            domain = None
    agent._active_strategy = STRATEGY_NAME if domain else None
    agent._active_strategy_task_class = "direct_url" if domain else None
    agent._active_strategy_metadata = (
        {"url_domain": domain, "target_url": target_url} if domain else {}
    )
    agent._active_strategy_evidence_pending = bool(domain)
    agent._active_strategy_tool_index = 0


def record_first_tool_evidence(
    agent: Any,
    *,
    tool_name: str,
    tool_args: Any,
    duration_seconds: float,
    is_error: bool,
) -> None:
    """Record exactly the first executed tool for the selected strategy."""
    if not getattr(agent, "_active_strategy_evidence_pending", False):
        return
    # Consume before persistence so a failed telemetry write cannot cause a
    # later, unrelated tool to be counted as the routing decision.
    agent._active_strategy_evidence_pending = False

    strategy = getattr(agent, "_active_strategy", None)
    session_db = getattr(agent, "_session_db", None)
    if not strategy or session_db is None:
        return

    active_metadata = getattr(agent, "_active_strategy_metadata", {}) or {}
    domain = str(active_metadata.get("url_domain") or "")
    target_url = str(active_metadata.get("target_url") or "")
    routing_miss = tool_name in _GENERIC_SEARCH_TOOLS
    native_route = _uses_native_route(domain, target_url, tool_name, tool_args)
    result = "success" if native_route and not is_error else "failure"
    metadata = dict(getattr(agent, "_active_strategy_metadata", {}) or {})
    metadata.update({
        "routing_miss": routing_miss,
        "native_route": native_route,
        "tool_called": tool_name,
        "tool_index": 0,
    })
    try:
        session_db.record_strategy_event(
            session_id=getattr(agent, "session_id", "") or "",
            event_type="tool_call",
            tool_name=tool_name,
            strategy=strategy,
            task_class=getattr(agent, "_active_strategy_task_class", None),
            result=result,
            latency_ms=round(max(duration_seconds, 0.0) * 1000, 1),
            metadata=metadata,
        )
    except Exception:
        logger.debug("source-first strategy telemetry failed", exc_info=True)


def discard_first_tool_evidence(agent: Any) -> None:
    """Consume a cancelled turn's evidence slot without scoring an outcome."""
    agent._active_strategy_evidence_pending = False
