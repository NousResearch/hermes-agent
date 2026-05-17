from __future__ import annotations

"""Opt-in Discord stock-research packet router.

This module is intentionally small and deterministic: it can generate a
Research Terminal instruction packet for configured Discord stock-research
channels, then inject the packet into the user message before the normal agent
loop starts. It does not generate reports, place trades, or change accounts.
"""

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_WORKDIR = "/Users/cai/Desktop/AI/research-terminal"
DEFAULT_TIMEOUT_SECONDS = 45
DEEP_PATTERNS = re.compile(
    r"\b(deep\s+(?:dive|research)|full\s+(?:memo|report)|institutional|equity\s+report|"
    r"initiating?\s+coverage|comprehensive)\b",
    re.IGNORECASE,
)
CASH_TICKER_RE = re.compile(r"\$([A-Z][A-Z0-9]{0,9}(?:\.[A-Z0-9]{1,4})?)\b")
EXCHANGE_TICKER_RE = re.compile(r"\b([A-Z0-9]{1,8}\.[A-Z0-9]{1,4})\b")
UPPER_TICKER_RE = re.compile(r"\b([A-Z]{2,6})\b")
AFTER_VERB_RE = re.compile(
    r"\b(?:research(?:\s+on)?|deep\s+dive|deep\s+research|look\s+at|on|for)\s+\$?([A-Za-z0-9][A-Za-z0-9._-]{0,24})\b",
    re.IGNORECASE,
)
STOPWORDS = {
    "A", "AN", "AND", "ARE", "CAN", "DEEP", "DIVE", "DO", "FOR", "FULL", "GO", "ME",
    "ON", "PLEASE", "RESEARCH", "STOCK", "THE", "THIS", "TO", "WHAT", "WITH",
}


@dataclass(frozen=True)
class StockResearchRouterConfig:
    enabled: bool = False
    channels: frozenset[str] = frozenset()
    workdir: str = DEFAULT_WORKDIR
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS


@dataclass(frozen=True)
class StockResearchRoute:
    topic: str
    mode: str
    command_alias: str
    command: tuple[str, ...]


def _coerce_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _split_ids(value: Any) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, (list, tuple, set)):
        parts = [str(v) for v in value]
    else:
        parts = re.split(r"[,\s]+", str(value))
    return frozenset(p.strip() for p in parts if p and p.strip())


def config_from_extra(extra: Mapping[str, Any] | None = None) -> StockResearchRouterConfig:
    extra = extra or {}
    nested = extra.get("stock_research_router") if isinstance(extra.get("stock_research_router"), Mapping) else {}

    enabled = os.getenv("DISCORD_STOCK_RESEARCH_ROUTER_ENABLED")
    if enabled is None:
        enabled = nested.get("enabled", extra.get("stock_research_router_enabled"))

    channels = os.getenv("DISCORD_STOCK_RESEARCH_CHANNELS")
    if channels is None:
        channels = nested.get("channels", extra.get("stock_research_router_channels"))

    workdir = os.getenv("DISCORD_STOCK_RESEARCH_WORKDIR")
    if workdir is None:
        workdir = nested.get("workdir", extra.get("stock_research_router_workdir", DEFAULT_WORKDIR))

    timeout_raw = os.getenv("DISCORD_STOCK_RESEARCH_TIMEOUT_SECONDS")
    if timeout_raw is None:
        timeout_raw = nested.get("timeout_seconds", extra.get("stock_research_router_timeout_seconds", DEFAULT_TIMEOUT_SECONDS))
    try:
        timeout_seconds = max(1, int(timeout_raw))
    except (TypeError, ValueError):
        timeout_seconds = DEFAULT_TIMEOUT_SECONDS

    return StockResearchRouterConfig(
        enabled=_coerce_bool(enabled, False),
        channels=_split_ids(channels),
        workdir=str(workdir or DEFAULT_WORKDIR),
        timeout_seconds=timeout_seconds,
    )


def should_route(config: StockResearchRouterConfig, channel_ids: Sequence[str | None], is_dm: bool = False) -> bool:
    if not config.enabled or is_dm:
        return False
    ids = {str(v) for v in channel_ids if v}
    return "*" in config.channels or bool(ids & set(config.channels))


def classify_mode(text: str) -> str:
    return "deep" if DEEP_PATTERNS.search(text or "") else "standard"


def extract_topic(text: str) -> str | None:
    text = (text or "").strip()
    if not text:
        return None
    for pattern in (CASH_TICKER_RE, EXCHANGE_TICKER_RE):
        match = pattern.search(text)
        if match:
            return match.group(1).upper()
    match = AFTER_VERB_RE.search(text)
    if match:
        candidate = match.group(1).strip(".,:;!?()[]{}\"'")
        if candidate and candidate.upper() not in STOPWORDS:
            return candidate.upper() if re.fullmatch(r"[A-Za-z0-9.]+", candidate) else candidate
    for match in UPPER_TICKER_RE.finditer(text):
        candidate = match.group(1).upper()
        if candidate not in STOPWORDS:
            return candidate
    return None


def route_for_text(text: str) -> StockResearchRoute | None:
    topic = extract_topic(text)
    if not topic:
        return None
    mode = classify_mode(text)
    command_alias = "research-deep" if mode == "deep" else "research"
    return StockResearchRoute(
        topic=topic,
        mode=mode,
        command_alias=command_alias,
        command=("npm", "--silent", "run", "rt", "--", command_alias, topic),
    )


def _packet_subset(packet: Mapping[str, Any]) -> dict[str, Any]:
    keys = [
        "kind", "version", "mode", "topic", "packet_path", "prompt_files", "freshness_guard",
        "preflight_commands", "output", "routing", "execution", "central_wiki_context", "safety",
    ]
    return {key: packet[key] for key in keys if key in packet}


def generate_packet(route: StockResearchRoute, config: StockResearchRouterConfig) -> tuple[dict[str, Any], str | None]:
    workdir = Path(config.workdir).expanduser()
    if not workdir.exists():
        raise FileNotFoundError(f"Research Terminal workdir not found: {workdir}")
    result = subprocess.run(
        list(route.command),
        cwd=str(workdir),
        text=True,
        capture_output=True,
        timeout=config.timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        stderr = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"rt packet command failed ({result.returncode}): {stderr[:1000]}")
    packet = json.loads(result.stdout)
    return packet, getattr(result, "stderr", None)


def build_injected_text(original_text: str, route: StockResearchRoute, packet: Mapping[str, Any]) -> str:
    compact_packet = json.dumps(_packet_subset(packet), indent=2, sort_keys=True)
    command = " ".join(route.command)
    return (
        "[Deterministic stock-research packet generated by Discord router]\n"
        f"Command: {command}\n"
        "Scope: packet/instruction only; no report generation, trades, transfers, or account actions.\n"
        "Packet JSON:\n"
        f"```json\n{compact_packet}\n```\n\n"
        "Original user request:\n"
        f"{original_text}"
    )


def maybe_build_injected_text(
    text: str,
    channel_ids: Sequence[str | None],
    extra: Mapping[str, Any] | None = None,
    is_dm: bool = False,
) -> str | None:
    config = config_from_extra(extra)
    if not should_route(config, channel_ids, is_dm=is_dm):
        return None
    route = route_for_text(text)
    if not route:
        return None
    packet, _ = generate_packet(route, config)
    return build_injected_text(text, route, packet)
