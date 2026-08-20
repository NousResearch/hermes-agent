"""Multi-platform delivery simulator — ``hermes gateway simulate-delivery``.

Runs input text through the REAL per-adapter ``format_message()`` renderer
and the shared ``BasePlatformAdapter.truncate_message()`` chunker — the
exact code paths ``gateway/delivery.py`` drives before a live send — so
rendering/chunking bugs (escaping, fence-splitting, oversized payloads)
surface without opening a socket or touching credentials.

Oracles are imported, never reimplemented — the render step reuses the
same unbound-``format_message`` trick as the merged conformance-vector
generator (``scripts/generate_conformance_vectors.py``): those methods are
asserted self-free by ``tests/conformance/test_vector_generator.py``, so
calling them with ``self=None`` (or a bare ``object.__new__`` instance for
the WhatsApp mixin) is safe and avoids constructing a real adapter (which
would need network config).

Scope: this simulates RENDERING and STATIC CHUNKING, not draft/streaming
diffs or wire-level SDK payloads — see the module docstring in the CLI
parser (``hermes_cli/subcommands/gateway.py``) for the exact contract.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "SUPPORTED_PLATFORMS",
    "DeliverySimulation",
    "simulate",
    "cmd_simulate_delivery",
]


@dataclass
class DeliverySimulation:
    """Result of rendering+chunking one message for one platform."""

    platform: str
    formatted: str
    length: int
    max_length: int
    exceeds_limit: bool
    splits_natively: bool
    chunks: Optional[List[str]]  # None when the platform has no native chunker


def _telegram_renderer() -> Dict[str, Any]:
    from gateway.platforms.base import utf16_len
    from plugins.platforms.telegram.adapter import TelegramAdapter

    return {
        "format": lambda text: TelegramAdapter.format_message(None, text),
        "max_length": TelegramAdapter.MAX_MESSAGE_LENGTH,
        "splits": TelegramAdapter.splits_long_messages,
        "len_fn": utf16_len,
        "truncate": TelegramAdapter.truncate_message,
    }


def _discord_renderer() -> Dict[str, Any]:
    from plugins.platforms.discord.adapter import DiscordAdapter

    return {
        "format": lambda text: DiscordAdapter.format_message(None, text),
        "max_length": DiscordAdapter.MAX_MESSAGE_LENGTH,
        "splits": DiscordAdapter.splits_long_messages,
        "len_fn": len,
        "truncate": DiscordAdapter.truncate_message,
    }


def _slack_renderer() -> Dict[str, Any]:
    from plugins.platforms.slack.adapter import SlackAdapter

    return {
        "format": lambda text: SlackAdapter.format_message(None, text),
        "max_length": SlackAdapter.MAX_MESSAGE_LENGTH,
        "splits": SlackAdapter.splits_long_messages,
        "len_fn": len,
        "truncate": SlackAdapter.truncate_message,
    }


def _whatsapp_renderer() -> Dict[str, Any]:
    from gateway.platforms.base import BasePlatformAdapter
    from gateway.platforms.whatsapp_common import WhatsAppBehaviorMixin

    # format_message needs no __init__ state — same construction the
    # conformance-vector generator uses (asserted by
    # tests/conformance/test_vector_generator.py).
    instance = object.__new__(WhatsAppBehaviorMixin)
    return {
        "format": instance.format_message,
        # WhatsAppBehaviorMixin doesn't set MAX_MESSAGE_LENGTH — real
        # adapters fall back to the literal 4096 documented in
        # BasePlatformAdapter.max_message_length_for_chat().
        "max_length": 4096,
        "splits": BasePlatformAdapter.splits_long_messages,
        "len_fn": len,
        "truncate": BasePlatformAdapter.truncate_message,
    }


_RENDERERS: Dict[str, Callable[[], Dict[str, Any]]] = {
    "telegram": _telegram_renderer,
    "discord": _discord_renderer,
    "slack": _slack_renderer,
    "whatsapp": _whatsapp_renderer,
}

SUPPORTED_PLATFORMS = tuple(sorted(_RENDERERS))


def simulate(platform: str, text: str) -> DeliverySimulation:
    """Render+chunk ``text`` for ``platform`` through the real adapter code."""
    if platform not in _RENDERERS:
        raise ValueError(
            f"unsupported platform {platform!r} — supported: "
            f"{', '.join(SUPPORTED_PLATFORMS)}"
        )
    spec = _RENDERERS[platform]()
    formatted = spec["format"](text)
    len_fn = spec["len_fn"]
    length = len_fn(formatted)
    max_length = spec["max_length"]
    exceeds = length > max_length
    splits = bool(spec["splits"])

    chunks: Optional[List[str]] = None
    if splits:
        chunks = spec["truncate"](formatted, max_length, len_fn=len_fn)

    return DeliverySimulation(
        platform=platform,
        formatted=formatted,
        length=length,
        max_length=max_length,
        exceeds_limit=exceeds,
        splits_natively=splits,
        chunks=chunks,
    )


def _format_human(sim: DeliverySimulation) -> str:
    lines = [f"Delivery simulation: {sim.platform}", ""]
    lines.append("--- formatted output ---")
    lines.append(sim.formatted)
    lines.append("--- end ---")
    lines.append("")
    lines.append(f"length: {sim.length} / {sim.max_length}")
    lines.append(f"exceeds limit: {'yes' if sim.exceeds_limit else 'no'}")
    lines.append(f"splits natively: {'yes' if sim.splits_natively else 'no'}")
    if sim.chunks is not None:
        lines.append(f"chunks: {len(sim.chunks)}")
        for idx, chunk in enumerate(sim.chunks, start=1):
            lines.append(f"  [{idx}/{len(sim.chunks)}] ({len(chunk)} chars)")
    elif sim.exceeds_limit:
        lines.append(
            "  this platform has no native chunker — oversized-content "
            "handling depends on the call path (e.g. cron truncates with "
            "a footer; other paths are adapter-specific)."
        )
    return "\n".join(lines)


def _format_json(sim: DeliverySimulation) -> str:
    return json.dumps(
        {
            "platform": sim.platform,
            "formatted": sim.formatted,
            "length": sim.length,
            "max_length": sim.max_length,
            "exceeds_limit": sim.exceeds_limit,
            "splits_natively": sim.splits_natively,
            "chunks": sim.chunks,
        },
        indent=2,
    )


def cmd_simulate_delivery(args: argparse.Namespace) -> int:
    platform = (getattr(args, "platform", "") or "").strip().lower()
    input_path = getattr(args, "input", None)
    text_arg = getattr(args, "text", None)

    if input_path:
        try:
            with open(input_path, encoding="utf-8") as f:
                text = f.read()
        except OSError as exc:
            print(f"error: cannot read {input_path}: {exc}", file=sys.stderr)
            return 1
    else:
        text = text_arg or ""

    try:
        sim = simulate(platform, text)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if getattr(args, "json", False):
        print(_format_json(sim))
    else:
        print(_format_human(sim))
    return 0
