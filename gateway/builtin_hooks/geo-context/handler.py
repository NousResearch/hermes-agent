"""
geo-context hook — fetches User Profile, Memory and Today blocks from Gabriel's
Geo app via the geo-mcp-bridge stdio MCP server, optionally summarizes Today
with Claude Haiku, and writes the result to ~/.hermes/memories/MEMORY.md so
hermes's own memory-injection picks it up when building system prompts.

Fires on agent:start (every turn) and session:reset. TTL+hash gated:
- Skip MCP entirely if last successful fetch was <TTL_SECONDS ago.
- Skip MEMORY.md rewrite if body hash is unchanged (preserves prompt cache).
Silently bails when Geo.app is closed (no socket).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

HERMES_HOME = Path(os.path.expanduser("~/.hermes"))
MEMORY_PATH = HERMES_HOME / "memories" / "MEMORY.md"
STATE_PATH = Path(__file__).parent / ".state.json"
GEO_BRIDGE_PATH = "/Users/biel/ARC/Forge/Geo/geo-mcp-bridge/geo-mcp-bridge"
HAIKU_MODEL = "claude-haiku-4-5"
TODAY_SUMMARIZE_THRESHOLD = 600
MAX_MEMORY_BODY = 4000
TTL_SECONDS = 60.0


def _log(msg: str) -> None:
    print(f"[geo-context] {msg}", flush=True)


def _extract_text(call_result) -> Optional[str]:
    if call_result is None:
        return None
    content = getattr(call_result, "content", None)
    if not content:
        return None
    parts = [c.text for c in content if hasattr(c, "text") and c.text]
    raw = "\n".join(parts).strip()
    return raw or None


def _unwrap_block(raw: Optional[str]) -> Optional[str]:
    """Geo's get_block_by_title returns a JSON envelope; pull `.markdown` and
    strip its frontmatter so we don't double-wrap. Returns None on miss
    sentinels ("No block found matching title: ...")."""
    if not raw:
        return None
    if raw.lstrip().lower().startswith("no block found"):
        return None
    try:
        obj = json.loads(raw)
        md = obj.get("markdown") or obj.get("body") or obj.get("content")
        if md:
            return _strip_frontmatter(md)
    except (json.JSONDecodeError, AttributeError):
        pass
    return _strip_frontmatter(raw)


def _format_today(raw: Optional[str]) -> Optional[str]:
    """Geo's get_today returns {block_ids, capture_count, id}; render as a
    one-line summary humans/LLMs can scan."""
    if not raw:
        return None
    try:
        obj = json.loads(raw)
        date = obj.get("id", "today")
        block_ids = obj.get("block_ids") or []
        capture_count = obj.get("capture_count", 0)
        parts = [f"date: {date}"]
        if block_ids:
            parts.append(f"linked blocks: {len(block_ids)}")
        if capture_count:
            parts.append(f"captures: {capture_count}")
        if len(parts) == 1:
            parts.append("nothing logged yet")
        return " · ".join(parts)
    except (json.JSONDecodeError, AttributeError):
        return raw


async def _safe_call(session: ClientSession, tool: str, args: dict) -> Optional[str]:
    try:
        result = await asyncio.wait_for(
            session.call_tool(tool, arguments=args),
            timeout=10.0,
        )
        return _extract_text(result)
    except Exception as e:
        _log(f"{tool}({args}) failed: {e}")
        return None


async def _fetch_geo_blocks():
    if not Path(GEO_BRIDGE_PATH).exists():
        _log(f"bridge binary missing at {GEO_BRIDGE_PATH}")
        return None, None, None
    params = StdioServerParameters(command=GEO_BRIDGE_PATH, args=[], env=None)
    try:
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), timeout=8.0)
                profile = await _safe_call(
                    session, "get_block_by_title", {"title": "User Profile"}
                )
                memory = await _safe_call(
                    session, "get_block_by_title", {"title": "Memory"}
                )
                today = await _safe_call(session, "get_today", {})
                return profile, memory, today
    except Exception as e:
        _log(f"bridge unreachable (Geo.app closed?): {e}")
        return None, None, None


async def _summarize_with_haiku(text: str, label: str) -> str:
    """Spawn `hermes -z` to summarize via Haiku. Uses the same auth as the
    running gateway so no API key juggling. Falls back to truncation."""
    hermes_bin = shutil.which("hermes") or os.path.expanduser("~/.local/bin/hermes")
    if not Path(hermes_bin).exists():
        return text[:1500]
    prompt = (
        f"Summarize {label} in <= 200 words. Be telegraphic, no preamble, "
        f"no markdown. Pure information density.\n\n{text}"
    )
    try:
        proc = await asyncio.create_subprocess_exec(
            hermes_bin,
            "-z", prompt,
            "-m", HAIKU_MODEL,
            "--provider", "anthropic",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=20.0)
        if proc.returncode == 0 and out:
            return out.decode("utf-8", errors="replace").strip()
        _log(f"haiku summarize exit {proc.returncode}: {err.decode()[:200]}")
    except Exception as e:
        _log(f"haiku summarize failed: {e}")
    return text[:1500]


def _strip_frontmatter(md: str) -> str:
    """Drop the leading YAML frontmatter from a markdown block, if present."""
    if not md.startswith("---"):
        return md.strip()
    parts = md.split("---", 2)
    return parts[2].strip() if len(parts) >= 3 else md.strip()


async def _build_body() -> Optional[str]:
    raw_profile, raw_memory, raw_today = await _fetch_geo_blocks()

    profile = _unwrap_block(raw_profile)
    memory = _unwrap_block(raw_memory)
    today = _format_today(raw_today)

    if not any([profile, memory, today]):
        return None

    sections: list[str] = [
        "<!-- auto-generated by hooks/geo-context — DO NOT edit by hand; "
        "edit the Soul/User-Profile/Memory blocks in Geo instead. -->\n"
    ]
    if profile:
        sections.append(f"## User profile\n\n{profile}")
    if memory:
        sections.append(f"## Memory\n\n{memory}")
    if today:
        body = today
        if len(body) > TODAY_SUMMARIZE_THRESHOLD:
            body = await _summarize_with_haiku(body, "today's day record")
        sections.append(f"## Today\n\n{body}")

    out = "\n\n".join(sections)
    if len(out) > MAX_MEMORY_BODY:
        out = out[:MAX_MEMORY_BODY].rsplit("\n", 1)[0] + "\n"
    return out


def _load_state() -> dict:
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state: dict) -> None:
    try:
        STATE_PATH.write_text(json.dumps(state), encoding="utf-8")
    except Exception as e:
        _log(f"state persist failed: {e}")


def _hash(body: str) -> str:
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


async def handle(event_type: str, context: dict) -> None:
    if event_type not in ("agent:start", "session:reset", "session:start"):
        return
    platform = context.get("platform", "?")
    state = _load_state()
    now = time.time()
    last_ts = float(state.get("last_fetch_ts") or 0)
    force = event_type == "session:reset"
    if not force and (now - last_ts) < TTL_SECONDS:
        return
    body = await _build_body()
    if not body:
        if event_type != "agent:start":
            _log(f"{event_type}: no Geo data fetched; leaving MEMORY.md untouched")
        return
    new_hash = _hash(body)
    if state.get("body_hash") == new_hash and MEMORY_PATH.exists():
        state["last_fetch_ts"] = now
        _save_state(state)
        return
    try:
        MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        MEMORY_PATH.write_text(body, encoding="utf-8")
        _log(f"{event_type} (platform={platform}) wrote {MEMORY_PATH.name} ({len(body)} chars)")
        _save_state({"last_fetch_ts": now, "body_hash": new_hash})
    except Exception as e:
        _log(f"write failed: {e}")
