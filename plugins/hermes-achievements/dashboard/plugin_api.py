"""Hermes Achievements dashboard plugin backend, mounted at /api/plugins/hermes-achievements/.

Scans the session history into per-session stats (checkpointed by fingerprint so warm
scans are cheap), aggregates them, and evaluates the tiered / multi-condition catalog.
Cold scans run on a background thread; ``/achievements`` serves the last snapshot.
"""
from __future__ import annotations

import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from fastapi import APIRouter

from hermes_constants import get_hermes_home

router = APIRouter()

SNAPSHOT_TTL_SECONDS = 120
_SCAN_LOCK = threading.Lock()
_SNAPSHOT_CACHE: Optional[Dict[str, Any]] = None
_SNAPSHOT_CACHE_AT = 0
# Key order is part of the /scan-status wire shape.
_SCAN_STATUS: Dict[str, Any] = {"state": "idle", "started_at": None, "finished_at": None, "last_error": None, "last_duration_ms": None, "run_count": 0}

ERROR_RE = re.compile(r"\b(error|failed|failure|traceback|exception|permission denied|not found|eaddrinuse|already in use|timed out|blocked)\b", re.I)
PORT_RE = re.compile(r"\b(port\s+)?(3000|5173|8000|8080|9119)\b.*\b(in use|already|taken|eaddrinuse)\b|\beaddrinuse\b", re.I)
INSTALL_RE = re.compile(r"\b(npm|pnpm|yarn|pip|uv)\b.*\b(install|add)\b", re.I)
SUCCESS_RE = re.compile(r"\b(success|passed|built|compiled|done|exit_code[\"']?\s*[:=]\s*0|verified|ok)\b", re.I)
FILE_RE = re.compile(r"(?:/home/|~/?|\./|/mnt/)[\w./-]+\.(?:py|js|ts|tsx|jsx|css|html|md|json|yaml|yml|svg|sql|sh)")

TIER_NAMES = ["Copper", "Silver", "Gold", "Diamond", "Olympian"]

def _ach(
    id: str, name: str, description: str, category: str, icon: str, *,
    metric: Optional[str] = None, tiers: Optional[List[int]] = None,
    requires: Optional[List[tuple]] = None, secret: bool = False) -> Dict[str, Any]:
    """Build one catalog entry. ``kind`` is derived: ``requires`` -> multi_condition; a
    ``max_*`` metric is a per-session best (best_session); anything else accumulates over
    the whole history (lifetime)."""
    kind = "multi_condition" if requires is not None else ("best_session" if metric.startswith("max_") else "lifetime")
    item: Dict[str, Any] = {"id": id, "name": name, "description": description, "category": category, "kind": kind, "icon": icon}
    if secret:
        item["secret"] = True
    if requires is not None:
        item["requirements"] = [{"metric": m, "gte": gte} for m, gte in requires]
    else:
        item["threshold_metric"] = metric
        item["tiers"] = [{"name": n, "threshold": t} for n, t in zip(TIER_NAMES, tiers)]
    return item


ACHIEVEMENTS: List[Dict[str, Any]] = [
    # Agent Autonomy — mostly best-session feats
    _ach("let_him_cook", "Let Him Cook", "Let Hermes run a serious autonomous tool chain in one session.", "Agent Autonomy", "flame", metric="max_tool_calls_in_session", tiers=[200, 500, 1200, 3000, 8000]),
    _ach("autonomous_avalanche", "Autonomous Avalanche", "Accumulate a lifetime avalanche of Hermes tool calls across sessions.", "Agent Autonomy", "avalanche", metric="total_tool_calls", tiers=[1000, 3000, 8000, 20000, 50000]),
    _ach("toolchain_maxxer", "Toolchain Maxxer", "Use a wide spread of distinct Hermes tools in one session.", "Agent Autonomy", "nodes", metric="max_distinct_tools_in_session", tiers=[18, 28, 45, 70, 100]),
    _ach("full_send", "Full Send", "Terminal, files, and web/browser all get involved in one real run.", "Agent Autonomy", "rocket", requires=[("max_terminal_calls_in_session", 180), ("max_file_tool_calls_in_session", 120), ("max_web_browser_calls_in_session", 60)]),
    _ach("subagent_commander", "Subagent Commander", "Coordinate delegated agent work.", "Agent Autonomy", "branch", metric="total_delegate_calls", tiers=[5, 40, 100, 1000, 5000]),
    _ach("background_process_enjoyer", "Background Process Enjoyer", "Start or control enough long-running processes to deserve the title.", "Agent Autonomy", "daemon", metric="total_process_calls", tiers=[300, 800, 2000, 6000, 15000]),
    _ach("cron_necromancer", "Cron Necromancer", "Raise scheduled autonomous jobs from the dead.", "Agent Autonomy", "clock", metric="total_cron_calls", tiers=[1000, 3000, 8000, 20000, 50000]),

    # Debugging Chaos — higher thresholds + multi-condition events
    _ach("red_text_connoisseur", "Red Text Connoisseur", "Encounter enough errors to develop a palate for red text.", "Debugging Chaos", "warning", metric="total_errors", tiers=[1500, 4000, 10000, 25000, 75000]),
    _ach("stack_trace_sommelier", "Stack Trace Sommelier", "Taste tracebacks by the flight, not by the sip.", "Debugging Chaos", "wine", metric="traceback_events", tiers=[300, 1000, 3000, 8000, 20000]),
    _ach("actually_read_the_logs", "Actually Read The Logs", "Inspect logs repeatedly instead of guessing.", "Debugging Chaos", "scroll", metric="log_read_events", tiers=[1000, 3000, 8000, 20000, 50000]),
    _ach("port_3000_taken", "Port 3000 Is Taken", "Discover dev-server port conflict patterns enough times to become numb.", "Debugging Chaos", "plug", metric="port_conflict_events", tiers=[15, 40, 100, 300, 1000], secret=True),
    _ach("permission_denied_any_percent", "Permission Denied Any%", "Speedrun into permission walls.", "Debugging Chaos", "lock", metric="permission_denied_events", tiers=[25, 75, 200, 600, 1500], secret=True),
    _ach("dependency_hell_tourist", "Dependency Hell Tourist", "Package installs fail, then somehow life continues.", "Debugging Chaos", "package_skull", requires=[("install_error_events", 25), ("install_success_events", 10)]),
    _ach("the_fix_was_restarting", "The Fix Was Restarting It", "Restart after enough error clusters to call it a technique.", "Debugging Chaos", "restart", requires=[("restart_after_error_events", 50), ("total_errors", 4000)]),
    _ach("forgot_the_env_var", "Forgot The Env Var", "Auth or configuration failed because an environment variable was missing.", "Debugging Chaos", "key", metric="env_var_error_events", tiers=[5000, 15000, 40000, 100000, 250000], secret=True),
    _ach("yaml_colon_incident", "YAML Colon Incident", "Configuration syntax bites back.", "Debugging Chaos", "colon", metric="yaml_error_events", tiers=[1000, 3000, 8000, 20000, 50000], secret=True),
    _ach("docker_name_collision", "Docker Name Collision", "A container name already exists. Of course it does.", "Debugging Chaos", "container", metric="docker_conflict_events", tiers=[75, 200, 600, 1500, 4000], secret=True),

    # Vibe Coding
    _ach("supposed_to_be_quick", "This Was Supposed To Be Quick", "A tiny ask becomes an entire expedition.", "Vibe Coding", "melting_clock", metric="max_messages_in_session", tiers=[300, 600, 1200, 2500, 6000]),
    _ach("one_more_small_change", "One More Small Change", "Make enough file edits in one session to invalidate the phrase small change.", "Vibe Coding", "pencil", metric="max_file_tool_calls_in_session", tiers=[150, 400, 1000, 3000, 8000]),
    _ach("vibe_architect", "Vibe Architect", "Touch a broad surface area in one project session.", "Vibe Coding", "blueprint", metric="max_files_touched_in_session", tiers=[300, 700, 1500, 4000, 10000]),
    _ach("pixel_goblin", "Pixel Goblin", "Do sustained frontend, CSS, SVG, or visual tuning.", "Vibe Coding", "pixel", metric="frontend_activity_events", tiers=[20000, 50000, 120000, 300000, 800000]),
    _ach("ship_first_ask_later", "Ship First, Ask Later", "Git activity after a serious tool chain.", "Vibe Coding", "ship", requires=[("git_events", 50), ("max_tool_calls_in_session", 500)]),
    _ach("css_exorcist", "CSS Exorcist", "Cast repeated styling demons out of the interface.", "Vibe Coding", "spark_cursor", metric="css_activity_events", tiers=[10000, 30000, 80000, 200000, 500000]),
    _ach("one_character_fix", "One Character Fix", "A tiny edit after a pile of errors. Painful. Beautiful.", "Vibe Coding", "needle", requires=[("tiny_patch_after_errors_events", 5), ("total_errors", 4000)], secret=True),

    # Hermes Native
    _ach("skillsmith", "Skillsmith", "Work with Hermes skills enough to leave fingerprints.", "Hermes Native", "hammer_scroll", metric="skill_events", tiers=[5000, 15000, 40000, 100000, 250000]),
    _ach("skill_issue_skill_created", "Skill Issue? Skill Created.", "Create or patch durable procedures instead of repeating yourself.", "Hermes Native", "anvil", metric="skill_manage_events", tiers=[25, 75, 200, 600, 1500]),
    _ach("memory_keeper", "Memory Keeper", "Persist durable knowledge with memory or Mnemosyne.", "Hermes Native", "crystal", metric="memory_events", tiers=[100, 300, 1000, 3000, 8000]),
    _ach("memory_palace", "Memory Palace", "Build a serious durable-memory trail.", "Hermes Native", "palace", metric="memory_write_events", tiers=[100, 300, 1000, 3000, 8000]),
    _ach("context_dragon", "Context Dragon", "Brush against compression, huge context, or token pressure repeatedly.", "Hermes Native", "dragon", metric="context_events", tiers=[5000, 15000, 40000, 100000, 250000]),
    _ach("gateway_dweller", "Gateway Dweller", "Live through gateway-connected Hermes workflows.", "Hermes Native", "antenna", metric="gateway_events", tiers=[5000, 15000, 40000, 100000, 250000]),
    _ach("plugin_goblin", "Plugin Goblin", "Use or develop plugins enough that the dashboard notices.", "Hermes Native", "puzzle", metric="plugin_events", tiers=[1000, 3000, 8000, 20000, 50000]),
    _ach("rollback_wizard", "Rollback Wizard", "Invoke rollback/checkpoint recovery magic.", "Hermes Native", "rewind", metric="rollback_events", tiers=[500, 1500, 4000, 10000, 25000], secret=True),

    # Research/Web
    _ach("rabbit_hole_certified", "Rabbit Hole Certified", "Search or extract enough web content to qualify as a research spiral.", "Research/Web", "spiral", metric="total_web_calls", tiers=[400, 1200, 3000, 8000, 20000]),
    _ach("citation_goblin", "Citation Goblin", "Extract enough web pages to become a tiny librarian.", "Research/Web", "quote", metric="total_web_extract_calls", tiers=[100, 300, 1000, 3000, 8000]),
    _ach("docs_archaeologist", "Docs Archaeologist", "Dig through documentation sources over and over.", "Research/Web", "compass", metric="docs_activity_events", tiers=[5000, 15000, 40000, 100000, 250000]),
    _ach("browser_possession", "Browser Possession", "Possess a browser through automation repeatedly.", "Research/Web", "browser", metric="browser_calls", tiers=[75, 200, 600, 1500, 4000]),

    # Tool Mastery
    _ach("terminal_goblin", "Terminal Goblin", "Spend serious time in shell-land.", "Tool Mastery", "terminal", metric="total_terminal_calls", tiers=[750, 2000, 6000, 15000, 50000]),
    _ach("patch_wizard", "Patch Wizard", "Bend files to your will with targeted patches.", "Tool Mastery", "wand", metric="total_patch_calls", tiers=[250, 750, 2000, 6000, 15000]),
    _ach("file_archaeologist", "File Archaeologist", "Dig through the filesystem with reads and searches.", "Tool Mastery", "folder", metric="total_file_reads_searches", tiers=[750, 2000, 6000, 15000, 50000]),
    _ach("image_whisperer", "Image Whisperer", "Use image generation or vision tools enough for visual work.", "Tool Mastery", "eye", metric="image_vision_calls", tiers=[100, 300, 1000, 3000, 8000]),
    _ach("voice_of_the_machine", "Voice Of The Machine", "Use text-to-speech or voice tooling repeatedly.", "Tool Mastery", "wave", metric="tts_calls", tiers=[10, 30, 100, 300, 800]),

    # Model Lore
    _ach("model_hopper", "Model Hopper", "Switch or inspect providers/models enough to count as a habit.", "Model Lore", "swap", metric="model_events", tiers=[10000, 30000, 80000, 200000, 500000]),
    _ach("openrouter_enjoyer", "OpenRouter Enjoyer", "Route model work through OpenRouter repeatedly.", "Model Lore", "router", metric="openrouter_events", tiers=[250, 750, 2000, 6000, 15000]),
    _ach("codex_conjurer", "Codex Conjurer", "Summon Codex-flavored assistance often enough for a ritual.", "Model Lore", "codex", metric="codex_events", tiers=[500, 1500, 4000, 10000, 25000]),
    _ach("multi_model_mage", "Multi-Model Mage", "Use a real spread of distinct model names across Hermes history.", "Model Lore", "prism", metric="distinct_model_count", tiers=[10, 20, 40, 80, 160]),
    _ach("five_model_flight", "Five-Model Flight", "Try at least five distinct LLMs instead of marrying the first model that answers.", "Model Lore", "prism", metric="distinct_model_count", tiers=[5, 10, 20, 40, 80]),
    _ach("provider_polyglot", "Provider Polyglot", "Use models from multiple providers across Hermes history.", "Model Lore", "swap", metric="distinct_provider_count", tiers=[2, 3, 5, 8, 12]),
    _ach("model_sommelier", "Model Sommelier", "Taste enough model/provider conversations to develop preferences.", "Model Lore", "wine", metric="model_events", tiers=[250, 750, 2000, 6000, 15000]),
    _ach("claude_confidant", "Claude Confidant", "Bring Claude-flavored reasoning into the workflow repeatedly.", "Model Lore", "quote", metric="claude_events", tiers=[50, 150, 500, 1500, 4000]),
    _ach("gemini_cartographer", "Gemini Cartographer", "Map enough Gemini-related workflows to know the terrain.", "Model Lore", "compass", metric="gemini_events", tiers=[50, 150, 500, 1500, 4000]),
    _ach("open_weights_pilgrim", "Open Weights Pilgrim", "Actually chat with local/open-weight models through Hermes session metadata.", "Model Lore", "terminal", metric="local_model_chat_sessions", tiers=[1, 3, 10, 30, 100]),

    # Workflow Intelligence
    _ach("toolset_cartographer", "Toolset Cartographer", "Navigate Hermes toolsets deliberately instead of treating tools as a blur.", "Hermes Native", "compass", metric="toolset_events", tiers=[20, 60, 200, 600, 1500]),
    _ach("config_surgeon", "Config Surgeon", "Operate on real config files, manifests, env files, and dashboard settings without flinching.", "Hermes Native", "key", metric="config_events", tiers=[100, 300, 1000, 3000, 10000]),
    _ach("rebase_acrobat", "Rebase Acrobat", "Handle real git history surgery: rebase, conflict, merge, fetch, push.", "Vibe Coding", "branch", metric="git_history_events", tiers=[10, 30, 100, 300, 800]),
    _ach("test_suite_tamer", "Test Suite Tamer", "Run enough verification commands that green text becomes part of the ritual.", "Tool Mastery", "daemon", metric="test_events", tiers=[100, 300, 800, 2400, 6000]),
    _ach("screenshot_hunter", "Screenshot Hunter", "Capture, inspect, and polish visual proof instead of just claiming it works.", "Tool Mastery", "eye", metric="screenshot_events", tiers=[50, 150, 500, 1500, 5000]),

    # Lifestyle
    _ach("marathon_operator", "Marathon Operator", "Accumulate a serious number of Hermes sessions.", "Lifestyle", "marathon", metric="session_count", tiers=[75, 200, 500, 1500, 5000]),
    _ach("weekend_warrior", "Weekend Warrior", "Run Hermes on weekends enough times to make it a lifestyle.", "Lifestyle", "calendar", metric="weekend_sessions", tiers=[25, 75, 200, 600, 1500]),
    _ach("night_shift_operator", "Night Shift Operator", "Run sessions during gremlin hours repeatedly.", "Lifestyle", "moon", metric="night_sessions", tiers=[25, 75, 200, 600, 1500]),
    _ach("cache_hit_appreciator", "Cache Hit Appreciator", "Notice or benefit from prompt/cache behavior.", "Lifestyle", "cache", metric="cache_events", tiers=[100, 300, 1000, 3000, 8000], secret=True),

]

# ---- Durable state files ----

SNAPSHOT_FILE = "scan_snapshot.json"
CHECKPOINT_FILE = "scan_checkpoint.json"


def _data_dir() -> Path:
    """Durable data root (``<hermes home>/plugin-data/hermes-achievements/``). State used to
    live in the install tree and died on ``hermes plugins remove``/``update``; legacy files
    migrate on first read (see ``_data_file``)."""
    try:
        from plugins.plugin_storage import plugin_data_dir
        return plugin_data_dir("hermes-achievements")
    except Exception:
        # Standalone dashboard import (no plugins package on sys.path): same layout, computed locally.
        root = get_hermes_home() / "plugin-data" / "hermes-achievements"
        root.mkdir(parents=True, exist_ok=True)
        return root


def _data_file(name: str) -> Path:
    path = _data_dir() / name
    if not path.exists():
        legacy = get_hermes_home() / "plugins" / "hermes-achievements" / name
        if legacy.exists():
            try:
                path.write_text(legacy.read_text(encoding="utf-8"), encoding="utf-8")
            except Exception:
                pass
    return path


def _read_json(name: str) -> Any:
    """Parsed data file, or ``None`` when missing/unreadable."""
    path = _data_file(name)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(name: str, data: Any) -> None:
    path = _data_file(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(data), indent=2, sort_keys=True), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, set):
        return sorted(_json_safe(v) for v in value)
    return value


def load_state() -> Dict[str, Any]:
    data = _read_json("state.json")
    return {"unlocks": {}} if data is None else data


def save_state(state: Dict[str, Any]) -> None:
    _write_json("state.json", state)


def load_checkpoint() -> Dict[str, Any]:
    data = _read_json(CHECKPOINT_FILE)
    if isinstance(data, dict):
        data.setdefault("schema_version", 1)
        data.setdefault("generated_at", 0)
        data.setdefault("sessions", {})
        if isinstance(data.get("sessions"), dict):
            return data
    return {"schema_version": 1, "generated_at": 0, "sessions": {}}


def session_fingerprint(meta: Dict[str, Any]) -> Dict[str, Any]:
    return {"last_active": meta.get("last_active"), "started_at": meta.get("started_at"), "model": meta.get("model"), "title": meta.get("title") or meta.get("preview") or "Untitled"}


def _cache_is_fresh(now: int) -> bool:
    return _SNAPSHOT_CACHE is not None and (now - _SNAPSHOT_CACHE_AT) <= SNAPSHOT_TTL_SECONDS


def _is_snapshot_stale(snapshot: Optional[Dict[str, Any]], now: Optional[int] = None) -> bool:
    ts = int(snapshot.get("generated_at") or 0) if isinstance(snapshot, dict) else 0
    return ts <= 0 or (int(now or time.time()) - ts) > SNAPSHOT_TTL_SECONDS


def _scan_status_payload(now: Optional[int] = None) -> Dict[str, Any]:
    current = int(now or time.time())
    snap = _SNAPSHOT_CACHE if isinstance(_SNAPSHOT_CACHE, dict) else None
    generated_at = int(snap.get("generated_at") or 0) if snap else 0
    return {
        **_SCAN_STATUS,
        "ttl_seconds": SNAPSHOT_TTL_SECONDS,
        "snapshot_generated_at": generated_at or None,
        "snapshot_age_seconds": (current - generated_at) if generated_at else None,
        "snapshot_stale": _is_snapshot_stale(snap, current)}


# ---- Per-session analysis ----

def _tool_name_from_call(call: Any) -> Optional[str]:
    if not isinstance(call, dict):
        return None
    return call.get("name") or (call.get("function") or {}).get("name")


def _content(msg: Dict[str, Any]) -> str:
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content)
    except Exception:
        return str(content)


def _count_tool(tool_names: List[str], *needles: str) -> int:
    lowered = [name.lower() for name in tool_names]
    return sum(1 for name in lowered if any(needle in name for needle in needles))


_PROVIDER_MARKERS = ["openai", "anthropic", "google", "gemini", "mistral", "meta", "qwen", "deepseek", "xai", "nous", "ollama", "groq", "openrouter", "codex"]
_LOCAL_MARKERS = ["ollama", "llama.cpp", "localhost", "127.0.0.1", "local/", "local:", "gguf", "vllm-local"]


def model_provider(model_name: str) -> Optional[str]:
    name = (model_name or "").strip().lower()
    if not name or name == "none":
        return None
    if "/" in name:
        return name.split("/", 1)[0]
    for provider in _PROVIDER_MARKERS:
        if provider in name:
            return "google" if provider == "gemini" else provider
    return name.split(":", 1)[0].split("-", 1)[0]


def is_local_model_name(model_name: str) -> bool:
    name = (model_name or "").strip().lower()
    return bool(name) and name != "none" and any(marker in name for marker in _LOCAL_MARKERS)


def analyze_messages(session_id: str, title: str, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    tool_names: Set[str] = set()
    tool_sequence: List[str] = []
    files_touched: Set[str] = set()
    full_text_parts: List[str] = []
    error_count = 0
    for msg in messages:
        text = _content(msg)
        full_text_parts.append(text)
        if msg.get("tool_name"):
            name = str(msg["tool_name"])
            tool_names.add(name)
            # Tool result rows name the tool that already appeared in the assistant tool_calls:
            # keep it for distinct-tool detection but don't double-count it as a new call.
            if msg.get("role") != "tool":
                tool_sequence.append(name)
        for call in msg.get("tool_calls") or []:
            name = _tool_name_from_call(call)
            if name:
                tool_names.add(name)
                tool_sequence.append(name)
        if ERROR_RE.search(text):
            error_count += 1
        blob = text
        if msg.get("tool_calls"):
            blob += " " + json.dumps(msg.get("tool_calls"), default=str)
        files_touched.update(FILE_RE.findall(blob))

    full_text = "\n".join(full_text_parts)
    lower = full_text.lower()

    def hits(pattern: str) -> int:
        return len(re.findall(pattern, full_text, re.I))
    web_calls = _count_tool(tool_sequence, "web_search", "web_extract")
    browser_calls = _count_tool(tool_sequence, "browser")

    return {
        "session_id": session_id,
        "title": title or "Untitled session",
        "message_count": len(messages),
        "tool_call_count": len(tool_sequence),
        "tool_names": tool_names,
        "distinct_tool_count": len(tool_names),
        "error_count": error_count,
        "terminal_calls": _count_tool(tool_sequence, "terminal"),
        "web_calls": web_calls,
        "web_extract_calls": _count_tool(tool_sequence, "web_extract"),
        "browser_calls": browser_calls,
        "web_browser_calls": web_calls + browser_calls,
        "patch_calls": _count_tool(tool_sequence, "patch"),
        "file_reads_searches": _count_tool(tool_sequence, "read_file", "search_files"),
        "file_tool_calls": _count_tool(tool_sequence, "read_file", "write_file", "patch", "search_files"),
        "files_touched_count": len(files_touched),
        "delegate_calls": _count_tool(tool_sequence, "delegate_task"),
        "process_calls": _count_tool(tool_sequence, "process") + hits(r"background\s*=\s*true"),
        "cron_calls": _count_tool(tool_sequence, "cronjob"),
        "image_vision_calls": _count_tool(tool_sequence, "image", "vision"),
        "tts_calls": _count_tool(tool_sequence, "tts", "text_to_speech"),
        "skill_events": _count_tool(tool_sequence, "skill") + len(re.findall(r"\bskill", lower)),
        "skill_manage_events": _count_tool(tool_sequence, "skill_manage"),
        "memory_events": _count_tool(tool_sequence, "memory", "mnemosyne"),
        "memory_write_events": _count_tool(tool_sequence, "mnemosyne_remember", "memory"),
        "port_conflict": bool(PORT_RE.search(full_text)),
        "port_conflict_events": 1 if PORT_RE.search(full_text) else 0,
        "traceback_events": hits(r"traceback|exception"),
        "log_read_events": hits(r"gateway\.log|errors\.log|agent\.log|/api/logs|\blogs\b"),
        "permission_denied_events": hits(r"permission denied|eacces|operation not permitted"),
        "install_error_events": 1 if INSTALL_RE.search(full_text) and ERROR_RE.search(full_text) else 0,
        "install_success_events": 1 if INSTALL_RE.search(full_text) and SUCCESS_RE.search(full_text) else 0,
        "restart_after_error_events": 1 if error_count and re.search(r"\brestart|reload|kill|start\b", full_text, re.I) else 0,
        "env_var_error_events": hits(r"missing .*env|api key|environment variable|not configured|unauthorized|auth"),
        "yaml_error_events": hits(r"yaml|yml|colon|parse error") if ERROR_RE.search(full_text) else 0,
        "docker_conflict_events": hits(r"docker.*(name|container).*already|container name conflict|Conflict\. The container"),
        "frontend_activity_events": hits(r"\.(css|svg|tsx|jsx)|frontend|tailwind|react"),
        "css_activity_events": hits(r"\.css|tailwind|style|className|visual"),
        "git_events": hits(r"\bgit\s+(commit|push|merge|rebase|status|diff)"),
        "tiny_patch_after_errors_events": 1 if error_count >= 5 and re.search(r"one character|single character|typo", full_text, re.I) else 0,
        "context_events": hits(r"compress|context window|token|cache"),
        "gateway_events": hits(r"gateway|discord|telegram|slack|api_server"),
        "plugin_events": hits(r"plugin|dashboard-plugins|__HERMES_PLUGIN|manifest\.json"),
        "rollback_events": hits(r"rollback|checkpoint"),
        "docs_activity_events": hits(r"docs|documentation|docusaurus|README"),
        "model_events": hits(r"model|provider|openrouter|codex|gemini|claude|anthropic|openai|mistral|qwen|deepseek|llama|ollama|vllm|gguf"),
        "openrouter_events": hits(r"openrouter"),
        "codex_events": hits(r"codex"),
        "claude_events": hits(r"claude|anthropic"),
        "gemini_events": hits(r"gemini|google ai|google model"),
        "local_model_events": hits(r"ollama|llama\.cpp|gguf|vllm|local model|open[- ]weight|open weights"),
        "toolset_events": hits(r"toolset|enabled_toolsets|browser tool|terminal tool|file tool|web tool"),
        "config_events": hits(r"config\.ya?ml|\b[a-z0-9_-]+config\.(?:js|ts|json|ya?ml)|\.env(?:\b|\.)|manifest\.json|settings\.json|pyproject\.toml|package\.json"),
        "git_history_events": hits(r"\bgit\s+(rebase|merge|fetch|pull|push|tag|checkout)|merge conflict|conflict\s*\(|rebase --continue"),
        "test_events": hits(r"pytest|unittest|vitest|playwright|npm test|pnpm test|node --check|py_compile|tests? passed|\bOK\b"),
        "screenshot_events": hits(r"screenshot|playwright|vision_analyze|browser_vision|\.png|image data"),
        "release_events": hits(r"\bgit\s+tag|release|version bump|changelog|publish|pushed? tag"),
        "cache_events": hits(r"cache hit|prompt caching|cache_read"),
        "model_names": set()}


# ---- Evaluation ----

def _result(*, unlocked: bool, discovered: bool, state: str, tier, progress: int, next_tier, next_threshold: int, progress_pct: int) -> Dict[str, Any]:
    """Uniform evaluation result (key order is part of the wire shape)."""
    return {"unlocked": unlocked, "discovered": discovered, "state": state, "tier": tier, "progress": progress, "next_tier": next_tier, "next_threshold": next_threshold, "progress_pct": progress_pct}


def _state(definition: Dict[str, Any], unlocked: bool, any_progress: bool) -> tuple[str, bool]:
    """``(state, discovered)``: secret badges stay hidden until the first matching signal."""
    secret = bool(definition.get("secret"))
    state = "unlocked" if unlocked else ("secret" if secret and not any_progress else "discovered")
    return state, any_progress or not secret


def evaluate_tiered(definition: Dict[str, Any], aggregate: Dict[str, Any]) -> Dict[str, Any]:
    progress = int(aggregate.get(definition["threshold_metric"], 0) or 0)
    tiers_list = sorted(definition.get("tiers", []), key=lambda t: t["threshold"])
    achieved = [t for t in tiers_list if progress >= t["threshold"]]
    next_tiers = [t for t in tiers_list if progress < t["threshold"]]
    next_threshold = next_tiers[0]["threshold"] if next_tiers else (tiers_list[-1]["threshold"] if tiers_list else 1)
    current_threshold = achieved[-1]["threshold"] if achieved else 0
    denom = max(1, next_threshold - current_threshold)
    pct = 100 if not next_tiers and achieved else max(0, min(99, math.floor(((progress - current_threshold) / denom) * 100)))
    state, discovered = _state(definition, bool(achieved), progress > 0)
    return _result(
        unlocked=bool(achieved), discovered=discovered, state=state, tier=achieved[-1]["name"] if achieved else None,
        progress=progress, next_tier=next_tiers[0]["name"] if next_tiers else None, next_threshold=next_threshold, progress_pct=pct)


def evaluate_requirements(definition: Dict[str, Any], aggregate: Dict[str, Any]) -> Dict[str, Any]:
    requirements = definition.get("requirements", [])
    if not requirements:
        state, discovered = _state(definition, False, False)
        return _result(unlocked=False, discovered=discovered, state=state, tier=None, progress=0, next_tier=None, next_threshold=1, progress_pct=0)
    parts = []
    any_progress = False
    complete = True
    for requirement in requirements:
        value = int(aggregate.get(requirement["metric"], 0) or 0)
        threshold = int(requirement.get("gte", 1))
        any_progress = any_progress or value > 0
        complete = complete and value >= threshold
        parts.append(min(1.0, value / max(1, threshold)))
    pct = math.floor((sum(parts) / len(parts)) * 100)
    state, discovered = _state(definition, complete, any_progress)
    return _result(
        unlocked=complete, discovered=discovered, state=state, tier=None, progress=pct, next_tier=None,
        next_threshold=100, progress_pct=100 if complete else min(99, pct))


def evaluate_definition(definition: Dict[str, Any], aggregate: Dict[str, Any]) -> Dict[str, Any]:
    if "threshold_metric" in definition:
        return evaluate_tiered(definition, aggregate)
    return evaluate_requirements(definition, aggregate)


METRIC_LABELS = {
    "max_tool_calls_in_session": "tool calls in one session",
    "max_distinct_tools_in_session": "distinct Hermes tools used in one session",
    "max_terminal_calls_in_session": "terminal calls in one session",
    "max_file_tool_calls_in_session": "file/search/patch calls in one session",
    "max_web_browser_calls_in_session": "web search/extract or browser calls in one session",
    "max_messages_in_session": "messages in one session",
    "max_files_touched_in_session": "files touched in one session",
    "total_delegate_calls": "lifetime delegate_task calls",
    "total_process_calls": "lifetime background process operations",
    "total_cron_calls": "lifetime scheduled-job operations",
    "total_errors": "error/failed/traceback messages observed",
    "traceback_events": "traceback or exception mentions",
    "log_read_events": "log inspections",
    "port_conflict_events": "dev-server port conflict detections",
    "permission_denied_events": "permission-denied errors",
    "install_error_events": "package-install failures",
    "install_success_events": "successful package installs after package work",
    "restart_after_error_events": "restart/reload actions after error clusters",
    "env_var_error_events": "missing auth/config/environment-variable events",
    "yaml_error_events": "YAML/config parse incidents",
    "docker_conflict_events": "Docker/container-name conflicts",
    "frontend_activity_events": "frontend/CSS/SVG/React activity mentions",
    "css_activity_events": "CSS, styling, Tailwind, or className activity",
    "git_events": "git workflow commands",
    "tiny_patch_after_errors_events": "tiny typo-style fixes after error clusters",
    "skill_events": "Hermes skill mentions or tool use",
    "skill_manage_events": "skill_manage create/patch/delete operations",
    "memory_events": "memory or Mnemosyne tool events",
    "memory_write_events": "durable memory writes",
    "context_events": "context, compression, token, or cache-pressure mentions",
    "gateway_events": "gateway/API/chat-platform activity",
    "plugin_events": "dashboard plugin development or usage signals",
    "rollback_events": "rollback/checkpoint recovery mentions",
    "docs_activity_events": "documentation/README/docs activity",
    "model_events": "model/provider-related activity",
    "openrouter_events": "OpenRouter mentions",
    "codex_events": "Codex mentions",
    "cache_events": "prompt-cache/cache-hit mentions",
    "total_web_calls": "lifetime web_search/web_extract calls",
    "total_web_extract_calls": "lifetime web_extract calls",
    "browser_calls": "lifetime browser automation calls",
    "total_tool_calls": "lifetime Hermes tool calls",
    "total_terminal_calls": "lifetime terminal calls",
    "total_patch_calls": "lifetime targeted patch edits",
    "total_file_reads_searches": "lifetime read_file/search_files calls",
    "image_vision_calls": "image generation or vision tool calls",
    "tts_calls": "text-to-speech or voice tool calls",
    "distinct_model_count": "distinct model names seen in session metadata",
    "distinct_provider_count": "distinct model providers inferred from session metadata",
    "claude_events": "Claude/Anthropic model mentions",
    "gemini_events": "Gemini/Google model mentions",
    "local_model_events": "local/open-weight model mentions",
    "local_model_chat_sessions": "Hermes sessions whose model metadata is local/open-weight",
    "toolset_events": "toolset or tool-family mentions",
    "config_events": "configuration/environment/manifest activity",
    "git_history_events": "git history operations such as rebase, merge, fetch, push, or tag",
    "test_events": "test/check/verification command mentions",
    "screenshot_events": "screenshot, Playwright, PNG, or vision-inspection activity",
    "release_events": "release, version, publish, or git tag events",
    "session_count": "Hermes sessions",
    "weekend_sessions": "sessions started on weekends",
    "night_sessions": "sessions started late night or before dawn"}


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric.replace("_", " "))


def criteria_for(definition: Dict[str, Any]) -> str:
    if definition.get("secret") and definition.get("state") == "secret":
        return "Secret: exact requirement hidden until Hermes sees the first matching signal. Keep using Hermes across debugging, tools, memory, skills, plugins, and model workflows to reveal it."
    if "threshold_metric" in definition:
        tiers_list = sorted(definition.get("tiers", []), key=lambda t: t["threshold"])
        if not tiers_list:
            return "Requirement: use Hermes in the matching workflow."
        ladder = ", ".join(f"{t['name']} {t['threshold']}" for t in tiers_list)
        return f"Requirement: {metric_label(definition['threshold_metric'])}. Tier ladder: {ladder}."
    requirements = definition.get("requirements") or []
    if requirements:
        return "Requirement: " + "; ".join(f"{metric_label(r['metric'])} ≥ {int(r.get('gte', 1))}" for r in requirements) + "."
    return "Requirement: complete the matching Hermes behavior."


def display_achievement(item: Dict[str, Any]) -> Dict[str, Any]:
    clean = dict(item)
    if clean.get("state") == "secret":
        return {**clean, "name": "???", "description": "Secret achievement: hidden until Hermes detects the first relevant behavior in your session history.", "criteria": criteria_for(clean), "icon": "secret"}
    clean["criteria"] = criteria_for(clean)
    return clean


# ---- Scanning + aggregation ----

def _scan_meta(mode: str, total: int, *, rescanned: int = 0, reused: int = 0, scanned_so_far: Optional[int] = None, expected_total: Optional[int] = None) -> Dict[str, Any]:
    meta = {"mode": mode, "sessions_total": total, "sessions_rescanned": rescanned, "sessions_reused": reused}
    if scanned_so_far is not None:
        meta.update(sessions_scanned_so_far=scanned_so_far, sessions_expected_total=expected_total)
    return meta


def scan_sessions(limit: Optional[int] = None, progress_callback: Optional[Any] = None, progress_every: int = 250) -> Dict[str, Any]:
    """Scan Hermes sessions and build per-session achievement stats.

    ``limit=None`` (default) scans the ENTIRE history (SQLite ``LIMIT -1``); a former cap
    of 200 silently shrank lifetime totals on long-running installs. The checkpoint stores
    per-session stats keyed by ``(started_at, last_active)`` fingerprint so warm scans only
    re-analyze changed sessions. ``progress_callback(partial_sessions, scanned_so_far,
    total)`` fires every ``progress_every`` sessions so background scans can publish
    intermediate snapshots.
    """
    try:
        from hermes_state import SessionDB
    except Exception as exc:
        return {"sessions": [], "aggregate": {}, "error": f"Could not import SessionDB: {exc}", "scan_meta": _scan_meta("failed", 0)}

    previous_sessions = load_checkpoint()["sessions"]  # load_checkpoint guarantees a dict
    reused = rescanned = 0
    db_limit = -1 if (limit is None or limit <= 0) else int(limit)
    db = SessionDB()
    try:
        sessions_meta = db.list_sessions_rich(limit=db_limit, include_children=True, project_compression_tips=False)
        total_sessions = len(sessions_meta)
        sessions: List[Dict[str, Any]] = []
        checkpoint_sessions: Dict[str, Any] = {}
        for idx, meta in enumerate(sessions_meta, start=1):
            sid = meta.get("id")
            if not sid:
                continue
            fp = session_fingerprint(meta)
            cached = previous_sessions.get(sid)
            cached = cached if isinstance(cached, dict) else {}
            title = meta.get("title") or meta.get("preview")
            if isinstance(cached.get("stats"), dict) and cached.get("fingerprint") == fp:
                stats = dict(cached["stats"])
                reused += 1
            else:
                stats = analyze_messages(sid, title or "Untitled", db.get_messages(sid))
                rescanned += 1
            stats.update(session_id=sid, title=title or stats.get("title") or "Untitled", started_at=meta.get("started_at"), last_active=meta.get("last_active"), source=meta.get("source"))
            if meta.get("model"):
                # Checkpoint round-trips turn the set into a list; handle both.
                model = str(meta.get("model"))
                names = stats.setdefault("model_names", set())
                if isinstance(names, set):
                    names.add(model)
                elif isinstance(names, list):
                    if model not in names:
                        names.append(model)
                else:
                    stats["model_names"] = {model}
            sessions.append(stats)
            checkpoint_sessions[sid] = {"fingerprint": fp, "stats": _json_safe(stats)}
            if progress_callback is not None and progress_every > 0 and (idx % progress_every == 0) and idx < total_sessions:
                try:
                    progress_callback(list(sessions), idx, total_sessions)
                except Exception:
                    pass  # Advisory — a broken publisher must never abort the scan.
        _write_json(CHECKPOINT_FILE, {"schema_version": 1, "generated_at": int(time.time()), "sessions": checkpoint_sessions})
    finally:
        db.close()
    return {
        "sessions": sessions,
        "aggregate": aggregate_stats(sessions),
        "scan_meta": _scan_meta(
            "incremental" if reused > 0 else "full", len(sessions), rescanned=rescanned, reused=reused,
            scanned_so_far=len(sessions), expected_total=total_sessions)}


# Per-session bests: aggregate metric -> session stat key (also drives evidence_for).
_SESSION_MAX_METRICS = {
    "max_tool_calls_in_session": "tool_call_count",
    "max_distinct_tools_in_session": "distinct_tool_count",
    "max_messages_in_session": "message_count",
    "max_terminal_calls_in_session": "terminal_calls",
    "max_file_tool_calls_in_session": "file_tool_calls",
    "max_web_calls_in_session": "web_calls",
    "max_web_browser_calls_in_session": "web_browser_calls",
    "max_files_touched_in_session": "files_touched_count"}
# Lifetime sums: aggregate metric -> session stat key.
_SESSION_SUM_METRICS = {
    "total_errors": "error_count",
    "total_tool_calls": "tool_call_count",
    "total_terminal_calls": "terminal_calls",
    "total_web_calls": "web_calls",
    "total_web_extract_calls": "web_extract_calls",
    "total_patch_calls": "patch_calls",
    "total_file_reads_searches": "file_reads_searches",
    "total_delegate_calls": "delegate_calls",
    "total_process_calls": "process_calls",
    "total_cron_calls": "cron_calls",
    "browser_calls": "browser_calls",
    "image_vision_calls": "image_vision_calls",
    "tts_calls": "tts_calls"}
# ``*_events`` counters summed under their own name.
_SESSION_EVENT_KEYS = [
    "traceback_events", "log_read_events", "port_conflict_events", "permission_denied_events", "install_error_events", "install_success_events", "restart_after_error_events", "env_var_error_events", "yaml_error_events", "docker_conflict_events", "frontend_activity_events", "css_activity_events", "git_events", "tiny_patch_after_errors_events", "skill_events", "skill_manage_events", "memory_events", "memory_write_events", "context_events", "gateway_events", "plugin_events", "rollback_events", "docs_activity_events", "model_events", "openrouter_events", "codex_events", "claude_events", "gemini_events", "local_model_events", "toolset_events", "config_events", "git_history_events", "test_events", "screenshot_events", "release_events", "cache_events",
]


def aggregate_stats(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Key order is part of the /rescan wire shape.
    agg: Dict[str, Any] = {"session_count": len(sessions)}
    for key in (*_SESSION_MAX_METRICS, *_SESSION_SUM_METRICS, "distinct_model_count", "distinct_provider_count", "local_model_chat_sessions", "weekend_sessions", "night_sessions", *_SESSION_EVENT_KEYS):
        agg[key] = 0
    model_names: Set[str] = set()
    provider_names: Set[str] = set()
    for s in sessions:
        for key, stat in _SESSION_MAX_METRICS.items():
            agg[key] = max(agg[key], s.get(stat, 0))
        for key, stat in _SESSION_SUM_METRICS.items():
            agg[key] += s.get(stat, 0)
        for key in _SESSION_EVENT_KEYS:
            agg[key] += s.get(key, 0)
        session_models = s.get("model_names") or set()
        model_names.update(session_models)
        provider_names.update(filter(None, (model_provider(str(m)) for m in session_models)))
        if any(is_local_model_name(str(m)) for m in session_models):
            agg["local_model_chat_sessions"] += 1
        if s.get("started_at"):
            try:
                lt = time.localtime(float(s.get("started_at")))
                if lt.tm_wday >= 5:
                    agg["weekend_sessions"] += 1
                if lt.tm_hour < 6 or lt.tm_hour >= 23:
                    agg["night_sessions"] += 1
            except Exception:
                pass
    agg["distinct_model_count"] = len({m for m in model_names if m and m != "None"})
    agg["distinct_provider_count"] = len(provider_names)
    return agg


def evidence_for(definition: Dict[str, Any], sessions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    key = _SESSION_MAX_METRICS.get(definition.get("threshold_metric"))
    if not sessions or key is None:
        return None
    s = max(sessions, key=lambda x: x.get(key, 0))
    return {"session_id": s.get("session_id"), "title": s.get("title"), "value": s.get(key, 0)}


# ---- Snapshot assembly ----

def _snapshot(evaluated: List[Dict[str, Any]], scan: Dict[str, Any], now: int) -> Dict[str, Any]:
    """Wire payload shared by finished, partial and pending snapshots."""
    return {
        "achievements": evaluated,
        "sessions": scan.get("sessions", []),
        "aggregate": scan.get("aggregate", {}),
        "scan_meta": scan.get("scan_meta", {}),
        "error": scan.get("error"),
        "unlocked_count": sum(1 for a in evaluated if a["unlocked"]),
        "discovered_count": sum(1 for a in evaluated if a.get("state") == "discovered"),
        "secret_count": sum(1 for a in evaluated if a.get("state") == "secret"),
        "total_count": len(evaluated),
        "generated_at": now}


def _compute_from_scan(scan: Dict[str, Any], *, is_partial: bool = False) -> Dict[str, Any]:
    """Evaluate every achievement definition against a scan result. Used by ``compute_all``
    for finished scans AND by the background progress callback for in-flight snapshots;
    ``is_partial=True`` skips persisting ``state.json`` unlocks — an "unlock time" from
    half a scan could be invalidated by a later session."""
    aggregate = scan.get("aggregate", {})
    state = load_state() if not is_partial else {"unlocks": {}}
    unlocks = state.setdefault("unlocks", {})
    now = int(time.time())
    evaluated = []
    for definition in ACHIEVEMENTS:
        result = evaluate_definition(definition, aggregate)
        unlock_id = definition["id"]
        if not is_partial and result["unlocked"] and unlock_id not in unlocks:
            unlocks[unlock_id] = {"unlocked_at": now, "first_tier": result.get("tier"), "evidence": evidence_for(definition, scan.get("sessions", []))}
        item = {**definition, **result}
        if result["unlocked"]:
            item["unlocked_at"] = unlocks.get(unlock_id, {}).get("unlocked_at")
            item["evidence"] = unlocks.get(unlock_id, {}).get("evidence") or evidence_for(definition, scan.get("sessions", []))
        evaluated.append(display_achievement(item))
    if not is_partial:
        save_state(state)
    return _snapshot(evaluated, scan, now)


def compute_all(progress_callback: Optional[Any] = None, progress_every: int = 250) -> Dict[str, Any]:
    scan = scan_sessions(progress_callback=progress_callback, progress_every=progress_every)
    return _compute_from_scan(scan, is_partial=False)


_BACKGROUND_SCAN_THREAD: Optional[threading.Thread] = None
_BACKGROUND_SCAN_LOCK = threading.Lock()


def _build_pending_snapshot(now: int) -> Dict[str, Any]:
    """Structurally-complete placeholder served while the first-ever scan runs, so the UI
    renders an empty list + spinner without special-casing "no data"."""
    evaluated = [
        display_achievement({
            **d, "unlocked": False, "discovered": False, "state": "secret" if d.get("secret") else "discovered", "progress": 0,
            "progress_pct": 0, "next_tier": (d.get("tiers") or [{}])[0].get("name"),
            "next_threshold": (d.get("tiers") or [{}])[0].get("threshold", 1), "tier": None})
        for d in ACHIEVEMENTS]
    return _snapshot(evaluated, {"scan_meta": _scan_meta("pending", 0), "error": None}, now)


def _set_cache(snapshot: Dict[str, Any], at: int) -> None:
    global _SNAPSHOT_CACHE, _SNAPSHOT_CACHE_AT
    _SNAPSHOT_CACHE = _json_safe(snapshot)
    _SNAPSHOT_CACHE_AT = at


def _run_scan_and_update_cache(publish_partial_snapshots: bool = True) -> None:
    """Execute a scan + snapshot update (synchronously or from a thread). With
    ``publish_partial_snapshots`` (background scans) the scanner periodically publishes
    in-progress snapshots to ``_SNAPSHOT_CACHE`` so a long cold scan unlocks badges
    incrementally; synchronous /rescan callers pass ``False`` since they block on the result."""
    with _SCAN_LOCK:
        started = int(time.time())
        _SCAN_STATUS.update(state="running", started_at=started, last_error=None)

        def _publish_partial(partial_sessions, scanned_so_far, total):
            try:
                partial_scan = {
                    "sessions": partial_sessions,
                    "aggregate": aggregate_stats(partial_sessions),
                    "scan_meta": _scan_meta("in_progress", scanned_so_far, scanned_so_far=scanned_so_far, expected_total=total),
                }
                # _SNAPSHOT_CACHE_AT stays 0 so partials remain in the 'stale' regime: the UI
                # keeps polling /scan-status and never mistakes an in-flight result for a finished one.
                _set_cache(_compute_from_scan(partial_scan, is_partial=True), 0)
            except Exception:
                pass  # Intermediate publication is best-effort; don't kill the scan.

        try:
            computed = _json_safe(compute_all(progress_callback=_publish_partial if publish_partial_snapshots else None))
            _set_cache(computed, int(computed.get("generated_at") or int(time.time())))
            _write_json(SNAPSHOT_FILE, _SNAPSHOT_CACHE)
            _SCAN_STATUS["state"] = "idle"
        except Exception as exc:
            _SCAN_STATUS.update(state="failed", last_error=str(exc))
        finally:
            finished = int(time.time())
            _SCAN_STATUS.update(finished_at=finished, last_duration_ms=int((finished - started) * 1000), run_count=int(_SCAN_STATUS.get("run_count", 0)) + 1)


def _start_background_scan() -> None:
    """Kick off a daemon-thread scan unless one is already running (idempotent)."""
    global _BACKGROUND_SCAN_THREAD
    with _BACKGROUND_SCAN_LOCK:
        existing = _BACKGROUND_SCAN_THREAD
        if existing is not None and existing.is_alive():
            return
        thread = threading.Thread(target=_run_scan_and_update_cache, kwargs={"publish_partial_snapshots": True}, name="hermes-achievements-scan", daemon=True)
        _BACKGROUND_SCAN_THREAD = thread
        thread.start()


def evaluate_all(force: bool = False) -> Dict[str, Any]:
    """Return the current achievements payload: a fresh in-memory cache is returned as is;
    a stale on-disk snapshot is served while a background rescan runs (UI decorates it with
    ``is_stale=True``); with no snapshot yet an empty-but-valid "pending" payload is served
    while the first scan runs; ``force=True`` (manual /rescan) scans synchronously. Cold
    scans on 8000+ session databases take minutes, hence the background thread."""
    global _SNAPSHOT_CACHE, _SNAPSHOT_CACHE_AT
    now = int(time.time())
    if not force and _cache_is_fresh(now):
        return _SNAPSHOT_CACHE or {}
    # Lazy-load the persisted snapshot so fresh process starts serve cached data.
    if _SNAPSHOT_CACHE is None:
        persisted = _read_json(SNAPSHOT_FILE)
        if isinstance(persisted, dict):
            _SNAPSHOT_CACHE = persisted
            _SNAPSHOT_CACHE_AT = int(persisted.get("generated_at") or 0) or now
    if force:
        # No partial publishing: the caller is blocking on the final result.
        _run_scan_and_update_cache(publish_partial_snapshots=False)
    elif not _cache_is_fresh(now):
        # Serve what we have (stale is fine) and refresh in the background; on a first-ever
        # run the UI polls /scan-status and re-fetches when the scan completes.
        _start_background_scan()
    return _SNAPSHOT_CACHE if _SNAPSHOT_CACHE is not None else _build_pending_snapshot(now)


# ---- Routes ----

@router.get("/achievements")
async def achievements():
    data = evaluate_all()
    payload = {k: data[k] for k in ["achievements", "unlocked_count", "discovered_count", "secret_count", "total_count", "error", "generated_at"] if k in data}
    payload["is_stale"] = _is_snapshot_stale(data)
    payload["scan_meta"] = {**(data.get("scan_meta") or {}), "status": _scan_status_payload()}
    return payload


@router.get("/scan-status")
async def scan_status():
    return _scan_status_payload()


@router.get("/recent-unlocks")
async def recent_unlocks():
    data = evaluate_all()
    return sorted([a for a in data["achievements"] if a["unlocked"]], key=lambda a: a.get("unlocked_at") or 0, reverse=True)[:20]


@router.get("/sessions/{session_id}/badges")
async def session_badges(session_id: str):
    data = evaluate_all()
    session = next((s for s in data["sessions"] if s["session_id"] == session_id), None)
    if not session:
        return {"session_id": session_id, "badges": []}
    aggregate = aggregate_stats([session])
    results = [(d, evaluate_definition(d, aggregate)) for d in ACHIEVEMENTS]
    return {"session_id": session_id, "badges": [display_achievement({**d, **r}) for d, r in results if r["unlocked"]]}


@router.post("/rescan")
async def rescan():
    return {"ok": True, **evaluate_all(force=True)}


@router.post("/reset-state")
async def reset_state():
    global _SNAPSHOT_CACHE, _SNAPSHOT_CACHE_AT
    save_state({"unlocks": {}})
    _SNAPSHOT_CACHE = None
    _SNAPSHOT_CACHE_AT = 0
    _SCAN_STATUS.update(state="idle", started_at=None, finished_at=None, last_error=None, last_duration_ms=None)
    for name in (SNAPSHOT_FILE, CHECKPOINT_FILE):
        try:
            _data_file(name).unlink(missing_ok=True)
        except Exception:
            pass
    return {"ok": True}
