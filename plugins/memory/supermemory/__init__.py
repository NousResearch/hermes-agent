"""Supermemory memory plugin (MemoryProvider): profile recall, semantic search, memory tools, turn capture, session ingest."""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import threading
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from agent.secret_scope import get_secret, is_multiplex_active
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_DEFAULT_CONTAINER_TAG = "hermes"
_VALID_SEARCH_MODES = ("hybrid", "memories", "documents")
_DEFAULT_BASE_URL = "https://api.supermemory.ai"
_API_KEY_URL = "http://app.supermemory.ai/integrations?connect=hermes"
# Strips injected <supermemory-context> / <supermemory-containers> blocks before capture.
_INJECTED_BLOCK_RE = re.compile(r"<supermemory-(context|containers)>[\s\S]*?</supermemory-\1>\s*", re.DOTALL)
_DEFAULT_ENTITY_CONTEXT = (
    "User-assistant conversation. Format: [role: user]...[user:end] and [role: assistant]...[assistant:end].\n\n"
    "Only extract things useful in future conversations. Most messages are not worth remembering.\n\n"
    "Remember lasting personal facts, preferences, routines, tools, ongoing projects, working context, "
    "and explicit requests to remember something.\n\n"
    "Do not remember temporary intents, one-time tasks, assistant actions, implementation details, or in-progress status.\n\n"
    "When in doubt, store less."
)
# snake_case tool name -> kebab-case alias exposed alongside it.
_KEBAB_ALIASES = {"supermemory_store": "supermemory-save", "supermemory_search": "supermemory-search",
                  "supermemory_forget": "supermemory-forget", "supermemory_profile": "supermemory-profile"}
_ALIAS_TO_TOOL = {kebab: snake for snake, kebab in _KEBAB_ALIASES.items()}
_BOOL_WORDS = {**dict.fromkeys(("true", "1", "yes", "y", "on"), True), **dict.fromkeys(("false", "0", "no", "n", "off"), False)}


def _quietly(fn: Callable[[], Any], fail_msg: str = "", *args: Any, level: int = logging.DEBUG, default: Any = None) -> Any:
    """Run ``fn()``; on any exception log ``fail_msg`` (if given) with traceback and return ``default``."""
    try:
        return fn()
    except Exception:
        if fail_msg:
            logger.log(level, fail_msg, *args, exc_info=True)
        return default


def _sanitize_tag(raw: str) -> str:
    return re.sub(r"_+", "_", re.sub(r"[^a-zA-Z0-9_]", "_", raw or "")).strip("_") or _DEFAULT_CONTAINER_TAG


def _resolve_base_url(config_value: Any = "") -> str:
    """config > SUPERMEMORY_BASE_URL env var > default (self-hosted support)."""
    raw = str(config_value or "").strip() or os.environ.get("SUPERMEMORY_BASE_URL", "").strip()
    return (raw or _DEFAULT_BASE_URL).rstrip("/") or _DEFAULT_BASE_URL


def _clamp_entity_context(text: str) -> str:
    return text.strip()[:1500] if text else _DEFAULT_ENTITY_CONTEXT


def _as_bool(value: Any, default: bool) -> bool:
    """bool passthrough; common true/false words parsed; anything else (incl. ints) -> default."""
    return value if isinstance(value, bool) else _BOOL_WORDS.get(value.strip().lower(), default) if isinstance(value, str) else default


def _clamp_number(value: Any, default, lo, hi, cast):
    """Cast ``value`` and clamp it to [lo, hi]; fall back to ``default`` on any conversion error."""
    return _quietly(lambda: max(lo, min(hi, cast(value))), default=default)


# config key -> (default, normalizer applied to the raw/merged value). Order = supermemory.json layout.
# container_tag is kept raw here: {identity} templates are resolved in initialize(), and
# _sanitize_tag runs AFTER that resolution. custom_containers, by contrast, are sanitized on load.
_CONFIG_SPEC: Dict[str, tuple] = {
    "container_tag": (_DEFAULT_CONTAINER_TAG, lambda v: str(v).strip() or _DEFAULT_CONTAINER_TAG),
    "auto_recall": (True, lambda v: _as_bool(v, True)),
    "auto_capture": (True, lambda v: _as_bool(v, True)),
    "max_recall_results": (10, lambda v: _clamp_number(v, 10, 1, 20, int)),
    "profile_frequency": (50, lambda v: _clamp_number(v, 50, 1, 500, int)),
    "capture_mode": ("all", lambda v: "everything" if v == "everything" else "all"),
    "search_mode": ("hybrid", lambda v: v if (v := str(v).strip().lower()) in _VALID_SEARCH_MODES else "hybrid"),
    "entity_context": (_DEFAULT_ENTITY_CONTEXT, lambda v: _clamp_entity_context(str(v))),
    "api_timeout": (5.0, lambda v: _clamp_number(v, 5.0, 0.5, 15.0, float)),
    "base_url": ("", lambda v: str(v or "").strip()),
    "enable_custom_container_tags": (False, lambda v: _as_bool(v, False)),
    "custom_containers": ([], lambda v: [_sanitize_tag(str(t)) for t in v if t] if isinstance(v, list) else []),
    "custom_container_instructions": ("", lambda v: str(v).strip()),
}


def _read_json_dict(path: Path) -> dict:
    raw = _quietly(lambda: json.loads(path.read_text(encoding="utf-8")), "Failed to parse %s", path) if path.exists() else None
    return raw if isinstance(raw, dict) else {}


def _load_supermemory_config(hermes_home: Optional[str] = None) -> dict:
    """Defaults overlaid with $hermes_home/supermemory.json (None = defaults only), every key normalized."""
    config = {k: (list(d) if isinstance(d, list) else d) for k, (d, _) in _CONFIG_SPEC.items()}
    if hermes_home is not None:
        config.update({k: v for k, v in _read_json_dict(Path(hermes_home) / "supermemory.json").items() if v is not None})
    for key, (_, normalize) in _CONFIG_SPEC.items():
        config[key] = normalize(config[key])
    return config


def _save_supermemory_config(values: dict, hermes_home: str) -> None:
    from utils import atomic_json_write
    config_path = Path(hermes_home) / "supermemory.json"
    atomic_json_write(config_path, {**_read_json_dict(config_path), **values}, mode=0o600, sort_keys=True)


def _detect_category(text: str) -> str:
    lowered = text.lower()  # first matching pattern wins
    return next((cat for cat, pat in (("preference", r"prefer|like|love|hate|want"), ("decision", r"decided|will use|going with"),
                                      ("fact", r"\bis\b|\bare\b|\bhas\b|\bhave\b")) if re.search(pat, lowered)), "other")


def _format_relative_time(iso_timestamp: str) -> str:
    """'just now' / '5m ago' / '3h ago' / '2d ago' / '%d %b[ %Y]'; '' when unparseable."""
    def _fmt():
        dt, now = datetime.fromisoformat(iso_timestamp.replace("Z", "+00:00")), datetime.now(timezone.utc)
        seconds = (now - dt).total_seconds()
        for limit, unit, label in ((1800, 0, "just now"), (3600, 60, "m ago"), (86400, 3600, "h ago"), (604800, 86400, "d ago")):
            if seconds < limit:
                return f"{int(seconds / unit)}{label}" if unit else label
        return dt.strftime("%d %b" if dt.year == now.year else "%d %b %Y")
    return _quietly(_fmt, default="")


def _similarity_pct(value: Any) -> Optional[int]:
    """0..1 similarity -> whole percent; None when absent or unparseable."""
    return _quietly(lambda: None if value is None else round(float(value) * 100))


def _profile_sections(static_facts: list, dynamic_facts: list) -> list[str]:
    return [f"## {title}\n" + "\n".join(f"- {item}" for item in items)
            for title, items in (("User Profile (Persistent)", static_facts), ("Recent Context", dynamic_facts)) if items]


def _format_prefetch_context(static_facts: list, dynamic_facts: list, search_results: list, max_results: int) -> str:
    """Dedupe across the three lists (earlier lists win: profile facts beat search hits), cap each, render."""
    seen: set = set()

    def _unique(items, key=lambda x: x):  # set.add() returns None, so `not seen.add(k)` records k and keeps the item
        return [i for i in items or [] if (k := key(i)) and k not in seen and not seen.add(k)][:max_results]
    sections = _profile_sections(_unique(static_facts), _unique(dynamic_facts))
    lines = []
    for item in _unique(search_results, key=lambda i: i.get("memory", "")):
        rel = _format_relative_time(item.get("updated_at") or item.get("updatedAt") or "")
        pct = _similarity_pct(item.get("similarity"))
        lines.append(f"- {' '.join(([f'[{rel}]'] if rel else []) + ([f'[{pct}%]'] if pct is not None else []))} {item['memory']}".strip())
    sections += ["## Relevant Memories\n" + "\n".join(lines)] if lines else []
    intro = "The following is background context from long-term memory. Use it silently when relevant. Do not force memories into the conversation."
    return f"<supermemory-context>\n{intro}\n\n" + "\n\n".join(sections) + "\n</supermemory-context>" if sections else ""


def _clean_text_for_capture(text: str) -> str:
    return _INJECTED_BLOCK_RE.sub("", text or "").strip()


def _memory_fields(item: Any, *keys: str) -> dict:
    """Pick SDK result attrs into a plain dict; ``updated_at`` also accepts camelCase ``updatedAt``."""
    defaults = {"id": "", "memory": "", "similarity": None, "metadata": None}
    return {k: getattr(item, "updated_at", None) or getattr(item, "updatedAt", None) if k == "updated_at" else getattr(item, k, defaults[k])
            for k in keys}


class _SupermemoryClient:
    def __init__(self, api_key: str, timeout: float, container_tag: str,
                 search_mode: str = "hybrid", base_url: str = ""):
        # Lazy-install the SDK on demand (honors security.allow_lazy_installs and sealed Docker
        # venvs). On failure fall through so the raw import produces the canonical ImportError.
        _quietly(lambda: importlib.import_module("tools.lazy_deps").ensure("memory.supermemory", prompt=False))
        from supermemory import Supermemory
        self._api_key, self._container_tag, self._timeout = api_key, container_tag, timeout
        self._search_mode = search_mode if search_mode in _VALID_SEARCH_MODES else "hybrid"
        self._base_url = _resolve_base_url(base_url)
        self._client = Supermemory(api_key=api_key, base_url=self._base_url, timeout=timeout, max_retries=0,
                                   default_headers={"x-sm-source": "hermes"})

    def _merge_metadata(self, metadata: Optional[dict]) -> dict:
        # sm_source routes Hermes writes into the "Hermes" Space in the Supermemory app so the user
        # can filter / bulk-manage them per source agent (a routing key for the user, not telemetry).
        merged = {"sm_source": "hermes", **(metadata or {})}
        if (legacy_source := merged.pop("source", None)) and "type" not in merged:
            merged["type"] = str(legacy_source)
        return merged

    def add_memory(self, content: str, metadata: Optional[dict] = None, *, entity_context: str = "",
                   container_tag: Optional[str] = None, custom_id: Optional[str] = None) -> dict:
        kwargs: dict[str, Any] = {"content": content.strip(), "container_tags": [container_tag or self._container_tag],
                                  **({"metadata": self._merge_metadata(metadata)} if metadata else {}),
                                  **({"entity_context": _clamp_entity_context(entity_context)} if entity_context else {}),
                                  **({"custom_id": custom_id} if custom_id else {})}
        return {"id": getattr(self._client.documents.add(**kwargs), "id", "")}

    def search_memories(self, query: str, *, limit: int = 5, container_tag: Optional[str] = None,
                        search_mode: Optional[str] = None) -> list[dict]:
        mode = search_mode or self._search_mode
        kwargs: dict[str, Any] = {"q": query, "container_tag": container_tag or self._container_tag, "limit": limit,
                                  **({"search_mode": mode} if mode in _VALID_SEARCH_MODES else {})}
        response = self._client.search.memories(**kwargs)
        return [{**_memory_fields(item, "id", "memory", "similarity", "updated_at", "metadata"), "memory": getattr(item, "memory", "") or ""}
                for item in (getattr(response, "results", None) or [])]

    def get_profile(self, query: Optional[str] = None, *, container_tag: Optional[str] = None) -> dict:
        response = self._client.profile(container_tag=container_tag or self._container_tag, **({"q": query} if query else {}))
        profile_data = getattr(response, "profile", None)
        search_data = getattr(response, "search_results", None) or getattr(response, "searchResults", None)
        raw_results = getattr(search_data, "results", None) or search_data or []
        return {
            **{k: (getattr(profile_data, k, []) or []) if profile_data else [] for k in ("static", "dynamic")},
            "search_results": [item if isinstance(item, dict) else _memory_fields(item, "memory", "updated_at", "similarity")
                               for item in raw_results] if isinstance(raw_results, list) else [],
        }

    def forget_memory(self, memory_id: str, *, container_tag: Optional[str] = None) -> None:
        self._client.memories.forget(container_tag=container_tag or self._container_tag, id=memory_id)

    def forget_by_query(self, query: str, *, container_tag: Optional[str] = None) -> dict:
        results = self.search_memories(query, limit=5, container_tag=container_tag)
        memory_id = results[0].get("id", "") if results else ""
        if not memory_id:
            return {"success": False, "message": "Best matching memory has no id." if results else "No matching memory found to forget."}
        self.forget_memory(memory_id, container_tag=container_tag)
        return {"success": True, "message": f'Forgot: "{(results[0].get("memory") or "")[:100]}"', "id": memory_id}

    def ingest_conversation(self, session_id: str, messages: list[dict], metadata: dict | None = None) -> None:
        payload: dict = {"conversationId": session_id, "messages": messages, "containerTags": [self._container_tag],
                         **({"metadata": self._merge_metadata(metadata)} if metadata else {})}
        req = urllib.request.Request(f"{self._base_url}/v4/conversations", data=json.dumps(payload).encode("utf-8"), method="POST",
                                     headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json",
                                              "x-sm-source": "hermes"})
        with urllib.request.urlopen(req, timeout=self._timeout + 3):
            return


def _build_client(api_key: str, config: dict, container_tag: str) -> _SupermemoryClient:
    return _SupermemoryClient(api_key=api_key, timeout=config["api_timeout"], container_tag=container_tag,
                              search_mode=config["search_mode"], base_url=_resolve_base_url(config["base_url"]))


def _resolve_container_tag(config_tag: str, identity: str) -> str:
    """SUPERMEMORY_CONTAINER_TAG env > config > default; {identity} expands to the agent identity, then sanitize."""
    raw_tag = os.environ.get("SUPERMEMORY_CONTAINER_TAG", "").strip() or config_tag
    return _sanitize_tag(raw_tag.replace("{identity}", identity))


def _probe_supermemory_connection(api_key: str, hermes_home: str, *, identity: str = "default") -> dict:
    config = _load_supermemory_config(hermes_home)
    status = {"ok": False, "error": "", "profile_facts": 0, "container_tag": _resolve_container_tag(config["container_tag"], identity),
              "auto_recall": bool(config["auto_recall"]), "auto_capture": bool(config["auto_capture"])}
    if not (api_key or "").strip():
        return {**status, "error": "SUPERMEMORY_API_KEY not set"}
    try:
        __import__("supermemory")
    except ImportError:
        return {**status, "error": "supermemory package not installed"}
    try:
        profile = _build_client(api_key.strip(), config, status["container_tag"]).get_profile()
    except Exception as exc:
        return {**status, "error": str(exc).strip()[:160] or "connection failed"}
    facts = sum(1 for f in (profile.get("static") or []) + (profile.get("dynamic") or []) if f and str(f).strip())
    return {**status, "ok": True, "profile_facts": facts}


def _format_connection_summary(status: dict) -> str:
    container = status.get("container_tag") or _DEFAULT_CONTAINER_TAG
    flags = f"auto_recall {'on' if status.get('auto_recall') else 'off'} · auto_capture {'on' if status.get('auto_capture') else 'off'}"
    if status.get("ok"):
        facts = int(status.get("profile_facts") or 0)
        return f"✓ Connected · container: {container} · {facts} profile {'fact' if facts == 1 else 'facts'} · {flags}"
    return f"✗ {status.get('error') or 'connection failed'} · container: {container} · {flags}"


# (name, description, ((prop, type, description), ...), required) -> tool schema; kebab aliases are added in get_tool_schemas().
_BASE_SCHEMAS = [
    {"name": name, "description": description,
     "parameters": {"type": "object", "properties": {p: {"type": t, "description": d} for p, t, d in props}, **({"required": req} if req else {})}}
    for name, description, props, req in (
        ("supermemory_store", "Store an explicit memory for future recall.",
         (("content", "string", "The memory content to store."), ("metadata", "object", "Optional metadata attached to the memory.")), ["content"]),
        ("supermemory_search", "Search long-term memory by semantic similarity.",
         (("query", "string", "What to search for."), ("limit", "integer", "Maximum results to return, 1 to 20.")), ["query"]),
        ("supermemory_forget", "Forget a memory by exact id or by best-match query.",
         (("id", "string", "Exact memory id to delete."), ("query", "string", "Query used to find the memory to forget.")), None),
        ("supermemory_profile", "Retrieve persistent profile facts and recent memory context.",
         (("query", "string", "Optional query to focus the profile response."),), None),
    )
]


class _TagError(Exception):
    """Tool call named a container_tag outside the whitelist."""


def _tagged(resp: dict, tag: Optional[str]) -> dict:
    return {**resp, "container_tag": tag} if tag else resp


class SupermemoryMemoryProvider(MemoryProvider):
    def __init__(self):
        self._api_key = self._session_id = self._hermes_home = ""
        self._client: Optional[_SupermemoryClient] = None
        self._container_tag, self._turn_count, self._write_enabled, self._active = _DEFAULT_CONTAINER_TAG, 0, True, False
        self._prefetch_thread = self._sync_thread = self._write_thread = None  # only _write_thread is ever started
        self._session_turns: List[Dict[str, str]] = []
        self._apply_config(_load_supermemory_config())
        self._base_url, self._allowed_containers = _DEFAULT_BASE_URL, []  # env var is only consulted in initialize()

    def _apply_config(self, config: dict) -> None:
        for key in ("auto_recall", "auto_capture", "max_recall_results", "profile_frequency", "capture_mode",
                    "search_mode", "entity_context", "api_timeout", "custom_containers", "custom_container_instructions"):
            setattr(self, f"_{key}", config[key])
        self._base_url, self._enable_custom_containers = _resolve_base_url(config["base_url"]), config["enable_custom_container_tags"]
        self._allowed_containers: List[str] = [self._container_tag] + list(self._custom_containers)

    @property
    def name(self) -> str:
        return "supermemory"

    def is_available(self) -> bool:
        # Key presence only, no SDK import check: the SDK is lazy-installed in initialize(), so gating on
        # importability here is a chicken-and-egg trap on sealed venvs. Mirrors honcho/mem0.
        return bool(get_secret("SUPERMEMORY_API_KEY", ""))

    def get_config_schema(self):
        # Only the API key is prompted during `hermes memory setup`; other options live in supermemory.json / env.
        return [{"key": "api_key", "description": "Supermemory API key", "secret": True, "required": True, "env_var": "SUPERMEMORY_API_KEY", "url": _API_KEY_URL}]

    def save_config(self, values, hermes_home):
        sanitized = dict(values or {})
        for key, fix in (("container_tag", _sanitize_tag), ("entity_context", _clamp_entity_context)):
            if key in sanitized:
                sanitized[key] = fix(str(sanitized[key]))
        _save_supermemory_config(sanitized, hermes_home)

    def get_status_config(self, provider_config: dict) -> dict:
        from hermes_constants import get_hermes_home
        return {"summary": _format_connection_summary(_probe_supermemory_connection(get_secret("SUPERMEMORY_API_KEY", "") or "", str(get_hermes_home())))}

    def post_setup(self, hermes_home: str, config: dict) -> None:
        from hermes_cli.config import save_config
        from hermes_cli.memory_setup import _prompt, _write_env_vars
        print(f"\n  Configuring supermemory:\n\n  Get your API key at {_API_KEY_URL}\n")
        existing = os.environ.get("SUPERMEMORY_API_KEY", "")
        masked = f"...{existing[-4:]}" if len(existing) > 4 else "set"
        val = _prompt(f"Supermemory API key (current: {masked}, blank to keep)" if existing else "Supermemory API key", secret=True)
        memory = config["memory"] = config["memory"] if isinstance(config.get("memory"), dict) else {}
        memory["provider"] = self.name
        save_config(config)
        if val:
            _write_env_vars({"SUPERMEMORY_API_KEY": val}, hermes_home=hermes_home)
        api_key = val or existing
        # Make the freshly-entered key visible to the probe below. Single-profile only: under a multiplexed
        # gateway, writing to the process-global environ would leak the key to sibling profiles and their subprocesses.
        if api_key and not is_multiplex_active() and os.environ.get("SUPERMEMORY_API_KEY") != api_key:
            os.environ["SUPERMEMORY_API_KEY"] = api_key
        status = _probe_supermemory_connection(api_key, hermes_home)
        print(f"\n  {_format_connection_summary(status)}\n\n  Memory provider: supermemory\n  Activation saved to config.yaml")
        if val:
            print("  API keys saved to .env")
        print("\n  Start a new session to activate.\n")

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home
        self._hermes_home = kwargs.get("hermes_home") or str(get_hermes_home())
        self._session_id, self._turn_count, self._session_turns = session_id, 0, []
        config = _load_supermemory_config(self._hermes_home)
        self._api_key = get_secret("SUPERMEMORY_API_KEY", "") or ""
        self._container_tag = _resolve_container_tag(config["container_tag"], kwargs.get("agent_identity", "default"))
        self._apply_config(config)
        self._write_enabled = kwargs.get("agent_context", "") not in {"cron", "flush", "subagent"}
        self._client = _quietly(lambda: _build_client(self._api_key, config, self._container_tag),
                                "Supermemory initialization failed", level=logging.WARNING) if self._api_key else None
        self._active = self._client is not None

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._turn_count = max(turn_number, 0)

    def system_prompt_block(self) -> str:
        lines = ["# Supermemory", f"Active. Container: {self._container_tag}.",
                 "Use supermemory-search, supermemory-save, supermemory-forget, and supermemory-profile (aliases: supermemory_search, supermemory_store, supermemory_forget, supermemory_profile)."]
        if self._enable_custom_containers and self._custom_containers:
            lines += [f"\nMulti-container mode enabled. Available containers: {', '.join(self._allowed_containers)}.",
                      "Pass an optional container_tag to supermemory_search, supermemory_store, supermemory_forget, and supermemory_profile to target a specific container."]
            lines += [f"\n{self._custom_container_instructions}"] if self._custom_container_instructions else []
        return "\n".join(lines) if self._active else ""

    def _can_write(self) -> bool:
        return bool(self._active and self._write_enabled and self._client)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._active or not self._auto_recall or not self._client or not query.strip():
            return ""
        def _recall():
            profile = self._client.get_profile(query=query[:200])
            include_profile = self._turn_count <= 1 or (self._turn_count % self._profile_frequency == 0)
            return _format_prefetch_context(profile["static"] if include_profile else [], profile["dynamic"] if include_profile else [],
                                            profile["search_results"], self._max_recall_results)
        return _quietly(_recall, "Supermemory prefetch failed", default="")

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._can_write() or not self._auto_capture:
            return
        turn = {"user": _clean_text_for_capture(user_content), "assistant": _clean_text_for_capture(assistant_content)}
        if any(turn.values()):  # buffered for the single full-session document written at end/switch/shutdown
            self._session_turns.append(turn)

    def _ingest(self, session_id: str, messages: list[dict], metadata: dict, fail_msg: str, level: int = logging.DEBUG) -> None:
        metadata = {"type": "full_session", "session_id": session_id, **metadata}
        _quietly(lambda: self._client.ingest_conversation(session_id, messages, metadata=metadata), fail_msg, level=level)

    def _flush_turns(self, session_id: str, *, partial: bool, fail_msg: str) -> None:
        turns = self._session_turns  # message_count reports 2 per buffered turn regardless of empty sides
        messages = [{"role": role, "content": t[role]} for t in turns for role in ("user", "assistant") if t.get(role)]
        self._ingest(session_id, messages, {"message_count": len(turns) * 2, "partial": partial}, fail_msg)

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._can_write() or not self._session_id:
            return
        cleaned = [{"role": m.get("role"), "content": content} for m in messages or []
                   if m.get("role") in {"user", "assistant"} and (content := _clean_text_for_capture(str(m.get("content", ""))))]
        if not cleaned or (len(cleaned) == 1 and len(cleaned[0]["content"]) < 20):
            return
        self._ingest(self._session_id, cleaned, {"message_count": len(cleaned)}, "Supermemory session ingest failed", level=logging.WARNING)
        self._session_turns = []  # so shutdown() doesn't duplicate on normal exit

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False, **kwargs) -> None:
        """Flush any buffered turns from the old session as one document, then reset for the new session."""
        old_session_id = self._session_id
        if self._can_write():
            if self._session_turns and old_session_id:
                self._flush_turns(old_session_id, partial=not reset, fail_msg="Supermemory session-switch ingest failed")
            self._turn_count = 0
        self._session_id = str(new_session_id or "").strip() or old_session_id
        self._session_turns = []

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if not self._can_write() or action != "add" or not (content or "").strip():
            return
        if self._write_thread and self._write_thread.is_alive():
            self._write_thread.join(timeout=2.0)
        self._write_thread = threading.Thread(
            target=_quietly, daemon=False, name="supermemory-memory-write",
            args=(lambda: self._client.add_memory(content.strip(), metadata={"target": target, "type": "explicit_memory"},
                                                  entity_context=self._entity_context), "Supermemory on_memory_write failed"))
        self._write_thread.start()

    def shutdown(self) -> None:
        # Emergency fallback (crashes only). Buffer is cleared on normal on_session_end().
        if self._can_write() and self._session_turns and self._session_id:
            logger.warning("Supermemory: Saving session via shutdown (session=%s, turns=%d)", self._session_id, len(self._session_turns))
            self._flush_turns(self._session_id, partial=True, fail_msg="Supermemory shutdown ingest failed")
        if self._write_thread and self._write_thread.is_alive():
            self._write_thread.join(timeout=5.0)
        self._prefetch_thread = self._sync_thread = self._write_thread = None

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        schemas = [json.loads(json.dumps(base)) for base in _BASE_SCHEMAS]  # deep copies
        for schema in schemas if self._enable_custom_containers else ():  # multi-container mode: every tool takes container_tag
            schema["parameters"]["properties"]["container_tag"] = {
                "type": "string", "description": f"Optional container tag. Allowed: {', '.join(self._allowed_containers)}. Defaults to primary ({self._container_tag})."}
        # Kebab-case aliases are appended after all snake_case schemas (deep-copied, name swapped).
        return schemas + [{**json.loads(json.dumps(s)), "name": _KEBAB_ALIASES[s["name"]]} for s in schemas]

    def _tool_container_tag(self, args: dict) -> Optional[str]:
        """Validated container_tag from args; None = primary. Raises _TagError when not whitelisted."""
        raw = str(args.get("container_tag") or "").strip() if self._enable_custom_containers else ""
        tag = _sanitize_tag(raw) if raw else None
        if tag and tag not in self._allowed_containers:
            raise _TagError(f"Container tag '{tag}' is not allowed. Allowed: {', '.join(self._allowed_containers)}")
        return tag

    def _tool_store(self, args: dict) -> dict | str:
        content = str(args.get("content") or "").strip()
        if not content:
            return tool_error("content is required")
        metadata = args.get("metadata") if isinstance(args.get("metadata"), dict) else {}
        metadata.setdefault("type", _detect_category(content))
        metadata.pop("source", None)
        tag = self._tool_container_tag(args)
        result = self._client.add_memory(content, metadata=metadata, entity_context=self._entity_context, container_tag=tag)
        return _tagged({"saved": True, "id": result.get("id", ""), "preview": content[:80] + ("..." if len(content) > 80 else "")}, tag)

    def _tool_search(self, args: dict) -> dict | str:
        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("query is required")
        limit = _clamp_number(args.get("limit", 5) or 5, 5, 1, 20, int)
        tag = self._tool_container_tag(args)
        results = [{"id": i.get("id", ""), "content": i.get("memory", ""), **({"similarity": pct} if (pct := _similarity_pct(i.get("similarity"))) is not None else {})}
                   for i in self._client.search_memories(query, limit=limit, container_tag=tag)]
        return _tagged({"results": results, "count": len(results)}, tag)

    def _tool_forget(self, args: dict) -> dict | str:
        memory_id, query = str(args.get("id") or "").strip(), str(args.get("query") or "").strip()
        if not memory_id and not query:
            return tool_error("Provide either id or query")
        tag = self._tool_container_tag(args)  # not echoed in the response
        if not memory_id:
            return self._client.forget_by_query(query, container_tag=tag)
        self._client.forget_memory(memory_id, container_tag=tag)
        return {"forgotten": True, "id": memory_id}

    def _tool_profile(self, args: dict) -> dict:
        tag = self._tool_container_tag(args)
        profile = self._client.get_profile(query=str(args.get("query") or "").strip() or None, container_tag=tag)
        return _tagged({"profile": "\n\n".join(_profile_sections(profile["static"], profile["dynamic"])),
                       "static_count": len(profile["static"]), "dynamic_count": len(profile["dynamic"])}, tag)

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handlers return a tool_error() string for bad args or a dict to JSON-encode; client failures get ``fail_prefix``."""
        if not self._active or not self._client:
            return tool_error("Supermemory is not configured")
        tool_name = _ALIAS_TO_TOOL.get(tool_name, tool_name)
        if tool_name not in self._TOOL_HANDLERS:
            return tool_error(f"Unknown tool: {tool_name}")
        handler, fail_prefix = self._TOOL_HANDLERS[tool_name]
        try:
            resp = handler(self, args)
        except Exception as exc:
            return tool_error(str(exc) if isinstance(exc, _TagError) else f"{fail_prefix}: {exc}")
        return resp if isinstance(resp, str) else json.dumps(resp)

    # snake_case tool name -> (handler, error prefix); kebab aliases are folded in via _ALIAS_TO_TOOL first.
    _TOOL_HANDLERS = {"supermemory_store": (_tool_store, "Failed to store memory"), "supermemory_search": (_tool_search, "Search failed"),
                      "supermemory_forget": (_tool_forget, "Forget failed"), "supermemory_profile": (_tool_profile, "Profile failed")}


def register(ctx):
    ctx.register_memory_provider(SupermemoryMemoryProvider())


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

FORGET_SCHEMA = {
    "name": "supermemory_forget",
    "description": "Forget a memory by exact id or by best-match query.",
    "parameters": {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Exact memory id to delete."},
            "query": {"type": "string", "description": "Query used to find the memory to forget."},
        },
    },
}

PROFILE_SCHEMA = {
    "name": "supermemory_profile",
    "description": "Retrieve persistent profile facts and recent memory context.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Optional query to focus the profile response."},
        },
    },
}

SEARCH_SCHEMA = {
    "name": "supermemory_search",
    "description": "Search long-term memory by semantic similarity.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "limit": {"type": "integer", "description": "Maximum results to return, 1 to 20."},
        },
        "required": ["query"],
    },
}

STORE_SCHEMA = {
    "name": "supermemory_store",
    "description": "Store an explicit memory for future recall.",
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The memory content to store."},
            "metadata": {"type": "object", "description": "Optional metadata attached to the memory."},
        },
        "required": ["content"],
    },
}
# ---- END PLUGIN-COMPAT ----
