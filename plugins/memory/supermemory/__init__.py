"""Supermemory memory plugin (MemoryProvider): profile recall, semantic search, memory tools, per-turn capture."""

from __future__ import annotations

import importlib
import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from agent.memory_provider import MemoryProvider, spawn_context_thread
from agent.secret_scope import get_secret, is_multiplex_active
from tools.registry import tool_error

logger = logging.getLogger(__name__)

_DEFAULT_CONTAINER_TAG = "hermes"
_VALID_SEARCH_MODES = ("hybrid", "memories", "documents")
_DEFAULT_BASE_URL = "https://api.supermemory.ai"
_API_KEY_URL = "http://app.supermemory.ai/integrations?connect=hermes"
# Strips injected <supermemory-context> / <supermemory-containers> blocks before capture.
_INJECTED_BLOCK_RE = re.compile(r"<supermemory-(context|containers)>[\s\S]*?</supermemory-\1>\s*", re.DOTALL)
_DATA_URI_RE = re.compile(r"data:[^;,\s]+;base64,[A-Za-z0-9+/=]+")  # pasted inline images are useless as memory text
_CAPTURE_BUCKET_HOURS = 4  # one capture document per session per 4h window (matches the other Supermemory agent integrations)
_FAILED = object()  # _quietly default for capture writes: an explicit failure marker (a client returning None still counts as success)
_MAX_PENDING_TURNS = 50  # a down service must not accumulate an unbounded retry buffer
_MAX_PENDING_BYTES = 256 * 1024
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
    """config > SUPERMEMORY_BASE_URL (profile-scoped) > default (self-hosted support)."""
    raw = str(config_value or "").strip() or (get_secret("SUPERMEMORY_BASE_URL", "") or "").strip()
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
    raw = _quietly(lambda: json.loads(path.read_text(encoding="utf-8-sig")), "Failed to parse %s", path) if path.exists() else None
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
    return _DATA_URI_RE.sub("[image]", _INJECTED_BLOCK_RE.sub("", text or "")).strip()


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


def _format_turn(user: str, assistant: str) -> str:
    """Render one turn in the [role: x]...[x:end] layout the entity context describes."""
    return "\n".join(f"[role: {role}]\n{text}\n[{role}:end]" for role, text in (("user", user), ("assistant", assistant)) if text)


def _capture_custom_id(session_id: str, now: Optional[datetime] = None) -> str:
    """<session>_<YYYY-MM-DD>_b<0..5>: same id within a 4h window, so the API appends turns to one document."""
    now = now or datetime.now(timezone.utc)
    return f"{_sanitize_tag(session_id)}_{now:%Y-%m-%d}_b{now.hour // _CAPTURE_BUCKET_HOURS}"


def _build_client(api_key: str, config: dict, container_tag: str) -> _SupermemoryClient:
    return _SupermemoryClient(api_key=api_key, timeout=config["api_timeout"], container_tag=container_tag,
                              search_mode=config["search_mode"], base_url=_resolve_base_url(config["base_url"]))


def _resolve_container_tag(config_tag: str, identity: str) -> str:
    """SUPERMEMORY_CONTAINER_TAG (profile-scoped) > config > default; {identity} expands to the agent
    identity, then sanitize. The container is the data partition, so it must never be borrowed from
    the default profile's environ under multiplexing."""
    raw_tag = (get_secret("SUPERMEMORY_CONTAINER_TAG", "") or "").strip() or config_tag
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
        self._pending_turns: List[Dict[str, str]] = []  # failed writes, each tagged with its session_id; retried on next write/end/switch/shutdown
        self._capture_lock = threading.Lock()  # sync_turn (worker) vs on_session_switch/shutdown (caller thread) both touch _pending_turns
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
        self._session_id, self._turn_count, self._pending_turns = session_id, 0, []
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

    def _write_turns(self, mode: str, new_turn: Optional[Dict[str, str]] = None) -> None:
        """Write pending turns (+ ``new_turn``) as one documents.add per session id; custom_id = session + 4h bucket, so the
        API appends deltas. Failed batches stay pending under their own session id, so a switch never re-homes them.
        Retries are at-least-once: a write the API accepted but whose response was lost is re-sent and appended again.
        The lock serializes the snapshot/write/replace sequence across the worker and caller threads."""
        with self._capture_lock:
            turns = self._pending_turns + ([new_turn] if new_turn else [])
            if not turns:
                return
            failed: List[Dict[str, str]] = []
            for sid in dict.fromkeys(t["session_id"] for t in turns):
                batch = [t for t in turns if t["session_id"] == sid]
                now = datetime.now(timezone.utc)
                content = "\n\n".join(_format_turn(t["user"], t["assistant"]) for t in batch)
                metadata = {"type": "conversation", "session_id": sid, "timestamp": now.isoformat()}  # no sm_capture_mode: Hermes policy
                result = _quietly(lambda: self._client.add_memory(content, metadata=metadata, entity_context=self._entity_context,
                                                                  custom_id=_capture_custom_id(sid, now)),
                                  "Supermemory capture failed (%s, session=%s, %d turns pending)", mode, sid, len(batch),
                                  level=logging.WARNING if mode != "turn" else logging.DEBUG, default=_FAILED)
                if result is _FAILED:  # only a raised exception re-queues the batch
                    failed += batch
            self._pending_turns = failed

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # Host runs this on a worker thread, so the blocking write is fine here.
        if not self._can_write() or not self._auto_capture:
            return
        turn = {"user": _clean_text_for_capture(user_content), "assistant": _clean_text_for_capture(assistant_content),
                "session_id": session_id or self._session_id}
        if turn["user"] or turn["assistant"]:
            self._write_turns("turn", turn)
            self._bound_pending_turns()

    def _flush_pending(self, mode: str) -> None:
        if self._can_write():
            self._write_turns(mode)
            self._bound_pending_turns()

    def _bound_pending_turns(self) -> None:
        """Keep the retry buffer bounded: drop OLDEST entries past the turn/byte caps.

        Without this, a persistently failing service accumulates one entry per turn for the
        process lifetime (gateway runs never re-initialize) and every retry re-sends the
        whole accumulated payload."""
        with self._capture_lock:
            if len(self._pending_turns) <= _MAX_PENDING_TURNS and \
                    sum(len(t["user"]) + len(t["assistant"]) for t in self._pending_turns) <= _MAX_PENDING_BYTES:
                return
            kept: List[Dict[str, str]] = list(self._pending_turns)
            total = sum(len(t["user"]) + len(t["assistant"]) for t in kept)
            while len(kept) > _MAX_PENDING_TURNS or total > _MAX_PENDING_BYTES:
                if not kept:
                    break
                dropped = kept.pop(0)
                total -= len(dropped["user"]) + len(dropped["assistant"])
            if len(kept) != len(self._pending_turns):
                logger.warning("Supermemory: dropped %d oldest pending turn(s) to keep the retry buffer bounded",
                               len(self._pending_turns) - len(kept))
            self._pending_turns = kept

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        # Turns were already written as they completed; only retry what failed.
        self._flush_pending("session_end")

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False, **kwargs) -> None:
        # Pending turns survive the switch: they carry their own session_id, so a later retry still lands on the old session.
        self._flush_pending("session_switch")
        if self._can_write():
            self._turn_count = 0
        self._session_id = str(new_session_id or "").strip() or self._session_id

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if not self._can_write() or action != "add" or not (content or "").strip():
            return
        if self._write_thread and self._write_thread.is_alive():
            self._write_thread.join(timeout=2.0)
        self._write_thread = spawn_context_thread(
            _quietly, daemon=False, name="supermemory-memory-write",
            args=(lambda: self._client.add_memory(content.strip(), metadata={"target": target, "type": "explicit_memory"},
                                                  entity_context=self._entity_context), "Supermemory on_memory_write failed"))
        self._write_thread.start()

    def shutdown(self) -> None:
        self._flush_pending("shutdown")
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

class SupermemoryMemoryProvider(MemoryProvider):
    def __init__(self):
        self._config = _default_config()
        self._api_key = ""
        self._client: Optional[_SupermemoryClient] = None
        self._container_tag = _DEFAULT_CONTAINER_TAG
        self._session_id = ""
        self._turn_count = 0
        self._prefetch_result = ""
        self._prefetch_lock = threading.Lock()
        self._prefetch_thread: Optional[threading.Thread] = None
        self._sync_thread: Optional[threading.Thread] = None
        self._write_thread: Optional[threading.Thread] = None
        self._auto_recall = True
        self._auto_capture = True
        self._max_recall_results = _DEFAULT_MAX_RECALL_RESULTS
        self._profile_frequency = _DEFAULT_PROFILE_FREQUENCY
        self._capture_mode = _DEFAULT_CAPTURE_MODE
        self._search_mode = _DEFAULT_SEARCH_MODE
        self._entity_context = _DEFAULT_ENTITY_CONTEXT
        self._api_timeout = _DEFAULT_API_TIMEOUT
        self._base_url = _DEFAULT_BASE_URL
        self._hermes_home = ""
        self._write_enabled = True
        self._active = False
        # Multi-container support
        self._enable_custom_containers = False
        self._custom_containers: List[str] = []
        self._custom_container_instructions = ""
        self._allowed_containers: List[str] = []
        self._session_turns: List[Dict[str, str]] = []

    @property
    def name(self) -> str:
        return "supermemory"

    def is_available(self) -> bool:
        # Key presence only — no SDK import check. The supermemory SDK is
        # lazy-installed when the client is first constructed in initialize()
        # (see _SupermemoryClient.__init__). Gating availability on the SDK
        # being importable here would be a chicken-and-egg trap: on a sealed
        # Docker venv the package isn't present until ensure() runs, but
        # ensure() only runs once the provider is loaded — which this gates.
        # Mirrors honcho/mem0, which check config only. No network calls.
        return bool(get_secret("SUPERMEMORY_API_KEY", ""))

    def get_config_schema(self):
        # Only prompt for the API key during `hermes memory setup`.
        # All other options are documented for $HERMES_HOME/supermemory.json
        # or the SUPERMEMORY_CONTAINER_TAG env var.
        return [
            {"key": "api_key", "description": "Supermemory API key", "secret": True, "required": True, "env_var": "SUPERMEMORY_API_KEY", "url": _API_KEY_URL},
        ]

    def save_config(self, values, hermes_home):
        sanitized = dict(values or {})
        if "container_tag" in sanitized:
            sanitized["container_tag"] = _sanitize_tag(str(sanitized["container_tag"]))
        if "entity_context" in sanitized:
            sanitized["entity_context"] = _clamp_entity_context(str(sanitized["entity_context"]))
        _save_supermemory_config(sanitized, hermes_home)

    def get_status_config(self, provider_config: dict) -> dict:
        from hermes_constants import get_hermes_home

        del provider_config
        hermes_home = str(get_hermes_home())
        api_key = get_secret("SUPERMEMORY_API_KEY", "") or ""
        status = _probe_supermemory_connection(api_key, hermes_home)
        return {"summary": _format_connection_summary(status)}

    def post_setup(self, hermes_home: str, config: dict) -> None:
        from hermes_cli.config import save_config
        from hermes_cli.memory_setup import _prompt, _write_env_vars

        print("\n  Configuring supermemory:\n")
        print(f"  Get your API key at {_API_KEY_URL}\n")

        env_writes: dict[str, str] = {}
        existing = os.environ.get("SUPERMEMORY_API_KEY", "")
        if existing:
            masked = f"...{existing[-4:]}" if len(existing) > 4 else "set"
            val = _prompt(f"Supermemory API key (current: {masked}, blank to keep)", secret=True)
        else:
            val = _prompt("Supermemory API key", secret=True)
        if val:
            env_writes["SUPERMEMORY_API_KEY"] = val

        if not isinstance(config.get("memory"), dict):
            config["memory"] = {}
        config["memory"]["provider"] = self.name
        save_config(config)

        if env_writes:
            _write_env_vars(env_writes, hermes_home=hermes_home)

        api_key = env_writes.get("SUPERMEMORY_API_KEY") or existing
        # Make the freshly-entered key visible to the connection probe below.
        # (Checks the VALUE of SUPERMEMORY_API_KEY, not whether the key string
        # happens to name some unrelated env var.)
        # Single-profile convenience only: never write a profile's key into
        # the process-global environ under a multiplexed gateway — sibling
        # profiles' turns (and any subprocess spawned with env=os.environ)
        # would inherit it.
        if (
            api_key
            and not is_multiplex_active()
            and os.environ.get("SUPERMEMORY_API_KEY") != api_key
        ):
            os.environ["SUPERMEMORY_API_KEY"] = api_key

        status = _probe_supermemory_connection(api_key, hermes_home)
        print(f"\n  {_format_connection_summary(status)}")
        print("\n  Memory provider: supermemory")
        print("  Activation saved to config.yaml")
        if env_writes:
            print("  API keys saved to .env")
        print("\n  Start a new session to activate.\n")

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home
        self._hermes_home = kwargs.get("hermes_home") or str(get_hermes_home())
        self._session_id = session_id
        self._turn_count = 0
        self._config = _load_supermemory_config(self._hermes_home)
        self._api_key = get_secret("SUPERMEMORY_API_KEY", "") or ""

        # Resolve container tag: env var > config > default.
        # Supports {identity} template for profile-scoped containers.
        env_tag = os.environ.get("SUPERMEMORY_CONTAINER_TAG", "").strip()
        raw_tag = env_tag or self._config["container_tag"]
        identity = kwargs.get("agent_identity", "default")
        self._container_tag = _sanitize_tag(raw_tag.replace("{identity}", identity))

        self._auto_recall = self._config["auto_recall"]
        self._auto_capture = self._config["auto_capture"]
        self._max_recall_results = self._config["max_recall_results"]
        self._profile_frequency = self._config["profile_frequency"]
        self._capture_mode = self._config["capture_mode"]
        self._search_mode = self._config["search_mode"]
        self._entity_context = self._config["entity_context"]
        self._api_timeout = self._config["api_timeout"]
        # Base URL: config > SUPERMEMORY_BASE_URL env var > api.supermemory.ai.
        # Supports self-hosted Supermemory servers.
        self._base_url = _resolve_base_url(self._config["base_url"])
        self._enable_custom_containers = self._config["enable_custom_container_tags"]
        self._custom_containers = self._config["custom_containers"]
        self._custom_container_instructions = self._config["custom_container_instructions"]
        self._allowed_containers = [self._container_tag] + list(self._custom_containers)

        self._session_turns = []

        agent_context = kwargs.get("agent_context", "")
        self._write_enabled = agent_context not in {"cron", "flush", "subagent"}
        self._active = bool(self._api_key)
        self._client = None
        if self._active:
            try:
                self._client = _SupermemoryClient(
                    api_key=self._api_key,
                    timeout=self._api_timeout,
                    container_tag=self._container_tag,
                    search_mode=self._search_mode,
                    base_url=self._base_url,
                )
            except Exception:
                logger.warning("Supermemory initialization failed", exc_info=True)
                self._active = False
                self._client = None

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._turn_count = max(turn_number, 0)

    def system_prompt_block(self) -> str:
        if not self._active:
            return ""
        lines = [
            "# Supermemory",
            f"Active. Container: {self._container_tag}.",
            "Use supermemory-search, supermemory-save, supermemory-forget, and supermemory-profile (aliases: supermemory_search, supermemory_store, supermemory_forget, supermemory_profile).",
        ]
        if self._enable_custom_containers and self._custom_containers:
            tags_str = ", ".join(self._allowed_containers)
            lines.append(f"\nMulti-container mode enabled. Available containers: {tags_str}.")
            lines.append("Pass an optional container_tag to supermemory_search, supermemory_store, supermemory_forget, and supermemory_profile to target a specific container.")
            if self._custom_container_instructions:
                lines.append(f"\n{self._custom_container_instructions}")
        return "\n".join(lines)

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._active or not self._auto_recall or not self._client or not query.strip():
            return ""
        try:
            profile = self._client.get_profile(query=query[:200])
            include_profile = self._turn_count <= 1 or (self._turn_count % self._profile_frequency == 0)
            context = _format_prefetch_context(
                static_facts=profile["static"] if include_profile else [],
                dynamic_facts=profile["dynamic"] if include_profile else [],
                search_results=profile["search_results"],
                max_results=self._max_recall_results,
            )
            return context
        except Exception:
            logger.debug("Supermemory prefetch failed", exc_info=True)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._active or not self._auto_capture or not self._write_enabled or not self._client:
            return

        clean_user = _clean_text_for_capture(user_content)
        clean_assistant = _clean_text_for_capture(assistant_content)
        if not clean_user and not clean_assistant:
            return

        # Buffer every turn for the single full-session document written at end/switch/shutdown
        self._session_turns.append({"user": clean_user, "assistant": clean_assistant})

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        if not self._active or not self._write_enabled or not self._client or not self._session_id:
            return
        cleaned = []
        for message in messages or []:
            role = message.get("role")
            if role not in {"user", "assistant"}:
                continue
            content = _clean_text_for_capture(str(message.get("content", "")))
            if content:
                cleaned.append({"role": role, "content": content})
        if not cleaned:
            return
        if len(cleaned) == 1 and len(cleaned[0].get("content", "")) < 20:
            return
        try:
            self._client.ingest_conversation(
                self._session_id,
                cleaned,
                metadata={
                    "type": "full_session",
                    "session_id": self._session_id,
                    "message_count": len(cleaned),
                },
            )
        except urllib.error.HTTPError:
            logger.warning("Supermemory session ingest failed", exc_info=True)
        except Exception:
            logger.warning("Supermemory session ingest failed", exc_info=True)

        # Clear buffer so shutdown() doesn't duplicate on normal exit
        self._session_turns = []

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        """Flush any buffered turns from the old session as one document, then reset for the new session."""
        if not self._active or not self._write_enabled or not self._client:
            self._session_id = str(new_session_id or "").strip() or self._session_id
            self._session_turns = []
            return

        old_session_id = self._session_id
        old_turns = list(self._session_turns)

        # Flush previous session via conversations ingest (with metadata)
        if old_turns and old_session_id:
            messages: list[dict] = []
            for turn in old_turns:
                if turn.get("user"):
                    messages.append({"role": "user", "content": turn["user"]})
                if turn.get("assistant"):
                    messages.append({"role": "assistant", "content": turn["assistant"]})

            try:
                self._client.ingest_conversation(
                    old_session_id,
                    messages,
                    metadata={
                        "type": "full_session",
                        "session_id": old_session_id,
                        "message_count": len(old_turns) * 2,
                        "partial": not reset,
                    },
                )
            except Exception:
                logger.debug("Supermemory session-switch ingest failed", exc_info=True)

        # Reset for new session
        self._session_id = str(new_session_id or "").strip() or old_session_id
        self._session_turns = []
        self._turn_count = 0

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        if not self._active or not self._write_enabled or not self._client:
            return
        if action != "add" or not (content or "").strip():
            return

        def _run():
            try:
                self._client.add_memory(
                    content.strip(),
                    metadata={"target": target, "type": "explicit_memory"},
                    entity_context=self._entity_context,
                )
            except Exception:
                logger.debug("Supermemory on_memory_write failed", exc_info=True)

        if self._write_thread and self._write_thread.is_alive():
            self._write_thread.join(timeout=2.0)
        self._write_thread = None
        self._write_thread = threading.Thread(target=_run, daemon=False, name="supermemory-memory-write")
        self._write_thread.start()

    def shutdown(self) -> None:
        # Emergency fallback (crashes only). Buffer is cleared on normal on_session_end().
        if self._active and self._write_enabled and self._client and self._session_turns and self._session_id:
            logger.warning("Supermemory: Saving session via shutdown (session=%s, turns=%d)", self._session_id, len(self._session_turns))

            messages: list[dict] = []
            for turn in self._session_turns:
                if turn.get("user"):
                    messages.append({"role": "user", "content": turn["user"]})
                if turn.get("assistant"):
                    messages.append({"role": "assistant", "content": turn["assistant"]})

            try:
                self._client.ingest_conversation(
                    self._session_id,
                    messages,
                    metadata={
                        "type": "full_session",
                        "session_id": self._session_id,
                        "message_count": len(self._session_turns) * 2,
                        "partial": True,
                    },
                )
            except Exception:
                logger.debug("Supermemory shutdown ingest failed", exc_info=True)

        for attr_name in ("_prefetch_thread", "_sync_thread", "_write_thread"):
            thread = getattr(self, attr_name, None)
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
            setattr(self, attr_name, None)

    def _resolve_tool_container_tag(self, args: dict) -> Optional[str]:
        """Validate and resolve container_tag from tool call args.

        Returns None (use primary) if multi-container is disabled or no tag provided.
        Returns the validated tag if it's in the allowed list.
        Raises ValueError if the tag is not whitelisted.
        """
        if not self._enable_custom_containers:
            return None
        tag = str(args.get("container_tag") or "").strip()
        if not tag:
            return None
        sanitized = _sanitize_tag(tag)
        if sanitized not in self._allowed_containers:
            raise ValueError(
                f"Container tag '{sanitized}' is not allowed. "
                f"Allowed: {', '.join(self._allowed_containers)}"
            )
        return sanitized

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        def with_kebab_aliases(schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            aliases = {
                "supermemory_store": "supermemory-save",
                "supermemory_search": "supermemory-search",
                "supermemory_forget": "supermemory-forget",
                "supermemory_profile": "supermemory-profile",
            }
            expanded = list(schemas)
            for schema in schemas:
                kebab = aliases.get(schema.get("name", ""))
                if not kebab:
                    continue
                copy = json.loads(json.dumps(schema))
                copy["name"] = kebab
                expanded.append(copy)
            return expanded

        if not self._enable_custom_containers:
            return with_kebab_aliases([STORE_SCHEMA, SEARCH_SCHEMA, FORGET_SCHEMA, PROFILE_SCHEMA])

        # When multi-container is enabled, add optional container_tag to relevant tools
        container_param = {
            "type": "string",
            "description": f"Optional container tag. Allowed: {', '.join(self._allowed_containers)}. Defaults to primary ({self._container_tag}).",
        }
        schemas = []
        for base in [STORE_SCHEMA, SEARCH_SCHEMA, FORGET_SCHEMA, PROFILE_SCHEMA]:
            schema = json.loads(json.dumps(base))  # deep copy
            schema["parameters"]["properties"]["container_tag"] = container_param
            schemas.append(schema)
        return with_kebab_aliases(schemas)

    def _tool_store(self, args: dict) -> str:
        content = str(args.get("content") or "").strip()
        if not content:
            return tool_error("content is required")
        try:
            tag = self._resolve_tool_container_tag(args)
        except ValueError as exc:
            return tool_error(str(exc))
        metadata = args.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata.setdefault("type", _detect_category(content))
        metadata.pop("source", None)
        try:
            result = self._client.add_memory(content, metadata=metadata, entity_context=self._entity_context, container_tag=tag)
            preview = content[:80] + ("..." if len(content) > 80 else "")
            resp: dict[str, Any] = {"saved": True, "id": result.get("id", ""), "preview": preview}
            if tag:
                resp["container_tag"] = tag
            return json.dumps(resp)
        except Exception as exc:
            return tool_error(f"Failed to store memory: {exc}")

    def _tool_search(self, args: dict) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return tool_error("query is required")
        try:
            tag = self._resolve_tool_container_tag(args)
        except ValueError as exc:
            return tool_error(str(exc))
        try:
            limit = max(1, min(20, int(args.get("limit", 5) or 5)))
        except Exception:
            limit = 5
        try:
            results = self._client.search_memories(query, limit=limit, container_tag=tag)
            formatted = []
            for item in results:
                entry: dict[str, Any] = {"id": item.get("id", ""), "content": item.get("memory", "")}
                if item.get("similarity") is not None:
                    try:
                        entry["similarity"] = round(float(item["similarity"]) * 100)
                    except Exception:
                        pass
                formatted.append(entry)
            resp: dict[str, Any] = {"results": formatted, "count": len(formatted)}
            if tag:
                resp["container_tag"] = tag
            return json.dumps(resp)
        except Exception as exc:
            return tool_error(f"Search failed: {exc}")

    def _tool_forget(self, args: dict) -> str:
        memory_id = str(args.get("id") or "").strip()
        query = str(args.get("query") or "").strip()
        if not memory_id and not query:
            return tool_error("Provide either id or query")
        try:
            tag = self._resolve_tool_container_tag(args)
        except ValueError as exc:
            return tool_error(str(exc))
        try:
            if memory_id:
                self._client.forget_memory(memory_id, container_tag=tag)
                return json.dumps({"forgotten": True, "id": memory_id})
            return json.dumps(self._client.forget_by_query(query, container_tag=tag))
        except Exception as exc:
            return tool_error(f"Forget failed: {exc}")

    def _tool_profile(self, args: dict) -> str:
        query = str(args.get("query") or "").strip() or None
        try:
            tag = self._resolve_tool_container_tag(args)
        except ValueError as exc:
            return tool_error(str(exc))
        try:
            profile = self._client.get_profile(query=query, container_tag=tag)
            sections = []
            if profile["static"]:
                sections.append("## User Profile (Persistent)\n" + "\n".join(f"- {item}" for item in profile["static"]))
            if profile["dynamic"]:
                sections.append("## Recent Context\n" + "\n".join(f"- {item}" for item in profile["dynamic"]))
            resp: dict[str, Any] = {
                "profile": "\n\n".join(sections),
                "static_count": len(profile["static"]),
                "dynamic_count": len(profile["dynamic"]),
            }
            if tag:
                resp["container_tag"] = tag
            return json.dumps(resp)
        except Exception as exc:
            return tool_error(f"Profile failed: {exc}")

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if not self._active or not self._client:
            return tool_error("Supermemory is not configured")
        aliases = {
            "supermemory-save": "supermemory_store",
            "supermemory-search": "supermemory_search",
            "supermemory-forget": "supermemory_forget",
            "supermemory-profile": "supermemory_profile",
        }
        tool_name = aliases.get(tool_name, tool_name)
        if tool_name == "supermemory_store":
            return self._tool_store(args)
        if tool_name == "supermemory_search":
            return self._tool_search(args)
        if tool_name == "supermemory_forget":
            return self._tool_forget(args)
        if tool_name == "supermemory_profile":
            return self._tool_profile(args)
        return tool_error(f"Unknown tool: {tool_name}")


def register(ctx):
    ctx.register_memory_provider(SupermemoryMemoryProvider())
