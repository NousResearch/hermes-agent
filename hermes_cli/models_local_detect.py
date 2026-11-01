"""Find model servers already running on this machine, for the provider pickers.

Ollama, LM Studio and vLLM each listen on a well-known localhost port and answer the
OpenAI-compatible ``GET /v1/models``. Users running one had to type its address into the custom
endpoint flow by hand; the pickers now offer it as a one-click ``Detected: Ollama (3 models)`` row.

Detection only runs when a picker opens (``hermes setup`` / ``hermes model``, ``/model``, the
Desktop model picker). It never touches the network otherwise. Probes go to 127.0.0.1 only, in
parallel, with a short deadline. They send no credential, ignore proxy settings, never follow a
redirect, and cap how much they read. Model ids come only from the server's own answer.

Picking a detected row writes an ordinary ``providers.<key>`` custom-endpoint entry; from then on
it is a saved provider like any other. Nothing is ever selected automatically.
"""

from __future__ import annotations

import http.client
import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field

# 127.0.0.1, never "localhost": resolving localhost costs ~2s per request on Windows.
_HOST = "127.0.0.1"
_PROBE_TIMEOUT_S = 1.0
_MAX_BODY_BYTES = 1 << 20
_CACHE_TTL_S = 30.0
PROVIDER_KEY_PREFIX = "local-"


@dataclass(frozen=True)
class _Kind:
    kind: str
    label: str
    port: int


# llama-server's :8080 is owned by the managed local runtime (``local_runtime/detect.py``).
KNOWN_SERVERS: tuple[_Kind, ...] = (
    _Kind("ollama", "Ollama", 11434),
    _Kind("lmstudio", "LM Studio", 1234),
    _Kind("vllm", "vLLM", 8000),
)
_KIND_BY_NAME = {k.kind: k for k in KNOWN_SERVERS}
# ``owned_by`` in each server's /v1/models rows: names the server even off its default port.
_OWNER_KIND = {"library": "ollama", "ollama": "ollama", "organization_owner": "lmstudio", "vllm": "vllm"}


@dataclass(frozen=True)
class DetectedServer:
    kind: str
    label: str
    port: int
    base_url: str
    models: tuple[str, ...] = field(default_factory=tuple)

    @property
    def provider_key(self) -> str:
        """The ``providers:`` key a pick writes: ``local-ollama``, ``local-vllm-8001``."""
        known = _KIND_BY_NAME.get(self.kind)
        suffix = "" if known and known.port == self.port else f"-{self.port}"
        return f"{PROVIDER_KEY_PREFIX}{self.kind}{suffix}"

    @property
    def slug(self) -> str:
        """Picker slug: a ``providers:`` entry's rows use its bare key, so the saved row matches."""
        return self.provider_key

    @property
    def display_name(self) -> str:
        return f"{self.label} (local)"

    @property
    def picker_name(self) -> str:
        """Picker row name; the pickers print the model count next to it themselves."""
        return f"Detected: {self.label}"

    @property
    def row_label(self) -> str:
        count = len(self.models)
        return f"Detected: {self.label} ({count} model{'' if count == 1 else 's'})"


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *_args, **_kwargs):
        return None


# No ProxyHandler entries: a system proxy must never see (or answer for) a loopback probe.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())


def _read_body(r, deadline: float) -> bytes | None:
    """The response body, or ``None`` past the size cap or the deadline. The socket timeout only
    bounds each read, so a server dripping bytes just under it would otherwise hold the probe."""
    chunks: list[bytes] = []
    size = 0
    while chunk := r.read1(64 * 1024):
        size += len(chunk)
        if size > _MAX_BODY_BYTES or time.monotonic() > deadline:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


def _get_json(url: str, timeout: float):
    deadline = time.monotonic() + timeout
    try:
        with _OPENER.open(urllib.request.Request(url, headers={"Accept": "application/json"}), timeout=timeout) as r:
            if r.status != 200:
                return None
            body = _read_body(r, deadline)
    # HTTPException: whatever answers on a scanned port may speak broken HTTP (bad status line,
    # truncated chunked body); that port yields no row instead of failing the whole scan.
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
        return None
    if body is None:
        return None
    try:
        return json.loads(body)
    except (ValueError, RecursionError):
        return None


def _openai_models(payload) -> tuple[list[str], str]:
    """``(ids, owner)`` from an OpenAI ``{"data": [{"id", "owned_by"}]}`` list."""
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return [], ""
    ids = [str(r["id"]).strip() for r in rows if isinstance(r, dict) and str(r.get("id") or "").strip()]
    owners = {str(r.get("owned_by") or "").strip().lower() for r in rows if isinstance(r, dict)}
    owner = next((o for o in owners if o in _OWNER_KIND), "")
    return ids, owner


def _ollama_tags(payload) -> list[str]:
    rows = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return []
    return [str(r.get("name") or r.get("model")).strip() for r in rows
            if isinstance(r, dict) and str(r.get("name") or r.get("model") or "").strip()]


def probe_port(port: int, *, expected: _Kind | None = None, timeout: float = _PROBE_TIMEOUT_S) -> DetectedServer | None:
    """One loopback port: an OpenAI-compatible model list (Ollama's native ``/api/tags`` as a
    fallback on its own port). ``None`` when nothing answers in that shape."""
    root = f"http://{_HOST}:{port}"
    ids, owner = _openai_models(_get_json(f"{root}/v1/models", timeout))
    if not ids and expected is not None and expected.kind == "ollama":
        ids = _ollama_tags(_get_json(f"{root}/api/tags", timeout))
        owner = "ollama" if ids else ""
    if not ids:
        return None
    kind = _KIND_BY_NAME.get(_OWNER_KIND.get(owner, "")) or expected
    if kind is None:
        return None
    return DetectedServer(kind.kind, kind.label, port, f"{root}/v1", tuple(dict.fromkeys(ids)))


_cache_lock = threading.Lock()
_cache: tuple[float, list[DetectedServer]] | None = None


def detect_local_servers(
    *, refresh: bool = False, cached_only: bool = False, timeout: float = _PROBE_TIMEOUT_S,
) -> list[DetectedServer]:
    """Every known local server answering on its default port, probed in parallel. The whole scan
    is bounded by ``timeout`` (a refused port returns at once); results are reused for a short
    while so repeated picker opens don't re-probe. ``cached_only`` never probes: it returns the
    last scan's result (any age), or nothing when no picker has scanned yet."""
    global _cache
    with _cache_lock:
        if cached_only:
            return list(_cache[1]) if _cache is not None else []
        if not refresh and _cache is not None and time.monotonic() - _cache[0] < _CACHE_TTL_S:
            return list(_cache[1])
    hits: dict[int, DetectedServer | None] = {}

    def _probe(k: _Kind) -> None:
        hits[k.port] = probe_port(k.port, expected=k, timeout=timeout)

    from agent.memory_provider import spawn_context_thread

    # Daemon threads: a probe still running when the scan gives up never holds interpreter exit.
    threads = [spawn_context_thread(_probe, args=(k,), name=f"local-model-probe-{k.port}") for k in KNOWN_SERVERS]
    for t in threads:
        t.start()
    # Ollama's /api/tags fallback can add a second request; never wait past two probe budgets.
    give_up = time.monotonic() + timeout * 2 + 0.2
    for t in threads:
        t.join(max(0.0, give_up - time.monotonic()))
    found = [hit for k in KNOWN_SERVERS if (hit := hits.get(k.port)) is not None]
    with _cache_lock:
        _cache = (time.monotonic(), found)
    return list(found)


def _norm_url(url: str) -> str:
    url = str(url or "").strip().rstrip("/").lower().replace("://localhost", f"://{_HOST}")
    return url.removesuffix("/v1")


def configured_base_urls(config: dict) -> set[str]:
    """Normalized base URLs already saved in ``providers:`` / ``custom_providers:`` / ``model:``."""
    urls: set[str] = set()
    entries = []
    providers = config.get("providers")
    if isinstance(providers, dict):
        entries.extend(e for e in providers.values() if isinstance(e, dict))
    custom = config.get("custom_providers")
    if isinstance(custom, list):
        entries.extend(e for e in custom if isinstance(e, dict))
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        entries.append(model_cfg)
    for entry in entries:
        for key in ("base_url", "url", "api"):
            if entry.get(key):
                urls.add(_norm_url(entry[key]))
    urls.discard("")
    return urls


def _taken_keys(config: dict) -> set[str]:
    providers = config.get("providers")
    return {str(k).strip().lower() for k in providers} if isinstance(providers, dict) else set()


def unsaved_servers(config: dict, *, cached_only: bool = False) -> list[DetectedServer]:
    """Detected servers the user has not saved yet (a saved one already has its own row). A server
    whose ``providers:`` key the user already gave to another address is not offered: picking it
    would select that other entry."""
    saved = configured_base_urls(config)
    taken = _taken_keys(config)
    return [s for s in detect_local_servers(cached_only=cached_only)
            if _norm_url(s.base_url) not in saved and s.provider_key not in taken]


def picker_row(server: DetectedServer, *, max_models: int | None = None) -> dict:
    """A ``list_authenticated_providers``-shaped row for /model, the TUI and Desktop."""
    models = list(server.models)
    return {
        "slug": server.slug, "name": server.picker_name, "is_current": False, "is_user_defined": False,
        "models": models[:max_models] if max_models is not None else models, "total_models": len(models),
        "source": "detected", "api_url": server.base_url, "native_catalog_empty": False,
    }


def find_detected(slug_or_key: str) -> DetectedServer | None:
    """The detected server behind a picker slug / ``providers:`` key (``local-ollama``)."""
    want = str(slug_or_key or "").strip().lower().removeprefix("custom:")
    if not want.startswith(PROVIDER_KEY_PREFIX):
        return None
    for refresh in (False, True):
        hit = next((s for s in detect_local_servers(refresh=refresh) if s.provider_key == want), None)
        if hit is not None:
            return hit
    return None


def save_detected_server(server: DetectedServer) -> dict | None:
    """Write ``providers.<key>`` for a picked server (no key: local servers take none) and return
    the entry. An entry already under that key is left as the user wrote it: returned when it points
    at this server, ``None`` when it points elsewhere (that key is not this server's to claim)."""
    from hermes_cli.config import _CONFIG_LOCK, read_user_config_raw, save_config

    entry = {"name": server.display_name, "base_url": server.base_url}
    # Check and write under one lock, merging only this entry into the file's latest content: two
    # chats picking at once (or a settings edit) never lose each other's write to an older copy.
    with _CONFIG_LOCK:
        providers = read_user_config_raw().get("providers")
        existing = providers.get(server.provider_key) if isinstance(providers, dict) else None
        if existing is not None:
            ours = _norm_url(server.base_url) in configured_base_urls({"providers": {server.provider_key: existing}})
            return existing if ours else None
        save_config({"providers": {server.provider_key: entry}}, merge_existing=True)
    return entry


def adopt_detected_provider(explicit_provider: str, user_providers: dict | None) -> dict | None:
    """``switch_model`` hook: a picked ``Detected:`` row's slug becomes a saved provider before the
    switch resolves it. Returns the ``user_providers`` dict to resolve against (unchanged for any
    slug that is not an unsaved detected server)."""
    key = str(explicit_provider or "").strip().lower().removeprefix("custom:")
    if not key.startswith(PROVIDER_KEY_PREFIX):
        return user_providers
    if isinstance(user_providers, dict) and key in user_providers:
        return user_providers
    server = find_detected(key)
    if server is None:
        return user_providers
    entry = save_detected_server(server)
    if entry is None:
        return user_providers
    merged = dict(user_providers) if isinstance(user_providers, dict) else {}
    merged[server.provider_key] = dict(entry)
    return merged
