"""
Finetune adapter-routing plugin.

Registers ``llm_request`` middleware that routes prompts against cluster
centroids and sets per-request LoRA scales for the adapters the managed
llama-server has PRELOADED (via ``--lora`` at startup).

llama.cpp cannot load an arbitrary adapter path per request. Its real
per-request API — a llama.cpp server extension field accepted through the
OpenAI-compatible endpoint — is ``"lora": [{"id": N, "scale": s}]``, where
``id`` is the positional index of a ``--lora`` flag in the server command.
So the middleware only activates when BOTH hold:

  - ``finetune.routing.enabled`` is true in config, AND
  - ``manage.py redeploy`` has written a serving manifest
    (``<finetune-dir>/serving.json``) describing the server it launched
    and the adapters that server preloaded.

If the routed cluster matches a served adapter, that adapter gets scale
1.0. If the prompt routes to a cluster that is NOT served (or to no
cluster at all), every served adapter gets scale 0.0 — off-domain prompts
deliberately fall back to the base model.

The routing decision is carried entirely on the request payload the
middleware returns — no environment variables or other process-global
state — so concurrent sessions and interleaved requests can't observe
each other's adapter selection.

This plugin ships inside the mlops/finetune optional skill
(``plugin/finetune-routing/``). ``/finetune route enable`` copies it into
``<hermes-home>/plugins/``, where the standard plugin discovery
(hermes_cli/plugins.py) loads it via this ``register(ctx)`` entry point.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

logger = logging.getLogger("hermes.finetune.routing")

# Lazy-initialized router; failure is sticky so a broken install doesn't
# re-attempt (and re-log) on every request.
_router = None
_router_failed = False
# The skill's `common` module (paths), captured alongside the router import.
_common_mod = None
# Serving manifest cache, invalidated by mtime so promotions/redeploys in
# other processes are picked up without restarting the session.
_manifest_cache: Dict[str, Any] = {"mtime_ns": None, "data": None}

# Hosts that all mean "this machine" — a request to any of them may be
# served by a llama-server bound to any other.
_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "0.0.0.0"}


def _find_scripts_dir() -> Optional[Path]:
    """Locate the finetune skill's scripts/ directory.

    Two layouts exist:
    - in-repo / in-bundle: this file lives at <skill>/plugin/finetune-routing/,
      so scripts/ is two levels up;
    - enabled install: this file lives at <hermes-home>/plugins/finetune-routing/,
      and the skill at <hermes-home>/skills/mlops/finetune/.
    """
    here = Path(__file__).resolve().parent
    candidates = [
        here.parent.parent / "scripts",
        here.parent.parent / "skills" / "mlops" / "finetune" / "scripts",
    ]
    for candidate in candidates:
        if (candidate / "route.py").exists():
            return candidate
    return None


def _get_router():
    global _router, _router_failed, _common_mod
    if _router is None and not _router_failed:
        try:
            scripts_dir = _find_scripts_dir()
            if scripts_dir is None:
                raise FileNotFoundError(
                    "finetune skill scripts not found next to the plugin"
                )
            # route.py does `from common import ...` — the skill scripts are
            # also run directly as files (`python scripts/route.py`), so they
            # can't use package-relative imports and a bare
            # spec_from_file_location load of route.py alone would fail.
            # Insert the scripts dir just long enough to import, then remove
            # it so the long-lived CLI process's sys.path isn't permanently
            # polluted.  Residual sys.modules entries ('route', 'common', …)
            # do remain after a successful import — an accepted tradeoff of
            # the scripts' flat top-level module names.
            scripts_path = str(scripts_dir)
            inserted = scripts_path not in sys.path
            if inserted:
                sys.path.insert(0, scripts_path)
            try:
                import common as common_mod
                from route import AdapterRouter
            finally:
                if inserted:
                    try:
                        sys.path.remove(scripts_path)
                    except ValueError:
                        pass
            _router = AdapterRouter()
            _common_mod = common_mod
        except Exception as e:
            logger.warning("Finetune routing disabled — router init failed: %s", e)
            _router_failed = True
    return _router


def _serving_manifest_path() -> Optional[Path]:
    if _common_mod is None:
        return None
    return Path(_common_mod.FINETUNE_DIR) / "serving.json"


def _load_serving_manifest() -> Optional[Dict[str, Any]]:
    """Load the manifest manage.py redeploy writes, cached by mtime.

    Returns None when no manifest exists — i.e. no server we manage is
    known to be running — which deactivates the middleware entirely.
    """
    path = _serving_manifest_path()
    if path is None:
        return None
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        _manifest_cache["mtime_ns"] = None
        _manifest_cache["data"] = None
        return None
    if _manifest_cache["mtime_ns"] != mtime_ns:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            data = None
        _manifest_cache["mtime_ns"] = mtime_ns
        _manifest_cache["data"] = data if isinstance(data, dict) else None
    return _manifest_cache["data"]


def _url_host(url: str) -> Optional[str]:
    try:
        host = urlsplit(url).hostname
    except ValueError:
        return None
    return host.lower() if host else None


def _host_matches_serving(base_url: str, health_url: str) -> bool:
    """Route only when the request targets the server the manifest describes.

    Compares parsed URL hosts — a substring test like `"localhost" in url`
    would match "notlocalhost.example.com" and leak local adapter routing
    to remote APIs. Loopback aliases (localhost/127.0.0.1/::1/0.0.0.0) are
    treated as equivalent. Anything unparseable fails closed.
    """
    request_host = _url_host(base_url or "")
    serving_host = _url_host(health_url or "")
    if not request_host or not serving_host:
        return False
    if request_host == serving_host:
        return True
    return request_host in _LOOPBACK_HOSTS and serving_host in _LOOPBACK_HOSTS


def _last_user_message(messages: List[Any]) -> str:
    """Extract the text of the most recent user message."""
    for msg in reversed(messages or []):
        if not isinstance(msg, dict) or msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = [
                p.get("text", "") for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            return "\n".join(p for p in parts if p)
    return ""


def finetune_llm_request_middleware(**kwargs) -> Optional[Dict[str, Any]]:
    """``llm_request`` middleware: set per-request LoRA scales.

    Returns ``{"request": <rewritten kwargs>}`` with
    ``extra_body["lora"] = [{"id": N, "scale": s}]`` (the llama.cpp server
    extension referencing adapters preloaded via ``--lora``), or ``None``
    to leave the request untouched.
    """
    request = kwargs.get("request")
    if not isinstance(request, dict):
        return None

    router = _get_router()
    if router is None or not router.enabled:
        return None

    # Only act when manage.py has actually deployed a server with adapters.
    manifest = _load_serving_manifest()
    if not manifest:
        return None
    served = [
        a for a in (manifest.get("adapters") or [])
        if isinstance(a, dict) and isinstance(a.get("id"), int)
    ]
    if not served:
        return None  # base-model-only deploy — nothing to scale

    health_url = str((manifest.get("server") or {}).get("health_url") or "")
    if not _host_matches_serving(kwargs.get("base_url") or "", health_url):
        return None

    user_message = _last_user_message(request.get("messages"))
    # Don't route empty or very short messages
    if not user_message or len(user_message.strip()) < 10:
        return None

    try:
        result = router.route(user_message)
    except Exception as e:
        logger.debug("Routing failed: %s", e)
        return None

    routed_cluster = result.get("cluster_id")
    lora = [
        {"id": a["id"], "scale": 1.0 if a.get("cluster") == routed_cluster else 0.0}
        for a in served
    ]
    if any(entry["scale"] == 1.0 for entry in lora):
        reason = f"adapter cluster {routed_cluster}"
    else:
        # Off-domain prompt: disable every preloaded adapter so the base
        # model answers. This is deliberate, not a failure mode.
        reason = "off-domain prompt — served adapters scaled to 0 (base model)"

    logger.info(
        "Finetune routing: %s (cluster=%s, confidence=%.3f, lora=%s)",
        result.get("label", ""), routed_cluster,
        result.get("confidence", 0.0), lora,
    )

    extra_body = dict(request.get("extra_body") or {})
    extra_body["lora"] = lora
    return {
        "request": {**request, "extra_body": extra_body},
        "source": "finetune-routing",
        "reason": reason,
    }


def register(ctx) -> None:
    """Plugin entry point (see hermes_cli/plugins.py)."""
    ctx.register_middleware("llm_request", finetune_llm_request_middleware)
    logger.debug("finetune-routing plugin registered llm_request middleware")
