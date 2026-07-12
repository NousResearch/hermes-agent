"""
Finetune adapter-routing plugin.

Registers ``llm_request`` middleware that routes prompts to the best
fine-tuned LoRA adapter based on cosine similarity with cluster centroids,
injecting the adapter as a per-request ``extra_body.lora_adapters`` entry
for local llama.cpp endpoints.

The routing decision is carried entirely on the request payload the
middleware returns — no environment variables or other process-global
state — so concurrent sessions and interleaved requests can't observe
each other's adapter selection.

This plugin ships inside the mlops/finetune optional skill
(``plugin/finetune-routing/``). ``/finetune route enable`` copies it into
``<hermes-home>/plugins/``, where the standard plugin discovery
(hermes_cli/plugins.py) loads it via this ``register(ctx)`` entry point.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("hermes.finetune.routing")

# Lazy-initialized router; failure is sticky so a broken install doesn't
# re-attempt (and re-log) on every request.
_router = None
_router_failed = False

_LOCAL_INDICATORS = ("localhost", "127.0.0.1", "0.0.0.0")


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
    global _router, _router_failed
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
                from route import AdapterRouter
            finally:
                if inserted:
                    try:
                        sys.path.remove(scripts_path)
                    except ValueError:
                        pass
            _router = AdapterRouter()
        except Exception as e:
            logger.warning("Finetune routing disabled — router init failed: %s", e)
            _router_failed = True
    return _router


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
    """``llm_request`` middleware: inject the routed LoRA adapter.

    Returns ``{"request": <rewritten kwargs>}`` when an adapter matches,
    or ``None`` to leave the request untouched.
    """
    request = kwargs.get("request")
    if not isinstance(request, dict):
        return None

    # Only route for local endpoints — remote providers can't load a LoRA
    # from a local path, and llama.cpp is what honors lora_adapters.
    base_url = (kwargs.get("base_url") or "").lower()
    if not any(ind in base_url for ind in _LOCAL_INDICATORS):
        return None

    router = _get_router()
    if router is None or not router.enabled:
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

    adapter_path = result.get("adapter_path")
    if not adapter_path:
        return None

    cluster_id = result.get("cluster_id", "unknown")
    logger.info(
        "Finetune routing: %s (cluster=%s, confidence=%.3f, adapter=%s)",
        result.get("label", ""), cluster_id,
        result.get("confidence", 0.0), adapter_path,
    )

    extra_body = dict(request.get("extra_body") or {})
    extra_body["lora_adapters"] = [{"path": adapter_path, "scale": 1.0}]
    return {
        "request": {**request, "extra_body": extra_body},
        "source": "finetune-routing",
        "reason": f"adapter cluster {cluster_id}",
    }


def register(ctx) -> None:
    """Plugin entry point (see hermes_cli/plugins.py)."""
    ctx.register_middleware("llm_request", finetune_llm_request_middleware)
    logger.debug("finetune-routing plugin registered llm_request middleware")
