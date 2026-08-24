"""Kimi Code and Z.AI endpoint auto-detection, LM Studio base-URL normalization.

Re-exported from ``hermes_cli/auth.py`` (patch targets unchanged); origin helpers are imported
lazily per function so ``hermes_cli.auth.<helper>`` patches still intercept and no cycle forms.
"""

from __future__ import annotations

import logging
import hashlib
from typing import Dict, Optional
from hermes_cli.auth_constants import httpx

logger = logging.getLogger("hermes_cli.auth")

# "sk-kimi-" keys only work on api.kimi.com/coding; legacy moonshot keys use the old default.
# NO /v1 suffix: the anthropic SDK appends "/v1/messages" itself ("/coding/v1" would 404).
KIMI_CODE_BASE_URL = "https://api.kimi.com/coding"


def _resolve_kimi_base_url(api_key: str, default_url: str, env_override: str) -> str:
    """Kimi base URL from the key prefix; an explicit KIMI_BASE_URL always wins."""
    if env_override:
        return env_override
    if api_key and api_key.startswith("sk-kimi-"):
        return KIMI_CODE_BASE_URL
    return default_url


# Z.AI bills general/coding plans and global/China endpoints separately ("Insufficient balance" on
# the wrong one), so probe once and cache. Candidate models are tried in order: newer coding-plan
# accounts may only have recent GLM slugs, older ones still glm-4.7.
_ZAI_CODING_PROBE_MODELS = ["glm-5.3", "glm-5.3-flash", "glm-5.2", "glm-5.1", "glm-5v-turbo", "glm-4.7"]
ZAI_ENDPOINTS = [
    # (id, base_url, probe_models, label)
    ("global",        "https://api.z.ai/api/paas/v4",        ["glm-5"],   "Global"),
    ("cn",            "https://open.bigmodel.cn/api/paas/v4", ["glm-5"],   "China"),
    ("coding-global", "https://api.z.ai/api/coding/paas/v4",  _ZAI_CODING_PROBE_MODELS, "Global (Coding Plan)"),
    ("coding-cn",     "https://open.bigmodel.cn/api/coding/paas/v4", _ZAI_CODING_PROBE_MODELS, "China (Coding Plan)"),
]


def _probe_single_zai_endpoint(api_key: str, endpoint: tuple, timeout: float) -> Optional[Dict[str, str]]:
    """Probe one Z.AI endpoint, trying its candidate models in order; None when none succeeds."""
    ep_id, base_url, probe_models, label = endpoint
    for model in probe_models:
        try:
            resp = httpx.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "stream": False, "max_tokens": 1, "messages": [{"role": "user", "content": "ping"}]},
                timeout=timeout,
            )
            if resp.status_code == 200:
                logger.debug("Z.AI endpoint probe: %s (%s) model=%s OK", ep_id, base_url, model)
                return {"id": ep_id, "base_url": base_url, "model": model, "label": label}
            logger.debug("Z.AI endpoint probe: %s model=%s returned %s", ep_id, model, resp.status_code)
        except Exception as exc:
            logger.debug("Z.AI endpoint probe: %s model=%s failed: %s", ep_id, model, exc)
    return None


def detect_zai_endpoint(api_key: str, timeout: float = 8.0) -> Optional[Dict[str, str]]:
    """Probe z.ai endpoints in parallel; first working one in ZAI_ENDPOINTS priority order, or None."""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    # No `with`: it would join ALL probes on exit, defeating the early return below.
    pool = ThreadPoolExecutor(max_workers=len(ZAI_ENDPOINTS))
    try:
        futures = {pool.submit(_probe_single_zai_endpoint, api_key, ep, timeout): ep[0] for ep in ZAI_ENDPOINTS}
        by_id = {ep_id: f for f, ep_id in futures.items()}
        results: Dict[str, Dict[str, str]] = {}

        def _first_ready(require_done: bool) -> Optional[Dict[str, str]]:
            # Walk endpoints in PRIORITY order; a lower-priority success only wins once every
            # higher-priority probe has finished without success.
            for ep in ZAI_ENDPOINTS:
                if require_done and not by_id[ep[0]].done():
                    return None  # a higher-priority probe is still in flight
                if ep[0] in results:
                    return results[ep[0]]
            return None

        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    results[futures[future]] = result
            except Exception:
                pass
            winner = _first_ready(require_done=True)
            if winner is not None:
                return winner
        return _first_ready(require_done=False)
    finally:
        pool.shutdown(wait=False)


def _resolve_zai_base_url(api_key: str, default_url: str, env_override: str) -> str:
    """Z.AI base URL by probing endpoints; an explicit GLM_BASE_URL always wins.

    The detected endpoint is cached in provider state (auth.json) keyed on a hash of the API key so
    subsequent starts skip the probe.
    """
    from hermes_cli.auth import _auth_store_lock, _load_auth_store, _load_provider_state, _save_auth_store, _store_provider_state, detect_zai_endpoint
    if env_override:
        return env_override
    # No key -> don't probe (N×M 401s); auxiliary-client auto-detection hits this for everyone.
    if not api_key:
        return default_url

    key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
    state = _load_provider_state(_load_auth_store(), "zai") or {}
    cached = state.get("detected_endpoint")
    if isinstance(cached, dict) and cached.get("base_url") and cached.get("key_hash", "") == key_hash:
        logger.debug("Z.AI: using cached endpoint %s", cached["base_url"])
        return cached["base_url"]

    # Probe — may take up to ~8s per endpoint.
    detected = detect_zai_endpoint(api_key)
    if not (detected and detected.get("base_url")):
        logger.debug("Z.AI: probe failed, falling back to default %s", default_url)
        return default_url

    detected_endpoint = {
        "base_url": detected["base_url"], "endpoint_id": detected.get("id", ""),
        "model": detected.get("model", ""), "label": detected.get("label", ""),
        "key_hash": key_hash,
    }
    # Persist failure must not break resolution; worst case the next start re-probes.
    try:
        with _auth_store_lock():
            auth_store = _load_auth_store()  # reload under lock to avoid overwriting concurrent changes
            state_under_lock = _load_provider_state(auth_store, "zai") or {}
            state_under_lock["detected_endpoint"] = detected_endpoint
            # set_active=False: runs from credential-pool env seeding; must not flip active provider.
            _store_provider_state(auth_store, "zai", state_under_lock, set_active=False)
            _save_auth_store(auth_store)
    except Exception as exc:
        logger.warning("Z.AI: could not persist detected endpoint (%s); will re-probe next start", exc)
    logger.info("Z.AI: auto-detected endpoint %s (%s)", detected["label"], detected["base_url"])
    return detected["base_url"]


def _normalize_lmstudio_runtime_base_url(base_url: str) -> str:
    """Return the OpenAI-compatible LM Studio runtime base URL.

    LM Studio's native management API lives under ``/api/v1`` while its OpenAI-compatible chat
    endpoint lives under ``/v1``; users paste either form, so normalize before the SDK appends
    ``/chat/completions``.
    """
    root = str(base_url or "").strip().rstrip("/")
    for suffix in ("/api/v1", "/api", "/v1"):
        if root.endswith(suffix):
            root = root[: -len(suffix)].rstrip("/")
            break
    return (root or "http://127.0.0.1:1234") + "/v1"
