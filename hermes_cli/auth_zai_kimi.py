"""Kimi Code endpoint constant, Z.AI endpoint auto-detection, LM Studio base-URL normalization.

Re-exported from ``hermes_cli/auth.py`` (patch targets unchanged); origin helpers are imported
lazily per function so ``hermes_cli.auth.<helper>`` patches still intercept and no cycle forms.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional
from hermes_cli.auth_constants import httpx

logger = logging.getLogger("hermes_cli.auth")

# In-process negative cache for Z.AI endpoint detection (ZaiProfile.resolve_base_url), keyed by key
# hash: a failed probe is not retried for this long (a success persists to auth.json instead).
_ZAI_PROBE_FAILURE_TTL_SECONDS = 300
_zai_probe_failed_until: Dict[str, float] = {}

# "sk-kimi-" keys only work on api.kimi.com/coding; legacy moonshot keys use the old default.
# NO /v1 suffix: the anthropic SDK appends "/v1/messages" itself ("/coding/v1" would 404).
KIMI_CODE_BASE_URL = "https://api.kimi.com/coding"


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
