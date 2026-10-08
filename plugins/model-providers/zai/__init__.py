"""ZAI / GLM provider profile.

GLM-4.5+ defaults to thinking ON, so ``reasoning_config`` is translated to
``extra_body.thinking``; GLM-5.2/5.3 also take a native ``reasoning_effort``.
"""

import hashlib
import logging
import re
import time
from typing import Any

from agent import reasoning_effort as re_
from providers import register_provider
from providers.base import ProviderProfile

# Same logger the probe used in hermes_cli.auth, so existing log filters keep matching.
logger = logging.getLogger("hermes_cli.auth")

_GLM_VERSION_RE = re.compile(r"^glm-(\d+)(?:\.(\d+))?")
# Alias spellings seen on relays (Fireworks ``glm-5p2``, ``zai-org-glm-5-2``…).
_GLM_5_3_TOKENS = ("glm-5.3", "glm-5-3", "glm-5p3")
_GLM_5_2_TOKENS = ("glm-5.2", "glm-5-2", "glm-5p2") + _GLM_5_3_TOKENS


def _model_supports_thinking(model: str | None) -> bool:
    """GLM thinking-capable model families: glm-4.5 and later (4.5, 4.6, 5…)."""
    match = _GLM_VERSION_RE.match((model or "").strip().lower())
    return bool(match) and (int(match.group(1)), int(match.group(2) or 0)) >= (4, 5)


def _has_token(model: str | None, tokens: tuple[str, ...]) -> bool:
    m = (model or "").strip().lower()
    return any(token in m for token in tokens)


def _glm_5_2_reasoning_effort(reasoning_config: dict | None, *, model: str | None = None) -> str | None:
    """Hermes effort -> GLM vocabulary (5.2: high/max; 5.3: low..max). Below-floor
    efforts clamp to the floor; disabled/unset leaves the server default."""
    effort = re_.requested_effort(reasoning_config)
    if effort is None or effort == "none":
        return None
    if _has_token(model, _GLM_5_3_TOKENS):
        efforts, overrides, floor = re_.GLM53_EFFORTS, re_.GLM53_OVERRIDES, "low"
    else:
        efforts, overrides, floor = re_.GLM52_EFFORTS, re_.GLM52_OVERRIDES, "high"
    clamped = re_.clamp_effort(effort, efforts, overrides)
    return clamped if clamped in efforts else floor


class ZaiProfile(ProviderProfile):
    """Z.AI / GLM — extra_body.thinking on/off + GLM-5.2 reasoning_effort."""

    def resolve_base_url(self, *, api_key: str, default_url: str, env_url: str, probe: bool = True) -> str:
        """Z.AI base URL by probing endpoints; an explicit GLM_BASE_URL always wins.

        The detected endpoint is cached in provider state (auth.json) keyed on a hash of the API key so
        subsequent starts skip the probe. ``probe=False`` (status display) never probes or reads the
        cache: it returns the override or the registry default.
        """
        if not probe:
            return super().resolve_base_url(api_key=api_key, default_url=default_url, env_url=env_url, probe=False)
        from hermes_cli import auth_zai_kimi
        from hermes_cli.auth import _auth_store_lock, _load_auth_store, _load_provider_state, _save_auth_store, _store_provider_state, detect_zai_endpoint
        if env_url:
            return env_url
        # No key -> don't probe (N×M 401s); auxiliary-client auto-detection hits this for everyone.
        if not api_key:
            return default_url

        key_hash = hashlib.sha256(api_key.encode()).hexdigest()[:16]
        state = _load_provider_state(_load_auth_store(), "zai") or {}
        cached = state.get("detected_endpoint")
        if isinstance(cached, dict) and cached.get("base_url") and cached.get("key_hash", "") == key_hash:
            logger.debug("Z.AI: using cached endpoint %s", cached["base_url"])
            return cached["base_url"]
        # Only a success is persisted, so a failing key (429/401 on every endpoint) would re-run the
        # four chat-completion probes on every credential-pool load — dozens of times per picker open.
        if auth_zai_kimi._zai_probe_failed_until.get(key_hash, 0.0) > time.time():
            return default_url

        # Probe — may take up to ~8s per endpoint.
        detected = detect_zai_endpoint(api_key)
        if not (detected and detected.get("base_url")):
            logger.debug("Z.AI: probe failed, falling back to default %s", default_url)
            auth_zai_kimi._zai_probe_failed_until[key_hash] = time.time() + auth_zai_kimi._ZAI_PROBE_FAILURE_TTL_SECONDS
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
            logger.warning("Z.AI: could not persist detected endpoint (%s); will re-probe next start", exc, exc_info=True)
        logger.info("Z.AI: auto-detected endpoint %s (%s)", detected["label"], detected["base_url"])
        return detected["base_url"]

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, model: str | None = None, **context
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        extra_body: dict[str, Any] = {}
        top_level: dict[str, Any] = {}
        is_5_2 = _has_token(model, _GLM_5_2_TOKENS)
        if not _model_supports_thinking(model) and not is_5_2:
            return extra_body, top_level
        # Only emit when the user expressed a preference (server default = enabled).
        if isinstance(reasoning_config, dict):
            enabled = reasoning_config.get("enabled") is not False
            if not enabled and _has_token(model, _GLM_5_3_TOKENS):
                # GLM-5.3 rejects disabled thinking; low is its lightest supported mode.
                extra_body["thinking"] = {"type": "enabled"}
                top_level["reasoning_effort"] = "low"
            else:
                extra_body["thinking"] = {"type": "enabled" if enabled else "disabled"}
        if is_5_2:
            effort = _glm_5_2_reasoning_effort(reasoning_config, model=model)
            if effort is not None:
                top_level["reasoning_effort"] = effort
        return extra_body, top_level


zai = ZaiProfile(
    name="zai", aliases=("glm", "z-ai", "z.ai", "zhipu"),
    env_vars=("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"), display_name="Z.AI (GLM)",
    description="Z.AI / GLM — Zhipu AI models", signup_url="https://z.ai/",
    fallback_models=("glm-5.2", "glm-5", "glm-4-9b"), base_url="https://api.z.ai/api/paas/v4",
    default_aux_model="glm-4.5-flash",
)

register_provider(zai)
