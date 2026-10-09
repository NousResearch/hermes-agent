"""Nous Portal provider profile."""

from typing import Any

from agent.portal_tags import get_affinity_scope, get_conversation_context, nous_portal_tags
from agent.transports.codex import _cache_scope_from_session_id
from providers import register_provider
from providers.base import ProviderProfile
from providers.model_normalizers import VendorQualifiedModelIdsMixin


class NousProfile(VendorQualifiedModelIdsMixin, ProviderProfile):
    """Nous Portal — product tags, reasoning with Nous-specific omission."""

    def resolve_route_policy(
        self, model: str, base_url: str = "", *, options: dict[str, Any] | None = None
    ) -> str | None:
        """Resolve the Portal\'s model-dependent wire from already-read route policy."""
        del base_url
        if not str(model or "").strip().lower().startswith("anthropic/"):
            return None
        wire = str((options or {}).get("anthropic_wire") or "chat").strip().lower()
        return "anthropic_messages" if wire == "native" else "chat_completions"

    def fetch_recommended_models(
        self, *, base_url: str = "https://portal.nousresearch.com", timeout: float = 5.0
    ) -> dict[str, Any] | None:
        """Provider-owned public Portal fetch (no account credential or model selection)."""
        import gzip
        import json
        import urllib.request

        from hermes_cli.urllib_security import open_credentialed_url
        from models.catalog_nous_recommendations import RECOMMENDED_MODELS_PATH

        req = urllib.request.Request(
            base_url.rstrip("/") + RECOMMENDED_MODELS_PATH,
            headers={"Accept": "application/json", "Accept-Encoding": "gzip"},
        )
        with open_credentialed_url(req, timeout=timeout) as response:
            body = response.read()
            if response.headers.get("Content-Encoding", "").lower() == "gzip":
                body = gzip.decompress(body)
        payload = json.loads(body.decode("utf-8"))
        return payload if isinstance(payload, dict) else None

    def resolve_aux_model(self, *, vision: bool = False, force_refresh: bool = False) -> str:
        """Use application-resolved Portal/account facts; never import CLI model policy."""
        try:
            from application_nous_recommendations import auxiliary_model

            return auxiliary_model(vision=vision, force_refresh=force_refresh)
        except Exception:
            return ""

    def default_vision_model(self) -> str | None:
        """Provider-owned live vision default for auxiliary selection."""
        return self.resolve_aux_model(vision=True) or None

    def build_extra_body(self, *, session_id: str | None = None, **context) -> dict[str, Any]:
        body: dict[str, Any] = {"tags": nous_portal_tags(session_id=session_id)}
        # Top-level session_id = sticky routing key, so Anthropic-style cache
        # breakpoints stay warm on one upstream instance. Resolved like the
        # ``conversation=`` tag: declared scope, then the ambient lineage ROOT
        # (covers aux call sites that pass no session_id), then the explicit argument.
        sticky_key = _cache_scope_from_session_id(get_affinity_scope() or get_conversation_context() or session_id)
        if sticky_key:
            body["session_id"] = sticky_key
        # Nous Portal inference rejects caller-supplied provider routing prefs
        # (only/ignore/order/sort/data_collection/zdr/require_parameters) with
        # HTTP 400 — routing is decided centrally per model. provider_routing
        # from config.yaml is OpenRouter-only, so it is not forwarded here.
        return body

    @staticmethod
    def _cannot_disable_reasoning(model: str | None) -> bool:
        """True when ``reasoning: {enabled: false}`` would 400 on *model*. Cache-only catalog
        lookup; unknown/cold (warmer kicked) and no-reasoning routes both answer True (omit > 400)."""
        try:
            from models.metadata.reasoning import nous_model_reasoning_capabilities, warm_nous_reasoning_caps_async

            caps = nous_model_reasoning_capabilities(model)
            if caps is None:
                warm_nous_reasoning_caps_async()
                return True
        except Exception:
            return True
        return not caps.supported or bool(caps.mandatory)

    def build_api_kwargs_extras(
        self, *, reasoning_config: dict | None = None, supports_reasoning: bool = False,
        model: str | None = None, **context,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Pass the full reasoning_config, disable included (the Portal honors it;
        omitting it means the upstream default, thinking ON for V4-class models)."""
        if not supports_reasoning:
            return {}, {}
        if reasoning_config is None:
            return {"reasoning": {"enabled": True, "effort": "medium"}}, {}
        rc = dict(reasoning_config)
        if rc.get("enabled") is False and self._cannot_disable_reasoning(model):
            return {}, {}
        return {"reasoning": rc}, {}


nous = NousProfile(
    name="nous", aliases=("nous-portal", "nousresearch"), env_vars=("NOUS_API_KEY",),
    display_name="Nous Portal", description="Nous Research — Hermes model family",
    signup_url="https://nousresearch.com/", fallback_models=("hermes-3-405b", "hermes-3-70b"),
    base_url="https://inference-api.nousresearch.com/v1", auth_type="oauth_device_code",
    fallback_aux_model="google/gemini-3.6-flash",
)

register_provider(nous)
