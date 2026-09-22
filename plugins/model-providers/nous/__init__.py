"""Nous Portal provider profile."""

from typing import Any

from agent.portal_tags import get_affinity_scope, get_conversation_context, nous_portal_tags
from agent.transports.codex import _cache_scope_from_session_id
from providers import register_provider
from providers.base import ProviderProfile


# Fixed descriptions identify the operation without copying prompts, tool output,
# or arbitrary task names supplied by plugins into request metadata.
_AUXILIARY_PURPOSES = {
    "compression": "Summarize conversation context so the assistant can continue.",
    "title_generation": "Generate a title for the conversation.",
    "vision": "Interpret an image for the current assistant task.",
    "skills_hub": "Select relevant skills for the assistant task.",
    "approval": "Evaluate whether a requested tool action needs approval.",
    "mcp": "Complete a model sampling request from a connected tool.",
    "memory_query_rewrite": "Rewrite a query to retrieve relevant memories.",
    "tts_audio_tags": "Prepare speech delivery annotations.",
    "triage_specifier": "Expand a task description into a specification.",
    "kanban_decomposer": "Break a task into actionable subtasks.",
    "profile_describer": "Generate a short profile description.",
    "goal_judge": "Define or assess the completion criteria for a goal.",
    "curator": "Review skills and identify useful improvements.",
    "monitor": "Assess the relevance of a monitored item.",
    "background_review": "Review the conversation for memory and skill improvements.",
    "moa_reference": "Produce a candidate response for the current assistant task.",
    "moa_aggregator": "Synthesize candidate responses for the current assistant task.",
}


class NousProfile(ProviderProfile):
    """Nous Portal — product tags, reasoning with Nous-specific omission."""

    def resolve_aux_model(self, *, vision: bool = False) -> str:
        """Portal's tier-aware ``/api/nous/recommended-models`` pick (cached, offline-safe)."""
        try:
            from hermes_cli.models import get_nous_recommended_aux_model

            return get_nous_recommended_aux_model(vision=vision) or ""
        except Exception:
            return ""

    def build_extra_body(
        self, *, session_id: str | None = None, task: str | None = None,
        messages: list | None = None, **context,
    ) -> dict[str, Any]:
        if task is None:
            activity = "assistant_chat"
            purpose = (
                "Continue the assistant response using the requested tool results."
                if messages and messages[-1].get("role") == "tool"
                else "Respond to the user's latest message."
            )
        else:
            activity = task if task in _AUXILIARY_PURPOSES else "auxiliary"
            purpose = _AUXILIARY_PURPOSES.get(task, "Complete a supporting operation for the assistant.")
        metadata = {"hermes_activity": activity, "hermes_purpose": purpose}
        conversation_id = get_conversation_context() or session_id
        if conversation_id and len(conversation_id) <= 512 and conversation_id.strip():
            metadata["hermes_activity_id"] = conversation_id
        body: dict[str, Any] = {"tags": nous_portal_tags(session_id=session_id), "metadata": metadata}
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
            from hermes_cli.models_reasoning_caps import nous_model_reasoning_capabilities, warm_nous_reasoning_caps_async

            caps = nous_model_reasoning_capabilities(model)
            if caps is None:
                warm_nous_reasoning_caps_async()
                return True
        except Exception:
            return True
        return not caps.get("supports_reasoning") or bool(caps.get("mandatory"))

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
    display_name="Nous Research", description="Nous Research — Hermes model family",
    signup_url="https://nousresearch.com/", fallback_models=("hermes-3-405b", "hermes-3-70b"),
    base_url="https://inference-api.nousresearch.com/v1", auth_type="oauth_device_code",
)

register_provider(nous)
