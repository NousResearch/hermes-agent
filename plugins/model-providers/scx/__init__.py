"""SCX.ai provider profile.

SCX.ai is a sovereign Australian inference provider with a plain
OpenAI-compatible chat-completions API at ``https://api.scx.ai/v1``.
The catalog features SCX's own models (MAGPiE, an Australian-context
LLM, and SCX Coder) alongside hosted open-weights models.
"""

from providers import register_provider
from providers.base import ProviderProfile


class _ScxProfile(ProviderProfile):
    """SCX profile with a curated flagship-only picker.

    SCX's ``/v1/models`` also lists hosted open-weights models and
    non-chat surfaces (embeddings, STT, moderation) that the agent loop
    cannot drive. The picker is pinned to SCX's own agentic flagship
    models via ``fallback_models``; any other hosted model still works
    when typed explicitly (e.g. ``/model scx:DeepSeek-V3.1``).
    """

    def fetch_models(self, *, api_key=None, base_url=None, timeout=8.0):
        # No live catalog merge — the curated list is authoritative.
        return None


scx = _ScxProfile(
    name="scx",
    aliases=("scx-ai",),
    display_name="SCX.ai",
    description="SCX.ai — sovereign Australian inference (MAGPiE, SCX Coder)",
    signup_url="https://scx.ai/",
    env_vars=("SCX_API_KEY", "SCX_BASE_URL"),
    base_url="https://api.scx.ai/v1",
    auth_type="api_key",
    # Cheap/fast model for auxiliary side tasks (compression, session
    # search, web extract). Aux resolution is synchronous, so one explicit
    # choice is needed here rather than a catalog round-trip per turn.
    default_aux_model="coder",
    # Curated picker: SCX's flagship agentic models. All support tool
    # calling and exceed MINIMUM_CONTEXT_LENGTH (coder: 196K, MAGPiE: 131K,
    # MiniMax-M2.7: 192K). Entry [0] is the setup default
    # (get_default_model_for_provider). SCX model IDs are bare (no org/
    # prefix) — use them exactly as /v1/models returns them.
    fallback_models=(
        "coder",
        "MAGPiE",
        "MiniMax-M2.7",
    ),
)

register_provider(scx)
