"""Pi RPC provider profile.

pi-rpc drives the local ``pi`` coding agent over its native JSONL RPC
protocol (``pi --mode rpc``) — no ACP bridge. The native protocol exposes
``extension_ui_request``, so delegated pi agents can ask the parent
questions and receive real free-text answers.

The profile captures auth + endpoint metadata for registry migration;
client construction is handled in run_agent.py / auxiliary_client.py,
which build ``PiRPCClient`` for ``provider == "pi-rpc"``.
"""

from providers import register_provider
from providers.base import ProviderProfile


class PiRPCProfile(ProviderProfile):
    """Pi coding agent — external JSONL RPC process, no REST endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Model listing is owned by the pi subprocess config."""
        return None


pi_rpc = PiRPCProfile(
    name="pi-rpc",
    aliases=("pi", "pi-agent"),
    api_mode="chat_completions",  # JSONL RPC is routed via chat_completions plumbing
    env_vars=(),  # Managed by the pi subprocess
    base_url="pi://rpc",  # internal marker scheme
    auth_type="external_process",
)

register_provider(pi_rpc)
