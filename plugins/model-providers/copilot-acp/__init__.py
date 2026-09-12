"""GitHub Copilot ACP provider profile.

copilot-acp does not speak OpenAI-over-HTTP: it drives an external ACP subprocess over
stdio, so the profile supplies its own client via :meth:`ProviderProfile.create_client`.
An out-of-tree ACP provider (``~/.hermes/plugins/model-providers/`` or a pip entry point)
uses the same three lines without touching core.
"""

from typing import Any

from providers import register_provider
from providers.base import ProviderProfile


class CopilotACPProfile(ProviderProfile):
    """GitHub Copilot ACP — external process, no REST models endpoint."""

    def create_client(self, **client_kwargs: Any) -> Any:
        """Build the ACP stdio shim rather than an HTTP client."""
        from agent.copilot_acp_client import CopilotACPClient

        return CopilotACPClient(**client_kwargs)

    def fetch_models(
        self, *, api_key: str | None = None, base_url: str | None = None, timeout: float = 15.0
    ) -> list[str] | None:
        """Enabled models advertised by a short-lived signed-in ACP session (``session/new``).

        The CLI may keep its login in an OS credential store with no token Hermes can reuse, so
        the session is the only source that reflects the account's enablement. ``api_key`` /
        ``base_url`` are ignored: the subprocess owns auth. None when the CLI is missing, refuses
        ``--acp``, or the probe fails/times out — callers fall back to their next source.
        """
        from hermes_cli.auth import resolve_external_process_provider_credentials

        creds = resolve_external_process_provider_credentials(self.name)
        if not str(creds.get("base_url") or "").startswith("acp://"):
            return None
        client = self.create_client(
            api_key=creds.get("api_key"), base_url=creds.get("base_url"),
            command=creds.get("command"), args=creds.get("args"))
        return client.list_models(timeout_seconds=timeout) or None


copilot_acp = CopilotACPProfile(
    name="copilot-acp", aliases=("github-copilot-acp", "copilot-acp-agent"),
    api_mode="chat_completions",  # ACP subprocess uses chat_completions routing
    env_vars=(),  # Managed by ACP subprocess
    base_url="acp://copilot",  # ACP internal scheme
    auth_type="external_process",
    # How to launch the CLI; env var names predate this profile (formerly hardcoded in
    # hermes_cli/auth.py), so existing setups keep working.
    process_command="copilot",
    process_args=("--acp", "--stdio"),
    process_command_env_vars=("HERMES_COPILOT_ACP_COMMAND", "COPILOT_CLI_PATH"),
    process_args_env_var="HERMES_COPILOT_ACP_ARGS",
)

register_provider(copilot_acp)
