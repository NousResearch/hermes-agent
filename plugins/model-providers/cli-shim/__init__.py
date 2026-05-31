"""cli-shim provider profile.

cli-shim shells out to locally-installed CLIs that the user already
authenticated with OAuth (claude, codex, gemini). NO API keys required —
the CLIs use their own credentials cache.

The `model` field on each request selects which CLI to invoke:
  - claude-sonnet-cli   -> `claude --print --model sonnet`
  - claude-opus-cli     -> `claude --print --model opus`
  - codex-gpt5-cli      -> `codex exec --model gpt-5`
  - gemini-cli          -> `gemini --acp ...` (ACP path, tool-use capable)

api_mode="chat_completions" routes through standard run_agent.py paths;
the cli-shim base_url `cli://shim` triggers the CliShimClient branch.
"""

from providers import register_provider
from providers.base import ProviderProfile


class CliShimProfile(ProviderProfile):
    """Local CLI shim — external process, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        return [
            "claude-sonnet-cli",
            "claude-opus-cli",
            "codex-gpt5-cli",
            "gemini-cli",
        ]


cli_shim = CliShimProfile(
    name="cli-shim",
    aliases=("local-cli-shim", "cli"),
    api_mode="chat_completions",
    env_vars=(),  # OAuth-only — managed by the CLIs themselves
    base_url="cli://shim",
    auth_type="external_process",
)

register_provider(cli_shim)
