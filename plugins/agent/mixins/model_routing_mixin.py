"""Model-routing URL / provider classification helpers for AIAgent.

Extracted verbatim from ``run_agent.py`` (godfile shard plan s1, cluster
c23 — 7 methods, 28/28 move agreement). The methods read ``self.*`` state
maintained by ``AIAgent.__init__`` / the ``base_url`` property
(``_base_url_lower``, ``_base_url_hostname``, ``provider``, ``model``,
``api_mode``); those class attributes stay on ``AIAgent`` and resolve
through the MRO.

``base_url_hostname`` / ``base_url_host_matches`` are shared module
helpers also used by methods that remain in ``run_agent.py``, so they are
imported here from ``utils`` exactly as ``run_agent.py`` imports them.
"""

from utils import base_url_host_matches, base_url_hostname


class ModelRoutingMixin:
    def _is_direct_openai_url(self, base_url: str = None) -> bool:
        """Return True when a base URL targets OpenAI's native API."""
        if base_url is not None:
            hostname = base_url_hostname(base_url)
        else:
            hostname = getattr(self, "_base_url_hostname", "") or base_url_hostname(
                getattr(self, "_base_url_lower", "")
            )
        return hostname == "api.openai.com"

    def _is_azure_openai_url(self, base_url: str = None) -> bool:
        """Return True when a base URL targets Azure OpenAI.

        Azure OpenAI exposes an OpenAI-compatible endpoint at
        ``{resource}.openai.azure.com/openai/v1`` that accepts the
        standard ``openai`` Python client.  Unlike api.openai.com it
        does NOT support the Responses API — gpt-5.x models are served
        on the regular ``/chat/completions`` path — so routing decisions
        must treat Azure separately from direct OpenAI.
        """
        if base_url is not None:
            url = str(base_url).lower()
        else:
            url = getattr(self, "_base_url_lower", "") or ""
        return "openai.azure.com" in url

    def _is_github_copilot_url(self, base_url: str = None) -> bool:
        """Return True when a base URL targets GitHub Copilot's OpenAI-compatible API."""
        if base_url is not None:
            hostname = base_url_hostname(base_url)
        else:
            hostname = getattr(self, "_base_url_hostname", "") or base_url_hostname(
                getattr(self, "_base_url_lower", "")
            )
        if not hostname:
            return False
        return hostname == "api.githubcopilot.com" or hostname.endswith(".githubcopilot.com")

    def _is_openrouter_url(self) -> bool:
        """Return True when the base URL targets OpenRouter."""
        return base_url_host_matches(self._base_url_lower, "openrouter.ai")

    def _is_copilot_url(self) -> bool:
        """Return True when the base URL targets GitHub Copilot or GitHub Models."""
        return (
            "api.githubcopilot.com" in self._base_url_lower
            or "models.github.ai" in self._base_url_lower
        )

    def _is_copilot_provider(self) -> bool:
        """True when the active provider is GitHub Copilot, however spelled.

        ``self.provider`` is not always the normalized slug: ``/model`` and
        profile configs can leave the alias ``github-copilot`` (or ``github``)
        in place — a single session log can show both ``provider=copilot`` and
        ``provider=github-copilot`` for the same account. A bare
        ``provider == "copilot"`` gate silently skips credential recovery for
        the alias spellings, so this is the single owner of the check; the
        Copilot base URL is accepted as a fallback signal.
        """
        if (self.provider or "").strip().lower() in {"copilot", "github-copilot", "github"}:
            return True
        return self._is_copilot_url()

    def _is_codex_backend(self) -> bool:
        """Return True for the ChatGPT OAuth Codex Responses backend."""
        return (
            getattr(self, "api_mode", None) == "codex_responses"
            and getattr(self, "_base_url_hostname", "") == "chatgpt.com"
            and "/backend-api/codex"
            in (getattr(self, "_base_url_lower", "") or "")
        )
