"""Per-provider OAuth credential refresh on the live client, for AIAgent.

Extracted from ``run_agent.py`` as part of the god-file decomposition
campaign, following the same mechanical mixin lift that produced
``gateway/authz_mixin.py`` and the ``SessionDB`` mixins in
``hermes_state_*.py``, and named for the same reason those are.

The cluster is five sibling methods with one shape: when a request is about to
go out (or has just failed on auth), re-resolve that provider's credentials
and swap them into the already-built client rather than rebuilding the world.
Codex, Nous, Vertex, Copilot and Anthropic each need a different dance to get
there, which is why it is 242 lines rather than one method with a switch.

Mixin contract: a plain mixin consumed by ``AIAgent``. It defines no
``__init__`` and no state of its own; the host's attributes (``self.client``,
``self.provider``, ``self.base_url``, ``self._is_anthropic_oauth``) and its
other methods (``self._rebuild_anthropic_client``,
``self._replace_primary_openai_client``, ``self._apply_client_headers_for_base_url``)
resolve through the MRO. It never imports ``run_agent``, so there is no cycle.

Every provider dependency this code needs was already imported lazily inside
the method that uses it (``from hermes_cli.auth import ...``, ``from
agent.vertex_adapter import ...``, and so on). Those imports are untouched and
resolve from the modules they name, so they are unaffected by the move; only
three module-level names had to follow, and they come from their original
sources.

Behavior-neutral: every method is lifted verbatim from ``AIAgent``.
"""

import logging

from hermes_cli.timeouts import get_provider_request_timeout
from utils import env_float

# Bind the run_agent logger by name so records lifted with these methods are
# emitted under exactly the name they were before. getLogger returns the same
# singleton object run_agent holds.
logger = logging.getLogger("run_agent")


class AgentCredentialRefreshMixin:
    """See module docstring - credential-refresh cluster lifted verbatim from AIAgent."""

    def _try_refresh_codex_client_credentials(self, *, force: bool = True) -> bool:
        if self.api_mode != "codex_responses" or self.provider not in {"openai-codex", "xai-oauth"}:
            return False

        # Guard against silent account swap.
        #
        # When an agent is using a non-singleton credential — e.g. a manual
        # pool entry (``hermes auth add xai-oauth``) whose tokens belong to
        # a different account than the device_code singleton, or an agent
        # constructed with an explicit ``api_key=`` arg — force-refreshing
        # the singleton here and adopting its tokens silently re-routes the
        # rest of the conversation onto the singleton's account.  The
        # credential pool's reactive recovery (``_recover_with_credential_pool``)
        # is the right channel for that case; this path is the
        # singleton-only fallback used when the pool can't recover, and
        # MUST only fire when the agent really is on singleton tokens.
        try:
            if self.provider == "openai-codex":
                from hermes_cli.auth import resolve_codex_runtime_credentials

                singleton_now = resolve_codex_runtime_credentials(
                    refresh_if_expiring=False,
                )
            else:
                from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

                singleton_now = resolve_xai_oauth_runtime_credentials(
                    refresh_if_expiring=False,
                )
        except Exception as exc:
            logger.debug("%s singleton read failed: %s", self.provider, exc)
            return False

        singleton_key = str(singleton_now.get("api_key") or "").strip()
        active_key = str(self.api_key or "").strip()
        if singleton_key and active_key and singleton_key != active_key:
            logger.debug(
                "%s singleton tokens differ from the active api_key; "
                "skipping singleton force-refresh to avoid silent account swap. "
                "Reactive credential rotation should go through the pool.",
                self.provider,
            )
            return False

        try:
            if self.provider == "openai-codex":
                from hermes_cli.auth import resolve_codex_runtime_credentials

                creds = resolve_codex_runtime_credentials(force_refresh=force)
            else:
                from hermes_cli.auth import resolve_xai_oauth_runtime_credentials

                creds = resolve_xai_oauth_runtime_credentials(force_refresh=force)
        except Exception as exc:
            logger.debug("%s credential refresh failed: %s", self.provider, exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url

        if not self._replace_primary_openai_client(reason=f"{self.provider}_credential_refresh"):
            return False

        return True

    def _try_refresh_nous_client_credentials(
        self,
        *,
        force: bool = True,
    ) -> bool:
        if self.provider != "nous":
            return False
        # Portal serves anthropic/* on the native Messages route, so a session
        # can be holding either client kind when its short-lived invoke JWT
        # expires. Both need the refresh or the turn dies on a 401.
        if self.api_mode not in ("chat_completions", "anthropic_messages"):
            return False

        try:
            from hermes_cli.auth import resolve_nous_runtime_credentials

            creds = resolve_nous_runtime_credentials(
                timeout_seconds=env_float("HERMES_NOUS_TIMEOUT_SECONDS", 15),
                force_refresh=force,
            )
        except Exception as exc:
            logger.debug("Nous credential refresh failed: %s", exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip("/")

        if self.api_mode == "anthropic_messages":
            self._anthropic_api_key = self.api_key
            self._anthropic_base_url = self.base_url
            self._rebuild_anthropic_client()
            return True

        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        # Nous requests should not inherit OpenRouter-only attribution headers.
        self._client_kwargs.pop("default_headers", None)

        if not self._replace_primary_openai_client(reason="nous_credential_refresh"):
            return False

        return True

    def _try_refresh_vertex_client_credentials(self) -> bool:
        """Re-mint the Vertex OAuth2 access token and rebuild the OpenAI client.

        Vertex tokens live ~1 hour. On a long-lived agent (gateway session) a
        cached client's bearer token will expire mid-session, producing a 401.
        This re-resolves credentials via the adapter (which refreshes the
        underlying google-auth Credentials object when near expiry), swaps the
        new token into the client kwargs, and rebuilds the primary OpenAI
        client. Returns True when a usable token+base_url were obtained.
        """
        if self.api_mode != "chat_completions" or self.provider != "vertex":
            return False

        try:
            from agent.vertex_adapter import get_vertex_config

            token, base_url = get_vertex_config()
        except Exception as exc:
            logger.debug("Vertex credential refresh failed: %s", exc)
            return False

        if not isinstance(token, str) or not token.strip():
            return False
        if not isinstance(base_url, str) or not base_url.strip():
            return False

        self.api_key = token.strip()
        self.base_url = base_url.strip().rstrip("/")
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url

        if not self._replace_primary_openai_client(reason="vertex_credential_refresh"):
            return False

        logger.info("Vertex AI OAuth token refreshed")
        return True

    def _try_refresh_copilot_client_credentials(self) -> bool:
        """Refresh Copilot credentials and rebuild the shared OpenAI client.

        Copilot tokens may remain the same string across refreshes (`gh auth token`
        returns a stable OAuth token in many setups). We still rebuild the client
        on 401 so retries recover from stale auth/client state without requiring
        a session restart.
        """
        if self.provider != "copilot":
            return False

        try:
            from hermes_cli.copilot_auth import resolve_copilot_token

            new_token, token_source = resolve_copilot_token()
        except Exception as exc:
            logger.debug("Copilot credential refresh failed: %s", exc)
            return False

        if not isinstance(new_token, str) or not new_token.strip():
            return False

        new_token = new_token.strip()

        self.api_key = new_token
        self._client_kwargs["api_key"] = self.api_key
        self._client_kwargs["base_url"] = self.base_url
        self._apply_client_headers_for_base_url(str(self.base_url or ""))

        if not self._replace_primary_openai_client(reason="copilot_credential_refresh"):
            return False

        logger.info("Copilot credentials refreshed from %s", token_source)
        return True

    def _try_refresh_anthropic_client_credentials(self) -> bool:
        if self.api_mode != "anthropic_messages" or not hasattr(self, "_anthropic_api_key"):
            return False
        # Only refresh credentials for the native Anthropic provider.
        # Other anthropic_messages providers (MiniMax, Alibaba, etc.) use their own keys.
        if self.provider != "anthropic":
            return False
        # Azure endpoints use static API keys — OAuth token rotation doesn't apply.
        # Refreshing would pick up ~/.claude/.credentials.json OAuth token and break auth.
        _base = getattr(self, "_anthropic_base_url", "") or ""
        if "azure.com" in _base:
            return False

        try:
            from agent.anthropic_adapter import resolve_anthropic_token, build_anthropic_client

            new_token = resolve_anthropic_token()
        except Exception as exc:
            logger.debug("Anthropic credential refresh failed: %s", exc)
            return False

        if not isinstance(new_token, str) or not new_token.strip():
            return False
        new_token = new_token.strip()
        if new_token == self._anthropic_api_key:
            return False

        try:
            self._anthropic_client.close()
        except Exception:
            pass

        try:
            self._anthropic_client = build_anthropic_client(
                new_token,
                getattr(self, "_anthropic_base_url", None),
                timeout=get_provider_request_timeout(self.provider, self.model),
            )
        except Exception as exc:
            logger.warning("Failed to rebuild Anthropic client after credential refresh: %s", exc)
            return False

        self._anthropic_api_key = new_token
        # Update OAuth flag — token type may have changed (API key ↔ OAuth).
        # Only treat as OAuth on native Anthropic; third-party endpoints using
        # the Anthropic protocol must not trip OAuth paths (#1739 & third-party
        # identity-injection guard).
        from agent.anthropic_adapter import _is_oauth_token
        self._is_anthropic_oauth = _is_oauth_token(new_token) if self.provider == "anthropic" else False
        return True
