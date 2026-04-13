import os
import threading
import socket
import logging
from typing import Any

from openai import OpenAI

logger = logging.getLogger(__name__)

class LLMClientManager:
    """Manages the TCP connections, SDK clients, and credential refreshments for LLM Providers.
    
    This class handles thread-safe lock mechanisms and graceful recovery sequences
    when TCP CLOSE-WAITs accumulate or API credentials expire.
    """
    def __init__(self, agent):
        self._agent = agent
        self.client = None
        self._client_lock = threading.RLock()
    
    def get_lock(self) -> threading.RLock:
        return self._client_lock

    def _client_log_context(self) -> str:
        provider = getattr(self._agent, "provider", "unknown")
        base_url = getattr(self._agent, "base_url", "unknown")
        model = getattr(self._agent, "model", "unknown")
        return (
            f"thread={self._agent._thread_identity()} provider={provider} "
            f"base_url={base_url} model={model}"
        )

    @staticmethod
    def _is_openai_client_closed(client: Any) -> bool:
        """Check if an OpenAI client is closed."""
        from unittest.mock import Mock

        if isinstance(client, Mock):
            return False

        is_closed_attr = getattr(client, "is_closed", None)
        if is_closed_attr is not None:
            if callable(is_closed_attr):
                if is_closed_attr():
                    return True
            elif bool(is_closed_attr):
                return True

        http_client = getattr(client, "_client", None)
        if http_client is not None:
            return bool(getattr(http_client, "is_closed", False))
        return False

    def create_client(self, client_kwargs: dict, *, reason: str, shared: bool) -> Any:
        # Avoid circular imports
        if getattr(self._agent, "provider", "") == "copilot-acp" or str(client_kwargs.get("base_url", "")).startswith("acp://copilot"):
            from agent.copilot_acp_client import CopilotACPClient
            client = CopilotACPClient(**client_kwargs)
            logger.info("Copilot ACP client created (%s, shared=%s) %s", reason, shared, self._client_log_context())
            return client
            
        client = OpenAI(**client_kwargs)
        logger.info("OpenAI client created (%s, shared=%s) %s", reason, shared, self._client_log_context())
        return client

    @staticmethod
    def _force_close_tcp_sockets(client: Any) -> int:
        """Force-close underlying TCP sockets to prevent CLOSE-WAIT accumulation."""
        closed = 0
        try:
            http_client = getattr(client, "_client", None)
            if http_client is None: return 0
            transport = getattr(http_client, "_transport", None)
            if transport is None: return 0
            pool = getattr(transport, "_pool", None)
            if pool is None: return 0
            
            connections = getattr(pool, "_connections", None) or getattr(pool, "_pool", None) or []
            for conn in list(connections):
                stream = getattr(conn, "_network_stream", None) or getattr(conn, "_stream", None)
                if stream is None: continue
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    sock = getattr(stream, "stream", None)
                    if sock is not None:
                        sock = getattr(sock, "_sock", None)
                if sock is None: continue
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError: pass
                try:
                    sock.close()
                except OSError: pass
                closed += 1
        except Exception as exc:
            logger.debug("Force-close TCP sockets sweep error: %s", exc)
        return closed

    def close_client(self, client: Any, *, reason: str, shared: bool) -> None:
        if client is None: return
        force_closed = self._force_close_tcp_sockets(client)
        try:
            client.close()
            logger.info("OpenAI client closed (%s, shared=%s, tcp_force_closed=%d) %s", reason, shared, force_closed, self._client_log_context())
        except Exception as exc:
            logger.debug("OpenAI client close failed (%s, shared=%s) %s error=%s", reason, shared, self._client_log_context(), exc)

    def recreate_client(self, *, reason: str) -> bool:
        with self.get_lock():
            old_client = getattr(self, "client", None)
            try:
                new_client = self.create_client(self._agent._client_kwargs, reason=reason, shared=True)
            except Exception as exc:
                logger.warning("Failed to rebuild shared OpenAI client (%s) %s error=%s", reason, self._client_log_context(), exc)
                return False
            self.client = new_client
            # Note: updating the agent's copy for backward compat during refactor transition
            self._agent.client = new_client
        self.close_client(old_client, reason=f"replace:{reason}", shared=True)
        return True

    def get_client(self, *, reason: str) -> Any:
        with self.get_lock():
            client = getattr(self, "client", None)
            if client is not None and not self._is_openai_client_closed(client):
                return client

        logger.warning("Detected closed shared OpenAI client; recreating before use (%s) %s", reason, self._client_log_context())
        if not self.recreate_client(reason=f"recreate_closed:{reason}"):
            raise RuntimeError("Failed to recreate closed OpenAI client")
        with self.get_lock():
            return self.client

    def cleanup_dead_connections(self) -> bool:
        client = getattr(self, "client", None)
        if client is None: return False
        try:
            http_client = getattr(client, "_client", None)
            if http_client is None: return False
            transport = getattr(http_client, "_transport", None)
            if transport is None: return False
            pool = getattr(transport, "_pool", None)
            if pool is None: return False
            connections = getattr(pool, "_connections", None) or getattr(pool, "_pool", None) or []
            dead_count = 0
            for conn in list(connections):
                stream = getattr(conn, "_network_stream", None) or getattr(conn, "_stream", None)
                if stream is None: continue
                sock = getattr(stream, "_sock", None)
                if sock is None:
                    sock = getattr(stream, "stream", None)
                    if sock is not None:
                        sock = getattr(sock, "_sock", None)
                if sock is None: continue
                try:
                    sock.setblocking(False)
                    data = sock.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT)
                    if data == b"": dead_count += 1
                except BlockingIOError: pass
                except OSError: dead_count += 1
                finally:
                    try: sock.setblocking(True)
                    except OSError: pass
            if dead_count > 0:
                logger.warning("Found %d dead connection(s) in client pool — rebuilding client", dead_count)
                self.recreate_client(reason="dead_connection_cleanup")
                return True
        except Exception as exc:
            logger.debug("Dead connection check error: %s", exc)
        return False

    def create_request_client(self, *, reason: str) -> Any:
        from unittest.mock import Mock
        primary_client = self.get_client(reason=reason)
        if isinstance(primary_client, Mock):
            return primary_client
        with self.get_lock():
            request_kwargs = dict(self._agent._client_kwargs)
        return self.create_client(request_kwargs, reason=reason, shared=False)

    def close_request_client(self, client: Any, *, reason: str) -> None:
        self.close_client(client, reason=reason, shared=False)

    def try_refresh_codex_credentials(self, *, force: bool = True) -> bool:
        if self._agent.api_mode != "codex_responses" or self._agent.provider != "openai-codex":
            return False
        try:
            from hermes_cli.auth import resolve_codex_runtime_credentials
            creds = resolve_codex_runtime_credentials(force_refresh=force)
        except Exception as exc:
            logger.debug("Codex credential refresh failed: %s", exc)
            return False
        
        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip() or not isinstance(base_url, str) or not base_url.strip():
            return False
            
        self._agent.api_key = api_key.strip()
        self._agent.base_url = base_url.strip().rstrip("/")
        self._agent._client_kwargs["api_key"] = self._agent.api_key
        self._agent._client_kwargs["base_url"] = self._agent.base_url

        if not self.recreate_client(reason="codex_credential_refresh"): return False
        return True

    def try_refresh_nous_credentials(self, *, force: bool = True) -> bool:
        if self._agent.api_mode != "chat_completions" or self._agent.provider != "nous":
            return False
        try:
            from hermes_cli.auth import resolve_nous_runtime_credentials
            creds = resolve_nous_runtime_credentials(
                min_key_ttl_seconds=max(60, int(os.environ.get("HERMES_NOUS_MIN_KEY_TTL_SECONDS", "1800"))),
                timeout_seconds=float(os.environ.get("HERMES_NOUS_TIMEOUT_SECONDS", "15")),
                force_mint=force,
            )
        except Exception as exc:
            logger.debug("Nous credential refresh failed: %s", exc)
            return False

        api_key = creds.get("api_key")
        base_url = creds.get("base_url")
        if not isinstance(api_key, str) or not api_key.strip() or not isinstance(base_url, str) or not base_url.strip():
            return False

        self._agent.api_key = api_key.strip()
        self._agent.base_url = base_url.strip().rstrip("/")
        self._agent._client_kwargs["api_key"] = self._agent.api_key
        self._agent._client_kwargs["base_url"] = self._agent.base_url
        self._agent._client_kwargs.pop("default_headers", None)

        if not self.recreate_client(reason="nous_credential_refresh"): return False
        return True

    def try_refresh_anthropic_credentials(self) -> bool:
        if self._agent.api_mode != "anthropic_messages" or not hasattr(self._agent, "_anthropic_api_key"):
            return False
        if self._agent.provider != "anthropic":
            return False
        try:
            from agent.anthropic_adapter import resolve_anthropic_token, build_anthropic_client
            new_token = resolve_anthropic_token()
        except Exception as exc:
            logger.debug("Anthropic credential refresh failed: %s", exc)
            return False

        if not isinstance(new_token, str) or not new_token.strip(): return False
        new_token = new_token.strip()
        if new_token == self._agent._anthropic_api_key: return False

        try:
            self._agent._anthropic_client.close()
        except Exception as exc:
            logger.debug("Anthropic client close during credential refresh failed: %s", exc)

        try:
            self._agent._anthropic_client = build_anthropic_client(new_token, getattr(self._agent, "_anthropic_base_url", None))
        except Exception as exc:
            logger.warning("Failed to rebuild Anthropic client after credential refresh: %s", exc)
            return False

        self._agent._anthropic_api_key = new_token
        from agent.anthropic_adapter import _is_oauth_token
        self._agent._is_anthropic_oauth = _is_oauth_token(new_token)
        return True

    def apply_client_headers_for_base_url(self, base_url: str) -> None:
        from agent.auxiliary_client import _OR_HEADERS
        normalized = (base_url or "").lower()
        if "openrouter" in normalized:
            self._agent._client_kwargs["default_headers"] = dict(_OR_HEADERS)
        elif "api.githubcopilot.com" in normalized:
            from hermes_cli.models import copilot_default_headers
            self._agent._client_kwargs["default_headers"] = copilot_default_headers()
        elif "api.kimi.com" in normalized:
            self._agent._client_kwargs["default_headers"] = {"User-Agent": "KimiCLI/1.3"}
        else:
            self._agent._client_kwargs.pop("default_headers", None)
