"""Route-derived client-config helpers (run_agent.py shard s4, cluster c3).

Extracted verbatim from run_agent.py (wave 1, shard s4, cluster c3,
3 methods).  Method bodies and the moved module-level helpers
(``_routermint_headers`` / ``_qwen_portal_headers`` + ``_QWEN_CODE_VERSION``)
are character-for-character copies; only this header and the import block
are new.  ``logger`` keeps run_agent's logger name so log records preserve
their origin.  No call site in run_agent.py remains for the moved helpers
(the only uses were inside ``_apply_client_headers_for_base_url``).
"""
from __future__ import annotations

import logging

from utils import base_url_host_matches

logger = logging.getLogger("run_agent")


# =========================================================================
# Qwen Portal headers — mimics QwenCode CLI for portal.qwen.ai compatibility.
# Extracted as a module-level helper so both __init__ and
# _apply_client_headers_for_base_url can share it.
# =========================================================================
_QWEN_CODE_VERSION = "0.14.1"


def _routermint_headers() -> dict:
    """Return the User-Agent RouterMint needs to avoid Cloudflare 1010 blocks."""
    from hermes_cli import __version__ as _HERMES_VERSION

    return {
        "User-Agent": f"HermesAgent/{_HERMES_VERSION}",
    }


def _qwen_portal_headers() -> dict:
    """Return default HTTP headers required by Qwen Portal API."""
    import platform as _plat

    _ua = f"QwenCode/{_QWEN_CODE_VERSION} ({_plat.system().lower()}; {_plat.machine()})"
    return {
        "User-Agent": _ua,
        "X-DashScope-CacheControl": "enable",
        "X-DashScope-UserAgent": _ua,
        "X-DashScope-AuthType": "qwen-oauth",
    }


class RouteClientConfigMixin:
    def _apply_client_headers_for_base_url(
        self,
        base_url: str,
        *,
        apply_user_headers: bool = True,
    ) -> None:
        from agent.auxiliary_client import (
            _AI_GATEWAY_HEADERS,
            build_nvidia_nim_headers,
            build_or_headers,
        )

        if base_url_host_matches(base_url, "openrouter.ai"):
            self._client_kwargs["default_headers"] = build_or_headers()
        elif base_url_host_matches(base_url, "ai-gateway.vercel.sh"):
            self._client_kwargs["default_headers"] = dict(_AI_GATEWAY_HEADERS)
        elif base_url_host_matches(base_url, "integrate.api.nvidia.com"):
            self._client_kwargs["default_headers"] = build_nvidia_nim_headers(base_url)
        elif base_url_host_matches(base_url, "api.routermint.com"):
            self._client_kwargs["default_headers"] = _routermint_headers()
        elif base_url_host_matches(base_url, "githubcopilot.com"):
            from hermes_cli.models import copilot_default_headers

            self._client_kwargs["default_headers"] = copilot_default_headers()
        elif base_url_host_matches(base_url, "api.kimi.com"):
            self._client_kwargs["default_headers"] = {"User-Agent": "claude-code/0.1.0"}
        elif base_url_host_matches(base_url, "portal.qwen.ai"):
            self._client_kwargs["default_headers"] = _qwen_portal_headers()
        elif base_url_host_matches(base_url, "chatgpt.com"):
            from agent.auxiliary_client import _codex_cloudflare_headers
            self._client_kwargs["default_headers"] = _codex_cloudflare_headers(
                self._client_kwargs.get("api_key", "")
            )
        elif base_url_host_matches(base_url, "x.ai"):
            # Cover both provider=xai and provider=xai-oauth (api.x.ai).
            from tools.xai_http import hermes_xai_default_headers

            self._client_kwargs["default_headers"] = hermes_xai_default_headers()
        else:
            # No URL-specific headers — check profile.default_headers before clearing.
            _ph_headers = None
            try:
                from providers import get_provider_profile as _gpf2
                _ph2 = _gpf2(self.provider)
                if _ph2 and _ph2.default_headers:
                    _ph_headers = dict(_ph2.default_headers)
            except Exception:
                pass
            if _ph_headers:
                self._client_kwargs["default_headers"] = _ph_headers
            else:
                self._client_kwargs.pop("default_headers", None)

        # User-configured overrides win over URL/profile defaults for the same
        # route. A credential swap to another endpoint must not inherit them.
        if apply_user_headers:
            self._apply_user_default_headers()

        # Per-provider extra HTTP headers (providers.<name>.extra_headers /
        # custom_providers[].extra_headers) — applied last so the most
        # specific config level survives credential swaps and rebuilds too.
        # SECURITY: values may carry credentials — never log them.
        if self.api_mode not in ("anthropic_messages", "bedrock_converse"):
            try:
                from hermes_cli.config import (
                    apply_custom_provider_extra_headers_to_client_kwargs,
                )

                apply_custom_provider_extra_headers_to_client_kwargs(
                    self._client_kwargs, base_url,
                )
            except Exception:
                logger.debug("custom-provider extra_headers skipped", exc_info=True)

    def _apply_user_default_headers(self) -> None:
        """Merge user-configured request headers onto the OpenAI client.

        Reads ``model.default_headers`` from config.yaml and merges it onto
        ``self._client_kwargs["default_headers"]``, with user values taking
        precedence over provider- and SDK-supplied defaults.

        This exists for ``custom`` OpenAI-compatible endpoints sitting behind
        a gateway/WAF that rejects the OpenAI Python SDK's identifying headers
        (``User-Agent: OpenAI/Python ...``, ``X-Stainless-*``). Setting e.g.
        ``model.default_headers: {User-Agent: curl/8.7.1}`` lets the request
        reach such an upstream instead of failing with an opaque 4xx/502 even
        though the same body works under ``curl``. (#40033)

        Delegates the config read + merge to
        ``agent.auxiliary_client._apply_user_default_headers`` so the main and
        auxiliary clients can never drift on precedence or value handling.

        No-op for Anthropic/Bedrock modes, which don't use the OpenAI client,
        and when no overrides are configured.
        """
        if self.api_mode in ("anthropic_messages", "bedrock_converse"):
            return
        from agent.auxiliary_client import (
            _apply_user_default_headers as _merge_user_headers,
        )
        merged = _merge_user_headers(self._client_kwargs.get("default_headers"))
        if merged:
            self._client_kwargs["default_headers"] = merged

    def _reapply_route_client_config(self, *, route_changed: bool) -> None:
        """Recompute route-derived client kwargs for the current ``self.base_url``.

        TLS material (``ssl_verify``/``ssl_ca_cert``) and default headers are
        derived from the endpoint, not the credential — any client rebuild
        that may have moved ``base_url`` must recompute them or the new
        endpoint inherits configuration computed for the old one. Shared by
        credential-pool rotation and the per-turn env refresh so the two
        paths cannot drift.
        """
        self._client_kwargs.pop("ssl_verify", None)
        self._client_kwargs.pop("ssl_ca_cert", None)
        try:
            from hermes_cli.config import (
                apply_custom_provider_tls_to_client_kwargs,
                get_compatible_custom_providers,
                load_config_readonly,
            )

            apply_custom_provider_tls_to_client_kwargs(
                self._client_kwargs,
                str(self.base_url or ""),
                get_compatible_custom_providers(load_config_readonly()),
            )
        except Exception:
            logger.debug(
                "custom-provider TLS resolution skipped on credential rotation",
                exc_info=True,
            )
        self._apply_client_headers_for_base_url(
            self.base_url,
            apply_user_headers=not route_changed,
        )
