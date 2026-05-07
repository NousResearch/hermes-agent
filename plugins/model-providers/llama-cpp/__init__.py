"""llama.cpp provider profile.

First-class support for llama.cpp's `llama-server` — the OpenAI-compatible
HTTP endpoint shipped with the upstream binary. Hermes can already reach
any OpenAI-compatible URL via `--provider custom`; this profile exists so
that local llama.cpp users get:

  - a zero-config default (http://127.0.0.1:8088/v1) that lines up with
    `scripts/start-llama-server.sh`
  - a `hermes model` menu entry without setting `OPENAI_BASE_URL` by hand
  - graceful behaviour when the local server is offline (fetch_models
    returns None instead of raising)

Auth: llama-server runs without auth by default. If users boot it with
`--api-key`, set `LLAMA_CPP_API_KEY` to the same value. Base URL is
overridable via `LLAMA_CPP_BASE_URL` for non-default ports / remote hosts.
"""

from __future__ import annotations

import logging

from providers import register_provider
from providers.base import ProviderProfile

logger = logging.getLogger(__name__)


class LlamaCppProfile(ProviderProfile):
    """llama.cpp llama-server profile — local-first, model-agnostic."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 4.0,
    ) -> list[str] | None:
        """Hit `${base_url}/models` quietly — local server may not be up."""
        if not self.base_url:
            return None
        try:
            return super().fetch_models(api_key=api_key, timeout=timeout)
        except Exception as exc:
            logger.debug("llama-cpp fetch_models failed (server offline?): %s", exc)
            return None


llama_cpp = LlamaCppProfile(
    name="llama-cpp",
    aliases=(
        "llamacpp",
        "llama.cpp",
        "llama_cpp",
        "llama-server",
    ),
    display_name="llama.cpp",
    description="llama.cpp — local OpenAI-compatible inference (llama-server)",
    signup_url="https://github.com/ggml-org/llama.cpp/releases",
    env_vars=("LLAMA_CPP_API_KEY", "LLAMA_CPP_BASE_URL"),
    base_url="http://127.0.0.1:8088/v1",
    auth_type="api_key",
)

register_provider(llama_cpp)
