"""Antigravity CLI (`agy`) provider profile.

`agy` is Google's Antigravity CLI — a stand-alone Go binary at
``~/.local/bin/agy`` that exposes 8 zero-cost models with 1M context:

  * gemini-3.5-flash (low/medium/high)  ← reasoning levels baked into model id
  * gemini-3.1-pro (low/high)            ← including the gemini-3.1-pro-preview
                                            that Copilot won't reliably serve
  * claude-sonnet-4.6 (thinking)
  * claude-opus-4.6 (thinking)
  * gpt-oss-120b                         ← NousResearch's open-weight 120B,
                                            FREE here, 131k context

The CLI's auth is OAuth-based and stored under ``~/.config/agy/`` (or in
the binary's own state); Hermes does NOT manage it. The user is expected to
have run ``agy install`` once and have a valid session.

Like ``copilot-acp``, this provider is a thin registry profile — the actual
subprocess transport (``api_mode="agy_cli"``) is dispatched in run_agent.py
via ``agent/agy_cli_client.py``.

Slug → display-name map (mirrors @gsd/agy-cli stream-adapter):
  ``--model "<display>"`` is what the CLI accepts; the slug is the Hermes-side
  id. Display strings come from ``agy models`` output and are pinned to the
  installed binary version (v1.0.5 as of 2026-06-04).
"""

from providers import register_provider
from providers.base import ProviderProfile


# Hermes slug → agy --model display string.
# Source: ~/.gsd/agent/extensions/agy-cli/models.js (AGY_MODEL_DISPLAY)
# and live ``agy models`` output 2026-06-04.
AGY_SLUG_TO_DISPLAY: dict[str, str] = {
    "default": "",  # omit --model; CLI default (currently Gemini 3.5 Flash)
    "gemini-3.5-flash-low":         "Gemini 3.5 Flash (Low)",
    "gemini-3.5-flash-medium":      "Gemini 3.5 Flash (Medium)",
    "gemini-3.5-flash-high":        "Gemini 3.5 Flash (High)",
    "gemini-3.1-pro-low":           "Gemini 3.1 Pro (Low)",
    "gemini-3.1-pro-high":          "Gemini 3.1 Pro (High)",
    "claude-sonnet-4.6-thinking":   "Claude Sonnet 4.6 (Thinking)",
    "claude-opus-4.6-thinking":     "Claude Opus 4.6 (Thinking)",
    "gpt-oss-120b":                 "GPT-OSS 120B (Medium)",
}


class AgyCliProfile(ProviderProfile):
    """Antigravity CLI — external subprocess, no REST models endpoint."""

    def fetch_models(
        self,
        *,
        api_key: str | None = None,
        timeout: float = 8.0,
    ) -> list[str] | None:
        """Return the pinned slug list. The CLI's own ``agy models`` is the
        canonical source but it's a subprocess; for catalog/UI purposes we
        return the pinned slugs synchronously."""
        return [s for s in AGY_SLUG_TO_DISPLAY.keys() if s != "default"]


agy_cli = AgyCliProfile(
    name="agy-cli",
    aliases=("agy", "antigravity", "antigravity-cli"),
    api_mode="agy_cli",  # routed to agent/agy_cli_client.py in run_agent.py
    env_vars=(),  # auth fully managed by the agy binary
    base_url="agy://antigravity",  # internal scheme; never hit over HTTP
    auth_type="external_process",
)

register_provider(agy_cli)
