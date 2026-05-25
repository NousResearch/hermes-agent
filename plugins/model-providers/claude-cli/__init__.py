"""Claude CLI provider profile.

Routes inference through a local OpenAI-compatible HTTP shim (default
``http://localhost:9180/v1``) that subprocesses Anthropic's official
``claude`` CLI. Each request uses the user's existing Claude Code OAuth
session (``~/.claude/.credentials.json``), so billing flows against the
**Claude Code subscription's base Max allowance** instead of pay-per-token
API credits or separately-purchased extra-usage credits.

This is the same auth path users see when running ``claude`` interactively
in a terminal. It does **not** call ``api.anthropic.com`` directly — the
shim is what makes it look like an OpenAI server to Hermes.

Why this is its own provider (rather than just using ``provider: custom``):

  - Picker visibility: ``hermes model`` lists named providers so users
    discover the Claude-via-subscription path without reading docs.
  - Curated model list: aliases (``sonnet``, ``opus``, ``haiku``) plus
    full IDs that the ``claude`` CLI accepts.
  - Default aux model: routes auxiliary tasks (compression, vision
    routing, title generation) to ``haiku`` so they don't burn the
    main-model allowance.

Setup requires an external shim daemon — Hermes intentionally does **not**
spawn or manage the ``claude`` subprocess itself, both to keep the core
free of Go/Node dependencies and to let the same shim serve non-Hermes
tools (Open WebUI, Cursor, etc.).

Reference shim: https://github.com/niski84/claude-bridge

Override the shim URL with ``CLAUDE_BRIDGE_URL`` if running on a
non-default host/port or behind a reverse proxy.
"""

from __future__ import annotations

import os

from providers import register_provider
from providers.base import ProviderProfile


claude_cli = ProviderProfile(
    name="claude-cli",
    aliases=("claude", "claude-code", "claude-max", "claude-subscription"),
    env_vars=("CLAUDE_BRIDGE_URL",),
    display_name="Claude CLI (Max subscription)",
    description=(
        "Claude via the local claude CLI subscription — uses Max base "
        "allowance instead of API credits. Requires the claude-bridge "
        "shim running on localhost (github.com/niski84/claude-bridge)."
    ),
    signup_url="https://claude.ai/code",
    base_url=os.getenv("CLAUDE_BRIDGE_URL", "http://localhost:9180/v1"),
    fallback_models=(
        "sonnet",
        "opus",
        "haiku",
        "claude-sonnet-4-6",
        "claude-opus-4-7",
        "claude-haiku-4-5",
    ),
    default_aux_model="haiku",
    default_headers={
        "X-Title": "Hermes Agent (via Claude CLI bridge)",
    },
)


register_provider(claude_cli)
