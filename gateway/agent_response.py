"""
Format gateway responses so the user can see which agent answered.

In a multi-agent gateway, the same chat may receive replies from
several profiles — one might be the ``coder`` agent, the next a
one-shot ``@data-sci`` turn.  Without a label, the user can't tell
them apart.  This module owns the labelling.

Design:

* The prefix is computed once, in one place, and injected before the
  final response is handed to the adapter ``send()`` chain.  We do not
  touch every adapter or every streaming delta — that would duplicate
  prefixes in multi-chunk replies.
* The prefix is configurable via ``gateway.show_agent_name`` in
  ``config.yaml`` (default: ``true``).
* Format is intentionally plain-text (``[name] content``) so it
  renders consistently across every platform.  Per-platform rich
  formatting (bold for Telegram MarkdownV2, etc.) can layer on top
  later without breaking this contract.
* Empty or whitespace-only responses are passed through unchanged so
  we don't surface a bare prefix when the agent had nothing to say.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional

from utils import is_truthy_value


_DEFAULT_SHOW_AGENT_NAME = True


def show_agent_name_enabled(user_config: Optional[Mapping[str, Any]]) -> bool:
    """Return whether the agent-name prefix is enabled in config.

    Defaults to True when the key is absent, matching the user's
    request that the prefix is shown "by default".
    """
    if not isinstance(user_config, Mapping):
        return _DEFAULT_SHOW_AGENT_NAME
    gateway_cfg = user_config.get("gateway")
    if not isinstance(gateway_cfg, Mapping):
        return _DEFAULT_SHOW_AGENT_NAME
    raw = gateway_cfg.get("show_agent_name", _DEFAULT_SHOW_AGENT_NAME)
    return is_truthy_value(raw, default=_DEFAULT_SHOW_AGENT_NAME)


def format_agent_response(
    content: str,
    agent_name: Optional[str],
    *,
    enabled: bool = True,
) -> str:
    """Prepend ``[<agent_name>] `` to ``content`` when conditions hold.

    * Returns ``content`` unchanged when ``enabled`` is False or
      ``agent_name`` is empty / None.
    * Returns ``content`` unchanged when it is empty / whitespace-only,
      so we never produce a "bare prefix" reply.
    * Idempotent: if the content already starts with the same agent
      prefix (e.g. the agent itself echoed it, or it was pre-formatted
      upstream), the prefix is NOT doubled.

    The first newline is preserved if present so multi-paragraph
    responses keep their original layout.
    """
    if not enabled:
        return content
    if not agent_name or not isinstance(agent_name, str):
        return content
    name = agent_name.strip()
    if not name:
        return content
    if not isinstance(content, str) or not content.strip():
        return content
    prefix = f"[{name}] "
    if content.startswith(prefix):
        return content
    return prefix + content


def apply_agent_prefix_to_result(
    result: Optional[dict],
    agent_name: Optional[str],
    user_config: Optional[Mapping[str, Any]],
) -> Optional[dict]:
    """Mutate ``result['final_response']`` in place to carry the agent prefix.

    Returns the same dict so callers can chain.  Safe with ``None``
    (returns None) and with dicts that don't have ``final_response``.
    """
    if not isinstance(result, dict):
        return result
    enabled = show_agent_name_enabled(user_config)
    if not enabled:
        return result
    raw = result.get("final_response")
    if not isinstance(raw, str):
        return result
    result["final_response"] = format_agent_response(
        raw, agent_name, enabled=enabled
    )
    return result
