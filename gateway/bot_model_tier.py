"""Classifies a bot-to-bot sender/agent as ``claude`` (expensive) or ``local`` (cheap), for
``gateway/bot_loop_guard.py`` and ``agent/bot_chain.py``'s tier-aware limits
(``bot_loop_guard_split_design.md`` §4). Both modules import this one module so they agree.

Classification order, cheapest/most explicit first:
1. ``gateway.bot_loop_guard.claude_profiles`` / ``.local_profiles`` (config.yaml, exact profile name).
2. The profile's own ``config.yaml`` ``model``/``provider`` text against
   ``gateway.bot_loop_guard.claude_model_patterns`` (default ``["claude", "anthropic"]``),
   case-insensitive substring.

Fail-closed: an unknown profile, an unreadable config, a model string that matches no pattern (and no
explicit list membership), or an empty/unclassifiable input is ``"claude"`` -- token protection over
convenience, per ``bot_loop_guard_split_design.md`` §2/§6. A set of several profiles (a mixed chain) is
``"claude"`` if any one of them is -- the same sticky-strict rule, applied across one classification
call instead of across a chain's lifetime (that part is ``_Chain.ever_claude`` / ``_CHAINS[...]["ever_claude"]``).
"""

from __future__ import annotations

__all__ = ["classify_tier", "classify_model_text"]

_DEFAULT_CLAUDE_PATTERNS = ("claude", "anthropic")


def _bot_loop_guard_block(cfg) -> dict:
    from hermes_cli.config import cfg_get

    block = cfg_get(cfg, "gateway", "bot_loop_guard", default=None)
    return block if isinstance(block, dict) else {}


def _load_cfg():
    from hermes_cli.config import load_config_readonly

    return load_config_readonly()


def _claude_patterns(block: dict) -> list:
    raw = block.get("claude_model_patterns")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple, set)) or not raw:
        raw = _DEFAULT_CLAUDE_PATTERNS
    return [str(p).lower() for p in raw if str(p)]


def _profile_model_text(profile_name: str) -> str:
    """``"<model> <provider>"`` from ``profile_name``'s own config.yaml (read-only, mtime-cached by
    ``hermes_cli.profiles``). Empty string when the profile or its config cannot be read."""
    try:
        from hermes_cli.profiles import _read_config_model, get_profile_dir

        model, provider = _read_config_model(get_profile_dir(profile_name))
    except Exception:
        return ""
    return " ".join(str(v) for v in (model, provider) if v)


def classify_model_text(model_text, cfg=None) -> str:
    """``"claude"``/``"local"`` from a model/provider string alone (no profile config.yaml lookup) --
    for a caller that already holds its own model name, like ``agent/bot_chain.py``'s running agent.
    Same patterns as ``classify_tier``; empty/unmatched text is ``"claude"`` (fail-closed)."""
    try:
        block = _bot_loop_guard_block(cfg if cfg is not None else _load_cfg())
    except Exception:
        block = {}
    text = str(model_text or "").strip().lower()
    if not text:
        return "claude"
    patterns = _claude_patterns(block)
    return "claude" if any(p in text for p in patterns) else "local"


def classify_tier(profile_names, cfg=None) -> str:
    """``"claude"``/``"local"`` for the profile name(s) behind one bot message or chain.

    ``profile_names`` is a profile name or an iterable of them (several only for a mixed chain already
    known to span profiles). ``cfg`` is an optional already-loaded config (same shape
    ``settings_from_config`` takes); omitted, the live config.yaml is read.
    """
    if isinstance(profile_names, str):
        names = {profile_names} if profile_names else set()
    else:
        names = {str(p) for p in (profile_names or ()) if p}
    if not names:
        return "claude"
    try:
        block = _bot_loop_guard_block(cfg if cfg is not None else _load_cfg())
    except Exception:
        return "claude"
    claude_profiles = {str(v) for v in (block.get("claude_profiles") or ())}
    local_profiles = {str(v) for v in (block.get("local_profiles") or ())}
    patterns = _claude_patterns(block)
    for name in names:
        if name in claude_profiles:
            return "claude"
        if name in local_profiles:
            continue
        text = _profile_model_text(name).lower()
        if not text or any(p in text for p in patterns):
            return "claude"
    return "local"
