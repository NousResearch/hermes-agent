"""Cache-rebuild notice for mid-session model switches.

Switching models mid-conversation invalidates the prompt cache the new
model would otherwise hit: providers key prompt caches per model, so the
first call after a switch re-reads the entire context fully uncached —
every tool result, file read, and web page in the session, billed at full
input price once. On a short session that costs pennies; on a long one it
is real money, and users have no way to see it coming.

This module builds a single informational line appended to the ``/model``
switch confirmation on every wired surface. It is NOT a confirmation gate
— the switch already happened; the notice only makes the one-time cost
visible and offers the session-scoped revert command.

Design constraints honored here (see AGENTS.md):

- Informational only. Never blocks, never prompts, never mutates context —
  the note is display output, so prompt caching and message alternation are
  untouched.
- Fires only when it matters: below ``MIN_CONTEXT_TOKENS`` the re-read cost
  is negligible and the line would be noise, so nothing is emitted. Empty
  and near-empty sessions stay silent.
- Gated by ``display.cache_switch_notice`` in config.yaml (never env vars).

CRITICAL CALL ORDER (see PR #94753 review): ``agent.switch_model()`` calls
``context_compressor.update_model()``, which zeroes ``last_prompt_tokens``
and clears ``_cached_system_prompt`` by design. Every caller MUST snapshot
the estimate (and any fields it wants) BEFORE calling ``switch_model()`` —
afterwards the agent looks like an empty session and the notice goes
silent exactly when it matters most.
"""

from __future__ import annotations

from typing import Any, List, Optional

# Below this estimated context size, the uncached re-read after a switch is
# cheap enough that warning about it is pure noise. ~30k tokens ≈ several
# substantial exchanges (or one long document read).
MIN_CONTEXT_TOKENS = 30_000


def _parse_bool(value: Any) -> bool:
    """Config-truthy parse: real booleans plus common string spellings.

    ``"false"``, ``"no"``, ``"0"``, ``"off"`` are False — a bare ``bool()``
    would treat every non-empty string (including "false") as True.
    """
    if isinstance(value, str):
        return value.strip().lower() not in {"false", "no", "0", "off", ""}
    return bool(value)


def cache_switch_notice_enabled() -> bool:
    """Read ``display.cache_switch_notice`` (default: enabled).

    Config failures degrade to "enabled" — a broken config.yaml should not
    silently suppress a cost signal.
    """
    try:
        from hermes_cli.config import load_config_readonly

        display = (load_config_readonly() or {}).get("display") or {}
        value = display.get("cache_switch_notice")
        if value is None:
            return True
        return _parse_bool(value)
    except Exception:
        return True


def _compressor_reported_tokens(agent: Any) -> int:
    """Provider-reported prompt tokens from the agent's context engine.

    Returns 0 when unknown. ``last_prompt_tokens`` parks at a -1 sentinel
    right after a compression (awaiting real usage) — clamp it to 0, the
    same treatment the status bar applies (cli.py snapshot path).
    """
    compressor = getattr(agent, "context_compressor", None)
    if compressor is None:
        return 0
    try:
        tokens = int(getattr(compressor, "last_prompt_tokens", 0) or 0)
    except (TypeError, ValueError):
        return 0
    return tokens if tokens > 0 else 0


def _agent_messages_and_prompt(agent: Any) -> tuple[List[dict], str]:
    """Read conversation history + system prompt off either agent shape.

    ``AIAgent`` keeps them in ``_session_messages`` / ``_cached_system_prompt``;
    the classic CLI object exposes ``conversation_history`` / ``system_prompt``.
    Probe the AIAgent names FIRST — ``switch_model()`` clears
    ``_cached_system_prompt``, which is exactly why callers must snapshot
    before switching (module docstring).
    """
    messages: List[dict] = []
    for source_name in ("_session_messages", "conversation_history"):
        raw = getattr(agent, source_name, None)
        if raw:
            messages = [m for m in raw if isinstance(m, dict)]
            break
    system_prompt = ""
    for prompt_name in ("_cached_system_prompt", "system_prompt"):
        raw = getattr(agent, prompt_name, None)
        if raw:
            system_prompt = str(raw)
            break
    return messages, system_prompt


def _agent_tools(agent: Any) -> Optional[List[dict]]:
    """Best-effort tool-schema list for the rough estimator.

    Prefers the live resolved schemas; ``tools`` may be names/strings on
    some shapes, in which case the estimator just counts less accurately.
    """
    for name in ("_tool_definitions", "tool_definitions", "tools"):
        raw = getattr(agent, name, None)
        if isinstance(raw, list) and raw:
            return raw
    return None


def _rough_estimate_tokens(agent: Any) -> int:
    """Rough token estimate over the live conversation as a fallback.

    Used when no provider-reported count exists yet (fresh session, or the
    first turn before any usage came back). Mirrors the payload buckets
    Hermes actually sends: system prompt, messages, tool schemas.
    """
    try:
        from agent.model_metadata import estimate_request_tokens_rough
    except Exception:
        return 0

    messages, system_prompt = _agent_messages_and_prompt(agent)
    tools = _agent_tools(agent)
    try:
        return int(
            estimate_request_tokens_rough(messages, system_prompt=system_prompt, tools=tools)
        )
    except Exception:
        return 0


def estimate_context_tokens(agent: Any) -> int:
    """Best-effort context size for the cache-notice decision.

    Prefers the provider-reported prompt token count from the last real API
    call; falls back to a rough structural estimate over whatever message /
    prompt / tool attributes this agent shape carries. Returns 0 when
    neither is available — callers treat 0 as "below threshold".

    MUST be called BEFORE ``agent.switch_model()`` — see module docstring.

    Approximation note (PR #94753 review): the provider-reported count comes
    from the OLD model's last API call. The NEW model may tokenize the same
    conversation differently, so this is an upper-bound estimate of what it
    will re-read, not an exact figure — hence the "up to ~Nk" wording in
    the notice. This is inherent: the new model has no call history yet, so
    a pre-switch estimate under the old tokenizer is the best signal
    available. The rough fallback (system + messages + tools) is
    model-agnostic and slightly more conservative.
    """
    if agent is None:
        return 0
    reported = _compressor_reported_tokens(agent)
    if reported > 0:
        return reported
    return _rough_estimate_tokens(agent)


def same_model_reselect(
    *,
    old_model: str,
    old_provider: str = "",
    new_model: str,
    new_provider: str = "",
) -> bool:
    """True when the switch is a no-op for caching purposes.

    Prompt caches are keyed per (provider, model). Same display string with
    a different provider IS a real cache miss (e.g. grok-4.6 xAI-OAuth →
    grok-4.6 OpenRouter), so identity requires BOTH to match. Unknown
    providers compare on the model alone.
    """
    om = (old_model or "").strip()
    nm = (new_model or "").strip()
    op = (old_provider or "").strip().lower()
    np_ = (new_provider or "").strip().lower()
    if not om or not nm:
        return False
    if om != nm:
        return False
    # Model IDs match: it's a re-select unless the providers differ AND we
    # know both of them.
    if op and np_ and op != np_:
        return False
    return True


def build_cache_switch_notice(
    *,
    old_model_display: str,
    new_model_display: str,
    est_context_tokens: int,
    is_reselect: bool = False,
    include_revert_hint: bool = True,
) -> Optional[str]:
    """Build the user-facing notice, or None when it should stay silent.

    Silent when:
    - ``is_reselect`` — same (provider, model) re-select keeps the cache
      warm, there is nothing to warn about,
    - the estimated context is below :data:`MIN_CONTEXT_TOKENS`,
    - the config toggle is off (checked by :func:`cache_switch_notice_enabled`).

    ``include_revert_hint=False`` for ``--once`` switches: those auto-restore
    after one turn, so telling the user to type ``/model <old>`` would be
    wrong.
    """
    if not old_model_display or not new_model_display:
        return None
    if is_reselect:
        return None
    if est_context_tokens < MIN_CONTEXT_TOKENS:
        return None

    from agent.i18n import t

    # Round to nearest thousand with half-up (not banker's), so 30_500 → 31k.
    k = max(1, int((est_context_tokens + 500) // 1000))
    lines = [
        t(
            "gateway.model.cache_switch_notice",
            model=new_model_display,
            tokens=f"{k}k",
        ),
    ]
    if include_revert_hint:
        lines.append(t("gateway.model.cache_switch_revert_hint", model=old_model_display))
    return "\n".join(lines)


def snapshot_pre_switch_state(agent: Any) -> int:
    """Capture the context-token estimate BEFORE ``switch_model()`` runs.

    The single integration point every surface must call pre-switch;
    ``switch_model()`` zeroes the compressor counters and clears the cached
    system prompt, so anything captured afterwards under-reports (PR
    #94753 review, P0).
    """
    return estimate_context_tokens(agent)


def cache_switch_notice_for_agent(
    *,
    agent: Any = None,
    old_model_display: str,
    new_model_display: str,
    est_context_tokens: Optional[int] = None,
    old_provider: str = "",
    new_provider: str = "",
    include_revert_hint: bool = True,
) -> Optional[str]:
    """Compose config gate + reselect check + builder in one call.

    ``est_context_tokens`` should come from :func:`snapshot_pre_switch_state`
    called BEFORE ``switch_model()``. When omitted, falls back to estimating
    from ``agent`` right now — only correct for surfaces that have not yet
    switched (or tests). ``agent`` may be None when only a pre-captured
    token count is available.
    """
    if not cache_switch_notice_enabled():
        return None
    if est_context_tokens is None:
        est_context_tokens = estimate_context_tokens(agent)
    return build_cache_switch_notice(
        old_model_display=old_model_display,
        new_model_display=new_model_display,
        est_context_tokens=est_context_tokens,
        is_reselect=same_model_reselect(
            old_model=old_model_display,
            old_provider=old_provider,
            new_model=new_model_display,
            new_provider=new_provider,
        ),
        include_revert_hint=include_revert_hint,
    )
