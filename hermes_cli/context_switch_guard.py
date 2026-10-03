"""Warn when an in-session model switch will trigger preflight compression on the next turn."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Callable, List, Optional

from agent.backend_identity import same_route
from hermes_cli.model_switch import ModelSwitchResult, resolve_display_context_length

# The switch summary is where the user learns what the next turn costs. Below the compression
# trigger there is still a cost: the first reply ships the whole history to a route whose prefix
# cache is cold (provider caches are per model/route), so a hosted prefill re-reads it at ~2-5k
# tok/s and the largest sessions feel that as a stall. Flagged once the session is at least halfway
# to the trigger this same summary already quotes. Calibration knob, not a contract.
COLD_READ_THRESHOLD_FRACTION = 0.5

# Why a target cannot promise preflight compression on the first turn. "never" = this runtime will
# not run it there, or the engine itself refuses a pass; "defer" = the figure is not anchored to
# provider usage (or not settled against the trigger the switch installs), so the turn is sent as-is
# and compression waits for the provider's own reading; "will" = an anchored figure that reaches the
# target trigger prices this transcript, and an anchored figure never defers.
COMPRESSION_NEVER = "never"
COMPRESSION_DEFER = "defer"
COMPRESSION_WILL = "will"

# Where a figure came from decides what the summary may say about it. A *measured* size — the
# payload this turn sends, the provider's price for it, or the compressor's own last provider
# reading — is a number the next reply really re-reads, and the only evidence that can locate the
# *next* request relative to a trigger. The compressor's display seed (``last_prompt_tokens`` with
# no paired provider reading behind it) is a *local estimate*: it sizes the session, but nothing
# provider-side has validated it, so it is named as an estimate and never quoted as "the session is
# ~N tokens". The session counter and the session row's recorded figure are *history*: they say what
# an earlier request cost, never what this one will.
SIZE_FROM_MEASURED = "measured"
SIZE_FROM_ESTIMATE = "estimate"
SIZE_FROM_COUNTER = "counter"
SIZE_FROM_RECORD = "record"
# A provider reading taken on the route the session was on. The figure is real, but it was priced
# on a payload and a route the destination does not necessarily share, so it sizes the display and
# never locates the next request relative to the destination's trigger.
SIZE_FROM_PRIOR_ROUTE = "prior_route"

# The phrase each non-measured source is quoted with. Keeping them distinct is the point: the two
# historical fallbacks are separate kinds of history, and neither is the local estimate.
_SIZE_LABELS = {
    SIZE_FROM_ESTIMATE: "Estimated current request size ~{figure:,} tokens (local estimate, "
                        "no provider reading for it yet)",
    # A cumulative sum over the session's requests, not the last measured one — the two are separate
    # fallbacks and must not be relabelled into each other.
    SIZE_FROM_COUNTER: "This session has sent ~{figure:,} prompt tokens in total; "
                       "the current request size is unavailable",
    SIZE_FROM_RECORD: "Last recorded prompt size was ~{figure:,} tokens; "
                      "the current request size is unavailable",
}


def size_label(figure: int, source: str) -> str:
    """The phrase the summary sizes the payload with: a measured size keeps the plain sentence, and
    every weaker class of evidence is named for what it is. Both surfaces that describe a switch
    quote a figure through this, so one assessment is not worded two ways."""
    label = _SIZE_LABELS.get(source)
    return label.format(figure=figure) if label else f"Session is ~{figure:,} tokens"


def _prices_the_next_request(source: str) -> bool:
    """Whether this class of evidence may locate the *next* request relative to a trigger.

    Only a provider-validated figure may. An estimate or a historical count says what some request
    cost, not what the request about to be built will — so the summary keeps its trigger language
    conditional on that evidence rather than lending it the authority of a measurement.
    """
    return source == SIZE_FROM_MEASURED


def _estimate_tokens(
    agent: Any, messages: Optional[List[dict]], durable_prompt_tokens: Optional[int] = None
) -> Optional[tuple]:
    """Size the payload the target route has to read, from the best evidence at hand.

    The durable transcript is the first choice — it exists whether or not a live agent does (the
    gateway evicts the cached agent on every committed switch), and it is what the new route
    re-reads. Live runtime facts only sharpen it (system prompt, tool schemas). Fallback chain for
    callers without a transcript: the compressor's last *provider* reading, then its display seed,
    then the session counter, then a durable prompt-token record (the session row's last
    API-reported figure).

    Returns ``(figure, source)``: the answer is only as current as its evidence, and the summary's
    wording is read off the source, so a historical figure can never be quoted as this request.
    """
    if messages is not None:
        try:
            from agent.model_metadata import estimate_request_tokens_rough

            system_prompt = getattr(agent, "_cached_system_prompt", None) or ""
            tools = getattr(agent, "tools", None)
            estimated = int(
                estimate_request_tokens_rough(
                    messages, system_prompt=system_prompt, tools=tools or None))
            # Only a positive figure is evidence. Every wired caller hands over a list — an empty one
            # for a session row with no transcript yet — and an empty list estimates 0, so returning
            # that would skip the fallback below on the exact state it exists for.
            if estimated > 0:
                return estimated, SIZE_FROM_MEASURED
        except Exception:
            pass

    cc = getattr(agent, "context_compressor", None)
    # ``update_from_response`` writes BOTH ``last_prompt_tokens`` and ``last_real_prompt_tokens``,
    # and ``update_model`` clears both, so a positive ``last_real_prompt_tokens`` is the provider's
    # own count for the last request and is the only compressor figure that measures traffic.
    # ``last_prompt_tokens`` alone may instead be a display-only preflight seed
    # (``maybe_seed_preflight_display_tokens``) written from a local estimate, which is a size for
    # the display and not a reading.
    real = int(getattr(cc, "last_real_prompt_tokens", 0) or 0)
    if real > 0:
        return real, SIZE_FROM_MEASURED
    display = int(getattr(cc, "last_prompt_tokens", 0) or 0)
    if display > 0:
        return display, SIZE_FROM_ESTIMATE
    session_prompt = int(getattr(agent, "session_prompt_tokens", 0) or 0)
    if session_prompt > 0:
        return session_prompt, SIZE_FROM_COUNTER
    recorded = int(durable_prompt_tokens or 0)
    return (recorded, SIZE_FROM_RECORD) if recorded > 0 else None


def _append_warning(result: ModelSwitchResult, text: str) -> None:
    if result.warning_message:
        result.warning_message = f"{result.warning_message} | {text}"
    else:
        result.warning_message = text


def _compression_config() -> dict:
    """The ``compression`` config section the engine is built from, over the shipped defaults.

    The guard's agentless callers — an evicted gateway session, an explicit ``--provider`` pick that
    never built an agent — still have to name the destination's policy, and the config section is
    where that policy lives. Defaults come from ``DEFAULT_CONFIG`` so a partial or unreadable config
    still states the shipped behaviour instead of silently dropping the note.
    """
    try:
        from hermes_cli.config import DEFAULT_CONFIG

        section = dict(DEFAULT_CONFIG.get("compression") or {})
    except Exception:
        section = {}
    try:
        from hermes_cli.config import load_config

        configured = (load_config() or {}).get("compression")
        if isinstance(configured, dict):
            section.update(configured)
    except Exception:
        pass
    return section


def _compression_enabled(agent: Any, resolved: Optional[bool] = None) -> bool:
    """Whether the destination runtime runs preflight compression at all.

    A live agent carries the resolved flag. Without one the flag comes from the initializer's own
    resolution when it is available (``resolved``), and from the section's flag parser otherwise —
    the same string-set truthiness the engine reads, so ``enabled: "false"`` is read as off rather
    than as a truthy string.
    """
    if agent is not None:
        return bool(getattr(agent, "compression_enabled", True))
    if resolved is not None:
        return bool(resolved)
    section = _compression_config()
    try:
        from agent.agent_init import _cfg_flag

        return bool(_cfg_flag(section, "enabled", True))
    except Exception:
        return bool(section.get("enabled", True))


def _resolved_installed_policy(
    model: str, provider: str, context_length: int, *, api_mode: str = "", base_url: str = "",
    custom_providers: list | None = None
) -> Optional[tuple]:
    """``(enabled, installed trigger)`` from the initializer's own producers, or ``None``.

    A destination with no live engine still has a policy: the one a fresh engine would install for
    it. Asking ``agent.agent_init.resolve_installed_compression_policy`` — the same parser and the
    same construction ``_build_context_engine`` uses — is what keeps the quoted trigger and the
    installed one from drifting over normalized flags, model adjustments and output reservation.
    ``None`` whenever those producers cannot answer, so the caller falls back to a figure it
    qualifies instead of presenting a re-derived one as the installed policy.
    """
    try:
        from agent.agent_init import resolve_installed_compression_policy
        from hermes_cli.config import load_config

        cfg = load_config() or {}
    except Exception:
        return None
    destination = SimpleNamespace(
        model=model, provider=provider, api_mode=api_mode, base_url=base_url, api_key="",
        max_tokens=None, quiet_mode=True, session_id=None,
        _compression_threshold_autoraised=None)
    try:
        resolved = resolve_installed_compression_policy(
            destination, cfg, int(context_length), custom_providers)
    except Exception:
        return None
    if resolved is None:
        return None
    enabled, threshold = resolved
    return bool(enabled), int(threshold or 0)


def _configured_trigger(
    model: str, context_length: int, provider: str, *, policy: Optional[tuple] = None,
    api_mode: str = "", base_url: str = "", custom_providers: list | None = None
) -> int:
    """The trigger the compressor would install for this destination, from its policy owner.

    Reached only when no live engine can answer for the destination, and resolved through the real
    configuration producer (``_resolved_installed_policy``) so what is quoted is what would be
    installed. The arithmetic below is the fallback for a config whose producer could not answer at
    all: it reproduces the engine's own trigger math from the ``compression`` section
    (``_effective_threshold_percent`` / ``_compute_threshold_tokens``) and states the full window,
    exactly as the engine's own ``max_tokens=None`` does — a figure to qualify, not an exact
    installed policy.
    """
    if policy is None:
        policy = _resolved_installed_policy(
            model, provider, context_length, api_mode=api_mode, base_url=base_url,
            custom_providers=custom_providers)
    if policy is not None:
        return int(policy[1])

    from agent.context_compressor import ContextCompressor, resolve_model_threshold

    section = _compression_config()
    overrides = section.get("model_thresholds")
    if not isinstance(overrides, dict):
        overrides = {}
    percent = resolve_model_threshold(
        model,
        {str(k): float(v) for k, v in overrides.items()
         if isinstance(v, (int, float)) and not isinstance(v, bool)},
        float(section.get("threshold", 0.5) or 0.5),
        provider)
    effective = ContextCompressor._effective_threshold_percent(context_length, percent)
    threshold = ContextCompressor._compute_threshold_tokens(context_length, effective, None)
    cap = section.get("threshold_tokens")
    if cap is not None:
        try:
            from agent.agent_init import _positive_int

            cap = _positive_int(cap)
        except Exception:
            cap = cap if isinstance(cap, (int, float)) and not isinstance(cap, bool) else None
    if isinstance(cap, (int, float)) and not isinstance(cap, bool) and cap > 0:
        threshold = min(threshold, min(int(cap), context_length))
    return int(threshold)


def _threshold_tokens(
    compressor: Any, model: str, context_length: int, provider: str = "", **policy_kwargs
) -> int:
    """The trigger the switch WILL install (cap, model_thresholds and small-window floor included),
    so the warning quotes the real number.

    A live engine answers for itself via ``preview_threshold_tokens``. Without one the same policy
    is resolved through the configuration producer the engine is built from, so an agentless caller
    states the installed trigger rather than a default ratio of its own.
    """
    preview = getattr(compressor, "preview_threshold_tokens", None)
    if callable(preview):
        return int(preview(model, context_length, provider))
    return _configured_trigger(model, context_length, provider, **policy_kwargs)


def _runtime_changes(result: ModelSwitchResult, current_route: tuple[str, str, str], agent: Any) -> bool:
    """Whether the switch installs a different runtime from the one the compressor is bound to.

    The comparison ``update_model`` itself makes (model, provider, endpoint, wire) — it decides
    which state the update keeps, so a forecast about the destination has to make the same one.
    """
    if not _same_route(result, current_route):
        return True
    return _mode_changes(result, agent)


def _mode_changes(result: ModelSwitchResult, agent: Any) -> bool:
    """Whether the switch moves the same deployment onto a different wire."""
    target_mode = str(getattr(result, "api_mode", "") or "")
    current_mode = str(getattr(agent, "api_mode", "") or "")
    return bool(target_mode and current_mode and target_mode != current_mode)


def _history_can_shrink(cc: Any, messages: Optional[List[dict]]) -> bool:
    """Whether preflight compression could actually drop anything from this payload.

    Read from the engine's own admission and retention rules: the protected head (the system
    prompt plus the decaying ``protect_first_n`` rows), the fixed minimum tail, and the
    token-bounded tail walk that decides where the compressible middle begins. ``protect_last_n``
    is a floor on what the tail keeps, not a veto on whether a pass runs, so a token-heavy history
    with fewer rows is still eligible and the compressor does shrink it.

    This is a fact about *compression*, not about the payload the new route still has to read —
    hence this gates the compression promise alone, never the cold-read note.
    """
    if messages is None:
        return True
    try:
        head_end = int(cc._protect_head_size(messages))
        minimum = head_end + 3 + 1
    except Exception:
        head_end = 0
        minimum = int(getattr(cc, "protect_first_n", 3)) + 3 + 1
    if len(messages) <= minimum:
        return False
    cut = getattr(cc, "_find_tail_cut_by_tokens", None)
    if not callable(cut):
        return True
    try:
        return int(cut(messages, head_end)) > head_end
    except Exception:
        return True


def _route_of(agent: Any) -> tuple[str, str, str]:
    """The route the session is on right now — ``("", "", "")`` when no live agent can say."""
    if agent is None:
        return ("", "", "")
    return (
        str(getattr(agent, "model", "") or ""),
        str(getattr(agent, "provider", "") or ""),
        str(getattr(agent, "base_url", "") or ""),
    )


def _same_route(result: ModelSwitchResult, current_route: tuple[str, str, str]) -> bool:
    """A reselect of the deployment the session already runs on. Every caller merges this warning
    before the swap, so the route facts still name the current route; its prefix cache is the warm
    one, and a cold-read note there would be false.

    The comparison itself is ``agent.backend_identity.same_route`` — the owner both this summary and
    the selection confirmation read, so the two surfaces cannot describe one transition two ways.
    """
    current_model, current_provider, current_url = current_route
    return same_route(
        old_model=current_model, old_provider=current_provider, old_base_url=current_url,
        new_model=result.new_model,
        new_provider=result.target_provider or current_provider,
        new_base_url=result.base_url or current_url,
        provider_changed=bool(result.provider_changed))


def _anchored_tokens(agent: Any, messages: Optional[List[dict]]) -> Optional[int]:
    """The provider's priced figure while the runtime's usage anchor still covers this transcript.

    This is the same evidence ``agent.turn_context._preflight_request_tokens`` consults, and the
    reason a preflight figure does not defer to a later provider reading — read from that owner
    rather than re-derived here.
    """
    if agent is None or messages is None:
        return None
    try:
        from agent.usage_anchor import anchored_context_tokens

        return anchored_context_tokens(messages, getattr(agent, "_usage_anchor", None))
    except Exception:
        return None


def _anchor_prices_destination(
    agent: Any, result: ModelSwitchResult, current_route: tuple[str, str, str]
) -> bool:
    """Whether the session's usage anchor may speak for the route the switch installs.

    The anchor is the provider's priced figure for *this* transcript, captured on the route the
    session runs on now, and it carries no route of its own — so it cannot show that the destination
    prices it. Deployment equality (resolved model, provider, endpoint) is the cold-read note's
    question and not proof of that either, and the wire is part of the route as well: a mode change
    is not the same read. Establishing anchor-validity for a destination belongs to the provenance
    owner in #129177, so until that lands only a switch that keeps the current deployment on the same
    wire keeps the definite promise; anything else prices the display, never the promise.
    """
    if not _same_route(result, current_route):
        return False
    return not _mode_changes(result, agent)


def _codex_owns_compaction(agent: Any, result: ModelSwitchResult) -> bool:
    """Whether a codex app-server side owns compaction for this switch.

    codex app-server threads are compacted by codex itself and Hermes runs no preflight there. The
    runtime gate reads the live ``api_mode``, which at merge time is still the wire the session came
    from, so reading it alone describes where the session *left* and promises a Hermes pass on a
    switch *into* codex app-server. Both ends are asked — the destination the switch installs
    (``result.api_mode``) and the mode the session is already on — through the runtime's own gate, so
    the summary and the runtime cannot drift apart.
    """
    try:
        from agent.turn_context_compaction import _codex_native_auto_compaction
    except Exception:
        return False

    def _owns(api_mode: str) -> bool:
        if not api_mode:
            return False
        try:
            return bool(_codex_native_auto_compaction(SimpleNamespace(
                api_mode=api_mode,
                codex_app_server_auto_compaction=getattr(
                    agent, "codex_app_server_auto_compaction", "native"))))
        except Exception:
            return False

    return (
        _owns(str(getattr(agent, "api_mode", "") or ""))
        or _owns(str(getattr(result, "api_mode", "") or ""))
    )


def _engine_block_reason(cc: Any, figure: int) -> Optional[str]:
    """The engine's own block reason for a pass at ``figure``, never a blocker list re-derived here.

    ``should_compress_info`` reports a reason only *past the compressor's own trigger*, so the
    question is asked at the higher of ``figure`` and that trigger: a figure below the trigger is an
    admission about neither the old nor the new one.
    """
    info = getattr(cc, "should_compress_info", None)
    if not callable(info):
        return None
    probe = max(int(figure or 0), int(getattr(cc, "threshold_tokens", 0) or 0))
    try:
        return info(probe)[1]
    except Exception:
        return None


def _blocker_outlives_switch(cc: Any, reason: str, runtime_changes: bool) -> bool:
    """Whether a block reason the engine reports *now* is still armed after ``update_model``.

    ``update_model`` voids the ineffective strikes unconditionally, and scopes the summary-failure
    cooldown and the fallback-summary streak to the runtime that produced them; the structural
    no-op backoff is transient and survives. A reason read before the swap is therefore only a
    verdict about the destination when it is one the update keeps — otherwise the switch clears it
    and the next turn runs the very pass the engine is refusing now.
    """
    if reason.startswith("structural_backoff"):
        return True
    if reason.startswith("cooldown"):
        return not runtime_changes
    # ``ineffective``: the strike counter is voided by the update; a fallback-summary streak is
    # cleared only when the runtime changes.
    return not runtime_changes and int(getattr(cc, "_fallback_compression_streak", 0) or 0) >= 2


def _compression_disposition(
    agent: Any, cc: Any, messages: Optional[List[dict]], estimate: int,
    anchored: Optional[int], target_threshold: int, result: ModelSwitchResult,
    current_route: tuple[str, str, str], resolved_enabled: Optional[bool] = None) -> str:
    """What the target's first turn does about compression, read from the owners that decide it.

    ``"defer"`` is the honest answer whenever the figure is not one the runtime will price: the
    switched compressor has just had its usage reset, so a rough estimate past the trigger waits one
    request for real usage. ``"will"`` needs a provider-priced figure (the usage anchor) that reaches
    the trigger the switch installs — a sub-trigger anchor is priced by the next turn as not needing
    a pass, however large the rough estimate is — *and* an anchor whose authority over this
    destination is established, which the caller decides (``_anchor_prices_destination``) and signals
    by handing over ``None`` here: an unverified anchor prices the display, never the promise.
    ``"never"`` is the engine's own refusal — forecast past the update, so a blocker the switch
    clears does not suppress a warning for a pass the next turn actually runs.
    """
    if not _compression_enabled(agent, resolved_enabled):
        return COMPRESSION_NEVER
    if not _history_can_shrink(cc, messages) or _codex_owns_compaction(agent, result):
        return COMPRESSION_NEVER
    # The engine's own admission, asked at the figure that comparison is defined at. On a window
    # shrink the quoted figure sits below the trigger this compressor still owns while the target
    # turn prices a lower one that the same figure is past — there the engine reports only that a
    # blocker is armed, past a trigger it is about to stop using, so that is uncertainty about the
    # target, not a verdict from it. A reason is only reported past a trigger, hence the probe.
    trigger = int(getattr(cc, "threshold_tokens", 0) or 0)
    shrink = bool(trigger) and estimate < trigger
    reason = _engine_block_reason(cc, trigger if shrink else estimate)
    if reason is not None and (
            shrink or _blocker_outlives_switch(cc, reason, _runtime_changes(result, current_route, agent))):
        return COMPRESSION_DEFER if shrink else COMPRESSION_NEVER
    if anchored is None or anchored < target_threshold:
        return COMPRESSION_DEFER
    return COMPRESSION_WILL


def _append_cold_read_note(
    result: ModelSwitchResult, current_route: tuple[str, str, str], estimate: int, threshold: int,
    source: str = SIZE_FROM_MEASURED
) -> None:
    """Note the pre-read a switch costs when no compression is on the way.

    Nothing in the delay is something Hermes rebuilds: the destination answers from a prefix cache
    only a route that already served this session could have warmed, so the cost is the history
    itself being re-read. Small sessions answer from a cold cache fast enough that the line would be
    noise, hence the floor. Only a measured figure may be quoted as this request's payload
    (``_prices_the_next_request``); a local estimate or a historical count carries the read instead,
    and the note says which evidence it read rather than presenting it as the next request's size.
    Nothing here records whether *this* destination served the session earlier, so the cold cache is
    stated as the condition it is — a route that has not served the session has no warm cache —
    never as a claim about this destination.
    """
    if _same_route(result, current_route) or estimate < int(threshold * COLD_READ_THRESHOLD_FRACTION):
        return
    if _prices_the_next_request(source):
        read = (f"Session is ~{estimate:,} tokens; the first reply on {result.new_model} re-reads "
                f"them before it answers")
    else:
        read = (f"{size_label(estimate, source)}; the first reply on {result.new_model} may have "
                f"to re-read the session's history before it answers")
    _append_warning(
        result,
        f"{read} — if that route has not served this session it has no warm prefix cache, so expect "
        f"a delay on large sessions.")


def merge_preflight_compression_warning(
    result: ModelSwitchResult,
    *,
    agent: Any = None,
    messages: Optional[List[dict]] = None,
    custom_providers: list | None = None,
    config_context_length: int | None = None,
    configured_model: str | None = None,
    configured_provider: str | None = None,
    configured_base_url: str | None = None,
    durable_prompt_tokens: int | None = None) -> None:
    """If the next user message will likely preflight-compress, append a warning.

    ``agent`` is optional: with the durable transcript (``messages``) a caller that has no live
    agent — an evicted gateway session, a restart — still gets the switch-cost note, and an
    unavailable runtime forecast reads as "compression disposition unknown", never as a small
    session. ``durable_prompt_tokens`` is the session row's last API-reported prompt size, used only
    when no transcript and no live compressor can size the payload; it is a record of an earlier
    request, so the copy it drives says so instead of quoting it as this turn's payload.
    """
    if not result.success:
        return

    cc = getattr(agent, "context_compressor", None)
    if agent is not None and cc is None:
        return
    if agent is None and messages is None:
        return

    # Fall back to the agent's custom providers: without them the shrink warning used the
    # hardcoded catalog (e.g. "qwen" → 131072) even when the provider declared 1M.
    if custom_providers is None:
        custom_providers = getattr(agent, "_custom_providers", None)

    def _or_agent(value, attr):
        return value if value is not None else getattr(agent, attr, None)

    old_ctx = int(getattr(cc, "context_length", 0) or 0)
    new_ctx = resolve_display_context_length(
        result.new_model,
        result.target_provider,
        base_url=result.base_url or getattr(agent, "base_url", "") or "",
        api_key=result.api_key or getattr(agent, "api_key", "") or "",
        model_info=result.model_info,
        custom_providers=custom_providers,
        config_context_length=config_context_length,
        configured_model=_or_agent(configured_model, "model"),
        configured_provider=_or_agent(configured_provider, "provider"),
        configured_base_url=_or_agent(configured_base_url, "base_url"))
    if not new_ctx:
        return

    sized = _estimate_tokens(agent, messages, durable_prompt_tokens)
    if sized is None:
        return
    estimate, source = sized
    current_route = _route_of(agent)
    # The next turn prices a provider usage anchor, not this rough figure. When one covers the
    # transcript it is the number to quote *and* the one the trigger comparison is made on, so the
    # sentence and the disposition cannot be read off two different figures — but it was priced on
    # the route the session runs on now, so it only *promises* for a destination it can speak for.
    # Until the provenance owner (#129177) can say that, an unverified anchor is handed on as "no
    # anchor": the figure still informs the display and the rest stays conditional.
    anchored = _anchored_tokens(agent, messages)
    if anchored is not None:
        estimate = anchored
        source = SIZE_FROM_MEASURED
    authoritative = anchored if _anchor_prices_destination(agent, result, current_route) else None

    # A destination with no live engine still has a resolved policy, and the same producer that
    # builds the engine answers for it: ask before quoting a trigger, and use its enabled flag too.
    policy = None
    if cc is None:
        policy = _resolved_installed_policy(
            result.new_model, result.target_provider, new_ctx,
            api_mode=str(result.api_mode or ""), base_url=str(result.base_url or ""),
            custom_providers=custom_providers)
        if policy is not None and not policy[1]:
            return
    new_threshold = _threshold_tokens(
        cc, result.new_model, new_ctx, result.target_provider, policy=policy)
    if estimate < new_threshold:
        _append_cold_read_note(result, current_route, estimate, new_threshold, source)
        return

    # A compression notice is a promise about the next turn; the cold-read note is a plain statement
    # about the cost the switch already carries. Only the promise needs the target runtime to be
    # able to run the pass — the payload is sent either way. Same for the suppression rules: they
    # decide whether to promise compression, not whether the switch costs a re-read.
    disposition = _compression_disposition(
        agent, cc, messages, estimate, authoritative, new_threshold, result, current_route,
        resolved_enabled=(policy[0] if policy else None))
    if disposition == COMPRESSION_NEVER:
        _append_cold_read_note(result, current_route, estimate, new_threshold, source)
        return

    parts: list[str] = []
    if old_ctx and new_ctx < old_ctx:
        parts.append(f"Context window shrinks ({old_ctx:,} → {new_ctx:,}). ")
    parts.append(
        f"{size_label(estimate, source)}; "
        f"{result.new_model} allows {new_ctx:,} "
        f"(auto-compress at ~{new_threshold:,}). ")
    if disposition == COMPRESSION_WILL:
        parts.append("Your next message will run preflight compression before the model replies.")
    elif not _prices_the_next_request(source):
        # The figure is a local estimate or a historical count, so nothing provider-side has priced
        # the request about to be sent: it cannot locate that request relative to the trigger, and
        # saying it did would give the figure authority its evidence does not carry. The run's own
        # decision is a deferral until real usage arrives either way — say both, promise neither.
        parts.append(
            "Whether the next message is past that trigger is unknown: the figure above is not a "
            "provider reading for this request, so the run either runs preflight compression before "
            "the model replies or sends it as-is; fresh provider usage will inform subsequent "
            "compression decisions.")
    else:
        # A measured figure past the trigger: the pass is not certain either, because the switched
        # compressor has had its usage reset — but the size itself is current, so the sentence may
        # state the comparison it makes.
        parts.append(
            "Your next message is past that trigger: the run either runs preflight compression "
            "before the model replies, or sends it as-is; fresh provider usage will inform "
            "subsequent compression decisions.")
    _append_warning(result, "".join(parts))


def enrich_model_switch_warnings_for_gateway(
    result: ModelSwitchResult,
    runner: Any,
    *,
    session_key: str,
    source: Any,
    custom_providers: list | None = None,
    load_gateway_config: Callable[[], dict] | None = None) -> None:
    """Gateway helper: cached agent when there is one, plus the durable session transcript.

    The transcript and the cached agent have separate lifetimes — the committed switch path evicts
    the agent so the next turn rebuilds from the override, and a second ``/model`` before that turn
    still faces the same large conversation. So history is read independently of agent residency,
    and the session row's last reported prompt size stands in for a live measurement — including for
    a session with no transcript rows yet, which is a size to quote, not an absent one.
    """
    lock = getattr(runner, "_agent_cache_lock", None)
    cache = getattr(runner, "_agent_cache", None)
    agent = None
    if lock is not None and cache is not None:
        with lock:
            entry = cache.get(session_key)
            if entry and entry[0] is not None:
                agent = entry[0]

    configured: dict = dict.fromkeys(
        ("config_context_length", "configured_model", "configured_provider", "configured_base_url"))
    if load_gateway_config is not None:
        try:
            cfg = load_gateway_config()
            model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
            if isinstance(model_cfg, dict) and model_cfg.get("context_length") is not None:
                configured.update(
                    config_context_length=int(model_cfg["context_length"]),
                    configured_model=model_cfg.get("default") or model_cfg.get("model"),
                    configured_provider=model_cfg.get("provider"),
                    configured_base_url=model_cfg.get("base_url"))
        except Exception:
            pass

    messages = None
    durable_prompt_tokens = None
    db = getattr(runner, "_session_db", None)
    store = getattr(runner, "session_store", None)
    if db is not None and store is not None:
        try:
            session_entry = store.get_or_create_session(source)
            messages = db.get_messages_as_conversation(session_entry.session_id)
            durable_prompt_tokens = int(getattr(session_entry, "last_prompt_tokens", 0) or 0) or None
        except Exception:
            pass

    if agent is None and not messages and not durable_prompt_tokens:
        return

    merge_preflight_compression_warning(
        result, agent=agent, messages=messages, custom_providers=custom_providers,
        durable_prompt_tokens=durable_prompt_tokens, **configured)
