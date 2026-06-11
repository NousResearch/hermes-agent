"""hexis — metacognitive appraisal plugin for hermes-agent.

Observational per-turn appraisal with zero autonomy. Phase 2: pre_llm_call
runs one deadline-bounded JSON appraisal via ctx.llm and injects a compact
sanitized block. Phase 3: post_llm_call captures the completed turn and
on_session_end / on_session_start route through reflection.maybe_reflect —
the debounced, idempotent pass that carries appraisal context across the
one-turn lag (REFL-01..05; hooks stay thin, reflection.py does the work).

Hard rules for this module (see .planning/research/ARCHITECTURE.md):
- Zero import-time side effects: no DB, no config reads, no host imports
  at module top level. Only stdlib functools/logging/os are imported here;
  store/config/appraisal/render are imported lazily inside functions.
- Every hook accepts the documented kwargs PLUS **kwargs (the dispatcher
  injects extras such as telemetry_schema_version).
- Every hook is wrapped in the fail-open guard and returns None on any
  failure. A hexis hook must never raise into the dispatcher.
"""

import functools
import logging
import os

logger = logging.getLogger("hermes.plugins.hexis_appraisal")

# Host plugin context, stashed at register() time. None until register() runs.
_ctx = None

# Per-session throttle state (G6: long-lived gateway processes — reset on
# session rollover in on_session_start and in pre_llm_call's guard).
_session_state = {"session_id": None, "last_msg_norm": None}


def _fail_open(fn):
    """Wrap a hook so any exception is logged and swallowed (returns None).

    Telemetry stub fulfilled: raised paths record an llm_error row, inside
    a nested guard so a telemetry failure can never resurrect the exception.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            logger.warning(
                "hexis hook %s failed (fail-open): %s",
                fn.__name__,
                exc,
                exc_info=True,
            )
            try:
                from . import store

                store.record_telemetry(
                    "llm_error", error=(fn.__name__ + ": " + str(exc))[:300]
                )
            except Exception:
                pass
            return None

    return wrapper


@_fail_open
def on_session_start(session_id="", platform="", **kwargs):
    """Verify state-store availability, reset per-session caches, and run
    the session-change reflection trigger.

    maybe_reflect here runs BEFORE the new session's first appraisal — the
    ordering that lands cross-session surfacing on turn 1 of session B
    (ROADMAP criterion 3). Double-triggering with on_session_end is
    harmless BY DESIGN (the watermark makes the second fire a no-op).
    Latency: at most one reflect deadline once per session boundary —
    accepted (03-CONTEXT cost decision).
    """
    logger.debug("hexis on_session_start fired (session_id=%s)", session_id)
    from . import config, reflection, store  # lazy — zero import-time side effects

    _session_state["session_id"] = session_id or None
    _session_state["last_msg_norm"] = None
    config.reset_cache()

    if not store.ensure_db():
        logger.debug("hexis state store unavailable — continuing without state")

    llm = getattr(_ctx, "llm", None) if _ctx is not None else None
    reflection.maybe_reflect(llm=llm, session_id=session_id)
    return None


@_fail_open
def pre_llm_call(session_id="", task_id="", turn_id="", user_message="",
                 conversation_history=None, is_first_turn=False, model="",
                 platform="", sender_id="", **kwargs):
    """Run the appraisal pre-phase. Returns {"context": block} or None.

    Order is contractual (02-CONTEXT.md): kill switch -> session rollover ->
    throttle gates -> snapshot -> appraisal -> telemetry -> render/suppress.
    PLUG-04: user-message injection only.
    """
    logger.debug("hexis pre_llm_call fired (session_id=%s)", session_id)
    from . import appraisal, config, render, store  # lazy

    cfg = config.get_cfg()
    if not cfg.get("enabled", True):  # kill switch FIRST (APPR-07)
        store.record_telemetry("skipped:disabled", session_id=session_id)
        return None

    if session_id != _session_state["session_id"]:  # session rollover guard
        _session_state["session_id"] = session_id
        _session_state["last_msg_norm"] = None

    reason = appraisal.should_skip(user_message, _session_state["last_msg_norm"])
    if reason:  # throttle gates (APPR-08)
        store.record_telemetry("skipped:" + reason, session_id=session_id)
        return None
    _session_state["last_msg_norm"] = appraisal.normalize_message(user_message)

    snapshot = store.read_snapshot()  # None is fine — message salience still applies

    if _ctx is None or getattr(_ctx, "llm", None) is None:
        store.record_telemetry("skipped:no_ctx", session_id=session_id)
        return None

    result = appraisal.run_appraisal(
        llm=_ctx.llm,
        user_message=user_message,
        conversation_history=conversation_history or [],
        snapshot=snapshot,
        cfg=cfg,
    )

    store.record_telemetry(  # before returning; single quick INSERT, fail-open
        result.outcome,
        wall_ms=result.wall_ms,
        model=result.model,
        tokens_in=result.tokens_in,
        tokens_out=result.tokens_out,
        error=result.error,
        session_id=session_id,
    )

    if result.signals is None:
        return None
    # snapshot rides along for REFL-05 trust hints (advisory only; empty-
    # signal suppression inside render_block still takes precedence).
    block = render.render_block(result.signals, snapshot=snapshot)
    if block is None:  # empty-signal suppression (APPR-05)
        return None

    dump_path = os.environ.get("HEXIS_APPRAISAL_DEBUG_DUMP")
    if dump_path:  # live-demo observability aid (Plan 02-02); off by default
        try:
            with open(dump_path, "a", encoding="utf-8") as fh:
                fh.write(block + "\n")
        except Exception:
            pass

    return {"context": block}


@_fail_open
def post_llm_call(session_id="", task_id="", turn_id="", user_message="",
                  assistant_response="", conversation_history=None, model="",
                  platform="", **kwargs):
    """Capture the completed turn for reflection (REFL-01 cheap bookkeeping).

    post_llm_call is the only hook carrying the assistant response (the
    host's on_session_end has no transcript — turn_finalizer.py:294/:415);
    it fires once per turn, only `if final_response and not interrupted`.
    Returns None always.
    """
    logger.debug("hexis post_llm_call fired (session_id=%s)", session_id)
    from . import reflection  # lazy

    reflection.record_turn(
        session_id=session_id,
        turn_id=turn_id,
        user_message=user_message,
        assistant_response=assistant_response,
    )
    return None


@_fail_open
def on_session_end(session_id="", task_id="", turn_id="", completed=False,
                   interrupted=False, model="", platform="", **kwargs):
    """Reflection trigger. Return value is ignored by the host.

    Fires per run_conversation (per turn), not per session — the debounce
    (session change OR every-N unreflected turns) lives in
    reflection.maybe_reflect, which never raises and records a reflect_*
    telemetry row per firing.
    """
    logger.debug("hexis on_session_end fired (session_id=%s)", session_id)
    from . import reflection  # lazy

    llm = getattr(_ctx, "llm", None) if _ctx is not None else None
    reflection.maybe_reflect(llm=llm, session_id=session_id)
    return None


def register(ctx):
    """Host entry point. Stash ctx and register the four lifecycle hooks.

    Nothing else happens here — no DB, no config, no I/O.
    """
    global _ctx
    _ctx = ctx
    ctx.register_hook("on_session_start", on_session_start)
    ctx.register_hook("pre_llm_call", pre_llm_call)
    ctx.register_hook("post_llm_call", post_llm_call)
    ctx.register_hook("on_session_end", on_session_end)
