"""Per-turn driver and the session-lifetime reader of ``ClaudeAgentSdkSession``.

``run_turn`` (watchdog-supervised turn from the caller thread), ``_consume_turn``
(claims the shared stream, projects messages, settles the terminal result),
``_reader_loop`` (the single stream reader arbitrating turn claims and publishing
stream death) and unsolicited-result handling. The reader and the turn stay
together on purpose: they jointly drive one claim/release protocol over
``_turn_claims`` / ``_turn_inbox`` / ``_stream_ended``. Extracted from
``claude_agent_sdk_session.py``; every method resolves through
``ClaudeAgentSdkSession``'s MRO unchanged. ``ClaudeSdkEventProjector`` is bound
here — tests replacing the projector patch this module's name.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Optional

# TurnResult is the shared contract with the runtime glue — reused verbatim
# from the codex session module (same fields, same semantics) so
# ``run_claude_agent_sdk_turn`` mirrors ``run_codex_app_server_turn`` 1:1.
from agent.transports.codex_app_server_session import TurnResult
from agent.transports.claude_sdk_event_projector import ClaudeSdkEventProjector
from agent.transports.claude_agent_sdk_session_availability import (
    _safe_sdk_error_text,
    _safe_sdk_traceback,
    classify_auth_failure,
)
from agent.transports.claude_agent_sdk_session_input import (
    _coerce_turn_input,
    _sdk_user_message_stream,
)
from agent.transports.claude_agent_sdk_session_config import (
    _configured_post_tool_quiet_timeout,
    _configured_turn_timeout,
)
from agent.transports.claude_agent_sdk_session_watchdog import (
    _DEFAULT_POST_TOOL_QUIET_STREAMING,
    _DEFAULT_TURN_TIMEOUT,
    _POLL_STALL_FACTOR,
    _StreamEnd,
    _TURN_ABORT_GRACE,
    _TurnWatch,
)

# Same logger name as the origin module so log records / caplog filters are unchanged.
logger = logging.getLogger("agent.transports.claude_agent_sdk_session")


class ClaudeSdkTurnMixin:
    """Turn driver, stream consumer and reader loop (see module docstring)."""

    # ---------- per-turn ----------

    def run_turn(
        self,
        user_input: Any,
        *,
        turn_timeout: Optional[float] = None,
        post_tool_quiet_timeout: Optional[float] = None,
        watch_poll_interval: float = 1.0,
        abort_grace: float = _TURN_ABORT_GRACE,
    ) -> TurnResult:
        """Send a user message and block until the SDK's ResultMessage,
        projecting the typed stream into Hermes' messages shape.

        Turn lifetime is activity-aware, not a bare wall clock (production
        forensics: four of six 600s kills were actively-working turns).
        `turn_timeout` (explicit arg > agent.claude_agent_sdk.turn_timeout >
        600s) is a SOFT budget: it only fires when nothing is outstanding —
        no tool running, no approval awaiting a human — and the stream has
        ALSO been quiet ≥ min(30s, budget). A post-tool quiet watchdog
        (`post_tool_quiet_timeout`; default 90s with streaming on, disabled
        with streaming off) catches wedges early: armed when a tool result
        arrives, cleared by any later activity. On a trip the CLI is
        interrupted and given `abort_grace` to unwind — a clean unwind keeps
        the partial transcript and the resumable session id (no retire);
        only a grace expiry hard-cancels and retires."""
        result = TurnResult()
        prompt = _coerce_turn_input(user_input)
        if isinstance(prompt, str) and not prompt.strip():
            self.consume_interrupt()
            result.final_text = (
                "This Claude Agent SDK route can't process an empty message. "
                "Please send text or a supported image."
            )
            result.error = result.final_text
            result.api_call_made = False
            return result
        try:
            self.ensure_started()
        except Exception as exc:
            safe_exc = _safe_sdk_error_text(exc)
            hint = classify_auth_failure(safe_exc)
            result.error = hint or f"claude-agent-sdk startup failed: {safe_exc}"
            result.should_retire = True
            # Keep the traceback: `str(exc)` alone collapses a KeyError to a
            # bare quoted key (e.g. "'anyio'"), which is undiagnosable from the
            # user-facing error string. WARNING so it reaches errors.log.
            logger.warning(
                "claude-agent-sdk startup failed (%s: %s)\n"
                "Traceback (most recent call last):\n%s",
                type(exc).__name__,
                safe_exc,
                _safe_sdk_traceback(exc),
            )
            # A refusal to start is fatal to the run, not turn-scoped: the
            # metered-key guard and an uninstallable SDK are config errors
            # ("startup" — NOT "billing": kanban maps failure_reason
            # "billing" to the transient EX_TEMPFAIL requeue sentinel, and
            # retrying cannot fix a present metered key); an auth-classified
            # failure is "auth".
            result.fatal_reason = "auth" if hint else "startup"
            return result

        with self._interrupt_commit_lock:
            # Re-open interrupt admission for this turn.  A post-terminal
            # request that no outer runtime consumed belongs to this next
            # direct turn and becomes its normal pre-set interrupt.
            self._terminal_result_committed = False
            if self._post_terminal_interrupt_pending:
                self._post_terminal_interrupt_pending = False
                self._interrupt_event.set()

        # Text buffered before this turn whose terminal ResultMessage never
        # arrived is partial mid-burst content — a later unrelated result
        # must never pick it up (misattribution is the proven worse failure;
        # texts parked DURING the turn re-buffer via the residue drain and
        # are unaffected). Never a silent drop: WARN what is discarded.
        stale = list(self._unsolicited_text)
        if stale:
            self._unsolicited_text.clear()
            logger.warning(
                "claude-agent-sdk: discarding %d stale unsolicited text(s) "
                "(%d chars) buffered before this turn — their terminal "
                "ResultMessage never arrived; attaching them to a later "
                "unrelated result is the proven worse failure",
                len(stale), sum(len(t) for t in stale),
            )

        # An interrupt that arrived between turns or during connect (up to
        # 60s) targets THIS turn — honor it instead of erasing it. (The old
        # unconditional clear() silently swallowed that window.)
        if self._interrupt_event.is_set():
            self.consume_interrupt()
            result.interrupted = True
            return result

        import concurrent.futures

        budget = (
            float(turn_timeout)
            if turn_timeout is not None
            else (_configured_turn_timeout() or _DEFAULT_TURN_TIMEOUT)
        )
        if post_tool_quiet_timeout is not None:
            quiet = float(post_tool_quiet_timeout)
        else:
            configured_quiet = _configured_post_tool_quiet_timeout()
            if configured_quiet is not None:
                quiet = configured_quiet  # 0 = explicitly disabled
            else:
                quiet = (
                    _DEFAULT_POST_TOOL_QUIET_STREAMING if self._streaming else 0.0
                )
        poll = max(0.01, float(watch_poll_interval))

        assert self._loop is not None, "loop thread not started"
        watch = _TurnWatch()
        self._turn_watch = watch
        trip: Optional[str] = None
        trip_elapsed = trip_idle = 0.0
        hard_trip = False
        turn_data: Optional[dict[str, Any]] = None
        future = asyncio.run_coroutine_threadsafe(
            self._consume_turn(prompt), self._loop
        )
        try:
            pending_verdict: Optional[str] = None
            prev_poll = time.monotonic()
            while turn_data is None and trip is None:
                try:
                    turn_data = future.result(timeout=poll)
                except (TimeoutError, concurrent.futures.TimeoutError):
                    # py3.11 unifies TimeoutError/asyncio.TimeoutError/
                    # concurrent.futures.TimeoutError — a DONE future here
                    # means the COROUTINE settled in the race window: with an
                    # exception (typically its own TimeoutError — a socket/
                    # pipe timeout under the CLI), re-raise into the
                    # classification path instead of misreading it as a poll
                    # expiry (which would busy-spin until the budget); with a
                    # RESULT, harvest it — the turn completed.
                    if future.done():
                        if future.exception() is not None:
                            raise
                        turn_data = future.result()
                        continue
                    now = time.monotonic()
                    if now - prev_poll > _POLL_STALL_FACTOR * poll:
                        # The PROCESS was descheduled (swap/OOM), not the
                        # turn — that gap proves nothing about the stream.
                        watch.rebaseline()
                        prev_poll = now
                        pending_verdict = None
                        continue
                    prev_poll = now
                    verdict = watch.check(budget=budget, quiet=quiet)
                    if verdict is None:
                        pending_verdict = None
                        continue
                    if pending_verdict != verdict:
                        # Debounce: two consecutive polls must agree before
                        # a trip (closes the tick-vs-check race window).
                        pending_verdict = verdict
                        continue
                    now = time.monotonic()
                    trip = verdict
                    trip_elapsed = now - watch.started
                    trip_idle = now - watch.last_activity
            if trip is not None and turn_data is None:
                if future.done():
                    # Completed between check() and the trip decision —
                    # completion wins, the trip is void.
                    turn_data = future.result()
                    trip = None
                else:
                    # Interrupt through request_interrupt(): setting
                    # _interrupt_event is REQUIRED — _consume_turn's W22 EDE
                    # mask keys on the turn-local flag, so the CLI's
                    # interrupt-ack is masked instead of delivered as a
                    # fresh error.
                    self.request_interrupt()
                    try:
                        turn_data = future.result(timeout=abort_grace)
                    except (TimeoutError, concurrent.futures.TimeoutError):
                        if future.done():
                            if future.exception() is not None:
                                raise
                            turn_data = future.result()
                        elif not future.cancel():
                            # Settled in the cancel race window — a real
                            # result beats a fabricated hard trip.
                            if future.exception() is not None:
                                raise
                            turn_data = future.result()
                        else:
                            hard_trip = True
            if trip is not None and turn_data is not None and not hard_trip:
                if (
                    not turn_data["error"]
                    and turn_data.get("result_uuid")
                    and turn_data["final_text"]
                ):
                    # The turn FINISHED inside the grace window with a real
                    # answer — deliver it in full; the trip never happened.
                    # final_text is REQUIRED: the W22-masked interrupt ack
                    # also has error=None + a result uuid, but its final_text
                    # is empty (projection stopped at the interrupt) — that
                    # shape must stay a trip, or it degrades into a silent
                    # dropped turn. Consume our own interrupt signal so the
                    # mapping below cannot misread it. (A real user /stop
                    # racing this exact window loses its signal too — the
                    # completed answer it targeted is delivered, same as any
                    # near-boundary stop today.)
                    self.consume_interrupt()
                    # The watchdog's internal interrupt was withdrawn because
                    # the turn completed in full during grace. Clear the
                    # stream-consumer snapshot too; otherwise final mapping
                    # resurrects the voided trip as a user interruption.
                    turn_data["interrupt_observed"] = False
                    logger.info(
                        "claude-agent-sdk: turn completed during watchdog "
                        "grace (%.0fs elapsed) — delivered in full",
                        time.monotonic() - watch.started,
                    )
                    trip = None
        except Exception as exc:
            self.consume_interrupt()
            safe_exc = _safe_sdk_error_text(exc)
            hint = classify_auth_failure(safe_exc)
            result.error = hint or f"claude-agent-sdk turn failed: {safe_exc}"
            result.should_retire = True
            if hint is not None:
                # Auth failures are fatal; other mid-turn exceptions stay
                # transient (retire + retry semantics unchanged).
                result.fatal_reason = "auth"
            return result
        finally:
            self._turn_watch = None

        if turn_data is None:
            # Hard trip: the CLI ignored the interrupt for the whole grace —
            # nothing was harvested. Retire (today's shape, now the rare
            # fallback for a genuinely unresponsive CLI).
            self.consume_interrupt()
            result.interrupted = True
            result.error = self._format_trip_error(
                trip, budget, quiet, trip_elapsed, trip_idle
            )
            result.should_retire = True
            return result

        result.final_text = turn_data["final_text"]
        result.projected_messages = turn_data["messages"]
        result.tool_iterations = turn_data["tool_iterations"]
        result.token_usage_last = turn_data["usage"]
        result.token_usage_total = turn_data["usage"]
        result.model_last = turn_data.get("model")
        result.billing_mode = turn_data.get("billing_mode", "unknown")
        result.billing_evidence = turn_data.get("billing_evidence", {})
        result.total_cost_usd = turn_data.get("total_cost_usd")
        result.api_call_made = turn_data.get("api_call_made", True)
        result.thread_id = self._session_id
        result.turn_id = turn_data.get("result_uuid")
        # The stream consumer records interruption at the last point where it
        # can still precede terminal acceptance.  Never re-read the live event
        # here: a /stop can arrive after ResultMessage while ownership release
        # is finishing, and must not retroactively downgrade the committed
        # result.  The SDK-specific marker is deliberately dynamic so the
        # shared Codex TurnResult contract stays unchanged.
        result.terminal_result_accepted = bool(
            turn_data.get("terminal_result_accepted", False)
        )
        result.interrupted = bool(turn_data.get("interrupt_observed", False))
        # A non-terminal turn can spend time releasing foreground ownership
        # after the stream consumer's last snapshot. Restore the live read for
        # that path so a stop admitted during a stream-death/release handshake
        # remains authoritative. Terminal results stay fenced: once accepted,
        # no later event may retroactively downgrade the completed answer.
        if not result.terminal_result_accepted:
            with self._interrupt_commit_lock:
                result.interrupted = (
                    result.interrupted or self._interrupt_event.is_set()
                )
        # Consume either the honored pre-terminal interrupt or a late stop
        # aimed at the turn that has already committed. Both the event and
        # post-terminal pending state must clear together so neither can bleed
        # into the next turn on this session object.
        self.consume_interrupt()
        # The fence covered commit -> mapping. It ends HERE, with the result in
        # the caller's hands: between turns the CLI can still be running an
        # unsolicited background task, and a /stop then must reach the child
        # (and pre-set the next turn) exactly as it did before the fence.
        with self._interrupt_commit_lock:
            self._terminal_result_committed = False
        if turn_data["error"]:
            # A prior MCP tool use is not evidence that this terminal SDK
            # error belongs to MCP; preserve fail-closed Claude auth handling
            # unless the error text itself identifies an MCP origin.
            hint = classify_auth_failure(turn_data["error"])
            result.error = hint or turn_data["error"]
            if hint is not None:
                result.should_retire = True
                result.fatal_reason = "auth"
        if turn_data.get("billing_guard_violation"):
            # This is durable account/config state, not a transient provider
            # billing error: the generic "billing" fatal_reason is a retry
            # sentinel, which would repeatedly spend calls against the same
            # unsafe lane. "startup" stops the run until the operator fixes
            # Extra Usage/credentials or explicitly opts into metering.
            result.should_retire = True
            result.fatal_reason = "startup"
        if turn_data.get("stream_ended"):
            # A dead stream is permanent on this session object: the reader
            # exited, _stream_ended stays set, and ensure_started() keeps
            # returning early while _client is non-None — so WITHOUT retire,
            # every later turn short-circuits to this same error with zero
            # model calls, forever (the CLI died once and poisoned the
            # session; observed as the unrecoverable half of the 2026-08-09
            # desync incident). Retire lets the runtime rebuild a fresh CLI
            # on the next attempt/turn. Model-level result subtypes
            # (error_max_turns, error_max_budget_usd) stay non-retiring.
            result.should_retire = True
        if trip is not None:
            # Clean-ack trip: the CLI honored our interrupt inside the grace,
            # the partial transcript above is preserved, and the session id
            # is resumable — mirror the user-interrupt lane (close-but-keep-
            # resume-id) instead of retiring. The trip text WINS over
            # whatever error the drain surfaced (typically the W22-masked
            # interrupt ack) — EXCEPT auth and stream-death, which keep
            # their retire verdicts from the mapping above.
            if result.fatal_reason != "auth":
                result.error = self._format_trip_error(
                    trip, budget, quiet, trip_elapsed, trip_idle
                )
            result.interrupted = True
        return result

    def _format_trip_error(
        self,
        kind: Optional[str],
        budget: float,
        quiet: float,
        elapsed: float,
        idle: float,
    ) -> str:
        """≤200 chars (the gateway truncates at str(err)[:200]); keeps the
        literal "turn timed out" needle; names the cause."""
        if kind == "post_tool_quiet":
            return (
                f"turn timed out: no SDK activity for {idle:.0f}s after a "
                f"tool result (turn ran {elapsed:.0f}s, quiet limit "
                f"{quiet:.0f}s)"
            )
        return (
            f"turn timed out after {elapsed:.0f}s "
            f"(budget {budget:.0f}s, idle {idle:.0f}s)"
        )

    # ---------- internals ----------

    async def _consume_turn(self, prompt: Any) -> dict[str, Any]:
        """The async side of one turn: query, then read THIS turn's messages
        off the reader loop's inbox until the ResultMessage.

        Deliberately NOT `receive_response()` — that helper serves the oldest
        buffered ResultMessage, which is not necessarily ours. See
        `_reader_loop` for why that is a permanent-corruption bug."""
        projector = ClaudeSdkEventProjector()
        out: dict[str, Any] = {
            "final_text": "",
            "messages": [],
            "tool_iterations": 0,
            "usage": None,
            "error": None,
            "result_uuid": None,
            "model": None,
            "stream_ended": False,
            "mcp_tool_seen": False,
            "billing_guard_violation": False,
            "billing_mode": self._reported_billing_mode(),
            "billing_evidence": dict(self._billing_evidence),
            "total_cost_usd": None,
            "api_call_made": True,
            "interrupt_observed": False,
            "terminal_result_accepted": False,
        }

        boundary_interrupt = False

        def _snapshot_interrupt() -> bool:
            with self._interrupt_commit_lock:
                observed = self._interrupt_event.is_set()
            out["interrupt_observed"] = observed
            return observed

        ended = self._stream_ended
        if ended is not None:
            _snapshot_interrupt()
            out["error"] = "SDK message stream ended before this turn" + (
                f": {_safe_sdk_error_text(ended.error)}" if ended.error else ""
            )
            out["stream_ended"] = True
            return out
        if self._billing_guard_error is not None:
            _snapshot_interrupt()
            out["error"] = self._billing_guard_error
            out["billing_guard_violation"] = True
            out["billing_mode"] = self._reported_billing_mode()
            out["billing_evidence"] = dict(self._billing_evidence)
            out["api_call_made"] = False
            return out
        inbox: Any = asyncio.Queue()
        # Ask the sole reader to drain everything already ahead of this turn
        # in the resumed client's FIFO, then atomically install our inbox.  A
        # direct assignment here races an already-buffered background result:
        # the reader can observe the new inbox before it observes that older
        # result and mis-serve it as this query's answer.
        claims = self._turn_claims
        if claims is None:
            _snapshot_interrupt()
            out["error"] = "SDK message reader is not ready to claim this turn"
            out["stream_ended"] = True
            return out
        claim_ack = asyncio.get_running_loop().create_future()
        self._turn_claim_requested = True
        claims.put_nowait(("claim", inbox, claim_ack))
        interrupted = False
        billing_guarded = False
        try:
            try:
                await claim_ack
            finally:
                self._turn_claim_requested = False
            ended = self._stream_ended
            if ended is not None:
                with self._interrupt_commit_lock:
                    interrupted = interrupted or self._interrupt_event.is_set()
                out["error"] = "SDK message stream ended before this turn" + (
                    f": {_safe_sdk_error_text(ended.error)}" if ended.error else ""
                )
                out["stream_ended"] = True
                out["interrupt_observed"] = interrupted
                return out
            query_input = (
                _sdk_user_message_stream(prompt)
                if isinstance(prompt, list)
                else prompt
            )
            await self._client.query(query_input)
            while True:
                message = await inbox.get()
                watch = self._turn_watch
                if watch is not None:
                    # Any stream message is liveness — stamp before anything
                    # else so the caller-thread watchdog sees it.
                    watch.tick()
                if isinstance(message, _StreamEnd):
                    with self._interrupt_commit_lock:
                        interrupted = interrupted or self._interrupt_event.is_set()
                    out["error"] = (
                        "SDK message stream ended before this turn's result"
                        + (f": {_safe_sdk_error_text(message.error)}" if message.error else "")
                    )
                    out["stream_ended"] = True
                    break
                # Capture the SDK session id from ANY message that carries it —
                # the init SystemMessage announces it first, so even a turn
                # interrupted before its ResultMessage keeps a resumable id.
                early_sid = getattr(message, "session_id", None)
                if early_sid:
                    self._session_id = early_sid
                self._handle_compact_boundary(message)
                if self._billing_guard_error is not None and not billing_guarded:
                    billing_guarded = True
                    out["error"] = self._billing_guard_error
                    out["billing_guard_violation"] = True
                    try:
                        await self._client.interrupt()
                    except Exception:
                        logger.debug(
                            "claude-agent-sdk billing-guard interrupt failed",
                            exc_info=True,
                        )
                if self._interrupt_event.is_set():
                    # Previously `break`. Bailing out before this turn's
                    # ResultMessage orphaned it in the shared stream, and the
                    # NEXT turn was then served that stale result — the same
                    # off-by-one corruption `_reader_loop` exists to prevent,
                    # reachable through the interrupt path with no Agent tool
                    # involved. Keep draining to the result; stop projecting.
                    interrupted = True
                if type(message).__name__ == "StreamEvent":
                    if watch is not None:
                        # A partial delta proves the post-tool model call is
                        # alive — the quiet watchdog stands down.
                        watch.disarm_post_tool()
                    if not interrupted and not billing_guarded:
                        self._forward_stream_delta(message)
                    continue
                self._note_mcp_tool_use(message, out)
                if not interrupted and not billing_guarded:
                    self._notify_tool_started(message)
                    self._notify_interim_assistant(message)
                projection, _result_is_error, _result_is_contradictory_success = (
                    self._project_message_step(projector, watch, message, out)
                )
                if projection.is_result:
                    # Terminal acceptance and interrupt observation are one
                    # atomic boundary.  If interrupt admission won the lock,
                    # REPORT it; if commit wins, later requests are queued and
                    # cannot call client.interrupt() during release.
                    #
                    # Reported, never folded into `interrupted`: that local
                    # gates delivery below, and the EDE mask further down whose
                    # contract is "never a fresh event read". A stop admitted
                    # during projection must not discard the answer this
                    # ResultMessage just completed, nor reclassify a genuine
                    # terminal error as an honored interrupt.
                    with self._interrupt_commit_lock:
                        boundary_interrupt = (
                            interrupted or self._interrupt_event.is_set()
                        )
                        self._terminal_result_committed = True
                    out["terminal_result_accepted"] = True
                if not interrupted and not billing_guarded:
                    if projection.messages:
                        out["messages"].extend(projection.messages)
                    if projection.is_tool_iteration:
                        out["tool_iterations"] += 1
                        self._notify_tool_iteration()
                    if projection.final_text is not None and (
                        not _result_is_error or _result_is_contradictory_success
                    ):
                        out["final_text"] = projection.final_text
                if projection.is_result:
                    usage = getattr(message, "usage", None)
                    if isinstance(usage, dict):
                        out["usage"] = dict(usage)
                    sid = getattr(message, "session_id", None)
                    if sid:
                        self._session_id = sid
                    out["result_uuid"] = getattr(message, "uuid", None)
                    out["total_cost_usd"] = getattr(
                        message, "total_cost_usd", None
                    )
                    subtype = getattr(message, "subtype", "") or ""
                    if getattr(message, "is_error", False):
                        errors = getattr(message, "errors", None) or []
                        api_error_status = getattr(message, "api_error_status", None)
                        if subtype == "success" and not errors and not api_error_status:
                            # Contradictory envelope: is_error=True yet
                            # subtype="success" with nothing in errors. The
                            # CLI emits this shape rarely (2026-08-11: it
                            # killed a cron run as "RuntimeError: SDK result
                            # error (subtype=success): success" — a dead job
                            # over a turn that had actually produced its
                            # answer). There is no error to report, so the
                            # error flag loses to the subtype; kept loud for
                            # diagnosis. A genuine failure carries a non-empty
                            # errors list or a non-success subtype and still
                            # takes the honest path below.
                            logger.warning(
                                "claude-agent-sdk: contradictory result "
                                "envelope (is_error=True, subtype=success, "
                                "no errors) — treated as success (diagnostic omitted)",
                            )
                            break
                        detail = _safe_sdk_error_text(
                            "; ".join(str(e) for e in errors)
                            or getattr(message, "result", None)
                            or subtype
                        )
                        err_text = f"SDK result error (subtype={subtype}): {detail}"
                        if api_error_status:
                            err_text += f" (HTTP {api_error_status})"
                        # A turn WE interrupted before any assistant content
                        # ends as is_error/error_during_execution in the CLI
                        # ("[ede_diagnostic] result_type=user…") — that is the
                        # interrupt being honored, not a failure; surfacing it
                        # paged the operator with a false "Processing stopped"
                        # (2026-08-09 barge-in incident). Mask ONLY that exact
                        # shape, only when THIS turn's interrupt flag is set
                        # (never a fresh event read — a late /stop must not
                        # reclassify a genuine error), and only when no auth
                        # needle is present (an auth failure must keep its
                        # error + retire path regardless of interrupts).
                        # A real transient failure colliding with an interrupt
                        # re-fires next turn with no interrupt pending and
                        # takes the honest path; the masked text is INFO-kept.
                        if (
                            interrupted
                            and subtype == "error_during_execution"
                            and classify_auth_failure(err_text) is None
                        ):
                            logger.info(
                                "claude-agent-sdk: masked %s on requested "
                                "interrupt (interrupt honored, not a "
                                "failure): %s",
                                subtype,
                                _safe_sdk_error_text(err_text),
                            )
                        else:
                            if not billing_guarded:
                                out["error"] = err_text
                    elif subtype not in ("", "success") and not billing_guarded:
                        # e.g. error_max_turns / error_max_budget_usd — surface
                        # honestly; the partial transcript is still projected.
                        out["error"] = f"SDK turn ended: {subtype}"
                    break
        finally:
            if self._turn_inbox is inbox:
                # Relinquish ownership through the same reader arbiter.  It
                # drains immediately buffered post-result residue into this
                # inbox before acknowledging release, preserving the existing
                # overlap/own-answer dedup handling below.
                if self._stream_ended is None and self._turn_claims is not None:
                    release_ack = asyncio.get_running_loop().create_future()
                    self._turn_claims.put_nowait(("release", inbox, release_ack))
                    await release_ack
                else:
                    self._turn_inbox = None
                # Stream death can win the release handshake after the
                # terminal ResultMessage was already delivered.  The reader's
                # death path acknowledges queued operations so waiters wake,
                # but it deliberately does not install/apply their ownership.
                # Re-check after the acknowledgement: preserve the valid turn
                # result while retiring the now-dead session, and never leave
                # a between-turn inbox falsely claimed.
                if self._stream_ended is not None:
                    if self._turn_inbox is inbox:
                        self._turn_inbox = None
                    out["stream_ended"] = True
            # Anything the reader parked after our ResultMessage belongs to a
            # CLI-initiated turn that overlapped ours. Route it now — left in
            # a discarded queue it would be lost, and left in the stream it
            # would become the next turn's answer. EXCEPT a residue result
            # that repeats THIS turn's own answer: delivering it through the
            # background lane re-presents the agent's own text as a fake
            # completion (the 2026-08-06 echo class) — suppress it, dedup-mark
            # it, and consume its burst buffer. Genuinely different residue is
            # a real background completion and still delivers.
            final_norm = " ".join(str(out["final_text"]).split())
            while not inbox.empty():
                residue = inbox.get_nowait()
                if final_norm and type(residue).__name__ == "ResultMessage":
                    res_text = getattr(residue, "result", None)
                    if (
                        isinstance(res_text, str)
                        and " ".join(res_text.split()) == final_norm
                    ):
                        uuid = getattr(residue, "uuid", None)
                        if uuid:
                            self._unsolicited_delivered.add(uuid)
                        self._unsolicited_results += 1
                        buffered = len(self._unsolicited_text)
                        self._unsolicited_text.clear()
                        logger.warning(
                            "claude-agent-sdk: residue ResultMessage %s "
                            "matches this turn's own answer — suppressed, "
                            "never delivered as a background result "
                            "(%d buffered text(s) consumed)",
                            uuid, buffered,
                        )
                        continue
                self._handle_unsolicited(residue)
        out["interrupt_observed"] = interrupted or boundary_interrupt
        out["billing_mode"] = self._reported_billing_mode()
        out["billing_evidence"] = dict(self._billing_evidence)
        return out

    # ---------- ungated per-message steps of _consume_turn ----------
    # Everything that depends on turn STATE (the interrupted / billing-guarded
    # gates on delivery, and every exit) stays in the loop scaffold above; the
    # two steps below run for every non-delta message regardless of that
    # state and only feed it.

    @staticmethod
    def _note_mcp_tool_use(message: Any, out: dict[str, Any]) -> None:
        """Record that this turn issued an ``mcp__`` tool (ungated)."""
        for block in getattr(message, "content", None) or []:
            if (
                type(block).__name__ == "ToolUseBlock"
                and str(getattr(block, "name", "")).startswith("mcp__")
            ):
                out["mcp_tool_seen"] = True

    @staticmethod
    def _project_message_step(
        projector: ClaudeSdkEventProjector,
        watch: Optional[_TurnWatch],
        message: Any,
        out: dict[str, Any],
    ) -> tuple[Any, bool, bool]:
        """Project one stream message and do the ungated bookkeeping around it.

        Runs whatever the turn state: the projector recognises the terminal
        result either way, the watchdog needs the outstanding-tool evidence, and
        the model id is captured even on interrupted turns. Returns the
        projection plus the two terminal-error flags that the gated delivery
        reads to withhold a diagnostic ``result`` string from the transcript.
        """
        projection = projector.project(message)
        if watch is not None:
            # Outstanding-tool evidence: ToolUseBlocks issue, tool
            # results resolve (server tools never enter — the
            # projector resolves them inside their own assistant
            # message). A turn with a tool in flight is suspended
            # from BOTH watchdog rules.
            issued = sum(
                len(m.get("tool_calls") or [])
                for m in projection.messages
                if m.get("role") == "assistant"
            )
            if issued:
                watch.note_tools_issued(issued)
            if projection.is_tool_iteration:
                watch.note_tools_resolved(
                    sum(
                        1
                        for m in projection.messages
                        if m.get("role") == "tool"
                    )
                )
                # Codex-parity arm point: a tool result just landed;
                # silence from here on is the wedge signature.
                watch.arm_post_tool()
            elif projection.messages or projection.final_text is not None:
                watch.disarm_post_tool()
        if projection.model:
            # Last reported id wins; captured even on interrupted
            # turns — the tokens were still spent on that model.
            out["model"] = projection.model
        # A genuine terminal SDK error carries a result string, but it
        # is transport diagnostics, not an assistant answer. The outer
        # runtime needs `out["error"]` to activate provider fallback;
        # emitting this text first persists it as an assistant message
        # and poisons the fallback's history with a false “I am
        # blocked” claim. Contradictory success envelopes remain a
        # success exactly as handled in the terminal branch.
        _result_subtype = getattr(message, "subtype", "") or ""
        _result_is_error = bool(
            projection.is_result
            and (
                getattr(message, "is_error", False)
                or _result_subtype not in ("", "success")
            )
        )
        _result_is_contradictory_success = bool(
            _result_is_error
            and _result_subtype == "success"
            and not (getattr(message, "errors", None) or [])
            and not getattr(message, "api_error_status", None)
        )
        return projection, _result_is_error, _result_is_contradictory_success

    # ---------- stream ownership ----------

    async def _reader_loop(self) -> None:
        """Sole owner of the SDK message stream for the client's lifetime.

        Hermes used to call `receive_response()` once per turn. That helper
        returns the OLDEST buffered ResultMessage — not necessarily this
        turn's — and the SDK offers no per-turn correlator (`query()` takes
        only a `session_id`, which is constant across turns; `ResultMessage`
        has a `uuid` with no request-side counterpart). Ordering is therefore
        the ONLY thing keeping answers matched to questions.

        The Claude Code CLI breaks that ordering by itself: when a background
        Agent task completes it injects a `<task-notification>` and runs a
        FULL assistant turn nobody asked for, leaving an unconsumed
        ResultMessage in the shared FIFO. Every later turn then pops a stale
        result. Because a desynced turn still looks like success — no error,
        no timeout, no interrupt — no retire path fires, so the skew is
        permanent and silent. Observed live 2026-07-25: four such turns left a
        constant off-by-4 for hours (dasbrow-hermes-coder#2).

        One reader for the whole client lifetime makes the rule enforceable:
        a message arriving while no turn is in flight is unsolicited BY
        DEFINITION, and gets routed away instead of poisoning the next turn."""
        end: _StreamEnd
        iterator = self._client.receive_messages().__aiter__()
        message_task = asyncio.ensure_future(iterator.__anext__())
        claim_task = asyncio.ensure_future(self._turn_claims.get())
        try:
            while True:
                done, _pending = await asyncio.wait(
                    (message_task, claim_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if message_task in done:
                    try:
                        message = message_task.result()
                    except StopAsyncIteration:
                        end = _StreamEnd(error=None)
                        break
                    self._observe_billing_evidence(message)
                    inbox = self._turn_inbox
                    if inbox is not None:
                        inbox.put_nowait(message)
                    else:
                        self._handle_unsolicited(message)
                    # Message wins ties with a claim. Re-arm first, then loop:
                    # every immediately available pre-claim FIFO entry is
                    # classified unsolicited before the claim is acknowledged.
                    message_task = asyncio.ensure_future(iterator.__anext__())
                    continue

                operation, inbox, claim_ack = claim_task.result()
                claim_task = asyncio.ensure_future(self._turn_claims.get())
                if claim_ack.cancelled():
                    continue
                if operation == "claim":
                    if self._turn_inbox is not None:
                        claim_ack.set_exception(
                            RuntimeError("SDK message stream already has a turn owner")
                        )
                        continue
                    self._turn_inbox = inbox
                elif operation == "release":
                    if self._turn_inbox is not inbox:
                        claim_ack.set_exception(
                            RuntimeError("SDK message stream release owner mismatch")
                        )
                        continue
                    self._turn_inbox = None
                else:  # pragma: no cover - internal invariant
                    claim_ack.set_exception(
                        RuntimeError(f"unknown SDK stream ownership operation: {operation}")
                    )
                    continue
                claim_ack.set_result(None)
        except asyncio.CancelledError:  # pragma: no cover - shutdown path
            message_task.cancel()
            claim_task.cancel()
            raise
        except Exception as exc:  # pragma: no cover - stream torn down
            logger.debug(
                "claude-agent-sdk reader loop ended: %s",
                _safe_sdk_error_text(exc),
            )
            end = _StreamEnd(error=_safe_sdk_error_text(exc))
        finally:
            if not message_task.done():
                message_task.cancel()
            if not claim_task.done():
                claim_task.cancel()
        # The stream is gone (CLI exited or transport died). Mark it before
        # acknowledging every queued ownership operation.  A foreground turn
        # can pass its initial stream check, enqueue a claim behind buffered
        # messages, and then lose the reader to EOF; resolving the claim makes
        # it re-check this terminal state instead of waiting out turn_timeout.
        self._stream_ended = end
        pending_claims = []
        if claim_task.done() and not claim_task.cancelled():
            try:
                pending_claims.append(claim_task.result())
            except Exception:  # pragma: no cover - queue task failed
                pass
        claims = self._turn_claims
        if claims is not None:
            while True:
                try:
                    pending_claims.append(claims.get_nowait())
                except asyncio.QueueEmpty:
                    break
        for _operation, _owner, claim_ack in pending_claims:
            if not claim_ack.done():
                claim_ack.set_result(None)
        inbox = self._turn_inbox
        if inbox is not None:
            inbox.put_nowait(end)

    def _handle_unsolicited(self, message: Any) -> None:
        """Route a message that arrived with no turn in flight.

        These are real CLI output (typically a finished background Agent task
        reporting in). They answer nothing Hermes asked, so they must never
        enter a turn's result — but their CONTENT is completed work the user
        is waiting on: with a delivery callback wired, capture each top-level
        assistant message's text and hand the FULL burst over as an ordered
        list on the terminal ResultMessage (uuid-deduped). Without a
        callback, the historical WARN-drop stands."""
        if isinstance(message, _StreamEnd):
            return
        sid = getattr(message, "session_id", None)
        if sid:
            self._session_id = sid
        name = type(message).__name__
        if name == "ResultMessage":
            self._unsolicited_results += 1
            if self._on_unsolicited_result is None:
                self._unsolicited_text.clear()
                logger.warning(
                    "claude-agent-sdk: dropped unsolicited ResultMessage (no "
                    "turn in flight, total=%d) — CLI-initiated turn; see "
                    "dasbrow-hermes-coder#2",
                    self._unsolicited_results,
                )
                return
            uuid = getattr(message, "uuid", None)
            if uuid and uuid in self._unsolicited_delivered:
                self._unsolicited_text.clear()
                logger.debug(
                    "claude-agent-sdk: duplicate unsolicited ResultMessage "
                    "%s ignored", uuid,
                )
                return
            result_text = getattr(message, "result", None)
            texts = list(self._unsolicited_text)
            self._unsolicited_text.clear()
            if isinstance(result_text, str) and result_text.strip():
                # The CLI's result text repeats the turn's final assistant
                # message — never hand the same text over twice.
                if not texts or texts[-1] != result_text:
                    texts.append(result_text)
            if uuid:
                self._unsolicited_delivered.add(uuid)
            if not texts:
                logger.warning(
                    "claude-agent-sdk: unsolicited ResultMessage carried no "
                    "text (total=%d) — nothing to deliver",
                    self._unsolicited_results,
                )
                return
            logger.info(
                "claude-agent-sdk: delivering unsolicited result burst "
                "(background task finished, total=%d, %d message(s), "
                "%d chars)",
                self._unsolicited_results, len(texts),
                sum(len(t) for t in texts),
            )
            try:
                self._on_unsolicited_result(texts)
            except Exception:
                logger.warning(
                    "claude-agent-sdk: unsolicited-result delivery callback "
                    "raised — answer may be lost", exc_info=True,
                )
        elif name == "AssistantMessage":
            # Buffer top-level text, one entry per message so the burst
            # delivers in message granularity; subagent streams
            # (parent_tool_use_id set) are noise — the same gate
            # _forward_stream_delta uses.
            if (
                self._on_unsolicited_result is not None
                and not getattr(message, "parent_tool_use_id", None)
            ):
                parts = [
                    getattr(block, "text", "") or ""
                    for block in getattr(message, "content", None) or []
                    if type(block).__name__ == "TextBlock"
                ]
                message_text = "\n".join(p for p in parts if p)
                if message_text:
                    self._unsolicited_text.append(message_text)
            logger.info(
                "claude-agent-sdk: unsolicited %s outside a turn", name,
            )
        else:
            logger.debug(
                "claude-agent-sdk: unsolicited %s outside a turn", name,
            )

    def _start_reader(self) -> None:
        """Spawn the reader on the session loop. Idempotent."""
        if self._reader_task is not None:
            return

        async def _spawn() -> Any:
            self._turn_claims = asyncio.Queue()
            return asyncio.ensure_future(self._reader_loop())

        self._reader_task = self._run_coro(_spawn(), timeout=10.0)

    def _stop_reader(self) -> None:
        task = self._reader_task
        self._reader_task = None
        if task is None or self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(task.cancel)
        except Exception:  # pragma: no cover - loop already gone
            pass
