"""run_inbound_hm — the _hm* methods split out of GatewayInboundMixin (mechanical extraction)."""

from __future__ import annotations
import logging
import asyncio
import dataclasses
import os
import time
from contextlib import suppress
from gateway.config import Platform
from gateway.platforms.base import EphemeralReply
from gateway.platforms.event import MessageType
from gateway.session import SessionSource
from typing import Any, Optional, Tuple
logger = logging.getLogger("gateway.run")

class GatewayInboundHmMixin:
    def _hm_pre_gateway_dispatch_hook(
        self, event: "MessageEvent", source: SessionSource
    ) -> Optional["MessageEvent"]:
        """Run the ``pre_gateway_dispatch`` plugin hook; None = drop, else the (maybe rewritten) event.
        Results: ``{"action": "skip"}`` → drop; ``{"action": "rewrite", "text"}`` → replace ``event.text``;
        ``allow``/None → normal dispatch. Runs BEFORE auth so plugins can handle unauthorized senders."""
        try:
            from hermes_cli.lifecycle import invoke_hook as _invoke_hook
            _hook_results = _invoke_hook(
                "pre_gateway_dispatch", event=event, gateway=self,
                # getattr: bare-runner tests build GatewayRunner via object.__new__ without __init__.
                session_store=getattr(self, "session_store", None),
            )
        except Exception as _hook_exc:
            logger.warning("pre_gateway_dispatch invocation failed: %s", _hook_exc)
            _hook_results = []

        for _result in _hook_results:
            if not isinstance(_result, dict):
                continue
            _action = _result.get("action")
            if _action == "skip":
                logger.info(
                    "pre_gateway_dispatch skip: reason=%s platform=%s chat=%s",
                    _result.get("reason"), source.platform.value if source.platform else "unknown",
                    source.chat_id or "unknown",
                )
                return None
            if _action == "rewrite":
                _new_text = _result.get("text")
                if isinstance(_new_text, str):
                    event = dataclasses.replace(event, text=_new_text)
                break
            if _action == "allow":
                break
        return event

    async def _hm_offer_pairing_code(self, source: SessionSource) -> None:
        """DM an unauthorized sender a pairing code (rate-limited; groups never reach here)."""
        platform_name = source.platform.value if source.platform else "unknown"
        pairing_store = self._pairing_store_for(source)
        if pairing_store is None:
            logger.error("Cannot offer pairing code on %s: no pairing store", platform_name)
            return
        # Rate-limit ALL pairing responses (code or rejection) so a burst of DMs doesn't spam.
        if pairing_store._is_rate_limited(platform_name, source.user_id):
            return
        code = pairing_store.generate_code(platform_name, source.user_id, source.user_name or "")
        adapter = self._adapter_for_source(source)
        if code:
            store_profile = getattr(pairing_store, "profile", None)
            profile_arg = (
                f"-p {store_profile} "
                if isinstance(store_profile, str) and store_profile and store_profile != "default"
                else ""
            )
            reply = (
                f"Hi~ I don't recognize you yet!\n\n"
                f"Here's your pairing code: `{code}`\n\n"
                f"Ask the bot owner to run:\n"
                f"`hermes {profile_arg}pairing approve "
                f"{platform_name} {code}`"
            )
        else:
            reply = "Too many pairing requests right now~ Please try again later!"
        if adapter:
            await adapter.send(source.chat_id, reply)
        if not code:
            # Record rate limit so subsequent messages are silently ignored
            pairing_store._record_rate_limit(platform_name, source.user_id)

    async def _hm_admit_event(
        self, event: "MessageEvent"
    ) -> Optional[Tuple["MessageEvent", SessionSource, bool]]:
        """Ingress gates for ``_handle_message``; None when dropped, else ``(event, source, is_internal)``
        (the ``pre_gateway_dispatch`` hook may have rewritten ``event``)."""
        from gateway.run import _is_slack_ignored_channel
        source = event.source
        # getattr(self, ...) throughout: bare test runners build GatewayRunner via object.__new__.
        _config = getattr(self, "config", None)

        # 🔴 Cross-session leak guard: this per-message task was create_task()'d with a copy of the
        # spawning context, which may carry ANOTHER message's HERMES_SESSION_* ContextVars; until
        # _set_session_env binds ours a subprocess would read the foreign identity. Reset to _UNSET.
        try:
            from gateway.session_context import reset_session_vars
            reset_session_vars()
        except Exception:
            logger.debug("reset_session_vars failed at handler entry", exc_info=True)

        # Most adapters resolve profile routes in build_source(); internal/voice paths construct
        # SessionSource directly, so resolve those here as the shared fail-closed ingress gate.
        # Strict boolean marker: require the literal True so duck-typed test/internal sources with
        # dynamic attributes are not mistaken for a rejection.
        if (
            getattr(_config, "multiplex_profiles", False)
            and not getattr(source, "profile", None)
            and getattr(source, "profile_route_rejected", False) is not True
        ):
            from gateway.profile_routing import ProfileRouteRejected

            try:
                source.profile = self._profile_name_for_source(source)
            except ProfileRouteRejected:
                source.profile_route_rejected = True
        if getattr(source, "profile_route_rejected", False) is True:
            logger.warning(
                "Dropping inbound message because its explicit profile route "
                "targets an unserved profile"
            )
            return None

        is_internal = bool(getattr(event, "internal", False))  # e.g. background-process notifications

        # Ignored-channel guard runs FIRST — before startup-restore queueing, plugin hooks, auth,
        # and session setup — so an ignored channel can never reach pairing/auth/session state.
        _chat_id = getattr(source, "chat_id", None)
        if (
            # See #51899.
            not is_internal
            and getattr(source, "platform", None) == Platform.SLACK
            and _is_slack_ignored_channel(_config, _chat_id)
        ):
            logger.info("Dropping Slack message from configured ignored channel %s", _chat_id)
            return None

        if (
            getattr(self, "_startup_restore_in_progress", False)
            and not is_internal
            and not getattr(event, "_hermes_startup_restore_replay", False)
        ):
            self._queue_startup_restore_event(event)
            return None

        if is_internal:
            return event, source, True

        # scale-to-zero: only real user-originated inbound stamps the last-inbound clock;
        # counting internal/system events would keep a genuinely idle gateway awake.
        self._scale_to_zero_note_real_inbound()
        event = self._hm_pre_gateway_dispatch_hook(event, source)
        if event is None:
            return None
        source = event.source

        if not self._is_user_authorized_for_source(source):
            if source.user_id is None:
                # No user identity (Telegram service messages, channel forwards, anonymous admin
                # posts, sender_chat): can't be paired but may be authorized via a chat allowlist.
                logger.debug("Ignoring message with no user_id from %s", source.platform.value)
                return None
            logger.warning("Unauthorized user: %s (%s) on %s", source.user_id, source.user_name, source.platform.value)
            # DMs get a pairing code, groups are ignored. A bot cannot pair, and answering one mid-cooldown is outbound traffic.
            if (
                source.chat_type == "dm"
                and not getattr(source, "is_bot", False)
                and self._get_unauthorized_dm_behavior(source.platform, profile=source.profile) == "pair"
            ):
                await self._hm_offer_pairing_code(source)
            return None
        # The busy path charged this event on arrival; a drained follow-up must not pay twice.
        if not getattr(event, "_bot_loop_admitted", False) and not self._admit_bot_message_for_source(source):
            return None
        return event, source, False

    def _hm_estop_turn_allowed(self, event: "MessageEvent", source: SessionSource) -> bool:
        """Whether a turn may bypass the global emergency stop: pause blocks NEW agent turns, never
        running work or control traffic — recognized slash commands (incl. /pause off, the in-band
        resume) and replies owned by in-flight work (pending update prompt, running session,
        pending slash-confirm, dangerous-command approval) all pass through."""
        with suppress(Exception):
            _estop_cmd = event.get_command()
            if _estop_cmd:
                from hermes_cli.commands import resolve_command as _resolve_estop_cmd
                if _resolve_estop_cmd(_estop_cmd) is not None:
                    return True
        with suppress(Exception):
            _estop_key = self._session_key_for_source(source)
            _estop_state = self._peek_session_state(_estop_key)
            if _estop_state is not None and _estop_state.persistent.update_prompt_pending:
                return True
            # A running session covers steering plus pending clarify / tool approvals it holds.
            if self._is_session_running(_estop_key):
                return True
            from tools import slash_confirm as _estop_confirm_mod
            if _estop_confirm_mod.get_pending(_estop_key):
                return True
            from tools.approval import has_blocking_approval as _estop_has_approval
            if _estop_has_approval(_estop_key):
                return True
        return False

    def _hm_estop_gate(
        self, event: "MessageEvent", source: SessionSource, is_internal: bool
    ) -> Optional[str]:
        """Global emergency-stop (`hermes pause`) notice when this turn must be blocked, else None.
        Placed after auth so unauthorized senders can't probe pause state."""
        if is_internal:
            return None
        try:
            from agent.estop import paused_reply as _estop_paused_reply
        except ImportError:
            return None
        _paused_notice = _estop_paused_reply()
        if _paused_notice is None or self._hm_estop_turn_allowed(event, source):
            return None
        logger.info(
            "Gateway turn paused by global emergency stop (platform=%s chat=%s)",
            getattr(getattr(source, "platform", None), "value", "unknown"),
            getattr(source, "chat_id", None) or "unknown",
        )
        return _paused_notice

    @staticmethod
    def _hm_write_update_response(response_text: str) -> Optional[str]:
        """Atomically hand *response_text* to the detached update process; returns the OSError str."""
        from gateway.run import _hermes_home
        response_path = _hermes_home / ".update_response"
        try:
            tmp = response_path.with_suffix(".tmp")
            tmp.write_text(response_text, encoding="utf-8")
            tmp.replace(response_path)
            (_hermes_home / ".update_prompt.json").unlink(missing_ok=True)
        except OSError as e:
            return str(e)
        return None

    def _hm_update_prompt_reply(self, event: "MessageEvent", _quick_key: str) -> Optional[str]:
        """Consume a reply to a pending ``/update`` prompt (routed to the detached update process via
        ``.update_response``); None when nothing was consumed. Recognized slash commands must bypass
        this or /new, /help etc. get silently consumed as update answers."""
        _up_state = self._peek_session_state(_quick_key)
        if _up_state is None or not _up_state.persistent.update_prompt_pending:
            return None
        # Accept /approve and /deny as shorthand for yes/no
        cmd = event.get_command()
        _recognized_cmd = None
        if cmd in {"approve", "yes"}:
            response_text = "y"
        elif cmd in {"deny", "no"}:
            response_text = "n"
        else:
            if cmd:
                with suppress(Exception):
                    from hermes_cli.commands import resolve_command as _resolve_update_cmd
                    _cmd_def = _resolve_update_cmd(cmd)
                    _recognized_cmd = _cmd_def.name if _cmd_def else None
            response_text = "" if _recognized_cmd else (event.text or "").strip()
        if response_text:
            err = self._hm_write_update_response(response_text)
            if err is not None:
                logger.warning("Failed to write update response: %s", err)
                return f"✗ Failed to send response to update process: {err}"
            _up_state.persistent.update_prompt_pending = False
            label = response_text if len(response_text) <= 20 else response_text[:20] + "…"
            return f"✓ Sent `{label}` to the update process."
        # Recognized slash command during a pending update prompt: write a blank response so the
        # detached update's ``_gateway_prompt`` returns the prompt's default (typically a safe
        # "n" / skip) and exits instead of blocking on stdin until the watcher timeout.
        if _recognized_cmd:
            err = self._hm_write_update_response("")
            if err is None:
                logger.info(
                    "Recognized /%s during pending update prompt for %s; "
                    "cancelled prompt with default and dispatching command",
                    _recognized_cmd, _quick_key,
                )
            else:
                logger.warning("Failed to write cancel response for pending update prompt: %s", err)
            _up_state.persistent.update_prompt_pending = False
        return None

    async def _hm_clarify_reply(
        self, event: "MessageEvent", source: SessionSource, _quick_key: str
    ) -> Optional[str]:
        """Intercept a reply to a pending clarify prompt; None when the message falls through.
        Free text answers open-ended/"Other" prompts; "2" answers a multi-choice one. Resolved/retained
        replies return "" so adapters don't double-post — the agent produces the next user-facing message."""
        try:
            from tools import clarify_gateway as _clarify_mod
            _pending_clarify = _clarify_mod.get_pending_for_session(_quick_key, include_choice_prompts=True)
        except Exception:
            return None
        if _pending_clarify is None:
            return None
        _clarify_has_audio = bool(self._pending_event_audio_paths(event))
        _raw_clarify_reply = await self._prepare_clarify_reply_text(event)

        def _retain(why: str) -> str:
            logger.info(
                "Gateway retained pending clarify after %s (session=%s, id=%s)",
                why, _quick_key, _pending_clarify.clarify_id,
            )
            return ""

        if _clarify_has_audio and not _raw_clarify_reply:
            return _retain("voice transcription produced no usable text")
        # Slash commands: the user wanted a command, not to answer the clarify. Leave it pending so
        # they can retry; on timeout the agent unblocks with an empty response.
        if not _raw_clarify_reply or _raw_clarify_reply.startswith("/"):
            return None
        _text_outcome = _clarify_mod.attempt_text_response_for_session(_quick_key, _raw_clarify_reply)
        if _text_outcome == _clarify_mod.TEXT_RESOLVED:
            logger.info(
                "Gateway intercepted clarify text response (session=%s, id=%s)",
                _quick_key, _pending_clarify.clarify_id,
            )
            # The clarify callback pauses the platform typing/status indicator while waiting so
            # Slack users can type; the active agent resumes now, so re-enable its indicator.
            _clarify_adapter = self._adapter_for_source(source)
            if _clarify_adapter:
                try:
                    _clarify_adapter.resume_typing_for_chat(source.chat_id)
                except Exception:
                    logger.debug("Failed to resume typing after clarify response", exc_info=True)
            return ""
        if _text_outcome == _clarify_mod.TEXT_REJECTED_SELECTION:
            # Selection-shaped but invalid (out-of-range number, bad comma-list): keep the clarify
            # armed for retry — don't cancel, don't treat as an unrelated follow-up.
            return _retain("invalid selection attempt")
        if _text_outcome == _clarify_mod.TEXT_REJECTED_PROSE:
            # Native-choice prompts reject unmatched prose so it continues through normal busy
            # routing. Release this clarify first: redirect() degrades to steer() while tools
            # execute, and that steer cannot drain until the clarify tool returns.
            _clarify_mod.resolve_gateway_clarify(_pending_clarify.clarify_id, "")
        return None

    async def _hm_slash_confirm_reply(self, event: "MessageEvent", _quick_key: str) -> Optional[str]:
        """Resolve a reply (/approve, /always, /cancel + aliases) to a pending slash-confirm prompt;
        None when it falls through — a stale pending confirm does NOT block other commands. A pending
        dangerous-command approval takes precedence: /approve there unblocks the waiting tool thread."""
        from tools import slash_confirm as _slash_confirm_mod
        _pending_confirm = _slash_confirm_mod.get_pending(_quick_key)
        if not _pending_confirm:
            return None
        with suppress(Exception):
            from tools.approval import has_blocking_approval
            if has_blocking_approval(_quick_key):
                return None
        # Accept bang-prefixed replies (`!always`, `!cancel`) verbatim: Slack/Matrix show the `!`
        # prefix (typed `/` is blocked in Slack threads) and adapters only rewrite
        # `!<known-command>` — confirm keywords aren't commands, so the `!` survives to here.
        _norm_reply = (event.text or "").strip().lstrip("!/").lower()
        _confirm_choice = (
            self._SLASH_CONFIRM_CMD_CHOICES.get(event.get_command())
            or self._SLASH_CONFIRM_TEXT_CHOICES.get(_norm_reply)
        )
        if _confirm_choice is not None:
            _resolved = await _slash_confirm_mod.resolve(
                _quick_key, _pending_confirm.get("confirm_id"), _confirm_choice,
            )
            return _resolved or ""
        # Stale pending + unrelated command: the user moved on, so drop the pending state rather
        # than let the confirm block normal usage indefinitely.
        _slash_confirm_mod.clear_if_stale(_quick_key)
        return None

    def _hm_evict_idle_stale_agent(self, _quick_key: str) -> None:
        """Evict a leaked lock from a hung/crashed handler: only when the agent has been *idle* past
        the threshold (active tasks can run for hours), or has no activity tracker and an extreme
        wall-clock age. The pending sentinel is never evicted (no get_activity_summary() → idle
        reads inf and would race the async setup path)."""
        from gateway.run import _AGENT_PENDING_SENTINEL, _float_env
        _raw_stale_timeout = _float_env("HERMES_AGENT_TIMEOUT", 1800)
        _quick_state = self._peek_session_state(_quick_key)
        _stale_ts = _quick_state.turn.started_ts if _quick_state else 0
        if _quick_state is None or _quick_state.turn.agent is None or not _stale_ts:
            return
        _stale_age = time.time() - _stale_ts
        _stale_agent = _quick_state.turn.agent
        _stale_idle = float("inf")  # assume idle if we can't check
        _stale_detail = ""
        _activity_summary_valid = False
        if _stale_agent and hasattr(_stale_agent, "get_activity_summary"):
            with suppress(Exception):
                _sa = _stale_agent.get_activity_summary()
                from gateway.session_stall import resolve_session_idle_seconds_from_activity

                _sa_d = _sa if isinstance(_sa, dict) else {}
                _resolved_idle = resolve_session_idle_seconds_from_activity(
                    _sa if isinstance(_sa, dict) else None, now=time.time(),
                )
                if _resolved_idle is not None:
                    _stale_idle = _resolved_idle
                    _activity_summary_valid = True
                _stale_detail = (
                    f" | last_activity={_sa_d.get('last_activity_desc', 'unknown')} "
                    f"({_stale_idle:.0f}s ago) "
                    f"| iteration={_sa_d.get('api_call_count', 0)}/{_sa_d.get('max_iterations', 0)}"
                )
        # A valid activity clock is authoritative: total age alone never makes an actively
        # progressing turn stale. The emergency wall TTL is only a fallback when the agent cannot
        # report usable activity.
        _wall_ttl = max(_raw_stale_timeout * 10, 7200) if _raw_stale_timeout > 0 else float("inf")
        _should_evict = _stale_agent is not _AGENT_PENDING_SENTINEL and (
            (_activity_summary_valid and _raw_stale_timeout > 0 and _stale_idle >= _raw_stale_timeout)
            or (not _activity_summary_valid and _stale_age > _wall_ttl)
        )
        if _should_evict:
            logger.warning(
                "Evicting stale _running_agents entry for %s "
                "(age: %.0fs, idle: %.0fs, timeout: %.0fs)%s",
                _quick_key, _stale_age, _stale_idle, _raw_stale_timeout, _stale_detail,
            )
            self._hm_evict_running_agent(_quick_key, "stale_running_agent_eviction")

    def _hm_evict_reaped_agent(self, _quick_key: str) -> None:
        """Evict the in-memory turn slot of a session whose durable row was ended while the gateway
        lived (``ws_orphan_reap`` / ``agent_close``): otherwise the fast-path queues every next
        message into the dead runtime. The cold path re-attaches via ``get_or_create_session``."""
        try:
            # #99106: durable-reaped guard. This is the live-gateway variant of #54878 and the #632
            # detached/ 405 suppressions in production. Evict the stale slot so the next message falls
            # through to the cold path and re-attaches or creates a fresh session; /status then correctly
            # shows 代理运行中: 否 before the heal and a live turn after.
            _reap_store = getattr(self, "session_store", None)
            # Public, lock-held accessors: peek_session_id returns a non-str on stubbed stores in
            # bare test runners — the isinstance() / ``is True`` gates keep this inert unless a
            # real SessionStore answers.
            _reap_peek = getattr(_reap_store, "peek_session_id", None)
            _is_ended = getattr(_reap_store, "_is_session_ended_in_db", None)
            _reap_sid = _reap_peek(_quick_key) if callable(_reap_peek) else None
            if isinstance(_reap_sid, str) and _reap_sid and callable(_is_ended) and _is_ended(_reap_sid) is True:
                logger.warning(
                    "Evicting stale _running_agents entry for %s — "
                    "durable session %s is ended (reaped) in state.db; "
                    "healing routing on next message (#99106)", _quick_key, _reap_sid,
                )
                self._hm_evict_running_agent(_quick_key, "reaped_session_eviction")
        except Exception:
            logger.debug("reaped-session staleness check failed", exc_info=True)

    def _hm_evict_running_agent(self, _quick_key: str, reason: str) -> None:
        self._invalidate_session_run_generation(_quick_key, reason=reason)
        self._release_running_agent_state(_quick_key)

    def _hm_merge_pending_for_source(
        self, source: SessionSource, _quick_key: str, event: "MessageEvent", *, merge_text: bool = False
    ) -> None:
        """Merge *event* into the source adapter's pending slot (no-op without an adapter)."""
        from gateway.platforms.base import merge_pending_message_event
        adapter = self._adapter_for_source(source)
        if adapter:
            merge_pending_message_event(adapter._pending_messages, _quick_key, event, merge_text=merge_text)

    async def _hm_busy_slash_or_photo(
        self, event: "MessageEvent", source: SessionSource, _quick_key: str
    ) -> Tuple[bool, Optional[str]]:
        """Slash-command / photo-burst handling on the busy fast-path → ``(handled, result)``. Each
        command's mid-run behavior is declared on its CommandDef (busy_policy / busy_handler)."""
        from hermes_cli.commands import resolve_command as _resolve_cmd_inner
        _evt_cmd = event.get_command()
        _cmd_def_inner = _resolve_cmd_inner(_evt_cmd) if _evt_cmd else None

        if _cmd_def_inner:
            # /status and /context are intentionally pre-gate so users always see session state.
            if _cmd_def_inner.name == "status":
                return True, await self._handle_status_command(event)
            if _cmd_def_inner.name == "context":
                return True, await self._handle_context_command(event)
            # Slash access control mirrors the cold-path gate so non-admins can't bypass gating
            # just because an agent is busy. /help and /whoami are the always-allowed floor.
            _denied = self._check_slash_access(source, _cmd_def_inner.name)
            if _denied is not None:
                return True, _denied
            # Any recognized slash command dispatches per its declared busy_policy (dispatch /
            # interrupt_then_dispatch / reject). Unrecognized commands and plain text fall through.
            return True, await self._dispatch_busy_slash_command(event, _cmd_def_inner, _quick_key, source)

        # Telegram photo bursts arrive as near-simultaneous updates — never interrupt for a
        # photo-only follow-up; adapter-level batching absorbs them.
        if event.message_type == MessageType.PHOTO:
            logger.debug("PRIORITY photo follow-up for session %s — queueing without interrupt", _quick_key)
            self._hm_merge_pending_for_source(source, _quick_key, event)
            return True, None
        return False, None

    def _hm_busy_telegram_grace_queue(
        self, event: "MessageEvent", source: SessionSource, _quick_key: str, effective_busy_input_mode: str
    ) -> bool:
        """Queue a Telegram text follow-up that lands within the post-start grace window."""
        _grace = float(os.getenv("HERMES_TELEGRAM_FOLLOWUP_GRACE_SECONDS", "3.0"))
        _grace_state = self._peek_session_state(_quick_key)
        _started_at = _grace_state.turn.started_ts if _grace_state else 0
        if not (
            source.platform == Platform.TELEGRAM and event.message_type == MessageType.TEXT
            and _grace > 0 and _started_at and (time.time() - _started_at) <= _grace
        ):
            return False
        logger.debug(
            "Telegram follow-up arrived %.2fs after run start for %s — queueing without interrupt",
            time.time() - _started_at, _quick_key,
        )
        if effective_busy_input_mode != "queue":
            self._hm_merge_pending_for_source(source, _quick_key, event, merge_text=True)
        else:
            adapter = self._adapter_for_source(source)
            if adapter:
                self._enqueue_fifo(_quick_key, event, adapter)
        return True

    @staticmethod
    def _hm_text_only(event: "MessageEvent") -> bool:
        return event.message_type == MessageType.TEXT and not event.media_urls and not event.media_types

    def _hm_busy_steer(self, event: "MessageEvent", running_agent: Any, _quick_key: str) -> None:
        """Steer mode: inject text mid-run via ``agent.steer()``, else fall back to queue semantics."""
        steer_text = (event.text or "").strip()
        steered = False
        if self._hm_text_only(event) and steer_text and hasattr(running_agent, "steer"):
            try:
                steered = bool(running_agent.steer(self._steer_text_with_origin(steer_text, event)))
            except Exception as exc:
                logger.warning("PRIORITY steer failed for session %s: %s", _quick_key, exc)
        if steered:
            logger.debug("PRIORITY steer for session %s", _quick_key)
            return
        logger.debug("PRIORITY steer-fallback-to-queue for session %s", _quick_key)
        self._queue_or_replace_pending_event(_quick_key, event)

    async def _hm_busy_interrupt(
        self, event: "MessageEvent", source: SessionSource, running_agent: Any, _quick_key: str
    ) -> None:
        """Interrupt path: redirect text-only corrections when supported, else ``agent.interrupt()``."""
        from gateway.run import _build_media_placeholder
        # Text-only corrections redirect the live turn (preserving displayed context) when the
        # runtime supports it; media/voice and older runtimes use the interrupt path below.
        _can_redirect = getattr(running_agent, "_supports_active_turn_redirect", False) is True
        if self._hm_text_only(event) and _can_redirect and hasattr(running_agent, "redirect"):
            try:
                if running_agent.redirect(
                    self._steer_text_with_origin((event.text or "").strip(), event)
                ):
                    logger.debug("PRIORITY redirect for session %s", _quick_key)
                    return
            except Exception as exc:
                logger.warning("PRIORITY redirect failed for session %s: %s", _quick_key, exc)
        logger.debug("PRIORITY interrupt for session %s", _quick_key)
        _interrupt_text = event.text
        if self._pending_event_audio_paths(event):
            _interrupt_text, _ = await self._transcribe_and_echo_pending_voice(
                event, self._adapter_for_source(source), source, event.text or "",
                log_context="Voice-priority-interrupt",
            )
        elif not _interrupt_text and getattr(event, "media_urls", None):
            _interrupt_text = _build_media_placeholder(event)
        # Delivered via adapter._pending_messages (read by _run_agent); never also buffered on self
        # — that copy was never consumed and grew unbounded.
        running_agent.interrupt(_interrupt_text)

    async def _hm_handle_running_session_message(
        self, event: "MessageEvent", source: SessionSource, _quick_key: str
    ) -> Optional[str]:
        """Fast-path while this session's agent is running: interrupt by default (minimal latency);
        busy_input_mode queue/steer, subagent and compression protection demote to queue."""
        from gateway.run import _AGENT_PENDING_SENTINEL
        _handled, _result = await self._hm_busy_slash_or_photo(event, source, _quick_key)
        if _handled:
            return _result

        effective_busy_input_mode = self._effective_busy_input_mode(source)
        if self._hm_busy_telegram_grace_queue(event, source, _quick_key, effective_busy_input_mode):
            return None

        _ra_state = self._peek_session_state(_quick_key)
        running_agent = _ra_state.turn.agent if _ra_state else None
        if running_agent is _AGENT_PENDING_SENTINEL:  # agent still being set up
            if event.get_command() == "stop":  # force-clean the sentinel so the session is unlocked
                self._release_running_agent_state(_quick_key)
                logger.info("HARD STOP (pending) for session %s — sentinel cleared", _quick_key)
                return EphemeralReply("⚡ Force-stopped. The agent was still starting — session unlocked.")
            self._hm_merge_pending_for_source(source, _quick_key, event, merge_text=True)  # picked up after start
            return None
        if self._draining:
            queue_during_drain = self._queue_during_drain_enabled(effective_busy_input_mode)
            if queue_during_drain:
                self._queue_or_replace_pending_event(_quick_key, event)
            return (
                f"⏳ Gateway {self._status_action_gerund()} — queued for the next turn after it comes back."
                if queue_during_drain
                else f"⏳ Gateway is {self._status_action_gerund()} and is not accepting another turn right now."
            )
        if effective_busy_input_mode == "queue":
            logger.debug("PRIORITY queue follow-up for session %s", _quick_key)
            self._queue_or_replace_pending_event(_quick_key, event)
            return None
        if effective_busy_input_mode == "steer":
            self._hm_busy_steer(event, running_agent, _quick_key)
            return None
        # Subagent protection: an interrupt cascades through ``_active_children`` and aborts
        # in-flight delegate_task work (/stop reached its handler above — still an escape hatch).
        # Compression protection: an interrupt would start a new turn on the pre-rotation parent
        # while compression rotates the id away, forking orphaned siblings.
        if self._agent_has_active_subagents(running_agent):
            _demote = "because the running agent has active subagents (#30170)"
        elif await self._session_has_compression_in_flight(_quick_key):
            _demote = "because context compression is in flight (#56391)"
        else:
            await self._hm_busy_interrupt(event, source, running_agent, _quick_key)
            return None
        logger.info("PRIORITY interrupt demoted to queue for session %s %s", _quick_key, _demote)
        self._queue_or_replace_pending_event(_quick_key, event)
        return None

    def _hm_quick_commands(self) -> dict:
        """User-defined ``quick_commands`` mapping from config (empty dict when unset/malformed)."""
        cfg = self.config
        qc = (cfg.get("quick_commands") if isinstance(cfg, dict) else getattr(cfg, "quick_commands", None)) or {}
        return qc if isinstance(qc, dict) else {}

    @staticmethod
    def _hm_expand_alias_quick_command(event: "MessageEvent", qcmd: dict) -> Optional[str]:
        """Rewrite ``event.text`` to an alias quick command's target; returns the new command name."""
        target = (qcmd.get("target") or "").strip()
        if not target:
            return None
        target = target if target.startswith("/") else f"/{target}"
        event.text = f"{target} {event.get_command_args().strip()}".strip()
        target_command = target.lstrip("/")
        return target_command.split()[0] if target_command else target_command

    async def _hm_command_hooks(
        self, event: "MessageEvent", source: SessionSource, _quick_key: str, command: str, canonical: str
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Fire ``pre_command`` (observer) and ``command:<canonical>`` (interceptor) hooks →
        ``(handled, result, new_command)`` (``new_command`` set when a handler rewrote the command).
        The running-agent path deliberately does NOT fire these — a slow or hostile plugin must not
        interfere with the operator's escape hatches for a live agent."""
        raw_args = event.get_command_args().strip()
        platform = source.platform.value if source.platform else ""
        try:
            from hermes_cli.plugins import fire_pre_command_hook
            fire_pre_command_hook(
                surface="gateway", command=str(canonical), alias_used=str(command),
                args_raw=raw_args, session_key=_quick_key, platform=platform,
            )
        except Exception as _pre_cmd_err:
            logger.debug("pre_command hook dispatch failed (non-fatal): %s", _pre_cmd_err)

        # Handlers may return ``{"decision": "deny" | "handled" | "rewrite", ...}`` to intercept
        # dispatch; handlers returning nothing behave as plain observers.
        hook_ctx = {
            "platform": platform, "user_id": source.user_id, "command": canonical,
            "raw_command": command, "args": raw_args, "raw_args": raw_args,
        }
        try:
            hook_results = await self.hooks.emit_collect(f"command:{canonical}", hook_ctx)
        except Exception as _hook_err:
            logger.debug("command:%s hook dispatch failed (non-fatal): %s", canonical, _hook_err)
            hook_results = []

        for hook_result in hook_results:
            if not isinstance(hook_result, dict):
                continue
            decision = str(hook_result.get("decision", "")).strip().lower()
            message = hook_result.get("message")
            message = message if isinstance(message, str) and message else None
            if decision == "deny":
                return True, message or f"Command `/{command}` was blocked by a hook.", None
            if decision == "handled":
                return True, message, None
            if decision == "rewrite":
                new_command = str(hook_result.get("command_name", "")).strip().lstrip("/")
                if new_command:
                    event.text = f"/{new_command} {str(hook_result.get('raw_args', '')).strip()}".strip()
                    return False, None, event.get_command()
        return False, None, None

    async def _hm_resolve_command(
        self, event: "MessageEvent", source: SessionSource, _quick_key: str
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[str]]:
        """Resolve the slash command (aliases, access gate, hooks) → ``(handled, result, command,
        canonical)``; when ``handled`` the caller returns ``result`` as-is (may be None)."""
        from hermes_cli.commands import is_gateway_known_command, resolve_command as _resolve_cmd

        def _canon(cmd):
            # Aliases resolve to the canonical name so dispatch and hook names don't depend on them.
            _def = _resolve_cmd(cmd) if cmd else None
            return _def, (_def.name if _def else cmd)

        command = event.get_command()
        _cmd_def, canonical = _canon(command)

        # Expand alias quick commands before built-in dispatch so targets like /model openai/gpt-5.5
        # --provider openrouter reach the /model handler. Built-ins keep precedence: aliases only
        # need early handling when the typed command is not already known.
        if command and _cmd_def is None:
            qcmd = self._hm_quick_commands().get(command)
            if qcmd is not None and qcmd.get("type") == "alias":
                new_command = self._hm_expand_alias_quick_command(event, qcmd)
                if new_command is not None:
                    command = new_command
                    _cmd_def, canonical = _canon(command)

        if not (command and canonical and is_gateway_known_command(canonical)):
            return False, None, command, canonical

        # Per-platform slash access control: only active when the operator set ``allow_admin_from``
        # for the source's scope; then non-admins get ``user_allowed_commands`` plus the
        # /help, /whoami floor. Plain chat is never gated.
        _denied = self._check_slash_access(source, canonical)
        if _denied is not None:
            return True, _denied, command, canonical

        _handled, _result, new_command = await self._hm_command_hooks(
            event, source, _quick_key, command, canonical
        )
        if _handled:
            return True, _result, command, canonical
        if new_command is not None:
            command = new_command
            _cmd_def, canonical = _canon(command)
        return False, None, command, canonical

    async def _hm_confirm_destructive(self, event, command: str, detail: str, handler) -> Tuple[bool, Optional[str]]:
        async def _execute():
            return await handler(event)
        return True, await self._maybe_confirm_destructive_slash(
            event=event, command=command, title=f"/{command}", detail=detail, execute=_execute,
        )

    async def _hm_cmd_new(self, event, source, _quick_key):
        if await asyncio.to_thread(self._is_telegram_topic_root_lobby, source):
            return True, self._telegram_topic_root_new_message()
        return await self._hm_confirm_destructive(
            event, "new", "This starts a fresh session and discards the current conversation history.",
            self._handle_reset_command,
        )

    async def _hm_cmd_start(self, event, source, _quick_key):
        if event.get_command_args().strip().startswith("wisdom_"):
            return True, await self._continue_wisdom_start(event, source)
        logger.info("Ignoring /start platform ping for session %s", _quick_key)
        return True, ""

    async def _hm_cmd_egress(self, event, source, _quick_key):
        from hermes_cli.proxy_cli import format_status_text
        return True, format_status_text()

    async def _hm_rewrite_turn_to_prompt(self, event, source, name: str, ack: str, build) -> Tuple[bool, Optional[str]]:
        """Ack, then rewrite the turn to ``build()`` and fall through to the agent (keeps role
        alternation; works on any backend). A failing builder replies with a retry hint."""
        await self._send_command_ack(source, ack, name)
        try:
            event.text = build()
        except Exception:
            return True, f"Could not start /{name} — please try again."
        return False, None

    # /learn and /plan: ack, rewrite the turn to a builder prompt, fall through to the agent.
    async def _hm_cmd_learn(self, event, source, _quick_key):
        from agent.learn_prompt import build_learn_prompt

        req = event.get_command_args().strip()
        _ack = f"Learning a skill from {'what you described' if req else 'this conversation'}…"
        return await self._hm_rewrite_turn_to_prompt(event, source, "learn", _ack, lambda: build_learn_prompt(req))

    async def _hm_cmd_plan(self, event, source, _quick_key):
        from agent.plan_prompt import build_plan_prompt

        task = event.get_command_args().strip()
        _ack = f"Planning: {task[:80]}{'…' if len(task) > 80 else ''}" if task else "Planning from this conversation's context…"
        return await self._hm_rewrite_turn_to_prompt(event, source, "plan", _ack, lambda: build_plan_prompt(task))

    async def _hm_cmd_init(self, event, source, _quick_key):
        # /init builds the prompt first: the ack wording depends on whether AGENTS.md exists.
        from hermes_cli.init_command import build_init_prompt_for_cwd

        try:
            _init_prompt = build_init_prompt_for_cwd(extra=event.get_command_args().strip())
        except Exception:
            return True, "Could not start /init — please try again."
        _ack = (
            "Updating AGENTS.md from a project scan…"
            if "UPDATE the existing AGENTS.md" in _init_prompt
            else "Generating AGENTS.md from a project scan…"
        )
        await self._send_command_ack(source, _ack, "init")
        event.text = _init_prompt
        return False, None

    async def _hm_cmd_blueprint(self, event, source, _quick_key):
        _blueprint_result = await self._handle_blueprint_command(event)
        _text = getattr(_blueprint_result, "text", "") or ""
        _blueprint_seed = getattr(_blueprint_result, "agent_seed", None)
        if not _blueprint_seed:
            return True, _text or None
        # Blueprint matched — rewrite the turn to the seed and fall through so the agent collects
        # each slot value conversationally, then calls the cronjob tool (the /steer pattern).
        if _text:
            await self._send_command_ack(source, _text, "blueprint")
        try:
            event.text = _blueprint_seed
        except Exception:
            return True, _text or None
        return False, None

    async def _hm_cmd_undo(self, event, source, _quick_key):
        _undo_n = 1
        _undo_raw = event.get_command_args().strip()
        if _undo_raw:
            with suppress(ValueError, IndexError):
                _undo_n = max(1, int(_undo_raw.split()[0]))
        _undo_detail = (
            "This removes the last user/assistant exchange from history."
            if _undo_n == 1
            else f"This removes the last {_undo_n} user turns from history."
        )
        return await self._hm_confirm_destructive(event, "undo", _undo_detail, self._handle_undo_command)

    # /queue and /steer on the idle path: no agent is running, so strip the prefix and send the
    # payload as a regular user turn; an empty payload surfaces the usage hint.
    async def _hm_cmd_queue(self, event, source, _quick_key):
        return self._hm_send_payload_as_turn(event, "Usage: /queue <prompt>")

    async def _hm_cmd_steer(self, event, source, _quick_key):
        return self._hm_send_payload_as_turn(
            event, "Usage: /steer <prompt>  (no agent is running; sending as a normal message)"
        )

    @staticmethod
    def _hm_send_payload_as_turn(event, usage: str) -> Tuple[bool, Optional[str]]:
        payload = event.get_command_args().strip()
        if not payload:
            return True, usage
        with suppress(Exception):
            event.text = payload
        return False, None

    async def _hm_cmd_moa(self, event, source, _quick_key):
        # /moa is one-shot sugar only: run a single prompt through the default MoA preset, then
        # restore the prior model. To *switch* to a MoA preset for the session, pick it from the
        # model picker (MoA presets surface as a virtual "Mixture of Agents" provider).
        from hermes_cli.moa_config import moa_usage, normalize_moa_config
        from hermes_cli.config import load_config

        moa_payload = event.get_command_args().strip()
        if not moa_payload:
            return True, moa_usage()
        try:
            cfg = load_config()
            moa_cfg = normalize_moa_config(cfg.get("moa") if isinstance(cfg, dict) else {})
        except Exception:
            moa_cfg = normalize_moa_config({})
        try:
            event.text = moa_payload
            _moa_state = self._session_state(_quick_key)
            event._moa_restore_override = _moa_state.conversation.model_override
            _moa_state.conversation.model_override = {
                "provider": "moa", "model": moa_cfg["default_preset"], "base_url": "moa://local",
                "api_key": "moa-virtual-provider", "api_mode": "chat_completions",
            }
            self._evict_cached_agent(_quick_key)
            event._moa_disable_after_turn = True
        except Exception:
            return True, "Failed to prepare MoA turn."
        return False, None

    async def _hm_dispatch_canonical_command(
        self, event: "MessageEvent", source: SessionSource, _quick_key: str,
        canonical: Optional[str],
    ) -> Tuple[bool, Optional[str]]:
        """Dispatch built-in idle-path commands → ``(handled, result)``; prompt-rewriting commands
        mutate ``event.text`` and return ``(False, None)`` to fall through to the agent."""
        plain_handler = (
            self._gateway_plain_command_handlers().get(canonical)
            or self._gateway_idle_command_handlers().get(canonical)
        )
        if plain_handler is not None:
            return True, await plain_handler(event)
        if canonical in self._HM_CANONICAL_COMMANDS:
            return await getattr(self, f"_hm_cmd_{canonical}")(event, source, _quick_key)
        return False, None

    async def _hm_run_exec_quick_command(self, command: str, exec_cmd: str) -> str:
        """Run a ``type: exec`` quick command in the gateway process (30 s cap, sanitized env — the
        gateway process has every API key in os.environ; output is redacted too)."""
        try:
            from tools.environments.local import build_subprocess_env
            proc = await asyncio.create_subprocess_shell(
                exec_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=build_subprocess_env(),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            output = (stdout or stderr).decode().strip()
            if output:
                from agent.redact import redact_sensitive_text
                output = redact_sensitive_text(output)
            return output or "Command returned no output."
        except asyncio.TimeoutError:
            return "Quick command timed out (30s)."
        except Exception as e:
            return f"Quick command error: {e}"

    async def _hm_dispatch_quick_and_plugin_commands(
        self, event: "MessageEvent", source: SessionSource, command: Optional[str]
    ) -> Tuple[bool, Optional[str], Optional[str]]:
        """Drain gate, user-defined quick commands (exec/alias) and plugin slash commands →
        ``(handled, result, command)``; an alias quick command rewrites ``command``."""
        if self._draining:
            return True, f"⏳ Gateway is {self._status_action_gerund()} and is not accepting new work right now.", command

        # User-defined quick commands (bypass agent loop, no LLM call)
        qcmd = self._hm_quick_commands().get(command) if command else None
        if qcmd is not None:
            # Quick commands are slash capabilities too — and type:exec ones run a shell command in
            # the gateway process. They are never in the registry, so the early gate never fires for
            # them; apply the same admin/user policy to the raw typed name here.
            # The early gate above only fires for registry-known commands, so quick commands (never in the
            # registry) would otherwise reach this dispatch sink unchecked. (#44727)
            _denied = self._check_slash_access(source, command)
            if _denied is not None:
                return True, _denied, command
            qtype = qcmd.get("type")
            if qtype == "exec":
                exec_cmd = qcmd.get("command", "")
                if not exec_cmd:
                    return True, f"Quick command '/{command}' has no command defined.", command
                return True, await self._hm_run_exec_quick_command(command, exec_cmd), command
            if qtype != "alias":
                return True, f"Quick command '/{command}' has unsupported type (supported: 'exec', 'alias').", command
            new_command = self._hm_expand_alias_quick_command(event, qcmd)
            if new_command is None:
                return True, f"Quick command '/{command}' has no target defined.", command
            command = new_command  # Fall through to normal command dispatch below

        # Plugin-registered slash commands. Underscores normalize to hyphens so Telegram's
        # underscored autocomplete form matches plugin commands registered with hyphens.
        if command:
            try:
                from hermes_cli.plugins import get_plugin_command_handler
                plugin_handler = get_plugin_command_handler(command.replace("_", "-"))
                if plugin_handler:
                    result = plugin_handler(event.get_command_args().strip())
                    if asyncio.iscoroutine(result):
                        result = await result
                    return True, str(result) if result else None, command
            except Exception as e:
                logger.warning("Plugin command dispatch failed: %s", e)
        return False, None, command

    def _hm_bundle_slash_rewrite(
        self, event: "MessageEvent", source: SessionSource, _quick_key: str, command: str
    ) -> bool:
        """Rewrite ``/<bundle>`` to the bundle invocation message; True when handled.
        Skill bundles take precedence over individual skill commands (mirrors CLI dispatch)."""
        try:
            from agent.skill_bundles import (
                build_bundle_invocation_message, resolve_bundle_command_key
            )
            bundle_key = resolve_bundle_command_key(command)
            if bundle_key is None:
                return False
            # Pass the platform explicitly: bundle skill loading bypasses get_skill_commands()'
            # scan-time disabled filter, and one gateway process serves several platforms, so
            # env-var platform resolution can't be trusted here.
            # Mirrors the stacked-skill gate (#58888).
            bundle_result = build_bundle_invocation_message(
                bundle_key, event.get_command_args().strip(), task_id=_quick_key,
                platform=source.platform.value if source.platform else None,
            )
            if not bundle_result:
                return False
            event.text, _loaded, missing = bundle_result
            if missing:
                logger.info("Bundle %s skipped missing skills: %s", bundle_key, ", ".join(missing))
            return True  # Fall through to normal message processing with bundle content
        except Exception as exc:
            logger.warning("Bundle dispatch failed: %s", exc)
            return False

    @staticmethod
    def _hm_unknown_slash_reply(command: str, source: SessionSource) -> Optional[str]:
        """Reply for a /command that is not built-in/plugin/skill; None when it is known."""
        from gateway.run import _check_unavailable_skill
        from hermes_cli.commands import GATEWAY_KNOWN_COMMANDS
        # Known-but-disabled or uninstalled skill → actionable guidance.
        _unavail_msg = _check_unavailable_skill(command)
        if _unavail_msg:
            return _unavail_msg
        # Genuinely unrecognized: warn instead of forwarding to the LLM as free text (it invents
        # tool calls). Normalize to hyphenated form first: the quick-command block may have set an
        # alias target, so the resolved def can be stale.
        if command.replace("_", "-") in GATEWAY_KNOWN_COMMANDS:
            return None
        logger.warning(
            "Unrecognized slash command /%s from %s — replying with unknown-command notice",
            command, source.platform.value if source.platform else "?",
        )
        return (
            f"Unknown command `/{command}`. "
            f"Type /commands to see what's available, "
            f"or resend without the leading slash to send "
            f"as a regular message."
        )

    def _hm_skill_slash_rewrite(
        self, event: "MessageEvent", source: SessionSource, _quick_key: str, command: Optional[str]
    ) -> Optional[str]:
        """Rewrite ``/<bundle>`` / ``/<skill>`` invocations into the skill prompt on ``event.text``;
        returns a reply string when the command is disabled/unknown/failed, else None.
        resolve_skill_command_key() handles the Telegram underscore/hyphen round-trip (/claude_code)."""
        if not command or self._hm_bundle_slash_rewrite(event, source, _quick_key, command):
            return None
        try:
            from agent.skill_commands import (
                get_skill_commands, build_skill_invocation_message, resolve_skill_command_key
            )
            skill_cmds = get_skill_commands()
            cmd_key = resolve_skill_command_key(command)
            if cmd_key is None:
                return self._hm_unknown_slash_reply(command, source)
            _plat = source.platform.value if source.platform else None
            user_instruction = event.get_command_args().strip()
            # Stacked slash-skill invocations: `/skill-a /skill-b do XYZ` loads every leading skill
            # (up to 5), not just the first. Mirrors CLI.
            try:
                from agent.skill_commands import (
                    build_stacked_skill_invocation_message as _build_stacked,
                    split_stacked_skill_commands,
                )
                extra_keys, stacked_instruction = split_stacked_skill_commands(user_instruction)
            except Exception:
                _build_stacked = None
                extra_keys, stacked_instruction = [], user_instruction
            _skill_name = skill_cmds[cmd_key].get("name", "")
            if _plat and (_skill_name or extra_keys):
                # Per-platform disabled check: get_skill_commands() only applies the *global*
                # disabled list at scan time (process-global cache across platforms), and
                # split_stacked_skill_commands() only checks each extra token is a KNOWN skill.
                from agent.skill_utils import get_disabled_skill_names as _get_plat_disabled
                _plat_disabled = _get_plat_disabled(platform=_plat)
                if _skill_name and _skill_name in _plat_disabled:
                    return (
                        f"The **{_skill_name}** skill is disabled for {_plat}.\n"
                        f"Enable it with: `hermes skills config`"
                    )
                _disabled_extra = [
                    skill_cmds.get(k, {}).get("name", "")
                    for k in extra_keys
                    if skill_cmds.get(k, {}).get("name", "") in _plat_disabled
                ]
                if _disabled_extra:
                    return (
                        f"The **{', '.join(_disabled_extra)}** skill(s) in this "
                        f"stacked invocation are disabled for {_plat}.\n"
                        f"Enable them with: `hermes skills config`"
                    )
            if extra_keys and _build_stacked is not None:
                stacked_result = _build_stacked(
                    [cmd_key, *extra_keys], stacked_instruction, task_id=_quick_key,
                )
                if not stacked_result:
                    return f"Failed to load stacked skills for /{command}."
                event.text, _loaded, _missing = stacked_result
            else:
                msg = build_skill_invocation_message(cmd_key, user_instruction, task_id=_quick_key)
                if msg:
                    event.text = msg
            # Fall through to normal message processing with skill content
        except Exception as e:
            logger.debug("Skill command check failed (non-fatal): %s", e)
        return None

    async def _hm_pending_reply_intercepts(
        self, event: "MessageEvent", source: SessionSource, _quick_key: str
    ) -> Optional[str]:
        """Replies owned by in-flight work: pending /update prompt, clarify, slash-confirm.
        Only events that may control the gateway (``allow_gateway_control``) can answer them."""
        if not event.allow_gateway_control:
            return None
        _reply = self._hm_update_prompt_reply(event, _quick_key)
        if _reply is None:
            _reply = await self._hm_clarify_reply(event, source, _quick_key)
        if _reply is None:
            _reply = await self._hm_slash_confirm_reply(event, _quick_key)
        return _reply

    async def _hm_dispatch_idle_commands(
        self, event: "MessageEvent", source: SessionSource, _quick_key: str
    ) -> Tuple[bool, Optional[str]]:
        """Idle path: resolve + dispatch slash commands; rewriting commands fall through to the agent."""
        _handled, _result, command, canonical = await self._hm_resolve_command(event, source, _quick_key)
        if not _handled:
            _handled, _result = await self._hm_dispatch_canonical_command(event, source, _quick_key, canonical)
        if not _handled:
            _handled, _result, command = await self._hm_dispatch_quick_and_plugin_commands(event, source, command)
        if not _handled:
            _result = self._hm_skill_slash_rewrite(event, source, _quick_key, command)
            _handled = _result is not None
        return _handled, _result

    def _hm_rescue_orphaned_fifo(
        self, event: "MessageEvent", source: SessionSource, is_internal: bool, _quick_key: str
    ) -> Tuple["MessageEvent", SessionSource, bool]:
        """FIFO orphan rescue: a session that went idle with a populated overflow (post-turn drain
        never promoted, e.g. a compression-demoted follow-up) silently orphaned those events. The
        oldest orphan runs as THIS turn and the incoming event is parked behind the chain. Skipped
        for control commands and internal events."""
        try:
            # ── FIFO orphan rescue (#99882) ──────────────────────────────── If this session went idle with
            # a populated overflow (queued during a busy window whose post-turn drain never promoted — e.g.
            # a compression-demoted follow-up after the compression window ended through an exit that
            # skipped the promotion site), those events were silently orphaned. We are starting the next
            # turn for this session NOW: re-stage the orphans in FIFO order and enqueue the incoming event
            # behind them, so arrival order (#28503) holds: oldest orphan runs as this turn, the rest drain
            # in order, the new message last.
            _orphan_adapter = self._adapter_for_source(source)
            if _orphan_adapter is None or getattr(event, "internal", False) or event.get_command():
                return event, source, is_internal
            _rescued = self._rescue_orphaned_overflow(_quick_key, _orphan_adapter)
            if _rescued is None:
                return event, source, is_internal
            # Into the slot when the chain was a single orphan (post-turn drain picks it up),
            # otherwise into overflow behind the already-staged next orphan.
            self._enqueue_fifo(_quick_key, event, _orphan_adapter)
            # Same session key by construction; carry the orphan's own source so reply anchors /
            # thread metadata point at the message actually being answered.
            _rescued_source = getattr(_rescued, "source", None)
            source = _rescued_source if _rescued_source is not None else source
            return _rescued, source, bool(getattr(_rescued, "internal", False))
        except Exception:
            logger.debug("FIFO orphan rescue pre-claim failed for %s", _quick_key, exc_info=True)
            return event, source, is_internal
