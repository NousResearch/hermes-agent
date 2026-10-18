"""The AIAgent surface main's HermesCLI uses, backed by the shared gateway session (CALLMAP §3.2-3.3).

``run_conversation`` submits ``prompt.submit`` and replays the gateway's event stream, on the
calling (agent) thread, into the SAME callbacks HermesCLI passed to ``AIAgent`` on main, then
returns a dict shaped like ``AIAgent.run_conversation``'s. The CLI process never builds an agent,
never runs a tool and never writes ``state.db``: the turn runs in the gateway.

Event contract: wave_a.md (frozen). Every frame is a gateway ``event`` with params
``{type, session_id, execution_generation, admission_id?, payload}``.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
import uuid

from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError
from hermes_cli.gateway_connection import DISCONNECTED, CallStopped, lost_before

logger = logging.getLogger(__name__)

# How long a turn waits for its settlement after an interrupt (main: cli_chat_turn_mixin 10 s).
INTERRUPT_SETTLE_SECONDS = 10.0
_POLL = 0.1

_RESULT_DEFAULTS = {
    "final_response": "", "failed": False, "partial": False, "error": None, "failure_reason": None,
    "completed": True, "interrupted": False, "interrupt_message": None, "api_calls": 0,
    "pending_steer": None, "last_reasoning": None, "response_previewed": False,
    "response_transformed": False, "pre_transform_response": None, "billing_block": None,
    "auxiliary_usage": {},
}
# ``session.usage`` / ``session.stats`` (wave_a2 contract, session-cumulative) → the agent attributes
# main's status bar, spinner (``↓ N tok``) and /usage read off ``AIAgent``.
_COUNTER_FROM_USAGE = {
    "session_input_tokens": "input_tokens", "session_output_tokens": "output_tokens",
    "session_prompt_tokens": "prompt_tokens", "session_completion_tokens": "completion_tokens",
    "session_total_tokens": "total_tokens", "session_cache_read_tokens": "cache_read_tokens",
    "session_cache_write_tokens": "cache_write_tokens", "session_reasoning_tokens": "reasoning_tokens",
    "session_api_calls": "api_calls",
}


class _CompressorView:
    """``agent.context_compressor`` as the status bar / banner read it (plain attributes)."""

    def __init__(self):
        self.last_prompt_tokens = 0
        self.last_real_prompt_tokens = None
        self.context_length = 0
        self.compression_count = 0
        self._config_context_length = None
        self.threshold_tokens = 0


class _TodoView:
    def __init__(self):
        self.items: list = []

    def read(self):
        return list(self.items)


class _Turn:
    """One ``run_conversation``; what each consumer got of the reply's current segment (``shown``: the
    terminal, ``spoken``: ``stream_callback``) and the closed segments and interims (``seen``), for the
    settlement after a replay gap."""

    def __init__(self, session_id):
        self.session_id, self.admission = session_id, None
        # Parts, joined on read: ``+=`` on an attribute copies the whole segment per delta (quadratic).
        self._shown, self._spoken = [], []
        self.seen, self.answered, self.gapped = set(), set(), False

    @property
    def shown(self):
        return "".join(self._shown)

    @property
    def spoken(self):
        return "".join(self._spoken)

    def close_segment(self):
        self.seen |= {self.shown.strip(), self.spoken.strip()}
        self._shown, self._spoken = [], []


class GatewayAgent:
    """Gateway-backed stand-in for ``run_agent.AIAgent`` (constructed at the same call site)."""

    platform = "cli"
    _supports_active_turn_redirect = True

    def __init__(self, *, session_db=None, session_id=None, model=None, provider=None, base_url=None,
                 api_mode=None, max_iterations=90, quiet_mode=True, tool_progress_mode="all",
                 reasoning_config=None, service_tier=None, clarify_callback=None, connection_callback=None,
                 reasoning_callback=None, thinking_callback=None, tool_progress_callback=None,
                 tool_start_callback=None, tool_complete_callback=None, stream_delta_callback=None,
                 tool_gen_callback=None, notice_callback=None, notice_clear_callback=None,
                 reaction_callback=None, interim_assistant_callback=None, connection=None,
                 push_reasoning=False, model_confirm_callback=None, modal_withdraw_callback=None,
                 one_turn=False, one_turn_reasoning=None, one_turn_alias="", **_ignored):
        conn = connection or getattr(session_db, "connection", None)
        if conn is None:
            raise GatewayClientError("Gateway session unavailable; no local fallback")
        self.connection = conn
        self._session_db = session_db
        self.session_id = session_id
        self.model = model or ""
        self.provider = provider or ""
        self.base_url = base_url or ""
        self.api_mode = api_mode
        self.api_key = None  # never on this side of the wire
        self.max_iterations = max_iterations
        self.quiet_mode = quiet_mode
        self._service_tier = None
        self.clarify_callback = clarify_callback
        self.connection_callback = connection_callback
        self._reasoning_callback = None
        self._show_reasoning_sent = None
        self.thinking_callback = thinking_callback
        self.tool_progress_callback = tool_progress_callback
        self.tool_start_callback = tool_start_callback
        self.tool_complete_callback = tool_complete_callback
        self.stream_delta_callback = stream_delta_callback
        self.tool_gen_callback = tool_gen_callback
        self.notice_callback = notice_callback
        self.notice_clear_callback = notice_clear_callback
        self.reaction_callback = reaction_callback
        self.interim_assistant_callback = interim_assistant_callback
        # The CLI's confirm prompt for the owner's model-guard refusal; None where nobody can answer (-q).
        self.model_confirm_callback = model_confirm_callback
        # Closes the CLI's open modal for one server-request method when the gateway withdraws it
        # (``request.cancel``: another viewer answered, the turn ended); called off the agent thread.
        self.modal_withdraw_callback = modal_withdraw_callback
        self._open_request = None  # (request id, method) of the server request a modal is answering
        self.suppress_status_output = False
        self._print_fn = None
        # Inert private attributes main's CLI writes (CALLMAP §3.2).
        self._last_flushed_db_idx = 0
        self._session_messages: list = []
        self._db_flush_scan_prefix = None
        self._persist_user_message_idx = self._persist_user_message_override = None
        self._persist_user_message_timestamp = None
        self._pending_cli_user_message = None
        self._voice_turn_pending = False
        self._auto_load_skills_result = None
        self._auto_load_skills_resolved = False
        self._session_db_created = True
        self._fallback_activated = False
        self._rate_limited_until = None
        self._primary_runtime = None
        self._interrupt_requested = False
        self._active_children: list = []
        self._api_latency_history: list = []
        self._api_output_history: list = []
        self._turn_base_usage_anchor = None
        self._context_tokens = 0
        self._turn_calls_base = 0
        self._turn_ask_index = None
        self._turn_api_calls = None
        self._usage_stale = False
        self._current_task_id = self._current_turn_id = self._current_api_request_id = ""
        self._memory_manager = None
        self._memory_store = None
        self._checkpoint_mgr = None
        self.session_api_calls = 0
        self.session_total_tokens = 0
        self.session_prompt_tokens = self.session_input_tokens = 0
        self.session_completion_tokens = self.session_output_tokens = 0
        self.session_cache_read_tokens = self.session_cache_write_tokens = 0
        self.session_reasoning_tokens = 0
        self.context_compressor = _CompressorView()
        self._todo_store = _TodoView()
        self._generation = None
        self._admission = None
        self._turn = None  # a turn a later one replaced stops taking frames
        self._epoch = None  # the connection epoch this turn was submitted on
        self._interrupt_sent_at = None
        self._rate_limit_state = None
        self._config_context_length = None
        # Session-scope settings the gateway applies (its turn renders tools in this mode, its agent asks
        # for this effort): written once main's rebuilt agent is ready, then on every change.
        self._tool_progress_mode = "all"
        self.tool_progress_mode = tool_progress_mode
        self._reasoning_config = None
        # Only a user-picked effort (/reasoning, a model pick's --reasoning) is committed at build; the
        # config-resolved level must never override the session's frozen launch policy on resume.
        # ``reasoning_pushed``: the commit landed, so the CLI stops re-sending it on every later rebuild.
        self.reasoning_pushed = False
        if push_reasoning:
            self.reasoning_pushed = self.pin_reasoning(reasoning_config)
        else:
            self._reasoning_config = reasoning_config
        # Main's /moa rebuilds its agent on the MoA virtual provider for one turn (the session's info
        # below then overwrites ``model``): that preset rides the turn's submit as its model_override.
        self._moa_preset = model if provider == "moa" else None
        # ``/model --once``: chat retires the agent that switch_model touched and builds this one.
        # Capture the pick, and the alias it resolved through, before ``_apply_info`` copies the session
        # model over it.
        self._turn_route = self._route_off_session(model, provider, one_turn_alias) if one_turn else None
        self._taken_turn_route = None  # the pick the admission being sent took (``_take_turn_route``)
        # ``/model B --once --reasoning X``: X rides that admission's ``model_override``, never a session pin.
        self._turn_reasoning = one_turn_reasoning if one_turn else None
        self._apply_info(self._state().info)
        if self._turn_route:
            self.model = model  # the resolved model, never the alias the admission sends
            if self._turn_route.get("provider"):
                self.provider = self._turn_route["provider"]
        self.reasoning_callback = reasoning_callback
        self.service_tier = service_tier
        state = self._state()
        state.usage_listeners.append(self.apply_session_usage)
        state.request_listeners.append(self._on_request_withdrawn)
        if state.usage:
            self.apply_session_usage(state.usage)

    # -- session-scope settings ------------------------------------------------------------------
    @property
    def tool_progress_mode(self):
        return self._tool_progress_mode

    @tool_progress_mode.setter
    def tool_progress_mode(self, mode):
        """``/verbose`` / ``/focus``: ``config.set verbose`` with the explicit mode (never ``cycle``, which
        could drift from the CLI's own cycle)."""
        if mode == self._tool_progress_mode or mode not in {"off", "new", "all", "verbose"}:
            self._tool_progress_mode = mode
            return
        self._tool_progress_mode = mode
        try:
            self.connection.call("config.set", session_id=self.session_id, key="verbose", value=mode)
        except GatewayClientError as exc:
            logger.debug("config.set verbose failed: %s", exc)

    @property
    def reasoning_config(self):
        return self._reasoning_config

    @reasoning_config.setter
    def reasoning_config(self, config):
        """/new's re-derived effort and a one-turn restore: ``session.mutate reasoning`` when it differs from
        the session's effort (an inherited level stays inherited)."""
        self._set_reasoning(config, explicit=False)

    def pin_reasoning(self, config):
        """``/reasoning <level>`` (session scope; main rebuilds its agent with the new config) and a model
        pick's ``--reasoning``: always ``session.mutate reasoning``, even when the session already runs that
        level by inheritance, so it becomes the session's own choice. True when the owner committed it."""
        return self._set_reasoning(config, explicit=True)

    def stage_turn_reasoning(self, config):
        """``/model B --once --reasoning X``: X for the one-turn admission only (its ``model_override``).
        Nothing is sent to the session."""
        self._reasoning_config = config
        self._turn_reasoning = _effort_word(config)

    def restore_reasoning(self, config):
        """A one-turn restore: the level goes back on this object only. The session was never written."""
        self._reasoning_config = config
        self._turn_reasoning = None

    def _set_reasoning(self, config, *, explicit):
        self._reasoning_config = config
        effort = _effort_word(config)
        if effort is None:
            return False
        info = self._state().info or {}
        if not explicit and info.get("reasoning_effort") == effort:
            return True
        try:
            snapshot = self.connection.call("session.resume", session_id=self.session_id)
            # A policy mutation is generation-fenced: the owner refuses it without the generation.
            self.connection.call("session.mutate", session_id=self.session_id, request_id=uuid.uuid4().hex,
                                 expected_revision=snapshot["revision"],
                                 expected_generation=snapshot["execution_generation"], operation="reasoning",
                                 payload={"effort": effort})
            info["reasoning_effort"] = effort
            return True
        except (GatewayClientError, KeyError, TypeError) as exc:
            logger.debug("session.mutate reasoning failed: %s", exc)
            return False

    def _route_off_session(self, model, provider="", alias=""):
        """``{model, provider}`` the turn sends when the session runs another route, else None.

        ``alias``, the ``model_aliases`` entry the pick resolved through, is sent instead of the bare
        model: two aliases may name one model at different endpoints, so the owner must not re-resolve
        it. An alias makes even the session's own model a one-turn override."""
        state = self._state()
        if not model or not (state.info or {}).get("model") or (state.runs_route(model, provider) and not alias):
            return None
        return {"model": alias or model, **({"provider": provider} if provider else {})}

    def switch_model(self, new_model, new_provider="", api_key="", base_url="", api_mode="", capabilities=None, *,
                     once=False, restore=False, alias=""):
        """Session-scoped ``/model`` and /resume commit ``session.mutate model``.

        ``once=True`` does not write the session. The next ``chat`` builds a new agent; that
        constructor carries the pick, because this agent is retired when the route signature
        changes. ``restore=True`` only puts the previous model back on this object and clears the
        one-turn route. The admission already has its ``model_override``.

        "Already on this route" is the session row, not ``self.model`` (a one-turn pick sets
        ``self.model`` without committing). A session switch of that same target still mutates
        on the generation fence.

        A guarded target the owner refuses with ``confirmation_required`` is re-sent with its token
        only on the user's yes to ``model_confirm_callback``: a no returns False with nothing
        written; with no prompt the refusal raises. Any refusal raises: main's callers roll their
        own fields back."""
        if once or restore:
            # ``alias``: the ``model_aliases`` entry a ``--once`` pick resolved through (``_route_off_session``).
            self._turn_route = None if restore else self._route_off_session(new_model, new_provider, alias)
            if restore:
                self._turn_reasoning = None
            if new_model:
                self.model = new_model
            if new_provider:
                self.provider = new_provider
            return True
        if new_model == self.model and (not new_provider or new_provider == self.provider):
            if self._route_off_session(new_model, new_provider) is None:
                self._turn_route = None
                return True
        from hermes_cli.gateway_mutations import model_refusal_text
        payload = {"model": new_model, **({"provider": new_provider} if new_provider else {})}
        snapshot = self.connection.call("session.resume", session_id=self.session_id)
        fence = dict(session_id=self.session_id, operation="model", expected_revision=snapshot["revision"],
                     expected_generation=snapshot["execution_generation"])
        result = self.connection.call("session.mutate", request_id=uuid.uuid4().hex, payload=payload, **fence) or {}
        if result.get("status") == "confirmation_required":
            if self.model_confirm_callback is None:
                raise GatewayClientError(model_refusal_text(result))
            if not self.model_confirm_callback(result):
                return False
            # The refusal wrote nothing: the same fence is re-sent with the owner's token.
            result = self.connection.call("session.mutate", request_id=uuid.uuid4().hex,
                                          payload={**payload, "confirm": result["confirm"]}, **fence) or {}
            if result.get("status") == "confirmation_required":
                raise GatewayClientError(model_refusal_text(result, confirmed=True))
        state = self._state()
        try:
            state.adopt_snapshot(self.connection.call("session.resume", session_id=self.session_id))
        except (GatewayClientError, KeyError, TypeError) as exc:
            # Committed: the next turn runs on the new model, so main must not roll its fields back.
            logger.warning("session.resume after the committed model switch failed: %s", exc)
        # A rebuilt stand-in reads the session's info: it must show the committed model.
        provider = result.get("provider") or new_provider
        state.info = {**(state.info or {}), "model": result.get("model") or new_model,
                      **({"provider": provider} if provider else {})}
        # The session now runs this route, so a leftover one-turn override must not ride the next submit.
        self._turn_route = None
        self._apply_info(state.info)
        return True

    # -- session identity ----------------------------------------------------------------------
    @property
    def session_id(self):
        return self._session_id

    @session_id.setter
    def session_id(self, value):
        """Main repoints ``agent.session_id`` on /new, /resume and /branch: attach this viewer to that
        gateway session (idempotent) and read its state from then on."""
        previous = getattr(self, "_session_id", None)
        self._session_id = value
        if previous is None or value == previous or not value:
            return
        store = self._session_db
        if store is not None and hasattr(store, "attach"):
            try:
                store.attach(value)
            except GatewayClientError as exc:
                logger.debug("session.resume %s failed: %s", value, exc)
        else:
            try:
                self.connection.call("session.resume", session_id=value)
            except GatewayClientError as exc:
                logger.debug("session.resume %s failed: %s", value, exc)
        state = self._state()
        if self.apply_session_usage not in state.usage_listeners:
            state.usage_listeners.append(self.apply_session_usage)
        if self._on_request_withdrawn not in state.request_listeners:
            state.request_listeners.append(self._on_request_withdrawn)
        self._apply_info(state.info)
        if state.usage:
            self.apply_session_usage(state.usage)

    # -- reasoning display -------------------------------------------------------------------
    @property
    def reasoning_callback(self):
        return self._reasoning_callback

    @reasoning_callback.setter
    def reasoning_callback(self, callback):
        """Main installs a reasoning sink at construction and on ``/reasoning show|hide`` / verbose; the
        sink's presence is what makes the agent ask the provider for reasoning. The gateway agent
        gets the same switch: the session's ``show_reasoning`` display flag, sent on each change."""
        self._reasoning_callback = callback
        wanted = (self.session_id, callback is not None)
        if wanted == self._show_reasoning_sent:  # per session: a rotated id gets its own flag
            return
        try:
            self.connection.call("config.set", session_id=self.session_id, key="show_reasoning",
                                 value="1" if wanted[1] else "0", timeout=10)
        except GatewayClientError as exc:
            logger.debug("config.set show_reasoning failed: %s", exc)
            return
        self._show_reasoning_sent = wanted

    @property
    def service_tier(self):
        return self._service_tier

    @service_tier.setter
    def service_tier(self, tier):
        """Main rebuilds AIAgent(service_tier=...) after ``/fast``; the gateway agent gets the same tier
        as the session's override (``config.set service_tier``), sent when it differs from the last
        one sent for this session. The session's own default (None at first build) is left alone."""
        self._service_tier = tier
        # Remembered on the session's shared state: /fast rebuilds this stand-in, the session stays.
        state = self._state()
        wanted = tier or "normal"
        sent = getattr(state, "service_tier_sent", None)
        if wanted == sent or (sent is None and not tier):
            return
        try:
            self.connection.call("config.set", session_id=self.session_id, key="service_tier",
                                 value=wanted, timeout=10)
        except GatewayClientError as exc:
            logger.debug("config.set service_tier failed: %s", exc)
            return
        state.service_tier_sent = wanted

    # -- snapshot-backed reads ---------------------------------------------------------------
    def _state(self):
        return self.connection.state(self.session_id)

    def _apply_info(self, info):
        info = info or {}
        if info.get("model"):
            self.model = info["model"]
        if info.get("provider"):
            self.provider = info["provider"]
        if info.get("context_length"):
            self.context_compressor.context_length = int(info["context_length"])

    @property
    def tools(self):
        """Tool schemas the gateway session will load (``info.tools`` {toolset: [names]})."""
        groups = (self._state().info or {}).get("tools") or {}
        names = [name for members in groups.values() for name in (members or [])]
        return [{"type": "function", "function": {"name": name}} for name in names]

    @property
    def valid_tool_names(self):
        return {tool["function"]["name"] for tool in self.tools}

    def get_rate_limit_state(self):
        return self._rate_limit_state

    def apply_session_usage(self, usage):
        """Install one session-cumulative usage dict (the ``session.usage`` event / ``session.stats``
        result) on main's attributes. Plain assignments: the render thread reads them lock-free."""
        if not isinstance(usage, dict):
            return
        for attr, key in _COUNTER_FROM_USAGE.items():
            value = usage.get(key)
            if isinstance(value, int) and not isinstance(value, bool):
                setattr(self, attr, value)
        compressor = self.context_compressor
        if isinstance(usage.get("last_prompt_tokens"), int):
            compressor.last_prompt_tokens = usage["last_prompt_tokens"]
        if usage.get("context_length"):
            compressor.context_length = int(usage["context_length"])
        if "config_context_length" in usage:
            pin = usage.get("config_context_length")
            compressor._config_context_length = self._config_context_length = pin if isinstance(pin, int) else None
        if isinstance(usage.get("compression_count"), int):
            compressor.compression_count = usage["compression_count"]
        source = usage.get("context_source")
        if source is not None:
            # context_display_source(): a "local_estimate" reading is a shown value != the real one.
            compressor.last_real_prompt_tokens = (
                None if source == "provider_usage" else compressor.last_prompt_tokens + 1)
        if isinstance(usage.get("context_tokens"), int):
            self._context_tokens = usage["context_tokens"]
            self._anchor_display()
        latency, tps = usage.get("avg_latency_s"), usage.get("tokens_per_s")
        if isinstance(latency, (int, float)) and latency > 0 and isinstance(tps, (int, float)):
            # The bar averages these deques (mean latency, sum/sum velocity): one aligned sample each.
            self._api_latency_history = [float(latency)]
            self._api_output_history = [float(tps) * float(latency)]
        if "rate_limit" in usage:
            self._rate_limit_state = _rate_limit_state(usage.get("rate_limit"))
        if usage.get("model"):
            self.model = usage["model"]
        if usage.get("provider"):
            self.provider = usage["provider"]

    # -- no-ops: the gateway owns persistence, prompt cache, memory and clients --------------
    def _flush_messages_to_session_db(self, *a, **k):
        return None

    def _persist_session(self, *a, **k):
        return None

    def _ensure_db_session(self, *a, **k):
        return None

    def _invalidate_system_prompt(self, *a, **k):
        return None

    def reset_session_state(self, *a, **k):
        return None

    def release_clients(self, *a, **k):
        return None

    def clear_interrupt(self, *a, **k):
        self._interrupt_requested = False

    def shutdown_memory_provider(self, *a, **k):
        return None

    def close(self, *a, **k):
        return None

    def _restore_primary_runtime(self, *a, **k):
        return None

    def _summarize_api_error(self, exc):
        return str(exc)[:300]

    # -- controls ------------------------------------------------------------------------------
    def interrupt(self, message=None):
        """``session.interrupt`` for the running generation; ``message`` stays CLI-side (main
        re-queues it as the next turn itself)."""
        self._interrupt_requested = True
        if self._interrupt_sent_at is None:
            self._interrupt_sent_at = time.monotonic()
        generation = self._generation
        if generation is None:
            generation = self._state().execution_generation
        try:
            self.connection.call("session.interrupt", session_id=self.session_id,
                                 execution_generation=generation, timeout=5)
        except Exception as exc:  # health: allow BLE001 -- the settle wait below bounds a lost interrupt
            logger.debug("session.interrupt failed: %s", exc)

    def hard_interrupt(self, message=None, tool_reason=None):
        self.interrupt(message)

    def _correct(self, verb, text):
        generation = self._generation
        if generation is None:
            return False
        try:
            result = self.connection.call(f"session.{verb}", session_id=self.session_id, text=text,
                                          execution_generation=generation)
        except GatewayClientError:
            return False
        return (result or {}).get("status") in {"queued", "redirected"}

    def _compress_context(self, messages, system_message=None, *, approx_tokens=None, focus_topic=None,
                          force=True, verbatim_tail=None, **_kw):
        """Main's manual ``/compress`` seam (``compress_now``) as ``session.mutate compress``: the
        gateway summarizes and commits; the returned transcript is the gateway's compacted history.
        ``verbatim_tail`` rows are stamped as stored, so ``compress_now`` does not rejoin them."""
        from agent.context_compressor import _DB_PERSISTED_MARKER
        from agent.conversation_compression import COMPACTION_STATUS
        from hermes_cli.gateway_client import COMPRESS_RPC_TIMEOUT
        payload = {}
        if focus_topic:
            payload["focus"] = focus_topic
        if verbatim_tail:
            payload["partial"] = True
            payload["keep_last"] = max(1, sum(1 for m in verbatim_tail if m.get("role") == "user"))
        self._emit_print(COMPACTION_STATUS)
        snapshot = self.connection.call("session.resume", session_id=self.session_id)
        self.connection.call("session.mutate", timeout=COMPRESS_RPC_TIMEOUT, session_id=self.session_id,
                             request_id=uuid.uuid4().hex, expected_revision=snapshot["revision"],
                             expected_generation=snapshot["execution_generation"],
                             operation="compress", payload=payload)
        for row in verbatim_tail or ():
            row[_DB_PERSISTED_MARKER] = True
        after = self.connection.call("session.resume", session_id=self.session_id)
        self._state().adopt_snapshot(after)
        compressor = self.context_compressor
        compressor.compression_count += 1
        compressor.last_prompt_tokens = -1
        self._turn_base_usage_anchor = None
        return [{k: v for k, v in m.items() if k != "row_id"} for m in after.get("messages") or []
                if not m.get("archived")], system_message

    def steer(self, text):
        return self._correct("steer", text)

    def redirect(self, text):
        accepted = self._correct("redirect", text)
        if accepted:
            # Main's turn then answers the redirect row first (the interrupted call has no usage), so
            # its anchor is priced through that row: it, not the original ask, is the turn's ask.
            self._turn_ask_index = None
        return accepted

    # -- the turn ------------------------------------------------------------------------------
    def _submit_params(self, user_message, persist_user_message, turn_author):
        attachments = None
        if isinstance(user_message, list):
            # Main's native image turn: OpenAI content parts built from local files. The gateway takes
            # text plus staged attachments and routes the images itself (native or described).
            user_message, attachments = self._stage_content_parts(user_message)
            if isinstance(persist_user_message, list):
                persist_user_message = user_message
        text = persist_user_message if isinstance(persist_user_message, str) else user_message
        params = {"session_id": self.session_id, "submission_id": uuid.uuid4().hex, "text": text}
        if attachments:
            params["attachments"] = attachments
        if isinstance(user_message, str) and user_message != text:
            params["api_text"] = user_message
        if turn_author:
            params["turn_author"] = turn_author
        if self._voice_turn_pending:
            params["voice_turn"] = True
        staged = self._pending_cli_user_message
        if isinstance(staged, dict) and staged.get("display_kind"):
            params["display_kind"] = staged["display_kind"]
        if getattr(self, "_finite", False):
            params["finite"] = True
        if self.clarify_callback is not None:
            # Main's batch panel answers the clarify tool's whole question set (one ``clarify`` request).
            params["clarify_batch"] = True
        info = self._state().info or {}
        pending = self._take_turn_route()
        if pending:
            # /model --once: this admission alone runs the pick. The session row stays put.
            params["model_override"] = pending
        elif self._moa_preset and (info.get("provider"), info.get("model")) != ("moa", self._moa_preset):
            # /moa one-shot: this admission alone runs the preset (unless it is the session's own
            # route); the session's route is untouched.
            params["model_override"] = {"model": self._moa_preset, "provider": "moa"}
        return params

    def _take_turn_route(self):
        """The ``/model --once`` pick for the admission being built, consumed here: one admission carries
        it, even when a later ``run_conversation`` submits while this one's ``prompt.submit`` still blocks.
        A refusal that admitted nothing (``session_busy``) puts it back with ``_give_back_turn_route``.

        Staging already compared the pick with the session's route (``_route_off_session``), so it is not
        compared again: its ``model`` may be an alias spelled like the session's model, and the owner runs a
        plain pick of the session's own route on the session's endpoint and key (``one_turn_runtime``)."""
        pending, reasoning = self._turn_route, self._turn_reasoning
        self._turn_route = self._turn_reasoning = self._taken_turn_route = None
        if not pending:
            return None
        self._taken_turn_route = (pending, reasoning)
        return {**pending, **({"reasoning": reasoning} if reasoning else {})}

    def _give_back_turn_route(self):
        taken, self._taken_turn_route = getattr(self, "_taken_turn_route", None), None
        if taken and self._turn_route is None:
            self._turn_route, self._turn_reasoning = taken

    def _stage_content_parts(self, parts):
        """``[{type: text}, {type: image_url, image_url: {url: data:...}}]`` -> (text, attachments): each
        image is staged on the owner with ``image.attach_bytes`` (bytes read here, locally, by main's
        ``build_native_content_parts``); main's ``[Image attached at: <local path>]`` hints are dropped
        because the gateway names its own staged copies."""
        texts, attachments = [], []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if part.get("type") == "text":
                lines = str(part.get("text") or "").splitlines()
                texts.append("\n".join(ln for ln in lines if not ln.startswith("[Image attached")).strip())
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url") if isinstance(part.get("image_url"), dict) else None
                if isinstance(url, str) and url.startswith("data:image/"):
                    staged = self.connection.call("image.attach_bytes", session_id=self.session_id, content_base64=url)
                    attachments.append({"path": staged["path"], "mime": staged["mime"]})
        return "\n\n".join(t for t in texts if t), attachments

    def run_conversation(self, user_message, conversation_history=None, stream_callback=None,
                         task_id=None, persist_user_message=None, turn_author=None, **_kw):
        turn = self._turn = _Turn(self.session_id)  # first: a turn main gave up on stops consuming
        history = list(conversation_history or [])
        self._turn_ask_index = len(history)  # the staged ask's row in this turn's transcript
        self._turn_api_calls = None
        self._turn_calls_base = self.session_api_calls
        self._usage_stale = True
        self._interrupt_requested = False
        self._interrupt_sent_at = None
        from hermes_cli.gateway_tools_notify import take_admitted_turn
        adopted = take_admitted_turn(self, persist_user_message if isinstance(persist_user_message, str)
                                     else user_message)
        if adopted is not None:
            # A notification turn the gateway already admitted and is running (its frames are queued).
            receipt = adopted
        else:
            self._drain_stale()
            self._taken_turn_route = None
            try:
                # Inside the try: staging an image is a gateway call, and its failure is a failed turn.
                # Building the params consumes a ``/model --once`` pick (``_take_turn_route``).
                state = self._state()
                if state.gaps != state.attached_gaps:  # a replay gap between turns retired the subscription
                    self.connection.reattach(turn.session_id, abandon=self._stop(turn))
                params = self._submit_params(user_message, persist_user_message, turn_author)
                if self._stop(turn)():
                    raise CallStopped("prompt.submit")
                receipt = self.connection.call("prompt.submit", **{**params, "session_id": turn.session_id})
            except CallStopped:  # main moved on (a later turn, /new, Ctrl-C's bound) before the submit
                self._give_back_turn_route()  # nothing was sent: the pick is still owed
                return self._failed(history, user_message, "Stopped before the prompt was sent; nothing was sent.")
            except GatewayRPCError as exc:
                # The owner answered and refused: not a lost connection. Only its own site code
                # (session_busy: send again) is a verdict; any other refusal stays unclassified.
                busy = str(exc) == "session_busy"
                if busy:
                    # Nothing was admitted: the resend carries the pick (the CLI keeps its restore).
                    self._give_back_turn_route()
                return self._failed(history, user_message, str(exc),
                                    failure_reason="session_busy" if busy else None, retryable=busy)
            except GatewayClientError as exc:
                return self._failed(history, user_message, str(exc), failure_reason="gateway_connection")
        self._epoch = getattr(self.connection, "epoch", None)
        turn.admission = self._admission = (receipt or {}).get("admission_id")
        if receipt and receipt.get("execution_generation") is not None:
            self._generation = receipt["execution_generation"]
        starts: dict = {}
        try:
            outcome = self._pump(stream_callback, starts, turn)
        finally:
            if self._owns(turn):
                self._generation_done()
        if isinstance(outcome, Exception) and self._owns(turn):
            return self._failed(history, user_message, str(outcome) or DISCONNECTED,
                                failure_reason="gateway_connection")
        result = self._result(history, user_message, outcome, turn) if self._owns(turn) else None
        if not self._owns(turn):  # a later turn or a session switch took over: it reads and paints nothing
            return self._failed(history, user_message, "A later turn took over this terminal.")
        if turn.gapped:
            self._paint_owed(turn, result, outcome, stream_callback)
        return result

    def _drain_stale(self):
        """Frames left from an earlier turn of this viewer (a late settlement) are not this turn's."""
        while True:
            try:
                frame = self.connection.events.get_nowait()
            except queue.Empty:
                return
            if isinstance(frame, Exception):
                if lost_before(frame, getattr(self.connection, "epoch", None)):
                    continue  # a connection a reconnect already replaced
                self.connection.events.put(frame)
                return

    def _generation_done(self):
        self._generation = None

    def _expired(self):
        return self._interrupt_sent_at is not None and time.monotonic() - self._interrupt_sent_at > INTERRUPT_SETTLE_SECONDS

    def _owns(self, turn):
        return self._turn is turn and self.session_id == turn.session_id

    def _stop(self, turn):
        """The ``abandon`` of a turn's blocking reads: it lost the terminal, or Ctrl-C's settle bound passed."""
        return lambda: not self._owns(turn) or self._expired()

    def _lost(self, turn):
        """The ``abandon`` of a turn's end-of-turn reads (its receipt, the cumulative stats): only losing the
        terminal stops them. Past Ctrl-C's settle bound they still run, bounded by their own timeouts."""
        return lambda: not self._owns(turn)

    def _pump(self, stream_callback, starts, turn):
        """Dispatch frames until this admission's ``message.complete``; returns its payload, or the
        transport error (disconnect)."""
        events = self.connection.events
        while self._owns(turn):
            if self._expired():
                return {"status": "interrupted", "outcome": "cancelled", "_settle_timeout": True}
            try:
                frame = events.get(timeout=_POLL)
            except queue.Empty:
                continue
            if not self._owns(turn):
                with events.mutex:  # the later turn's frame goes back where it was
                    events.queue.appendleft(frame)
                break
            if isinstance(frame, Exception):
                if lost_before(frame, self._epoch):
                    continue  # the previous connection's loss, already reported; this turn is live
                return frame
            params = frame.get("params") if isinstance(frame, dict) else None
            if not isinstance(params, dict) or params.get("session_id") != self.session_id:
                continue
            if frame.get("method") in _SERVER_REQUESTS and "id" in frame:
                self._answer_request(frame)
                continue
            kind, payload = params.get("type"), params.get("payload") or {}
            admission = params.get("admission_id") or (payload.get("admission_id") if isinstance(payload, dict) else None)
            if self._admission and admission and admission != self._admission:
                continue
            if params.get("execution_generation") is not None:
                self._generation = params["execution_generation"]
                self._state().execution_generation = params["execution_generation"]
            if kind == "message.complete":
                return payload
            if kind == "session.replay_gap":
                outcome = self._recover(turn)
                if outcome is not None:
                    return outcome
                continue
            try:
                self._dispatch(kind, payload, stream_callback, starts)
            except Exception:
                logger.debug("event %s dispatch failed", kind, exc_info=True)
        return None

    def _recover(self, turn):
        """``gateway_replay_gap``: re-attach, answer the controls the hole hid, stop painting live text. A
        turn the snapshot no longer lists settles from its receipt; None waits for the live completion."""
        from hermes_cli.gateway_replay_gap import settled
        lost, stop = GatewayClientError("session_replay_gap: " + DISCONNECTED), self._stop(turn)
        try:
            snapshot = self.connection.reattach(turn.session_id, abandon=stop)
            row = next((r for r in snapshot.get("pending") or [] if r.get("admission_id") == turn.admission), {})
            if row.get("status") == "unknown":
                return lost  # its completion was the one the hole lost: no other is owed
            turn.gapped |= row.get("status") != "queued"  # one still queued lost nothing
            for prompt in snapshot.get("prompts") or []:
                if row.get("status") == "started" and prompt.get("execution_generation") == row.get("execution_generation") \
                        and prompt.get("prompt_id") not in turn.answered and not stop():
                    turn.answered.add(prompt.get("prompt_id"))
                    try:
                        self._dispatch(f"{prompt.get('kind')}.request", prompt, None, {})
                    except Exception:  # a broken modal is a skipped answer, never a failed turn
                        logger.debug("control %s dispatch failed", prompt.get("prompt_id"), exc_info=True)
            if row or not turn.admission or stop():
                return None
            receipt = self.connection.call("prompt.receipt", session_id=turn.session_id, admission_id=turn.admission,
                                           include_result=True, abandon=stop)
        except CallStopped:
            return None
        except GatewayClientError:
            return lost
        verdict = settled(receipt or {})
        return {"_receipt": receipt} if verdict else lost if verdict is None else None

    def _dispatch(self, kind, payload, stream_callback, starts):
        handler = _HANDLERS.get(kind)
        if handler is not None:
            handler(self, payload, stream_callback, starts)

    # event handlers (agent thread) ------------------------------------------------------------
    def _on_delta(self, payload, stream_callback, starts):
        text, turn = payload.get("text"), self._turn
        if not isinstance(text, str):
            return self._close_segment(starts, separator=True)
        if turn.gapped:
            return  # live text stopped at the replay gap; the settlement shows the reply once
        if self.stream_delta_callback is not None:
            self.stream_delta_callback(text)
            turn._shown.append(text)
        if stream_callback is not None:
            stream_callback(text)
            turn._spoken.append(text)

    def _close_segment(self, starts, separator=False):
        """A segment boundary: the ``text: None`` separator, or a later boundary when it fell in a gap."""
        turn = self._turn
        if self.stream_delta_callback is not None and (separator or any(turn._shown)):
            self.stream_delta_callback(None)
        turn.close_segment()

    def _on_interim(self, payload, stream_callback, starts):
        text = payload.get("text")
        self._close_segment(starts)
        self._turn.seen.add(str(text or "").strip())
        if self.interim_assistant_callback is not None:
            self.interim_assistant_callback(text, already_streamed=bool(payload.get("already_streamed")))
        elif text and str(text).strip():
            self._emit_print(text)

    def _on_tool_generating(self, payload, stream_callback, starts):
        if self.tool_gen_callback is not None:
            self.tool_gen_callback(payload.get("name"))

    def _on_tool_start(self, payload, stream_callback, starts):
        self._close_segment(starts)
        name = payload.get("name") or payload.get("tool_name") or "tool"
        args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
        call_id = payload.get("tool_id") or payload.get("tool_call_id") or ""
        starts[call_id] = time.monotonic()
        if self._usage_stale:
            # Main's counters move right after each API call, before its tools run (the spinner's
            # `↓ N tok` beside a tool or approval); the gateway's next session.usage frame comes only
            # at the next step, so the first tool of each batch reads session.stats once.
            self._usage_stale = False
            self._refresh_stats()
        if self.tool_progress_callback is not None:
            from agent.display import build_tool_preview
            self.tool_progress_callback("tool.started", name, build_tool_preview(name, args), args)
        # No tool_start_callback: main's snapshots the local FS for inline diffs; the gateway
        # snapshots before its tool writes and ships ``tool.complete.inline_diff`` instead.

    def _on_tool_complete(self, payload, stream_callback, starts):
        name = payload.get("name") or payload.get("tool_name") or "tool"
        args = payload.get("args") if isinstance(payload.get("args"), dict) else {}
        call_id = payload.get("tool_id") or payload.get("tool_call_id") or ""
        result = payload.get("result")
        duration = payload.get("duration_s")
        if duration is None:
            started = starts.pop(call_id, None)
            duration = (time.monotonic() - started) if started is not None else 0.0
        if self.tool_progress_callback is not None:
            self.tool_progress_callback("tool.completed", name, None, None, duration=duration,
                                        is_error=bool(payload.get("is_error")), result=result)
        self._usage_stale = True
        from hermes_cli.gateway_tools import mark_dirty
        mark_dirty()  # a terminal/process call may have started or stopped a background process
        if name == "todo":
            self._todo_from_result(result)
        diff = payload.get("inline_diff")
        if isinstance(diff, str) and diff:
            from cli import _cprint
            for line in diff.rstrip("\n").splitlines():
                _cprint(line)
            if name == "delegate_task" and self.tool_complete_callback is not None:
                self.tool_complete_callback(call_id, name, args, result)
        elif self.tool_complete_callback is not None:
            self.tool_complete_callback(call_id, name, args, result)

    def _todo_from_result(self, result):
        try:
            data = json.loads(result) if isinstance(result, str) else result
        except ValueError:
            return
        if isinstance(data, dict) and isinstance(data.get("todos"), list):
            self._todo_store.items = list(data["todos"])

    def _on_todo(self, payload, stream_callback, starts):
        if isinstance(payload.get("todos"), list):
            self._todo_store.items = list(payload["todos"])

    def _on_thinking(self, payload, stream_callback, starts):
        if self.thinking_callback is not None:
            self.thinking_callback(payload.get("text") or "")

    def _on_reasoning(self, payload, stream_callback, starts):
        if self.reasoning_callback is not None:
            self.reasoning_callback(payload.get("text") or "")

    def _on_reasoning_available(self, payload, stream_callback, starts):
        if self.tool_progress_callback is not None:
            self.tool_progress_callback("reasoning.available", "_thinking", (payload.get("text") or "")[:500], None)

    def _on_notice(self, payload, stream_callback, starts):
        if self.notice_callback is None:
            return
        from agent.credits_tracker import AgentNotice
        self.notice_callback(AgentNotice(text=str(payload.get("text") or ""), level=payload.get("level") or "info",
                                         kind=payload.get("kind") or "sticky", ttl_ms=payload.get("ttl_ms"),
                                         key=payload.get("key"), id=payload.get("id")))

    def _on_notice_clear(self, payload, stream_callback, starts):
        if self.notice_clear_callback is not None:
            self.notice_clear_callback(payload.get("key"))

    def _on_reaction(self, payload, stream_callback, starts):
        if self.reaction_callback is not None:
            self.reaction_callback(payload.get("kind"))

    def _on_print(self, payload, stream_callback, starts):
        self._emit_print(payload.get("text") or "")

    def _on_review_summary(self, payload, stream_callback, starts):
        line = review_summary_line(payload)
        if line:
            self._emit_print(line)

    def _emit_print(self, text):
        if self.suppress_status_output:
            return
        (self._print_fn or print)(text)

    def _on_approval(self, payload, stream_callback, starts):
        from tools.terminal_tool import _get_approval_callback
        callback = _get_approval_callback()
        choices = list(payload.get("choices") or [])
        choice = "deny"
        if callback is not None:
            try:
                choice = callback(payload.get("command") or "", payload.get("description") or "",
                                  allow_permanent="always" in choices, allow_session="session" in choices,
                                  smart_denied=False)
            except TypeError:
                choice = callback(payload.get("command") or "", payload.get("description") or "")
        if choice not in choices:
            choice = "deny"
        self.connection.call("approval.respond", session_id=self.session_id,
                             execution_generation=payload.get("execution_generation", self._generation),
                             prompt_id=payload.get("prompt_id"), choice=choice)

    def _on_clarify(self, payload, stream_callback, starts):
        answer = ""
        if self.clarify_callback is not None:
            qid = payload.get("prompt_id") or "q0"
            reply = self.clarify_callback([{"qid": qid, "question": payload.get("question") or "",
                                            "choices": list(payload.get("choices") or []),
                                            "multi_select": bool(payload.get("multi_select"))}])
            if isinstance(reply, dict):
                value = (reply.get("answers") or {}).get(qid)
                answer = ", ".join(map(str, value)) if isinstance(value, (list, tuple)) else (value or "")
            elif isinstance(reply, str):
                answer = reply
        self.connection.call("clarify.respond", session_id=self.session_id,
                             execution_generation=payload.get("execution_generation", self._generation),
                             prompt_id=payload.get("prompt_id"), answer=str(answer))

    def _answer_request(self, frame):
        """A gateway server→client request (``sudo``, ``secret``, ``clarify``): main's own modal
        answers it on this (agent) thread, through the callback HermesCLI registered for the turn; the
        answer travels only in the JSON-RPC response frame."""
        method = frame["method"]
        self._open_request = (frame["id"], method)
        try:
            answer = _SERVER_REQUESTS[method](self, frame.get("params") or {})
        except Exception:  # health: allow BLE001 -- a broken modal is a skip, never a hung tool
            logger.debug("server request %s failed", method, exc_info=True)
            answer = {"outcome": "cancelled"} if method == "clarify" else ""
        finally:
            self._open_request = None
        if method == "clarify":
            self.connection.reply(frame["id"], answer)
        else:
            self.connection.reply(frame["id"], {"value": answer if isinstance(answer, str) else ""})

    def _on_request_withdrawn(self, payload):
        """``request.cancel`` (connection loop thread): the gateway settled or withdrew a request. When
        this agent thread is parked in the modal answering it (another viewer completed the form, the
        turn timed out or was interrupted), release that modal so the pump reads the turn's remaining
        frames; its late reply to the already-settled request is dropped by the gateway."""
        current = self._open_request
        if current is None or payload.get("id") != current[0] or self.modal_withdraw_callback is None:
            return
        try:
            self.modal_withdraw_callback(current[1])
        except Exception:  # health: allow BLE001 -- a broken teardown must not stop the connection pump
            logger.debug("modal withdraw for %s failed", current[1], exc_info=True)

    def _ask_clarify(self, params):
        """Main's batch clarify panel (``clarify_callback(questions)``) for the whole question set."""
        questions = [{"qid": q.get("qid"), "question": q.get("question") or "",
                      "choices": list(q.get("choices") or []), "multi_select": bool(q.get("multi_select"))}
                     for q in params.get("questions") or [] if isinstance(q, dict) and q.get("qid")]
        if self.clarify_callback is None or not questions:
            return {"outcome": "cancelled"}
        reply = self.clarify_callback(questions)
        if not isinstance(reply, dict):
            return {"outcome": "cancelled"}
        answers = {qid: (value if isinstance(value, str) or value is None else json.dumps(value))
                   for qid, value in (reply.get("answers") or {}).items()}
        out = {"answers": answers}
        if reply.get("outcome") in ("submitted", "timed_out", "cancelled"):
            out["outcome"] = reply["outcome"]
        return out

    def _ask_sudo(self, params):
        from tools.terminal_tool import _get_sudo_password_callback
        callback = _get_sudo_password_callback()
        return (callback() or "") if callback is not None else ""

    def _ask_secret(self, params):
        """Main's masked "Skill Setup Required" panel (the CLI's registered secret-capture callback,
        ``hermes_cli.callbacks.prompt_for_secret``) collects the value and prints main's lines; the
        gateway, not this process, saves it in the profile ``.env``, so the panel's save is captured."""
        from tools import skills_tool
        callback = skills_tool._secret_capture_callback
        if callback is None:
            return ""
        from hermes_cli import callbacks
        captured = []

        def capture(key, value):
            captured.append(value)
            return {"success": True, "stored_as": key, "validated": False}
        with _SECRET_LOCK:
            saver, callbacks.save_env_value_secure = callbacks.save_env_value_secure, capture
            try:
                callback(params.get("env_var") or "", params.get("prompt") or "", params.get("metadata"))
            finally:
                callbacks.save_env_value_secure = saver
        return captured[0] if captured else ""

    # -- result --------------------------------------------------------------------------------
    def _user_row(self, user_message):
        staged = self._pending_cli_user_message
        if isinstance(staged, dict):
            return staged
        return {"role": "user", "content": user_message}

    def _failed(self, history, user_message, error, *, failure_reason=None, retryable=False):
        """``failure_reason="gateway_connection"``: the CLI shows ``error`` as written instead of
        classifying a lost gateway as a model failure ("Run /model to switch...")."""
        out = {**_RESULT_DEFAULTS, "final_response": "", "messages": history + [self._user_row(user_message)],
               "failed": True, "completed": False, "error": error, "session_id": self.session_id}
        if failure_reason:
            out.update(failure_reason=failure_reason, failure_retryable=retryable)
        return out

    def _result(self, history, user_message, complete, turn):
        receipt = (complete or {}).get("_receipt") or {}
        if receipt:  # settled from it after a replay gap
            complete = {"outcome": receipt.get("outcome")}
        elif turn.admission:
            try:
                receipt = self.connection.call("prompt.receipt", session_id=turn.session_id, abandon=self._lost(turn),
                                               admission_id=turn.admission, include_result=True) or {}
            except GatewayClientError as exc:
                logger.debug("prompt.receipt failed: %s", exc)
        stored = dict(receipt.get("result") or {})
        usage = dict(receipt.get("usage") or {})
        status = (complete or {}).get("status") or (complete or {}).get("outcome")
        out = {**_RESULT_DEFAULTS, **stored}
        if "final_response" not in stored:
            out["final_response"] = (complete or {}).get("text") or (complete or {}).get("content") or ""
        if status in {"interrupted", "cancelled"} or (complete or {}).get("outcome") == "cancelled":
            out["interrupted"] = True
            out["completed"] = False
        elif status in {"error", "failed"} and not stored:
            out.update(failed=True, completed=False)
        suffix = stored.get("messages") if isinstance(stored.get("messages"), list) else None
        if suffix is not None and stored.get("_compressed"):
            out["messages"] = list(suffix)
        else:
            out["messages"] = history + [self._user_row(user_message)] + list(suffix or [])
        if not suffix and out.get("final_response") and not out.get("failed"):
            out["messages"].append({"role": "assistant", "content": out["final_response"]})
        out["session_id"] = self.session_id
        if not self._owns(turn):  # taken over during that read: the newer turn's state stays untouched
            return out
        self._session_messages = out["messages"]
        calls = stored.get("api_calls")
        # An interrupted turn's own count includes the attempt Ctrl-C cut off (the agent counts a call
        # before the request goes out). Only a response prices the ask: fall back to the gateway's
        # cumulative count, which moves per response, so an unanswered ask stays main's ``~`` estimate.
        counted = isinstance(calls, int) and not isinstance(calls, bool) and not out.get("interrupted")
        self._turn_api_calls = calls if counted else None
        self._apply_usage(stored, usage, turn)
        return out

    def _paint_owed(self, turn, out, complete, stream_callback):
        """After a replay gap, the reply once (``owed_reply``): a reuse this terminal showed whole stays
        the only copy; otherwise main's panel shows what streaming did not."""
        from hermes_cli.gateway_replay_gap import owed_reply
        reply = out.get("pre_transform_response") if out.get("response_transformed") else None
        reply = reply if isinstance(reply, str) else out.get("final_response")
        if out.get("failed") or not isinstance(reply, str) or not reply.strip():
            return
        reused = out.get("response_previewed") or out.get("response_reused") or complete.get("response_reused")
        out["response_previewed"] = bool(reused) and reply.strip() in turn.seen | {turn.shown.strip(), turn.spoken.strip()}
        for consumer, got in ((self.stream_delta_callback, turn.shown), (stream_callback, turn.spoken)):
            owed = owed_reply(got, reply) if consumer is not None and not out["response_previewed"] else ""
            if owed:  # each consumer completes what it got itself, never twice
                consumer(owed)

    def _anchor_display(self):
        """Main's bar shows the anchored ``context_tokens`` by re-deriving it from
        ``_turn_base_usage_anchor`` over ``_session_messages`` (``anchored_context_tokens``), and marks
        it ``~`` when the turn appended rows after its first response. Rebuild an anchor over this
        viewer's transcript that yields the gateway's figure: rows after the first API call of a
        multi-call turn are the estimated delta (main's ``~``); a single reply is covered by the anchor."""
        from agent.model_metadata import estimate_messages_tokens_rough
        from agent.usage_anchor import capture_usage_anchor
        messages = self._session_messages if isinstance(self._session_messages, list) else []
        tokens = getattr(self, "_context_tokens", 0) or 0
        if tokens <= 0 or not messages or self._generation is not None:
            self._turn_base_usage_anchor = None  # mid-turn: the bar falls back to last_prompt_tokens
            return
        # Main anchors on the turn's FIRST response, priced over the transcript through the turn's ask.
        # A turn with no completed call (interrupted before a response) keeps the previous anchor, so
        # its ask is estimated delta too.
        # The turn's own ask, never a later user-role row of the same turn (a mid-turn /steer row).
        ask = getattr(self, "_turn_ask_index", None)
        last_user = ask if isinstance(ask, int) and 0 <= ask < len(messages) and messages[ask].get("role") == "user" \
            else next((i for i in range(len(messages) - 1, -1, -1) if messages[i].get("role") == "user"),
                      len(messages) - 1)
        # Did this turn complete an API call? The turn result's own count when the gateway sent one: a
        # rebuilt gateway agent (/personality, /model, and /undo, /retry bumping the runtime generation)
        # restarts its session counters, so the cumulative delta can read "no call" for a turn that
        # answered; without that count a LOWER cumulative is a new agent that made calls, never "no call".
        turn_calls = self._turn_api_calls
        calls, turn_base = self.session_api_calls, self._turn_calls_base
        made_call = turn_calls > 0 if isinstance(turn_calls, int) else (calls > turn_base or 0 < calls < turn_base)
        base = last_user + 1 if made_call else last_user
        if base <= 0:
            self._turn_base_usage_anchor = None
            return
        delta = messages[base:]
        if delta and delta[0].get("role") == "assistant":
            delta = delta[1:]
        estimate = estimate_messages_tokens_rough(delta, charge_stale_thinking=False) if delta else 0
        self._turn_base_usage_anchor = capture_usage_anchor(max(1, tokens - estimate), 0, messages[:base])

    def _refresh_stats(self, turn=None):
        """``session.stats`` → the same attributes ``session.usage`` sets; False when unavailable."""
        try:
            stats = self.connection.call("session.stats", session_id=self.session_id, timeout=10,
                                         abandon=turn and self._lost(turn))
        except GatewayClientError as exc:
            logger.debug("session.stats unavailable: %s", exc)
            return False
        stats = stats.get("usage") if isinstance(stats, dict) and isinstance(stats.get("usage"), dict) else stats
        if not (isinstance(stats, dict) and isinstance(stats.get("api_calls"), int)):
            return False
        if turn is None or self._owns(turn):
            self._state().apply_usage(stats)
        return True

    def _apply_usage(self, stored, receipt_usage, turn):
        """Turn end: the session-cumulative ``session.stats`` (the same dict ``session.usage`` carries);
        a gateway without the verb leaves the per-turn receipt figures as before."""
        if self._refresh_stats(turn) or not self._owns(turn):
            return
        # Fallback: the classic-CLI result carries the gateway agent's session_* counters
        # (turn_finalizer _SESSION_TOKEN_KEYS); per-turn receipt usage only accumulates what is missing.
        usage = {key: stored[key] for key in _COUNTER_FROM_USAGE.values() if isinstance(stored.get(key), int)}
        for key in ("input_tokens", "output_tokens"):
            if key not in usage and isinstance(receipt_usage.get(key), int):
                attr = "session_" + key
                usage[key] = getattr(self, attr) + receipt_usage[key]
        usage.setdefault("prompt_tokens", usage.get("input_tokens"))
        usage.setdefault("completion_tokens", usage.get("output_tokens"))
        if isinstance(usage.get("prompt_tokens"), int) and isinstance(usage.get("completion_tokens"), int):
            usage.setdefault("total_tokens", usage["prompt_tokens"] + usage["completion_tokens"])
        if isinstance(stored.get("api_calls"), int):
            usage["api_calls"] = self.session_api_calls + stored["api_calls"]
        for key in ("last_prompt_tokens", "context_length", "model"):
            if receipt_usage.get(key) or stored.get(key):
                usage[key] = receipt_usage.get(key) or stored.get(key)
        self.apply_session_usage(usage)


def _effort_word(config):
    """The ``session.mutate reasoning`` effort for main's ``reasoning_config`` dict (None: no override)."""
    if not isinstance(config, dict):
        return None
    if config.get("enabled") is False:
        return "none"
    effort = config.get("effort")
    return effort if isinstance(effort, str) and effort else None


def _rate_limit_state(data):
    """``get_rate_limit_state()`` rebuilt from its serialized dict (None stays None)."""
    if not isinstance(data, dict):
        return None
    from agent.rate_limit_tracker import RateLimitBucket, RateLimitState
    buckets = {name: RateLimitBucket(**{k: v for k, v in (data.get(name) or {}).items()
                                        if k in {"limit", "remaining", "reset_seconds", "captured_at"}})
               for name in ("requests_min", "requests_hour", "tokens_min", "tokens_hour")}
    return RateLimitState(**buckets, captured_at=float(data.get("captured_at") or 0.0),
                          provider=str(data.get("provider") or ""))


def review_summary_line(payload):
    """Main's ``display.review.summary_cli`` line for the gateway's ``review.summary`` frame: the background
    review's "💾 Self-improvement review: ..." (``/refine``, the post-turn review), which the gateway
    publishes to a local viewer once the turn settled. None for an empty one."""
    text = str(payload.get("text") or "").strip() if isinstance(payload, dict) else ""
    return f"  {text}" if text else None


def _moa_handler(kind):
    def handler(agent, payload, stream_callback, starts):
        if agent.tool_progress_callback is None:
            return
        extra = {k: v for k, v in payload.items() if k not in {"label", "text", "name", "preview", "index", "count"}}
        # Main's _on_tool_progress reads moa_index / moa_count (the gateway frame uses the TUI keys).
        if payload.get("index") is not None:
            extra["moa_index"] = payload["index"]
        if payload.get("count") is not None:
            extra["moa_count"] = payload["count"]
        agent.tool_progress_callback(kind, payload.get("label") or payload.get("name"),
                                     payload.get("text") or payload.get("preview"), None, **extra)
    return handler


_SECRET_LOCK = threading.Lock()
_SERVER_REQUESTS = {
    "sudo": GatewayAgent._ask_sudo,
    "secret": GatewayAgent._ask_secret,
    "clarify": GatewayAgent._ask_clarify,
}

_HANDLERS = {
    "message.delta": GatewayAgent._on_delta,
    "message.interim": GatewayAgent._on_interim,
    "tool.generating": GatewayAgent._on_tool_generating,
    "tool.start": GatewayAgent._on_tool_start,
    "tool.complete": GatewayAgent._on_tool_complete,
    "todo.updated": GatewayAgent._on_todo,
    "thinking.delta": GatewayAgent._on_thinking,
    "reasoning.delta": GatewayAgent._on_reasoning,
    "reasoning.available": GatewayAgent._on_reasoning_available,
    "moa.reference": _moa_handler("moa.reference"),
    "moa.aggregating": _moa_handler("moa.aggregating"),
    "notification.show": GatewayAgent._on_notice,
    "notification.clear": GatewayAgent._on_notice_clear,
    "reaction": GatewayAgent._on_reaction,
    "agent.print": GatewayAgent._on_print,
    "review.summary": GatewayAgent._on_review_summary,
    "approval.request": GatewayAgent._on_approval,
    "clarify.request": GatewayAgent._on_clarify,
}
