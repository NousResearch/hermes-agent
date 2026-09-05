"""Platform-agnostic sequential Bot Mode chain runner.

Each chain run gets a unique titled conversation in every target profile and
delivers turns through the same ``tui_gateway`` session RPC core used by the
Desktop.  The in-process runtime stays warm for the life of the CLI/gateway;
the established ``hermes -p <profile> chat`` subprocess path remains a
pre-admission fallback.  Both paths own provider/model resolution,
credentials, tools, memory, and session behavior; this module does not
implement a second inference client.
"""

from __future__ import annotations

import contextlib
import os
import itertools
import logging
import queue
import re
import signal
import subprocess
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Mapping, Optional, Protocol, Sequence

from hermes_cli.bot_profiles import BotProfile


BOT_CHAIN_USAGE = "Usage: $Bot1 [$Bot2 ...] <prompt>"
BOT_CHAIN_CONVERSATION_PREFIX = "Bot Chain "
_NICKNAME_RE = re.compile(r"\$([A-Za-z0-9][A-Za-z0-9_-]{0,63})(?=\s|$)")
logger = logging.getLogger(__name__)


class BotChainError(RuntimeError):
    """Base error surfaced to CLI/gateway users."""


class BotChainSyntaxError(BotChainError):
    """The message started with ``$`` but did not contain a valid chain."""


class BotChainCancelled(BotChainError):
    """The operator stopped an active chain."""


class BotChainRecoveryUnavailable(BotChainError):
    """Durable step state cannot be proven; retry admission, never replay blindly."""


class BotTopicBindingError(BotChainError):
    """A ``$Name`` Telegram topic requested a bot that cannot run.

    Raised only once ``chat_topic`` provably starts with ``$``: at that
    point the operator has explicitly bound the topic to a bot identity,
    so the message must NOT fall through to the default agent. The message
    is refused with this typed, user-visible error instead.
    """


class BotRuntimeUnavailable(BotChainError):
    """The primary RPC runtime failed before a bot turn was admitted.

    Only this exception activates the subprocess fallback.  A failure after
    ``prompt.submit`` may already have reached the model, so replaying it over
    another transport would risk a duplicate turn.
    """


class BotTurnError(BotChainError):
    """One bot failed; downstream bots were not started."""

    def __init__(self, bot_name: str, detail: str, *, reason: str = "unknown"):
        self.bot_name = bot_name
        self.detail = str(detail or "Unknown error").strip()
        self.reason = reason
        super().__init__(f"${bot_name} failed [{reason}]: {self.detail}")


@dataclass(frozen=True)
class BotChainRequest:
    names: tuple[str, ...]
    prompt: str


@dataclass(frozen=True)
class BotChainStep:
    profile: BotProfile
    input_text: str
    output: str


@dataclass(frozen=True)
class BotChainResult:
    prompt: str
    steps: tuple[BotChainStep, ...]

    @property
    def final_output(self) -> str:
        return self.steps[-1].output if self.steps else ""


def parse_bot_chain_message(text: str) -> Optional[BotChainRequest]:
    """Parse leading ``$Name`` tokens; return ``None`` for ordinary chat."""
    source = str(text or "").lstrip()
    if not source.startswith("$"):
        return None

    names: list[str] = []
    offset = 0
    while offset < len(source) and source[offset] == "$":
        match = _NICKNAME_RE.match(source, offset)
        if match is None:
            raise BotChainSyntaxError(BOT_CHAIN_USAGE)
        names.append(match.group(1))
        offset = match.end()
        while offset < len(source) and source[offset].isspace():
            offset += 1

    if not names:
        raise BotChainSyntaxError(BOT_CHAIN_USAGE)
    prompt = source[offset:].strip()
    if not prompt:
        raise BotChainSyntaxError(BOT_CHAIN_USAGE)
    return BotChainRequest(names=tuple(names), prompt=prompt)


def bind_topic_bot(
    request: Optional[BotChainRequest],
    bound_bot: Optional[str],
    text: str,
) -> Optional[BotChainRequest]:
    """Prepend a topic-bound bot to a chain request.

    A Telegram topic titled ``$Name`` (e.g. ``$writer``) is bound to that bot
    profile: plain messages become a single-bot chain for it, and explicit
    ``$Other`` tokens in the message are treated as additional invited bots
    (the bound bot stays first, deduplicated case-insensitively). Returns
    ``request`` unchanged when no bot is bound.
    """
    if not bound_bot:
        return request
    if request is None:
        return BotChainRequest(names=(bound_bot,), prompt=str(text or ""))
    names = list(request.names)
    if bound_bot.casefold() not in {name.casefold() for name in names}:
        names.insert(0, bound_bot)
    return BotChainRequest(names=tuple(names), prompt=request.prompt)


class BotChainControl:
    """Thread-safe cancellation target compatible with gateway/CLI interrupts."""

    _is_bot_chain_control = True
    _supports_active_turn_redirect = True

    def __init__(self, *, on_redirect: Optional[Callable[[str], None]] = None):
        self.cancel_event = threading.Event()
        self._interrupt_requested = False
        self._interrupt_message: Optional[str] = None
        self._active_children: list = []
        self._active_children_lock = threading.Lock()
        self._on_redirect = on_redirect
        self._last_activity = time.time()
        self.publication_guard = contextlib.nullcontext
        self.source_home: Optional[Path] = None

    @contextlib.contextmanager
    def guard_publication(self):
        from hermes_state_bot_chain import BotChainClaimLostError

        try:
            with self.publication_guard():
                if self.cancel_event.is_set():
                    raise BotChainCancelled("Bot chain stopped.")
                yield
        except BotChainClaimLostError as exc:
            self.cancel_event.set()
            raise BotChainCancelled(str(exc)) from exc

    def touch(self) -> None:
        self._last_activity = time.time()

    def interrupt(self, message: Optional[str] = None, **_kwargs) -> None:
        self._interrupt_message = message
        self._interrupt_requested = True
        self.touch()
        self.cancel_event.set()

    def hard_interrupt(self, message: Optional[str] = None, **kwargs) -> None:
        self.interrupt(message, **kwargs)

    def clear_interrupt(self) -> None:
        self._interrupt_requested = False
        self._interrupt_message = None
        self.cancel_event.clear()

    def redirect(self, message: str) -> bool:
        payload = str(message or "").strip()
        if not payload:
            return False
        if self._on_redirect is not None:
            self._on_redirect(payload)
        self.interrupt(payload)
        return True

    def get_activity_summary(self) -> dict:
        return {
            "last_activity_ts": self._last_activity,
            "last_activity_desc": "bot chain turn",
            "api_call_count": 0,
            "max_iterations": 0,
        }


class BotTurnExecutor(Protocol):
    def __call__(
        self,
        profile: BotProfile,
        prompt: str,
        control: BotChainControl,
        *,
        conversation_name: str,
    ) -> str: ...


class _SessionRPCRejected(RuntimeError):
    """A registered session RPC returned a JSON-RPC error envelope."""

    def __init__(
        self,
        method: str,
        code: int,
        message: str,
        data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.method = method
        self.code = code
        self.data = dict(data or {})
        super().__init__(message)


class _SessionRPCProtocolError(RuntimeError):
    """An installed in-process handler raised or returned an invalid shape."""


class _SessionRPCClient:
    """Small in-process adapter over the installed ``tui_gateway`` registry."""

    _REQUIRED_METHODS = frozenset(
        {
            "session.create",
            "session.compress",
            "prompt.submit",
            "session.status",
            "session.interrupt",
            "session.close",
        }
    )

    def __init__(self, server: Optional[ModuleType] = None) -> None:
        self._server = server
        self._load_lock = threading.Lock()
        self._ids = itertools.count(1)

    def server(self) -> ModuleType:
        if self._server is None:
            with self._load_lock:
                if self._server is None:
                    try:
                        from tui_gateway import server
                    except Exception as exc:
                        raise BotRuntimeUnavailable(
                            f"session RPC runtime could not start: {exc}"
                        ) from exc
                    self._server = server

        methods = getattr(self._server, "_methods", None)
        missing = sorted(
            method
            for method in self._REQUIRED_METHODS
            if not isinstance(methods, dict) or method not in methods
        )
        proof = getattr(self._server, "_IN_PROCESS_SINGLE_QUERY_PROOF", None)
        if missing or proof is None:
            detail = (
                f"missing methods: {', '.join(missing)}"
                if missing
                else "missing in-process unattended-turn proof"
            )
            raise BotRuntimeUnavailable(f"session RPC runtime is incompatible ({detail})")
        return self._server

    @property
    def single_query_proof(self) -> object:
        return self.server()._IN_PROCESS_SINGLE_QUERY_PROOF

    def call(
        self,
        method: str,
        params: dict[str, Any],
        *,
        transport: Any,
    ) -> dict[str, Any]:
        server = self.server()
        handler = server._methods.get(method)
        if handler is None:
            raise _SessionRPCProtocolError(f"session RPC method is unavailable: {method}")
        try:
            from tui_gateway.transport import bind_transport, reset_transport
        except Exception as exc:
            raise _SessionRPCProtocolError(
                f"session RPC transport could not load: {exc}"
            ) from exc

        token = bind_transport(transport)
        try:
            envelope = handler(f"bot-chain-{next(self._ids)}", params)
        except Exception as exc:
            raise _SessionRPCProtocolError(f"{method} raised: {exc}") from exc
        finally:
            reset_transport(token)

        error = envelope.get("error") if isinstance(envelope, dict) else None
        if isinstance(error, dict):
            data = error.get("data")
            raise _SessionRPCRejected(
                method,
                int(error.get("code") or 5000),
                str(error.get("message") or "gateway rejected the request"),
                data if isinstance(data, Mapping) else None,
            )
        result = envelope.get("result") if isinstance(envelope, dict) else None
        if not isinstance(result, dict):
            raise _SessionRPCProtocolError(f"{method} returned no result")
        return result


class _BotChainTransport:
    """Thread-safe event sink for one in-process bot session."""

    def __init__(self) -> None:
        self._events: queue.Queue[dict[str, Any]] = queue.Queue()
        self._closed = threading.Event()

    def write(self, obj: dict) -> bool:
        if self._closed.is_set():
            return False
        self._events.put(obj)
        return True

    def close(self) -> None:
        self._closed.set()

    def next_event(self, timeout: float) -> dict[str, Any]:
        return self._events.get(timeout=timeout)


@dataclass(frozen=True)
class _RpcTurnOutcome:
    status: str
    text: str
    detail: str = ""
    reason: str = "unknown"


class HermesSessionRpcTurnExecutor:
    """Run profile turns on Hermes' warm in-process session RPC runtime."""

    def __init__(self, rpc: Optional[_SessionRPCClient] = None) -> None:
        self._rpc = rpc or _SessionRPCClient()

    @staticmethod
    def _event(frame: Any, session_id: str) -> tuple[str, dict[str, Any]] | None:
        if not isinstance(frame, dict) or frame.get("method") != "event":
            return None
        params = frame.get("params")
        if not isinstance(params, dict) or params.get("session_id") != session_id:
            return None
        payload = params.get("payload")
        return str(params.get("type") or ""), payload if isinstance(payload, dict) else {}

    @staticmethod
    def _reason(machine_reason: Any, detail: str) -> str:
        from tools.bot_failure_reasons import classify_session_error

        return classify_session_error(machine_reason, detail)

    def _rejected_turn_error(
        self, profile: BotProfile, exc: _SessionRPCRejected
    ) -> BotTurnError:
        return BotTurnError(
            profile.name,
            str(exc),
            reason=self._reason(exc.data.get("reason"), str(exc)),
        )

    def _interrupt(
        self,
        session_id: str,
        profile: BotProfile,
        transport: _BotChainTransport,
    ) -> None:
        try:
            self._rpc.call(
                "session.interrupt",
                {"session_id": session_id, "profile": profile.name},
                transport=transport,
            )
        except Exception:
            logger.debug("bot-chain RPC interrupt failed", exc_info=True)

    def _wait_for_terminal(
        self,
        session_id: str,
        profile: BotProfile,
        control: BotChainControl,
        transport: _BotChainTransport,
    ) -> _RpcTurnOutcome:
        while True:
            if control.cancel_event.is_set():
                self._interrupt(session_id, profile, transport)
                raise BotChainCancelled("Bot chain stopped.")
            try:
                frame = transport.next_event(timeout=0.2)
            except queue.Empty:
                control.touch()
                continue
            event = self._event(frame, session_id)
            if event is None:
                continue
            event_type, payload = event
            if event_type != "message.complete":
                continue

            status = str(payload.get("status") or "complete").strip().lower()
            text = str(payload.get("text") or "").strip()
            if status == "complete":
                return _RpcTurnOutcome(status=status, text=text)
            if status == "interrupted":
                if control.cancel_event.is_set():
                    raise BotChainCancelled("Bot chain stopped.")
                return _RpcTurnOutcome(
                    status=status,
                    text=text,
                    detail=str(payload.get("error") or "Bot turn was interrupted."),
                    reason="cancelled",
                )

            detail = str(payload.get("error") or text or "Bot turn failed.").strip()
            machine_reason = payload.get("failure_reason")
            if not machine_reason and isinstance(payload.get("error_surface"), dict):
                machine_reason = payload["error_surface"].get("code")
            return _RpcTurnOutcome(
                status="error",
                text=text,
                detail=detail,
                reason=self._reason(machine_reason, detail),
            )

    def _wait_for_idle(
        self,
        session_id: str,
        profile: BotProfile,
        control: BotChainControl,
        transport: _BotChainTransport,
        *,
        timeout: float = 30.0,
    ) -> None:
        """Wait for the server's post-terminal turn teardown to finish.

        ``message.complete`` intentionally precedes ``session.running = False``
        and the end of the run thread so interactive clients can paint the
        answer immediately.  A bot retry cannot submit in that window: the
        busy-input path de-duplicates the identical prompt and would leave this
        caller waiting forever.
        """
        deadline = time.monotonic() + timeout
        while True:
            if control.cancel_event.is_set():
                self._interrupt(session_id, profile, transport)
                raise BotChainCancelled("Bot chain stopped.")
            try:
                status = self._rpc.call(
                    "session.status",
                    {"profile": profile.name, "session_id": session_id},
                    transport=transport,
                )
            except _SessionRPCRejected as exc:
                raise self._rejected_turn_error(profile, exc) from exc
            except _SessionRPCProtocolError as exc:
                raise BotTurnError(profile.name, str(exc)) from exc
            settled = status.get("turn_settled")
            if settled is True:
                return
            if settled is not False:
                raise BotTurnError(
                    profile.name,
                    "session.status returned no turn-settled state",
                )
            if time.monotonic() >= deadline:
                raise BotTurnError(
                    profile.name,
                    "Bot turn did not settle after its terminal response.",
                    reason="delivery_timeout",
                )
            time.sleep(0.05)

    def _submit_once(
        self,
        session_id: str,
        profile: BotProfile,
        prompt: str,
        control: BotChainControl,
        transport: _BotChainTransport,
    ) -> _RpcTurnOutcome:
        try:
            self._rpc.call(
                "prompt.submit",
                {
                    "profile": profile.name,
                    "session_id": session_id,
                    "text": prompt,
                    "source": "cli",
                },
                transport=transport,
            )
        except _SessionRPCRejected as exc:
            # A handler error envelope is returned before prompt admission.
            raise self._rejected_turn_error(profile, exc) from exc
        except _SessionRPCProtocolError as exc:
            # Handler execution may have crossed admission before raising.
            # Never replay this turn through the subprocess fallback.
            raise BotTurnError(profile.name, str(exc)) from exc
        outcome = self._wait_for_terminal(
            session_id, profile, control, transport
        )
        self._wait_for_idle(
            session_id, profile, control, transport
        )
        return outcome

    def __call__(
        self,
        profile: BotProfile,
        prompt: str,
        control: BotChainControl,
        *,
        conversation_name: str,
    ) -> str:
        from tools.bot_failure_reasons import (
            RETRY_COMPRESS_THEN_RESUME,
            RETRY_NONE,
            retry_action,
        )

        # Resolve and validate the registry before creating anything.  A
        # failure here is the only condition allowed to activate fallback.
        self._rpc.server()
        transport = _BotChainTransport()
        session_id = ""
        try:
            try:
                created = self._rpc.call(
                    "session.create",
                    {
                        "profile": profile.name,
                        "title": conversation_name,
                        "source": "cli",
                        "cwd": str(Path.home()),
                        "hidden": True,
                        "follow_profile_config": True,
                        "close_on_disconnect": False,
                        "_single_query_proof": self._rpc.single_query_proof,
                    },
                    transport=transport,
                )
            except _SessionRPCRejected as exc:
                raise BotTurnError(profile.name, str(exc)) from exc
            except _SessionRPCProtocolError as exc:
                raise BotRuntimeUnavailable(str(exc)) from exc

            session_id = str(created.get("session_id") or "").strip()
            if not session_id:
                raise BotRuntimeUnavailable("session.create returned no runtime session id")

            if control.cancel_event.is_set():
                raise BotChainCancelled("Bot chain stopped.")

            outcome = self._submit_once(
                session_id, profile, prompt, control, transport
            )
            action = retry_action(outcome.reason)
            if outcome.status != "complete" and action != RETRY_NONE:
                if action == RETRY_COMPRESS_THEN_RESUME:
                    try:
                        compression = self._rpc.call(
                            "session.compress",
                            {"profile": profile.name, "session_id": session_id},
                            transport=transport,
                        )
                    except Exception as exc:
                        raise BotTurnError(
                            profile.name,
                            f"{outcome.detail} Compression failed: {exc}",
                            reason=outcome.reason,
                        ) from exc
                    if compression.get("status") != "compressed":
                        summary = compression.get("summary")
                        detail = str(
                            compression.get("message")
                            or (
                                summary.get("message")
                                if isinstance(summary, Mapping)
                                else ""
                            )
                            or compression.get("status")
                            or "compression did not complete"
                        )
                        raise BotTurnError(
                            profile.name,
                            f"{outcome.detail} Compression did not complete: {detail}",
                            reason=outcome.reason,
                        )
                outcome = self._submit_once(
                    session_id, profile, prompt, control, transport
                )

            if outcome.status != "complete":
                raise BotTurnError(
                    profile.name,
                    outcome.detail or outcome.text or "Bot turn failed.",
                    reason=outcome.reason,
                )
            if not outcome.text:
                raise BotTurnError(
                    profile.name,
                    "Hermes returned an empty response.",
                )
            return outcome.text
        finally:
            if session_id:
                try:
                    self._rpc.call(
                        "session.close",
                        {"profile": profile.name, "session_id": session_id},
                        transport=transport,
                    )
                except Exception:
                    logger.warning(
                        "Failed to close bot-chain RPC session %s", session_id,
                        exc_info=True,
                    )
            transport.close()


def publish_bot_chain_history(
    profile: BotProfile,
    conversation_name: str,
    *,
    control: Optional[BotChainControl] = None,
) -> str:
    """Project one completed isolated chain turn into the bot's ``Bot Chat``.

    The chain keeps its unique hidden session for execution and diagnostics so
    it never contends with an open Desktop runtime.  Bot Mode, however, exposes
    exactly one user-facing forever-chat per profile.  After the isolated turn
    settles, copy its durable transcript into that canonical chat under the
    same cross-process turn lease normal Hermes turns use.

    When the profile has no canonical chat yet, promote the completed isolated
    session itself by renaming it.  That preserves the full first transcript
    without a duplicate write and makes the Bots row immediately resolvable by
    its exact-title registry key.
    """
    from hermes_state import SessionDB
    from hermes_state_registry import acquire, release_or_close
    from tools.bot_mode_probe import BOT_CHAT_TITLE
    from tools.bot_relay import acquire_turn_lock, turn_wait_seconds

    title = str(conversation_name or "").strip()
    if not title:
        raise ValueError("conversation_name cannot be empty")

    profile_home = Path(profile.path)
    root = (
        profile_home.parent.parent
        if profile_home.parent.name == "profiles"
        else profile_home
    )
    db = acquire(profile_home / "state.db")
    try:
        with acquire_turn_lock(root, profile.name):
            if control is not None and control.cancel_event.is_set():
                # The chain's claim was lost (or the turn was stopped) while
                # the model turn ran: the new receipt owner publishes — this
                # stale generation must not create/rename/write Bot Chat.
                raise BotChainCancelled("Bot chain stopped.")
            canonical = db.get_session_by_title(BOT_CHAT_TITLE)
            if canonical is not None:
                canonical_id = str(canonical["id"])
                canonical_tip = db.get_compression_tip(canonical_id) or canonical_id
                if _published_chain_output(db, canonical_id, title) is not None:
                    return canonical_tip
            source = db.get_session_by_title(title)
            if source is None:
                raise RuntimeError(
                    f"completed Bot Chain session {title!r} was not persisted"
                )
            source_root_id = str(source.get("id") or "")
            source_tip_id = db.get_compression_tip(source_root_id) or source_root_id

            if canonical is None:
                with control.guard_publication() if control is not None else contextlib.nullcontext():
                    if control is not None and control.cancel_event.is_set():
                        # Re-checked under the DB lock, immediately before the
                        # first mutation (receipt stamp + rename into Bot Chat).
                        raise BotChainCancelled("Bot chain stopped.")
                    # The rename retires the chain-titled session, so the exact
                    # chain identity must survive on the message rows themselves:
                    # recovery from Bot Chat is keyed by that receipt, never by
                    # prompt text.
                    stamped_rows = db.get_messages_as_conversation(
                        source_tip_id,
                        repair_alternation=False,
                        include_row_ids=True,
                    )
                    db.stamp_bot_chain_receipt(
                        [m.get("_row_id") for m in stamped_rows], title
                    )
                    try:
                        promoted = db.set_session_title(source_root_id, BOT_CHAT_TITLE)
                    except ValueError:
                        # Desktop may have persisted its lazy Bot Chat between the
                        # lookup and rename. Resolve that winner and copy below.
                        canonical = db.get_session_by_title(BOT_CHAT_TITLE)
                        if canonical is None:
                            raise
                    else:
                        if not promoted:
                            raise RuntimeError(
                                f"could not promote Bot Chain session {source_root_id}"
                            )
                        db.set_session_hidden(source_root_id, True)
                        return source_tip_id

            canonical_root_id = str(canonical.get("id") or "")
            if not canonical_root_id:
                raise RuntimeError("canonical Bot Chat has no session id")
            if canonical_root_id == source_root_id:
                with control.guard_publication() if control is not None else contextlib.nullcontext():
                    db.set_session_hidden(canonical_root_id, True)
                return source_tip_id
            if canonical.get("archived"):
                with control.guard_publication() if control is not None else contextlib.nullcontext():
                    if not db.unarchive_recoverable_session(canonical_root_id):
                        raise RuntimeError(
                            "canonical Bot Chat is deliberately archived; refusing "
                            "to override that user boundary"
                        )
                    canonical = db.get_session(canonical_root_id) or canonical

            source_messages = db.get_messages_as_conversation(
                source_tip_id,
                repair_alternation=False,
                include_row_ids=False,
            )
            if not source_messages:
                raise RuntimeError(
                    f"completed Bot Chain session {source_root_id} has no transcript"
                )
            copied_messages = []
            for message in source_messages:
                copied = dict(message)
                # These are identities of rows in the isolated source session,
                # not repair targets in Bot Chat. The persistence marker is
                # likewise source-local and must not suppress the fresh insert.
                copied.pop("_row_id", None)
                copied.pop("_db_persisted", None)
                # Chain-qualified receipt: recovery from Bot Chat may skip
                # re-execution only for this exact chain identity, never for
                # a mere prompt-text match.
                receipt_meta = copied.get("display_metadata")
                if not isinstance(receipt_meta, dict):
                    receipt_meta = {}
                copied["display_metadata"] = {
                    **receipt_meta,
                    SessionDB.BOT_CHAIN_RECEIPT_METADATA_KEY: {"chain": title},
                }
                copied_messages.append(copied)

            canonical_tip_id = (
                db.get_compression_tip(canonical_root_id) or canonical_root_id
            )
            holder = (
                f"pid={os.getpid()}:bot-chain-history={uuid.uuid4().hex}:"
                f"profile={profile.name}"
            )
            wait_seconds = turn_wait_seconds()
            acquired = db.acquire_session_turn_lease(
                canonical_tip_id,
                holder,
                ttl_seconds=max(30.0, wait_seconds + 5.0),
                wait_seconds=wait_seconds,
                poll_interval_seconds=0.1,
                should_abort=(
                    control.cancel_event.is_set if control is not None else None
                ),
            )
            if not acquired:
                raise RuntimeError(
                    "canonical Bot Chat stayed busy while publishing chain history"
                )
            try:
                with control.guard_publication() if control is not None else contextlib.nullcontext():
                    # A normal inbound delivery reopens the canonical conversation;
                    # the transcript projection must have the same lifecycle shape.
                    if control is not None and control.cancel_event.is_set():
                        # Claim lost while waiting on the canonical chat's turn
                        # lease: no stale append under the new owner's receipt.
                        raise BotChainCancelled("Bot chain stopped.")
                    if _published_chain_output(db, canonical_root_id, title) is not None:
                        return canonical_tip_id
                    db.reopen_session(canonical_tip_id)
                    db.append_messages_batch(
                        canonical_tip_id,
                        copied_messages,
                        turn_lease_holder=holder,
                        turn_lease_ttl_seconds=max(30.0, wait_seconds + 5.0),
                    )
                    db.set_session_hidden(canonical_root_id, True)
            finally:
                db.release_session_turn_lease(canonical_tip_id, holder)
            return canonical_tip_id
    finally:
        release_or_close(db)


class FallbackBotTurnExecutor:
    """Use subprocess delivery only when RPC failed before admission."""

    def __init__(
        self,
        primary: BotTurnExecutor,
        fallback: BotTurnExecutor,
        *,
        history_publisher: Optional[Callable[..., str]] = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.history_publisher = history_publisher

    def __call__(
        self,
        profile: BotProfile,
        prompt: str,
        control: BotChainControl,
        *,
        conversation_name: str,
    ) -> str:
        try:
            output = self.primary(
                profile,
                prompt,
                control,
                conversation_name=conversation_name,
            )
        except BotRuntimeUnavailable as exc:
            logger.warning(
                "Bot-chain session RPC unavailable before admission; using "
                "subprocess fallback: %s",
                exc,
            )
            output = self.fallback(
                profile,
                prompt,
                control,
                conversation_name=conversation_name,
            )
        self.publish_history(profile, conversation_name, control)
        return output

    def publish_history(self, profile, conversation_name, control):
        """Complete (or recover) the history projection without replaying inference."""
        if self.history_publisher is not None:
            if control.cancel_event.is_set():
                # The execution claim was lost mid-turn (or the chain was
                # stopped): the receipt's new owner publishes — never project
                # a stale generation's output into the canonical Bot Chat.
                raise BotChainCancelled("Bot chain stopped.")
            try:
                self.history_publisher(
                    profile,
                    conversation_name,
                    control=control,
                )
            except BotChainCancelled:
                raise
            except Exception as exc:
                raise BotChainRecoveryUnavailable(
                    f"${profile.name}'s turn is durable but its Bot Chat publication is pending."
                ) from exc


def _terminate_process(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGTERM)  # windows-footgun: ok — POSIX branch
        else:  # pragma: no cover - Windows
            proc.terminate()
        proc.wait(timeout=2)
        return
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        if os.name != "nt":
            os.killpg(proc.pid, signal.SIGKILL)  # windows-footgun: ok — POSIX branch
        else:  # pragma: no cover - Windows
            proc.kill()
    except OSError:
        pass


class HermesProfileTurnExecutor:
    """Run one profile turn through Hermes' established chat transport."""

    def _run_once(
        self,
        argv: list[str],
        control: BotChainControl,
    ) -> subprocess.CompletedProcess[str]:
        try:
            proc = subprocess.Popen(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                start_new_session=os.name != "nt",
            )
        except OSError as exc:
            raise BotTurnError(argv[2] if len(argv) > 2 else "?", str(exc)) from exc

        while True:
            if control.cancel_event.is_set():
                _terminate_process(proc)
                try:
                    proc.communicate(timeout=1)
                except subprocess.TimeoutExpired:
                    pass
                raise BotChainCancelled("Bot chain stopped.")
            try:
                stdout, stderr = proc.communicate(timeout=0.2)
                return subprocess.CompletedProcess(
                    argv, proc.returncode, stdout=stdout, stderr=stderr
                )
            except subprocess.TimeoutExpired:
                control.touch()

    def __call__(
        self,
        profile: BotProfile,
        prompt: str,
        control: BotChainControl,
        *,
        conversation_name: str,
    ) -> str:
        from tools.bot_failure_reasons import (
            RETRY_NONE,
            classify_agent_error,
            retry_action,
        )
        from tools.bot_relay import acquire_turn_lock, local_delivery_command

        fd, query_file = tempfile.mkstemp(
            prefix=f"hermes-chain-{profile.name}-", suffix=".txt"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(prompt)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.chmod(query_file, 0o600)
            except OSError:
                pass

            argv = local_delivery_command(
                profile.name,
                query_file,
                conversation_name=conversation_name,
            )
            root = (
                profile.path.parent.parent
                if profile.path.parent.name == "profiles"
                else profile.path
            )
            try:
                with acquire_turn_lock(root, profile.name):
                    result = self._run_once(argv, control)
                    if result.returncode != 0:
                        detail = (result.stderr or result.stdout or "").strip()
                        reason = classify_agent_error(detail)
                        if retry_action(reason) != RETRY_NONE:
                            result = self._run_once(argv, control)
                            if result.returncode != 0:
                                detail = (result.stderr or result.stdout or "").strip()
                                reason = classify_agent_error(detail)
                        if result.returncode != 0:
                            raise BotTurnError(
                                profile.name,
                                detail[-2000:] or f"Hermes exited with code {result.returncode}",
                                reason=reason,
                            )
            except BotChainCancelled:
                raise
            except BotTurnError:
                raise
            except Exception as exc:
                raise BotTurnError(profile.name, str(exc)) from exc

            output = (result.stdout or "").strip()
            if not output:
                raise BotTurnError(profile.name, "Hermes returned an empty response.")
            return output
        finally:
            try:
                Path(query_file).unlink(missing_ok=True)
            except OSError:
                pass


_DEFAULT_TURN_EXECUTOR: Optional[BotTurnExecutor] = None
_DEFAULT_TURN_EXECUTOR_LOCK = threading.Lock()


def default_bot_turn_executor() -> BotTurnExecutor:
    """Return the process-wide warm RPC executor with subprocess fallback."""
    global _DEFAULT_TURN_EXECUTOR
    if _DEFAULT_TURN_EXECUTOR is None:
        with _DEFAULT_TURN_EXECUTOR_LOCK:
            if _DEFAULT_TURN_EXECUTOR is None:
                _DEFAULT_TURN_EXECUTOR = FallbackBotTurnExecutor(
                    HermesSessionRpcTurnExecutor(),
                    HermesProfileTurnExecutor(),
                    history_publisher=publish_bot_chain_history,
                )
    return _DEFAULT_TURN_EXECUTOR


def build_handoff_prompt(
    original_prompt: str,
    previous_profile: BotProfile,
    previous_output: str,
) -> str:
    """Create the next turn while retaining the original user intent."""
    return (
        "Continue the user's task using the previous bot's work.\n\n"
        f"Original user request:\n{original_prompt}\n\n"
        f"Previous bot (${previous_profile.name}) output:\n{previous_output}"
    )


def _last_assistant_text(messages: Sequence[Mapping[str, Any]]) -> Optional[str]:
    if messages:
        message = messages[-1]
        if (message.get("role") == "assistant" and not message.get("tool_calls")
                and not message.get("_compressed_summary")):
            content = str(message.get("content") or "")
            if content.strip():
                return content
    return None


def _receipt_stamped_output(
    messages: Sequence[Mapping[str, Any]], conversation_name: str
) -> Optional[str]:
    """Latest assistant reply carrying this exact chain identity's receipt."""
    from hermes_state import SessionDB

    receipt_key = SessionDB.BOT_CHAIN_RECEIPT_METADATA_KEY
    for message in reversed(messages):
        if message.get("role") != "assistant" or message.get("tool_calls") or message.get("_compressed_summary"):
            continue
        metadata = message.get("display_metadata")
        if not isinstance(metadata, dict):
            continue
        receipt = metadata.get(receipt_key)
        if not isinstance(receipt, dict):
            continue
        if receipt.get("chain") != conversation_name:
            continue
        content = str(message.get("content") or "")
        if content.strip():
            return content
    return None


def _published_chain_output(db, canonical_id: str, conversation_name: str) -> Optional[str]:
    """Publication receipts survive rotation, compaction and transcript edits.

    Historical rows prove an executed side effect even when no longer part of
    the model context. Never feed these audit rows back into the conversation.
    """
    # ponytail: scan retained lineage; index receipts if long Bot Chats make this expensive.
    for session_id in reversed(db.get_compression_chain(canonical_id)):
        output = _receipt_stamped_output(
            db.get_messages(session_id, include_inactive=True), conversation_name
        )
        if output is not None:
            return output
    return None


def recover_durable_step_output(
    profile: BotProfile, conversation_name: str
) -> Optional[str]:
    """Recover one chain step's output from the profile's durable state.

    The chain identity (``conversation_name``) is the idempotency key: a
    crash after a step's model turn persisted but before the next side
    effect must resume AFTER that step, never re-execute it. Two durable
    shapes prove that identity: the isolated chain session that kept its
    title (the append-publish path), and message rows stamped with the
    chain receipt inside the canonical Bot Chat (the rename-publish path
    retires the chain title, so the receipt rides on the promoted/copied
    rows). Recovery NEVER matches on prompt text: a brand-new chain that
    repeats an older prompt must execute its own model turn. Returns
    ``None`` only when no durable proof exists. An unreadable store is not
    evidence that execution never happened, so errors defer the delivery.
    """
    from hermes_state import SessionDB
    from tools.bot_mode_probe import BOT_CHAT_TITLE

    db_path = Path(profile.path) / "state.db"
    db = None
    try:
        try:
            db_path.lstat()
        except FileNotFoundError:
            return None
        db = SessionDB(db_path, read_only=True)
        source = db.get_session_by_title(conversation_name)
        if source is not None and source.get("id"):
            source_id = str(source["id"])
            source_tip = db.get_compression_tip(source_id) or source_id
            recovered = _last_assistant_text(
                db.get_messages(source_tip)
            )
            if recovered is not None:
                return recovered
        canonical = db.get_session_by_title(BOT_CHAT_TITLE)
        if canonical is not None and canonical.get("id"):
            return _published_chain_output(db, str(canonical["id"]), conversation_name)
        return None
    except Exception as exc:
        logger.warning(
            "bot-chain durable recovery probe failed for $%s",
            profile.name,
            exc_info=True,
        )
        raise BotChainRecoveryUnavailable(
            f"Cannot verify ${profile.name}'s durable chain history; no turn was replayed."
        ) from exc
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


class BotChainRunner:
    """Run an already-resolved ordered profile chain."""

    def __init__(self, turn_executor: Optional[BotTurnExecutor] = None):
        self.turn_executor = turn_executor or default_bot_turn_executor()

    def run(
        self,
        profiles: Sequence[BotProfile],
        prompt: str,
        *,
        control: Optional[BotChainControl] = None,
        on_step: Optional[Callable[[BotChainStep, int, int], None]] = None,
        conversation_name: Optional[str] = None,
    ) -> BotChainResult:
        ordered = list(profiles)
        if not ordered:
            raise BotChainSyntaxError(BOT_CHAIN_USAGE)
        original_prompt = str(prompt or "").strip()
        if not original_prompt:
            raise BotChainSyntaxError(BOT_CHAIN_USAGE)
        control = control or BotChainControl()
        if conversation_name is None:
            conversation_name = f"{BOT_CHAIN_CONVERSATION_PREFIX}{uuid.uuid4().hex}"
        elif not str(conversation_name).startswith(BOT_CHAIN_CONVERSATION_PREFIX):
            raise ValueError(
                "conversation_name must keep the "
                f"{BOT_CHAIN_CONVERSATION_PREFIX!r} prefix"
            )

        steps: list[BotChainStep] = []
        seen_profiles: set[str] = set()
        next_input = original_prompt
        total = len(ordered)
        for index, profile in enumerate(ordered):
            # Keep existing receipts valid for the first occurrence of a profile;
            # later occurrences are distinct turns, not redeliveries of that turn.
            profile_key = profile.name.casefold()
            step_name = (
                f"{conversation_name} / step {index + 1}"
                if profile_key in seen_profiles else conversation_name
            )
            seen_profiles.add(profile_key)
            if control.cancel_event.is_set():
                raise BotChainCancelled("Bot chain stopped.")
            with control.guard_publication():
                pass  # Recheck receipt authority before starting the next step.
            if control.source_home is not None:
                from hermes_cli.bot_profiles import check_bot_chain_profile_access

                try:
                    check_bot_chain_profile_access(profile, control.source_home)
                except (OSError, ValueError) as exc:
                    raise BotTurnError(profile.name, str(exc), reason="missing_config") from exc
            control.touch()
            # Idempotent recipient processing (#100758): when this exact chain
            # identity already has a durable completed turn for this profile,
            # recover its output instead of re-executing — a redelivery after
            # a crash resumes after the last durable side effect, and the
            # canonical-history projection is never published twice. Recovery
            # is keyed by the chain identity only (session title or the
            # stamped chain receipt), never by prompt text.
            recovered = recover_durable_step_output(
                profile, step_name
            )
            if recovered is not None:
                output = recovered
                publish = getattr(self.turn_executor, "publish_history", None)
                if publish is not None:
                    publish(profile, step_name, control)
            else:
                output = self.turn_executor(
                    profile,
                    next_input,
                    control,
                    conversation_name=step_name,
                )
            with control.guard_publication():
                pass  # Do not emit a stale generation's result before heartbeat catches up.
            step = BotChainStep(profile=profile, input_text=next_input, output=output)
            steps.append(step)
            if on_step is not None:
                on_step(step, index, total)
            if index + 1 < total:
                next_input = build_handoff_prompt(
                    original_prompt,
                    profile,
                    output,
                )

        return BotChainResult(prompt=original_prompt, steps=tuple(steps))


def format_bot_chain_step(step: BotChainStep, *, final: bool = False) -> str:
    suffix = " (final)" if final else ""
    return f"${step.profile.name}{suffix}:\n{step.output}"


def format_bot_chain_result(result: BotChainResult) -> str:
    last = len(result.steps) - 1
    return "\n\n".join(
        format_bot_chain_step(step, final=index == last)
        for index, step in enumerate(result.steps)
    )
