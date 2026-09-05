"""Telegram bot-chain routing, durable admission, execution and settlement."""

import asyncio
import logging
import time
import uuid
from typing import Optional

from gateway.config import Platform
from gateway.platforms.base import MessageType
from gateway.session import SessionSource

logger = logging.getLogger("gateway.run")


class GatewayBotChainMixin:
    async def _prepare_bot_chain_request(self, event, source):
        """None for ordinary traffic, a request for a chain, or a routing refusal."""
        if (
            source.platform != Platform.TELEGRAM or event.message_type != MessageType.TEXT
            or getattr(event, "internal", False)
        ):
            return None
        from agent.bot_chain import (
            BotChainSyntaxError, BotTopicBindingError, bind_topic_bot, parse_bot_chain_message,
        )

        text = event.text or ""
        request = syntax_error = None
        try:
            request = parse_bot_chain_message(text)
        except BotChainSyntaxError as exc:
            syntax_error = str(exc)
        try:
            bound = await asyncio.to_thread(self._telegram_topic_bound_bot, source)
        except BotTopicBindingError as exc:
            return str(exc)
        if bound is not None:
            return bind_topic_bot(request, bound, text)
        return syntax_error if syntax_error is not None else request

    async def _admit_bot_chain_turn(self, session_entry, message_id):
        """Return (chain identity, claim token), a refusal string, or None to stand down."""
        from agent.bot_chain import BOT_CHAIN_CONVERSATION_PREFIX

        # The admission receipt is the authority whenever one exists;
        # the legacy transcript-row dedupe below applies ONLY to events
        # processed before the admission table existed. A released or
        # resumable receipt must reach the state machine below — stopping
        # at the transcript row would leave it admitted (non-terminal)
        # forever even though its durable outcome is recoverable.
        receipt = None
        try:
            get_receipt = getattr(
                self.async_session_store, "get_bot_chain_delivery", None
            )
            if get_receipt is not None:
                receipt = await get_receipt(
                    session_entry.session_id, message_id
                )
        except Exception:
            # Probe failure must not block: admission below decides.
            receipt = None
            logger.debug(
                "bot-chain receipt probe failed (message_id=%s)",
                message_id,
                exc_info=True,
            )
        if (
            receipt is None
            and await self.async_session_store.has_platform_message_id(
                session_entry.session_id, message_id
            )
        ):
            logger.info(
                "Skipping duplicate bot-chain turn (message_id=%s) in session %s",
                message_id,
                session_entry.session_id,
            )
            return None
        conversation_name = (
            f"{BOT_CHAIN_CONVERSATION_PREFIX}{uuid.uuid4().hex}"
        )
        try:
            admission = await self.async_session_store.admit_bot_chain_delivery(
                session_entry.session_id, message_id, conversation_name
            )
        except Exception:
            # Fail closed: executing without a durable receipt would
            # reopen the duplicate-execution window on redelivery.
            logger.warning(
                "Bot-chain admission write failed (message_id=%s, session %s)",
                message_id,
                session_entry.session_id,
                exc_info=True,
            )
            return (
                "Bot chain is temporarily unavailable: the delivery "
                "receipt could not be persisted. Please resend the message."
            )
        if admission != "admitted":
            # "settled": already ran to completion. "running": a live
            # owner holds the execution claim. Either way — zero turns.
            logger.info(
                "Skipping %s bot-chain turn (message_id=%s) in session %s",
                admission,
                message_id,
                session_entry.session_id,
            )
            return None
        # The receipt, not this process, owns the chain identity: a
        # resumed admission (crash before execution) reuses the name
        # bound at the FIRST delivery, so durable step recovery continues
        # the same chain sessions instead of minting a parallel chain.
        # The identity MUST come from a successful receipt read-back —
        # executing under a freshly minted random name after a failed
        # read would fork the chain away from its durable receipt and
        # invite a duplicate model turn on redelivery. Stand down until
        # the platform redelivers instead.
        receipt = None
        try:
            get_receipt = getattr(
                self.async_session_store, "get_bot_chain_delivery", None
            )
            if get_receipt is not None:
                receipt = await get_receipt(session_entry.session_id, message_id)
        except Exception:
            receipt = None
            logger.warning(
                "bot-chain receipt read-back failed (message_id=%s)",
                message_id,
                exc_info=True,
            )
        chain_name = (
            str(receipt.get("chain_name") or "").strip()
            if isinstance(receipt, dict)
            else ""
        )
        if not chain_name:
            logger.warning(
                "Bot-chain receipt unreadable after admission "
                "(message_id=%s, session %s); standing down until "
                "redelivery — no model turn without the durable chain "
                "identity",
                message_id,
                session_entry.session_id,
            )
            return None
        conversation_name = chain_name
        # The atomic execution claim is the durable boundary before any
        # side effect: a lost or failed claim means zero model turns —
        # the platform redelivery will resume the still-admitted receipt.
        try:
            claim_token = await self.async_session_store.mark_bot_chain_delivery_running(
                session_entry.session_id, message_id
            )
        except Exception:
            logger.warning(
                "Bot-chain execution claim failed (message_id=%s, session %s); "
                "executing zero turns",
                message_id,
                session_entry.session_id,
                exc_info=True,
            )
            return None
        if not claim_token:
            logger.info(
                "Bot-chain execution claim lost to a concurrent attempt "
                "(message_id=%s) in session %s; standing down",
                message_id,
                session_entry.session_id,
            )
            return None
        return conversation_name, claim_token

    def _telegram_topic_bound_bot(self, source: SessionSource) -> Optional[str]:
        """Bot profile bound to a Telegram topic titled ``$Name`` (or None).

        A Telegram DM/forum topic whose title starts with ``$`` (e.g.
        ``$writer``) is bound to that Bot Mode profile: every text message in the
        topic runs as a bot chain headed by the bound bot and the default
        profile does not answer there.

        Returns None only when this is not a ``$Name`` bot topic at all. Once
        the topic title begins with ``$`` the route identity is explicit, so
        an unresolved/disabled/unconfigured/unreadable bot fails CLOSED by
        raising ``BotTopicBindingError`` — the caller turns that into a
        user-visible routing refusal and never routes the message to the
        default agent.
        """
        if source.platform != Platform.TELEGRAM:
            return None
        topic_name = str(getattr(source, "chat_topic", "") or "").strip()
        if not topic_name.startswith("$"):
            return None

        from agent.bot_chain import BotTopicBindingError

        candidate = topic_name[1:].strip()
        if not candidate:
            raise BotTopicBindingError(
                f"Topic '{topic_name}' looks like a bot topic but names no bot. "
                "Rename it to '$<bot-name>' or drop the '$' prefix. The "
                "default agent does not answer in bound bot topics."
            )
        try:
            from hermes_cli.bot_profiles import get_bot_profile

            profile = get_bot_profile(candidate)
        except FileNotFoundError:
            raise BotTopicBindingError(
                f"Bot topic '{topic_name}' is bound to '${candidate}', but no "
                f"profile with that name exists. Create it with: "
                f"hermes bots create {candidate} ... — the default agent does "
                "not answer in bound bot topics."
            ) from None
        except ValueError:
            raise BotTopicBindingError(
                f"Bot topic '{topic_name}' is bound to '${candidate}', which "
                "is not a valid profile name. Rename the topic to "
                "'$<bot-name>' (lowercase letters, digits, '-' or '_'). The "
                "default agent does not answer in bound bot topics."
            ) from None
        except Exception as exc:
            logger.debug(
                "topic-bot-binding: could not read profile for topic '%s'",
                topic_name,
                exc_info=True,
            )
            raise BotTopicBindingError(
                f"Bot topic '{topic_name}' is bound to '${candidate}', but "
                f"that profile cannot be read right now ({exc}). The default "
                "agent does not answer in bound bot topics."
            ) from None
        if not profile.enabled:
            raise BotTopicBindingError(
                f"Bot topic '{topic_name}' is bound to '${profile.name}', but "
                "that bot is disabled (or its profile metadata is unreadable, "
                "which fails closed). Enable it with: "
                f"hermes bots enable {profile.name} — the default agent does "
                "not answer in bound bot topics."
            )
        if not profile.provider or not profile.model:
            raise BotTopicBindingError(
                f"Bot topic '{topic_name}' is bound to '${profile.name}', but "
                "that bot has no model/provider configured. Run: "
                f"hermes bots configure {profile.name} "
                "--provider <provider> --model <model> — the default agent "
                "does not answer in bound bot topics."
            )
        return profile.name

    async def _bot_chain_claim_heartbeat(
        self,
        session_id: str,
        message_id: str,
        owner_token: str,
        control,
        claim_state: dict,
    ) -> None:
        """Renew the bot-chain execution claim's lease while the chain runs.

        The lease (SessionDB.BOT_CHAIN_CLAIM_LEASE_SECONDS) is what bounds a
        dead runtime generation's claim; this loop keeps a healthy execution
        authoritative across multi-minute model turns. When the claim is
        lost — a renewal comes back False (reclaimed), or two renewals in a
        row fail to land — the chain is cancelled and the caller stands
        down: another delivery now owns the receipt and will settle it. The
        two-miss rule trips at ~2/3 of the lease window, strictly BEFORE a
        redelivery may reclaim the still-live-looking receipt.
        """
        from hermes_state import SessionDB

        lease_seconds = SessionDB.BOT_CHAIN_CLAIM_LEASE_SECONDS
        interval = max(0.1, lease_seconds / 3.0)
        missed_renewals = 0
        while True:
            await asyncio.sleep(interval)
            renew = getattr(
                self.async_session_store, "renew_bot_chain_delivery_claim", None
            )
            if renew is None:
                return
            try:
                renewed = await renew(session_id, message_id, owner_token)
            except Exception:
                logger.warning(
                    "Bot-chain claim renewal failed (message_id=%s, session %s)",
                    message_id,
                    session_id,
                    exc_info=True,
                )
                renewed = None
            if renewed:
                missed_renewals = 0
                continue
            if renewed is False:
                # Authoritative answer from the receipt: this claim was
                # reclaimed — stand down immediately.
                missed_renewals = 2
            else:
                missed_renewals += 1
            if missed_renewals < 2:
                continue
            logger.warning(
                "Bot-chain claim lost mid-execution (message_id=%s, "
                "session %s); cancelling the chain and standing down",
                message_id,
                session_id,
            )
            claim_state["lost"] = True
            control.cancel_event.set()
            return

    async def _handle_bot_chain_turn(
        self,
        event,
        session_entry,
        routing_key: str,
        request,
    ) -> Optional[str]:
        """Run and persist one ``$Bot`` chain under the gateway turn lease.

        Idempotent recipient processing for at-least-once platform delivery:
        a durable admission receipt (SessionDB ``bot_chain_deliveries``) is
        written BEFORE any model execution and decides whether this platform
        message may start a chain. A redelivery after a crash or a failed
        transcript write finds the receipt and never re-executes the chain.
        """
        from agent.bot_chain import (
            BotChainCancelled,
            BotChainControl,
            BotChainError,
            BotChainRunner,
            BotChainRecoveryUnavailable,
            format_bot_chain_result,
        )
        from hermes_cli.bot_profiles import resolve_bot_chain

        session_id = session_entry.session_id
        message_id = str(event.message_id) if event.message_id else None
        conversation_name = claim_token = None
        if message_id:
            admitted = await self._admit_bot_chain_turn(session_entry, message_id)
            if not isinstance(admitted, tuple):
                return admitted
            conversation_name, claim_token = admitted

        control = BotChainControl()
        from hermes_constants import get_hermes_home

        control.source_home = get_hermes_home()
        if message_id and claim_token:
            from functools import partial

            control.publication_guard = partial(
                self.session_store.bot_chain_publication_guard,
                session_id, message_id, claim_token,
            )
        claim_state: dict = {"lost": False}
        heartbeat = None
        if message_id and claim_token:
            # The claim is a bounded lease: renew it while the chain runs so
            # a healthy execution is never reclaimed, while a dead runtime
            # generation becomes reclaimable when its lease lapses. Losing
            # the claim mid-execution cancels the chain (handled below).
            heartbeat = asyncio.create_task(
                self._bot_chain_claim_heartbeat(
                    session_id,
                    message_id,
                    claim_token,
                    control,
                    claim_state,
                )
            )
        chain_state = self._session_state(routing_key)
        chain_state.turn.agent = control
        chain_state.turn.started_ts = time.time()

        response: str
        cancelled = False
        outcome = "completed"
        try:
            profiles = await asyncio.to_thread(resolve_bot_chain, request.names)
            # A bare ``await asyncio.to_thread(...)`` survives cancellation of
            # THIS handler: the worker thread would keep executing an orphaned
            # generation while the heartbeat finally-block below had already
            # stopped renewing the claim — the lease then lapses and a
            # redelivery starts a parallel execution of the same receipt.
            # Run the chain as a shielded task instead: an external cancel
            # signals the chain's cancel_event, keeps the claim heartbeat
            # alive until the worker thread has ACTUALLY finished, and only
            # then unwinds this handler.
            worker_task = asyncio.create_task(
                asyncio.to_thread(
                    BotChainRunner().run,
                    profiles,
                    request.prompt,
                    control=control,
                    conversation_name=conversation_name,
                )
            )
            try:
                result = await asyncio.shield(worker_task)
            except asyncio.CancelledError:
                control.cancel_event.set()
                while not worker_task.done():
                    try:
                        await asyncio.shield(worker_task)
                    except asyncio.CancelledError:
                        # Repeated external cancels: keep signalling and keep
                        # waiting — the heartbeat must outlive the worker.
                        control.cancel_event.set()
                    except Exception:
                        pass
                raise
            response = format_bot_chain_result(result)
        except BotChainRecoveryUnavailable as exc:
            if message_id and claim_token:
                try:
                    await self.async_session_store.release_bot_chain_delivery_claim(session_id, message_id, claim_token)
                except Exception:
                    logger.warning("Could not release deferred bot-chain recovery claim", exc_info=True)
            return f"Bot chain recovery deferred: {exc} Retry this delivery when its state store is available."
        except BotChainCancelled:
            cancelled = True
            outcome = "cancelled"
            response = "Bot chain stopped."
        except (BotChainError, FileNotFoundError, OSError, ValueError) as exc:
            outcome = "failed"
            response = f"Bot chain failed: {exc}"
        except Exception as exc:
            outcome = "failed"
            logger.exception("Unexpected bot-chain failure for session %s", routing_key)
            response = f"Bot chain failed: {exc}"
        finally:
            if heartbeat is not None:
                heartbeat.cancel()
                try:
                    await heartbeat
                except asyncio.CancelledError:
                    pass
                except Exception:
                    logger.debug(
                        "bot-chain claim heartbeat failed on shutdown",
                        exc_info=True,
                    )

        timestamp = time.time()
        if message_id and claim_state["lost"]:
            # Our claim was reclaimed mid-execution (the heartbeat cancelled
            # the chain): another delivery now owns the receipt and will
            # settle and answer it. The stale owner stands down fully — no
            # transcript copy, no user-facing response.
            logger.warning(
                "Bot-chain claim was reclaimed mid-execution (message_id=%s, "
                "session %s); standing down without transcript writes",
                message_id,
                session_id,
            )
            return None
        if message_id:
            # Settlement lands BEFORE the transcript rows and is never
            # swallowed silently: it is the receipt that forbids a second
            # execution when a later write fails and the platform redelivers.
            # Settlement is scoped to our claim's owner_token; a False
            # result means the claim was reclaimed (lease lapsed) and the
            # new owner is now responsible for the terminal write.
            settled = False
            for _attempt in range(2):  # one immediate retry for a transient wedge
                try:
                    settle_result = await self.async_session_store.settle_bot_chain_delivery(
                        session_id,
                        message_id,
                        outcome=outcome,
                        detail=response[:500],
                        owner_token=claim_token,
                    )
                    if settle_result is False:
                        logger.info(
                            "Bot-chain settlement refused: claim no longer ours "
                            "(message_id=%s, session %s); standing down",
                            message_id,
                            session_id,
                        )
                        claim_state["lost"] = True
                    settled = True
                    break
                except Exception:
                    logger.warning(
                        "Bot-chain settlement write failed (message_id=%s, "
                        "session %s, attempt %d/2)",
                        message_id,
                        session_id,
                        _attempt + 1,
                        exc_info=True,
                    )
            if not settled:
                # The receipt is still "running" under OUR claim; left
                # as-is, redeliveries would stand down until the lease
                # lapses. Release our own claim so a redelivery resumes the
                # admission and recovers every durably persisted step
                # instead of re-executing blindly. The release is scoped by
                # owner_token, so a concurrent or newer claim is never
                # revoked; if the release write also fails, the claim holds
                # only until its lease expires — reclaim then resumes the
                # admission.
                try:
                    released = await self.async_session_store.release_bot_chain_delivery_claim(
                        session_id, message_id, claim_token
                    )
                except Exception:
                    released = False
                    logger.warning(
                        "Bot-chain claim release failed (message_id=%s, "
                        "session %s); the claim holds until its lease "
                        "expires, then a redelivery reclaims it",
                        message_id,
                        session_id,
                        exc_info=True,
                    )
                if released:
                    logger.warning(
                        "Bot-chain settlement could not be persisted "
                        "(message_id=%s, session %s); released the execution "
                        "claim so a redelivery resumes the admission and "
                        "recovers every durably persisted step",
                        message_id,
                        session_id,
                    )
            if claim_state["lost"]:
                # The settlement write landed nowhere we own: the receipt was
                # reclaimed between the last heartbeat and settlement. The
                # new owner settles and answers — stand down without
                # transcript writes.
                return None
        user_entry = {
            "role": "user",
            "content": event.text or "",
            "timestamp": timestamp,
        }
        if message_id:
            user_entry["message_id"] = message_id
        try:
            await self.async_session_store.append_to_transcript(
                session_id,
                user_entry,
            )
            await self.async_session_store.append_to_transcript(
                session_id,
                {
                    "role": "assistant",
                    "content": response,
                    "timestamp": timestamp,
                },
            )
            await self.async_session_store.update_session(
                session_entry.session_key,
                touch_activity=not bool(getattr(event, "internal", False)),
            )
        except Exception:
            logger.warning(
                "Failed to persist bot-chain exchange for session %s",
                session_id,
                exc_info=True,
            )

        # /stop already sends the operator its own acknowledgement. Suppress a
        # second delivery from the unwinding chain task.
        return None if cancelled else response
