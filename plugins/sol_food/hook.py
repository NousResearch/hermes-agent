"""Sol food transport hook — the German-specific behavior, kept OUTSIDE
the generic Hermes core behind the authenticated topic-plugin seams.

The generic core knows nothing about food. This hook binds to the ``sol``
route profile (owner chat + General topic id 1) and owns:

- pre-download gating of Sol photos (album rejection, ceilings,
  single-active-proposal discipline) — all before any byte downloads;
- the ``sf1:`` opaque callback namespace (restart-safe: tokens resolve
  against the durable store, not process memory);
- proposal presentation state (owner-private, expiring, noncanonical);
- the one reviewed Health writer: the v3 commit client with frozen
  envelopes and exact-retry reconciliation.

Ordinary Sol conversation stays ordinary: ``on_message`` never consumes
text and never writes. Candidate creation happens only through the
explicit ``propose_*`` entry points (wired to an explicit "Log meal"
affordance by the Sol session layer), and no Health write ever happens
before an explicit Confirm callback.

Privacy: no food text, labels, candidate payloads, photo paths, callback
data, or credentials are ever logged — stable reason codes only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable, Dict, List, Mapping, Optional, Sequence

from gateway.topic_hooks import (
    HookDecision,
    MediaDescriptor,
    ReplyFn,
    TopicPluginHook,
)
from gateway.topic_routing import GENERAL_TOPIC_THREAD_ID, RouteOrigin, TopicRoute
from plugins.sol_food.cache import FoodImageCache
from plugins.sol_food.health_client import (
    FrozenEnvelope,
    HealthClientError,
    HealthFoodClient,
    build_commit_envelope,
)
from plugins.sol_food.legacy_guard import assert_legacy_helper_disabled
from plugins.sol_food.limits import (
    FOOD_CACHE_FILE_MODE,
    FOOD_CAPTION_MAX_CHARS,
    FOOD_PARSE_DEADLINE_SECONDS,
    FOOD_PARSE_MAX_ATTEMPTS,
    FOOD_TEXT_MAX_CHARS,
)
from plugins.sol_food.media_guard import MediaRejected, predownload_check
from plugins.sol_food.proposal import (
    ACTION_CANCEL,
    ACTION_CONFIRM,
    ACTION_EDIT,
    Candidate,
    ProposalError,
    ProposalState,
    render_display,
)
from plugins.sol_food.store import CallbackOutcome, FoodProposalStore
from plugins.sol_food.tokens import parse_token

logger = logging.getLogger(__name__)

__all__ = ["SolFoodHook", "ParserFn"]

#: Injected parser: (text, image_path) -> candidate list. The model/policy
#: side lives with the session layer; the transport enforces only the
#: deadline/attempt ceilings and never lets parsing commit anything.
ParserFn = Callable[[Optional[str], Optional[Path]], Awaitable[Sequence[Candidate]]]

# Stable value-free reason codes.
REASON_PARSE_TIMEOUT = "sol_food_parse_timeout"
REASON_PARSE_FAILED = "sol_food_parse_failed"
REASON_TEXT_TOO_LONG = "sol_food_text_too_long"
REASON_CAPTION_TOO_LONG = "sol_food_caption_too_long"
REASON_COMMIT_PENDING = "sol_food_commit_pending"

# Bounded user-visible strings (no user content interpolated).
_MSG_ALBUM = "Please send one representative image."
_MSG_ACTIVE = "A food proposal is already open — finish or cancel it first."
_MSG_EXPIRED = "These options expired. Send the meal again to log it."
_MSG_STALE = "These buttons are from an older version — use the newest message."
_MSG_RESOLVED = "This has already been resolved."
_MSG_DENIED = "That action isn't valid here."
_MSG_CANCELLED = "Cancelled — nothing was logged."
_MSG_EDIT = "Send a corrected description and I'll update the options."
_MSG_SELECT_FIRST = "Pick one option first, then confirm."
_MSG_COMMIT_PENDING = "Confirmed — logging is pending and will retry safely."
_MSG_MEDIA_REJECTED = "That image can't be used (format or size). JPEG/PNG/WebP up to 10 MiB."


def _occurred_at_now(clock: Callable[[], float]) -> str:
    stamp = datetime.fromtimestamp(clock(), tz=timezone.utc)
    return stamp.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class _EnvelopeStore:
    """Durable frozen-envelope storage (owner-private, 0600, atomic).

    An envelope is written BEFORE its first send and deleted only after a
    fully verified acknowledgement, so a crash between send and ack
    reconciles by replaying identical bytes.
    """

    def __init__(self, state_dir: Path) -> None:
        self._dir = Path(state_dir) / "food-envelopes"
        self._dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self._dir, 0o700)

    def _path(self, proposal_id: str) -> Path:
        return self._dir / f"{proposal_id}.json"

    def save(self, proposal_id: str, envelope: FrozenEnvelope, update_id: int) -> None:
        payload = dict(envelope.to_json())
        payload["update_id"] = int(update_id)
        blob = json.dumps(payload, sort_keys=True).encode("utf-8")
        path = self._path(proposal_id)
        tmp = path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, FOOD_CACHE_FILE_MODE)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(blob)
                handle.flush()
                os.fsync(handle.fileno())
        except OSError:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise
        os.replace(tmp, path)
        os.chmod(path, FOOD_CACHE_FILE_MODE)

    def load(self, proposal_id: str) -> Optional[tuple[FrozenEnvelope, int]]:
        path = self._path(proposal_id)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            update_id = int(data.pop("update_id"))
            return FrozenEnvelope.from_json(data), update_id
        except (OSError, ValueError, KeyError, HealthClientError):
            return None

    def delete(self, proposal_id: str) -> None:
        try:
            os.unlink(self._path(proposal_id))
        except OSError:
            pass

    def pending_ids(self) -> List[str]:
        return [p.stem for p in self._dir.glob("*.json")]


class SolFoodHook(TopicPluginHook):
    profile = "sol"
    callback_prefixes = ("sf1:",)

    def __init__(
        self,
        *,
        state_dir: Path,
        hermes_home: Path,
        health_client: HealthFoodClient,
        parser: Optional[ParserFn] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        # Single-writer guard: refuse to exist while the legacy
        # append-style helper is present (raises LegacyHelperPresent).
        # Caller contract: ``hermes_home`` MUST be the ACTIVE profile's
        # Hermes home (the directory the running gateway resolves) — the
        # guard inspects exactly that tree, so pointing it anywhere else
        # would void the single-writer property.
        assert_legacy_helper_disabled(hermes_home)
        self._store = FoodProposalStore(Path(state_dir), clock=clock)
        self._cache = FoodImageCache(Path(state_dir))
        self._envelopes = _EnvelopeStore(Path(state_dir))
        self._health = health_client
        self._parser = parser
        self._clock = clock
        # proposal_id -> cached image id (transient; store survives restart,
        # images are re-derived or already terminal-deleted).
        self._proposal_images: Dict[str, str] = {}
        self._cache.sweep_orphans()

    # ── generic seam: messages ──────────────────────────────────────────
    async def on_message(
        self, route: TopicRoute, origin: RouteOrigin, text: str, reply: ReplyFn
    ) -> HookDecision:
        # Ordinary Sol conversation remains ordinary conversation; food
        # candidate creation is explicit (propose_from_text) and never
        # implicit on every message.
        return HookDecision.CONTINUE

    # ── generic seam: media pre-download ────────────────────────────────
    async def on_media_pre_download(
        self,
        route: TopicRoute,
        origin: RouteOrigin,
        media: MediaDescriptor,
        reply: ReplyFn,
    ) -> HookDecision:
        if origin.thread_id != GENERAL_TOPIC_THREAD_ID:
            # Foreign thread should be unreachable post-route; fail closed.
            return HookDecision.DENY
        if media.kind != "photo":
            return HookDecision.CONTINUE
        # While an owner+Sol photo/text proposal is non-terminal, any later
        # photo update fails BEFORE download; origins are never merged.
        if await self._store.has_active_proposal(
            origin.owner_chat_id, origin.thread_id
        ):
            await reply(_MSG_ACTIVE)
            return HookDecision.DENY
        try:
            predownload_check(
                file_size=media.file_size,
                width=media.width,
                height=media.height,
                media_group_id=media.media_group_id,
            )
        except MediaRejected as rejected:
            logger.info("[sol-food] %s", rejected.reason_code)
            await reply(
                _MSG_ALBUM if rejected.reason_code == "food_media_album_rejected" else _MSG_MEDIA_REJECTED
            )
            return HookDecision.DENY
        if media.caption_length > FOOD_CAPTION_MAX_CHARS:
            logger.info("[sol-food] %s", REASON_CAPTION_TOO_LONG)
            # Over-limit caption: the photo is not treated as food input;
            # generic pipeline may continue (conversation), no proposal.
            return HookDecision.CONTINUE
        return HookDecision.CONTINUE

    # ── explicit proposal entry points (invoked by the Sol session layer) ──
    async def propose_from_text(
        self, origin: RouteOrigin, text: str, reply: ReplyFn
    ) -> Optional[str]:
        """Explicit 'Log meal' action for a text description.

        Returns the proposal id, or None when refused (the user was told
        why via bounded strings). Never writes to Health.
        """
        if len(text) > FOOD_TEXT_MAX_CHARS:
            logger.info("[sol-food] %s", REASON_TEXT_TOO_LONG)
            await reply("That description is too long to log — please shorten it.")
            return None
        return await self._propose(origin, reply, text=text, image_path=None)

    async def propose_from_photo(
        self,
        origin: RouteOrigin,
        image_bytes: bytes,
        caption: Optional[str],
        reply: ReplyFn,
    ) -> Optional[str]:
        """Explicit photo-meal proposal from already-downloaded bytes.

        Bytes are re-validated (magic/dimensions/size), cached under the
        0700/0600 food cache, parsed under the deadline, and deleted on
        every terminal outcome (plus the 60-second backstop).
        """
        from plugins.sol_food.media_guard import validate_image_bytes

        if caption is not None and len(caption) > FOOD_CAPTION_MAX_CHARS:
            logger.info("[sol-food] %s", REASON_CAPTION_TOO_LONG)
            await reply("That caption is too long — please shorten it.")
            return None
        try:
            validate_image_bytes(image_bytes)
        except MediaRejected as rejected:
            logger.info("[sol-food] %s", rejected.reason_code)
            await reply(_MSG_MEDIA_REJECTED)
            return None
        image_id = self._cache.store(bytes(image_bytes))
        # Backstop first: even if this task dies mid-parse, the image is
        # gone no later than 60 s after this terminal-parsing window.
        self._cache.arm_terminal_backstop(image_id)
        try:
            proposal_id = await self._propose(
                origin,
                reply,
                text=caption,
                image_path=self._cache.path_for(image_id),
            )
        finally:
            # Candidate extraction succeeded or failed — either way the
            # image has served its purpose. Delete on every path.
            self._cache.delete(image_id)
        return proposal_id

    async def _propose(
        self,
        origin: RouteOrigin,
        reply: ReplyFn,
        *,
        text: Optional[str],
        image_path: Optional[Path],
    ) -> Optional[str]:
        if self._parser is None:
            logger.info("[sol-food] %s", REASON_PARSE_FAILED)
            return None
        candidates = await self._run_parser(text, image_path)
        if candidates is None:
            await reply("I couldn't read that as a meal — nothing was logged.")
            return None
        try:
            proposal, tokens = await self._store.create(
                bot_id=origin.bot_id,
                owner_chat_id=origin.owner_chat_id,
                thread_id=origin.thread_id,
                origin_update_id=origin.update_id,
                origin_message_id=origin.message_id,
                candidates=list(candidates),
            )
        except ProposalError as err:
            logger.info("[sol-food] %s", err.reason_code)
            await reply(_MSG_ACTIVE)
            return None
        await reply(render_display(proposal.candidates))
        return proposal.proposal_id

    async def _run_parser(
        self, text: Optional[str], image_path: Optional[Path]
    ) -> Optional[Sequence[Candidate]]:
        """Enforce the 90-second total deadline and 2-attempt ceiling."""
        deadline = self._clock() + FOOD_PARSE_DEADLINE_SECONDS
        loop_clock = self._clock
        for attempt in range(FOOD_PARSE_MAX_ATTEMPTS):
            remaining = deadline - loop_clock()
            if remaining <= 0:
                logger.info("[sol-food] %s", REASON_PARSE_TIMEOUT)
                return None
            try:
                assert self._parser is not None
                result = await asyncio.wait_for(
                    self._parser(text, image_path), timeout=remaining
                )
            except asyncio.TimeoutError:
                logger.info("[sol-food] %s", REASON_PARSE_TIMEOUT)
                return None
            except Exception:
                logger.info("[sol-food] %s", REASON_PARSE_FAILED)
                continue
            if result:
                return result
        return None

    # ── presentation binding ────────────────────────────────────────────
    async def bind_presentation(self, proposal_id: str, message_id: int) -> None:
        await self._store.record_presentation(proposal_id, message_id)

    def tokens_for_keyboard(
        self, tokens: Mapping[str, str]
    ) -> List[tuple[str, str]]:
        """Order actions for an inline keyboard: choices, then verbs."""
        ordered: List[tuple[str, str]] = []
        for action in sorted(a for a in tokens if a.startswith("choice:")):
            ordered.append((action, tokens[action]))
        for action in (ACTION_CONFIRM, ACTION_EDIT, ACTION_CANCEL):
            if action in tokens:
                ordered.append((action, tokens[action]))
        return ordered

    # ── generic seam: callbacks ─────────────────────────────────────────
    async def on_callback(
        self,
        route: TopicRoute,
        origin: RouteOrigin,
        callback_data: str,
        reply: ReplyFn,
    ) -> HookDecision:
        token = parse_token(callback_data)
        if token is None:
            logger.info("[sol-food] food_callback_malformed")
            return HookDecision.DENY
        callback_identity = dict(
            token=token,
            update_id=origin.update_id,
            bot_id=origin.bot_id,
            chat_id=origin.owner_chat_id,
            thread_id=origin.thread_id,
            callback_message_id=origin.message_id,
        )
        # Crash-safety ordering: if this callback would consume a Confirm,
        # FREEZE the commit envelope durably BEFORE the token is consumed.
        # A crash between freeze and consume discards the unsent envelope
        # (proposal stays confirmable); a crash after consume always
        # completes via reconcile() with the exact frozen bytes. The peek
        # is advisory only — resolve_callback stays the authority.
        pre_froze_pid = None
        peek = await self._store.peek_callback(**callback_identity)
        if (
            peek.kind == CallbackOutcome.KIND_ACTION
            and peek.action == ACTION_CONFIRM
            and peek.proposal_id is not None
        ):
            pre_froze_pid = await self._prefreeze_confirm_envelope(
                peek.proposal_id, origin.update_id
            )
        outcome = await self._store.resolve_callback(**callback_identity)
        if outcome.kind == CallbackOutcome.KIND_REPLAY:
            # Exact transport replay: original effect stands; surface the
            # original acknowledgement, no second effect.
            await reply(_MSG_RESOLVED)
            return HookDecision.CONSUME
        if outcome.kind == CallbackOutcome.KIND_DENIED:
            if pre_froze_pid is not None:
                await self._discard_stale_envelope(pre_froze_pid)
            logger.info("[sol-food] %s", outcome.reason_code)
            await reply(self._denial_message(outcome.reason_code))
            return HookDecision.CONSUME
        assert outcome.action is not None and outcome.proposal_id is not None
        await self._perform_action(
            outcome.action, outcome.proposal_id, origin, reply
        )
        return HookDecision.CONSUME

    @staticmethod
    def _denial_message(reason_code: Optional[str]) -> str:
        mapping = {
            "food_store_expired": _MSG_EXPIRED,
            "food_store_stale_version": _MSG_STALE,
            "food_store_already_resolved": _MSG_RESOLVED,
        }
        return mapping.get(reason_code or "", _MSG_DENIED)

    async def _prefreeze_confirm_envelope(
        self, proposal_id: str, update_id: int
    ) -> Optional[str]:
        """Durably freeze (fsync) the commit envelope for a confirmable
        single-candidate proposal. Returns the proposal id when an
        envelope is (now) frozen, else None. Reuses an already-frozen
        envelope byte-for-byte (mutation identity never changes)."""
        proposal = await self._store.get(proposal_id)
        if proposal is None or proposal.state is not ProposalState.PENDING:
            return None
        if len(proposal.candidates) != 1:
            return None
        if self._envelopes.load(proposal_id) is None:
            envelope = build_commit_envelope(
                operation="create",
                occurred_at=_occurred_at_now(self._clock),
                items=[dict(i) for i in proposal.candidates[0].items],
                expected_revision=0,
            )
            self._envelopes.save(proposal_id, envelope, update_id)
        return proposal_id

    async def _discard_stale_envelope(self, proposal_id: str) -> None:
        """Drop a pre-frozen envelope whose Confirm never (durably)
        consumed — e.g. an interleaved Cancel won the race. Never drops
        an envelope owned by a consumed Confirm awaiting its receipt."""
        proposal = await self._store.get(proposal_id)
        if proposal is None:
            return
        if proposal.awaiting_commit or proposal.state is ProposalState.CONFIRMED:
            return
        self._envelopes.delete(proposal_id)

    async def _perform_action(
        self,
        action: str,
        proposal_id: str,
        origin: RouteOrigin,
        reply: ReplyFn,
    ) -> None:
        if action == ACTION_CANCEL:
            await self._store.mark_terminal(proposal_id, ProposalState.CANCELLED)
            self._envelopes.delete(proposal_id)
            self._drop_image(proposal_id)
            await reply(_MSG_CANCELLED)
            return
        if action == ACTION_EDIT:
            # Transport keeps the proposal pending; the session layer
            # collects the correction and calls edit_new_version().
            await reply(_MSG_EDIT)
            return
        if action.startswith("choice:"):
            index = int(action.split(":", 1)[1])
            proposal = await self._store.get(proposal_id)
            if proposal is None or index >= len(proposal.candidates):
                await reply(_MSG_DENIED)
                return
            chosen = proposal.candidates[index]
            new_proposal, tokens = await self._store.edit_new_version(
                proposal_id, [chosen]
            )
            await reply(render_display(new_proposal.candidates))
            return
        if action == ACTION_CONFIRM:
            await self._confirm(proposal_id, origin, reply)
            return
        await reply(_MSG_DENIED)

    # ── commit path ─────────────────────────────────────────────────────
    async def _confirm(
        self, proposal_id: str, origin: RouteOrigin, reply: ReplyFn
    ) -> None:
        proposal = await self._store.get(proposal_id)
        if proposal is None:
            await reply(_MSG_DENIED)
            return
        if len(proposal.candidates) != 1:
            # Explicit selection first; un-consume is impossible (tokens
            # are one-shot) so re-present a fresh version for selection.
            new_proposal, _tokens = await self._store.edit_new_version(
                proposal_id, list(proposal.candidates)
            )
            await reply(_MSG_SELECT_FIRST)
            return
        candidate = proposal.candidates[0]
        # Freeze the exact Health envelope durably BEFORE first send.
        existing = self._envelopes.load(proposal_id)
        if existing is not None:
            envelope, _update = existing
        else:
            envelope = build_commit_envelope(
                operation="create",
                occurred_at=_occurred_at_now(self._clock),
                items=[dict(i) for i in candidate.items],
                expected_revision=0,
            )
            self._envelopes.save(proposal_id, envelope, origin.update_id)
        await self._send_frozen(proposal_id, envelope, origin.update_id, reply)

    async def _send_frozen(
        self,
        proposal_id: str,
        envelope: FrozenEnvelope,
        update_id: int,
        reply: ReplyFn,
    ) -> None:
        try:
            verified = await asyncio.get_running_loop().run_in_executor(
                None, self._health.commit, envelope
            )
        except HealthClientError as err:
            logger.info("[sol-food] %s", err.reason_code)
            if err.retryable:
                # Envelope stays frozen; reconcile() retries identical bytes.
                await reply(_MSG_COMMIT_PENDING)
            else:
                await self._store.mark_terminal(proposal_id, ProposalState.CANCELLED)
                self._envelopes.delete(proposal_id)
                self._drop_image(proposal_id)
                await reply("Logging failed — nothing was recorded.")
            return
        # Acknowledge only after receipt + readback verification passed.
        await self._store.record_receipt(
            proposal_id, update_id, verified.receipt_sha256
        )
        self._envelopes.delete(proposal_id)
        self._drop_image(proposal_id)
        await reply(f"Logged ✅ (receipt {verified.receipt_sha256[:12]})")

    async def reconcile(self, reply: Optional[ReplyFn] = None) -> int:
        """Retry every frozen-but-unacknowledged envelope (startup/sweep).

        Returns the number of envelopes that verified. Retries reuse the
        exact frozen bytes; a replayed commit returns the original
        receipt with one effect.
        """
        verified_count = 0
        for proposal_id in self._envelopes.pending_ids():
            loaded = self._envelopes.load(proposal_id)
            if loaded is None:
                self._envelopes.delete(proposal_id)
                continue
            proposal = await self._store.get(proposal_id)
            if proposal is not None:
                if proposal.state in (ProposalState.CANCELLED, ProposalState.EXPIRED):
                    # A terminal non-confirmed proposal owns no commit.
                    self._envelopes.delete(proposal_id)
                    continue
                if (
                    proposal.state is ProposalState.PENDING
                    and not proposal.awaiting_commit
                ):
                    # Crash between freeze and consume: the Confirm never
                    # durably consumed and these bytes were never sent.
                    # Restore-to-confirmable — discard; a fresh Confirm
                    # freezes a new envelope.
                    self._envelopes.delete(proposal_id)
                    continue
            envelope, update_id = loaded
            try:
                verified = await asyncio.get_running_loop().run_in_executor(
                    None, self._health.commit, envelope
                )
            except HealthClientError as err:
                logger.info("[sol-food] %s", err.reason_code)
                continue
            try:
                await self._store.record_receipt(
                    proposal_id, update_id, verified.receipt_sha256
                )
            except ProposalError:
                # Orphaned envelope (proposal pruned): the canonical commit
                # verified; nothing transport-side is left to bind. Do not
                # abort the loop for the remaining envelopes.
                logger.info("[sol-food] sol_food_reconcile_orphan")
            self._envelopes.delete(proposal_id)
            verified_count += 1
            if reply is not None:
                await reply(f"Logged ✅ (receipt {verified.receipt_sha256[:12]})")
        return verified_count

    def _drop_image(self, proposal_id: str) -> None:
        image_id = self._proposal_images.pop(proposal_id, None)
        if image_id is not None:
            self._cache.delete(image_id)
