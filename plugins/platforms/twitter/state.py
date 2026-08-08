from __future__ import annotations

import json
import logging
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from hermes_constants import get_hermes_home
from utils import atomic_json_write

from .oauth import _twitter_scoped_lock, active_profile_key

logger = logging.getLogger(__name__)
STATE_VERSION = 1


def state_path() -> Path:
    return get_hermes_home() / "twitter" / "state.json"


@asynccontextmanager
async def twitter_state_lock(profile_key: str, account_id: str):
    async with _twitter_scoped_lock("twitter-state", profile_key, account_id):
        yield


@dataclass
class TwitterState:
    mention_since_id: str = ""
    dm_last_seen_event_id: str = ""
    dm_delivery_uncertain: bool = False
    known_dm_conversations: set[str] = field(default_factory=set)
    opted_out_dm_conversations: set[str] = field(default_factory=set)
    _seen: OrderedDict[str, None] = field(default_factory=OrderedDict)
    _branches: OrderedDict[str, str] = field(default_factory=OrderedDict)
    _public_reply_routes: OrderedDict[str, str] = field(default_factory=OrderedDict)
    _dm_reply_routes: OrderedDict[str, str] = field(default_factory=OrderedDict)
    _reply_reservations: OrderedDict[str, str] = field(default_factory=OrderedDict)
    max_seen: int = 2000
    max_branches: int = 2000

    @classmethod
    def load(
        cls, *, max_seen: int = 2000, max_branches: int = 2000
    ) -> "TwitterState":
        path = state_path()
        try:
            payload = json.loads(path.read_text())
            if payload.get("version") != STATE_VERSION:
                raise ValueError("unsupported state version")
            state = cls(
                mention_since_id=str(payload.get("mention_since_id") or ""),
                dm_last_seen_event_id=str(
                    payload.get("dm_last_seen_event_id")
                    or payload.get("dm_since_id")
                    or ""
                ),
                dm_delivery_uncertain=bool(payload.get("dm_delivery_uncertain", False)),
                max_seen=max_seen,
                max_branches=max_branches,
            )
            for item in payload.get("seen_ids") or []:
                state._seen[str(item)] = None
            for item in payload.get("bot_post_anchors") or []:
                if isinstance(item, list) and len(item) == 2:
                    state._branches[str(item[0])] = str(item[1])
            state.known_dm_conversations = {
                str(item) for item in payload.get("known_dm_conversations") or []
            }
            state.opted_out_dm_conversations = {
                str(item)
                for item in payload.get("opted_out_dm_conversations") or []
            }
            for item in payload.get("public_reply_routes") or []:
                if isinstance(item, list) and len(item) == 2:
                    state._public_reply_routes[str(item[0])] = str(item[1])
            for item in payload.get("dm_reply_routes") or []:
                if isinstance(item, list) and len(item) == 2:
                    state._dm_reply_routes[str(item[0])] = str(item[1])
            for item in payload.get("reply_reservations") or []:
                if (
                    isinstance(item, list)
                    and len(item) == 2
                    and item[1]
                    in {"reserved", "writing", "uncertain", "reconciled", "replied"}
                ):
                    state._reply_reservations[str(item[0])] = str(item[1])
            state._trim()
            return state
        except FileNotFoundError:
            return cls(max_seen=max_seen, max_branches=max_branches)
        except (AttributeError, OSError, TypeError, ValueError, json.JSONDecodeError):
            cls._quarantine(path)
            return cls(max_seen=max_seen, max_branches=max_branches)

    @staticmethod
    def _quarantine(path: Path) -> None:
        if not path.exists():
            return
        destination = path.with_name(f"{path.name}.corrupt-{time.time_ns()}")
        try:
            path.replace(destination)
            logger.warning("Quarantined corrupt Twitter state at %s", destination)
        except OSError:
            logger.warning("Ignoring unreadable Twitter state at %s", path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": STATE_VERSION,
            "mention_since_id": self.mention_since_id,
            "dm_last_seen_event_id": self.dm_last_seen_event_id,
            "dm_delivery_uncertain": self.dm_delivery_uncertain,
            "known_dm_conversations": sorted(self.known_dm_conversations),
            "opted_out_dm_conversations": sorted(
                self.opted_out_dm_conversations
            ),
            "seen_ids": list(self._seen),
            "bot_post_anchors": [list(item) for item in self._branches.items()],
            "public_reply_routes": [
                list(item) for item in self._public_reply_routes.items()
            ],
            "dm_reply_routes": [list(item) for item in self._dm_reply_routes.items()],
            "reply_reservations": [
                list(item) for item in self._reply_reservations.items()
            ],
        }

    def save(self) -> None:
        atomic_json_write(state_path(), self.to_dict())

    def seen(self, event_id: str) -> bool:
        return str(event_id) in self._seen

    def mark_seen(self, event_id: str) -> None:
        key = str(event_id)
        self._seen.pop(key, None)
        self._seen[key] = None
        self._trim()

    def map_bot_post(self, post_id: str, anchor_id: str) -> None:
        key = str(post_id)
        self._branches.pop(key, None)
        self._branches[key] = str(anchor_id)
        self._trim()

    def record_public_interaction(self, interaction_id: str, route: str) -> None:
        key = str(interaction_id)
        if self._make_route_capacity("tweet", key):
            self._public_reply_routes.setdefault(key, str(route))

    def reserve_public_reply(self, interaction_id: str, route: str) -> bool:
        key = str(interaction_id)
        reservation = self._reservation_key("tweet", key)
        if (
            self._public_reply_routes.get(key) != str(route)
            or reservation in self._reply_reservations
            or not self._make_reservation_capacity()
        ):
            return False
        self._reply_reservations[reservation] = "reserved"
        self._trim()
        return True

    def release_public_reply(self, interaction_id: str) -> None:
        self._release_reply("tweet", interaction_id)

    def mark_public_reply_uncertain(self, interaction_id: str) -> None:
        self._mark_reply_uncertain("tweet", interaction_id)

    def confirm_public_reply(
        self, interaction_id: str, post_id: str, anchor_id: str
    ) -> None:
        key = str(interaction_id)
        reservation = self._reservation_key("tweet", key)
        if self._reply_reservations.get(reservation) != "reserved":
            raise ValueError("Twitter reply reservation is not active")
        self._reply_reservations[reservation] = "replied"
        self.map_bot_post(post_id, anchor_id)

    def record_dm_inbound(
        self, conversation_id: str, interaction_id: str = ""
    ) -> None:
        conversation = str(conversation_id)
        self.known_dm_conversations.add(conversation)
        if interaction_id:
            interaction = str(interaction_id)
            if self._make_route_capacity("dm", interaction):
                self._dm_reply_routes.setdefault(interaction, conversation)

    def reserve_dm_reply(self, interaction_id: str, conversation_id: str) -> bool:
        key = str(interaction_id)
        reservation = self._reservation_key("dm", key)
        if (
            self._dm_reply_routes.get(key) != str(conversation_id)
            or reservation in self._reply_reservations
            or not self.can_send_dm(conversation_id)
            or not self._make_reservation_capacity()
        ):
            return False
        self._reply_reservations[reservation] = "reserved"
        self._trim()
        return True

    def begin_dm_reply(self, interaction_id: str, conversation_id: str) -> bool:
        reservation = self._reservation_key("dm", interaction_id)
        if (
            self._reply_reservations.get(reservation) != "reserved"
            or not self.can_send_dm(conversation_id)
        ):
            return False
        self._reply_reservations[reservation] = "writing"
        return True

    def release_dm_reply(self, interaction_id: str) -> None:
        self._release_reply("dm", interaction_id)

    def mark_dm_reply_uncertain(self, interaction_id: str) -> None:
        self._mark_reply_uncertain("dm", interaction_id)

    def confirm_dm_reply(self, interaction_id: str) -> None:
        reservation = self._reservation_key("dm", interaction_id)
        if self._reply_reservations.get(reservation) not in {"reserved", "writing"}:
            raise ValueError("Twitter DM reply reservation is not active")
        self._reply_reservations[reservation] = "replied"

    @staticmethod
    def _reservation_key(kind: str, interaction_id: str) -> str:
        return f"{kind}:{interaction_id}"

    def _release_reply(self, kind: str, interaction_id: str) -> None:
        reservation = self._reservation_key(kind, interaction_id)
        if self._reply_reservations.get(reservation) in {"reserved", "writing"}:
            self._reply_reservations.pop(reservation, None)

    def _mark_reply_uncertain(self, kind: str, interaction_id: str) -> None:
        reservation = self._reservation_key(kind, interaction_id)
        if self._reply_reservations.get(reservation) in {"reserved", "writing"}:
            self._reply_reservations[reservation] = "uncertain"
            if kind == "dm":
                self.dm_delivery_uncertain = True

    def opt_out_dm(self, conversation_id: str) -> None:
        conversation = str(conversation_id)
        self.known_dm_conversations.add(conversation)
        self.opted_out_dm_conversations.add(conversation)

    def clear_dm_opt_out(self, conversation_id: str) -> None:
        self.opted_out_dm_conversations.discard(str(conversation_id))

    def clear_dm_delivery_uncertainty(self) -> None:
        self.dm_delivery_uncertain = False
        for reservation, status in self._reply_reservations.items():
            if reservation.startswith("dm:") and status in {"writing", "uncertain"}:
                self._reply_reservations[reservation] = "reconciled"

    def has_ambiguous_dm_write(self) -> bool:
        return self.dm_delivery_uncertain or any(
            reservation.startswith("dm:") and status in {"writing", "uncertain"}
            for reservation, status in self._reply_reservations.items()
        )

    def has_unanchored_dm_recovery_block(self, conversation_id: str) -> bool:
        conversation = str(conversation_id)
        return any(
            reservation.startswith("dm:")
            and status in {"reconciled", "replied"}
            and self._dm_reply_routes.get(reservation.removeprefix("dm:"))
            == conversation
            for reservation, status in self._reply_reservations.items()
        )

    def can_send_dm(self, conversation_id: str) -> bool:
        conversation = str(conversation_id)
        return (
            conversation in self.known_dm_conversations
            and conversation not in self.opted_out_dm_conversations
        )

    def is_bot_post(self, post_id: str) -> bool:
        return str(post_id) in self._branches

    def bot_posts_for_anchor(self, anchor_id: str) -> set[str]:
        anchor = str(anchor_id)
        return {post_id for post_id, value in self._branches.items() if value == anchor}

    def recent_bot_posts_for_anchor(self, anchor_id: str, limit: int) -> list[str]:
        anchor = str(anchor_id)
        return [
            post_id
            for post_id, value in reversed(self._branches.items())
            if value == anchor
        ][:limit]

    def resolve_anchor(self, trigger_id: str, ancestor_ids: list[str]) -> str:
        for ancestor_id in ancestor_ids:
            anchor = self._branches.get(str(ancestor_id))
            if anchor:
                return anchor
        return str(trigger_id)

    def advance_mentions(self, post_id: str) -> None:
        self.mention_since_id = str(post_id)

    def advance_dms(self, event_id: str) -> None:
        self.dm_last_seen_event_id = str(event_id)

    def _trim(self) -> None:
        while len(self._seen) > self.max_seen:
            self._seen.popitem(last=False)
        while len(self._branches) > self.max_branches:
            self._branches.popitem(last=False)

    def _make_route_capacity(self, kind: str, interaction_id: str) -> bool:
        routes = (
            self._public_reply_routes if kind == "tweet" else self._dm_reply_routes
        )
        if interaction_id in routes or len(routes) < self.max_seen:
            return True
        unreserved = next(
            (
                key
                for key in routes
                if self._reservation_key(kind, key) not in self._reply_reservations
            ),
            None,
        )
        if unreserved is not None:
            routes.pop(unreserved)
            return True
        return self._evict_oldest_terminal_pair(kind)

    def _make_reservation_capacity(self) -> bool:
        return (
            len(self._reply_reservations) < self.max_seen
            or self._evict_oldest_terminal_pair()
        )

    def _evict_oldest_terminal_pair(self, kind: str = "") -> bool:
        for reservation, status in self._reply_reservations.items():
            reservation_kind, _, interaction_id = reservation.partition(":")
            if (
                status not in {"replied", "reconciled"}
                or (kind and reservation_kind != kind)
            ):
                continue
            routes = (
                self._public_reply_routes
                if reservation_kind == "tweet"
                else self._dm_reply_routes
            )
            if interaction_id not in routes:
                continue
            routes.pop(interaction_id)
            self._reply_reservations.pop(reservation)
            return True
        return False


async def mutate_state(
    account_id: str, mutation: Callable[[TwitterState], None]
) -> TwitterState:
    async with twitter_state_lock(active_profile_key(), account_id):
        state = TwitterState.load()
        mutation(state)
        state.save()
        return state
