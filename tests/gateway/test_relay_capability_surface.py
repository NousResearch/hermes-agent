"""Phase 0 regression harness for the relay/connector work.

Locks the *behavioral contract* that the future ``RelayAdapter`` must reproduce:
the gateway's ``stream_consumer`` and ``BasePlatformAdapter`` read per-platform
capabilities through a small, stable surface. A relay adapter that exposes the
same surface (``MAX_MESSAGE_LENGTH`` attribute, ``message_len_fn`` property,
``supports_draft_streaming`` probe, and only the abstract methods) slots into
the existing consumer with no consumer changes.

These are deliberately *behavioral* (construct an adapter, drive the code,
assert the observable outcome) rather than source-string snapshots, per the
repo's "don't write change-detector tests" rule. They pass on ``main`` before
any ``RelayAdapter`` exists — they describe the contract, not the relay.
"""

import inspect

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter


class _MinAdapter(BasePlatformAdapter):
    """Smallest concrete adapter: implements exactly the abstract methods."""

    async def connect(self, *, is_reconnect: bool = False):  # pragma: no cover - not called
        return True

    async def disconnect(self):  # pragma: no cover - not called
        return None

    async def send(self, *args, **kwargs):  # pragma: no cover - not called
        return None

    async def get_chat_info(self, chat_id):  # pragma: no cover - not called
        return {}


def _make() -> BasePlatformAdapter:
    return _MinAdapter(PlatformConfig(), Platform.LOCAL)


def test_abstract_methods_are_the_known_set():
    """The relay adapter must implement exactly this set of abstract methods.

    Everything else on BasePlatformAdapter has a default, so ONE generic
    RelayAdapter overriding the right subset is feasible without per-platform
    gateway classes. If a new abstractmethod is added here, the relay design
    (and the cross-repo contract) must be revisited — hence the lock.

    NOTE: this is four methods, not three — ``get_chat_info`` is abstract too
    (defined far below the connect/disconnect/send cluster in base.py). The
    RelayAdapter must implement it (proxying a chat-info lookup to the
    connector, or returning a descriptor-derived stub).
    """
    abstract = {
        name
        for name, member in inspect.getmembers(BasePlatformAdapter)
        if getattr(member, "__isabstractmethod__", False)
    }
    assert abstract == {"connect", "disconnect", "send", "get_chat_info"}


