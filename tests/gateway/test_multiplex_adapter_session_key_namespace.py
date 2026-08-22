"""Regression tests: adapter-level session keys must carry the profile (#88391).

Incident shape (Aug 2026, local install, two Telegram bots on one multiplexed
gateway): a Telegram private chat reports the user's own id as ``chat.id``, so
every bot in the multiplexer sees an identical ``chat_id`` for the same human.

``BasePlatformAdapter.handle_message`` and each adapter's ``_text_batch_key`` /
``_photo_batch_key`` derive a session key at INGRESS — before
``GatewayRunner._make_profile_message_handler`` stamps ``source.profile``. The
key therefore fell back to the *active* profile's namespace, so both bots
produced ``agent:main:telegram:dm:<uid>`` and shared one lane: the text-batching
dict, ``_active_sessions`` and the busy-session guard are all keyed on that
string. Observed in production logs: 60 flushes, zero carrying ``agent:medicina:``.

Fix under test: ``set_owner_profile`` records credential ownership on the
adapter, and ``_session_key_profile`` resolves the namespace as
``source.profile`` → ``_owner_profile`` → session-store resolver, so a secondary
adapter keys into its own namespace even before the runner stamps the source.
"""

import pytest

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter
from gateway.session import SessionSource, build_session_key


UID = "8693894969"


def _source(profile=None):
    """Telegram DM shape: chat_id == user_id, thread_id None — same for every bot."""
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id=UID,
        chat_type="dm",
        user_id=UID,
        user_name="Lucas",
        profile=profile,
    )


class _Adapter(BasePlatformAdapter):
    """Minimal concrete adapter — only the key-derivation seam is under test."""

    name = "stub"

    def __init__(self):
        self._session_store = None
        self._owner_profile = None

    # BasePlatformAdapter declares these abstract; none is exercised here.
    async def connect(self): ...
    async def disconnect(self): ...
    async def send(self, *a, **k): ...
    async def send_message(self, *a, **k): ...
    async def get_chat_info(self, *a, **k): ...
    async def start_listening(self): ...


class _Store:
    """Stand-in for GatewaySessionStore's namespace resolver."""

    def __init__(self, active="default", multiplex=True):
        self._active = active
        self._multiplex = multiplex

    def _resolve_profile_for_key(self, source=None):
        if not self._multiplex:
            return None
        if source is not None and getattr(source, "profile", None):
            return source.profile
        return self._active


class TestOwnerProfileKeying:
    def test_secondary_adapter_keys_into_own_namespace(self):
        """The bug: unstamped source + active profile 'default' collapsed a
        secondary bot's key onto agent:main:."""
        a = _Adapter()
        a._session_store = _Store(active="default")
        a.set_owner_profile("medicina")
        key = build_session_key(_source(), profile=a._session_key_profile(_source()))
        assert key.startswith("agent:medicina:"), key

    def test_two_bots_same_chat_do_not_collide(self):
        """Two adapters, one chat id: the keys must differ or the batching dict,
        _active_sessions and the busy guard merge both bots into one lane."""
        default_a, secondary_a = _Adapter(), _Adapter()
        default_a._session_store = _Store(active="default")
        secondary_a._session_store = _Store(active="default")
        secondary_a.set_owner_profile("medicina")
        src = _source()
        k_default = build_session_key(src, profile=default_a._session_key_profile(src))
        k_secondary = build_session_key(src, profile=secondary_a._session_key_profile(src))
        assert k_default != k_secondary, f"both bots share one lane: {k_default}"
        assert k_default.startswith("agent:main:")
        assert k_secondary.startswith("agent:medicina:")

    def test_stamped_source_wins_over_owner(self):
        """Connector/relay ingress stamps source.profile — it must take priority
        so a shared-ingress adapter routes per event, not per credential."""
        a = _Adapter()
        a._session_store = _Store(active="default")
        a.set_owner_profile("medicina")
        assert a._session_key_profile(_source(profile="finances")) == "finances"

    def test_primary_adapter_unchanged(self):
        """No owner + active default ⇒ legacy agent:main:, byte-identical."""
        a = _Adapter()
        a._session_store = _Store(active="default")
        key = build_session_key(_source(), profile=a._session_key_profile(_source()))
        assert key == build_session_key(_source())
        assert key.startswith("agent:main:")

    def test_single_profile_gateway_unchanged(self):
        """Multiplexing off ⇒ resolver returns None ⇒ legacy namespace."""
        a = _Adapter()
        a._session_store = _Store(multiplex=False)
        assert a._session_key_profile(_source()) is None
        key = build_session_key(_source(), profile=a._session_key_profile(_source()))
        assert key == build_session_key(_source())

    def test_owner_default_is_normalized_to_none(self):
        """'default' must collapse to None, not produce 'agent:default:'."""
        a = _Adapter()
        a._session_store = None
        a.set_owner_profile("default")
        assert a._owner_profile is None
        assert build_session_key(_source(), profile=a._session_key_profile(_source())) \
            == build_session_key(_source())

    @pytest.mark.parametrize("blank", [None, "", "   "])
    def test_blank_owner_is_none(self, blank):
        a = _Adapter()
        a._session_store = None
        a.set_owner_profile(blank)
        assert a._owner_profile is None

    def test_owner_used_when_store_absent(self):
        """A secondary adapter must not depend on the store being installed."""
        a = _Adapter()
        a._session_store = None
        a.set_owner_profile("medicina")
        assert a._session_key_profile(_source()) == "medicina"

    def test_adapter_without_base_init_does_not_raise(self):
        """Adapters are routinely built via ``object.__new__`` (tests) or by a
        subclass that never calls ``BasePlatformAdapter.__init__``, so
        ``_owner_profile``/``_session_store`` may be entirely absent. Resolution
        must degrade to the legacy namespace instead of AttributeError — this
        broke every text-batching suite on the first cut of the fix.
        """
        bare = object.__new__(_Adapter)
        assert not hasattr(bare, "_owner_profile")
        assert bare._session_key_profile(_source()) is None
        assert build_session_key(_source(), profile=bare._session_key_profile(_source())) \
            == build_session_key(_source())

    def test_resolver_exception_falls_back_to_none(self):
        class _Boom:
            def _resolve_profile_for_key(self, source=None):
                raise RuntimeError("store unavailable")

        a = _Adapter()
        a._session_store = _Boom()
        assert a._session_key_profile(_source()) is None

    def test_non_string_resolver_result_is_rejected(self):
        """A duck-typed/mock session store returns a truthy non-string, which
        would be interpolated into the key as ``agent:<MagicMock ...>:`` and
        corrupt every lookup. Real regression: it broke Slack's thread-reply
        suite, whose fixture store is a bare MagicMock.
        """
        from unittest.mock import MagicMock

        a = _Adapter()
        a._session_store = MagicMock()          # resolver returns a MagicMock
        assert a._session_key_profile(_source()) is None
        assert build_session_key(_source(), profile=a._session_key_profile(_source())) \
            == build_session_key(_source())

    @pytest.mark.parametrize("junk", [123, object(), ["medicina"], b"medicina", "   "])
    def test_non_string_or_blank_owner_is_ignored(self, junk):
        a = _Adapter()
        a._session_store = None
        a._owner_profile = junk                 # bypass the setter's normalisation
        assert a._session_key_profile(_source()) is None

    def test_no_source_still_resolves_owner(self):
        """Some call sites derive a key without an event (idle/wake paths)."""
        a = _Adapter()
        a._session_store = _Store(active="default")
        a.set_owner_profile("medicina")
        assert a._session_key_profile(None) == "medicina"
