"""Tests for the Honcho representation cap (context bloat fix).

Covers:
- representationMaxConclusions parsing: default applies when unset, host > root
  precedence, explicit 0/negative means uncapped (None), positive values
  clamped to the SDK's 1..100 range.
- The cap is passed through to peer.context(max_conclusions=...) on the
  prefetch path so Honcho filters the representation server-side, while cards
  and session summary are not affected by it.
"""

from unittest.mock import MagicMock

from plugins.memory.honcho.client import (
    HonchoClientConfig,
    _parse_representation_max_conclusions,
)
from plugins.memory.honcho.session import HonchoSessionManager

_DEFAULT = HonchoClientConfig.__dataclass_fields__["representation_max_conclusions"].default


class TestParseRepresentationMaxConclusions:
    def test_unset_returns_default(self):
        assert _parse_representation_max_conclusions(None, None, default=25) == 25

    def test_host_wins_over_root(self):
        assert _parse_representation_max_conclusions(40, 10, default=25) == 40

    def test_root_used_when_host_unset(self):
        assert _parse_representation_max_conclusions(None, 10, default=25) == 10

    def test_explicit_zero_means_uncapped(self):
        assert _parse_representation_max_conclusions(0, None, default=25) is None

    def test_negative_means_uncapped(self):
        assert _parse_representation_max_conclusions(-1, None, default=25) is None

    def test_clamped_to_sdk_max(self):
        assert _parse_representation_max_conclusions(500, None, default=25) == 100

    def test_clamped_to_sdk_min(self):
        # A positive sub-1 value can't occur for ints, but 1 is the floor.
        assert _parse_representation_max_conclusions(1, None, default=25) == 1

    def test_garbage_falls_through_to_default(self):
        assert _parse_representation_max_conclusions("nan", None, default=25) == 25

    def test_string_int_parsed(self):
        assert _parse_representation_max_conclusions("30", None, default=25) == 30


class TestDataclassDefault:
    def test_has_bounded_default(self):
        """The default must be a bounded positive int, not None — an unset
        install must not inject an uncapped representation."""
        cfg = HonchoClientConfig()
        assert isinstance(cfg.representation_max_conclusions, int)
        assert 1 <= cfg.representation_max_conclusions <= 100


class TestCapThreadedToPeerContext:
    def _peer_returning(self, representation: str, card):
        peer = MagicMock()
        ctx = MagicMock()
        ctx.representation = representation
        ctx.peer_card = card
        peer.context.return_value = ctx
        return peer

    def test_cap_passed_as_max_conclusions_kwarg(self):
        """A set cap must reach peer.context(max_conclusions=...). We do not
        force include_most_frequent — Honcho's default ordering favors recency."""
        mgr = HonchoSessionManager()
        mgr._config = HonchoClientConfig(representation_max_conclusions=25)
        peer = self._peer_returning("rep text", ["Name: X"])
        mgr._get_or_create_peer = MagicMock(return_value=peer)

        out = mgr._fetch_peer_context("peer-1", target="peer-1")

        assert out["representation"] == "rep text"
        _, kwargs = peer.context.call_args
        assert kwargs.get("max_conclusions") == 25
        assert "include_most_frequent" not in kwargs

    def test_uncapped_omits_max_conclusions_kwarg(self):
        """representation_max_conclusions=None must NOT pass max_conclusions= (uncapped)."""
        mgr = HonchoSessionManager()
        mgr._config = HonchoClientConfig(representation_max_conclusions=None)
        peer = self._peer_returning("rep text", ["Name: X"])
        mgr._get_or_create_peer = MagicMock(return_value=peer)

        mgr._fetch_peer_context("peer-1", target="peer-1")

        _, kwargs = peer.context.call_args
        assert "max_conclusions" not in kwargs

    def test_card_unaffected_by_cap(self):
        """The peer card comes back whole regardless of the representation cap."""
        mgr = HonchoSessionManager()
        mgr._config = HonchoClientConfig(representation_max_conclusions=5)
        peer = self._peer_returning("rep", ["Name: X", "Role: dev", "Pref: dark"])
        mgr._get_or_create_peer = MagicMock(return_value=peer)

        out = mgr._fetch_peer_context("peer-1", target="peer-1")

        assert out["card"] == ["Name: X", "Role: dev", "Pref: dark"]
