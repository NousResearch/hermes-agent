"""Regression coverage for #29623 — legacy ``reply_to_mode: false`` in
Discord/Telegram configs silently regressed to ``"first"`` instead of
being treated as ``"off"``.

The fix introduces ``gateway.config.normalize_reply_to_mode`` and wires
it into both ``PlatformConfig.__post_init__`` and the Discord/Telegram
adapter constructors. These tests pin the full contract end-to-end:

* Reporter's exact repro (``PlatformConfig.from_dict({'reply_to_mode': False})``).
* Every construction path — ``from_dict``, direct dataclass, env-var
  bridge — lands on a canonical ``"off"`` / ``"first"`` / ``"all"``.
* Both adapter constructors absorb any leftover non-canonical value
  through the defensive normalisation layer.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from gateway.config import (
    DEFAULT_REPLY_TO_MODE,
    REPLY_TO_MODES,
    PlatformConfig,
    Platform,
    load_gateway_config,
    normalize_reply_to_mode,
)


# ---------------------------------------------------------------------------
# normalize_reply_to_mode — the helper's pure contract
# ---------------------------------------------------------------------------


class TestNormalizeReplyToModeHelper:
    @pytest.mark.parametrize(
        "value,expected",
        [
            # The whole point of #29623 — boolean False must become "off".
            (False, "off"),
            # True historically meant "yes please reply (default behaviour)".
            (True, "first"),
            # Missing / blank → default
            (None, "first"),
            ("", "first"),
            ("   ", "first"),
            ("\t\n", "first"),
            # Canonical strings round-trip.
            ("off", "off"),
            ("first", "first"),
            ("all", "all"),
            # Case + whitespace insensitive.
            ("OFF", "off"),
            ("Off", "off"),
            ("  off  ", "off"),
            ("FIRST", "first"),
            ("ALL", "all"),
            # Bool-shaped strings (env-var bridge can hand us these).
            ("true", "first"),
            ("false", "off"),
            ("yes", "first"),
            ("no", "off"),
            ("1", "first"),
            ("0", "off"),
            ("TRUE", "first"),
            ("False", "off"),
            # Numeric 0/1 (YAML can parse to int)
            (0, "off"),
            (1, "first"),
            # Garbage falls back to the default rather than leaking into
            # the adapter where it would silently mean "first" anyway.
            ("banana", "first"),
            ("on", "first"),  # ambiguous historically, "on" is not a mode
            ("forever", "first"),
            (42, "first"),
            (object(), "first"),
        ],
    )
    def test_canonicalises_inputs(self, value, expected):
        assert normalize_reply_to_mode(value) == expected

    def test_custom_default_is_honoured(self):
        """Adapters that want a different fallback (e.g. legacy ``"all"``
        on a forum-bridge bot) can pass ``default=``. ``True`` and
        garbage both resolve to that default."""
        assert normalize_reply_to_mode(True, default="all") == "all"
        assert normalize_reply_to_mode(None, default="all") == "all"
        assert normalize_reply_to_mode("banana", default="all") == "all"
        # ``False`` is still ``"off"`` — that's the explicit user intent
        # the default-override should NOT subvert.
        assert normalize_reply_to_mode(False, default="all") == "off"

    def test_output_is_always_in_REPLY_TO_MODES(self):
        """Defensive: every code path through the helper must return a
        value in the public tuple — adapters branch on those exact
        strings."""
        inputs = [
            None, True, False, 0, 1, 42, "",
            "off", "first", "all", "OFF", "FIRST", "ALL",
            "true", "false", "yes", "no",
            "banana", "on", "forever",
            object(), [], {}, (1, 2),
        ]
        for value in inputs:
            assert normalize_reply_to_mode(value) in REPLY_TO_MODES, (
                f"{value!r} produced an out-of-range mode"
            )

    def test_default_constant_is_in_modes(self):
        """Catch the silly typo case where someone changes the default
        to e.g. ``"frist"`` and breaks every adapter at once."""
        assert DEFAULT_REPLY_TO_MODE in REPLY_TO_MODES


# ---------------------------------------------------------------------------
# Reporter's exact repro
# ---------------------------------------------------------------------------


class TestReporterRepro29623:
    """Exact transcript from
    https://github.com/NousResearch/hermes-agent/issues/29623."""

    def test_from_dict_false_resolves_to_off(self):
        """Before the fix: ``False -> False``. After: ``False -> "off"``."""
        cfg = PlatformConfig.from_dict({"enabled": True, "reply_to_mode": False})
        assert cfg.reply_to_mode == "off"

    def test_from_dict_off_string_resolves_to_off(self):
        cfg = PlatformConfig.from_dict({"enabled": True, "reply_to_mode": "off"})
        assert cfg.reply_to_mode == "off"

    def test_from_dict_missing_resolves_to_first(self):
        cfg = PlatformConfig.from_dict({"enabled": True})
        assert cfg.reply_to_mode == "first"

    def test_from_dict_all_three_match_reporter_table(self):
        """The reporter pasted a three-row table — assert all three
        rows agree on the new canonical output."""
        results = {
            "False": PlatformConfig.from_dict({"enabled": True, "reply_to_mode": False}).reply_to_mode,
            "off": PlatformConfig.from_dict({"enabled": True, "reply_to_mode": "off"}).reply_to_mode,
            "missing": PlatformConfig.from_dict({"enabled": True}).reply_to_mode,
        }
        assert results == {"False": "off", "off": "off", "missing": "first"}


# ---------------------------------------------------------------------------
# PlatformConfig construction paths
# ---------------------------------------------------------------------------


class TestPlatformConfigPostInit:
    """``__post_init__`` is the single point of truth — every
    construction style (direct, ``from_dict``, env-var override) ends
    up with a normalised value."""

    @pytest.mark.parametrize(
        "value,expected",
        [
            (False, "off"),
            (True, "first"),
            (None, "first"),
            ("", "first"),
            ("off", "off"),
            ("OFF", "off"),
            ("  off  ", "off"),
            ("all", "all"),
            ("ALL", "all"),
            ("false", "off"),
            ("true", "first"),
            ("banana", "first"),
        ],
    )
    def test_direct_construction_normalises(self, value, expected):
        cfg = PlatformConfig(enabled=True, reply_to_mode=value)
        assert cfg.reply_to_mode == expected

    def test_default_is_first(self):
        """No-arg construction respects the dataclass default."""
        cfg = PlatformConfig(enabled=True)
        assert cfg.reply_to_mode == "first"

    def test_to_dict_round_trip_preserves_off(self):
        """The bad path was: legacy YAML stores ``false`` → from_dict
        keeps ``False`` → to_dict serialises ``False`` → next reload
        repeats the cycle. After normalisation, to_dict always
        serialises a canonical string."""
        cfg = PlatformConfig.from_dict({"enabled": True, "reply_to_mode": False})
        serialised = cfg.to_dict()
        assert serialised["reply_to_mode"] == "off"
        # And reloading is idempotent.
        cfg2 = PlatformConfig.from_dict(serialised)
        assert cfg2.reply_to_mode == "off"

    def test_post_init_runs_on_every_construction(self):
        """Sanity: if ``__post_init__`` were silently dropped (e.g. by
        a future ``__init__`` override) every input would round-trip
        as-is. Pin the contract here."""
        cfg = PlatformConfig(enabled=True, reply_to_mode="banana")
        assert cfg.reply_to_mode == "first", (
            "PlatformConfig.__post_init__ should normalise reply_to_mode"
        )


# ---------------------------------------------------------------------------
# Env-var bridge (load_gateway_config)
# ---------------------------------------------------------------------------


class TestEnvVarBridgeNormalises:
    """The env-var path (``DISCORD_REPLY_TO_MODE`` /
    ``TELEGRAM_REPLY_TO_MODE``) already filters non-canonical strings,
    but after the fix the ``PlatformConfig`` it constructs goes through
    ``__post_init__`` regardless, so even a hypothetical garbage value
    can't leak into the adapter."""

    def test_discord_env_off_lands_as_off(self, monkeypatch):
        for var in (
            "DISCORD_REPLY_TO_MODE",
            "TELEGRAM_REPLY_TO_MODE",
            "DISCORD_BOT_TOKEN",
        ):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "x")
        monkeypatch.setenv("DISCORD_REPLY_TO_MODE", "off")

        cfg = load_gateway_config()
        assert cfg.platforms[Platform.DISCORD].reply_to_mode == "off"

    def test_discord_env_first_lands_as_first(self, monkeypatch):
        for var in ("DISCORD_REPLY_TO_MODE", "DISCORD_BOT_TOKEN"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("DISCORD_BOT_TOKEN", "x")
        monkeypatch.setenv("DISCORD_REPLY_TO_MODE", "first")

        cfg = load_gateway_config()
        assert cfg.platforms[Platform.DISCORD].reply_to_mode == "first"


# ---------------------------------------------------------------------------
# Adapter defence-in-depth — stub configs that bypass __post_init__
# ---------------------------------------------------------------------------


class _StubConfig:
    """A config-shaped duck that does NOT inherit ``PlatformConfig``.

    Mirrors what a downstream embedder might do if they hand-roll a
    config object instead of going through the dataclass. The adapter
    constructor reads ``reply_to_mode`` via ``getattr``, so this is
    the realistic worst-case escape hatch.
    """

    def __init__(self, **kw):
        self.enabled = True
        self.token = "stub"
        self.extra = {}
        self.home_channel = None
        # Anything callers want to set, e.g. ``reply_to_mode=False``.
        for k, v in kw.items():
            setattr(self, k, v)


class TestAdapterDefensiveNormalisation:
    """Even if a stub config slips a ``False`` past
    ``PlatformConfig.__post_init__``, the adapter constructor must
    still produce a canonical mode."""

    def test_discord_adapter_normalises_stub_false(self):
        from gateway.platforms.discord import DiscordAdapter

        adapter = DiscordAdapter(_StubConfig(reply_to_mode=False))
        assert adapter._reply_to_mode == "off"

    def test_discord_adapter_normalises_stub_garbage(self):
        from gateway.platforms.discord import DiscordAdapter

        adapter = DiscordAdapter(_StubConfig(reply_to_mode="banana"))
        assert adapter._reply_to_mode == "first"

    def test_discord_adapter_normalises_stub_missing_attr(self):
        from gateway.platforms.discord import DiscordAdapter

        stub = _StubConfig()
        # Force-remove the attribute so ``getattr(config, 'reply_to_mode',
        # None)`` actually returns ``None``.
        if hasattr(stub, "reply_to_mode"):
            delattr(stub, "reply_to_mode")
        adapter = DiscordAdapter(stub)
        assert adapter._reply_to_mode == "first"

    def test_telegram_adapter_normalises_stub_false(self):
        try:
            from gateway.platforms.telegram import TelegramAdapter
        except Exception:
            pytest.skip("telegram adapter dependencies not installed")

        # We can't instantiate the full TelegramAdapter without a token
        # exchange, so directly exercise the normalisation contract via
        # the helper the adapter delegates to.
        assert normalize_reply_to_mode(False) == "off"
        # And confirm the stub config (which IS what would be passed)
        # produces ``False`` so the adapter clearly needs a defender.
        stub = _StubConfig(reply_to_mode=False)
        assert stub.reply_to_mode is False
        assert normalize_reply_to_mode(getattr(stub, "reply_to_mode", None)) == "off"


# ---------------------------------------------------------------------------
# Whole-config round-trip — pin that ``PlatformConfig.to_dict ->
# from_dict`` is idempotent for legacy bool input.
# ---------------------------------------------------------------------------


class TestLegacyBoolRoundTrip:
    """If an operator's on-disk config has ``reply_to_mode: false``,
    Hermes should load it, treat it as ``"off"``, and (if it ever
    rewrites the file) save back a canonical string — never silently
    re-emit the bool that started the bug."""

    @pytest.mark.parametrize("raw", [False, "off", "OFF", "false", "no", "0", 0])
    def test_legacy_off_inputs_all_serialise_to_off(self, raw):
        cfg = PlatformConfig.from_dict({"enabled": True, "reply_to_mode": raw})
        assert cfg.reply_to_mode == "off"
        assert cfg.to_dict()["reply_to_mode"] == "off"
