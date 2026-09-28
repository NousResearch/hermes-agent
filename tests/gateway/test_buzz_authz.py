"""Gateway authz tests: BUZZ_ALLOWED_USERS accepts npub or hex (#78428).

Inbound Buzz events carry the sender as a 64-char hex pubkey, while
``BUZZ_ALLOWED_USERS`` historically accepted npubs too.  The gateway's
central allowlist comparison must decode npub entries to hex at comparison
time, or an operator who listed only their npub is rejected with
"Unauthorized user: <hex pubkey>" (gateway drops the message).
"""

import pytest

from gateway.config import Platform
from gateway.platform_registry import PlatformEntry, platform_registry
from gateway.session import SessionSource

# Chip's public identity (public information, not a secret) — the same pair
# used by tests/gateway/test_buzz_adapter.py.
SELF_PUBKEY = "9fd5c7ba6d3ef224da78f541e0fcb9c50f72cc63edb19aae76ac6a0474dfa860"
SELF_NPUB = "npub1nl2u0wnd8mezfknc74q7pl9ec58h9nrrakce4tnk434qgaxl4psqe5twr6"
OTHER_PUBKEY = "a" * 64

_BUZZ_PLATFORM = Platform("buzz")

_AUTH_ENV_VARS = (
    "BUZZ_ALLOWED_USERS",
    "BUZZ_ALLOW_ALL_USERS",
    "GATEWAY_ALLOWED_USERS",
    "GATEWAY_ALLOW_ALL_USERS",
)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch):
    for var in _AUTH_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def buzz_registered():
    """Register a minimal Buzz platform entry (allowed_users_env contract)."""
    platform_registry.register(
        PlatformEntry(
            name="buzz",
            label="Buzz",
            adapter_factory=lambda cfg: None,
            check_fn=lambda: True,
            allowed_users_env="BUZZ_ALLOWED_USERS",
            allow_all_env="BUZZ_ALLOW_ALL_USERS",
        )
    )
    yield
    platform_registry.unregister("buzz")


def _make_runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner.pairing_store = None
    return runner


def _make_source(user_id: str, chat_type: str = "dm"):
    return SessionSource(
        platform=_BUZZ_PLATFORM,
        chat_id="ccc2bc1a-7a82-5a8f-8c4e-57a070cbe7cd",
        chat_type=chat_type,
        user_id=user_id,
        user_name="Chip",
        is_bot=False,
    )


def test_npub_only_allowlist_authorizes_hex_identity(monkeypatch, buzz_registered):
    """The reported bug: listing only the npub must authorize the hex pubkey."""
    monkeypatch.setenv("BUZZ_ALLOWED_USERS", SELF_NPUB)
    runner = _make_runner()
    assert runner._is_user_authorized(_make_source(SELF_PUBKEY)) is True


def test_hex_only_allowlist_still_authorizes(monkeypatch, buzz_registered):
    """Existing hex-only allowlists keep working unchanged."""
    monkeypatch.setenv("BUZZ_ALLOWED_USERS", SELF_PUBKEY)
    runner = _make_runner()
    assert runner._is_user_authorized(_make_source(SELF_PUBKEY)) is True


def test_mixed_npub_and_hex_allowlist_authorizes(monkeypatch, buzz_registered):
    """Both forms in one allowlist authorize the same identity."""
    monkeypatch.setenv("BUZZ_ALLOWED_USERS", f"{SELF_NPUB},{SELF_PUBKEY}")
    runner = _make_runner()
    assert runner._is_user_authorized(_make_source(SELF_PUBKEY)) is True


def test_npub_allowlist_still_denies_other_user(monkeypatch, buzz_registered):
    """Normalization must not turn into fail-open for unrelated senders."""
    monkeypatch.setenv("BUZZ_ALLOWED_USERS", SELF_NPUB)
    runner = _make_runner()
    assert runner._is_user_authorized(_make_source(OTHER_PUBKEY)) is False


def test_uppercase_npub_allowlist_authorizes(monkeypatch, buzz_registered):
    """npub entries are case-insensitive, like the adapter's own decoder."""
    monkeypatch.setenv("BUZZ_ALLOWED_USERS", SELF_NPUB.upper())
    runner = _make_runner()
    assert runner._is_user_authorized(_make_source(SELF_PUBKEY)) is True


def test_normalize_nostr_allow_entries():
    from gateway.authz_mixin import _normalize_nostr_allow_entries

    expanded = _normalize_nostr_allow_entries({SELF_NPUB, OTHER_PUBKEY, "not-a-key"})
    assert SELF_PUBKEY in expanded
    assert SELF_NPUB in expanded  # original kept, harmless
    assert OTHER_PUBKEY in expanded
    assert "not-a-key" in expanded


def test_npub_to_hex_roundtrip():
    from gateway.authz_mixin import _npub_to_hex

    assert _npub_to_hex(SELF_NPUB) == SELF_PUBKEY
    assert _npub_to_hex(SELF_PUBKEY) is None  # hex input is not an npub
    assert _npub_to_hex("npub1garbage!!") is None  # invalid bech32
    assert _npub_to_hex("") is None


# ─────────────────────────────────────────────────────────────────────
# #82871: single-profile gateway must consult the Buzz adapter's own
# config.extra.allowed_users when no env allowlist is configured.  The
# reported symptom was a default-deny of EVERY Buzz user ("Unauthorized
# user: <hex> on buzz") even though the sender's npub was correctly
# listed in gateway.platforms.buzz.extra.allowed_users.
# ─────────────────────────────────────────────────────────────────────


def _make_runner_with_adapter(extra: dict):
    """Single-profile (multiplex OFF) runner with a live Buzz adapter."""
    from types import SimpleNamespace

    from gateway.authz_mixin import _npub_to_hex
    from gateway.config import GatewayConfig, PlatformConfig
    from gateway.run import GatewayRunner

    def _normalize(entry):
        entry = (entry or "").strip()
        if entry.lower().startswith("npub1"):
            return _npub_to_hex(entry)
        return entry.lower() if entry else None

    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.pairing_store = None
    runner.adapters = {
        _BUZZ_PLATFORM: SimpleNamespace(
            config=PlatformConfig(enabled=True, extra=extra),
            normalize_user_id=_normalize,
        )
    }
    return runner


def test_config_only_npub_allowlist_authorizes_single_profile(buzz_registered):
    """#82871 repro: npub listed ONLY in config.extra.allowed_users (no env
    var at all) must authorize the sender's hex pubkey."""
    runner = _make_runner_with_adapter({"allowed_users": [SELF_NPUB]})
    assert runner._is_user_authorized(_make_source(SELF_PUBKEY)) is True


def test_config_only_hex_allowlist_authorizes_single_profile(buzz_registered):
    runner = _make_runner_with_adapter({"allowed_users": [SELF_PUBKEY]})
    assert runner._is_user_authorized(_make_source(SELF_PUBKEY)) is True


def test_config_only_allowlist_still_denies_unlisted_sender(buzz_registered):
    """Consulting the adapter allowlist must not fail open."""
    runner = _make_runner_with_adapter({"allowed_users": [SELF_NPUB]})
    assert runner._is_user_authorized(_make_source(OTHER_PUBKEY)) is False


def test_config_only_empty_allowlist_keeps_default_deny(buzz_registered):
    """No allowlist anywhere: the default-deny is preserved (SECURITY.md §2.6)."""
    runner = _make_runner_with_adapter({})
    assert runner._is_user_authorized(_make_source(SELF_PUBKEY)) is False
