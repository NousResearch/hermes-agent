"""Guard: the Telegram outbound throttle must stay WIRED into the adapter.

The limiter module existing on disk proves nothing. It only paces sends if
`connect()` passes it to `Application.builder().rate_limiter(...)`.

On 2026-09-09 a `hermes update` autostash restored the checkout WITHOUT the
wiring block on both the Mac and the hub. `rate_limiter.py` was even deleted
outright on the Mac. Nothing failed loudly: the gateway connected fine, the
`[telegram-throttle] active:` banner simply stopped appearing, and Argo sent
unpaced until Telegram issued multi-minute `Flood control exceeded` bans
(3,762 flood lines in one morning). Sam saw only a 👎 reaction on every
message, because reactions ride a path that survives a send ban.

This test asserts the wiring, not the module, so an update that silently
drops it goes red instead of going quiet.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ADAPTER = (
    Path(__file__).resolve().parents[4]
    / "plugins"
    / "platforms"
    / "telegram"
    / "adapter.py"
)
LIMITER = ADAPTER.parent / "rate_limiter.py"


def _adapter_source() -> str:
    assert ADAPTER.is_file(), f"adapter.py missing at {ADAPTER}"
    return ADAPTER.read_text()


def test_rate_limiter_module_present() -> None:
    """The limiter itself must exist next to the adapter."""
    assert LIMITER.is_file(), (
        f"{LIMITER} is missing. Restore it from another box; without it the "
        "adapter falls back to unpaced sending and Telegram will ban the bot."
    )


def test_limiter_factory_is_importable_from_module() -> None:
    """The factory the adapter calls must actually be defined."""
    src = LIMITER.read_text()
    assert "def build_rate_limiter_from_config(" in src


def test_adapter_calls_the_limiter_factory() -> None:
    """connect() must build a limiter from config, not skip it."""
    src = _adapter_source()
    assert "build_rate_limiter_from_config" in src, (
        "The Telegram adapter no longer builds a rate limiter. Re-add the "
        "throttle block after `builder = Application.builder().token(...)`."
    )


def test_adapter_passes_limiter_to_the_ptb_builder() -> None:
    """Building one is not enough: it must reach PTB's builder chain."""
    src = _adapter_source()
    assert re.search(r"builder\s*=\s*builder\.rate_limiter\(", src), (
        "A limiter is built but never passed to Application.builder(). "
        "Every send stays unpaced."
    )


def test_wiring_sits_inside_connect_after_the_builder_anchor() -> None:
    """Order matters: the call must follow builder creation, precede build()."""
    src = _adapter_source()
    anchor = src.find("builder = Application.builder().token(")
    wiring = src.find("builder = builder.rate_limiter(")
    assert anchor != -1, "builder anchor missing; adapter shape changed"
    assert wiring != -1, "rate_limiter wiring missing"
    # adapter.py has more than one builder.build() call (a retry helper builds
    # too); the one that matters is the first AFTER the token anchor.
    built = src.find("self._app = builder.build()", anchor)
    assert built != -1, "builder.build() after anchor missing; shape changed"
    assert anchor < wiring < built, (
        "rate_limiter wiring is outside the builder window "
        f"(anchor={anchor}, wiring={wiring}, build={built})"
    )


def test_throttle_failure_does_not_block_connect() -> None:
    """A broken throttle must warn, never take the platform down."""
    src = _adapter_source()
    idx = src.find("build_rate_limiter_from_config")
    assert idx != -1
    window = src[idx - 400 : idx + 900]
    assert "try:" in window and "except Exception" in window, (
        "The throttle wiring must be wrapped so an import error degrades to "
        "unpaced sending instead of failing connect()."
    )


@pytest.mark.parametrize(
    "banner_fragment",
    ["telegram-throttle", "msg/s per chat"],
)
def test_activation_banner_is_still_logged(banner_fragment: str) -> None:
    """The banner is the only live proof the throttle attached; keep it."""
    src = LIMITER.read_text()
    assert banner_fragment in src, (
        f"Activation banner fragment {banner_fragment!r} gone from "
        "rate_limiter.py. Without it there is no way to verify from logs "
        "that a running gateway is actually paced."
    )
