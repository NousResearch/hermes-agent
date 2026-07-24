"""Regression: _format_session_info must honour channel_overrides + /model session
overrides so the /new banner and /status reflect the model the next turn will
actually use. Before the fix, it read model.default directly and reported
``super-super-model`` even when channel_overrides pinned the thread to a
different model.

Reported by a user running a Telegram thread pinned via channel_overrides to a
custom provider model. The /new auto-reset notice kept advertising the global
``super-super-model`` default, which contradicted the model the next turn
actually ran on. The same banner appears for /status, so any thread using
channel_overrides or a /model session override was affected — not just
custom-provider setups. The bug was in the gateway, not in the user's
provider config; the per-thread override just made it visible.

Fix: _format_session_info accepts an optional ``source`` and runs the same
priority chain as _resolve_session_agent_runtime (``/model`` session
override → channel_overrides → global default). _reset_notice_session_info,
the only production caller from the reset path, passes its source through.
The no-source path is preserved for CLI introspection where no channel
context exists."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from gateway.config import (
    ChannelOverride,
    GatewayConfig,
    Platform,
    PlatformConfig,
)
from gateway.run import GatewayRunner
from gateway.session import SessionSource


def _make_runner(home: Path) -> GatewayRunner:
    """Build a GatewayRunner without going through the heavy __init__ chain.

    _format_session_info only touches a handful of attributes — we stub the
    rest with ``object.__new__`` so the test stays fast and hermetic.
    """
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.TELEGRAM: PlatformConfig(
                enabled=True,
                channel_overrides={
                    "2147": ChannelOverride(model="ol/minimax-m3"),
                    "9999": ChannelOverride(model="openrouter/healer-alpha"),
                },
            ),
        },
    )
    runner._session_model_overrides = {}
    return runner


def _telegram_source(thread_id: str | None) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1004295889073",
        chat_type="group",
        user_id="ddky",
        thread_id=thread_id,
    )


def test_format_session_info_no_source_uses_global_default(home: Path):
    runner = _make_runner(home)
    out = runner._format_session_info()
    # Without a source we cannot resolve channel context, so the banner shows
    # the raw config default. This is intentional — _format_session_info() is
    # also called for CLI introspection where no channel exists.
    assert "super-super-model" in out


def test_format_session_info_applies_channel_override(home: Path):
    """Thread 2147 pinned to ol/minimax-m3 must surface in the banner."""
    runner = _make_runner(home)
    src = _telegram_source("2147")

    # Stub the helper hooks the banner also needs so the test focuses on the
    # override resolution — these helpers read config the runner doesn't own.
    with patch(
        "gateway.run._resolve_gateway_model", return_value="super-super-model"
    ), patch(
        "gateway.run._load_gateway_config", return_value={}
    ), patch(
        "gateway.run._resolve_runtime_agent_kwargs", return_value={}
    ):
        out = runner._format_session_info(source=src)

    assert "ol/minimax-m3" in out, f"expected ol/minimax-m3, got: {out!r}"
    assert "super-super-model" not in out


def test_format_session_info_no_override_thread_falls_back(home: Path):
    """A thread that has no override should still report the global default."""
    runner = _make_runner(home)
    src = _telegram_source("8888")  # not in channel_overrides

    with patch(
        "gateway.run._resolve_gateway_model", return_value="super-super-model"
    ), patch(
        "gateway.run._load_gateway_config", return_value={}
    ), patch(
        "gateway.run._resolve_runtime_agent_kwargs", return_value={}
    ):
        out = runner._format_session_info(source=src)

    assert "super-super-model" in out


def test_format_session_info_session_model_wins_over_channel_override(home: Path):
    """Priority: session /model > channel_overrides > global. Regression
    guard for the docstring claim at line 3752."""
    runner = _make_runner(home)
    src = _telegram_source("2147")
    sk = "agent:main:telegram:group:-1004295889073:2147"
    runner._session_model_overrides[sk] = {"model": "ol/special-override"}

    with patch(
        "gateway.run._resolve_gateway_model", return_value="super-super-model"
    ), patch(
        "gateway.run._load_gateway_config", return_value={}
    ), patch(
        "gateway.run._resolve_runtime_agent_kwargs", return_value={}
    ), patch.object(
        GatewayRunner, "_session_key_for_source", return_value=sk
    ), patch.object(
        GatewayRunner, "_rehydrate_session_model_override", lambda self, k: None
    ):
        out = runner._format_session_info(source=src)

    assert "ol/special-override" in out, f"expected /model override, got: {out!r}"
    assert "ol/minimax-m3" not in out


@pytest.fixture
def home(monkeypatch, tmp_path) -> Path:
    """Route HERMES_HOME at a temp dir so the runner's config + session lookups
    stay hermetic. Tests that don't touch the filesystem don't care, but the
    session-key builder does, and we want a clean baseline."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path
