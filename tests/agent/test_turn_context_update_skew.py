"""Regression coverage for in-process source-update skew at turn start.

A long-lived dashboard can import ``agent.turn_context`` before an in-place
source update, then lazily import ``agent.conversation_loop`` afterwards.  The
new loop must not pass newly-added optional keywords to that already-loaded
legacy builder, or the next ordinary prompt fails before reaching the model.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import conversation_loop


class _LegacyBuilderReached(RuntimeError):
    """Sentinel proving argument binding reached the legacy builder body."""


class _CurrentBuilderReached(RuntimeError):
    def __init__(self, display_kind, display_metadata):
        self.display_kind = display_kind
        self.display_metadata = display_metadata


class _UninspectableBuilderReached(RuntimeError):
    def __init__(self, kwargs):
        self.kwargs = kwargs


def _legacy_build_turn_context(
    agent,
    user_message,
    system_message,
    conversation_history,
    task_id,
    stream_callback,
    persist_user_message,
    persist_user_timestamp=None,
    *,
    restore_or_build_system_prompt,
    install_safe_stdio,
    sanitize_surrogates,
    summarize_user_message_for_log,
    set_session_context,
    set_current_write_origin,
    ra,
    moa_active=False,
):
    """Signature from before ``persist_user_display_*`` was introduced."""
    raise _LegacyBuilderReached


def _current_build_turn_context(
    *args,
    persist_user_display_kind=None,
    persist_user_display_metadata=None,
    restore_or_build_system_prompt,
    install_safe_stdio,
    sanitize_surrogates,
    summarize_user_message_for_log,
    set_session_context,
    set_current_write_origin,
    ra,
    moa_active=False,
):
    """Current keyword seam without a permissive ``**kwargs`` escape hatch."""
    raise _CurrentBuilderReached(
        persist_user_display_kind,
        persist_user_display_metadata,
    )


def _kwargs_build_turn_context(*args, **kwargs):
    """A wrapper-style builder advertises support through ``**kwargs``."""
    raise _CurrentBuilderReached(
        kwargs.get("persist_user_display_kind"),
        kwargs.get("persist_user_display_metadata"),
    )


class _UninspectableBuildTurnContext:
    """Remain callable even when signature introspection is unavailable."""

    __signature__ = object()

    def __call__(self, *args, **kwargs):
        raise _UninspectableBuilderReached(kwargs)


_uninspectable_build_turn_context = _UninspectableBuildTurnContext()


def test_normal_turn_after_model_switch_tolerates_loaded_legacy_builder(monkeypatch):
    monkeypatch.setattr(
        conversation_loop,
        "build_turn_context",
        _legacy_build_turn_context,
    )

    # ``moa_config`` skips unrelated decoding. The real turn reaches the builder
    # after initializing only these per-turn attributes on the agent.
    agent = SimpleNamespace()
    with pytest.raises(_LegacyBuilderReached):
        conversation_loop.run_conversation(
            agent,
            "ordinary prompt after /model",
            moa_config={},
        )


def test_current_builder_still_receives_synthetic_turn_display_fields(monkeypatch):
    monkeypatch.setattr(
        conversation_loop,
        "build_turn_context",
        _current_build_turn_context,
    )

    with pytest.raises(_CurrentBuilderReached) as reached:
        conversation_loop.run_conversation(
            SimpleNamespace(),
            "model-switch context",
            persist_user_display_kind="model_switch",
            persist_user_display_metadata={"model": "test-model"},
            moa_config={},
        )

    assert reached.value.display_kind == "model_switch"
    assert reached.value.display_metadata == {"model": "test-model"}


def test_kwargs_builder_receives_synthetic_turn_display_fields(monkeypatch):
    monkeypatch.setattr(
        conversation_loop,
        "build_turn_context",
        _kwargs_build_turn_context,
    )

    with pytest.raises(_CurrentBuilderReached) as reached:
        conversation_loop.run_conversation(
            SimpleNamespace(),
            "delegation context",
            persist_user_display_kind="delegation",
            persist_user_display_metadata={"task_count": 2},
            moa_config={},
        )

    assert reached.value.display_kind == "delegation"
    assert reached.value.display_metadata == {"task_count": 2}


def test_uninspectable_builder_fails_closed_without_display_fields(monkeypatch):
    monkeypatch.setattr(
        conversation_loop,
        "build_turn_context",
        _uninspectable_build_turn_context,
    )

    with pytest.raises(_UninspectableBuilderReached) as reached:
        conversation_loop.run_conversation(
            SimpleNamespace(),
            "model-switch context",
            persist_user_display_kind="model_switch",
            persist_user_display_metadata={"model": "test-model"},
            moa_config={},
        )

    assert "persist_user_display_kind" not in reached.value.kwargs
    assert "persist_user_display_metadata" not in reached.value.kwargs
