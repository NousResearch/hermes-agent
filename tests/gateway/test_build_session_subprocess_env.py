"""Tests for gateway.session_context.build_session_subprocess_env.

Companion to tests/gateway/test_session_context_inheritance.py. That file
covers ContextVar inheritance leaks between concurrent asyncio tasks; this
file covers the narrowed forwarding SET itself — only
HERMES_SESSION_PLATFORM/CHAT_ID/THREAD_ID are forwarded into spawned
subprocesses (e.g. kanban workers), not the full _VAR_MAP (key, id, profile,
user, cron auto-deliver targets, ...). See #57356 review finding: the
original implementation forwarded every _VAR_MAP entry, broadening a
worker's inherited session identity with no demonstrated consumer.
"""
import os

import pytest

from gateway.session_context import (
    _SUBPROCESS_FORWARD_VARS,
    _UNSET,
    _VAR_MAP,
    build_session_subprocess_env,
    clear_session_vars,
    set_session_vars,
)

SESSION_VARS = list(_VAR_MAP.keys())
NON_FORWARDED_VARS = [v for v in SESSION_VARS if v not in _SUBPROCESS_FORWARD_VARS]


@pytest.fixture(autouse=True)
def _isolate_session_context():
    """Clean ContextVar + os.environ slate per test, restored afterwards."""
    saved_env = {k: os.environ.get(k) for k in SESSION_VARS}
    saved_ctx = {name: var.get() for name, var in _VAR_MAP.items()}
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    for k in SESSION_VARS:
        os.environ.pop(k, None)
    try:
        yield
    finally:
        for var, val in zip(_VAR_MAP.values(), saved_ctx.values()):
            var.set(val)
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_forward_set_is_exactly_platform_chat_thread():
    """The narrowed forwarding tuple is exactly platform/chat_id/thread_id —
    not the full _VAR_MAP (which also carries key, id, profile, user_id,
    user_name, message_id, chat_name, source)."""
    assert set(_SUBPROCESS_FORWARD_VARS) == {
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_SESSION_THREAD_ID",
    }
    # Sanity: the excluded set is non-empty, i.e. _VAR_MAP is strictly wider.
    assert NON_FORWARDED_VARS


def test_build_env_forwards_only_narrow_set_from_contextvars():
    """When ContextVars are bound (real gateway path), only the narrow set
    reaches the subprocess env — session key/id/profile/user are dropped."""
    set_session_vars(
        session_key="agent:main:telegram:thread:CHAT:THREAD",
        platform="telegram",
        chat_id="-1001",
        thread_id="17585",
        user_id="USER1",
        chat_name="my-chat",
        message_id="MSG1",
    )

    env = build_session_subprocess_env({})

    assert env == {
        "HERMES_SESSION_PLATFORM": "telegram",
        "HERMES_SESSION_CHAT_ID": "-1001",
        "HERMES_SESSION_THREAD_ID": "17585",
    }
    for var_name in NON_FORWARDED_VARS:
        assert var_name not in env, f"{var_name} leaked into subprocess env"


def test_build_env_forwards_only_narrow_set_from_os_environ_fallback():
    """CLI/cron paths that never call set_session_vars fall back to
    os.environ — still narrowed to the same three vars."""
    os.environ["HERMES_SESSION_PLATFORM"] = "discord"
    os.environ["HERMES_SESSION_CHAT_ID"] = "CHAT9"
    os.environ["HERMES_SESSION_THREAD_ID"] = "THREAD9"
    os.environ["HERMES_SESSION_KEY"] = "agent:main:discord:thread:CHAT9:THREAD9"
    os.environ["HERMES_SESSION_USER_ID"] = "USER9"

    env = build_session_subprocess_env({})

    assert env == {
        "HERMES_SESSION_PLATFORM": "discord",
        "HERMES_SESSION_CHAT_ID": "CHAT9",
        "HERMES_SESSION_THREAD_ID": "THREAD9",
    }
    assert "HERMES_SESSION_KEY" not in env
    assert "HERMES_SESSION_USER_ID" not in env


def test_build_env_preserves_base_env_and_does_not_mutate_it():
    """base_env is copied, not mutated, and its own keys are preserved
    alongside the forwarded session vars."""
    set_session_vars(platform="slack", chat_id="C1", thread_id="T1")
    base = {"PATH": "/usr/bin", "HERMES_KANBAN_TASK": "t_123"}

    env = build_session_subprocess_env(base)

    assert env["PATH"] == "/usr/bin"
    assert env["HERMES_KANBAN_TASK"] == "t_123"
    assert env["HERMES_SESSION_PLATFORM"] == "slack"
    assert "HERMES_SESSION_PLATFORM" not in base, "base_env dict was mutated"


def test_build_env_omits_var_absent_from_both_contextvar_and_environ():
    """When a forwarded var is never bound (ContextVar still _UNSET) and
    absent from os.environ, it's simply absent from the result."""
    for var_name in _SUBPROCESS_FORWARD_VARS:
        os.environ.pop(var_name, None)

    env = build_session_subprocess_env({})

    for var_name in _SUBPROCESS_FORWARD_VARS:
        assert var_name not in env


def test_build_env_forwards_explicitly_cleared_var_as_empty_string():
    """clear_session_vars() sets vars to "" (explicitly cleared, distinct
    from _UNSET) so the os.environ fallback is suppressed -- the forwarded
    value is an empty string, not an omission."""
    tokens = set_session_vars(platform="telegram", chat_id="-1001", thread_id="17585")
    clear_session_vars(tokens)

    env = build_session_subprocess_env({})

    assert env["HERMES_SESSION_PLATFORM"] == ""
    assert env["HERMES_SESSION_CHAT_ID"] == ""
    assert env["HERMES_SESSION_THREAD_ID"] == ""
