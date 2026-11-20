"""Delegated children must not replace their parent's session identity."""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent.delegation_context import delegated_child_context
from gateway.session_context import (
    _SESSION_ID,
    _UNSET,
    get_session_env,
    set_current_session_id,
)
from tools.environments.local import build_subprocess_env


@pytest.fixture(autouse=True)
def _isolate_session_context():
    saved_env = os.environ.get("HERMES_SESSION_ID")
    saved_ctx = _SESSION_ID.get()
    _SESSION_ID.set(_UNSET)
    os.environ.pop("HERMES_SESSION_ID", None)
    try:
        yield
    finally:
        _SESSION_ID.set(saved_ctx)
        if saved_env is None:
            os.environ.pop("HERMES_SESSION_ID", None)
        else:
            os.environ["HERMES_SESSION_ID"] = saved_env


def _construct_child(session_id: str) -> tuple[object, str | None]:
    """Model the session mutation performed by AIAgent.__init__."""
    with delegated_child_context():
        set_current_session_id(session_id)
        return _SESSION_ID.get(), os.environ.get("HERMES_SESSION_ID")


def test_root_agent_keeps_contextvar_and_environment_in_sync():
    set_current_session_id("parent-session")

    assert _SESSION_ID.get() == "parent-session"
    assert os.environ["HERMES_SESSION_ID"] == "parent-session"
    assert get_session_env("HERMES_SESSION_ID") == "parent-session"


def test_child_construction_restores_both_parent_id_paths():
    set_current_session_id("parent-session")

    inside_context, inside_environment = _construct_child("child-session")

    assert inside_context == "child-session"
    assert inside_environment == "parent-session"
    assert _SESSION_ID.get() == "parent-session"
    assert os.environ["HERMES_SESSION_ID"] == "parent-session"
    assert get_session_env("HERMES_SESSION_ID") == "parent-session"


def test_child_execution_binds_own_id_then_restores_parent():
    set_current_session_id("parent-session")

    with delegated_child_context("child-session"):
        assert _SESSION_ID.get() == "child-session"
        assert get_session_env("HERMES_SESSION_ID") == "child-session"
        assert os.environ["HERMES_SESSION_ID"] == "parent-session"

    assert _SESSION_ID.get() == "parent-session"
    assert get_session_env("HERMES_SESSION_ID") == "parent-session"
    assert os.environ["HERMES_SESSION_ID"] == "parent-session"


def test_child_subprocess_environment_receives_child_id():
    set_current_session_id("parent-session")

    with delegated_child_context("child-session"):
        child_env = build_subprocess_env(
            base={"HERMES_SESSION_ID": "foreign-session"},
        )

    assert child_env["HERMES_SESSION_ID"] == "child-session"
    assert os.environ["HERMES_SESSION_ID"] == "parent-session"
    assert _SESSION_ID.get() == "parent-session"


def test_parallel_children_keep_parent_environment_and_own_contexts():
    set_current_session_id("parent-session")
    child_ids = [f"child-{index}" for index in range(8)]

    def run_child(child_id: str) -> tuple[str, str, str | None, object]:
        with delegated_child_context(child_id):
            observed_context = str(_SESSION_ID.get())
            observed_subprocess = build_subprocess_env(base={}).get(
                "HERMES_SESSION_ID"
            )
            observed_environment = os.environ.get("HERMES_SESSION_ID")
        return (
            observed_context,
            str(observed_subprocess),
            observed_environment,
            _SESSION_ID.get(),
        )

    with ThreadPoolExecutor(max_workers=len(child_ids)) as pool:
        observations = list(pool.map(run_child, child_ids))

    for child_id, observation in zip(child_ids, observations):
        context_id, subprocess_id, environment_id, after_context = observation
        assert context_id == child_id
        assert subprocess_id == child_id
        assert environment_id == "parent-session"
        # ThreadPoolExecutor workers do not inherit the caller's ContextVar;
        # the scope must restore that worker's original unset value.
        assert after_context is _UNSET

    assert _SESSION_ID.get() == "parent-session"
    assert os.environ["HERMES_SESSION_ID"] == "parent-session"


def test_nested_child_scope_restores_outer_child_then_parent():
    set_current_session_id("parent-session")

    with delegated_child_context("child-outer"):
        assert get_session_env("HERMES_SESSION_ID") == "child-outer"
        with delegated_child_context("child-inner"):
            assert get_session_env("HERMES_SESSION_ID") == "child-inner"
        assert get_session_env("HERMES_SESSION_ID") == "child-outer"

    assert get_session_env("HERMES_SESSION_ID") == "parent-session"
    assert os.environ["HERMES_SESSION_ID"] == "parent-session"


def test_root_rotation_still_updates_both_paths_after_child():
    set_current_session_id("parent-v1")
    _construct_child("child-session")

    set_current_session_id("parent-v2")

    assert _SESSION_ID.get() == "parent-v2"
    assert os.environ["HERMES_SESSION_ID"] == "parent-v2"
