"""
tests/test_session_orchestration_spawn_tool.py — Unit tests for the
session_spawn LLM tool defined in session_orchestration/spawn_tool.py.

Acceptance criteria covered
----------------------------
(a) Resolved repo fills workdir in the SpawnRequest.
(b) Unresolved repo returns an error string WITHOUT calling spawn_session.
(c) agent defaults to "omp" when omitted from args (DEFAULT_AGENT).
(d) spawn_session is called with the correct SpawnRequest when repo resolves.
"""

from __future__ import annotations

import pytest

from session_orchestration.repo_registry import (
    DEFAULT_AGENT,
    RepoEntry,
    RepoRegistry,
    ResolvedRepo,
    UnresolvedRepo,
    build_repo_registry,
)
from session_orchestration.spawn import SpawnRequest, SpawnResult
from session_orchestration.spawn_tool import spawn_tool_handler
from session_orchestration.types import SessionHandle


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_FAKE_WORKDIR = "/home/user/dev/myrepo"
_FAKE_TASK_ID = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
_FAKE_SESSION = "hermes-omp-abc12-def34"


def _make_resolved_repo(path: str = _FAKE_WORKDIR, agent: str = DEFAULT_AGENT) -> ResolvedRepo:
    return ResolvedRepo(
        path=path,
        default_agent=agent,
        matched_name="myrepo",
        match_kind="exact",
    )


def _make_fake_registry(resolved) -> RepoRegistry:
    """Build a RepoRegistry whose resolve() always returns *resolved*."""
    class _FakeRegistry:
        def resolve(self, name: str):
            return resolved

    return _FakeRegistry()  # type: ignore[return-value]


def _make_fake_spawn_result(**overrides) -> SpawnResult:
    from datetime import datetime, timezone

    defaults = dict(
        task_id=_FAKE_TASK_ID,
        handle=SessionHandle(
            session_id=_FAKE_TASK_ID,
            tmux_session=_FAKE_SESSION,
            pane=f"{_FAKE_SESSION}:0.0",
            launch_ts=datetime.now(tz=timezone.utc),
        ),
        session_name=_FAKE_SESSION,
        thread_id=None,
    )
    defaults.update(overrides)
    return SpawnResult(**defaults)


# ---------------------------------------------------------------------------
# (a) Resolved repo fills workdir
# ---------------------------------------------------------------------------


def test_resolved_repo_fills_workdir():
    """When repo resolves, the SpawnRequest workdir must equal resolved.path."""
    captured: list[SpawnRequest] = []

    def mock_spawn(request: SpawnRequest, **_kw) -> SpawnResult:
        captured.append(request)
        return _make_fake_spawn_result()

    registry = _make_fake_registry(_make_resolved_repo(path="/srv/code/projectX"))

    spawn_tool_handler(
        {"repo": "projectX", "prompt": "Do the thing"},
        _repo_registry=registry,
        _spawn_fn=mock_spawn,
    )

    assert len(captured) == 1
    assert captured[0].workdir == "/srv/code/projectX", (
        "workdir should be set to the resolved repo path"
    )


# ---------------------------------------------------------------------------
# (b) Unresolved repo → error string, spawn_session NOT called
# ---------------------------------------------------------------------------


def test_unresolved_repo_returns_error_without_calling_spawn():
    """UnresolvedRepo must return an error string; spawn_session must not be called."""
    spawn_called = []

    def mock_spawn(request: SpawnRequest, **_kw) -> SpawnResult:
        spawn_called.append(request)
        return _make_fake_spawn_result()

    registry = _make_fake_registry(UnresolvedRepo(name="nosuchrepo"))

    result = spawn_tool_handler(
        {"repo": "nosuchrepo", "prompt": "something"},
        _repo_registry=registry,
        _spawn_fn=mock_spawn,
    )

    assert spawn_called == [], "spawn_session must NOT be called for an unresolved repo"
    assert "nosuchrepo" in result, "error message should mention the unresolved name"
    # Must ask the user to supply a full path or configure an alias
    assert "path" in result.lower() or "alias" in result.lower(), (
        "error should guide the user toward a full path or alias"
    )


# ---------------------------------------------------------------------------
# (c) agent defaults to DEFAULT_AGENT ("omp") when omitted
# ---------------------------------------------------------------------------


def test_agent_defaults_to_omp_when_omitted():
    """When 'agent' is absent from args and resolved repo has no default, agent must be 'omp'."""
    captured: list[SpawnRequest] = []

    def mock_spawn(request: SpawnRequest, **_kw) -> SpawnResult:
        captured.append(request)
        return _make_fake_spawn_result()

    # Resolved repo with default_agent=DEFAULT_AGENT (matches the fallback)
    registry = _make_fake_registry(_make_resolved_repo(agent=DEFAULT_AGENT))

    spawn_tool_handler(
        # "agent" key absent — must default to DEFAULT_AGENT
        {"repo": "myrepo", "prompt": "run tests"},
        _repo_registry=registry,
        _spawn_fn=mock_spawn,
    )

    assert len(captured) == 1
    assert captured[0].agent == DEFAULT_AGENT, (
        f"agent should default to '{DEFAULT_AGENT}' (DEFAULT_AGENT) when omitted"
    )


def test_agent_defaults_to_omp_when_resolved_repo_has_no_agent():
    """Even if resolved.default_agent is empty, the final agent must be DEFAULT_AGENT."""
    captured: list[SpawnRequest] = []

    def mock_spawn(request: SpawnRequest, **_kw) -> SpawnResult:
        captured.append(request)
        return _make_fake_spawn_result()

    # Resolved repo with an empty default_agent to exercise the fallback chain
    resolved = ResolvedRepo(
        path=_FAKE_WORKDIR,
        default_agent="",  # empty → must fall back to DEFAULT_AGENT
        matched_name="myrepo",
        match_kind="exact",
    )
    registry = _make_fake_registry(resolved)

    spawn_tool_handler(
        {"repo": "myrepo", "prompt": "run tests"},
        _repo_registry=registry,
        _spawn_fn=mock_spawn,
    )

    assert len(captured) == 1
    assert captured[0].agent == DEFAULT_AGENT, (
        f"agent must fall back to DEFAULT_AGENT ('{DEFAULT_AGENT}') "
        "when resolved.default_agent is empty and arg is absent"
    )


# ---------------------------------------------------------------------------
# (d) spawn_session called with correct SpawnRequest
# ---------------------------------------------------------------------------


def test_spawn_session_called_with_correct_request():
    """spawn_session must receive a SpawnRequest with the right fields."""
    captured: list[SpawnRequest] = []

    def mock_spawn(request: SpawnRequest, **_kw) -> SpawnResult:
        captured.append(request)
        return _make_fake_spawn_result()

    registry = _make_fake_registry(_make_resolved_repo(path=_FAKE_WORKDIR, agent="claude"))

    spawn_tool_handler(
        {
            "repo": "myrepo",
            "prompt": "implement feature X",
            "agent": "claude",
            "z_command": "/z-plan",
        },
        _repo_registry=registry,
        _spawn_fn=mock_spawn,
    )

    assert len(captured) == 1
    req = captured[0]
    assert req.prompt == "implement feature X"
    assert req.agent == "claude"
    assert req.workdir == _FAKE_WORKDIR
    assert req.z_command == "/z-plan"


def test_spawn_session_called_with_resolved_agent_over_registry_default():
    """When arg 'agent' is explicit, it must win over resolved.default_agent."""
    captured: list[SpawnRequest] = []

    def mock_spawn(request: SpawnRequest, **_kw) -> SpawnResult:
        captured.append(request)
        return _make_fake_spawn_result()

    # resolved.default_agent is "omp" but caller explicitly asks for "claude"
    registry = _make_fake_registry(_make_resolved_repo(agent="omp"))

    spawn_tool_handler(
        {"repo": "myrepo", "prompt": "some task", "agent": "claude"},
        _repo_registry=registry,
        _spawn_fn=mock_spawn,
    )

    assert len(captured) == 1
    assert captured[0].agent == "claude", (
        "explicit 'agent' arg must override resolved.default_agent"
    )


# ---------------------------------------------------------------------------
# Reply string shape
# ---------------------------------------------------------------------------


def test_reply_contains_task_id_and_session():
    """On success the reply string must include task_id and session_name."""

    def mock_spawn(request: SpawnRequest, **_kw) -> SpawnResult:
        return _make_fake_spawn_result(
            task_id="test-task-id",
            session_name="hermes-omp-xxxxx-yyyyy",
        )

    registry = _make_fake_registry(_make_resolved_repo())

    reply = spawn_tool_handler(
        {"repo": "myrepo", "prompt": "do stuff"},
        _repo_registry=registry,
        _spawn_fn=mock_spawn,
    )

    assert "test-task-id" in reply
    assert "hermes-omp-xxxxx-yyyyy" in reply


def test_reply_missing_required_fields():
    """Missing 'repo' or 'prompt' must return an error without calling spawn."""
    spawn_called = []

    def mock_spawn(request: SpawnRequest, **_kw) -> SpawnResult:
        spawn_called.append(request)
        return _make_fake_spawn_result()

    registry = _make_fake_registry(_make_resolved_repo())

    for bad_args in [
        {},
        {"repo": "myrepo"},
        {"prompt": "something"},
        {"repo": "", "prompt": "something"},
        {"repo": "myrepo", "prompt": ""},
    ]:
        result = spawn_tool_handler(bad_args, _repo_registry=registry, _spawn_fn=mock_spawn)
        assert spawn_called == [], f"spawn must not be called for args={bad_args!r}"
        assert "required" in result.lower() or "repo" in result or "prompt" in result, (
            f"expected an error message for args={bad_args!r}, got: {result!r}"
        )


# ---------------------------------------------------------------------------
# Tool registration smoke-test
# ---------------------------------------------------------------------------


def test_tool_registered_in_registry():
    """session_spawn must appear in the tool registry after module import."""
    import session_orchestration.spawn_tool  # noqa: F401 — triggers registry.register

    from tools.registry import registry as tool_registry

    entry = tool_registry.get_entry("session_spawn")
    assert entry is not None, "session_spawn must be registered in the tool registry"
    assert entry.toolset == "session_spawn"
    assert entry.handler is not None
