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
# (b2) Missing repo → asks for a repo, spawn_session NOT called
# ---------------------------------------------------------------------------


def test_missing_repo_asks_for_repo_without_calling_spawn():
    """No repo supplied (e.g. "so omp 'fix blah'") must return a plain-language
    ask for a repo and must NOT call spawn_session — never break, just ask."""
    spawn_called = []

    def mock_spawn(request: SpawnRequest, **_kw) -> SpawnResult:
        spawn_called.append(request)
        return _make_fake_spawn_result()

    # repo key absent entirely; a valid prompt is present
    result = spawn_tool_handler(
        {"prompt": "fix the flaky test", "agent": "omp"},
        _repo_registry=_make_fake_registry(_make_resolved_repo()),
        _spawn_fn=mock_spawn,
    )

    assert spawn_called == [], "spawn_session must NOT be called when repo is missing"
    assert "repo" in result.lower(), "ask should mention 'repo'"
    assert "?" in result, "ask should be phrased as a question, not an error"


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
# Discord thread wiring (gap #1)
# ---------------------------------------------------------------------------


def test_thread_context_threaded_into_spawn(monkeypatch):
    """When a live Discord turn resolves a thread context, the handler must set
    ``parent_chat_id`` on the SpawnRequest and pass ``thread_creator`` to
    spawn_session — so spawn.py step 6 creates the project thread."""
    import session_orchestration.spawn_tool as st

    sentinel_creator = lambda pcid, name: "thread-123"  # noqa: E731

    monkeypatch.setattr(
        st, "_resolve_discord_thread_context",
        lambda: ("channel-999", sentinel_creator),
    )

    captured = {}

    def mock_spawn(request: SpawnRequest, **kw) -> SpawnResult:
        captured["request"] = request
        captured["kw"] = kw
        return _make_fake_spawn_result(thread_id="thread-123")

    registry = _make_fake_registry(_make_resolved_repo())

    reply = st.spawn_tool_handler(
        {"repo": "myrepo", "prompt": "do stuff"},
        _repo_registry=registry,
        _spawn_fn=mock_spawn,
    )

    assert captured["request"].parent_chat_id == "channel-999", (
        "resolved parent chat id must be set on the SpawnRequest"
    )
    assert captured["kw"].get("thread_creator") is sentinel_creator, (
        "resolved thread_creator must be forwarded to spawn_session"
    )
    # The reply should surface the created thread link.
    assert "thread-123" in reply


def test_thread_context_absent_spawns_threadless(monkeypatch):
    """Off Discord / no live gateway, the resolver returns (None, None); the
    handler must still spawn, with parent_chat_id=None and thread_creator=None
    (prior behavior — never block a spawn on thread wiring)."""
    import session_orchestration.spawn_tool as st

    monkeypatch.setattr(st, "_resolve_discord_thread_context", lambda: (None, None))

    captured = {}

    def mock_spawn(request: SpawnRequest, **kw) -> SpawnResult:
        captured["request"] = request
        captured["kw"] = kw
        return _make_fake_spawn_result(thread_id=None)

    registry = _make_fake_registry(_make_resolved_repo())

    reply = st.spawn_tool_handler(
        {"repo": "myrepo", "prompt": "do stuff"},
        _repo_registry=registry,
        _spawn_fn=mock_spawn,
    )

    assert captured["request"].parent_chat_id is None
    assert captured["kw"].get("thread_creator") is None
    assert "No project thread" in reply


def test_resolver_is_discord_gated(monkeypatch):
    """_resolve_discord_thread_context must short-circuit to (None, None) for a
    non-Discord platform without touching the gateway runner."""
    import session_orchestration.spawn_tool as st
    from gateway import session_context as sc

    monkeypatch.setattr(
        sc, "get_session_env",
        lambda name, default="": "telegram" if name == "HERMES_SESSION_PLATFORM" else default,
    )
    # If gating fails, this would blow up rather than return cleanly.
    assert st._resolve_discord_thread_context() == (None, None)


def test_resolver_adopts_existing_thread(monkeypatch):
    """When the Discord turn is already in a thread (the Hermès window), the
    resolver must ADOPT that thread — return it as the target and a no-op
    creator that yields the same id — without creating a new thread or needing
    the live gateway/adapter."""
    import session_orchestration.spawn_tool as st
    from gateway import session_context as sc

    def fake_env(name, default=""):
        return {
            "HERMES_SESSION_PLATFORM": "discord",
            "HERMES_SESSION_CHAT_ID": "1521983360040304871",
            "HERMES_SESSION_THREAD_ID": "1521983360040304871",
        }.get(name, default)

    monkeypatch.setattr(sc, "get_session_env", fake_env)
    # Guard: if the adopt path wrongly fell through to creation it would call
    # _gateway_runner_ref; make that explode so the test fails loudly instead.
    import gateway.run as gr
    monkeypatch.setattr(
        gr, "_gateway_runner_ref",
        lambda: (_ for _ in ()).throw(AssertionError("must not reach create path")),
    )

    target, creator = st._resolve_discord_thread_context()
    assert target == "1521983360040304871", "must adopt the window thread as target"
    assert creator("ignored-parent", "ignored-name") == "1521983360040304871", (
        "adopt creator must return the existing thread id, not create a new one"
    )


def test_discord_user_id_captured_onto_request(monkeypatch):
    """The handler must read HERMES_SESSION_USER_ID (Discord turn) and set it as
    discord_user_id on the SpawnRequest, so the row can @-mention the user."""
    import session_orchestration.spawn_tool as st

    monkeypatch.setattr(st, "_resolve_discord_thread_context", lambda: (None, None))
    monkeypatch.setattr(st, "_resolve_discord_user_id", lambda: "555000111")

    captured = {}

    def mock_spawn(request: SpawnRequest, **kw) -> SpawnResult:
        captured["request"] = request
        return _make_fake_spawn_result()

    registry = _make_fake_registry(_make_resolved_repo())

    st.spawn_tool_handler(
        {"repo": "myrepo", "prompt": "do stuff"},
        _repo_registry=registry,
        _spawn_fn=mock_spawn,
    )

    assert captured["request"].discord_user_id == "555000111"


def test_user_id_resolver_is_discord_gated(monkeypatch):
    """_resolve_discord_user_id must return None off Discord even if a user id
    is present in the session context."""
    import session_orchestration.spawn_tool as st
    from gateway import session_context as sc

    def fake_env(name, default=""):
        return {
            "HERMES_SESSION_PLATFORM": "telegram",
            "HERMES_SESSION_USER_ID": "999",
        }.get(name, default)

    monkeypatch.setattr(sc, "get_session_env", fake_env)
    assert st._resolve_discord_user_id() is None


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
