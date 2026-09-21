"""Per-process and per-session ACP tool scoping (#45955).

One ``hermes acp`` process serves many sessions for many projects, but every session was built
with a hardcoded ``["hermes-acp"]``: ``hermes acp --toolsets`` did not exist and a client had no
way to scope a single session. These tests pin the resolution chain — per-session
``_meta.hermes.toolsets`` > process ``--toolsets`` > the platform default — and, more importantly,
that a scoped session never silently widens on any sibling path: model-switch rebuild, restore
from the DB after a process restart, fork, and ACP MCP registration.

``_meta.hermes`` is the namespace the adapter already uses for its own ``_meta`` output
(``provenance.py``); every other ``_meta`` key, a bare ``_meta.toolsets`` included, is ignored.

The agent constructor and provider resolution are stubbed; the toolset resolution under test is
the real one (``_agent_factory`` short-circuits ``_make_agent``, so these tests never use it).
"""

from __future__ import annotations

import json
import threading
import types

import pytest

from acp_adapter.server import HermesACPAgent
from acp_adapter.session import (
    SessionManager,
    _expand_acp_enabled_toolsets,
    _normalize_acp_toolsets,
)


class _CapturingAgent:
    """Stand-in for AIAgent that records the kwargs ``_make_agent`` resolved."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.model = kwargs.get("model") or "stub-model"
        self.provider = kwargs.get("provider")
        self.base_url = kwargs.get("base_url")
        self.api_key = kwargs.get("api_key")
        self.api_mode = kwargs.get("api_mode")
        self.enabled_toolsets = kwargs.get("enabled_toolsets")
        self.disabled_toolsets = kwargs.get("disabled_toolsets")


@pytest.fixture
def acp_env(monkeypatch, tmp_path):
    """Real config/session plumbing on a throwaway HERMES_HOME; only the agent constructor,
    provider resolution and MCP discovery are stubbed."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr("run_agent.AIAgent", _CapturingAgent)
    monkeypatch.setattr(
        "hermes_cli.runtime_provider.resolve_runtime_provider",
        lambda requested=None, **_kw: {"provider": requested or "openrouter", "api_mode": "chat_completions",
                                       "base_url": "https://example.invalid/v1", "api_key": "test-key"},
    )
    monkeypatch.setattr("acp_adapter.session._register_task_cwd", lambda task_id, cwd: None)
    monkeypatch.setattr("hermes_cli.mcp_startup.ensure_mcp_discovery_before_agent_build", lambda **_kw: None)


def _manager(default_toolsets=None, db=None) -> SessionManager:
    manager = SessionManager(default_toolsets=default_toolsets, db=db)
    if db is None:
        manager._get_db = lambda: None
    return manager


# ---------------------------------------------------------------------------
# Normalization and the precedence chain
# ---------------------------------------------------------------------------


def test_normalize_accepts_arrays_strings_and_rejects_malformed():
    assert _normalize_acp_toolsets(["web", "file"]) == ["web", "file"]
    assert _normalize_acp_toolsets("web,file") == ["web", "file"]
    assert _normalize_acp_toolsets("web, file web") == ["web", "file"]  # deduped, whitespace-tolerant
    assert _normalize_acp_toolsets(["web,file", " terminal "]) == ["web", "file", "terminal"]
    # Nothing usable -> None, so the caller keeps whatever default applies.
    for empty in (None, "", "  ", ",,", [], ["", " "]):
        assert _normalize_acp_toolsets(empty) is None
    # Malformed wire values must not turn into bogus toolset names.
    for malformed in ({"web": True}, 7, True, object()):
        assert _normalize_acp_toolsets(malformed) is None


def test_precedence_session_then_process_then_platform_default(acp_env):
    manager = _manager(default_toolsets="web,file")
    assert manager._session_toolsets(None) == ["web", "file"]           # process default
    assert manager._session_toolsets(["memory"]) == ["memory"]          # per-session wins
    assert manager._session_toolsets("") == ["web", "file"]             # empty is not a selection
    assert _manager()._session_toolsets(None) is None                   # platform default


def test_default_process_and_session_are_unchanged_when_nothing_is_requested(acp_env):
    """Users who pass nothing must get byte-for-byte the historical toolset list."""
    state = _manager().create_session(cwd=".")
    assert state.toolsets is None
    assert state.agent.kwargs["enabled_toolsets"] == _expand_acp_enabled_toolsets(["hermes-acp"])


def test_scoped_session_builds_its_agent_with_only_the_requested_toolsets(acp_env):
    state = _manager().create_session(cwd=".", toolsets=["web", "file"])
    assert state.toolsets == ["web", "file"]
    assert state.agent.kwargs["enabled_toolsets"] == ["web", "file"]
    assert "hermes-acp" not in state.agent.kwargs["enabled_toolsets"]


def test_process_default_applies_to_sessions_that_do_not_scope_themselves(acp_env):
    manager = _manager(default_toolsets="web,file")
    assert manager.create_session(cwd=".").agent.kwargs["enabled_toolsets"] == ["web", "file"]
    # …and a per-session selection still overrides it.
    scoped = manager.create_session(cwd=".", toolsets="memory")
    assert scoped.agent.kwargs["enabled_toolsets"] == ["memory"]


def test_explicit_enabled_toolsets_still_win_over_the_session_scope(acp_env):
    """The rebuild seam (``enabled_toolsets=``) is authoritative: it carries a live agent's
    expanded list, ``mcp-*`` entries included, into a rebuild."""
    manager = _manager(default_toolsets="web")
    agent = manager._make_agent(session_id="s1", cwd=".", toolsets=["file"],
                                enabled_toolsets=["file", "mcp-demo"])
    assert agent.kwargs["enabled_toolsets"] == ["file", "mcp-demo"]


# ---------------------------------------------------------------------------
# _meta extraction and validation (session/new, session/load, session/resume)
# ---------------------------------------------------------------------------


def test_meta_toolsets_extraction_from_handler_kwargs(acp_env):
    """The acp library splats ``_meta`` into the handler's kwargs, so ``_meta.hermes.toolsets``
    arrives as ``kwargs["hermes"]["toolsets"]`` — see acp.router._make_func."""
    extract = HermesACPAgent._requested_toolsets
    assert extract({"hermes": {"toolsets": ["web", "file"]}}) == ["web", "file"]
    assert extract({"hermes": {"toolsets": "web,file"}}) == ["web", "file"]
    # Absent, or an explicit JSON null, is "no selection".
    assert extract({}) is None
    assert extract({"hermes": {"toolsets": None}}) is None
    assert extract({"hermes": {"sessionProvenance": {}}}) is None  # unrelated _meta.hermes payload


@pytest.mark.parametrize("meta", [
    {"toolsets": ["web"]},                    # bare key: another client's extension, never ours
    {"toolsets": []},                         # …and not an error either
    {"hermes": "not-a-dict"},                 # a foreign _meta.hermes value is ignored, not fatal
    {"hermes": []},
    {"somethingElse": {"toolsets": ["web"]}},
])
def test_only_the_namespaced_key_scopes_a_session(acp_env, meta):
    """``_meta`` is a shared extension map, so every key but ``_meta.hermes.toolsets`` is ignored
    — silently, because another client may legitimately use the name ``toolsets`` for its own
    purposes. Ignoring is safe here only because it can neither narrow nor widen anything: the
    session just keeps whatever default applies."""
    assert HermesACPAgent._requested_toolsets(meta) is None


@pytest.mark.parametrize("value", [[], "", "  ", ",,", ["", " "], {}, 7, True, {"web": True}])
def test_present_but_unusable_meta_toolsets_is_rejected_not_silently_widened(acp_env, value):
    """A client that meant to narrow must never be handed the FULL default toolset because its
    value was empty or malformed. Silent widen-on-garbage is the flaw the maintainer called out
    on #70326 — a fallback that defeats an explicit-empty contract."""
    from acp.exceptions import RequestError

    with pytest.raises(RequestError) as exc:
        HermesACPAgent._requested_toolsets({"hermes": {"toolsets": value}})
    assert exc.value.code == -32602
    assert "_meta.hermes.toolsets must be a non-empty array" in exc.value.data["details"]


def test_meta_toolsets_is_what_the_acp_library_delivers_for_session_new(acp_env):
    """Wire-format guard: ``_meta`` reaches the handler, a top-level field does not."""
    from acp.schema import NewSessionRequest

    request = NewSessionRequest.model_validate(
        {"cwd": "/tmp", "mcpServers": [], "toolsets": ["ignored"],
         "_meta": {"hermes": {"toolsets": ["web"]}}})
    assert not hasattr(request, "toolsets")  # dropped by the request model, never splatted
    kwargs = {k: getattr(request, k) for k in NewSessionRequest.model_fields if k != "field_meta"}
    kwargs.update(request.field_meta or {})
    assert HermesACPAgent._requested_toolsets(kwargs) == ["web"]


@pytest.mark.asyncio
async def test_session_new_over_the_wire_scopes_the_created_session(acp_env):
    """End to end through the real ACP router: a ``session/new`` carrying ``_meta.hermes.toolsets``
    produces a session whose agent was built with exactly those toolsets."""
    from acp.agent.router import build_agent_router

    manager = _manager()
    server = HermesACPAgent(session_manager=manager)
    result = await build_agent_router(server)(
        "session/new", {"cwd": ".", "mcpServers": [], "_meta": {"hermes": {"toolsets": ["web", "file"]}}},
        False)

    state = manager.get_session(result.session_id)
    assert state.toolsets == ["web", "file"]
    assert state.agent.kwargs["enabled_toolsets"] == ["web", "file"]


@pytest.mark.asyncio
async def test_bare_meta_toolsets_over_the_wire_does_not_scope_the_session(acp_env):
    """The un-namespaced key is somebody else's extension, so a ``session/new`` carrying it gets
    the ordinary default session — not a scoped one, and not an error."""
    from acp.agent.router import build_agent_router

    manager = _manager()
    result = await build_agent_router(HermesACPAgent(session_manager=manager))(
        "session/new", {"cwd": ".", "mcpServers": [], "_meta": {"toolsets": ["web"]}}, False)

    state = manager.get_session(result.session_id)
    assert state.toolsets is None
    assert state.agent.kwargs["enabled_toolsets"] == _expand_acp_enabled_toolsets(["hermes-acp"])


@pytest.mark.asyncio
async def test_toolset_validation_runs_off_the_event_loop(acp_env):
    """Validation imports the toolset registry and may run plugin discovery — both blocking.
    ``session/new`` already builds the agent off the loop (#58083, and
    test_session_construction_off_loop.py); the validation in front of it must not undo that."""
    import acp_adapter.server as acp_server
    from acp.agent.router import build_agent_router

    seen: list[str] = []
    real = acp_server._unknown_acp_toolsets

    def _recording(names):
        seen.append(threading.current_thread().name)
        return real(names)

    acp_server._unknown_acp_toolsets = _recording
    try:
        await build_agent_router(HermesACPAgent(session_manager=_manager()))(
            "session/new", {"cwd": ".", "mcpServers": [], "_meta": {"hermes": {"toolsets": ["web"]}}}, False)
    finally:
        acp_server._unknown_acp_toolsets = real

    assert seen and threading.current_thread().name not in seen


@pytest.mark.asyncio
async def test_unusable_meta_toolsets_over_the_wire_creates_no_session(acp_env):
    """The rejection must happen before construction: an empty selection leaves no half-built,
    full-toolset session behind."""
    from acp.agent.router import build_agent_router
    from acp.exceptions import RequestError

    manager = _manager()
    with pytest.raises(RequestError):
        await build_agent_router(HermesACPAgent(session_manager=manager))(
            "session/new", {"cwd": ".", "mcpServers": [], "_meta": {"hermes": {"toolsets": []}}}, False)
    assert manager._sessions == {}


@pytest.mark.asyncio
async def test_session_new_without_meta_is_unchanged_over_the_wire(acp_env):
    from acp.agent.router import build_agent_router

    manager = _manager()
    result = await build_agent_router(HermesACPAgent(session_manager=manager))(
        "session/new", {"cwd": ".", "mcpServers": []}, False)

    state = manager.get_session(result.session_id)
    assert state.toolsets is None
    assert state.agent.kwargs["enabled_toolsets"] == _expand_acp_enabled_toolsets(["hermes-acp"])


@pytest.mark.asyncio
async def test_session_resume_of_an_unknown_session_keeps_the_requested_scope(acp_env):
    """``session/resume`` for an id this process does not know falls back to creating a session —
    which must still honour the requested scope."""
    manager = _manager()
    server = HermesACPAgent(session_manager=manager)
    await server.resume_session(cwd=".", session_id="gone", hermes={"toolsets": ["web"]})

    # ResumeSessionResponse carries no session id, so read the one session the manager now holds.
    (state,) = manager._sessions.values()
    assert state.toolsets == ["web"]
    assert state.agent.kwargs["enabled_toolsets"] == ["web"]


def test_unknown_toolset_name_is_an_invalid_params_error(acp_env):
    """A typo must not produce a session with a silently empty toolset."""
    from acp.exceptions import RequestError

    with pytest.raises(RequestError) as exc:
        HermesACPAgent._requested_toolsets({"hermes": {"toolsets": ["web", "definitely-not-a-toolset"]}})
    assert exc.value.code == -32602
    # Only the offending name is reported, so the client can fix exactly it.
    assert exc.value.data["details"] == "Unknown toolset(s): definitely-not-a-toolset"


def test_known_and_mcp_names_are_accepted(acp_env):
    """Real registry names pass, and ``mcp-*`` (per-connection servers, registered later in the
    session) is never blocked."""
    assert HermesACPAgent._requested_toolsets({"hermes": {"toolsets": ["web", "file", "mcp-anything"]}}) == [
        "web", "file", "mcp-anything"]


# ---------------------------------------------------------------------------
# Sibling paths: a scoped session must never widen
# ---------------------------------------------------------------------------


def test_model_switch_rebuild_keeps_the_session_scope(acp_env, monkeypatch):
    """Defect (1) of the original review: ``/model`` rebuilt the agent with the default toolset.
    Exercises the real ``_switch_model`` rebuild in server.py."""
    from hermes_cli.model_switch import ModelSwitchResult

    monkeypatch.setattr(
        "hermes_cli.model_switch.switch_model",
        lambda **_kw: ModelSwitchResult(success=True, target_provider="openrouter", new_model="gpt-5.6"))

    manager = _manager()
    server = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".", toolsets=["web", "file"])

    server._switch_model(state, "gpt-5.6")

    assert state.model == "gpt-5.6"
    assert state.agent.kwargs["enabled_toolsets"] == ["web", "file"]
    assert state.toolsets == ["web", "file"]


def test_process_default_is_not_persisted_onto_the_session(acp_env, tmp_path):
    """``--toolsets`` is a property of the *process*, so it must not be baked into a session's
    persisted state: relaunch without the flag and those sessions are back on the default.
    Persisting the resolved list instead would pin one launch's flag onto the session forever."""
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    try:
        manager = _manager(default_toolsets="web", db=db)
        state = manager.create_session(cwd=str(tmp_path))
        assert state.toolsets is None                                   # no request of its own…
        assert state.agent.kwargs["enabled_toolsets"] == ["web"]        # …but the flag applied
        state.history = [{"role": "user", "content": "hi"}]
        manager.save_session(state.session_id)

        meta = json.loads(db.get_session(state.session_id)["model_config"])
        assert "toolsets" not in meta

        # A later launch without the flag restores it on the platform default, not on ``web``.
        restored = _manager(db=db).get_session(state.session_id)
        assert restored.toolsets is None
        assert restored.agent.kwargs["enabled_toolsets"] == _expand_acp_enabled_toolsets(["hermes-acp"])
    finally:
        db.close()


def test_unscoped_session_persists_no_toolsets_key_at_all(acp_env, tmp_path):
    """Default invariance: with no flag and no ``_meta``, the persistence blob is byte-for-byte
    what upstream wrote — ``session_meta`` gains a key only when the session scoped itself."""
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    try:
        manager = _manager(db=db)
        state = manager.create_session(cwd=str(tmp_path))
        state.history = [{"role": "user", "content": "hi"}]
        manager.save_session(state.session_id)

        assert "toolsets" not in json.loads(db.get_session(state.session_id)["model_config"])
    finally:
        db.close()


@pytest.mark.parametrize("persisted", ["web", 7, {"web": True}, ["", " "], None])
def test_restore_tolerates_a_malformed_persisted_selection(acp_env, tmp_path, persisted):
    """The blob is not schema-checked on the way back in (an older or hand-edited row); anything
    unusable degrades to the default rather than building a bogus toolset name into the agent."""
    manager = _manager()
    row = {"model": "m", "source": "acp", "model_config": json.dumps({"toolsets": persisted})}
    manager._get_db = lambda: types.SimpleNamespace(
        get_session=lambda _sid: row,
        get_messages_as_conversation=lambda *_a, **_kw: [])

    state = manager._restore("sid")
    expected = ["web"] if persisted == "web" else None
    assert state.toolsets == expected  # and not the raw blob value, which save_session re-persists
    assert state.agent.kwargs["enabled_toolsets"] == (
        expected or _expand_acp_enabled_toolsets(["hermes-acp"]))


def test_restore_after_process_restart_keeps_the_session_scope(acp_env, tmp_path):
    """The selection rides in the persisted ``model_config`` meta blob, so a session restored by a
    brand-new process (new SessionManager, no in-memory state) is still scoped."""
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    try:
        manager = _manager(db=db)
        state = manager.create_session(cwd=str(tmp_path), toolsets=["web", "file"])
        state.history = [{"role": "user", "content": "hi"}]
        manager.save_session(state.session_id)

        meta = json.loads(db.get_session(state.session_id)["model_config"])
        assert meta["toolsets"] == ["web", "file"]  # the base selection only

        restored = _manager(db=db).get_session(state.session_id)
        assert restored is not None
        assert restored.toolsets == ["web", "file"]
        assert restored.agent.kwargs["enabled_toolsets"] == ["web", "file"]
    finally:
        db.close()


def test_restore_does_not_resurrect_mcp_toolsets_whose_servers_are_gone(acp_env, tmp_path):
    """``mcp-*`` entries belong to a live connection's servers; persisting the expanded list would
    hand a restored session toolsets for servers that no longer exist."""
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    try:
        manager = _manager(db=db)
        state = manager.create_session(cwd=str(tmp_path), toolsets=["web"])
        state.agent.enabled_toolsets = ["web", "mcp-editor-provided"]  # as MCP registration leaves it
        state.history = [{"role": "user", "content": "hi"}]
        manager.save_session(state.session_id)

        restored = _manager(db=db).get_session(state.session_id)
        assert restored.agent.kwargs["enabled_toolsets"] == ["web"]
    finally:
        db.close()


def test_load_session_meta_toolsets_overrides_the_persisted_scope(acp_env, tmp_path):
    from hermes_state import SessionDB

    db = SessionDB(tmp_path / "state.db")
    try:
        manager = _manager(db=db)
        state = manager.create_session(cwd=str(tmp_path), toolsets=["web", "file"])
        state.history = [{"role": "user", "content": "hi"}]
        manager.save_session(state.session_id)

        fresh = _manager(db=db)
        restored = fresh.update_cwd(state.session_id, str(tmp_path), ["memory"])
        assert restored.toolsets == ["memory"]
        assert restored.agent.kwargs["enabled_toolsets"] == ["memory"]
    finally:
        db.close()


def test_fork_inherits_the_parent_scope(acp_env):
    manager = _manager()
    parent = manager.create_session(cwd=".", toolsets=["web", "file"])
    parent.history = [{"role": "user", "content": "hi"}]

    child = manager.fork_session(parent.session_id, cwd=".")

    assert child.session_id != parent.session_id
    assert child.toolsets == ["web", "file"]
    assert child.agent.kwargs["enabled_toolsets"] == ["web", "file"]


@pytest.mark.asyncio
async def test_mcp_registration_keeps_the_scope_and_adds_only_mcp_entries(acp_env):
    from unittest.mock import patch

    from acp.schema import McpServerStdio

    manager = _manager()
    server = HermesACPAgent(session_manager=manager)
    state = manager.create_session(cwd=".", toolsets=["web", "file"])
    state.agent.tools = []
    state.agent.valid_tool_names = set()

    mcp_server = McpServerStdio(name="editor-provided", command="/usr/bin/test", args=[], env=[])
    with patch("tools.mcp_tool_discovery.register_mcp_servers", return_value=[]), \
         patch("model_tools.get_tool_definitions", return_value=[]):
        await server._register_session_mcp_servers(state, [mcp_server])

    assert state.agent.enabled_toolsets == ["web", "file", "mcp-editor-provided"]
    assert state.toolsets == ["web", "file"]  # the persisted base selection is untouched


# ---------------------------------------------------------------------------
# CLI wiring: `hermes acp --toolsets a,b` reaches HermesACPAgent(default_toolsets=...)
# ---------------------------------------------------------------------------


def test_hermes_acp_toolsets_flag_reaches_the_agent_end_to_end(monkeypatch):
    """Defect (2) of the original review: ``cmd_acp`` forwarded ``args.toolsets`` but the live
    parser never declared ``--toolsets``, so the documented flag could not be parsed. Goes through
    the real ``hermes`` argparse tree and the real ``acp_adapter.entry`` parser."""
    import acp_adapter.entry as entry
    from hermes_cli.main import _build_cli_parser

    built: dict = {}
    monkeypatch.setattr(entry, "_setup_logging", lambda: None)
    monkeypatch.setattr(entry, "_load_env", lambda: None)
    monkeypatch.setattr(entry.asyncio, "run", lambda *_a, **_kw: None)
    monkeypatch.setattr("acp_adapter.server.HermesACPAgent.__init__",
                        lambda self, session_manager=None, default_toolsets=None:
                        built.update(default_toolsets=default_toolsets))
    monkeypatch.setattr("acp.run_agent", lambda *_a, **_kw: None)

    args = _build_cli_parser()[0].parse_args(["acp", "--toolsets", "web,file"])
    assert args.toolsets == "web,file"
    args.func(args)
    assert built["default_toolsets"] == "web,file"

    # And a process started that way scopes every session it creates.
    assert SessionManager(default_toolsets=built["default_toolsets"])._session_toolsets(None) == ["web", "file"]


def test_hermes_acp_without_toolsets_forwards_nothing(monkeypatch):
    """No flag -> no ``--toolsets`` on the adapter argv, so the default path is untouched."""
    from hermes_cli.main import _build_cli_parser
    from hermes_cli.main_agent_cmds import cmd_acp

    forwarded: dict = {}
    monkeypatch.setattr("acp_adapter.entry.main", lambda argv: forwarded.update(argv=argv))

    args = _build_cli_parser()[0].parse_args(["acp"])
    cmd_acp(args)
    assert forwarded["argv"] == []


def test_entry_parser_accepts_short_and_long_toolsets_flag():
    from acp_adapter.entry import _parse_args

    assert _parse_args(["--toolsets", "web,file"]).toolsets == "web,file"
    assert _parse_args(["-t", "web"]).toolsets == "web"
    assert _parse_args([]).toolsets is None


# ---------------------------------------------------------------------------
# Concurrency: one manager, many sessions
# ---------------------------------------------------------------------------


def test_concurrent_sessions_keep_independent_scopes(acp_env):
    """One SessionManager serves every chat in the process; scopes must not leak between them."""
    manager = _manager(default_toolsets="web")
    results: dict[str, types.SimpleNamespace] = {}
    scopes = {"a": ["file"], "b": ["memory"], "c": None}
    barrier = threading.Barrier(len(scopes))

    def _create(key):
        barrier.wait()
        results[key] = manager.create_session(cwd=".", toolsets=scopes[key])

    threads = [threading.Thread(target=_create, args=(key,)) for key in scopes]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert results["a"].agent.kwargs["enabled_toolsets"] == ["file"]
    assert results["b"].agent.kwargs["enabled_toolsets"] == ["memory"]
    assert results["c"].agent.kwargs["enabled_toolsets"] == ["web"]  # process default


# ---------------------------------------------------------------------------
# A persisted scope that no longer resolves
# ---------------------------------------------------------------------------


def _row_scoped_to(*names):
    return {"model": "m", "source": "acp", "model_config": json.dumps({"toolsets": list(names)})}


def _manager_restoring(row):
    manager = _manager()
    manager._get_db = lambda: types.SimpleNamespace(
        get_session=lambda _sid: row,
        get_messages_as_conversation=lambda *_a, **_kw: [])
    return manager


def test_restore_fails_loudly_when_a_persisted_toolset_no_longer_exists(acp_env, caplog):
    """A plugin uninstall (or a custom toolset dropped from config.yaml) can strand a scoped
    session. Restoring it on the platform default would hand it tools it was scoped away from;
    passing the dead name through builds an agent with no tools that still accepts prompts. So
    the restore fails, and the error names the session and the toolsets."""
    from acp.exceptions import RequestError

    manager = _manager_restoring(_row_scoped_to("file", "ghost-from-a-removed-plugin"))

    with caplog.at_level("WARNING"):
        with pytest.raises(RequestError) as excinfo:
            manager._restore("sid")

    assert excinfo.value.code == -32603
    details = excinfo.value.data["details"]
    assert "sid" in details and "ghost-from-a-removed-plugin" in details
    assert "file" not in details  # only the names that actually failed to resolve
    assert "ghost-from-a-removed-plugin" in caplog.text
    # Nothing half-built was installed, and above all nothing widened.
    assert manager._sessions == {}


def test_restore_of_a_still_valid_persisted_scope_is_unaffected(acp_env):
    """The new guard must not cost a healthy scoped session its restore: every name still
    resolves, so the session comes back scoped exactly as persisted."""
    manager = _manager_restoring(_row_scoped_to("file", "web"))

    state = manager._restore("sid")

    assert state.toolsets == ["file", "web"]
    assert state.agent.kwargs["enabled_toolsets"] == ["file", "web"]


def test_an_explicit_request_on_load_rescues_a_session_with_a_dead_persisted_scope(acp_env):
    """The persisted value is only consulted when the call makes no request of its own, so a
    client that re-scopes on ``session/load`` can still open a stranded session."""
    manager = _manager_restoring(_row_scoped_to("ghost-from-a-removed-plugin"))

    state = manager._restore("sid", toolsets=["file"])

    assert state.toolsets == ["file"]
    assert state.agent.kwargs["enabled_toolsets"] == ["file"]


def test_invalid_params_is_logged_as_one_warning_line_not_a_crash(monkeypatch):
    """An ``invalid_params`` rejection is the adapter working as designed, but the acp supervisor
    logs every background-task exception at ERROR with a traceback, so a malformed
    ``_meta.hermes.toolsets`` read like a server crash. Collapsed to one WARNING line."""
    import logging

    from acp.exceptions import RequestError

    from acp_adapter.entry import _BenignProbeMethodFilter

    def _record(exc):
        try:
            raise exc
        except RequestError:
            import sys
            return logging.LogRecord("root", logging.ERROR, __file__, 1, "Background task failed",
                                     (), sys.exc_info())

    rejection = _record(RequestError.invalid_params({"details": "toolsets must be a non-empty array"}))
    assert _BenignProbeMethodFilter().filter(rejection) is True  # still reported...
    assert rejection.exc_info is None                            # ...without the traceback
    assert rejection.levelname == "WARNING"
    assert "toolsets must be a non-empty array" in rejection.getMessage()

    # A genuine internal error keeps its ERROR level and its traceback.
    internal = _record(RequestError.internal_error({"details": "boom"}))
    assert _BenignProbeMethodFilter().filter(internal) is True
    assert internal.exc_info is not None and internal.levelname == "ERROR"
