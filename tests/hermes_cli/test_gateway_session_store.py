"""GatewaySessionStore: main's ``_session_db`` facade over gateway verbs (scripted gateway, no live one)."""
import asyncio
import types
from contextlib import asynccontextmanager

import pytest

from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError
from hermes_cli.gateway_connection import GatewayConnection
from hermes_cli.gateway_session_store import GatewaySessionStore

INFO = {"model": "fake-model", "cwd": "/w", "context_length": 128000, "enabled_toolsets": ["file"],
        "tools": {"file": ["read_file", "write_file"]}, "skills": {"devops": ["sdlc-review"]},
        "mcp_servers": [], "tool_availability": {"available_toolsets": ["file"], "unavailable_toolsets": [],
                                                 "lazy_tools": [], "disabled_tools": []}}


def snapshot(sid, revision=3):
    return {"session_id": sid, "stored_session_id": sid, "messages": [], "revision": revision,
            "execution_generation": 0, "subscription_id": "sub-" + sid, "info": dict(INFO)}


class FakeClient:
    def __init__(self, known=("local-old",)):
        self.events = asyncio.Queue()
        self.calls = []
        self.known = set(known)

    async def rpc(self, method, *, _timeout=None, **params):
        self.calls.append((method, params))
        if method == "session.create":
            self.known.add("local-new")
            return snapshot("local-new")
        if method == "session.resume":
            if params["session_id"] not in self.known:
                raise GatewayClientError("not_found")
            return snapshot(params["session_id"])
        return {"status": "ok", "detached": True}


@pytest.fixture
def store():
    client = FakeClient()

    @asynccontextmanager
    async def connect():
        yield client
    conn = GatewayConnection(connect=connect)
    conn.start(5)
    yield GatewaySessionStore(conn), client
    conn.close()


def cli(session_id="20261012_000000_abcdef", resumed=False):
    return types.SimpleNamespace(session_id=session_id, _resumed=resumed)


def test_new_session_is_created_on_the_gateway_and_its_id_adopted(store, monkeypatch):
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    db, client = store
    sid = db.open_session(cli(), {"model": "m1", "toolsets": ["file", "web"], "provider": None})
    assert sid == "local-new"
    # The store asks the owner's create contract first (does it take ``client_env``?).
    method, params = next(call for call in client.calls if call[0] != "runtime.describe")
    assert method == "session.create"
    assert params["model"] == "m1" and params["toolsets"] == ["file", "web"] and params["source"] == "cli"
    assert "provider" not in params and params["cwd"]
    assert db.connection.state("local-new").info["model"] == "fake-model"


def test_a_finite_run_creates_mains_oneshot_session_so_bare_c_finds_it(store, monkeypatch):
    """Main labels a `chat -q` / `-Q` session ``oneshot`` (agent.session_source.session_source_for), which
    `hermes -c` / `--resume latest` count as CLI history; created as ``cli`` the owner stores the
    transport's ``local`` and the next bare `-c` finds nothing."""
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    monkeypatch.delenv("HERMES_SESSION_SOURCE_EXPLICIT", raising=False)
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    db, client = store
    db.open_session(cli(), {})
    assert client.calls[-1][1]["source"] == "oneshot"
    # An explicit `--source tool` still wins, and an interactive session stays `cli`.
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "tool")
    monkeypatch.setenv("HERMES_SESSION_SOURCE_EXPLICIT", "1")
    GatewaySessionStore(db.connection).open_session(cli(), {})
    assert client.calls[-1][1]["source"] == "tool"
    for name in ("HERMES_SINGLE_QUERY_SESSION", "HERMES_SESSION_SOURCE", "HERMES_SESSION_SOURCE_EXPLICIT"):
        monkeypatch.delenv(name)
    GatewaySessionStore(db.connection).open_session(cli(), {})
    assert client.calls[-1][1]["source"] == "cli"


def test_resume_attaches_and_unknown_target_is_left_to_mains_not_found_path(store):
    db, client = store
    assert db.open_session(cli("local-old", resumed=True)) == "local-old"
    assert client.calls[-1][0] == "session.resume"
    assert db.open_session(cli("missing", resumed=True)) is None


def test_title_rename_delete_and_detach_are_gateway_verbs(store):
    db, client = store
    db.open_session(cli("local-old", resumed=True))
    db.set_session_title("local-old", "Some Title")
    mutate = [p for m, p in client.calls if m == "session.mutate"][-1]
    assert mutate["operation"] == "rename" and mutate["payload"] == {"title": "Some Title"}
    assert mutate["expected_revision"] == 3
    assert db.connection.state("local-old").title == "Some Title"
    db.end_session("local-old", "cli_close")
    assert client.calls[-1] == ("session.detach", {"session_id": "local-old", "subscription_id": "sub-local-old",
                                                   "end_reason": "cli_close"})
    db.delete_session("local-old")
    delete = [p for m, p in client.calls if m == "session.mutate"][-1]
    # The owner refuses a delete without the generation it was read at (invalid_params).
    assert delete["operation"] == "delete" and delete["expected_generation"] == 0


def _refusing_delete(client, reason="revision_conflict"):
    original = client.rpc

    async def rpc(method, *, _timeout=None, **params):
        if method == "session.mutate" and params.get("operation") == "delete":
            client.calls.append((method, params))
            raise GatewayClientError(reason)
        return await original(method, _timeout=_timeout, **params)
    client.rpc = rpc


def test_a_refused_delete_detaches_the_subscription_its_read_attached(store):
    """#134889 TUI-19: the delete's CAS read (``session.resume``) attaches a viewer; a refused delete
    must detach it, or the closed session stays in the live list until the CLI exits. A plain detach:
    an ``end_reason`` would run the owner's viewer close on the session."""
    db, client = store
    db.open_session(cli("local-old", resumed=True))
    db.end_session("local-old", "new_session")
    _refusing_delete(client)
    with pytest.raises(GatewayClientError, match="revision_conflict"):
        db.delete_session("local-old")
    assert client.calls[-1] == ("session.detach", {"session_id": "local-old", "subscription_id": "sub-local-old"})


def test_a_refused_delete_detaches_by_the_id_the_read_resolved_to(store):
    """A compressed session's history row names a segment the owner resolves to its logical owner
    and subscribes under that id; the release must detach by the resolved id, or it is a no-op."""
    db, client = store
    original = client.rpc

    async def rpc(method, *, _timeout=None, **params):
        if method == "session.resume" and params["session_id"] == "segment-1":
            client.calls.append((method, params))
            return snapshot("local-owner")
        return await original(method, _timeout=_timeout, **params)
    client.rpc = rpc
    _refusing_delete(client)
    with pytest.raises(GatewayClientError, match="revision_conflict"):
        db.delete_session("segment-1")
    assert client.calls[-1] == ("session.detach", {"session_id": "local-owner", "subscription_id": "sub-local-owner"})


def test_a_refused_delete_of_a_segment_keeps_the_resolved_session_this_viewer_watches(store):
    """The held check uses the resolved id too: the CLI watches the logical owner, the read of its
    segment reuses that subscription, and a refusal must not end the CLI's own view."""
    db, client = store
    client.known.add("local-owner")
    db.open_session(cli("local-owner", resumed=True))
    original = client.rpc

    async def rpc(method, *, _timeout=None, **params):
        if method == "session.resume" and params["session_id"] == "segment-1":
            client.calls.append((method, params))
            return snapshot("local-owner")
        return await original(method, _timeout=_timeout, **params)
    client.rpc = rpc
    _refusing_delete(client, "permission_denied")
    with pytest.raises(GatewayClientError, match="permission_denied"):
        db.delete_session("segment-1")
    assert "session.detach" not in [m for m, _ in client.calls]
    assert db.connection.state("local-owner").subscription_id == "sub-local-owner"


def test_a_refused_delete_keeps_a_session_this_cli_is_viewing(store):
    db, client = store
    db.open_session(cli("local-old", resumed=True))
    real_rpc = client.rpc

    async def refusing(method, *, _timeout=None, **params):
        if method == "session.mutate":
            raise GatewayRPCError("session_busy")
        return await real_rpc(method, _timeout=_timeout, **params)

    client.rpc = refusing
    with pytest.raises(GatewayClientError):
        db.delete_session("local-old")
    assert "session.detach" not in [method for method, _ in client.calls]


def test_gateway_owned_writes_are_noops_and_unknown_writes_refuse(store):
    db, client = store
    before = len(client.calls)
    db.create_session("x", source="cli")
    db.reopen_session("x")
    db.append_messages_batch("x", [])
    assert db.prune_empty_ghost_sessions() == 0
    assert len(client.calls) == before
    with pytest.raises(ValueError):
        db.request_handoff("x", "telegram")


def test_banner_inventory_comes_from_the_session_snapshot(store):
    db, _ = store
    assert db.banner_inventory() is None
    db.open_session(cli("local-old", resumed=True))
    inv = db.banner_inventory()
    assert [t["function"]["name"] for t in inv["tools"]] == ["read_file", "write_file"]
    assert inv["get_toolset_for_tool"]("read_file") == "file"
    assert inv["skills_by_category"] == {"devops": ["sdlc-review"]}
    assert inv["context_length"] == 128000 and inv["cwd"] == "/w"


def test_store_is_falsy_once_the_gateway_connection_closed(store):
    db, _ = store
    assert db
    db.connection.close()
    assert not db


def test_reader_opens_once_the_gateway_created_state_db(store, tmp_path, monkeypatch):
    """The startup maintenance probe runs before the gateway exists; that miss is not latched."""
    db, _ = store
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    assert db.get_session("x") is None and db._reader is None
    from hermes_state import SessionDB
    SessionDB(db_path=tmp_path / "state.db").close()
    assert db.get_session("x") is None and db._reader is not None


def test_banner_session_label_is_mains_eight_characters_for_a_canonical_id():
    from hermes_cli.gateway_session_store import session_label
    assert session_label("local-ab12cd34" + "0" * 56) == "ab12cd34"
    assert session_label("20261012_134501_abcdef") == "20261012_134501_abcdef"
    assert session_label("local-abc") == "local-abc"
    assert session_label(None) is None


def test_cli_startup_notices_render_from_the_snapshot(store, monkeypatch):
    """Main's tool-availability warning prints from the gateway snapshot (the gateway resolved the
    session's tools); the CLI process enumerates nothing."""
    from types import SimpleNamespace

    from hermes_cli.cli_info_mixin import CLIInfoMixin
    db, _ = store
    db.open_session(cli("local-old", resumed=True))
    db.current["info"].update(tool_availability_warnings=["[yellow]⚠ terminal off[/]"])
    printed = []
    host = SimpleNamespace(_session_db=db, _console_print=lambda *a: printed.append(a[0] if a else ""))
    monkeypatch.setattr("cli._build_compact_banner", lambda: "compact", raising=False)
    banner = SimpleNamespace(**vars(host), console=SimpleNamespace(clear=lambda: None), compact=True,
                             session_id="local-old", _show_status=lambda: None, agent=None,
                             _show_tool_availability_warnings=lambda: printed.append("LOCAL-WARN"))
    banner._print_gateway_tool_warnings = lambda lines: CLIInfoMixin._print_gateway_tool_warnings(banner, lines)
    CLIInfoMixin.show_banner(banner)
    assert "[yellow]⚠ terminal off[/]" in printed and "LOCAL-WARN" not in printed


def test_rename_refusal_prints_mains_title_in_use_text(store, tmp_path, monkeypatch):
    """The owner's ValueError text never crosses the wire (bounded reason codes); the rename copy
    main prints comes from the read-only reader."""
    db, client = store
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    from hermes_state import SessionDB
    writer = SessionDB(db_path=tmp_path / "state.db")
    writer.create_session("local-other", source="local")
    writer.set_session_title("local-other", "Alpha Title")
    writer.create_session("local-old", source="local")
    writer.close()
    db.open_session(cli("local-old", resumed=True))

    async def refuse(method, *, _timeout=None, **params):
        client.calls.append((method, params))
        if method == "session.mutate":
            raise GatewayClientError("request_failed")
        return snapshot(params.get("session_id", "local-old"))
    client.rpc = refuse
    with pytest.raises(ValueError) as err:
        db.set_session_title("local-old", "Alpha Title")
    assert str(err.value) == "Title 'Alpha Title' is already in use by session local-other"
    with pytest.raises(ValueError) as err:
        db.set_session_title("local-old", "Free Title")
    assert str(err.value) == "request_failed"


HISTORY = [{"role": "user", "content": "one", "row_id": 11}, {"role": "assistant", "content": "a1", "row_id": 12},
           {"role": "user", "content": "two", "row_id": 13}, {"role": "assistant", "content": "a2", "row_id": 14}]


def _with_history(client):
    base = client.rpc

    async def rpc(method, *, _timeout=None, **params):
        if method == "session.resume":
            client.calls.append((method, params))
            return {**snapshot(params["session_id"]), "messages": [dict(m) for m in HISTORY],
                    "execution_generation": 7}
        if method == "session.mutate" and params["operation"] == "rewind":
            client.calls.append((method, params))
            return {"rewound_count": 2, "target_message": {"role": "user", "content": "two"}}
        return await base(method, _timeout=_timeout, **params)
    client.rpc = rpc


def test_undo_and_retry_rewind_by_the_durable_row_id(store):
    """main's ``rewind_user_turn(sid, ordinal, warm_history=...)`` as ``session.mutate rewind``
    addressed by the row id of that user turn in the gateway's snapshot."""
    db, client = store
    _with_history(client)
    db.open_session(cli("local-old", resumed=True))
    warm = [{k: v for k, v in m.items() if k != "row_id"} for m in HISTORY]
    outcome = db.rewind_user_turn("local-old", 1, warm_history=warm)
    mutate = [p for m, p in client.calls if m == "session.mutate"][-1]
    assert mutate["operation"] == "rewind" and mutate["payload"] == {"target_message_id": 13}
    assert mutate["expected_generation"] == 7
    assert outcome.prefix == warm[:2] and outcome.live_text == "two" and outcome.rewound_count == 2
    retry = db.rewind_user_turn("local-old", -1, warm_history=warm, require_retryable=True)
    assert [p for m, p in client.calls if m == "session.mutate"][-1]["payload"] == {"target_message_id": 13}
    assert retry.live_text == "two" and retry.turns_undone == 1


def test_rewind_refuses_when_the_warm_view_diverged(store):
    db, client = store
    _with_history(client)
    db.open_session(cli("local-old", resumed=True))
    before = len([c for c in client.calls if c[0] == "session.mutate"])
    with pytest.raises(RuntimeError):
        db.rewind_user_turn("local-old", -1, warm_history=[{"role": "user", "content": "other"}])
    assert len([c for c in client.calls if c[0] == "session.mutate"]) == before


def test_new_session_id_is_a_fresh_gateway_session(store, monkeypatch):
    """/new: the id main mints is the gateway's new session (created with the profile's default model:
    an explicit -m was for the previous session only)."""
    from hermes_cli.cli_init_mixin import CLIInitMixin
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    db, client = store
    host = types.SimpleNamespace(session_id="local-old", _resumed=True, _session_db=db, model="m-explicit",
                                 _explicit_model_override=True, requested_provider=None, enabled_toolsets=["file"],
                                 _explicit_toolsets=True, _gateway_session_open=True)
    host._ensure_gateway_session = lambda: CLIInitMixin._ensure_gateway_session(host)
    db.open_session(cli("local-old", resumed=True))
    sid = CLIInitMixin._new_session_id(host, None)
    assert sid == "local-new" and host.session_id == "local-new"
    _method, params = [c for c in client.calls if c[0] == "session.create"][-1]
    assert "model" not in params and params["toolsets"] == ["file"]
    # The profile default is not a pick: the gateway resolves it, so a stale name can't refuse /new (#138670).
    host._explicit_toolsets = False
    CLIInitMixin._new_session_id(host, None)
    _method, params = [c for c in client.calls if c[0] == "session.create"][-1]
    assert params.get("toolsets") is None


def test_branch_session_id_is_the_gateway_branch_child(store):
    from hermes_cli.cli_init_mixin import CLIInitMixin
    db, client = store
    base = client.rpc

    async def rpc(method, *, _timeout=None, **params):
        if method == "session.mutate" and params["operation"] == "branch":
            client.calls.append((method, params))
            client.known.add("local-child")
            return {"branched_session_id": "local-child", "copied_messages": 2}
        return await base(method, _timeout=_timeout, **params)
    client.rpc = rpc
    db.open_session(cli("local-old", resumed=True))
    host = types.SimpleNamespace(session_id="local-old", _session_db=db, _gateway_session_open=True)
    assert CLIInitMixin._new_session_id(host, None, branch_of="local-old", title="fork") == "local-child"
    mutate = [p for m, p in client.calls if m == "session.mutate"][-1]
    assert mutate["payload"] == {"title": "fork"} and "expected_generation" in mutate
    assert "local-child" in db.snapshots


def test_new_session_id_off_the_gateway_is_mains_mint():
    from hermes_cli.cli_init_mixin import CLIInitMixin
    host = types.SimpleNamespace(_session_db=None)
    from datetime import datetime
    assert CLIInitMixin._new_session_id(host, datetime(2026, 10, 12, 1, 2, 3)).startswith("20261012_010203_")


def test_cli_listings_include_gateway_rows_created_by_the_cli(store, tmp_path, monkeypatch):
    """The gateway labels an interactive local row ``local`` (``created_source`` keeps the surface), so
    main's ``source='cli'`` listings (/resume picker, /sessions, /history, bare -c) read both labels."""
    import sqlite3
    db, _ = store
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    from hermes_state import SessionDB
    writer = SessionDB(db_path=tmp_path / "state.db")
    rows = (("local-cli", "local", "cli"), ("local-tui", "local", "tui"), ("legacy", "cli", "cli"),
            ("one", "oneshot", "oneshot"))
    for sid, source, _created in rows:
        writer.create_session(sid, source=source)
        writer.append_message(sid, "user", "hi " + sid)
    writer.close()
    conn = sqlite3.connect(tmp_path / "state.db")
    conn.executemany("UPDATE sessions SET created_source=? WHERE id=?", [(c, sid) for sid, _s, c in rows])
    conn.commit()
    conn.close()
    from hermes_cli.session_listing import query_session_listing
    listed = query_session_listing(db, source="cli", include_unnamed=True, limit=10)
    assert sorted(r["id"] for r in listed) == ["legacy", "local-cli"]
    found = db.search_sessions(source=["cli", "oneshot"], limit=10)
    assert sorted(r["id"] for r in found) == ["legacy", "local-cli", "one"]
    assert [r["id"] for r in db.search_sessions(source="tui", limit=10)] == ["local-tui"]
    assert len(db.list_sessions_rich(limit=10)) == 4


def test_create_if_missing_titles_a_gateway_created_session(store, monkeypatch):
    """``chat -c NAME --create-if-missing``: the gateway creates the titled session (its own
    resolve -> create -> title step); the CLI then resumes the id it returns. No client write."""
    db, client = store
    from hermes_cli import gateway_session_store as gss
    monkeypatch.setattr(gss, "session_store", lambda: db)
    assert gss.create_titled_session("Fresh Named") == "local-new"
    _method, params = [c for c in client.calls if c[0] == "session.create"][-1]
    assert params["title"] == "Fresh Named" and params["source"] and params["cwd"]


def test_bare_continue_finds_the_last_session_the_cli_created(tmp_path, monkeypatch):
    """Bare ``-c`` (main.py ``_resolve_last_session``) reads main's labels too: a gateway row labelled
    ``local`` that the CLI created is CLI history."""
    import sqlite3
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    from hermes_state import SessionDB
    writer = SessionDB(db_path=tmp_path / "state.db")
    writer.create_session("local-cli", source="local")
    writer.append_message("local-cli", "user", "hi")
    writer.create_session("local-tui", source="local")
    writer.append_message("local-tui", "user", "later")
    writer.close()
    conn = sqlite3.connect(tmp_path / "state.db")
    conn.executemany("UPDATE sessions SET created_source=? WHERE id=?", [("cli", "local-cli"), ("tui", "local-tui")])
    conn.commit()
    conn.close()
    monkeypatch.setattr("hermes_cli.main._resolve_workspace_key", lambda: None)
    from hermes_cli.main import _resolve_last_session
    assert _resolve_last_session("cli") == "local-cli"
    assert _resolve_last_session("tui") == "local-tui"


def test_start_and_quit_closes_the_gateway_session_with_no_agent():
    """The gateway session exists from the banner on, before any agent is built (main created its row
    lazily at the first turn): quit-immediately still ends and discards it, as main leaves no row."""
    from hermes_cli.cli_tui_runtime_mixin import CLITuiRuntimeMixin
    host = types.SimpleNamespace(agent=None, session_id="local-new", _gateway_session_open=True)
    assert CLITuiRuntimeMixin._closing_session_id(host) == "local-new"
    host._gateway_session_open = False
    assert CLITuiRuntimeMixin._closing_session_id(host) is None
    host.agent = types.SimpleNamespace(session_id="local-agent")
    assert CLITuiRuntimeMixin._closing_session_id(host) == "local-agent"


def test_yolo_toggle_persists_as_the_gateway_session_yolo(store):
    """Main's ``/yolo`` persist hook (``db.set_session_yolo``) is the gateway's ``config.set yolo`` with an
    explicit value: approvals run in the gateway, so its bypass is the one that matters."""
    db, client = store
    db.open_session(cli("local-old", resumed=True))
    db.set_session_yolo("local-old", True)
    db.set_session_yolo("local-old", False)
    sets = [p for m, p in client.calls if m == "config.set"]
    assert sets == [{"session_id": "local-old", "key": "yolo", "value": "1"},
                    {"session_id": "local-old", "key": "yolo", "value": "0"}]


def test_new_session_before_the_gateway_session_opened_is_mains_placeholder(store):
    from hermes_cli.cli_init_mixin import CLIInitMixin
    from datetime import datetime
    db, client = store
    host = types.SimpleNamespace(session_id="x", _session_db=db, _gateway_session_open=False)
    assert CLIInitMixin._new_session_id(host, datetime(2026, 10, 12, 1, 2, 3)).startswith("20261012_010203_")
    assert not [c for c in client.calls if c[0] == "session.create"]


def test_run_budget_flag_rides_session_create():
    from hermes_cli.gateway_session_store import _creation_policy
    assert _creation_policy({"run_budget": 90})["run_budget"] == 90.0
    assert "run_budget" not in _creation_policy({"run_budget": None})


def test_verbose_flag_rides_session_create():
    from hermes_cli.gateway_session_store import _creation_policy
    assert _creation_policy({"verbose": True})["verbose"] is True
    assert "verbose" not in _creation_policy({"verbose": None})


def test_row_cwd_is_mains_stamp_while_the_tools_run_in_terminal_cwd(store, monkeypatch, tmp_path):
    """Main's row cwd: the launching process's cwd on its first session (``-w`` never chdirs, so the
    repo root), none on ``/new`` and ``--create-if-missing``. The execution cwd stays TERMINAL_CWD."""
    from hermes_cli.cli_init_mixin import CLIInitMixin
    from hermes_cli import gateway_session_store as gss
    db, client = store
    repo, tree = tmp_path / "repo", tmp_path / "repo" / ".worktrees" / "wt"
    tree.mkdir(parents=True)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("TERMINAL_CWD", str(tree))
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    db.open_session(cli(), {"toolsets": ["file"]})
    first = [p for m, p in client.calls if m == "session.create"][-1]
    assert first["cwd"] == str(tree.resolve()) and first["row_cwd"] == str(repo.resolve())
    host = types.SimpleNamespace(session_id="local-new", _session_db=db, _gateway_session_open=True)
    CLIInitMixin._new_session_id(host, None)
    assert [p for m, p in client.calls if m == "session.create"][-1]["row_cwd"] == ""
    monkeypatch.setattr(gss, "session_store", lambda: db)
    gss.create_titled_session("Fresh Named")
    assert [p for m, p in client.calls if m == "session.create"][-1]["row_cwd"] == ""
    monkeypatch.setenv("TERMINAL_ENV", "docker")  # main stamps no host cwd for a remote backend
    db.create({"toolsets": ["file"]})
    assert [p for m, p in client.calls if m == "session.create"][-1]["row_cwd"] == ""


@pytest.mark.parametrize("backend", ["docker", "ssh"])
def test_a_sandbox_backend_sends_the_host_launch_dir_not_its_terminal_cwd(store, monkeypatch, tmp_path, backend):
    """A docker/ssh ``terminal.cwd`` (bridged to TERMINAL_CWD) names a path inside the sandbox. The
    owner checks ``cwd`` on its own host, so sending it refused every launch with ``invalid_params``;
    like the thin client and the TUI, the classic CLI sends its host launch dir."""
    from hermes_cli import gateway_session_store as gss
    db, client = store
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TERMINAL_CWD", "/project/not-on-this-host")
    monkeypatch.setenv("TERMINAL_ENV", backend)
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    db.create({"toolsets": ["file"]})
    assert [p for m, p in client.calls if m == "session.create"][-1]["cwd"] == str(tmp_path.resolve())
    monkeypatch.setattr(gss, "session_store", lambda: db)
    gss.create_titled_session("Remote Named")
    assert [p for m, p in client.calls if m == "session.create"][-1]["cwd"] == str(tmp_path.resolve())


def test_a_mounting_docker_backend_sends_the_w_worktree_not_the_launch_dir(store, monkeypatch, tmp_path):
    """``-w`` puts its host worktree in TERMINAL_CWD without a chdir. A docker backend mounting the
    session cwd at ``/workspace`` would otherwise mount the original checkout, so the tools edit it
    instead of the isolated tree."""
    db, client = store
    repo, tree = tmp_path / "repo", tmp_path / "repo" / ".worktrees" / "wt"
    tree.mkdir(parents=True)
    monkeypatch.chdir(repo)
    monkeypatch.setenv("TERMINAL_CWD", str(tree))
    monkeypatch.setenv("TERMINAL_ENV", "docker")
    monkeypatch.setenv("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "true")
    monkeypatch.delenv("HERMES_SESSION_SOURCE", raising=False)
    db.create({"toolsets": ["file"]})
    assert [p for m, p in client.calls if m == "session.create"][-1]["cwd"] == str(tree.resolve())


def test_explicit_reasoning_max_turns_and_checkpoints_reach_session_create(store, monkeypatch):
    """Main's HermesCLI built AIAgent with the run's `--reasoning`, `--max-turns` and `--checkpoints`;
    the gateway now builds the agent, so `session.create` must carry them or the run silently uses
    the profile default. Unset flags stay unset (the gateway resolves the profile default)."""
    from types import SimpleNamespace

    from hermes_cli.cli_init_mixin import CLIInitMixin
    db, client = store

    def host(**flags):
        h = SimpleNamespace(model="m1", requested_provider="auto", enabled_toolsets=["terminal"],
                            session_id="20261012_000000_abcdef", _resumed=False, _session_db=GatewaySessionStore(db.connection))
        CLIInitMixin._init_turn_limits(h, flags.get("max_turns"), None)
        CLIInitMixin._init_checkpoints_and_rules(h, flags.get("checkpoints", False), False, False)
        CLIInitMixin._init_prompt_and_reasoning(h, flags.get("reasoning"))
        return h

    CLIInitMixin._ensure_gateway_session(host(reasoning="high", max_turns=1, checkpoints=True))
    params = client.calls[-1][1]
    assert (params.get("reasoning"), params.get("max_turns"), params.get("checkpoints")) == ("high", 1, True), params
    CLIInitMixin._ensure_gateway_session(host())
    assert not {"reasoning", "max_turns", "checkpoints"} & set(client.calls[-1][1]), client.calls[-1][1]
    # An unparseable --reasoning keeps main's warn-and-ignore (never an invalid_params refusal).
    CLIInitMixin._ensure_gateway_session(host(reasoning="bogus"))
    assert "reasoning" not in client.calls[-1][1]


def test_resume_hands_the_cli_lease_over_while_the_gateway_imports_the_row(store):
    """The gateway imports a resumed legacy row under the active-session registry lock and refuses
    (runtime_coordination_required) while any live lease holds that id. The -q/-Q path claims the
    CLI's lease before the first turn, so the attach must run without it and re-claim afterwards."""
    from types import SimpleNamespace

    from hermes_cli.cli_init_mixin import CLIInitMixin
    db, client = store
    events = []
    h = SimpleNamespace(model="m1", requested_provider="auto", enabled_toolsets=["terminal"],
                        session_id="local-old", _resumed=True, _session_db=GatewaySessionStore(db.connection),
                        _active_session_lease=object())
    h._release_active_session = lambda: (events.append("release"), setattr(h, "_active_session_lease", None))
    h._claim_active_session = lambda *a, **k: (events.append(("claim", h.session_id)), True)[1]
    base = client.rpc

    async def rpc(method, **params):
        events.append((method, h._active_session_lease is None))
        return await base(method, **params)
    client.rpc = rpc
    CLIInitMixin._ensure_gateway_session(h)
    assert events == ["release", ("session.resume", True), ("claim", "local-old")], events


# Regressions for #134889 (classic CLI launch flags reaching the gateway-owned session).

def test_skills_flag_reaches_session_create_without_auto_load_names(store, monkeypatch):
    """CLI-1: `-s/--skills` rides ``session.create`` (the gateway renders it into the session once); the
    CLI's own system prompt never reaches the gateway. ``skills.auto_load`` names the gateway loads itself."""
    from hermes_cli.cli_init_mixin import CLIInitMixin
    monkeypatch.setattr("agent.skill_commands.resolve_auto_load_skills", lambda config=None: ["pinned"])
    db, client = store
    host = types.SimpleNamespace(model="m1", requested_provider="auto", enabled_toolsets=["file"],
                                 session_id="20261012_000000_abcdef", _resumed=False, _session_db=db,
                                 _preload_skills_requested=["pinned", "arxiv"])
    CLIInitMixin._ensure_gateway_session(host)
    assert [p for m, p in client.calls if m == "session.create"][-1]["skills"] == ["arxiv"]


def _frozen_launch(client, launch_request):
    base = client.rpc

    async def rpc(method, *, _timeout=None, **params):
        result = await base(method, _timeout=_timeout, **params)
        if method == "session.resume":
            result["info"]["launch_request"] = dict(launch_request)
        return result
    client.rpc = rpc


def test_explicit_flags_on_resume_are_checked_against_the_frozen_route(store):
    """CLI-2: `--resume` / `-c` with `-m`, `--max-turns`, `--checkpoints`, ... is refused when it would
    change the frozen route (as the thin client refuses it), never silently ignored; repeating it is fine."""
    db, client = store
    _frozen_launch(client, {"model": "m-frozen", "model_requested": "m-alias", "max_turns": 5, "run_budget": 60.0})
    for launch in ({"max_turns": 1}, {"model": "m-other"}, {"checkpoints": True}, {"run_budget": 30},
                   {"verbose": True}, {"skills": ["arxiv"]}):
        with pytest.raises(GatewayClientError, match="Resume retains gateway session policy"):
            GatewaySessionStore(db.connection).open_session(cli("local-old", resumed=True), launch)
    for launch in ({}, {"max_turns": 5, "model": "m-alias", "run_budget": 60, "toolsets": ["web"]}):
        assert GatewaySessionStore(db.connection).open_session(cli("local-old", resumed=True), launch) == "local-old"


def test_explicit_toolsets_on_resume_are_checked_against_the_frozen_route(store):
    """CLI-2 `-t`: only an explicit `-t` (``cli._typed_toolsets``) is checked on resume; the CLI's
    resolved default selection in ``launch`` never refuses one."""
    db, client = store
    _frozen_launch(client, {"toolsets": ["file"]})
    host = cli("local-old", resumed=True)
    host._typed_toolsets = ["web"]
    with pytest.raises(GatewayClientError, match="Resume retains gateway session policy"):
        GatewaySessionStore(db.connection).open_session(host, {"toolsets": ["web"]})
    host._typed_toolsets = ["file"]
    assert GatewaySessionStore(db.connection).open_session(host, {"toolsets": ["file"]}) == "local-old"
    host._typed_toolsets = None
    assert GatewaySessionStore(db.connection).open_session(host, {"toolsets": ["web"]}) == "local-old"


def test_unchanged_model_flag_on_resume_ignores_the_configured_provider(store, monkeypatch):
    """CLI-2: `--resume <id> -m my-model` repeats the session's frozen `-m` while the CLI resolved the
    profile's provider; only a `--provider` the user typed is launch policy, so the repeat is accepted
    (as the thin client accepts it) and a different `--provider` is still refused."""
    import cli as cli_module
    from hermes_cli.cli_init_mixin import CLIInitMixin
    monkeypatch.setitem(cli_module.CLI_CONFIG, "model", {"default": "cfg-model", "provider": "openrouter"})
    db, client = store
    _frozen_launch(client, {"model": "my-model"})

    def host(provider=None, resumed=True):
        h = types.SimpleNamespace(session_id="local-old", _resumed=resumed, enabled_toolsets=["file"],
                                  _session_db=GatewaySessionStore(db.connection))
        CLIInitMixin._init_model_and_provider(h, "my-model", provider, None, None)
        return h

    repeat = host()
    assert repeat.requested_provider == "openrouter"
    CLIInitMixin._ensure_gateway_session(repeat)
    assert repeat.session_id == "local-old"
    with pytest.raises(GatewayClientError, match="Resume retains gateway session policy"):
        CLIInitMixin._ensure_gateway_session(host("anthropic"))
    # Creation sends a typed --provider with -m, and leaves the configured one to the gateway.
    CLIInitMixin._ensure_gateway_session(host("anthropic", resumed=False))
    assert [p for m, p in client.calls if m == "session.create"][-1]["provider"] == "anthropic"
    CLIInitMixin._ensure_gateway_session(host(resumed=False))
    created = [p for m, p in client.calls if m == "session.create"][-1]
    assert created["model"] == "my-model" and "provider" not in created, created


def test_new_session_keeps_the_runs_launch_policy(store, monkeypatch):
    """CLI-3: `/new` keeps `--max-turns`, `--checkpoints`, `--run-budget`, `-v` and `-s` (main kept one
    AIAgent across /new); only the explicit model route is left to the profile default."""
    from hermes_cli.cli_init_mixin import CLIInitMixin
    monkeypatch.setattr("agent.skill_commands.resolve_auto_load_skills", lambda config=None: [])
    db, client = store
    db.open_session(cli("local-old", resumed=True))
    host = types.SimpleNamespace(session_id="local-old", _session_db=db, _gateway_session_open=True,
                                 model="m-explicit", _explicit_model_override=True, requested_provider="openrouter",
                                 enabled_toolsets=["file"], _explicit_toolsets=True, _explicit_max_turns=3, _explicit_checkpoints=True,
                                 _explicit_run_budget=60, verbose=True, _preload_skills_requested=["arxiv"])
    CLIInitMixin._new_session_id(host, None)
    params = [p for m, p in client.calls if m == "session.create"][-1]
    assert (params["max_turns"], params["checkpoints"], params["run_budget"], params["verbose"],
            params["skills"], params["toolsets"]) == (3, True, 60.0, True, ["arxiv"], ["file"]), params
    assert "model" not in params and "provider" not in params


def test_branch_carries_a_live_yolo_onto_the_gateway_child(store):
    """CLI-20: the gateway's branch child starts without the parent's `/yolo` toggle while the CLI moves
    its own flag over (status bar YOLO); the CLI sets the child's bypass on the gateway too."""
    from hermes_cli.cli_init_mixin import CLIInitMixin
    from tools import approval
    db, client = store
    base = client.rpc

    async def rpc(method, *, _timeout=None, **params):
        if method == "session.mutate" and params["operation"] == "branch":
            client.calls.append((method, params))
            child = f"local-child-{len(client.calls)}"
            client.known.add(child)
            return {"branched_session_id": child, "copied_messages": 0}
        return await base(method, _timeout=_timeout, **params)
    client.rpc = rpc
    db.open_session(cli("local-old", resumed=True))
    host = types.SimpleNamespace(session_id="local-old", _session_db=db, _gateway_session_open=True)
    plain = CLIInitMixin._new_session_id(host, None, branch_of="local-old")
    assert not [p for m, p in client.calls if m == "config.set"]
    approval.enable_session_yolo("local-old")
    try:
        child = CLIInitMixin._new_session_id(host, None, branch_of="local-old")
    finally:
        for key in ("local-old", plain, "local-child"):
            approval.disable_session_yolo(key)
    assert [p for m, p in client.calls if m == "config.set"] == [{"session_id": child, "key": "yolo", "value": "1"}]


def test_branch_never_shows_yolo_the_gateway_refused(store):
    """CLI-20: approvals run on the gateway, so a failed ``config.set yolo`` on the child surfaces instead of
    being logged as a best-effort persist, and the child's local flag stays off (no YOLO display while the
    gateway still asks)."""
    from hermes_cli.cli_init_mixin import CLIInitMixin
    from tools import approval
    db, client = store
    base = client.rpc

    async def rpc(method, *, _timeout=None, **params):
        if method == "session.mutate" and params["operation"] == "branch":
            client.known.add("local-child")
            return {"branched_session_id": "local-child", "copied_messages": 0}
        if method == "config.set":
            raise GatewayClientError("Gateway did not answer config.set")
        return await base(method, _timeout=_timeout, **params)
    client.rpc = rpc
    db.open_session(cli("local-old", resumed=True))
    host = types.SimpleNamespace(session_id="local-old", _session_db=db, _gateway_session_open=True)
    approval.enable_session_yolo("local-old")
    try:
        with pytest.raises(GatewayClientError, match="config.set"):
            CLIInitMixin._new_session_id(host, None, branch_of="local-old")
        assert not approval.is_session_yolo_enabled("local-child")
    finally:
        for key in ("local-old", "local-child"):
            approval.disable_session_yolo(key)


def test_exported_ignore_rules_never_refuses_a_bare_resume(store, monkeypatch):
    """CLI-2: an exported HERMES_IGNORE_RULES is creation policy only, so a bare `hermes -c` continues a
    session created without it (Desktop, Ink, before the export); a typed --ignore-rules is still refused,
    as the thin client refuses it."""
    from hermes_cli.cli_init_mixin import CLIInitMixin, _gateway_launch
    monkeypatch.setenv("HERMES_IGNORE_RULES", "1")
    db, client = store
    _frozen_launch(client, {"model": "m-frozen"})

    def host(typed):
        h = cli("local-old", resumed=True)
        CLIInitMixin._init_checkpoints_and_rules(h, False, False, typed)
        return h

    bare = host(False)
    assert bare.ignore_rules and _gateway_launch(bare)["ignore_rules"] is True  # a new session still gets it
    assert GatewaySessionStore(db.connection).open_session(bare, _gateway_launch(bare)) == "local-old"
    typed = host(True)
    with pytest.raises(GatewayClientError, match="Resume retains gateway session policy"):
        GatewaySessionStore(db.connection).open_session(typed, _gateway_launch(typed))


def test_resume_refusal_names_the_clashing_flag_and_the_way_forward(store):
    """CLI-2: the refusal (one text for both clients) names the flag that differs from the frozen launch
    and how to go on: resume without it, then the slash command that switches it live."""
    import argparse
    from hermes_cli.gateway_chat import check_resume_policy
    db, client = store
    _frozen_launch(client, {"model": "m-frozen", "max_turns": 5})
    host = cli("local-old", resumed=True)
    with pytest.raises(GatewayClientError, match="Resume retains gateway session policy") as refused:
        GatewaySessionStore(db.connection).open_session(host, {"model": "m-other"})
    assert "-m/--model differs" in str(refused.value) and "use /model after resuming" in str(refused.value)
    with pytest.raises(GatewayClientError, match="--max-turns differs") as refused:
        GatewaySessionStore(db.connection).open_session(host, {"max_turns": 1})
    assert "start a new session" in str(refused.value)
    with pytest.raises(GatewayClientError, match="--run-budget differs"):
        GatewaySessionStore(db.connection).open_session(host, {"run_budget": 30})
    # The thin client raises the same text.
    with pytest.raises(GatewayClientError, match="--provider differs .* use /model after resuming"):
        check_resume_policy(argparse.Namespace(provider="anthropic"),
                            {"info": {"launch_request": {"provider": "openrouter"}}})


def test_clear_banner_shows_the_new_gateway_sessions_tools_and_short_label(store, monkeypatch):
    """#134889 CLI-22: `/clear` inside the running CLI redraws the banner from the new gateway
    session's snapshot (its MCP tools, main's 8-character label), as the startup banner does; this
    process's own tool registry has no MCP servers and is never walked."""
    import cli as cli_mod
    from cli import HermesCLI
    db, client = store
    canonical = "local-ab12cd34" + "e" * 56
    info = {**INFO, "tools": {"file": ["read_file"], "mcp-zeta": ["mcp__zeta__ro_probe"]}}
    created = {**snapshot(canonical), "info": info}

    async def rpc(method, *, _timeout=None, **params):
        client.calls.append((method, params))
        return created if method == "session.create" else {}
    client.rpc = rpc
    banners = []
    monkeypatch.setattr("hermes_cli.banner.build_welcome_banner", lambda **kw: banners.append(kw))
    monkeypatch.setattr(cli_mod, "get_tool_definitions", lambda **_kw: pytest.fail("walked the CLI's registry"))
    monkeypatch.setattr(cli_mod, "_cprint", lambda *_a: None)
    monkeypatch.setattr("shutil.get_terminal_size", lambda *_a: types.SimpleNamespace(columns=200, lines=50))
    shell = HermesCLI.__new__(HermesCLI)
    shell.__dict__.update(
        _session_db=db, session_id="local-old", compact=False, model="fake-model", provider="custom",
        enabled_toolsets=["hermes-cli"], disabled_toolsets=[], agent=None,
        _app=types.SimpleNamespace(output=types.SimpleNamespace(
            erase_screen=lambda: None, cursor_goto=lambda *_a: None, flush=lambda: None)))
    shell._confirm_destructive_slash = lambda *_a, **_k: "once"
    shell._print_random_tip = lambda: None
    shell.new_session = lambda silent=False: setattr(shell, "session_id", db.create({}))

    shell._cmd_clear("/clear")

    (banner,) = banners
    assert banner["session_id"] == "ab12cd34"
    assert sorted(t["function"]["name"] for t in banner["tools"]) == ["mcp__zeta__ro_probe", "read_file"]
    assert banner["get_toolset_for_tool"]("mcp__zeta__ro_probe") == "mcp-zeta"
