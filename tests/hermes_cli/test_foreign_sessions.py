"""Tests for hermes_cli.foreign_sessions — Claude Code / Codex CLI import.

Fixture JSONL is synthesized inline (tmp_path); imports reach a gateway owner over a temp
state.db, so nothing touches the real HERMES_HOME store.
"""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.foreign_sessions import (
    _list_sessions,
    gather_foreign_sessions,
    import_foreign_session,
    parse_claude_session,
    parse_codex_session,
)
import itertools


# ── fixture builders ─────────────────────────────────────────────────────


def _claude_lines():
    def msg(role, content):
        return {
            "type": role,
            "sessionId": "abc-123",
            "cwd": "/home/user/proj",
            "message": {"role": role, "content": content},
        }

    return [
        {"type": "summary", "summary": "Fix the flaky test"},
        msg("user", "Please fix the flaky test in CI."),
        msg(
            "assistant",
            [
                {"type": "text", "text": "Looking into it now."},
                {"type": "tool_use", "name": "Bash", "id": "t1", "input": {}},
            ],
        ),
        # tool_result echoed back as a user message — must NOT become a turn
        msg("user", [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]),
        msg("assistant", [{"type": "text", "text": "Fixed — the sleep was too short."}]),
        msg("user", "Great, thanks!"),
        msg("assistant", [{"type": "text", "text": "Anytime."}]),
    ]


def _write_claude_fixture(tmp_path, extra_lines=None):
    proj = tmp_path / ".claude" / "projects" / "-home-user-proj"
    proj.mkdir(parents=True)
    f = proj / "abc-123.jsonl"
    lines = [json.dumps(entry) for entry in _claude_lines()]
    if extra_lines:
        lines = extra_lines + lines
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f


def _codex_lines():
    def item(payload):
        return {"timestamp": "2026-08-15T21:35:28Z", "type": "response_item", "payload": payload}

    def message(role, kind, text):
        return item({"type": "message", "role": role, "content": [{"type": kind, "text": text}]})

    return [
        {
            "type": "session_meta",
            "payload": {"session_id": "0000-1111", "cwd": "/home/user/repo"},
        },
        # developer/system payloads must never be imported
        message("developer", "input_text", "<skills_instructions>secret system stuff"),
        message("user", "input_text", "<recommended_plugins>\ninjected wrapper"),
        message("user", "input_text", "Summarize the transcripts please."),
        message("assistant", "output_text", "Reading them one at a time."),
        item({"type": "custom_tool_call", "name": "shell", "call_id": "c1"}),
        item({"type": "custom_tool_call_output", "call_id": "c1", "output": "big output"}),
        message("assistant", "output_text", "Done — here is the summary."),
        message("user", "input_text", "Now write it to a file."),
        message("assistant", "output_text", "Written to summary.md."),
    ]


def _write_codex_fixture(tmp_path, extra_lines=None):
    day = tmp_path / ".codex" / "sessions" / "2026" / "08" / "15"
    day.mkdir(parents=True)
    f = day / "rollout-2026-08-15T21-35-28-0000-1111.jsonl"
    lines = [json.dumps(entry) for entry in _codex_lines()]
    if extra_lines:
        lines = extra_lines + lines
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return f


def _route_store(monkeypatch, mutate):
    """Answer the importer's gateway calls with *mutate* (``session.mutate`` params -> receipt)."""
    from hermes_cli.gateway_client import GatewayClientError
    from hermes_state_runtime import RuntimeStoreError
    calls = []

    def call(method, **params):
        calls.append((method, params["operation"]))
        try:
            return mutate(params)
        except (RuntimeStoreError, ValueError) as exc:  # a refusal crosses the wire as a bounded code
            raise GatewayClientError(getattr(exc, "reason", "invalid_params")) from exc

    monkeypatch.setattr("hermes_cli.gateway_session_store.session_store", lambda: SimpleNamespace(call=call))
    return calls


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    """A gateway owner over a temp state.db that the importer's ``session.mutate`` calls reach."""
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import Principal, SessionRef
    from gateway.session_mutations import mutate_session
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch

    db = SessionDB(db_path=tmp_path / "state.db")
    authority = SessionAuthority(SimpleNamespace(_draining=False), profile_id="p", instance_id="owner",
                                 db=db, epoch=begin_runtime_epoch(db, instance_id="owner"))
    actor = Principal("owner", "p", frozenset({"session:read", "session:control", "session:create"}), "cli")
    calls = _route_store(monkeypatch, lambda params: asyncio.run(
        mutate_session(authority, actor, SessionRef("p", params["session_id"]), params)))
    yield SimpleNamespace(db=db, calls=calls)
    db.close()


def _assert_alternating(messages):
    roles = [m["role"] for m in messages]
    assert roles, "no messages"
    assert roles[0] == "user"
    for a, b in itertools.pairwise(roles):
        assert a != b, f"two consecutive {a} messages"


# ── parsing ──────────────────────────────────────────────────────────────


def test_parse_claude_session(tmp_path):
    f = _write_claude_fixture(tmp_path)
    parsed = parse_claude_session(f)
    turns = parsed["turns"]
    _assert_alternating(turns)
    assert len(turns) == 4  # tool_result-only user line skipped; assistants merge
    assert parsed["cwd"] == "/home/user/proj"
    assert parsed["title_guess"] == "Fix the flaky test"
    assert parsed["session_id"] == "abc-123"
    # tool_use flattened to a bracketed summary, merged with adjacent text
    joined_assistant = "\n".join(t["content"] for t in turns if t["role"] == "assistant")
    assert "[ran tool: Bash]" in joined_assistant
    # no fabricated tool_call structures
    assert all(set(t) == {"role", "content"} for t in turns)


def test_parse_codex_session(tmp_path):
    f = _write_codex_fixture(tmp_path)
    parsed = parse_codex_session(f)
    turns = parsed["turns"]
    _assert_alternating(turns)
    assert len(turns) == 4
    assert parsed["cwd"] == "/home/user/repo"
    assert parsed["session_id"] == "0000-1111"
    assert parsed["title_guess"].startswith("Summarize the transcripts")
    # developer + wrapper user lines excluded
    all_text = "\n".join(t["content"] for t in turns)
    assert "skills_instructions" not in all_text
    assert "recommended_plugins" not in all_text
    # tool call summarized in assistant text
    assert "[ran tool: shell]" in all_text


def test_malformed_lines_are_skipped(tmp_path):
    garbage = [
        "this is not json at all {{{",
        '"just a string"',
        json.dumps({"type": "user", "message": "not-a-dict-payload... wait, string"}),
        json.dumps({"type": "user"}),  # missing message
        "",
    ]
    f = _write_claude_fixture(tmp_path, extra_lines=garbage)
    parsed = parse_claude_session(f)
    assert len(parsed["turns"]) == 4  # good lines still parse

    g = _write_codex_fixture(tmp_path, extra_lines=["not json", json.dumps({"type": "response_item"})])
    parsed_codex = parse_codex_session(g)
    assert len(parsed_codex["turns"]) == 4


# ── discovery ────────────────────────────────────────────────────────────


def test_list_sessions(tmp_path):
    _write_claude_fixture(tmp_path)
    _write_codex_fixture(tmp_path)
    claude = _list_sessions("claude", tmp_path / ".claude" / "projects")
    codex = _list_sessions("codex", tmp_path / ".codex" / "sessions")
    assert len(claude) == 1 and claude[0].source == "claude"
    assert claude[0].turn_count == 4
    assert len(codex) == 1 and codex[0].source == "codex"
    assert codex[0].turn_count == 4
    both = gather_foreign_sessions(
        claude_root=tmp_path / ".claude" / "projects",
        codex_root=tmp_path / ".codex" / "sessions",
    )
    assert len(both) == 2
    assert both[0].mtime >= both[1].mtime  # newest first


def test_list_sessions_missing_roots(tmp_path):
    assert _list_sessions("claude", tmp_path / "nope") == []
    assert _list_sessions("codex", tmp_path / "nope") == []


def test_env_overrides_relocate_default_roots(tmp_path, monkeypatch):
    """CLAUDE_CONFIG_DIR / CODEX_HOME relocate discovery (ported from cline/cline#13827)."""
    claude_cfg = tmp_path / "relocated-claude"
    codex_home = tmp_path / "relocated-codex"
    _write_claude_fixture(tmp_path)  # writes under tmp_path/.claude — becomes the store root below
    (tmp_path / ".claude").rename(claude_cfg)
    _write_codex_fixture(tmp_path)
    (tmp_path / ".codex").rename(codex_home)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)  # default roots are empty
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(claude_cfg))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    both = gather_foreign_sessions()
    assert {s.source for s in both} == {"claude", "codex"}


def test_blank_env_overrides_fall_back_to_home(tmp_path, monkeypatch):
    """A blank/whitespace override is unset, not a CWD-relative path (cline/cline#13827)."""
    _write_claude_fixture(tmp_path)
    _write_codex_fixture(tmp_path)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "")
    monkeypatch.setenv("CODEX_HOME", "   ")
    both = gather_foreign_sessions()
    assert {s.source for s in both} == {"claude", "codex"}


# ── import through the gateway ───────────────────────────────────────────


def test_import_claude_session(tmp_path, gateway):
    f = _write_claude_fixture(tmp_path)
    session_id = import_foreign_session("claude", f)
    row = gateway.db.get_session(session_id)
    assert row is not None
    assert row["source"] == "cli"
    assert row["origin_json"] is None
    assert row["cwd"] is None  # /home/user/proj is not a folder here, and resume would refuse it
    assert row["message_count"] == 4
    title = gateway.db.get_session_title(session_id)
    assert title.startswith("Imported from Claude Code: ")
    assert "Please fix the flaky test" in title
    messages = gateway.db.get_messages(session_id)
    assert len(messages) == 4
    _assert_alternating(messages)
    # A re-import keeps its conversation; only the taken (UNIQUE) title is dropped.
    again = import_foreign_session("claude", f)
    assert gateway.db.get_session(again)["message_count"] == 4
    assert gateway.db.get_session_title(again) is None


def test_import_codex_session(tmp_path, gateway):
    f = _write_codex_fixture(tmp_path)
    f.write_text(f.read_text(encoding="utf-8").replace("/home/user/repo", str(tmp_path)), encoding="utf-8")
    session_id = import_foreign_session("@codex", f)
    row = gateway.db.get_session(session_id)
    assert row is not None
    assert row["source"] == "cli"
    assert row["cwd"] == str(tmp_path)
    assert row["message_count"] == 4
    title = gateway.db.get_session_title(session_id)
    assert title.startswith("Imported from Codex CLI: ")
    messages = gateway.db.get_messages(session_id)
    _assert_alternating(messages)


def test_import_rejects_bad_input(tmp_path, gateway):
    with pytest.raises(ValueError, match="Unknown foreign session source"):
        import_foreign_session("gemini", tmp_path / "x.jsonl")
    with pytest.raises(ValueError, match="not found"):
        import_foreign_session("claude", tmp_path / "missing.jsonl")
    empty = tmp_path / "empty.jsonl"
    empty.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="No user/assistant conversation"):
        import_foreign_session("claude", empty)
    assert gateway.calls == []


def test_import_is_written_by_the_gateway_not_this_process(tmp_path, gateway, monkeypatch):
    """The gateway is the only state.db writer (#106742): the importer only sends ``session.mutate``."""
    from hermes_state import SessionDB
    from hermes_state_mutation_binding import history_binding

    def refused(*_a, **_k):
        raise AssertionError("the importer wrote state.db itself")
    monkeypatch.setattr(SessionDB, "create_session", refused)
    monkeypatch.setattr(SessionDB, "append_message", refused)

    session_id = import_foreign_session("claude", _write_claude_fixture(tmp_path))

    assert gateway.calls == [("session.mutate", "import"), ("session.mutate", "rename")]
    with gateway.db._read_ctx() as conn:  # the import receipt bound the row to its importer
        assert history_binding(conn, session_id) == json.dumps(["p", "owner"], separators=(",", ":"))


@pytest.mark.asyncio
async def test_imported_chat_resumes_writable_as_cli(tmp_path, monkeypatch):
    """``hermes --resume @claude`` continues the import as an ordinary CLI session, not a read-only mount."""
    from gateway import run
    from gateway.config import GatewayConfig
    from gateway.session import SessionStore
    from gateway.session_authority import initialize_session_authority
    from gateway.session_contract import SessionRef
    from gateway.session_controls import AuthorityConnection
    monkeypatch.setattr(run, "_load_gateway_config", lambda: {"model": {"default": "m1", "provider": "openrouter"}})
    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    runner = SimpleNamespace(session_store=store, _session_db=store._db, adapters={}, _draining=False,
                             _adapter_for_source=lambda source: None, _cached_agent_for=lambda route: None)
    authority = await initialize_session_authority(runner, profile_id=str(tmp_path), instance_id="i")
    conn = AuthorityConnection(authority, object(), {
        "user_id": "owner", "provider": "local", "profile_id": str(tmp_path), "instance_id": "i",
        "capabilities": ["session:create", "session:read", "session:submit", "session:control"],
        "native_bootstrap": True})
    loop = asyncio.get_running_loop()
    _route_store(monkeypatch, lambda params: asyncio.run_coroutine_threadsafe(
        conn.mutate(SessionRef(str(tmp_path), params["session_id"]), params), loop).result())

    session_id = await asyncio.to_thread(import_foreign_session, "claude", _write_claude_fixture(tmp_path))
    ref = SessionRef(str(tmp_path), session_id)
    opened = await conn.resume(ref, {})

    assert len(opened["messages"]) == 4
    assert not authority.sessions[session_id].detached
    authority.authorize(conn.actor, ref, "session:submit")


def test_leading_assistant_gets_single_stub(tmp_path):
    day = tmp_path / ".codex" / "sessions" / "2026" / "01" / "01"
    day.mkdir(parents=True)
    f = day / "rollout-x.jsonl"
    lines = [
        json.dumps(
            {
                "type": "response_item",
                "payload": {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Continuing from before."}],
                },
            }
        )
    ]
    f.write_text("\n".join(lines) + "\n", encoding="utf-8")
    parsed = parse_codex_session(f)
    _assert_alternating(parsed["turns"])
    assert len(parsed["turns"]) == 2
    assert parsed["turns"][0]["role"] == "user"


def test_whitespace_only_user_turn_does_not_break_discovery(tmp_path):
    """A blank user message (image-only / tool-only turn) must not crash listing; the title
    comes from the first non-blank user line and the blank turn is dropped."""
    project = tmp_path / ".claude" / "projects" / "p"
    project.mkdir(parents=True)
    f = project / "blank.jsonl"
    lines = [
        {"type": "user", "sessionId": "w", "message": {"role": "user", "content": "   \n  "}},
        {"type": "assistant", "sessionId": "w", "message": {"role": "assistant", "content": "hi"}},
        {"type": "user", "sessionId": "w", "message": {"role": "user", "content": "real question"}},
    ]
    f.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    listed = _list_sessions("claude", project.parent)
    assert [s.title_guess for s in listed] == ["real question"]
    assert listed[0].turn_count == 3  # leading assistant reply gets the user stub
