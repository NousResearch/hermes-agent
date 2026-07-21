"""Tests for /save auto-generated title embedding (#61278).

The classic CLI ``/save`` and the gateway ``session.save`` RPC must record
the conversation's descriptive title in the JSON snapshot (not just in the
filename's timestamp). Title source:

  1. If the session DB already has a title (auto-generated earlier in the
     conversation, set via ``/title``, or imported from Honcho/petdex),
     reuse that — no extra LLM call.
  2. Otherwise, fall back to :func:`agent.title_generator.generate_title`
     with a short timeout. The legacy title used to live only in the
     session DB; preserving it inside the JSON snapshot makes the
     ``~/.hermes/sessions/saved/`` directory browsable by name without
     opening every file. #tom1007 #61278.
"""

from __future__ import annotations

import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    import hermes_constants

    if hasattr(hermes_constants, "_hermes_home_cache"):
        hermes_constants._hermes_home_cache = None
    return home


def _make_stub_cli(history, session_id="20260101_120000_abc123", session_db=None):
    """Build a minimal object exposing what save_conversation expects.

    _session_db is exposed for the cache-first path; save_conversation should
    call get_session_title on it when present.
    """
    return SimpleNamespace(
        conversation_history=history,
        model="test-model",
        session_id=session_id,
        session_start=datetime(2026, 1, 1, 12, 0, 0),
        _session_db=session_db,
    )


def _reload_cli():
    """Force-reload cli + hermes_constants so HERMES_HOME fixture takes effect."""
    for mod in [
        m for m in sys.modules if m.startswith("cli") or m == "hermes_constants"
    ]:
        sys.modules.pop(mod, None)
    import cli  # noqa: F401
    return cli


# ---------------------------------------------------------------------------
# Behavior 1: title is always present in the JSON payload (key invariant).
# ---------------------------------------------------------------------------


def test_save_conversation_includes_title_key_in_json(hermes_home, tmp_path, monkeypatch):
    """Snapshot JSON must carry a top-level ``title`` field — even before
    any title-generation logic runs, the key must exist so downstream
    readers can rely on it without branching on KeyError."""
    work = tmp_path / "somewhere-else"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr(
        "agent.title_generator.generate_title",
        lambda *a, **kw: "Stubbed Auto Title",
    )

    cli = _reload_cli()
    stub = _make_stub_cli([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])

    cli.HermesCLI.save_conversation(stub)

    saved_dir = hermes_home / "sessions" / "saved"
    files = list(saved_dir.glob("hermes_conversation_*.json"))
    assert len(files) == 1, files
    payload = json.loads(files[0].read_text())

    # Behavior contract: the key exists. Its value may be None (untitled
    # snapshot) or a string (resolved title), but the field is always there.
    assert "title" in payload, (
        "snapshot JSON must include a 'title' field; got: " + repr(list(payload.keys()))
    )


# ---------------------------------------------------------------------------
# Behavior 2: DB-cached title is reused — no auto-title call.
# ---------------------------------------------------------------------------


def test_save_conversation_reuses_db_title_without_calling_llm(hermes_home, tmp_path, monkeypatch):
    """When the session DB already has a title, save_conversation must NOT
    trigger a fresh LLM call — that would be wasteful and could overwrite
    an intentional /title rename with a different generated string."""
    gen_calls = []

    def spy(*args, **kwargs):
        gen_calls.append((args, kwargs))
        return "LLM Should Not Have Run"

    monkeypatch.setattr(
        "agent.title_generator.generate_title", spy,
    )

    # Fake a DB whose get_session_title returns a real title for our session.
    fake_db = SimpleNamespace(
        get_session_title=lambda sid: "Manual Renamed Title",
    )
    cli = _reload_cli()
    stub = _make_stub_cli(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        session_db=fake_db,
    )

    cli.HermesCLI.save_conversation(stub)

    saved_dir = hermes_home / "sessions" / "saved"
    payload = json.loads(
        list(saved_dir.glob("hermes_conversation_*.json"))[0].read_text()
    )
    assert payload["title"] == "Manual Renamed Title"
    assert gen_calls == [], (
        "save_conversation must not call generate_title when the DB already "
        f"has a title; observed calls: {gen_calls}"
    )


# ---------------------------------------------------------------------------
# Behavior 3: failing title generation is non-fatal — snapshot still saves.
# ---------------------------------------------------------------------------


def test_save_conversation_records_null_title_when_generation_fails(
    hermes_home, tmp_path, monkeypatch
):
    """If ``generate_title`` raises or returns None, the snapshot must still
    be written with ``title: null`` — /save must never block waiting on a
    title that doesn't exist, and must never lose the conversation."""
    monkeypatch.setattr(
        "agent.title_generator.generate_title",
        lambda *a, **kw: None,
    )
    cli = _reload_cli()
    stub = _make_stub_cli([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])

    cli.HermesCLI.save_conversation(stub)

    saved_dir = hermes_home / "sessions" / "saved"
    files = list(saved_dir.glob("hermes_conversation_*.json"))
    assert len(files) == 1, "snapshot must still be written when title gen fails"
    payload = json.loads(files[0].read_text())
    assert payload["title"] is None


def test_save_conversation_records_null_title_when_generation_raises(
    hermes_home, tmp_path, monkeypatch
):
    """Same as above, but the failure is an exception (network drop,
    provider 5xx, parser bug) — /save must swallow it and still write."""
    def blow_up(*a, **kw):
        raise RuntimeError("provider unreachable")

    monkeypatch.setattr(
        "agent.title_generator.generate_title", blow_up,
    )
    cli = _reload_cli()
    stub = _make_stub_cli([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])

    cli.HermesCLI.save_conversation(stub)

    saved_dir = hermes_home / "sessions" / "saved"
    files = list(saved_dir.glob("hermes_conversation_*.json"))
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["title"] is None


# ---------------------------------------------------------------------------
# Behavior 4: terminal confirmation echoes the resolved title when present.
# ---------------------------------------------------------------------------


def test_save_conversation_terminal_output_includes_title(
    hermes_home, tmp_path, monkeypatch, capsys
):
    """When a title is resolved (DB or freshly generated), the confirmation
    output must surface it so the user sees the title and can locate the
    file later without opening it.

    Drives the title through the DB-cached path (the more common one in
    practice — auto-titling already populated the DB at this point). This
    avoids relying on monkeypatching ``generate_title`` across a module
    reload, which interacts poorly with some pytest layouts.
    """
    gen_calls = []

    def spy(*a, **kw):
        gen_calls.append((a, kw))
        return "LLM would have generated this"

    monkeypatch.setattr("agent.title_generator.generate_title", spy)

    fake_db = SimpleNamespace(
        get_session_title=lambda sid: "Cached Manual Title",
    )
    cli = _reload_cli()
    stub = _make_stub_cli(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ],
        session_db=fake_db,
    )

    cli.HermesCLI.save_conversation(stub)

    out = capsys.readouterr().out
    assert gen_calls == [], "DB-cached title must skip generate_title"
    assert "Cached Manual Title" in out, (
        "resolved title must appear in /save confirmation output, got:\n" + out
    )


def test_save_conversation_terminal_output_omits_title_label_when_none(
    hermes_home, tmp_path, monkeypatch, capsys
):
    """When no title can be resolved, the confirmation output must still
    be valid — but it should NOT print a misleading 'Title:' line with
    'None' or empty. Either silence the line entirely or fall through."""
    monkeypatch.setattr(
        "agent.title_generator.generate_title",
        lambda *a, **kw: None,
    )
    cli = _reload_cli()
    stub = _make_stub_cli([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ])

    cli.HermesCLI.save_conversation(stub)

    out = capsys.readouterr().out
    # Snapshot path confirmation must still appear — /save worked.
    assert "saved to" in out
    # No bare "Title: None" string in the output (would look like a bug).
    assert "Title: None" not in out
    assert "Title: " not in out  # we either show a title or show nothing.


# ---------------------------------------------------------------------------
# Behavior 5: gateway session.save RPC mirrors the same JSON contract.
# ---------------------------------------------------------------------------


def test_tui_session_save_includes_title_key(tmp_path, monkeypatch):
    """The gateway ``session.save`` RPC (TUI / desktop save action) must
    write the same ``title`` field as the classic CLI. Same fix the bug
    across both surfaces, not just the CLI."""
    from tui_gateway import server

    home = tmp_path / ".hermes"
    home.mkdir()
    saved_dir = home / "sessions" / "saved"
    monkeypatch.setenv("HERMES_HOME", str(home))

    monkeypatch.setattr(
        "agent.title_generator.generate_title",
        lambda *a, **kw: "RPC Stub Title",
    )

    sid = "save-sid-rpc"
    agent = SimpleNamespace(
        model="hermes-test",
        session_id="20260101_120000_abc123",
        session_start=datetime(2026, 1, 1, 12, 0, 0),
        _cached_system_prompt="You are Hermes.",
        _session_db=None,
    )
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    server._sessions[sid] = {
        "agent": agent,
        "session_key": "save-key-rpc",
        "history": history,
        "history_lock": threading.Lock(),
        "created_at": 1735732800.0,
    }
    try:
        resp = server._methods["session.save"]("1", {"session_id": sid})
    finally:
        server._sessions.pop(sid, None)

    assert "result" in resp, resp
    saved_file = Path(resp["result"]["file"])
    payload = json.loads(saved_file.read_text())
    assert "title" in payload
    assert payload["title"] == "RPC Stub Title", resp
