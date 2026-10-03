import asyncio
import json
import sqlite3
import time
from types import SimpleNamespace

from hermes_cli.checkpoint_commands import (
    APPROVAL_FLAG,
    checkpoint_meta_key,
    checkpoint_state_path,
    run_checkpoint_command,
)
from hermes_cli.commands import gateway_help_lines, resolve_command


def _seed_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = tmp_path / "state.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp REAL, active INTEGER)"
        )
        conn.execute(
            "CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.executemany(
            "INSERT INTO messages (id, session_id, role, content, timestamp, active) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, "s1", "user", "Important rule: never send client emails without approval. API_KEY=secret-value", 1.0, 1),
                (2, "s1", "assistant", "Understood. I will keep that as an approval gate candidate.", 2.0, 1),
                (3, "s1", "tool", "raw tool output should not be included", 3.0, 1),
            ],
        )
    return db


def _read_meta(db_path, session_id="s1"):
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            "SELECT value FROM state_meta WHERE key = ?",
            (checkpoint_meta_key(session_id),),
        ).fetchone()
    return json.loads(row[0]) if row else None


def test_checkpoint_commands_registered_and_in_gateway_help():
    lock = resolve_command("lock-this-chat-into-law")
    promote = resolve_command("promote_since_last_checkpoint")
    assert lock is not None
    assert lock.name == "lock-this-chat-into-law"
    assert promote is not None
    assert promote.name == "promote-since-last-checkpoint"
    help_text = "\n".join(gateway_help_lines())
    assert "/lock-this-chat-into-law" in help_text
    assert "/promote-since-last-checkpoint" in help_text


def test_dry_run_writes_candidate_packet_but_does_not_advance_checkpoint(tmp_path, monkeypatch):
    db_path = _seed_home(tmp_path, monkeypatch)
    result = run_checkpoint_command("/lock-this-chat-into-law", session_id="s1", args="--dry-run")
    assert result.approved is False
    assert "No durable memory/SOUL/reference writes" in result.text
    assert result.state_path == str(checkpoint_state_path())
    assert _read_meta(db_path) is None

    packet = json.loads(open(result.packet_path, encoding="utf-8").read())
    assert packet["raw_transcript_included"] is False
    assert packet["classification_policy"]["method"] == "english-keyword-heuristic"
    assert packet["classification_policy"]["requires_review"] is True
    assert packet["redaction_policy"]["raw_transcript_included"] is False
    assert packet["state_store"] == "SessionDB.state_meta"
    assert packet["state_key"] == checkpoint_meta_key("s1")
    assert packet["candidate_count"] == 2
    assert all(c["role"] != "tool" for c in packet["candidates"])


def test_redaction_and_snippets_avoid_raw_transcript_dump(tmp_path, monkeypatch):
    _seed_home(tmp_path, monkeypatch)
    result = run_checkpoint_command("/promote-since-last-checkpoint", session_id="s1", args="")
    packet_text = open(result.packet_path, encoding="utf-8").read()
    assert "secret-value" not in packet_text
    assert "API_KEY=[REDACTED]" in packet_text
    assert "raw tool output should not be included" not in packet_text
    packet = json.loads(packet_text)
    assert all(len(c["snippet"]) <= 181 for c in packet["candidates"])
    assert all(c["classification_requires_review"] is True for c in packet["candidates"])


def test_explicit_approval_advances_checkpoint_in_native_state_meta_without_memory_writes(tmp_path, monkeypatch):
    db_path = _seed_home(tmp_path, monkeypatch)
    result = run_checkpoint_command(
        "/promote-since-last-checkpoint",
        session_id="s1",
        args=APPROVAL_FLAG,
    )
    assert result.approved is True
    assert "native SessionDB.state_meta" in result.text
    state = _read_meta(db_path)
    assert state["last_message_id"] == 2
    assert state["packet_path"] == result.packet_path
    packet = json.loads(open(result.packet_path, encoding="utf-8").read())
    assert packet["durable_writes"] == [
        "native state_meta checkpoint boundary advanced",
        "approved candidate packet recorded",
    ]


def test_since_last_checkpoint_window_starts_after_previous_state_meta_boundary(tmp_path, monkeypatch):
    db_path = _seed_home(tmp_path, monkeypatch)
    run_checkpoint_command("/lock-this-chat-into-law", session_id="s1", args=APPROVAL_FLAG)
    assert _read_meta(db_path)["last_message_id"] == 2
    with sqlite3.connect(tmp_path / "state.db") as conn:
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, timestamp, active) VALUES (?, ?, ?, ?, ?, ?)",
            (4, "s1", "user", "Decision: use candidate-first memory promotion.", 4.0, 1),
        )
    result = run_checkpoint_command("/promote-since-last-checkpoint", session_id="s1", args="--dry-run")
    assert result.window["from_exclusive_message_id"] == 2
    assert result.window["to_inclusive_message_id"] == 4
    assert len(result.candidates) == 1



def test_secret_token_redaction_removes_recognisable_prefixes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    db = tmp_path / "state.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp REAL, active INTEGER)")
        conn.execute("CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, timestamp, active) VALUES (?, ?, ?, ?, ?, ?)",
            (1, "s1", "user", "Tokens sk-ABCDEFGHIJKLMNOPQRST ghp_abcdefghijklmnopqrstuvwxyz xoxb-1234567890-abcd", time.time(), 1),
        )
    result = run_checkpoint_command("/lock-this-chat-into-law", session_id="s1", args="--dry-run")
    packet_text = open(result.packet_path, encoding="utf-8").read()
    assert "sk-" not in packet_text
    assert "ghp_" not in packet_text
    assert "xoxb-" not in packet_text
    assert "[REDACTED]" in packet_text


def test_packet_filename_is_safe_when_session_id_contains_slashes(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    session_id = "spaces/AAA/thread/BBB"
    db = tmp_path / "state.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp REAL, active INTEGER)")
        conn.execute("CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.execute(
            "INSERT INTO messages (id, session_id, role, content, timestamp, active) VALUES (?, ?, ?, ?, ?, ?)",
            (1, session_id, "user", "Decision: safe filename test", time.time(), 1),
        )
    result = run_checkpoint_command("/lock-this-chat-into-law", session_id=session_id, args="--dry-run")
    assert result.packet_path.endswith(".json")
    assert "spaces/AAA" not in result.packet_path
    assert json.loads(open(result.packet_path, encoding="utf-8").read())["session_id"] == session_id


def test_explicit_lookback_filters_by_timestamp_and_overrides_checkpoint_boundary(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    now = time.time()
    db = tmp_path / "state.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, content TEXT, timestamp REAL, active INTEGER)")
        conn.execute("CREATE TABLE state_meta (key TEXT PRIMARY KEY, value TEXT)")
        conn.executemany(
            "INSERT INTO messages (id, session_id, role, content, timestamp, active) VALUES (?, ?, ?, ?, ?, ?)",
            [
                (1, "s1", "user", "Decision: old outside lookback", now - 7200, 1),
                (2, "s1", "user", "Decision: recent inside lookback", now - 120, 1),
            ],
        )
        conn.execute(
            "INSERT INTO state_meta (key, value) VALUES (?, ?)",
            (checkpoint_meta_key("s1"), json.dumps({"last_message_id": 99})),
        )
    result = run_checkpoint_command("/promote-since-last-checkpoint", session_id="s1", args="90m --dry-run")
    assert result.window["boundary_mode"] == "lookback"
    assert result.window["lookback_raw"] == "90m"
    assert result.window["from_exclusive_message_id"] == 0
    assert [c["message_id"] for c in result.candidates] == [2]


def test_invalid_lookback_is_rejected_clearly(tmp_path, monkeypatch):
    _seed_home(tmp_path, monkeypatch)
    try:
        run_checkpoint_command("/lock-this-chat-into-law", session_id="s1", args="yesterday --dry-run")
    except ValueError as exc:
        assert "Use forms like 90m, 12h, or 4d" in str(exc)
    else:
        raise AssertionError("invalid lookback should raise")


def test_approval_flag_plus_dry_run_does_not_claim_boundary_advanced(tmp_path, monkeypatch):
    db_path = _seed_home(tmp_path, monkeypatch)
    result = run_checkpoint_command(
        "/lock-this-chat-into-law",
        session_id="s1",
        args=f"{APPROVAL_FLAG} --dry-run",
    )
    assert result.approved is False
    assert "ignored because dry-run" in result.text
    assert "No checkpoint boundary advanced" in result.text
    assert "Checkpoint boundary advanced in native" not in result.text
    assert _read_meta(db_path) is None


def test_gateway_checkpoint_uses_actual_session_id_not_route_key(monkeypatch):
    from gateway.run import GatewayRunner

    captured = {}

    def fake_run_checkpoint_command(command, *, session_id, args=""):
        captured["command"] = command
        captured["session_id"] = session_id
        captured["args"] = args
        return SimpleNamespace(text="ok")

    class Store:
        async def get_or_create_session(self, source, touch_activity=True):
            assert touch_activity is False
            return SimpleNamespace(session_id="actual-session-id", session_key="route-key")

        async def lookup_by_session_key(self, key):
            raise AssertionError("source resolution should win")

    event = SimpleNamespace(
        get_command=lambda: "lock-this-chat-into-law",
        get_command_args=lambda: "--dry-run",
        session_key="route-key",
    )
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.session_store = object()
    store = Store()
    store._store = runner.session_store
    runner._async_session_store = store
    monkeypatch.setattr("hermes_cli.checkpoint_commands.run_checkpoint_command", fake_run_checkpoint_command)

    text = asyncio.run(GatewayRunner._handle_checkpoint_command(runner, event, session_id="route-key", source=object()))
    assert text == "ok"
    assert captured["session_id"] == "actual-session-id"
    assert captured["command"] == "/lock-this-chat-into-law"

def test_redact_multitoken_authorization_scheme():
    from hermes_cli.checkpoint_commands import redact

    opaque = "opaque" + "x" * 32
    text = "Authorization: " + "Bearer " + opaque
    redacted = redact(text)
    assert opaque not in redacted
    assert "Bearer " not in redacted
    assert redacted == "Authorization=[REDACTED]"


def test_redact_authorization_header_redacts_generic_schemes():
    from hermes_cli.checkpoint_commands import redact

    bearer_payload = "opaque" + "x" * 32
    samples = [
        ("Authorization: " + "Basic " + "dXNlcjpwYXNz"),
        ("Authorization: " + "Token " + bearer_payload),
        "Authorization: Digest username=alice, realm=prod, nonce=abc123, response=deadbeef",
        "Authorization: AWS4-HMAC-SHA256 Credential=AKIAEXAMPLE/20260913/ap-southeast-2/service/aws4_request, SignedHeaders=host;x-amz-date, Signature=abcdef1234567890",
        "Authorization: Negotiate YIIGZQYJKoZIhvcSAQICAQBuggZU",
        "authorization=Custom-Scheme param1=value1 param2=value2",
    ]
    for sample in samples:
        redacted = redact(sample)
        assert redacted == "Authorization=[REDACTED]" or redacted == "authorization=[REDACTED]"
        assert "Bearer" not in redacted
        assert "Basic" not in redacted
        assert "Token" not in redacted
        assert "Digest" not in redacted
        assert "AWS4-HMAC" not in redacted
        assert "Negotiate" not in redacted
        assert "Custom-Scheme" not in redacted
        assert "Signature" not in redacted
        assert "response" not in redacted
        assert bearer_payload not in redacted
