"""Tests for the persistent claude-cli live CLIENT (delta-vs-reseed logic).

These exercise the seam that the session-manager tests do not: how
``ClaudeLiveClient`` decides whether to send only the newest turn (warm reuse)
or reseed the full conversation (a fresh process). Getting this wrong silently
drops all prior context whenever a session respawns — e.g. a live ``/model`` or
``/effort`` switch, which changes the fingerprint and forces a respawn.

The real ``claude`` binary is never spawned; a scripted FakeProc captures the
stream-json envelope written to stdin so we can assert exactly what each turn
sent.
"""

import json
import os
import sys
import threading

import pytest

import agent.claude_live_client as c
from agent import claude_live_session as s


# ---------------------------------------------------------------------------
# Fake subprocess that records the stdin envelope and scripts a full turn.
# ---------------------------------------------------------------------------


class _FakeStream:
    """Blocking readline over a growable line buffer (fed per input turn)."""

    def __init__(self):
        self._lines = []
        self._idx = 0
        self._cv = threading.Condition()
        self._closed = False

    def feed(self, lines):
        with self._cv:
            self._lines.extend(lines)
            self._cv.notify_all()

    def readline(self):
        with self._cv:
            while self._idx >= len(self._lines) and not self._closed:
                self._cv.wait(timeout=0.05)
            if self._idx < len(self._lines):
                line = self._lines[self._idx]
                self._idx += 1
                return line
            return ""

    def close(self):
        with self._cv:
            self._closed = True
            self._cv.notify_all()


class _FakeStdin:
    """Each written envelope drives one scripted turn onto the proc's stdout,
    exactly as the real streaming binary responds to a user turn."""

    def __init__(self, proc):
        self._proc = proc
        self.written = []

    def write(self, data):
        self.written.append(data)
        self._proc.on_input()

    def flush(self):
        pass

    def close(self):
        pass


def _turn_lines(session_id, text):
    return [
        json.dumps({"type": "system", "subtype": "init", "session_id": session_id}) + "\n",
        json.dumps(
            {
                "type": "assistant",
                "message": {
                    "usage": {"input_tokens": 3, "cache_read_input_tokens": 900, "output_tokens": 2},
                    "content": [{"type": "text", "text": text}],
                },
            }
        )
        + "\n",
        json.dumps({"type": "result", "subtype": "success", "is_error": False}) + "\n",
    ]


class FakeProc:
    def __init__(self, argv=None, session_id="sess", reply="ok", **kwargs):
        self.argv = argv or []
        self.pid = 4242
        self.returncode = None
        self.stdin = _FakeStdin(self)
        self.stdout = _FakeStream()
        self.stderr = _FakeStream()
        self._session_id = session_id
        self._reply = reply
        self._turns = 0
        self._alive = True

    def on_input(self):
        # First turn emits the init line; every turn emits assistant text+result.
        self._turns += 1
        lines = _turn_lines(self._session_id, self._reply)
        if self._turns > 1:
            lines = lines[1:]  # init only once per process
        self.stdout.feed(lines)

    def poll(self):
        return None if self._alive else 0

    def wait(self, timeout=None):
        self._alive = False
        self.returncode = 0
        return 0

    def send_signal(self, sig):
        self._alive = False

    def kill(self):
        self._alive = False


class _ProcFactory:
    """Hands out a fresh FakeProc per spawn and keeps them for inspection."""

    def __init__(self):
        self.spawned = []
        self._n = 0

    def __call__(self, argv=None, **kwargs):
        self._n += 1
        proc = FakeProc(argv=argv, session_id=f"sess-{self._n}", reply=f"reply-{self._n}")
        self.spawned.append(proc)
        return proc

    def last_envelope(self, spawn_index):
        """The user-turn content string written to the Nth spawned process."""
        proc = self.spawned[spawn_index]
        assert proc.stdin.written, "no envelope written"
        return json.loads(proc.stdin.written[0])["message"]["content"]


@pytest.fixture
def wired(monkeypatch):
    """Point the client's registry at a fake-popen registry we can inspect."""
    # Small watchdog budgets so any genuine starvation fails fast, not in 90s.
    monkeypatch.setenv("HERMES_CLAUDE_LIVE_FRESH_QUIET_S", "3")
    monkeypatch.setenv("HERMES_CLAUDE_LIVE_RESUME_QUIET_S", "3")
    factory = _ProcFactory()
    registry = s.LiveSessionRegistry(popen=factory)
    monkeypatch.setattr(c, "get_registry", lambda: registry)
    return factory, registry


def _client(session_id="conv-1"):
    client = c.ClaudeLiveClient(session_key=session_id)
    return client


def _messages_turn1():
    return [
        {"role": "system", "content": "STABLE-SYS"},
        {"role": "user", "content": "first question"},
    ]


def _messages_turn2():
    return [
        {"role": "system", "content": "STABLE-SYS"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "reply-1"},
        {"role": "user", "content": "second question"},
    ]


# ---------------------------------------------------------------------------
# Fresh vs warm envelope selection
# ---------------------------------------------------------------------------


def test_first_turn_seeds_and_warm_reuse_sends_delta(wired):
    factory, _ = wired
    client = _client()

    r1 = client._create_chat_completion(model="sonnet", messages=_messages_turn1(), tools=[])
    assert r1.choices[0].message.content == "reply-1"
    # Only one process spawned; it received the (whole) first turn.
    assert len(factory.spawned) == 1
    assert factory.last_envelope(0) == "first question"

    # Second turn, SAME model → warm reuse, no respawn, delta only.
    r2 = client._create_chat_completion(model="sonnet", messages=_messages_turn2(), tools=[])
    assert r2.choices[0].message.content == "reply-1"  # same warm proc replies again
    assert len(factory.spawned) == 1  # NO respawn
    assert factory.spawned[0].stdin.written[-1]
    second = json.loads(factory.spawned[0].stdin.written[-1])["message"]["content"]
    assert second == "second question"  # delta, not full history
    client.close()


def test_model_switch_respawns_and_reseeds_full_history(wired):
    """The core regression: a live /model switch changes the fingerprint and
    respawns the process. The fresh process MUST be reseeded with the full prior
    conversation, not just the newest turn, or all context is lost."""
    factory, _ = wired
    client = _client()

    client._create_chat_completion(model="sonnet", messages=_messages_turn1(), tools=[])
    assert len(factory.spawned) == 1

    # /model opus mid-conversation → fingerprint drift → respawn.
    client._create_chat_completion(model="opus", messages=_messages_turn2(), tools=[])
    assert len(factory.spawned) == 2, "model switch must respawn"

    reseed = factory.last_envelope(1)
    # The fresh (2nd) process must see the WHOLE conversation, including the
    # prior user question and assistant answer — not merely "second question".
    assert "first question" in reseed
    assert "second question" in reseed
    assert "reply-1" in reseed
    assert reseed != "second question"
    client.close()


def test_effort_switch_also_reseeds_full_history(wired):
    factory, _ = wired
    client = _client()

    client._create_chat_completion(
        model="sonnet", messages=_messages_turn1(), tools=[],
        extra_body={"_hermes_claude_effort": "low"},
    )
    client._create_chat_completion(
        model="sonnet", messages=_messages_turn2(), tools=[],
        extra_body={"_hermes_claude_effort": "high"},
    )
    assert len(factory.spawned) == 2, "effort switch must respawn"
    reseed = factory.last_envelope(1)
    assert "first question" in reseed and "reply-1" in reseed
    client.close()


def test_compression_respawns_and_reseeds_compressed_history(wired):
    """The long-session regression: when Hermes compresses its history (old turns
    replaced by a summary → the message list shrinks materially), the warm child
    still holds the OLD uncompressed transcript. The client must respawn and reseed
    the child with the CURRENT compressed history so the child's context realigns —
    otherwise it grows without bound (inflated meter + rising latency)."""
    factory, _ = wired
    client = _client()

    # Turn 1: a large history so the child is aligned to ~2000 tokens.
    big = [
        {"role": "system", "content": "STABLE-SYS"},
        {"role": "user", "content": "x" * 8000},
    ]
    client._create_chat_completion(model="sonnet", messages=big, tools=[])
    assert len(factory.spawned) == 1

    # Turn 2: Hermes compressed — the history is now tiny (a summary + new turn).
    compressed = [
        {"role": "system", "content": "STABLE-SYS"},
        {"role": "assistant", "content": "SUMMARY-of-earlier"},
        {"role": "user", "content": "new question"},
    ]
    client._create_chat_completion(model="sonnet", messages=compressed, tools=[])
    assert len(factory.spawned) == 2, "compression must respawn to reset the child"

    reseed = factory.last_envelope(1)
    # The fresh child is reseeded with the compressed history (full render), not a
    # bare delta — so it holds the summary and the new turn, and nothing more.
    assert "SUMMARY-of-earlier" in reseed
    assert "new question" in reseed
    assert reseed != "new question"
    client.close()


def test_context_drift_ceiling_respawns(wired, monkeypatch):
    """Even without a compression pass, if the child's reported context grows well
    past Hermes's view (its own transcript bloating), a drift ceiling respawns and
    reseeds so the child cannot climb toward the model window unbounded."""
    factory, _ = wired
    # FakeProc reports cache_read=900; make the ceiling bite at a tiny scale.
    monkeypatch.setenv("HERMES_CLAUDE_LIVE_DRIFT_MIN", "100")
    monkeypatch.setenv("HERMES_CLAUDE_LIVE_DRIFT_RATIO", "1.5")
    client = _client()

    small1 = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "hi"},
    ]
    client._create_chat_completion(model="sonnet", messages=small1, tools=[])
    assert len(factory.spawned) == 1

    # Child context (900) far exceeds Hermes's tiny view → drift → respawn.
    small2 = [
        {"role": "system", "content": "SYS"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "again"},
    ]
    client._create_chat_completion(model="sonnet", messages=small2, tools=[])
    assert len(factory.spawned) == 2, "drift ceiling must respawn the child"
    client.close()


def test_normal_growth_does_not_respawn(wired):
    """The guard: a normally GROWING history (no compression, child context within
    bounds) must NOT respawn — that would throw away the warm cache every turn."""
    factory, _ = wired
    client = _client()

    client._create_chat_completion(model="sonnet", messages=_messages_turn1(), tools=[])
    # A much larger second turn — history grows, does not shrink.
    grown = _messages_turn2() + [
        {"role": "assistant", "content": "reply-2"},
        {"role": "user", "content": "third question " * 50},
    ]
    client._create_chat_completion(model="sonnet", messages=grown, tools=[])
    assert len(factory.spawned) == 1, "growth alone must not respawn"
    client.close()


def test_has_prior_context_transition():
    """A fresh session reseeds; once it has taken a turn it sends deltas."""
    session = s.LiveSession(
        s.LiveSessionConfig(
            command="claude", argv=("claude", "-p"), cwd="/tmp", env={"HOME": "/h"},
            model="sonnet", effort="low", system_prompt_hash="a",
            mcp_config_hash="b", auth_identity="oauth:x",
        ),
        popen=lambda *a, **k: FakeProc(session_id="sess-x", reply="hi"),
    )
    assert session.has_prior_context is False
    session.spawn()
    assert session.has_prior_context is False
    session.send_turn("go", fresh=True, quiet_budget=2.0, hard_deadline=10.0)
    assert session.has_prior_context is True
    session.teardown()


def test_close_releases_tool_server_and_is_idempotent():
    """close() must shut the tool server (socket FD + accept thread) and be safe
    to call twice; the finalizer guards against Hermes dropping the client with
    `agent.client = None` and never calling close()."""
    client = c.ClaudeLiveClient(session_key="conv-close")
    client._tool_server.start()
    path = client._tool_server.socket_path
    assert os.path.exists(path)
    client.close()
    assert client.is_closed is True
    assert not os.path.exists(path)  # socket file removed
    client.close()  # idempotent, no raise


def test_finalizer_closes_tool_server_on_gc():
    """When the client is garbage-collected without close(), the weakref
    finalizer still tears the tool server down (breaks the accept-thread pin)."""
    import gc

    client = c.ClaudeLiveClient(session_key="conv-gc")
    client._tool_server.start()
    server = client._tool_server
    path = server.socket_path
    assert os.path.exists(path)
    del client
    gc.collect()
    assert not os.path.exists(path)


def test_env_scrub_removes_billing_diversion_vars(monkeypatch):
    """The child must run on the subscription OAuth session only. Every API-key,
    custom-header, and cloud-gateway (Bedrock/Vertex) routing var must be gone;
    CLAUDE_CODE_OAUTH_TOKEN + HOME must survive."""
    dangerous = {
        "ANTHROPIC_API_KEY": "sk-ant-xxx",
        "ANTHROPIC_AUTH_TOKEN": "tok",
        "ANTHROPIC_OAUTH_TOKEN": "otok",
        "ANTHROPIC_BASE_URL": "https://evil.example",
        "ANTHROPIC_CUSTOM_HEADERS": "x-api-key: sk-leak",
        "CLAUDE_CODE_PROVIDER_MANAGED_BY_HOST": "1",
        "CLAUDE_CODE_USE_BEDROCK": "1",
        "CLAUDE_CODE_USE_VERTEX": "1",
        "CLAUDE_CODE_SKIP_BEDROCK_AUTH": "1",
        "CLAUDE_CODE_SKIP_VERTEX_AUTH": "1",
        "ANTHROPIC_BEDROCK_BASE_URL": "https://bedrock.example",
        "ANTHROPIC_VERTEX_BASE_URL": "https://vertex.example",
        "ANTHROPIC_VERTEX_PROJECT_ID": "proj",
    }
    for key, value in dangerous.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "oauth-keep-me")

    env = c.build_live_subprocess_env()

    for key in dangerous:
        assert key not in env, f"{key} leaked into child env"
    assert env.get("CLAUDE_CODE_OAUTH_TOKEN") == "oauth-keep-me"
    assert env.get("HOME")


def test_delta_is_robust_to_compression_relocating_assistant_boundary():
    """After compression the summary lands as an assistant message, so the last
    assistant is no longer the previous turn. The warm delta must still send only
    the newest user turn, not re-send post-summary history."""
    compressed = [
        {"role": "system", "content": "SYS"},
        {"role": "assistant", "content": "[summary of earlier turns]"},  # compression artifact
        {"role": "user", "content": "older question that was already sent"},
        {"role": "assistant", "content": "older answer"},
        {"role": "user", "content": "the brand new question"},
    ]
    assert c._delta_user_text(compressed) == "the brand new question"


def test_build_turn_content_passes_data_uri_images():
    """A user turn carrying a data: image is forwarded as content blocks (text +
    image), not silently reduced to text."""
    png = "data:image/png;base64,iVBORw0KGgoAAAANS"
    messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": [
            {"type": "text", "text": "what is in this image?"},
            {"type": "image_url", "image_url": {"url": png}},
        ]},
    ]
    content = c._build_turn_content(messages, full=False)
    assert isinstance(content, list)
    text_block = next(b for b in content if b["type"] == "text")
    image_block = next(b for b in content if b["type"] == "image")
    assert "what is in this image" in text_block["text"]
    assert image_block["source"] == {
        "type": "base64", "media_type": "image/png", "data": "iVBORw0KGgoAAAANS",
    }


def test_build_turn_content_passes_http_image_and_plain_text():
    messages = [{"role": "user", "content": [
        {"type": "text", "text": "describe"},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.jpg"}},
    ]}]
    content = c._build_turn_content(messages, full=False)
    image_block = next(b for b in content if b["type"] == "image")
    assert image_block["source"] == {"type": "url", "url": "https://example.com/a.jpg"}
    # No images → plain string, not a list.
    assert c._build_turn_content([{"role": "user", "content": "hi"}], full=False) == "hi"


def test_fingerprint_ignores_volatile_prompt_tail():
    """Two system prompts differing only in the day-granularity date, session id,
    model, and provider lines must hash the same so a day-rollover or identity
    refresh does not evict the warm cache."""
    base = "You are Hermes.\nStable instruction block.\n<!-- hermes-cache-boundary -->"
    day1 = base + "\nConversation started: Monday, July 13, 2026\nSession ID: abc\nModel: sonnet\nProvider: claude-cli"
    day2 = base + "\nConversation started: Tuesday, July 14, 2026\nSession ID: abc\nModel: sonnet\nProvider: claude-cli"
    assert c._fingerprint_system_prompt(day1) == c._fingerprint_system_prompt(day2)
    # A genuine instruction change still differs.
    changed = base.replace("Stable instruction block.", "Different instructions.")
    assert c._fingerprint_system_prompt(changed) != c._fingerprint_system_prompt(day1)


def test_mcp_config_uses_this_interpreter(wired, monkeypatch):
    """The bridge must be launched with Hermes's own interpreter, not a bare
    python3 that may be missing in claude's spawn env on a server."""
    monkeypatch.delenv("HERMES_CLAUDE_LIVE_PYTHON", raising=False)
    client = _client("conv-interp")
    client._tool_server.start()
    client._tools_path = "/tmp/tools.json"
    path, _ = client._mcp_config_path()
    cfg = json.loads(open(path).read())
    assert cfg["mcpServers"]["hermes"]["command"] == sys.executable
    client.close()


def test_resume_spawn_marks_prior_context():
    session = s.LiveSession(
        s.LiveSessionConfig(
            command="claude", argv=("claude", "-p"), cwd="/tmp", env={"HOME": "/h"},
            model="sonnet", effort="low", system_prompt_hash="a",
            mcp_config_hash="b", auth_identity="oauth:x",
        ),
        popen=lambda *a, **k: FakeProc(session_id="sess-y", reply="hi"),
    )
    session.spawn(resume_session_id="sess-prev")
    # A resumed process reloads the transcript → treat as already holding context.
    assert session.has_prior_context is True
    session.teardown()
