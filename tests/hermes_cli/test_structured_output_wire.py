"""Real one-shot CLI corrections against synthetic loopback providers, never live keys."""
import json
from contextlib import ExitStack
import subprocess
import sys

import pytest

from tests.e2e.core.providers._openai_helpers import Home, REPO_ROOT
from tests.fakes.providers.anthropic_messages import (
    AnthropicMessagesServer, MODEL_ID, Reply, Text, Thinking, ToolUse,
)
from tests.fakes.providers.chat_variants import CText, FakeChatVariantServer


@pytest.fixture
def correction_home(tmp_path):
    return Home(tmp_path)


def _configure(home, server, provider="anthropic"):
    home.write({
        "model": {"provider": provider, "base_url": server.base_url,
                  "default": MODEL_ID if provider == "anthropic" else "fixture-chat-model",
                  "context_length": 200000},
        "agent": {"api_max_retries": 0, "auto_recovery_cycles": 0,
                  "oneshot_completion_wait_seconds": 0},
        "compression": {"enabled": False},
        "curator": {"enabled": False},
        "auxiliary": {"title_generation": {"enabled": False}},
    }, dotenv={"ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY":
               "fixture-only-not-a-credential"})
    (home.project / "schema.json").write_text(json.dumps({"const": {"ok": True}}), encoding="utf-8")


def _run(home, *extra, env=None):
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "chat", "-q", "Return the fixture answer.",
         "--format", "json", "--output-schema", "schema.json", "-o", "answer.json",
         "--max-turns", "3", "--run-budget", "25", "--ignore-rules", "--cli", *extra],
        cwd=home.project,
        env=home.env({"PYTHONPATH": str(REPO_ROOT), "HERMES_DISABLE_LAZY_INSTALLS": "1",
                      "HERMES_DISABLE_PLUGINS": "1", "NO_PROXY": "127.0.0.1,localhost", **(env or {})}),
        stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=50,
    )


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("correction_tool", [False, True], ids=["answer", "tool-rejected"])
def test_anthropic_correction_preserves_split_system_and_tool_cache(correction_home, correction_tool):
    home = correction_home
    forbidden = home.project / "forbidden.txt"
    correction = (ToolUse("write_file", {"path": str(forbidden), "content": "must not execute"})
                  if correction_tool else Text('{"ok":true}'))
    with AnthropicMessagesServer([Reply([Text("not JSON")]), Reply([correction])]) as server:
        _configure(home, server)
        artifact = home.project / "answer.json"
        artifact.write_text("old artifact", encoding="utf-8")
        proc = _run(home)
        records = [r for r in server.requests if r.get("kind") == "main"]

    assert proc.returncode == (1 if correction_tool else 0), proc.stdout + proc.stderr
    result = json.loads(proc.stdout)
    assert result["exit_code"] == proc.returncode
    assert artifact.read_text(encoding="utf-8-sig") == ("old artifact" if correction_tool else '{"ok":true}')
    assert not forbidden.exists()
    assert len(records) == 2
    assert not [r["schema_errors"] for r in records if r.get("schema_errors")]
    original, retry = [r["body"] for r in records]
    assert isinstance(original["system"], list) and len(original["system"]) == 2
    assert all(block["cache_control"] == {"type": "ephemeral"} for block in original["system"])
    assert retry["tools"] == original["tools"]
    assert retry["system"] == original["system"], "correction changed the original split/cache-marked system"


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("provider", ["custom", "anthropic"])
def test_corrected_answer_is_durable_and_replays_on_resume(correction_home, provider):
    from hermes_state import SessionDB

    home = correction_home
    corrected = '{"ok":true}'
    thinking = {"type": "thinking", "thinking": "fixture reasoning", "signature": "fixture-signature"}
    server = (AnthropicMessagesServer([
        Reply([Text("not JSON")]), Reply([Thinking(thinking["thinking"], thinking["signature"]), Text(corrected)]),
        Reply([Text(corrected)]),
    ]) if provider == "anthropic" else FakeChatVariantServer([
        CText("not JSON"), CText(corrected), CText(corrected),
    ], strict_user_turn=True))
    with server, ExitStack() as stack:
        _configure(home, server, provider)
        env = {}
        if provider == "anthropic":
            # Native signature policy strips thinking for third-party base URLs.
            # Intercept TLS locally to exercise the real direct-Anthropic replay path.
            from tests.fakes.providers.oauth_token_server import TLSInterceptProxy, make_test_ca
            host = "api.anthropic.com"
            ca = make_test_ca(home.root / "ca", [host])
            proxy = stack.enter_context(TLSInterceptProxy(server, ca, [host]))
            env = {"HTTPS_PROXY": proxy.url, "https_proxy": proxy.url, "SSL_CERT_FILE": str(ca.ca_pem)}
            home.update_config(lambda cfg: cfg["model"].update(base_url=f"https://{host}"))
        proc = _run(home, env=env)
        assert proc.returncode == 0, proc.stdout + proc.stderr
        result = json.loads(proc.stdout)
        assert result["structured_output"] == {"ok": True}
        assert (home.project / "answer.json").read_text(encoding="utf-8-sig") == corrected
        db = SessionDB(home.db_path, read_only=True)
        try:
            messages = db.get_messages_as_conversation(result["session_id"])
            session = db.get_session(result["session_id"])
        finally:
            db.close()
        assert messages[-1]["content"] == corrected, "SQLite still ends with the invalid original answer"
        assert [m["role"] for m in messages if m["role"] != "system"] == ["user", "assistant", "user", "assistant"]
        assert messages[-3]["content"] == "not JSON"
        assert session["input_tokens"] == result["tokens"]["input"] == 200
        assert session["output_tokens"] == result["tokens"]["output"] == 40
        assert session["api_call_count"] == 2
        if provider == "anthropic":
            assert messages[-1]["reasoning_details"] == [thinking]

        resumed = _run(home, "--resume", result["session_id"], env=env)
        assert resumed.returncode == 0, resumed.stdout + resumed.stderr
        assert json.loads(resumed.stdout)["session_id"] == result["session_id"]
        if provider == "anthropic":
            records = [r for r in server.requests if r.get("kind") == "main"]
            assert not [r["schema_errors"] for r in records if r.get("schema_errors")]
            requests = [r["body"] for r in records]
        else:
            assert not server.invalid_requests()
            requests = server.main_requests()
        assert len(requests) == 3  # one original, one correction, one resumed turn
        replay = [m for m in requests[-1]["messages"] if m["role"] != "system"]
        assert [m["role"] for m in replay] == ["user", "assistant", "user", "assistant", "user"]
        if provider == "anthropic":
            blocks = [{k: v for k, v in b.items() if k != "cache_control"} for b in replay[-2]["content"]]
            assert blocks == [thinking, {"type": "text", "text": corrected}]
        else:
            assert replay[-2]["content"] == corrected


@pytest.mark.platforms("posix")
@pytest.mark.parametrize("output_format", ["json", "stream-json"])
def test_surrogate_json_uses_utf8_safe_envelope_before_publishing(correction_home, output_format):
    home = correction_home
    raw = '{"x":"\\ud800"}'  # ASCII JSON whose decoded value contains a lone surrogate
    with FakeChatVariantServer([CText(raw)], strict_user_turn=True) as server:
        _configure(home, server, "custom")
        (home.project / "schema.json").write_text("{}", encoding="utf-8")
        artifact = home.project / "answer.json"
        artifact.write_text("old artifact", encoding="utf-8")
        proc = _run(home, "--format", output_format, env={"PYTHONIOENCODING": "utf-8:strict"})

    if proc.returncode:
        assert artifact.read_text(encoding="utf-8-sig") == "old artifact", (
            f"failed run replaced artifact: exit={proc.returncode}, stdout={proc.stdout!r}, stderr={proc.stderr!r}")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    events = [json.loads(line) for line in proc.stdout.splitlines()]
    results = [event for event in events if event["type"] == "result"]
    assert len(results) == 1
    assert results[0]["exit_code"] == 0
    assert results[0]["structured_output"] == json.loads(raw)
    assert artifact.read_text(encoding="utf-8-sig") == raw
    assert len(server.main_requests()) == 1
