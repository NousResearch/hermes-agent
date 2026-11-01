"""Regression for #121317 through the real CLI, HTTP adapter, retry loops and SQLite."""

import sqlite3

import pytest

from tests.e2e.core.providers import _native_helpers as nh
from tests.fakes.providers.chat_variants import CText, FakeChatVariantServer
from tests.fakes.providers.gemini_native import HERMES_ENV, Blocked, GeminiFake, hermes_model


@pytest.mark.parametrize("stream", [True, False])
def test_prompt_block_surfaces_once_and_accounts_usage(tmp_path, stream):
    with GeminiFake(tmp_path / "fake", [], route=lambda rec: Blocked("SAFETY", prompt=True)) as fake:
        home = nh.make_home(tmp_path / "turn", {**hermes_model(), "streaming": stream}, env_file=HERMES_ENV,
                            extra_config={"agent": {"api_max_retries": 3}, "custom_providers": [{
                                "name": "gemini-test", "base_url": "https://generativelanguage.googleapis.com/v1beta",
                                "ssl_ca_cert": str(fake.ca_pem),
                            }]})
        result = nh.run_chat(home, "Say hello.", env=fake.child_env())
        calls = fake.generate_calls()
    assert len(calls) == 1, f"terminal refusal was retried: {len(calls)} calls\n{result.describe()}"
    assert calls[0].status == 200
    assert calls[0].stream is stream
    assert result.returncode != 0, result.describe()
    output = result.stdout + result.stderr
    assert "SAFETY" in output and "block" in output.lower(), result.describe()
    assert "temporarily unavailable" not in output.lower()
    assert "HTTP 400" not in output
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    assert lines and len(lines) == len(set(lines)), result.describe()
    assert not [row for row in nh.messages(home) if row["role"] == "assistant" and not row.get("content")]
    with sqlite3.connect(home.db_path) as db:
        usage = db.execute("SELECT input_tokens, output_tokens, api_call_count FROM sessions").fetchall()
    assert usage == [(900, 0, 1)], usage
    logs = "\n".join(path.read_text(encoding="utf-8") for path in (home.hermes_home / "logs").glob("*.log"))
    assert "Gemini blocked the prompt: SAFETY" in logs
    assert "HTTP 400" not in logs


def test_prompt_block_can_use_configured_fallback_without_rebilling_primary(tmp_path):
    with GeminiFake(tmp_path / "fake", [], route=lambda rec: Blocked("SAFETY", prompt=True)) as fake, \
            FakeChatVariantServer([CText("GEMINI-FALLBACK-ANSWER", prompt_tokens=1200, completion_tokens=40)]) as fallback:
        home = nh.make_home(tmp_path / "turn", hermes_model(), env_file=HERMES_ENV, extra_config={
            "agent": {"api_max_retries": 3},
            "custom_providers": [{"name": "gemini-test",
                "base_url": "https://generativelanguage.googleapis.com/v1beta", "ssl_ca_cert": str(fake.ca_pem)}],
            "fallback_providers": [{"provider": "custom", "model": "fake-model",
                "base_url": fallback.base_url, "api_key": "test-fallback-key"}],
        })
        result = nh.run_chat(home, "Say hello.", env={**fake.child_env(),
            "NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"})
        calls = fake.generate_calls()
        fallback_calls = fallback.main_requests()
    assert result.returncode == 0, result.describe()
    assert result.stdout.count("GEMINI-FALLBACK-ANSWER") == 1, result.describe()
    assert len(calls) == len(fallback_calls) == 1, (calls, fallback_calls)
    assert fallback_calls[0]["model"] == "fake-model"
    with sqlite3.connect(home.db_path) as db:
        usage = db.execute("SELECT input_tokens, output_tokens, api_call_count FROM sessions").fetchall()
    assert usage == [(2100, 40, 2)], usage
