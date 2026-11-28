"""``prompt_caching.openai_key_scope``: which conversations share one OpenAI ``prompt_cache_key``."""

from agent.transports import get_transport


def test_install_key_scope_shares_one_key_only_on_openai(tmp_path, monkeypatch):
    """``prompt_caching.openai_key_scope: install``: two unrelated sessions with one static prefix
    share a key on api.openai.com (OpenAI shares a cached prefix only inside one key), while the
    default stays per session and other hosts never get the install scope."""
    home = tmp_path / "hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))

    def keys(base_url):
        return [get_transport("codex_responses").build_kwargs(
            model="gpt-5.4", messages=[{"role": "user", "content": "Hi"}], tools=[],
            session_id=sid, base_url=base_url,
        )["prompt_cache_key"] for sid in ("session_alice_1", "session_bob_1")]

    openai_url, proxy_url = "https://api.openai.com/v1", "https://proxy.example.com/v1"
    a, b = keys(openai_url)
    assert a != b  # default: per-session scope (#78941)
    (home / "config.yaml").write_text("prompt_caching:\n  openai_key_scope: install\n", encoding="utf-8")
    a, b = keys(openai_url)
    assert a == b
    c, d = keys(proxy_url)
    assert c != d
