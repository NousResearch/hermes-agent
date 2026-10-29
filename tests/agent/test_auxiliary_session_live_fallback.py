"""A session whose frozen ``auxiliary`` section predates a task pin follows live config.

Regression for #145493: gateway turns run under ``session_auxiliary_scope()`` holding the
session's creation-time ``auxiliary`` section, and ``auxiliary`` is not a followed section, so a
compression pin added after session creation never reached the session — while a bare
``call_llm(task="compression")`` outside the scope reads live config and honors it.
"""

from agent import auxiliary_client as ac
from agent.auxiliary_task_config import _get_auxiliary_task_config, session_auxiliary_scope

# A pin added to live config.yaml after the session was created.
LIVE_PIN = {
    "provider": "custom", "model": "pinned-combo",
    "base_url": "http://127.0.0.1:9/api/v1", "api_key": "k-aux", "timeout": 3,
}
# What a session created before the pin froze: the merged defaults (no route).
FROZEN_DEFAULTS = {
    "provider": "auto", "model": "", "base_url": "", "api_key": "", "timeout": 120,
    "extra_body": {}, "reasoning_effort": "",
}
# A session created after a (different) pin froze: the pin must keep winning.
FROZEN_PIN = {
    "provider": "custom", "model": "frozen-combo",
    "base_url": "http://127.0.0.1:7/api/v1", "api_key": "k-frozen", "timeout": 3,
}
# The session's own model runtime, as the compressor binds it on every summary call.
MAIN_RUNTIME = {
    "model": "session-model", "provider": "custom:probe-main",
    "base_url": "http://127.0.0.1:8/v1", "api_key": "k-main",
    "api_mode": "chat_completions",
}


def _live_pin(monkeypatch, pin):
    monkeypatch.setattr(
        "hermes_cli.config.load_config_readonly",
        lambda: {"auxiliary": {"compression": dict(pin)}},
    )


def _prepared(task_config_source=None):
    return ac._prepare_aux_request(
        "compression", provider=None, model=None, base_url=None, api_key=None,
        main_runtime=dict(MAIN_RUNTIME), messages=[{"role": "user", "content": "x"}],
        temperature=None, max_tokens=None, tools=None, timeout=3,
        extra_body=None, reasoning_config=None, extra_headers=None,
        api_mode=None, route_info={}, async_mode=False,
    )


def test_unpinned_frozen_task_follows_live_pin(monkeypatch):
    """The session predates the pin: the summary must still reach the pinned route."""
    _live_pin(monkeypatch, LIVE_PIN)
    with session_auxiliary_scope({"compression": dict(FROZEN_DEFAULTS)}):
        assert _get_auxiliary_task_config("compression").get("model") == "pinned-combo"
        req = _prepared()
    assert req.resolved_model == "pinned-combo"
    assert (req.resolved_base_url or "") == "http://127.0.0.1:9/api/v1"


def test_pinned_frozen_task_keeps_freeze(monkeypatch):
    """A frozen pin still wins over a later-changed live config (freeze preserved)."""
    _live_pin(monkeypatch, LIVE_PIN)
    with session_auxiliary_scope({"compression": dict(FROZEN_PIN)}):
        assert _get_auxiliary_task_config("compression").get("model") == "frozen-combo"
        req = _prepared()
    assert req.resolved_model == "frozen-combo"
    assert (req.resolved_base_url or "") == "http://127.0.0.1:7/api/v1"


def test_unscoped_task_reads_live_pin(monkeypatch):
    """Outside a session scope (CLI control path) live config applies as before."""
    _live_pin(monkeypatch, LIVE_PIN)
    assert _get_auxiliary_task_config("compression").get("model") == "pinned-combo"
