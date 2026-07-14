import json
from types import SimpleNamespace


def test_two_sessions_dispatch_to_distinct_browser_scopes(monkeypatch):
    import model_tools

    seen = []
    monkeypatch.setattr(
        model_tools.registry,
        "dispatch",
        lambda name, args, **kw: seen.append((name, kw["task_id"])) or "ok",
    )

    assert model_tools.handle_function_call(
        "browser_navigate", {"url": "https://example.com"},
        task_id="turn-1", browser_scope="session-a",
        skip_pre_tool_call_hook=True, skip_tool_request_middleware=True,
    ) == "ok"
    assert model_tools.handle_function_call(
        "browser_navigate", {"url": "https://example.com"},
        task_id="turn-1", browser_scope="session-b",
        skip_pre_tool_call_hook=True, skip_tool_request_middleware=True,
    ) == "ok"
    assert seen == [
        ("browser_navigate", "session-a"),
        ("browser_navigate", "session-b"),
    ]


def test_resume_reuses_browser_identity_and_compression_does_not_rotate_it():
    from agent.agent_init import resolve_browser_scope

    first = resolve_browser_scope(None, "resumed-session")
    resumed = resolve_browser_scope(None, "resumed-session")
    assert resumed == first

    agent = SimpleNamespace(session_id="resumed-session", browser_scope=resumed)
    agent.session_id = "compression-continuation"
    assert agent.browser_scope == "resumed-session"


def test_browser_dispatch_without_scope_fails_closed(monkeypatch):
    import model_tools

    monkeypatch.setattr(
        model_tools.registry, "dispatch",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not dispatch")),
    )
    result = model_tools.handle_function_call(
        "browser_snapshot", {}, task_id="turn-only",
        skip_pre_tool_call_hook=True, skip_tool_request_middleware=True,
    )
    assert json.loads(result) == {"error": "browser scope is required for browser tools"}


def test_per_thread_mode_disables_shared_camofox_url(monkeypatch):
    from tools import browser_camofox

    monkeypatch.setenv("CAMOFOX_URL", "http://127.0.0.1:9377")
    monkeypatch.setattr(
        browser_camofox, "_get_camofox_config",
        lambda: {"mode": "per_thread_instances"},
    )
    browser_camofox._request_base_url.set(None)
    assert browser_camofox.get_camofox_url() == ""
    assert browser_camofox.is_camofox_mode() is True


def test_per_turn_cleanup_preserves_browser(monkeypatch):
    from agent import chat_completion_helpers

    monkeypatch.setattr(chat_completion_helpers, "is_persistent_env", lambda _: False)
    monkeypatch.setattr(
        chat_completion_helpers, "_ra",
        lambda: SimpleNamespace(
            cleanup_vm=lambda _: None,
            cleanup_browser=lambda _: (_ for _ in ()).throw(
                AssertionError("browser cleanup is a session-boundary operation")
            ),
        ),
    )
    chat_completion_helpers.cleanup_task_resources(
        SimpleNamespace(verbose_logging=False), "turn-id"
    )
