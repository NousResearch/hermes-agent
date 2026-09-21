"""RED tests for exact host lifecycle (REM-304..REM-308).

Pins the generic lifecycle contract:

- ``on_session_open`` fires once when a live addressable session is
  registered, BEFORE its first model turn (distinct from on_session_start).
- ``on_session_reset`` threads ``old_session_id`` so plugins can finalise
  exactly the rotated session (REM-307).
- reset/close/expiry do not double-fire or leak unrelated sessions (REM-309).

The tests drive the real CLI lifecycle path with a disposable fake plugin
manager recording hook invocations.
"""

from __future__ import annotations

import types

import pytest


class RecordingPlugin:
    """Records every hook invocation with its kwargs."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    def register_hook(self, name, callback) -> None:
        pass

    def __call__(self, **kwargs) -> None:
        pass


def _make_manager_with_hook(hook_name: str, callback) -> object:
    """A plugin manager exposing invoke_hook through the real seam."""

    class M:
        def __init__(self) -> None:
            self._hooks: dict[str, list] = {hook_name: [callback]}
            self._plugin_commands: dict = {}
            self._plugin_tool_names: set = set()

        def invoke_hook(self, name: str, **kwargs):
            results = []
            for cb in self._hooks.get(name, []):
                try:
                    r = cb(**kwargs)
                    if r is not None:
                        results.append(r)
                except Exception:
                    pass
            return results

    return M()


class _ResumeDB:
    """Small in-memory DB contract for the real TUI resume handler."""

    def __init__(self, target: str) -> None:
        self.target = target

    def get_session(self, session_id: str):
        if session_id == self.target:
            return {"id": self.target, "message_count": 0}
        return None

    def get_session_by_title(self, _title: str):
        return None

    def resolve_resume_session_id(self, session_id: str) -> str:
        return session_id

    def reopen_session(self, _session_id: str) -> None:
        return None

    def get_resume_conversations(self, _session_id: str):
        return [], []

    def get_ancestor_display_prefix(self, _session_id: str):
        return []


class TestOnSessionOpen:
    def test_on_session_open_is_a_valid_hook(self):
        """on_session_open must be registered as a valid hook name."""
        import hermes_cli.plugins as plugins

        assert "on_session_open" in plugins.VALID_HOOKS

    def test_cli_run_fires_on_session_open_before_first_turn(self, monkeypatch):
        """The CLI run() path must invoke on_session_open with the exact
        session id and platform before any model turn."""
        events: list[dict] = []

        def fake_invoke_hook(name, **kwargs):
            if name == "on_session_open":
                events.append(kwargs)
            return []

        # The CLI run() block imports invoke_hook from hermes_cli.plugins.
        import hermes_cli.plugins as plugins_mod

        monkeypatch.setattr(plugins_mod, "invoke_hook", fake_invoke_hook)
        from cli import HermesCLI

        cli = HermesCLI.__new__(HermesCLI)
        cli.session_id = "cli-sess-xyz"
        cli._resumed = False
        cli._should_exit = True  # exit immediately after the open hook
        cli._claim_active_session = lambda *a, **k: True
        # Stub the many attributes run() touches after the hook.
        cli._detect_light_mode = lambda: None
        cli.show_banner = lambda: None
        cli._show_security_advisories = lambda: None
        cli._console_print = lambda *a, **k: None
        cli._offer_first_run_setup = lambda: None
        cli._preload_resumed_session = lambda: False
        cli._display_resumed_history = lambda: None
        cli._pending_input = types.SimpleNamespace(put=lambda *a: None, empty=lambda: True)
        cli._injected_input = types.SimpleNamespace(put=lambda *a: None)
        cli._interrupt_queue = types.SimpleNamespace(put=lambda *a: None)
        cli._agent_running = False
        cli._status_bar_suppressed_after_resize = False
        cli._pending_resume_sessions = None
        cli._typed_voice_stop = lambda *a: False

        try:
            cli.run()
        except Exception:
            pass  # run() may fail on unrelated stubs after the hook
        assert events, "on_session_open must fire from CLI run()"
        assert events[0]["session_id"] == "cli-sess-xyz"
        assert events[0]["platform"] == "cli"


class TestResetThreadsOldSession:
    def test_notify_session_boundary_passes_old_session_id(self, monkeypatch):
        """_notify_session_boundary('on_session_reset', old_session_id=X)
        must pass old_session_id to the plugin hook."""
        from cli import HermesCLI

        received: dict = {}

        def fake_invoke_hook(name, **kwargs):
            received.update(kwargs)
            return []

        import hermes_cli.plugins as plugins_mod

        monkeypatch.setattr(plugins_mod, "invoke_hook", fake_invoke_hook)
        cli = HermesCLI.__new__(HermesCLI)
        cli.agent = types.SimpleNamespace(session_id="new-sess")
        cli.platform = "cli"
        cli._notify_session_boundary("on_session_reset", old_session_id="old-sess")
        assert received["old_session_id"] == "old-sess"
        assert received["session_id"] == "new-sess"


class TestExactLifecycle:
    def test_open_then_reset_finalise_only_old_session(self, monkeypatch):
        """A plugin observing open(A), reset(A->B) must see exactly one open
        for B and the old_session_id A on reset — never a second open for A."""
        import hermes_cli.plugins as plugins_mod

        events: list[tuple[str, dict]] = []

        def fake_invoke_hook(name, **kwargs):
            events.append((name, kwargs))
            return []

        monkeypatch.setattr(plugins_mod, "invoke_hook", fake_invoke_hook)
        from cli import HermesCLI

        cli = HermesCLI.__new__(HermesCLI)
        cli.session_id = "sess-b"
        cli.agent = types.SimpleNamespace(session_id="sess-b")
        cli.platform = "cli"
        # Simulate: open fires with new id, reset fires with old id.
        fake_invoke_hook("on_session_open", session_id="sess-b", platform="cli")
        cli._notify_session_boundary("on_session_reset", old_session_id="sess-a")
        opens = [k for n, k in events if n == "on_session_open"]
        resets = [k for n, k in events if n == "on_session_reset"]
        assert opens == [{"session_id": "sess-b", "platform": "cli"}]
        assert resets == [{"session_id": "sess-b", "old_session_id": "sess-a", "platform": "cli", "reason": "new_session"}]


class TestHostOpenAcrossSurfaces:
    def test_generic_host_open_rejects_empty_and_is_idempotent(self, monkeypatch):
        import hermes_cli.plugins as plugins_mod

        notify = getattr(plugins_mod, "notify_session_open", None)
        assert callable(notify), "core requires a generic non-empty host-open seam"
        events: list[tuple[str, dict]] = []

        monkeypatch.setattr(
            plugins_mod,
            "invoke_hook",
            lambda name, **kwargs: events.append((name, kwargs)) or [],
        )
        unique = "host-open-idempotent-r2"
        assert notify("", "gateway") is False
        assert notify(unique, "gateway") is True
        assert notify(unique, "gateway") is False
        assert events == [
            ("on_session_open", {"session_id": unique, "platform": "gateway"})
        ]

    @pytest.mark.parametrize(
        ("boundary", "boundary_kwargs"),
        [
            (
                "on_session_finalize",
                {"session_id": "resume-after-finalize-r2", "platform": "gateway"},
            ),
            (
                "on_session_reset",
                {
                    "session_id": "replacement-session-r2",
                    "old_session_id": "resume-after-reset-r2",
                    "platform": "cli",
                },
            ),
        ],
    )
    def test_closed_session_id_can_be_opened_again(
        self, monkeypatch, boundary, boundary_kwargs
    ):
        import hermes_cli.plugins as plugins_mod

        session_id = boundary_kwargs.get("old_session_id") or boundary_kwargs["session_id"]
        events: list[tuple[str, dict]] = []

        class FakeManager:
            def invoke_hook(self, name, **kwargs):
                events.append((name, kwargs))
                return []

        monkeypatch.setattr(plugins_mod, "get_plugin_manager", lambda: FakeManager())
        assert plugins_mod.notify_session_open(session_id, "discord") is True
        assert plugins_mod.notify_session_open(session_id, "discord") is False

        # Finalisers do not always report the same platform label used by the
        # opener (for example a Discord session may finalise as "gateway").
        plugins_mod.invoke_hook(boundary, **boundary_kwargs)

        assert plugins_mod.notify_session_open(session_id, "discord") is True
        assert [name for name, _kwargs in events] == [
            "on_session_open",
            boundary,
            "on_session_open",
        ]

    def test_tui_session_create_opens_before_deferred_agent_build(self, monkeypatch, tmp_path):
        from tui_gateway import server

        order: list[tuple[str, str, str]] = []
        monkeypatch.setattr(server, "_completion_cwd", lambda params=None: str(tmp_path))
        monkeypatch.setattr(
            server,
            "_schedule_agent_build",
            lambda sid: order.append(("build", sid, "")),
        )
        monkeypatch.setattr(
            server,
            "_notify_session_open",
            lambda session_id, platform: order.append(("open", session_id, platform)),
            raising=False,
        )
        response = server._methods["session.create"]("r2", {"cols": 80, "source": "tui"})
        sid = response["result"]["session_id"]
        key = response["result"]["stored_session_id"]
        try:
            assert key
            assert order == [("open", key, "tui"), ("build", sid, "")]
        finally:
            session = server._sessions.pop(sid, None)
            if session is not None:
                server._teardown_session(session)

    def test_tui_session_resume_opens_before_deferred_history_hydration(
        self, monkeypatch, tmp_path
    ):
        from tui_gateway import server

        target = "tui-resume-deferred-history-order-r4"
        order: list[tuple[str, str]] = []
        monkeypatch.setattr(server, "_get_db", lambda: _ResumeDB(target))
        monkeypatch.setattr(server, "_default_session_cwd", lambda: str(tmp_path))
        monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
        monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
        monkeypatch.setattr(server, "_lazy_resume_info", lambda *_args, **_kwargs: {})
        monkeypatch.setattr(
            server,
            "_notify_session_open",
            lambda session_id, _platform: order.append(("open", session_id)),
        )
        monkeypatch.setattr(
            server,
            "_schedule_resume_hydration",
            lambda sid, *_args, **_kwargs: order.append(("hydrate", sid)),
        )

        response = server._methods["session.resume"](
            "r4",
            {"session_id": target, "defer_history": True},
        )
        assert "error" not in response
        sid = response["result"]["session_id"]
        try:
            assert order == [("open", target), ("hydrate", sid)]
        finally:
            session = server._sessions.pop(sid, None)
            if session is not None:
                server._teardown_session(session)

    def test_tui_session_resume_opens_before_deferred_agent_build(
        self, monkeypatch, tmp_path
    ):
        from tui_gateway import server

        target = "tui-resume-deferred-order-r3"
        order: list[tuple[str, str]] = []
        monkeypatch.setattr(server, "_get_db", lambda: _ResumeDB(target))
        monkeypatch.setattr(server, "_default_session_cwd", lambda: str(tmp_path))
        monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
        monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
        monkeypatch.setattr(server, "_lazy_resume_info", lambda *_args, **_kwargs: {})
        monkeypatch.setattr(
            server,
            "_notify_session_open",
            lambda session_id, _platform: order.append(("open", session_id)),
        )
        monkeypatch.setattr(
            server,
            "_schedule_agent_build",
            lambda sid: order.append(("build", sid)),
        )

        response = server._methods["session.resume"](
            "r3",
            {"session_id": target},
        )
        assert "error" not in response
        sid = response["result"]["session_id"]
        try:
            assert order == [("open", target), ("build", sid)]
        finally:
            session = server._sessions.pop(sid, None)
            if session is not None:
                server._teardown_session(session)

    def test_tui_session_resume_opens_before_eager_agent_build(self, monkeypatch):
        from tui_gateway import server

        target = "tui-resume-eager-order-r3"
        order: list[tuple[str, str]] = []
        monkeypatch.setattr(server, "_get_db", lambda: _ResumeDB(target))
        monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
        monkeypatch.setattr(
            server,
            "_notify_session_open",
            lambda session_id, _platform: order.append(("open", session_id)),
        )

        def _make_agent(_sid, _key, *, session_id=None, **_kwargs):
            assert session_id is not None
            order.append(("build", session_id))
            return object()

        monkeypatch.setattr(server, "_make_agent", _make_agent)
        monkeypatch.setattr(server, "_init_session", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(server, "_session_info", lambda *_args, **_kwargs: {})

        response = server._methods["session.resume"](
            "r3",
            {"session_id": target, "eager_build": True},
        )
        assert "error" not in response
        assert order == [("open", target), ("build", target)]

    def test_gateway_constructor_opens_before_agent_creation(self, monkeypatch):
        import gateway.run as gateway_run
        import hermes_cli.plugins as plugins_mod

        construct = getattr(gateway_run, "_construct_agent_with_session_open", None)
        assert callable(construct), "gateway requires a host-open constructor seam"
        order: list[tuple[str, str]] = []
        monkeypatch.setattr(
            plugins_mod,
            "notify_session_open",
            lambda session_id, platform: order.append(("open", session_id)) or True,
        )

        def constructor(**kwargs):
            order.append(("construct", kwargs["session_id"]))
            return kwargs

        result = construct(
            lambda: constructor(
                session_id="gateway-session-r2",
                platform="discord",
            ),
            session_id="gateway-session-r2",
            platform="discord",
        )
        assert result["session_id"] == "gateway-session-r2"
        assert order == [
            ("open", "gateway-session-r2"),
            ("construct", "gateway-session-r2"),
        ]
