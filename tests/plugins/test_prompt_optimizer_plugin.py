"""Tests for the prompt-optimizer plugin.

Covers the plugin at ``$HERMES_HOME/plugins/prompt-optimizer/`` (not
bundled under ``hermes-agent/plugins/`` — it lives in the user's HERMES
home and is loaded by ``hermes_cli.plugins`` at startup):

  * ``engine.create_preview`` / ``resolve_preview`` / ``expire_previews``
    — thread-safe pending-preview store used by the TUI overlay.
  * ``get_tui_preview`` bridge: happy path, mode=off, empty input, bypass
    prefixes, optimiser returns ``None``.
  * TUI gateway RPC ``prompt.optimize.preview``: param validation, model
    and provider extracted from the session, exception swallowed into a
    bypass response, plugin-not-importable bypass.
  * Slash command handlers: ``/prompt-optimizer status|auto|off|bogus``,
    ``/prompt-stats --raw``, ``/prompt-insights``.
  * ``register(ctx)`` wires three hooks and six commands.

The LLM is never invoked: ``_run_optimizer_bridge`` is monkeypatched.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "prompt-optimizer"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def hermes_home(tmp_path, monkeypatch):
    """Isolate HERMES_HOME so the plugin writes to a temp metrics DB."""
    home = tmp_path / "hermes_home"
    (home / "plugins" / "prompt-optimizer").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


@pytest.fixture
def plugin(hermes_home, monkeypatch):
    """Load the plugin fresh so HERMES_HOME monkeypatch takes effect.

    The plugin uses ``from .engine import …`` relative imports, so it
    must be loaded as a package. We register a synthetic ``hermes_plugins``
    parent package, then import the plugin as
    ``hermes_plugins.prompt_optimizer`` — matching what the real
    ``hermes_cli.plugins`` loader does in production.
    """
    # KENSEI (test-isolation fix): the sys.modules purge was removed — deleting
    # the hermes_plugins namespace split module identity for later test files.
    # The synthetic parent package registered below replaces the cache entry.

    ns = types.ModuleType("hermes_plugins")
    ns.__path__ = []
    sys.modules["hermes_plugins"] = ns

    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.prompt_optimizer",
        PLUGIN_DIR / "__init__.py",
        submodule_search_locations=[str(PLUGIN_DIR)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.prompt_optimizer"
    mod.__path__ = [str(PLUGIN_DIR)]
    sys.modules["hermes_plugins.prompt_optimizer"] = mod
    spec.loader.exec_module(mod)

    engine = sys.modules["hermes_plugins.prompt_optimizer.engine"]

    # Clear all module-level state between tests
    with mod._mode_lock:
        mod._mode = mod._DEFAULT_MODE
    mod._session_rewrites.clear()
    mod._pending_approvals.clear()
    with engine._pending_store_lock:
        engine._pending_store.clear()

    yield mod


def _fake_record(plugin, original="raw prompt", rewritten="tight prompt"):
    return plugin.RewriteRecord(
        original=original,
        rewritten=rewritten,
        quality_before=42.0,
        quality_after=78.5,
        token_delta_pct=-22.0,
        model_profile="deepseek-v4-flash",
    )


# ---------------------------------------------------------------------------
# Group 1 — get_tui_preview bridge
# ---------------------------------------------------------------------------


class TestGetTuiPreview:
    def test_happy_path_returns_preview_and_stores_it(self, plugin, monkeypatch):
        engine = sys.modules["hermes_plugins.prompt_optimizer.engine"]
        record = _fake_record(plugin)
        monkeypatch.setattr(plugin, "_run_optimizer_bridge",
                            lambda text, model, provider: record)

        out = plugin.get_tui_preview("sess-1", "please make this nicer", "m", "p")

        assert out["status"] == "preview"
        preview = out["preview"]
        # Returned as dict (JSON-serializable) so the TUI gateway transport
        # can encode it. Original dataclass lives in the pending store.
        assert isinstance(preview, dict)
        assert preview["session_key"] == "sess-1"
        assert preview["rewritten"] == "tight prompt"
        assert preview["quality_after"] == 78.5

        # The dataclass instance is stored in the pending store under the
        # session key so the agent thread can resolve it on acceptance.
        stored = engine.resolve_preview("sess-1")
        assert isinstance(stored, engine.PromptOptimizationPreview)
        assert stored.rewritten == "tight prompt"
        assert engine.resolve_preview("sess-1") is None

    def test_mode_off_bypasses(self, plugin, monkeypatch):
        calls = []
        monkeypatch.setattr(plugin, "_run_optimizer_bridge",
                            lambda *a, **kw: calls.append(a) or _fake_record(plugin))
        plugin._mode = "off"

        out = plugin.get_tui_preview("sess-1", "hello", "", "")

        assert out == {"status": "bypass", "reason": "disabled"}
        assert calls == []  # optimiser never invoked

    def test_empty_text_bypasses(self, plugin):
        assert plugin.get_tui_preview("s", "", "", "") == {
            "status": "bypass", "reason": "empty"}
        assert plugin.get_tui_preview("s", "   \n  ", "", "") == {
            "status": "bypass", "reason": "empty"}

    @pytest.mark.parametrize("prefix", ["/quick", "*simple", "#basic"])
    def test_bypass_prefixes(self, plugin, monkeypatch, prefix):
        monkeypatch.setattr(plugin, "_run_optimizer_bridge",
                            lambda *a, **kw: pytest.fail("optimiser should not run"))
        out = plugin.get_tui_preview("s", f"{prefix} do the thing", "", "")
        assert out == {"status": "bypass", "reason": "bypass_prefix"}

    def test_optimiser_returns_none_bypasses(self, plugin, monkeypatch):
        monkeypatch.setattr(plugin, "_run_optimizer_bridge",
                            lambda *a, **kw: None)
        out = plugin.get_tui_preview("s", "real prompt", "", "")
        assert out == {"status": "bypass", "reason": "no_rewrite_produced"}


# ---------------------------------------------------------------------------
# Group 2 — pending preview store
# ---------------------------------------------------------------------------


class TestPendingStore:
    def test_create_then_resolve_roundtrip(self, plugin):
        engine = sys.modules["hermes_plugins.prompt_optimizer.engine"]
        preview = engine.PromptOptimizationPreview(
            session_key="abc", original="o", rewritten="r",
            quality_before=1, quality_after=2, token_delta_pct=0,
            model_profile="x",
        )
        engine.create_preview("abc", preview)
        assert engine.get_pending_preview("abc") is preview
        assert engine.resolve_preview("abc") is preview
        # Resolved → removed
        assert engine.get_pending_preview("abc") is None

    def test_resolve_unknown_key_returns_none(self, plugin):
        engine = sys.modules["hermes_plugins.prompt_optimizer.engine"]
        assert engine.resolve_preview("nope") is None

    def test_expire_drops_stale_entries(self, plugin):
        engine = sys.modules["hermes_plugins.prompt_optimizer.engine"]
        old = engine.PromptOptimizationPreview(
            session_key="old", original="o", rewritten="r",
            quality_before=0, quality_after=0, token_delta_pct=0,
            model_profile="x", created_at=time.time() - 9999,
        )
        fresh = engine.PromptOptimizationPreview(
            session_key="fresh", original="o", rewritten="r",
            quality_before=0, quality_after=0, token_delta_pct=0,
            model_profile="x",
        )
        engine.create_preview("old", old)
        engine.create_preview("fresh", fresh)

        expired = engine.expire_previews(timeout=60)

        assert expired == ["old"]
        assert engine.get_pending_preview("old") is None
        assert engine.get_pending_preview("fresh") is fresh


# ---------------------------------------------------------------------------
# Group 3 — TUI gateway RPC: prompt.optimize.preview
# ---------------------------------------------------------------------------


@pytest.fixture
def tui_server(plugin):
    """Import tui_gateway.server and expose its method registry."""
    # Importing the server is heavy; we only need the dispatcher and _sessions.
    from tui_gateway import server  # noqa: WPS433 — test-only import
    return server


class TestTuiGatewayRpc:
    def test_missing_text_returns_error(self, tui_server):
        fn = tui_server._methods["prompt.optimize.preview"]
        resp = fn("rid-1", {"session_id": "s"})
        assert resp["error"]["code"] == 4004

    def test_missing_session_id_returns_error(self, tui_server):
        fn = tui_server._methods["prompt.optimize.preview"]
        resp = fn("rid-2", {"text": "hi"})
        assert resp["error"]["code"] == 4001

    def test_happy_path_forwards_model_and_provider(self, tui_server, plugin,
                                                    monkeypatch):
        captured = {}

        def fake_get_tui_preview(sid, text, model, provider):
            captured.update(sid=sid, text=text, model=model, provider=provider)
            return {"status": "preview", "preview": {"rewritten": "out"}}

        monkeypatch.setattr(plugin, "get_tui_preview", fake_get_tui_preview)
        tui_server._sessions["sess-7"] = {
            "agent": SimpleNamespace(model="deepseek-v4-flash", provider="nous"),
        }
        try:
            fn = tui_server._methods["prompt.optimize.preview"]
            resp = fn("rid-3", {"text": "hello", "session_id": "sess-7"})
            assert resp["result"]["status"] == "preview"
            assert captured == {
                "sid": "sess-7",
                "text": "hello",
                "model": "deepseek-v4-flash",
                "provider": "nous",
            }
        finally:
            tui_server._sessions.pop("sess-7", None)

    def test_exception_is_swallowed_into_bypass(self, tui_server, plugin,
                                                monkeypatch):
        def boom(*a, **kw):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(plugin, "get_tui_preview", boom)
        tui_server._sessions["sess-x"] = {"agent": SimpleNamespace()}
        try:
            fn = tui_server._methods["prompt.optimize.preview"]
            resp = fn("rid-4", {"text": "hi", "session_id": "sess-x"})
            assert resp["result"]["status"] == "bypass"
            assert "kaboom" in resp["result"]["reason"]
        finally:
            tui_server._sessions.pop("sess-x", None)

    def test_plugin_not_loaded_returns_bypass(self, tui_server, monkeypatch):
        # Replace the package with one that raises on attribute access.
        class _Raises:
            def __getattr__(self, name):
                raise ImportError("plugin gone")

        monkeypatch.setitem(sys.modules, "hermes_plugins.prompt_optimizer",
                            _Raises())
        tui_server._sessions["sess-y"] = {"agent": SimpleNamespace()}
        try:
            fn = tui_server._methods["prompt.optimize.preview"]
            resp = fn("rid-5", {"text": "hi", "session_id": "sess-y"})
            assert resp["result"] == {"status": "bypass",
                                       "reason": "plugin_not_loaded"}
        finally:
            tui_server._sessions.pop("sess-y", None)


# ---------------------------------------------------------------------------
# Group 4 — slash command handlers
# ---------------------------------------------------------------------------


class TestSlashCommands:
    def test_status_reports_current_mode(self, plugin):
        out = plugin._handle_prompt_optimizer("status")
        assert "Mode: auto" in out
        assert "Stored rewrites" in out

    def test_set_mode_transitions(self, plugin):
        out = plugin._handle_prompt_optimizer("off")
        assert "auto" in out and "off" in out
        assert plugin._mode == "off"

        plugin._handle_prompt_optimizer("interactive")
        assert plugin._mode == "interactive"

        plugin._handle_prompt_optimizer("auto")
        assert plugin._mode == "auto"

    def test_unknown_subcommand_does_not_change_mode(self, plugin):
        plugin._mode = "auto"
        out = plugin._handle_prompt_optimizer("bogus")
        assert "Unknown subcommand" in out
        assert plugin._mode == "auto"

    def test_help_when_no_args(self, plugin):
        out = plugin._handle_prompt_optimizer("")
        # The help text mentions the subcommands
        assert "auto" in out and "off" in out

    def test_stats_raw_returns_valid_json_on_empty_db(self, plugin):
        out = plugin._handle_prompt_stats("--raw")
        payload = json.loads(out)
        assert set(payload.keys()) >= {"today", "week", "month"}
        # No rewrites yet → counts are zero/empty.
        assert payload["today"].get("count", 0) == 0

    def test_insights_does_not_crash_on_empty_db(self, plugin):
        out = plugin._handle_prompt_insights("")
        assert isinstance(out, str)
        assert len(out) > 0


# ---------------------------------------------------------------------------
# Group 5 — register() wires hooks and commands
# ---------------------------------------------------------------------------


class TestRegister:
    def test_register_wires_three_hooks_and_six_commands(self, plugin):
        hooks: list[tuple[str, object]] = []
        commands: list[str] = []

        class FakeCtx:
            llm = SimpleNamespace()

            def register_hook(self, event, handler):
                hooks.append((event, handler))

            def register_command(self, name, handler, description="", args_hint=""):
                commands.append(name)

        plugin.register(FakeCtx())

        events = [e for e, _ in hooks]
        assert events == [
            "pre_gateway_dispatch",
            "pre_user_message",
            "transform_llm_output",
        ]
        assert sorted(commands) == sorted([
            "prompt-optimizer",
            "prompt-insights",
            "prompt-compare",
            "prompt-suggestions",
            "prompt-analytics",
            "prompt-stats",
        ])


# ---------------------------------------------------------------------------
# Group 6 — skill invocation skip + hard rewrite timeout
# ---------------------------------------------------------------------------


SKILL_PROMPT = (
    '[IMPORTANT: The user has invoked the "simplify-swarm" skill, '
    "indicating they want you to simplify the current diff.]"
)


class TestSkillInvocationSkip:
    def test_pre_user_message_skips_skill_invocations(self, plugin, monkeypatch):
        monkeypatch.setattr(
            plugin, "_run_optimizer_bridge",
            lambda *a, **k: pytest.fail("optimizer must not run for skill invocations"))
        assert plugin._on_pre_user_message(message=SKILL_PROMPT, session_id="s1") is None

    def test_pre_gateway_dispatch_skips_skill_invocations(self, plugin, monkeypatch):
        monkeypatch.setattr(
            plugin, "_run_optimizer_bridge",
            lambda *a, **k: pytest.fail("optimizer must not run for skill invocations"))
        event = SimpleNamespace(text=SKILL_PROMPT)
        assert plugin._on_pre_gateway_dispatch(event=event) is None

    def test_get_tui_preview_bypasses_skill_invocations(self, plugin, monkeypatch):
        monkeypatch.setattr(
            plugin, "_run_optimizer_bridge",
            lambda *a, **k: pytest.fail("optimizer must not run for skill invocations"))
        out = plugin.get_tui_preview("sess", SKILL_PROMPT)
        assert out == {"status": "bypass", "reason": "skill_invocation"}

    def test_ordinary_prompts_still_optimised(self, plugin, monkeypatch):
        record = _fake_record(plugin)
        monkeypatch.setattr(plugin, "_run_optimizer_bridge", lambda *a, **k: record)
        out = plugin._on_pre_user_message(message="raw prompt", session_id="s1")
        assert out == {"action": "rewrite", "text": record.rewritten}


DELEGATE_PROMPT = (
    "delegate_task with tasks=['research headroom', 'audit compressor', "
    "'benchmark caching'] synthesize=true verify_rubric='cite line numbers'"
)


class TestStructuredCommandSkip:
    def _forbid_optimizer(self, plugin, monkeypatch):
        monkeypatch.setattr(
            plugin, "_run_optimizer_bridge",
            lambda *a, **k: pytest.fail("optimizer must not run for structured commands"))

    def test_pre_user_message_skips_delegate_commands(self, plugin, monkeypatch):
        self._forbid_optimizer(plugin, monkeypatch)
        assert plugin._on_pre_user_message(message=DELEGATE_PROMPT, session_id="s1") is None

    def test_pre_gateway_dispatch_skips_delegate_commands(self, plugin, monkeypatch):
        self._forbid_optimizer(plugin, monkeypatch)
        event = SimpleNamespace(text=DELEGATE_PROMPT)
        assert plugin._on_pre_gateway_dispatch(event=event) is None

    def test_get_tui_preview_bypasses_delegate_commands(self, plugin, monkeypatch):
        self._forbid_optimizer(plugin, monkeypatch)
        out = plugin.get_tui_preview("sess", DELEGATE_PROMPT)
        assert out == {"status": "bypass", "reason": "structured_command"}

    def test_fenced_code_blocks_bypass(self, plugin, monkeypatch):
        self._forbid_optimizer(plugin, monkeypatch)
        msg = "please review this\n```python\nprint('hi')\n```"
        assert plugin._on_pre_user_message(message=msg, session_id="s1") is None

    def test_verb_with_trailing_colon_bypasses(self, plugin, monkeypatch):
        self._forbid_optimizer(plugin, monkeypatch)
        msg = "delegate: split this into three subtasks and synthesise"
        assert plugin._on_pre_user_message(message=msg, session_id="s1") is None

    def test_env_var_extends_verb_list(self, plugin, monkeypatch):
        engine = sys.modules["hermes_plugins.prompt_optimizer.engine"]
        monkeypatch.setenv("PROMPT_OPTIMIZER_BYPASS_VERBS", "fanout, council")
        assert engine.is_structured_command("fanout these 4 tasks")
        assert engine.is_structured_command("council review this plan")
        assert not engine.is_structured_command("plan my week")

    def test_mid_sentence_verb_still_optimised(self, plugin, monkeypatch):
        record = _fake_record(plugin)
        monkeypatch.setattr(plugin, "_run_optimizer_bridge", lambda *a, **k: record)
        out = plugin._on_pre_user_message(
            message="should I delegate this work to someone?", session_id="s1")
        assert out == {"action": "rewrite", "text": record.rewritten}


class TestRewriteWallClockTimeout:
    def test_hung_llm_call_fails_open(self, plugin, monkeypatch):
        """A provider call that ignores its timeout must not block the turn."""
        engine = sys.modules["hermes_plugins.prompt_optimizer.engine"]
        monkeypatch.setattr(engine, "OPTIMIZER_TIMEOUT_S", 0.2)

        class HungLLM:
            def complete(self, **kw):
                time.sleep(5)
                return SimpleNamespace(text="should never be used")

        start = time.monotonic()
        result = engine._try_rewrite_sync("some prompt", HungLLM())
        elapsed = time.monotonic() - start
        assert result is None
        assert elapsed < 2, f"wall-clock cap not enforced ({elapsed:.1f}s)"

    def test_fast_llm_call_still_works(self, plugin):
        engine = sys.modules["hermes_plugins.prompt_optimizer.engine"]

        class FastLLM:
            def complete(self, **kw):
                return SimpleNamespace(text=(
                    "better prompt\n---SCORES---\n"
                    '{"clarity": 80, "specificity": 80, "terminology": 80, '
                    '"actionability": 80, "structure": 80}'
                ))

        result = engine._try_rewrite_sync("some prompt", FastLLM())
        assert result == ("better prompt", 80.0)
