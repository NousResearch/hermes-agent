"""Tests for the bundled observability/langfuse plugin."""
from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path

import pytest

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_DIR = REPO_ROOT / "plugins" / "observability" / "langfuse"


# ---------------------------------------------------------------------------
# Manifest + layout
# ---------------------------------------------------------------------------

class TestManifest:

    def test_manifest_fields(self):
        data = yaml.safe_load((PLUGIN_DIR / "plugin.yaml").read_text())
        assert data["name"] == "langfuse"
        assert data["version"]
        # All six hooks the plugin implements.
        assert set(data["hooks"]) == {
            "pre_api_request", "post_api_request",
            "pre_llm_call", "post_llm_call",
            "pre_tool_call", "post_tool_call",
        }
        # Required env vars are the user-facing HERMES_ prefixed keys.
        assert "HERMES_LANGFUSE_PUBLIC_KEY" in data["requires_env"]
        assert "HERMES_LANGFUSE_SECRET_KEY" in data["requires_env"]


# ---------------------------------------------------------------------------
# Plugin discovery: langfuse is opt-in (not loaded unless explicitly enabled).
# This guards against someone accidentally re-introducing a per-hook
# load_config() gate or making the plugin auto-load.
# ---------------------------------------------------------------------------

class TestDiscovery:
    def test_plugin_is_discovered_as_standalone_opt_in(self, tmp_path, monkeypatch):
        """Scanner should find the plugin but NOT load it by default."""
        from hermes_cli import plugins as plugins_mod

        # Isolated HERMES_HOME so we don't read the developer's config.yaml.
        home = tmp_path / ".hermes"
        home.mkdir()
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setattr(Path, "home", lambda: tmp_path)

        manager = plugins_mod.PluginManager()
        manager.discover_and_load()

        # observability/langfuse appears in the plugin registry …
        loaded = manager._plugins.get("observability/langfuse")
        assert loaded is not None, "plugin not discovered"
        # … but is not loaded (opt-in default → no config.yaml means nothing enabled)
        assert loaded.enabled is False
        assert "not enabled" in (loaded.error or "").lower()


# ---------------------------------------------------------------------------
# Runtime gate: _get_langfuse() returns None and caches _INIT_FAILED when
# credentials are missing. Guards against regressing toward the rejected
# per-hook load_config() design.
# ---------------------------------------------------------------------------

class TestRuntimeGate:
    def _fresh_plugin(self):
        """Import the plugin module fresh (clears any cached client)."""
        mod_name = "plugins.observability.langfuse"
        sys.modules.pop(mod_name, None)
        return importlib.import_module(mod_name)

    def test_get_langfuse_returns_none_without_credentials(self, monkeypatch):
        for k in (
            "HERMES_LANGFUSE_PUBLIC_KEY", "HERMES_LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
        ):
            monkeypatch.delenv(k, raising=False)

        langfuse_plugin = self._fresh_plugin()
        assert langfuse_plugin._get_langfuse() is None

    def test_get_langfuse_caches_failure_no_config_load(self, monkeypatch):
        """A miss must be cached — no per-hook config.yaml reads, no env re-reads."""
        for k in (
            "HERMES_LANGFUSE_PUBLIC_KEY", "HERMES_LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
        ):
            monkeypatch.delenv(k, raising=False)

        langfuse_plugin = self._fresh_plugin()

        # Prime the cache with one call.
        assert langfuse_plugin._get_langfuse() is None

        # Now block os.environ.get — a correctly-cached plugin must not
        # touch env again.
        import os
        called = {"n": 0}
        real_get = os.environ.get

        def tracking_get(key, default=None):
            if key.startswith(("HERMES_LANGFUSE_", "LANGFUSE_")):
                called["n"] += 1
            return real_get(key, default)

        monkeypatch.setattr(os.environ, "get", tracking_get)

        for _ in range(20):
            assert langfuse_plugin._get_langfuse() is None

        assert called["n"] == 0, (
            f"_get_langfuse() re-read env {called['n']} times after cache miss — "
            "it should short-circuit via _INIT_FAILED"
        )


# ---------------------------------------------------------------------------
# Hooks are inert when the client is unavailable.
# ---------------------------------------------------------------------------

class TestHooksInert:
    def test_hooks_noop_without_client(self, monkeypatch):
        """All 6 hooks must return without raising when _get_langfuse() is None."""
        for k in (
            "HERMES_LANGFUSE_PUBLIC_KEY", "HERMES_LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
        ):
            monkeypatch.delenv(k, raising=False)

        sys.modules.pop("plugins.observability.langfuse", None)
        import importlib
        mod = importlib.import_module("plugins.observability.langfuse")

        # Each hook should just return; no exceptions.
        mod.on_pre_llm_call(task_id="t", session_id="s", messages=[{"role": "user", "content": "hi"}])
        mod.on_pre_llm_request(task_id="t", session_id="s", api_call_count=1, request_messages=[])
        mod.on_post_llm_call(task_id="t", session_id="s", api_call_count=1)
        mod.on_pre_tool_call(tool_name="read_file", args={}, task_id="t", session_id="s")
        mod.on_post_tool_call(tool_name="read_file", args={}, result="ok", task_id="t", session_id="s")


class TestPayloadSanitization:
    def test_safe_value_redacts_base64_data_uri_instead_of_truncating(self):
        sys.modules.pop("plugins.observability.langfuse", None)
        import importlib
        mod = importlib.import_module("plugins.observability.langfuse")

        payload = "data:image/png;base64," + ("a" * 20000)
        result = mod._safe_value(payload)

        assert result == {
            "type": "data_uri",
            "media_type": "image/png",
            "omitted": True,
            "length": len(payload),
        }

    def test_serialize_messages_redacts_data_uri_parts(self):
        sys.modules.pop("plugins.observability.langfuse", None)
        import importlib
        mod = importlib.import_module("plugins.observability.langfuse")

        payload = "data:image/jpeg;base64," + ("b" * 20000)
        serialized = mod._serialize_messages([
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": payload}}]}
        ])

        assert serialized[0]["content"][0]["image_url"]["url"] == {
            "type": "data_uri",
            "media_type": "image/jpeg",
            "omitted": True,
            "length": len(payload),
        }


class TestTraceScopeKey:
    def _fresh_plugin(self):
        mod_name = "plugins.observability.langfuse"
        sys.modules.pop(mod_name, None)
        return importlib.import_module(mod_name)

    def test_trace_key_scopes_by_turn_id_when_available(self):
        plugin = self._fresh_plugin()

        key_a = plugin._trace_key("task-1", "session-1", turn_id="turn-a")
        key_b = plugin._trace_key("task-1", "session-1", turn_id="turn-b")

        assert key_a != key_b
        assert "turn:turn-a" in key_a
        assert "turn:turn-b" in key_b


# ---------------------------------------------------------------------------
# End-to-end collision regression: two turns of ONE gateway session must not
# share trace state.  The helper-level tests above prove _trace_key returns
# distinct keys; this drives the real pre/post hooks to prove the keys are
# actually threaded through so the second turn gets its own root trace.
#
# Gateway reality this reproduces:
#   * task_id == session_id for every turn        (gateway/run.py)
#   * turn_id is unique per turn                   (turn_context.py)
#   * api_call_count resets to 1 each turn         (conversation_loop.py)
#
# Before the turn/request scoping, _trace_key collapsed to the constant
# session_id.  That worked only because _finish_trace pops the key on a clean
# turn end.  When turn 1 does NOT finalize (interrupted, tool-only final step,
# or empty final content), its state lingered under session_id and turn 2
# silently merged into turn 1's trace instead of opening its own.
# ---------------------------------------------------------------------------


class TestTurnTraceIsolation:
    def _fresh_plugin(self):
        sys.modules.pop("plugins.observability.langfuse", None)
        return importlib.import_module("plugins.observability.langfuse")

    @staticmethod
    def _fake_client(started):
        """A minimal Langfuse stand-in that records each root trace opened.

        ``_start_root_trace`` calls ``create_trace_id`` then opens a root via
        ``start_as_current_observation(...)`` (a context manager whose
        ``__enter__`` returns the root span).  We record one entry per root
        actually opened so the test can count distinct traces.
        """

        class _Span:
            def update(self, **kw):
                pass

            def end(self, **kw):
                pass

            def set_trace_io(self, **kw):
                pass

            def start_observation(self, **kw):
                return _Span()

        class _RootCM:
            def __enter__(self):
                return _Span()

            def __exit__(self, *exc):
                return False

        class _Client:
            def create_trace_id(self, seed=None):
                return f"trace::{seed}"

            def start_as_current_observation(self, **kw):
                started.append(kw.get("trace_context", {}).get("trace_id"))
                return _RootCM()

            def flush(self):
                pass

        return _Client()

    def _run_turn(self, mod, *, session, turn_n, finalize):
        """Drive one turn through the request-scoped hooks the gateway fires."""
        task_id = session  # gateway sets task_id == session_id
        turn_id = f"{session}:{task_id}:turn{turn_n}"
        api_call_count = 1  # resets every turn
        api_request_id = f"{turn_id}:api:{api_call_count}"

        mod.on_pre_llm_request(
            task_id=task_id,
            session_id=session,
            model="m",
            provider="p",
            api_mode="chat",
            api_call_count=api_call_count,
            request_messages=[{"role": "user", "content": "hi"}],
            turn_id=turn_id,
            api_request_id=api_request_id,
        )
        # finalize=False => leave a tool call on the final response so
        # _finish_trace is skipped and the turn's state lingers.
        mod.on_post_llm_call(
            task_id=task_id,
            session_id=session,
            model="m",
            provider="p",
            api_mode="chat",
            api_call_count=api_call_count,
            assistant_content_chars=5 if finalize else 0,
            assistant_tool_call_count=0 if finalize else 1,
            usage={"input_tokens": 10, "output_tokens": 5},
            turn_id=turn_id,
            api_request_id=api_request_id,
        )

    def test_unfinalized_turn_does_not_capture_next_turn(self, monkeypatch):
        """A turn that never finalizes must not absorb the following turn."""
        mod = self._fresh_plugin()
        started: list = []
        monkeypatch.setattr(mod, "_get_langfuse", lambda: self._fake_client(started))
        monkeypatch.setattr(mod, "_end_observation", lambda *a, **k: None)
        mod._TRACE_STATE.clear()

        # Turn 1 ends without finalizing (its final step still has a tool call).
        self._run_turn(mod, session="sess-iso", turn_n=1, finalize=False)
        # Turn 2 is a normal, fully finalizing turn in the SAME session.
        self._run_turn(mod, session="sess-iso", turn_n=2, finalize=True)

        # Each turn opened its OWN root trace.  On the pre-fix code the second
        # turn reused turn 1's lingering state and only one trace was opened.
        assert len(started) == 2

        # Turn 2 finalized and was popped by _finish_trace; only turn 1's
        # (non-finalizing) state lingers.  Assert the surviving key is turn 1's
        # and that turn 2 never merged into it — `all(...)` over an empty set
        # would pass vacuously, so pin the exact surviving key instead.
        keys = list(mod._TRACE_STATE.keys())
        assert len(keys) == 1
        assert "turn1" in keys[0]
        assert "turn2" not in keys[0]

    def test_pre_and_post_hooks_share_one_key_within_a_turn(self, monkeypatch):
        """turn_id is preferred over api_request_id so the turn-scoped
        post_llm_call (which carries no api_request_id) still resolves to the
        same key as the request-scoped pre/post_api_request hooks.  If the
        ordering were reversed, finalization would silently break."""
        mod = self._fresh_plugin()
        turn_id = "S:T:turnX"
        api_request_id = f"{turn_id}:api:1"

        k_pre_api = mod._trace_key("T", "S", turn_id=turn_id, api_request_id=api_request_id)
        k_post_api = mod._trace_key("T", "S", turn_id=turn_id, api_request_id=api_request_id)
        k_post_turn = mod._trace_key("T", "S", turn_id=turn_id, api_request_id="")

        assert k_pre_api == k_post_api == k_post_turn

    def test_non_finalizing_turns_do_not_grow_state_unboundedly(self, monkeypatch):
        """Per-turn keys mean a turn that never finalizes leaves a lingering
        entry.  Without a cap that grows once per non-finalizing turn forever;
        the LRU eviction must bound _TRACE_STATE at _MAX_TRACE_STATE.
        """
        mod = self._fresh_plugin()
        started: list = []
        monkeypatch.setattr(mod, "_get_langfuse", lambda: self._fake_client(started))
        monkeypatch.setattr(mod, "_end_observation", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_MAX_TRACE_STATE", 8)
        mod._TRACE_STATE.clear()

        # Far more non-finalizing turns than the cap.
        for n in range(50):
            self._run_turn(mod, session="sess-leak", turn_n=n, finalize=False)

        assert len(mod._TRACE_STATE) <= 8
        # The survivors are the most-recently-updated turns (LRU eviction).
        surviving = sorted(int(k.rsplit("turn", 1)[1]) for k in mod._TRACE_STATE)
        assert surviving == list(range(42, 50))


# ---------------------------------------------------------------------------
# Placeholder-credential guard (#23823).
#
# Regression coverage for the silent-failure bug: when an operator leaves
# HERMES_LANGFUSE_PUBLIC_KEY / SECRET_KEY at a template value like
# "placeholder", "test-key", or "your-langfuse-key", the SDK accepts the
# credentials at construction time (it does no server-side validation
# eagerly) but drops every trace at flush time, with no signal in the
# Hermes logs.  The fix in `_get_langfuse()` validates the documented
# `pk-lf-` / `sk-lf-` prefix Langfuse always issues, surfaces a one-shot
# warning naming the offending env var(s), and short-circuits via the
# same `_INIT_FAILED` path used for missing credentials so subsequent
# hook invocations don't re-log.
# ---------------------------------------------------------------------------


class _FakeLangfuse:
    """Stand-in for the real :class:`langfuse.Langfuse` so tests don't
    need the optional ``langfuse`` SDK installed.  The plugin's runtime
    gate refuses to proceed past ``if Langfuse is None`` when the SDK
    is missing, which would short-circuit before the placeholder check
    can fire.  Patching ``plugin.Langfuse`` with this class lets the
    placeholder validator exercise its full code path."""

    instances: list["_FakeLangfuse"] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        _FakeLangfuse.instances.append(self)


class TestPlaceholderKeyDetection:
    LOGGER_NAME = "plugins.observability.langfuse"

    def _fresh_plugin(self, monkeypatch=None):
        mod_name = "plugins.observability.langfuse"
        sys.modules.pop(mod_name, None)
        mod = importlib.import_module(mod_name)
        if monkeypatch is not None:
            # Pretend the SDK is installed so `_get_langfuse()` actually
            # reaches the placeholder check.  Real SDK calls are never
            # made because the placeholder/missing-credentials paths
            # return before constructing a client.
            _FakeLangfuse.instances.clear()
            monkeypatch.setattr(mod, "Langfuse", _FakeLangfuse, raising=False)
        return mod

    @staticmethod
    def _clear_env(monkeypatch):
        for k in (
            "HERMES_LANGFUSE_PUBLIC_KEY", "HERMES_LANGFUSE_SECRET_KEY",
            "LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY",
        ):
            monkeypatch.delenv(k, raising=False)

    # -- helper unit tests (no SDK stub needed: these don't go through
    #    _get_langfuse, they exercise the pure-Python helpers directly) ------


    def test_validate_langfuse_key_accepts_documented_prefix(self, monkeypatch):
        self._clear_env(monkeypatch)
        plugin = self._fresh_plugin()
        assert plugin._validate_langfuse_key(
            "HERMES_LANGFUSE_PUBLIC_KEY", "pk-lf-real-public-xyz"
        ) is None
        assert plugin._validate_langfuse_key(
            "HERMES_LANGFUSE_SECRET_KEY", "sk-lf-real-secret-xyz"
        ) is None


    # -- end-to-end _get_langfuse() behaviour --------------------------------
    # These tests pass `monkeypatch` to _fresh_plugin() so the helper can
    # stub out `Langfuse` (the optional SDK).  Without that, every call
    # short-circuits at `if Langfuse is None` before reaching the
    # placeholder validator — masking the very behaviour we're testing.

    def test_placeholder_public_key_warns_and_skips(self, monkeypatch, caplog):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_LANGFUSE_PUBLIC_KEY", "placeholder")
        monkeypatch.setenv("HERMES_LANGFUSE_SECRET_KEY", "sk-lf-real-secret-xyz")
        plugin = self._fresh_plugin(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            assert plugin._get_langfuse() is None
        text = caplog.text
        assert "HERMES_LANGFUSE_PUBLIC_KEY" in text
        assert "'placeholder'" in text
        assert "pk-lf-" in text
        # The valid secret value must NOT appear (the var NAME does, in
        # the "or unset ..." hint, but the value preview shouldn't).
        assert "'sk-lf-" not in text
        # Never constructed the SDK client — short-circuited before that.
        assert _FakeLangfuse.instances == []

    def test_placeholder_secret_key_warns_and_skips(self, monkeypatch, caplog):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_LANGFUSE_PUBLIC_KEY", "pk-lf-real-public-xyz")
        monkeypatch.setenv("HERMES_LANGFUSE_SECRET_KEY", "test-key")
        plugin = self._fresh_plugin(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            assert plugin._get_langfuse() is None
        text = caplog.text
        assert "HERMES_LANGFUSE_SECRET_KEY" in text
        assert "'test-key'" in text
        assert "sk-lf-" in text
        # The valid public value must NOT appear.
        assert "'pk-lf-" not in text
        assert _FakeLangfuse.instances == []

    def test_both_placeholders_one_warning_with_both_keys(self, monkeypatch, caplog):
        self._clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_LANGFUSE_PUBLIC_KEY", "placeholder")
        monkeypatch.setenv("HERMES_LANGFUSE_SECRET_KEY", "placeholder")
        plugin = self._fresh_plugin(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            assert plugin._get_langfuse() is None
        warnings = [r for r in caplog.records if r.levelname == "WARNING"
                    and r.name == self.LOGGER_NAME]
        assert len(warnings) == 1, (
            f"Expected a single combined warning; got {len(warnings)}:\n"
            + "\n".join(r.getMessage() for r in warnings)
        )
        text = warnings[0].getMessage()
        assert "HERMES_LANGFUSE_PUBLIC_KEY" in text
        assert "HERMES_LANGFUSE_SECRET_KEY" in text

    def test_repeated_calls_do_not_re_warn(self, monkeypatch, caplog):
        """The cached ``_INIT_FAILED`` sentinel must short-circuit
        subsequent calls so each hook invocation isn't a fresh log
        line — otherwise a busy gateway will spam the operator's
        terminal."""
        self._clear_env(monkeypatch)
        monkeypatch.setenv("HERMES_LANGFUSE_PUBLIC_KEY", "placeholder")
        monkeypatch.setenv("HERMES_LANGFUSE_SECRET_KEY", "placeholder")
        plugin = self._fresh_plugin(monkeypatch)
        with caplog.at_level(logging.WARNING, logger=self.LOGGER_NAME):
            for _ in range(15):
                assert plugin._get_langfuse() is None
        warnings = [r for r in caplog.records if r.levelname == "WARNING"
                    and r.name == self.LOGGER_NAME]
        assert len(warnings) == 1, (
            f"Warning fired {len(warnings)} times across 15 calls; "
            "expected 1 (cached via _INIT_FAILED)"
        )


class TestRequestMessageCoercion:
    def test_prefers_request_messages_then_messages_then_history_then_user_message(self):
        sys.modules.pop("plugins.observability.langfuse", None)
        mod = importlib.import_module("plugins.observability.langfuse")

        assert mod._coerce_request_messages(
            request_messages=[{"role": "system", "content": "s"}],
            messages=[{"role": "user", "content": "m"}],
            conversation_history=[{"role": "user", "content": "h"}],
            user_message="u",
        ) == [{"role": "system", "content": "s"}]
        assert mod._coerce_request_messages(
            messages=[{"role": "user", "content": "m"}],
            conversation_history=[{"role": "user", "content": "h"}],
            user_message="u",
        ) == [{"role": "user", "content": "m"}]
        assert mod._coerce_request_messages(
            conversation_history=[{"role": "user", "content": "h"}],
            user_message="u",
        ) == [{"role": "user", "content": "h"}]
        assert mod._coerce_request_messages(user_message="u") == [{"role": "user", "content": "u"}]


class TestToolCallOutputBackfill:
    def test_post_tool_call_backfills_matching_turn_tool_call_output(self, monkeypatch):
        sys.modules.pop("plugins.observability.langfuse", None)
        mod = importlib.import_module("plugins.observability.langfuse")

        observation = object()
        state = mod.TraceState(trace_id="trace-1", root_ctx=None, root_span=None)
        state.tools["call-1"] = observation
        state.turn_tool_calls.append({
            "id": "call-1",
            "type": "function",
            "name": "web_extract",
            "arguments": '{"urls": ["https://example.com"]}',
            "function": {
                "name": "web_extract",
                "arguments": '{"urls": ["https://example.com"]}',
            },
        })

        task_key = mod._trace_key("task-1", "session-1")
        monkeypatch.setitem(mod._TRACE_STATE, task_key, state)

        ended = {}

        def fake_end_observation(obs, *, output=None, metadata=None, usage_details=None, cost_details=None):
            ended["observation"] = obs
            ended["output"] = output
            ended["metadata"] = metadata

        monkeypatch.setattr(mod, "_end_observation", fake_end_observation)

        mod.on_post_tool_call(
            tool_name="web_extract",
            args={"urls": ["https://example.com"]},
            result='{"results": [{"url": "https://example.com", "content": "Example Domain"}]}',
            task_id="task-1",
            session_id="session-1",
            tool_call_id="call-1",
        )

        assert ended["observation"] is observation
        assert state.turn_tool_calls[0]["output"] == ended["output"]
        assert state.turn_tool_calls[0]["function"]["output"] == ended["output"]
        assert state.turn_tool_calls[0]["output"] == {
            "results": [{"url": "https://example.com", "content": "Example Domain"}]
        }

    def test_serialize_messages_keeps_tool_name_and_call_id(self):
        sys.modules.pop("plugins.observability.langfuse", None)
        mod = importlib.import_module("plugins.observability.langfuse")

        messages = [{
            "role": "tool",
            "name": "web_extract",
            "tool_call_id": "call-1",
            "content": '{"ok": true}',
        }]

        assert mod._serialize_messages(messages) == [{
            "role": "tool",
            "name": "web_extract",
            "tool_call_id": "call-1",
            "content": {"ok": True},
        }]


class TestToolObservationKeying:
    """Tests for pre/post tool_call observation matching when tool_call_id is absent."""

    def _make_mod(self):
        sys.modules.pop("plugins.observability.langfuse", None)
        return importlib.import_module("plugins.observability.langfuse")

    def test_empty_tool_call_id_single_tool_sets_output(self, monkeypatch):
        mod = self._make_mod()
        obs = object()
        state = mod.TraceState(trace_id="t", root_ctx=None, root_span=None)
        state.pending_tools_by_name.setdefault("my_tool", []).append(obs)

        task_key = mod._trace_key("task-1", "sess-1")
        monkeypatch.setitem(mod._TRACE_STATE, task_key, state)

        ended = {}

        def fake_end(o, *, output=None, metadata=None, **kw):
            ended["obs"] = o
            ended["output"] = output

        monkeypatch.setattr(mod, "_end_observation", fake_end)

        mod.on_post_tool_call(
            tool_name="my_tool",
            args={},
            result='{"ok": true}',
            task_id="task-1",
            session_id="sess-1",
            tool_call_id="",
        )

        assert ended["obs"] is obs
        assert ended["output"] == {"ok": True}
        assert state.pending_tools_by_name.get("my_tool") is None


    def test_threaded_post_calls_preserve_fifo_under_lock(self, monkeypatch):
        """The actual concurrency contract: when 8 threads race to drain
        the pending queue, no observation is consumed twice and none is
        lost.  Validates ``_STATE_LOCK`` discipline, not Python list
        semantics."""
        import threading

        mod = self._make_mod()
        n = 8
        observations = [object() for _ in range(n)]
        state = mod.TraceState(trace_id="t", root_ctx=None, root_span=None)
        state.pending_tools_by_name["web_extract"] = list(observations)

        task_key = mod._trace_key("task-thr", "sess-thr")
        monkeypatch.setitem(mod._TRACE_STATE, task_key, state)

        recorded: list = []
        lock = threading.Lock()

        def fake_end(o, *, output=None, metadata=None, **kw):
            with lock:
                recorded.append(o)

        monkeypatch.setattr(mod, "_end_observation", fake_end)

        barrier = threading.Barrier(n)

        def worker():
            barrier.wait()
            mod.on_post_tool_call(
                tool_name="web_extract", args={}, result='{"ok": true}',
                task_id="task-thr", session_id="sess-thr", tool_call_id="",
            )

        threads = [threading.Thread(target=worker) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Every observation was consumed exactly once; queue is empty.
        assert len(recorded) == n
        assert set(map(id, recorded)) == set(map(id, observations))
        assert state.pending_tools_by_name.get("web_extract") is None

    def test_explicit_tool_call_id_uses_tools_dict(self, monkeypatch):
        """When tool_call_id is present, pending_tools_by_name is not touched."""
        mod = self._make_mod()
        obs = object()
        state = mod.TraceState(trace_id="t", root_ctx=None, root_span=None)
        state.tools["call-99"] = obs

        task_key = mod._trace_key("task-1", "sess-1")
        monkeypatch.setitem(mod._TRACE_STATE, task_key, state)

        ended = {}

        def fake_end(o, *, output=None, metadata=None, **kw):
            ended["obs"] = o
            ended["output"] = output

        monkeypatch.setattr(mod, "_end_observation", fake_end)

        mod.on_post_tool_call(
            tool_name="my_tool", args={}, result='{"status": "done"}',
            task_id="task-1", session_id="sess-1", tool_call_id="call-99",
        )

        assert ended["obs"] is obs
        assert ended["output"] == {"status": "done"}
        assert not state.tools


class TestUsageFromSanitizedResponse:
    """Regression: ``post_api_request`` delivers ``response`` as a sanitized
    dict (no ``.usage`` attribute) plus a separate ``usage`` summary dict. The
    post-call handler must read the ``usage`` dict instead of treating the dict
    response as a usage-bearing object and dropping all token/cost data."""

    def _setup(self, mod, monkeypatch):
        # Active client so on_post_llm_call does not early-return.
        monkeypatch.setattr(mod, "_get_langfuse", lambda: object())
        observation = object()
        state = mod.TraceState(trace_id="trace-1", root_ctx=None, root_span=None)
        state.generations[mod._request_key(1)] = observation
        monkeypatch.setitem(mod._TRACE_STATE, mod._trace_key("task-1", "session-1"), state)
        captured = {}

        def fake_end_observation(obs, *, output=None, metadata=None, usage_details=None, cost_details=None):
            captured["usage_details"] = usage_details

        monkeypatch.setattr(mod, "_end_observation", fake_end_observation)
        return captured

    def test_sanitized_dict_response_uses_usage_dict(self, monkeypatch):
        sys.modules.pop("plugins.observability.langfuse", None)
        mod = importlib.import_module("plugins.observability.langfuse")
        captured = self._setup(mod, monkeypatch)

        # A plain dict has no ``.usage`` attribute — mirrors post_api_request.
        mod.on_post_llm_call(
            task_id="task-1",
            session_id="session-1",
            api_call_count=1,
            model="gemini-3-flash-preview",
            response={"model": "gemini-3-flash-preview", "usage": {"input_tokens": 100, "output_tokens": 20}},
            usage={"input_tokens": 100, "output_tokens": 20},
            assistant_content_chars=42,
        )

        # Before the fix the dict response shadowed the usage dict and tokens
        # were lost (usage_details == {}).
        assert captured["usage_details"] == {"input": 100, "output": 20}

    def test_real_response_object_with_usage_still_used(self, monkeypatch):
        sys.modules.pop("plugins.observability.langfuse", None)
        mod = importlib.import_module("plugins.observability.langfuse")
        captured = self._setup(mod, monkeypatch)

        # A response object that genuinely carries usage must still take the
        # response-object path (post_llm_call / legacy behavior).
        seen = {}

        def fake_usage_and_cost(resp, **_):
            seen["resp"] = resp
            return {"input": 7, "output": 3}, {}

        monkeypatch.setattr(mod, "_usage_and_cost", fake_usage_and_cost)

        class _Resp:
            usage = {"prompt_tokens": 7, "completion_tokens": 3}

        resp = _Resp()
        mod.on_post_llm_call(
            task_id="task-1",
            session_id="session-1",
            api_call_count=1,
            model="gemini-3-flash-preview",
            response=resp,
            usage={"input_tokens": 999, "output_tokens": 999},
            assistant_content_chars=42,
        )

        assert seen["resp"] is resp
        assert captured["usage_details"] == {"input": 7, "output": 3}


# ---------------------------------------------------------------------------
# Shutdown drain (#shutdown-typeerror).
#
# Regression: the root observation context manager was entered via
# ``__enter__`` but only ``span.end()`` was ever called — the matching
# ``__exit__`` never ran. The suspended generator was left at ``yield``
# until interpreter teardown GC'd it, by which point
# ``opentelemetry.trace.Span`` had been nulled out during module
# finalization, surfacing as
# ``TypeError: isinstance() arg 2 must be a type`` from
# ``opentelemetry.trace.use_span.__exit__`` on every CLI exit.
#
# The fix: ``_exit_root_context`` calls ``__exit__`` on the stored CM,
# ``_finish_trace`` and ``_evict_stale_locked`` invoke it, and an
# ``atexit``-registered ``_drain_all_traces`` finalizes any live traces
# before teardown. These tests pin all three paths.
# ---------------------------------------------------------------------------


class TestShutdownDrain:
    def _fresh_plugin(self):
        sys.modules.pop("plugins.observability.langfuse", None)
        return importlib.import_module("plugins.observability.langfuse")

    @staticmethod
    def _tracking_client():
        """Langfuse stand-in that records root-CM __exit__ calls."""

        exited: list[str] = []

        class _Span:
            def update(self, **kw):
                pass

            def end(self, **kw):
                pass

            def set_trace_io(self, **kw):
                pass

            def start_observation(self, **kw):
                return _Span()

        class _RootCM:
            def __init__(self, trace_id):
                self._trace_id = trace_id

            def __enter__(self):
                return _Span()

            def __exit__(self, *exc):
                exited.append(self._trace_id)
                return False

        class _Client:
            def create_trace_id(self, seed=None):
                return f"trace::{seed}"

            def start_as_current_observation(self, **kw):
                return _RootCM(kw.get("trace_context", {}).get("trace_id"))

            def flush(self):
                pass

        return _Client(), exited

    def test_finish_trace_exits_root_context_manager(self, monkeypatch):
        """_finish_trace must call __exit__ on the root CM, not just span.end()."""
        mod = self._fresh_plugin()
        client, exited = self._tracking_client()
        monkeypatch.setattr(mod, "_get_langfuse", lambda: client)
        monkeypatch.setattr(mod, "_end_observation", lambda *a, **k: None)
        mod._TRACE_STATE.clear()

        # Start a root trace by firing the pre-LLM request hook.
        mod.on_pre_llm_request(
            task_id="task-fin",
            session_id="sess-fin",
            model="m",
            provider="p",
            api_mode="chat",
            api_call_count=1,
            request_messages=[{"role": "user", "content": "hi"}],
            turn_id="sess-fin:task-fin:turn1",
            api_request_id="sess-fin:task-fin:turn1:api:1",
        )
        assert not exited, "root CM should still be suspended mid-turn"

        # Finalize the turn — this pops state and must close the CM.
        mod.on_post_llm_call(
            task_id="task-fin",
            session_id="sess-fin",
            model="m",
            provider="p",
            api_mode="chat",
            api_call_count=1,
            assistant_content_chars=5,
            assistant_tool_call_count=0,
            usage={"input_tokens": 1, "output_tokens": 1},
            turn_id="sess-fin:task-fin:turn1",
            api_request_id="sess-fin:task-fin:turn1:api:1",
        )

        assert len(exited) == 1
        assert mod._TRACE_STATE == {}

    def test_drain_all_traces_finalizes_live_traces(self, monkeypatch):
        """_drain_all_traces (atexit handler) closes CMs for non-finalizing turns."""
        mod = self._fresh_plugin()
        client, exited = self._tracking_client()
        monkeypatch.setattr(mod, "_get_langfuse", lambda: client)
        monkeypatch.setattr(mod, "_end_observation", lambda *a, **k: None)
        mod._TRACE_STATE.clear()

        # Two turns that never finalize (tool-only final step).
        for n in (1, 2):
            mod.on_pre_llm_request(
                task_id=f"task-{n}",
                session_id="sess-drain",
                model="m",
                provider="p",
                api_mode="chat",
                api_call_count=1,
                request_messages=[{"role": "user", "content": "hi"}],
                turn_id=f"sess-drain:task-{n}:turn{n}",
                api_request_id=f"sess-drain:task-{n}:turn{n}:api:1",
            )
            mod.on_post_llm_call(
                task_id=f"task-{n}",
                session_id="sess-drain",
                model="m",
                provider="p",
                api_mode="chat",
                api_call_count=1,
                assistant_content_chars=0,
                assistant_tool_call_count=1,
                usage={"input_tokens": 1, "output_tokens": 1},
                turn_id=f"sess-drain:task-{n}:turn{n}",
                api_request_id=f"sess-drain:task-{n}:turn{n}:api:1",
            )

        assert len(mod._TRACE_STATE) == 2
        assert exited == []

        # The atexit drain must finalize both.
        mod._drain_all_traces()

        assert len(exited) == 2
        assert mod._TRACE_STATE == {}

    def test_evict_stale_exits_root_context_manager(self, monkeypatch):
        """LRU eviction must also close the CM, not just end the span."""
        mod = self._fresh_plugin()
        client, exited = self._tracking_client()
        monkeypatch.setattr(mod, "_get_langfuse", lambda: client)
        monkeypatch.setattr(mod, "_end_observation", lambda *a, **k: None)
        monkeypatch.setattr(mod, "_MAX_TRACE_STATE", 4)
        mod._TRACE_STATE.clear()

        # Overflow the cap so the oldest entry gets evicted.
        for n in range(6):
            mod.on_pre_llm_request(
                task_id=f"task-ev-{n}",
                session_id="sess-ev",
                model="m",
                provider="p",
                api_mode="chat",
                api_call_count=1,
                request_messages=[{"role": "user", "content": "hi"}],
                turn_id=f"sess-ev:task-ev-{n}:turn{n}",
                api_request_id=f"sess-ev:task-ev-{n}:turn{n}:api:1",
            )

        # At least the evicted entries had their CMs closed.
        assert len(exited) >= 2
        assert len(mod._TRACE_STATE) <= 4

    def test_drain_is_idempotent(self, monkeypatch):
        """Calling _drain_all_traces twice must not raise (atexit + explicit)."""
        mod = self._fresh_plugin()
        client, _ = self._tracking_client()
        monkeypatch.setattr(mod, "_get_langfuse", lambda: client)
        monkeypatch.setattr(mod, "_end_observation", lambda *a, **k: None)
        mod._TRACE_STATE.clear()

        mod._drain_all_traces()  # empty — no-op
        mod._drain_all_traces()  # still no-op
