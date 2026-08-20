"""Tests for the hermes-approval-guard plugin.

Covers ``plugins/hermes-approval-guard/``:

  * ``_SAFE_TOOLS`` — ``browser_console`` must NOT be unconditionally
    bypassed (it can evaluate JavaScript in the page context).
  * Config reload — a failed ``load_config`` must not be cached, so
    enabling the plugin post-startup takes effect on the next TTL refresh.
  * Fail-closed paths — a Stage 2 exception returns a ``block`` directive
    when ``fail_open: false`` and ``None`` when ``fail_open: true``.
  * Stage 2 reviewer boundary — the ACP subprocess strips
    ``HERMES_YOLO_MODE`` from the child env and restricts the reviewer to
    the read-only ``session_search`` toolset.
  * Windows behaviour — the timeout kill path falls back to
    ``proc.terminate()/kill()`` when ``os.killpg`` / ``signal.SIGKILL``
    are unavailable.
  * Hook dispatch — the registered ``pre_tool_call`` callback tolerates
    the extra kwargs the core dispatcher passes (turn_id,
    api_request_id, middleware_trace, telemetry_schema_version).
  * Bundled-plugin discovery via ``PluginManager.discover_and_load``.
"""

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.delenv("HERMES_SAFE_MODE", raising=False)
    yield hermes_home


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

_PKG = "approval_guard_ut"


def _plugin_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "plugins" / "hermes-approval-guard"


def _load_guard_package():
    """Import the plugin modules as a package so relative imports work.

    Uses a private package name so these instances never collide with the
    PluginManager-loaded copies (``hermes_plugins.hermes_approval_guard.*``).
    """
    plugin_dir = _plugin_dir()
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [str(plugin_dir)]
    pkg.__package__ = _PKG
    sys.modules[_PKG] = pkg
    mods = {}
    for name in (
        "feedback", "stage1_rules", "stage1_llm",
        "stage2_acp", "hindsight_store", "guard",
    ):
        full = f"{_PKG}.{name}"
        spec = importlib.util.spec_from_file_location(full, plugin_dir / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = _PKG
        sys.modules[full] = mod
        spec.loader.exec_module(mod)
        mods[name] = mod
    return mods


@pytest.fixture
def guard_pkg(monkeypatch):
    mods = _load_guard_package()
    # Each test starts with an empty config cache.
    monkeypatch.setattr(mods["guard"], "_config_cache", None)
    monkeypatch.setattr(mods["guard"], "_config_cache_time", 0.0)
    return mods


def _write_guard_config(hermes_home, guard_cfg):
    import yaml
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"plugin_guard": guard_cfg})
    )


# ---------------------------------------------------------------------------
# _SAFE_TOOLS bypass boundary (review: browser_console is not read-only)
# ---------------------------------------------------------------------------

class TestSafeToolsBoundary:
    def test_browser_console_not_in_safe_tools(self, guard_pkg):
        """browser_console(expression=...) executes JavaScript in the page
        context (tools/browser_tool.py) — it must go through review."""
        assert "browser_console" not in guard_pkg["guard"]._SAFE_TOOLS

    def test_browser_console_reaches_stage1(self, guard_pkg, monkeypatch, _isolate_env):
        """Dispatch-level proof: browser_console is reviewed, not bypassed."""
        _write_guard_config(_isolate_env, {"enabled": True})
        calls = []
        monkeypatch.setattr(
            guard_pkg["stage1_llm"], "llm_classify",
            lambda *a, **k: calls.append(a[0]) or "ALLOW",
        )
        guard_pkg["guard"].pre_tool_call_handler(
            "browser_console", {"expression": "document.cookie"}
        )
        assert calls == ["browser_console"]

    def test_read_only_browser_tools_still_bypassed(self, guard_pkg, monkeypatch, _isolate_env):
        _write_guard_config(_isolate_env, {"enabled": True})
        calls = []
        monkeypatch.setattr(
            guard_pkg["stage1_llm"], "llm_classify",
            lambda *a, **k: calls.append(a[0]) or "ALLOW",
        )
        assert guard_pkg["guard"].pre_tool_call_handler(
            "browser_snapshot", {}
        ) is None
        assert calls == []


# ---------------------------------------------------------------------------
# Config reload (review: failed/disabled config must not latch forever)
# ---------------------------------------------------------------------------

class TestConfigReload:
    def test_load_failure_is_not_cached(self, guard_pkg, monkeypatch):
        """A broken config backend disables the guard for that call only —
        the next TTL refresh must retry (no permanent _config_disable latch)."""
        import hermes_cli.config as hc

        def _boom():
            raise RuntimeError("config backend down")

        monkeypatch.setattr(hc, "load_config", _boom)
        guard = guard_pkg["guard"]
        assert guard.pre_tool_call_handler("write_file", {"path": "/tmp/x"}) is None
        assert guard._config_cache is None

    def test_enable_after_startup_takes_effect(self, guard_pkg, monkeypatch, _isolate_env):
        """Config written after startup is picked up — no restart needed."""
        import hermes_cli.config as hc

        guard = guard_pkg["guard"]
        original_load = hc.load_config
        monkeypatch.setattr(
            hc, "load_config",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        assert guard.pre_tool_call_handler("write_file", {"path": "/tmp/x"}) is None

        # Config backend recovers with the guard now enabled.
        monkeypatch.setattr(hc, "load_config", original_load)
        _write_guard_config(_isolate_env, {"enabled": True})
        calls = []
        monkeypatch.setattr(
            guard_pkg["stage1_llm"], "llm_classify",
            lambda *a, **k: calls.append(a[0]) or "ALLOW",
        )
        assert guard.pre_tool_call_handler("write_file", {"path": "/tmp/x"}) is None
        assert calls == ["write_file"], "enabled config never took effect"


# ---------------------------------------------------------------------------
# Fail-closed Stage 2 (review: fail_open=false must block, not pass)
# ---------------------------------------------------------------------------

class TestStage2FailureBranches:
    def _dispatch_with_stage2_error(self, guard_pkg, monkeypatch, hermes_home, fail_open):
        _write_guard_config(hermes_home, {
            "enabled": True,
            "fail_open": fail_open,
            "stage2": {"enabled": True},
        })
        monkeypatch.setattr(
            guard_pkg["stage1_llm"], "llm_classify", lambda *a, **k: "ESCALATE"
        )

        def _explode(*a, **k):
            raise RuntimeError("acp subprocess died")

        monkeypatch.setattr(guard_pkg["stage2_acp"], "acp_agent_review", _explode)
        return guard_pkg["guard"].pre_tool_call_handler(
            "write_file", {"path": "/etc/nginx/nginx.conf"}
        )

    def test_fail_open_true_passes_on_stage2_exception(
        self, guard_pkg, monkeypatch, _isolate_env
    ):
        result = self._dispatch_with_stage2_error(
            guard_pkg, monkeypatch, _isolate_env, fail_open=True
        )
        assert result is None

    def test_fail_open_false_blocks_on_stage2_exception(
        self, guard_pkg, monkeypatch, _isolate_env
    ):
        """Regression: both branches of the exception handler used to
        return None, silently allowing the tool call."""
        result = self._dispatch_with_stage2_error(
            guard_pkg, monkeypatch, _isolate_env, fail_open=False
        )
        assert isinstance(result, dict)
        assert result["action"] == "block"
        assert "stage2_exception" in result["message"]

    def test_stage2_deny_produces_block_directive(
        self, guard_pkg, monkeypatch, _isolate_env
    ):
        _write_guard_config(_isolate_env, {
            "enabled": True,
            "fail_open": True,
            "stage2": {"enabled": True},
        })
        monkeypatch.setattr(
            guard_pkg["stage1_llm"], "llm_classify", lambda *a, **k: "ESCALATE"
        )
        monkeypatch.setattr(
            guard_pkg["stage2_acp"], "acp_agent_review",
            lambda *a, **k: ("DENY", {"reason": "unit_test_deny"}),
        )
        result = guard_pkg["guard"].pre_tool_call_handler(
            "write_file", {"path": "/etc/nginx/nginx.conf"}
        )
        assert isinstance(result, dict)
        assert result["action"] == "block"
        assert "unit_test_deny" in result["message"]


# ---------------------------------------------------------------------------
# Stage 2 reviewer boundary (review: reviewer must be genuinely read-only)
# ---------------------------------------------------------------------------

class _FakeProc:
    pid = 1234

    def __init__(self, stdout='{"verdict":"ALLOW","reason":"ok"}', stderr="", returncode=0):
        self._stdout = stdout
        self._stderr = stderr
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def communicate(self, timeout=None):
        return self._stdout, self._stderr

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True


class TestStage2ReviewerBoundary:
    def _run_review(self, guard_pkg, monkeypatch):
        stage2 = guard_pkg["stage2_acp"]
        captured = {}

        def _fake_popen(argv, **kwargs):
            captured["argv"] = argv
            captured["kwargs"] = kwargs
            return _FakeProc()

        monkeypatch.setattr(stage2.subprocess, "Popen", _fake_popen)
        # Keep Hindsight memory I/O out of the test.
        monkeypatch.setattr(
            stage2, "_query_context",
            lambda *a, **k: {
                "session_history": "",
                "pattern_history": "",
                "compression_summary": "",
            },
        )
        monkeypatch.setattr(stage2, "_store_decision", lambda *a, **k: None)

        verdict, _detail = stage2.acp_agent_review(
            "terminal",
            {"command": "rm -rf build"},
            {"fail_open": True, "stage2": {"enabled": True, "timeout": 15}},
            {"signals": ["WARNING: Dangerous pattern triggered"]},
            {"turns": ["User: clean the build output"], "tool_calls": ["terminal ls"]},
            "t1",
            "s1",
        )
        return verdict, captured

    def test_reviewer_env_strips_yolo_mode(self, guard_pkg, monkeypatch):
        """HERMES_YOLO_MODE is frozen at tools.approval import time — the
        reviewer subprocess must not inherit it from a --yolo parent."""
        monkeypatch.setenv("HERMES_YOLO_MODE", "1")
        verdict, captured = self._run_review(guard_pkg, monkeypatch)
        assert verdict == "ALLOW"
        env = captured["kwargs"]["env"]
        assert "HERMES_YOLO_MODE" not in env

    def test_reviewer_restricted_to_read_only_toolset(self, guard_pkg, monkeypatch):
        """The reviewer only gets session_search — no file/terminal toolset."""
        verdict, captured = self._run_review(guard_pkg, monkeypatch)
        assert verdict == "ALLOW"
        argv = captured["argv"]
        assert "-t" in argv
        assert argv[argv.index("-t") + 1] == "session_search"


class TestReviewProcessKill:
    def test_posix_kills_process_group(self, guard_pkg, monkeypatch):
        stage2 = guard_pkg["stage2_acp"]
        sent = []
        monkeypatch.setattr(stage2._os, "getpgid", lambda pid: 4242)
        monkeypatch.setattr(
            stage2._os, "killpg",
            lambda pgid, sig: sent.append((pgid, sig)),
            raising=False,
        )
        proc = _FakeProc()
        stage2._kill_review_process(proc)
        assert sent == [(4242, stage2.signal.SIGTERM)]

    def test_windows_fallback_kills_direct_child(self, guard_pkg, monkeypatch):
        """Windows has no os.killpg / signal.SIGKILL — the timeout path must
        not raise AttributeError; it terminates the direct child instead."""
        stage2 = guard_pkg["stage2_acp"]
        monkeypatch.delattr(stage2._os, "killpg")
        monkeypatch.delattr(stage2.signal, "SIGKILL")
        proc = _FakeProc()
        out, _err = stage2._kill_review_process(proc)
        assert proc.terminated
        assert out == proc._stdout


# ---------------------------------------------------------------------------
# Bundled-plugin discovery
# ---------------------------------------------------------------------------

class TestBundledDiscovery:
    def test_discovered_but_not_loaded_by_default(self, _isolate_env):
        """Bundled plugins are discovered but NOT loaded without opt-in."""
        from hermes_cli import plugins as pmod

        mgr = pmod.PluginManager()
        mgr.discover_and_load()
        assert "hermes-approval-guard" in mgr._plugins
        loaded = mgr._plugins["hermes-approval-guard"]
        assert loaded.manifest.source == "bundled"
        assert not loaded.enabled

    def test_loads_via_plugin_manager_when_enabled(self, _isolate_env):
        """End-to-end: enable in config.yaml and verify the PluginManager
        picks it up via the standard discovery path and registers the hook."""
        import yaml

        config = {"plugins": {"enabled": ["hermes-approval-guard"]}}
        (_isolate_env / "config.yaml").write_text(yaml.safe_dump(config))

        # Wipe any cached plugin state from earlier tests in this worker.
        for k in list(sys.modules):
            if k.startswith(("hermes_plugins", "hermes_cli.plugins")):
                del sys.modules[k]

        from hermes_cli.plugins import _ensure_plugins_discovered

        mgr = _ensure_plugins_discovered(force=True)
        assert "hermes-approval-guard" in mgr._plugins
        assert mgr._plugins["hermes-approval-guard"].enabled
        assert mgr.has_hook("pre_tool_call")


# ---------------------------------------------------------------------------
# pre_tool_call hook dispatch (through the real core dispatcher)
# ---------------------------------------------------------------------------

class TestHookDispatch:
    @pytest.fixture
    def dispatch(self, _isolate_env, monkeypatch):
        """Enable the plugin and return the fresh plugins module + manager."""
        import yaml

        config = {
            "plugins": {"enabled": ["hermes-approval-guard"]},
            "plugin_guard": {
                "enabled": True,
                "fail_open": True,
                "stage2": {"enabled": True},
            },
        }
        (_isolate_env / "config.yaml").write_text(yaml.safe_dump(config))

        for k in list(sys.modules):
            if k.startswith(("hermes_plugins", "hermes_cli.plugins")):
                del sys.modules[k]

        import hermes_cli.plugins as pmod

        mgr = pmod._ensure_plugins_discovered(force=True)
        assert mgr._plugins["hermes-approval-guard"].enabled
        return pmod, mgr

    def _patch_stages(self, monkeypatch, stage1_verdict, stage2_result):
        s1 = importlib.import_module(
            "hermes_plugins.hermes_approval_guard.stage1_llm"
        )
        s2 = importlib.import_module(
            "hermes_plugins.hermes_approval_guard.stage2_acp"
        )
        monkeypatch.setattr(
            s1, "llm_classify", lambda *a, **k: stage1_verdict
        )
        monkeypatch.setattr(
            s2, "acp_agent_review", lambda *a, **k: stage2_result
        )

    def test_safe_tool_passes_through_dispatch(self, dispatch):
        pmod, _mgr = dispatch
        directive, message = pmod.get_pre_tool_call_directive(
            "read_file", {"path": "/etc/passwd"}
        )
        assert directive is None
        assert message is None

    def test_allow_verdict_passes_through_dispatch(self, dispatch, monkeypatch):
        pmod, _mgr = dispatch
        self._patch_stages(monkeypatch, "ALLOW", ("ALLOW", {"reason": "ok"}))
        directive, _message = pmod.get_pre_tool_call_directive(
            "write_file", {"path": "/etc/nginx/nginx.conf"}
        )
        assert directive is None

    def test_deny_verdict_becomes_block_directive(self, dispatch, monkeypatch):
        """Full path: core dispatcher → plugin hook → stage1 ESCALATE →
        stage2 DENY → {"action": "block"} directive."""
        pmod, _mgr = dispatch
        self._patch_stages(monkeypatch, "ESCALATE", ("DENY", {"reason": "dispatch_deny"}))
        directive, message = pmod.get_pre_tool_call_directive(
            "write_file", {"path": "/etc/nginx/nginx.conf"}
        )
        assert directive == "block"
        assert "dispatch_deny" in message

    def test_handler_tolerates_extra_dispatch_kwargs(self, dispatch):
        """The core dispatcher passes turn_id / api_request_id /
        middleware_trace / telemetry_schema_version — the handler must
        absorb them instead of raising TypeError (which invoke_hook would
        swallow, silently neutering the plugin)."""
        pmod, _mgr = dispatch
        directive, _message = pmod.get_pre_tool_call_directive(
            "read_file",
            {"path": "/tmp/x"},
            task_id="t1",
            session_id="s1",
            tool_call_id="tc1",
            turn_id="turn1",
            api_request_id="req1",
            middleware_trace=[{"mw": "test"}],
        )
        assert directive is None
