"""Tests for the enhanced security-guidance plugin.

Covers ``plugins/security-guidance/``:

* patterns.py data integrity and severity mapping
* _scan_content — true positives, true negatives, path filtering
* Hooks — transform_tool_result with severity ranking, pre_tool_call blocking,
  post_tool_call + pre_llm_call advisory buffer, execute_code / terminal coverage
* security_scan tool — file, directory, and text scopes
* Bundled-plugin discovery via PluginManager.discover_and_load
"""

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
    monkeypatch.delenv("SECURITY_GUIDANCE_BLOCK", raising=False)
    monkeypatch.delenv("SECURITY_GUIDANCE_DISABLE", raising=False)
    yield hermes_home


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_plugin_init():
    """Import the plugin __init__.py with patterns.py as a sibling."""
    plugin_dir = _repo_root() / "plugins" / "security-guidance"
    if "hermes_plugins" not in sys.modules:
        ns = types.ModuleType("hermes_plugins")
        ns.__path__ = []
        sys.modules["hermes_plugins"] = ns
    spec = importlib.util.spec_from_file_location(
        "hermes_plugins.security_guidance",
        plugin_dir / "__init__.py",
        submodule_search_locations=[str(plugin_dir)],
    )
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = "hermes_plugins.security_guidance"
    mod.__path__ = [str(plugin_dir)]
    sys.modules["hermes_plugins.security_guidance"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Data integrity
# ---------------------------------------------------------------------------

class TestSeverityMapping:
    def test_all_rules_have_severity(self):
        mod = _load_plugin_init()
        assert len(mod._COMPILED) > 0
        for entry in mod._COMPILED:
            assert entry["severity"] in ("critical", "high", "medium", "low", "info")

    def test_critical_rules_include_exec_injection(self):
        mod = _load_plugin_init()
        crit = {e["ruleName"] for e in mod._COMPILED if e["severity"] == "critical"}
        assert "eval_injection" in crit
        assert "os_system_injection" in crit
        assert "python_subprocess_shell" in crit

    def test_high_rules_include_xss_and_crypto(self):
        mod = _load_plugin_init()
        high = {e["ruleName"] for e in mod._COMPILED if e["severity"] == "high"}
        assert "tls_verification_disabled" in high
        assert "innerHTML_xss" in high
        assert "github_actions_workflow" in high


# ---------------------------------------------------------------------------
# _scan_content
# ---------------------------------------------------------------------------

class TestScanContent:
    def test_eval_is_critical(self):
        mod = _load_plugin_init()
        findings = mod._scan_content("/tmp/foo.py", "result = eval(user_input)")
        assert len(findings) >= 1
        rule, severity, reminder = findings[0]
        assert rule == "eval_injection"
        assert severity == "critical"

    def test_script_src_without_sri_is_medium(self):
        mod = _load_plugin_init()
        findings = mod._scan_content(
            "/tmp/foo.html",
            '<script src="https://cdn.example.com/lib.js"></script>',
        )
        names = {f[0] for f in findings}
        assert "script_src_without_sri" in names
        for name, severity, _ in findings:
            if name == "script_src_without_sri":
                assert severity == "medium"

    def test_execute_code_scanned(self):
        mod = _load_plugin_init()
        findings = mod._scan_content("", "eval(x)")
        assert any(f[0] == "eval_injection" for f in findings)


# ---------------------------------------------------------------------------
# Hooks — severity-ranked warnings
# ---------------------------------------------------------------------------

class TestTransformToolResultHook:
    def test_warns_with_severity_ranking(self):
        mod = _load_plugin_init()
        args = {
            "path": "/tmp/foo.py",
            "content": "eval(x)\n\nverify = False\n",
        }
        result = mod._on_transform_tool_result(
            tool_name="write_file",
            args=args,
            result='{"success": true}',
        )
        assert isinstance(result, str)
        assert "Security guidance" in result
        # Both critical (eval) and high (verify=False) should be present
        assert "CRITICAL" in result
        assert "HIGH" in result

    def test_no_warn_on_clean_content(self):
        mod = _load_plugin_init()
        args = {"path": "/tmp/foo.py", "content": "import json\nx = json.loads(b)\n"}
        assert mod._on_transform_tool_result(
            tool_name="write_file", args=args, result='{"success": true}'
        ) is None

    def test_no_warn_when_result_is_error(self):
        mod = _load_plugin_init()
        args = {"path": "/tmp/foo.py", "content": "pickle.load(f)\n"}
        assert mod._on_transform_tool_result(
            tool_name="write_file", args=args, result='{"error": "boom"}'
        ) is None

    def test_disable_kill_switch(self, monkeypatch):
        mod = _load_plugin_init()
        monkeypatch.setenv("SECURITY_GUIDANCE_DISABLE", "1")
        args = {"path": "/tmp/foo.py", "content": "pickle.load(f)\n"}
        assert mod._on_transform_tool_result(
            tool_name="write_file", args=args, result='{"ok": true}'
        ) is None

    def test_block_mode_makes_transform_hook_quiet(self, monkeypatch):
        mod = _load_plugin_init()
        monkeypatch.setenv("SECURITY_GUIDANCE_BLOCK", "1")
        args = {"path": "/tmp/foo.py", "content": "pickle.load(f)\n"}
        assert mod._on_transform_tool_result(
            tool_name="write_file", args=args, result='{"ok": true}'
        ) is None


class TestPreToolCallHook:
    def test_no_block_in_warn_mode(self):
        mod = _load_plugin_init()
        args = {"path": "/tmp/foo.py", "content": "pickle.load(f)\n"}
        assert mod._on_pre_tool_call(tool_name="write_file", args=args) is None

    def test_blocks_in_block_mode_on_dangerous_pattern(self, monkeypatch):
        mod = _load_plugin_init()
        monkeypatch.setenv("SECURITY_GUIDANCE_BLOCK", "1")
        args = {"path": "/tmp/foo.py", "content": "pickle.load(f)\n"}
        out = mod._on_pre_tool_call(tool_name="write_file", args=args)
        assert isinstance(out, dict)
        assert out["action"] == "block"
        assert "pickle_deserialization" in out["message"]
        assert "SECURITY_GUIDANCE_BLOCK" in out["message"]

    def test_no_block_in_block_mode_on_clean_content(self, monkeypatch):
        mod = _load_plugin_init()
        monkeypatch.setenv("SECURITY_GUIDANCE_BLOCK", "1")
        args = {"path": "/tmp/foo.py", "content": "import json\n"}
        assert mod._on_pre_tool_call(tool_name="write_file", args=args) is None


class TestPostToolCallBuffer:
    def test_buffers_findings_for_pre_llm_call(self, monkeypatch):
        mod = _load_plugin_init()
        mod._TURN_ADVISORIES.clear()
        args = {"path": "/tmp/foo.py", "content": "eval(x)\n"}
        mod._on_post_tool_call(tool_name="write_file", args=args, result='{"ok": true}')
        assert len(mod._TURN_ADVISORIES) >= 1
        advisory = mod._flush_advisories()
        assert "Security Guidance" in advisory
        assert "CRITICAL" in advisory

    def test_pre_llm_call_returns_advisory(self, monkeypatch):
        mod = _load_plugin_init()
        mod._TURN_ADVISORIES.clear()
        mod._TURN_ADVISORIES.append({
            "ruleName": "eval_injection",
            "severity": "critical",
            "reminder": "eval is dangerous",
        })
        ctx = mod._on_pre_llm_call(system_message="test")
        assert ctx is not None
        assert "context" in ctx
        assert "CRITICAL" in ctx["context"]

    def test_pre_llm_call_returns_none_when_empty(self, monkeypatch):
        mod = _load_plugin_init()
        mod._TURN_ADVISORIES.clear()
        assert mod._on_pre_llm_call(system_message="test") is None


class TestExecuteCodeAndTerminalCoverage:
    def test_execute_code_scanned_by_scan_args(self):
        mod = _load_plugin_init()
        findings = mod._scan_args(
            "execute_code", {"code": "eval(x)\n"}
        )
        assert any(f[0] == "eval_injection" for f in findings)

    def test_terminal_command_scanned_by_scan_args(self):
        mod = _load_plugin_init()
        # subprocess shell=True fires even without path filter
        findings = mod._scan_args(
            "terminal", {"command": "subprocess.run('ls', shell=True)"}
        )
        assert any(
            f[0] == "python_subprocess_shell" for f in findings
        )


# ---------------------------------------------------------------------------
# security_scan tool
# ---------------------------------------------------------------------------

class TestSecurityScanTool:
    def test_scan_text_directly(self):
        mod = _load_plugin_init()
        result = mod.security_scan({"target": "eval(x)", "scope": "text"})
        assert "eval_injection" in result
        assert "CRITICAL" in result

    def test_scan_file(self, tmp_path: Path):
        mod = _load_plugin_init()
        f = tmp_path / "test.py"
        f.write_text("pickle.load(open('x.pkl','rb'))")
        result = mod.security_scan({"target": str(f), "scope": "file"})
        assert "pickle_deserialization" in result
        assert "CRITICAL" in result
        assert "Summary" in result

    def test_scan_directory(self, tmp_path: Path):
        mod = _load_plugin_init()
        (tmp_path / "a.py").write_text("eval(x)")
        result = mod.security_scan({"target": str(tmp_path), "scope": "directory"})
        assert "eval_injection" in result
        assert "CRITICAL" in result

    def test_scan_returns_clean_message_when_no_findings(self):
        mod = _load_plugin_init()
        result = mod.security_scan({"target": "print('hello')", "scope": "text"})
        assert "No known vulnerability patterns" in result

    def test_scan_accepts_auto_inference(self, tmp_path: Path):
        mod = _load_plugin_init()
        f = tmp_path / "auto.py"
        f.write_text("eval(x)")
        # scope omitted — should auto-infer from path
        result = mod.security_scan({"target": str(f)})
        assert "eval_injection" in result


# ---------------------------------------------------------------------------
# Bundled-plugin discovery
# ---------------------------------------------------------------------------

class TestPluginDiscovery:
    def test_loads_via_plugin_manager(self, _isolate_env, monkeypatch):
        import yaml

        config = {"plugins": {"enabled": ["security-guidance"]}}
        (_isolate_env / "config.yaml").write_text(yaml.safe_dump(config))

        for k in list(sys.modules):
            if k.startswith(("hermes_plugins", "hermes_cli.plugins")):
                del sys.modules[k]

        from hermes_cli.plugins import _ensure_plugins_discovered

        mgr = _ensure_plugins_discovered(force=True)
        loaded = set()
        if hasattr(mgr, "_plugins"):
            loaded = set(mgr._plugins.keys())
        assert "security-guidance" in loaded
