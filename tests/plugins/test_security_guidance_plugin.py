"""Tests for the security-guidance plugin.

Covers ``plugins/security-guidance/``:

  * ``patterns.py`` data integrity — every rule has a ``RuleId``, the
    fail-loud import assertion is wired.
  * ``_scan_content`` — true positives (pickle.load, yaml.load, eval,
    dangerouslySetInnerHTML, GitHub Actions workflow), true negatives
    (.md skips Python rules, ``model.eval()`` doesn't trip eval),
    path-only rules (``path_check``), content-only rules
    (``path_filter``).
  * Hooks — ``transform_tool_result`` appends a warning block in warn
    mode and stays out of error results; ``pre_tool_call`` blocks
    writes when ``SECURITY_GUIDANCE_BLOCK=1`` and stays silent
    otherwise.
  * Bundled-plugin discovery via ``PluginManager.discover_and_load``.
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


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_patterns():
    """Import patterns.py in isolation (no plugin glue)."""
    pat_path = _repo_root() / "plugins" / "security-guidance" / "patterns.py"
    spec = importlib.util.spec_from_file_location(
        "security_guidance_patterns_under_test", pat_path
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
# patterns.py data integrity
# ---------------------------------------------------------------------------

class TestPatternsData:
    def test_has_at_least_one_rule(self):
        p = _load_patterns()
        assert len(p.SECURITY_PATTERNS) >= 1

    def test_every_rule_has_required_fields(self):
        p = _load_patterns()
        for rule in p.SECURITY_PATTERNS:
            assert "ruleName" in rule
            assert "reminder" in rule and rule["reminder"]
            # At least one of substrings/regex/path_check must be present —
            # otherwise the rule could never fire.
            assert any(k in rule for k in ("substrings", "regex", "path_check")), rule

    def test_rule_names_are_unique(self):
        p = _load_patterns()
        names = [r["ruleName"] for r in p.SECURITY_PATTERNS]
        assert len(names) == len(set(names))

    def test_rule_id_enum_in_sync(self):
        # The upstream patterns.py asserts this at import time. If the
        # set diverges, the import itself raises and this test fails.
        p = _load_patterns()
        rule_names = {r["ruleName"] for r in p.SECURITY_PATTERNS}
        enum_names = set(p._RULE_NAME_TO_ID)
        assert rule_names == enum_names

    def test_rule_names_to_mask_packs_bits(self):
        p = _load_patterns()
        # PICKLE_DESERIALIZATION = 8, EVAL_INJECTION = 4 → bits 8 and 4 set.
        mask = p.rule_names_to_mask({"pickle_deserialization", "eval_injection"})
        assert mask & (1 << p.RuleId.PICKLE_DESERIALIZATION)
        assert mask & (1 << p.RuleId.EVAL_INJECTION)


# ---------------------------------------------------------------------------
# _scan_content
# ---------------------------------------------------------------------------

class TestScanContent:
    def test_pickle_load_in_py_warns(self):
        mod = _load_plugin_init()
        findings = mod._scan_content(
            "/tmp/foo.py", "import pickle\nx = pickle.load(open('p.pkl', 'rb'))\n"
        )
        names = [n for n, _ in findings]
        assert "pickle_deserialization" in names

    def test_pickle_load_in_md_skipped_by_path_filter(self):
        mod = _load_plugin_init()
        findings = mod._scan_content(
            "/tmp/foo.md", "import pickle\nx = pickle.load(open('p.pkl', 'rb'))\n"
        )
        assert findings == []

    def test_method_call_eval_does_not_trip(self):
        """model.eval() / redis.eval() / spec.eval() must not match eval_injection."""
        mod = _load_plugin_init()
        findings = mod._scan_content("/tmp/foo.py", "model.eval()\nout = model(x)\n")
        assert "eval_injection" not in [n for n, _ in findings]

    def test_bare_eval_in_py_warns(self):
        mod = _load_plugin_init()
        findings = mod._scan_content("/tmp/foo.py", "result = eval(user_input)\n")
        assert "eval_injection" in [n for n, _ in findings]

    def test_subprocess_shell_true_warns(self):
        mod = _load_plugin_init()
        findings = mod._scan_content(
            "/tmp/foo.py", "subprocess.run('ls ' + path, shell=True)\n"
        )
        assert "python_subprocess_shell" in [n for n, _ in findings]

    def test_dangerously_set_inner_html_warns(self):
        mod = _load_plugin_init()
        findings = mod._scan_content(
            "/tmp/foo.tsx", "<div dangerouslySetInnerHTML={{__html: x}} />"
        )
        assert "react_dangerously_set_html" in [n for n, _ in findings]

    def test_new_function_in_markdown_is_skipped(self):
        mod = _load_plugin_init()
        findings = mod._scan_content(
            "/tmp/foo.md", "Avoid new Function('return x') in browser code."
        )
        assert "new_function_injection" not in [n for n, _ in findings]

    def test_inner_html_in_markdown_is_skipped(self):
        mod = _load_plugin_init()
        findings = mod._scan_content(
            "/tmp/foo.md", "Example: el.innerHTML = '<b>demo</b>'"
        )
        assert "innerHTML_xss" not in [n for n, _ in findings]

    def test_github_workflow_path_check_fires_on_path_alone(self):
        """github_actions_workflow has no regex/substring — fires on path."""
        mod = _load_plugin_init()
        findings = mod._scan_content(
            ".github/workflows/test.yml", "name: CI\non: pull_request"
        )
        assert "github_actions_workflow" in [n for n, _ in findings]

    def test_non_workflow_path_doesnt_trip_workflow_rule(self):
        mod = _load_plugin_init()
        findings = mod._scan_content("/tmp/foo.py", "name: CI")
        assert "github_actions_workflow" not in [n for n, _ in findings]

    def test_empty_content_returns_no_findings(self):
        mod = _load_plugin_init()
        assert mod._scan_content("/tmp/foo.py", "") == []

    def test_huge_content_skipped(self):
        mod = _load_plugin_init()
        # 1 MB of content with a dangerous pattern at the end — scanner caps
        # out at _MAX_SCAN_BYTES (256 KB), so this should return [].
        big = "x" * (1024 * 1024) + "\npickle.load(open('p.pkl', 'rb'))\n"
        assert mod._scan_content("/tmp/foo.py", big) == []


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

class TestTransformToolResultHook:
    def test_warns_on_write_file_with_dangerous_content(self):
        mod = _load_plugin_init()
        args = {
            "path": "/tmp/foo.py",
            "content": "import pickle\nx = pickle.loads(b)\n",
        }
        result = mod._on_transform_tool_result(
            tool_name="write_file",
            args=args,
            result='{"success": true, "bytes_written": 30}',
        )
        assert isinstance(result, str)
        assert "Security guidance" in result
        assert "pickle_deserialization" in result
        # The original JSON should still be there at the start of the string.
        assert result.startswith('{"success": true')

    def test_no_warn_on_clean_content(self):
        mod = _load_plugin_init()
        args = {"path": "/tmp/foo.py", "content": "import json\nx = json.loads(b)\n"}
        assert (
            mod._on_transform_tool_result(
                tool_name="write_file", args=args, result='{"success": true}'
            )
            is None
        )

    def test_no_warn_when_result_is_error(self):
        mod = _load_plugin_init()
        args = {"path": "/tmp/foo.py", "content": "pickle.load(f)\n"}
        # When the tool itself errored, we don't pile a security warning on
        # top — the model has bigger problems to solve.
        assert (
            mod._on_transform_tool_result(
                tool_name="write_file", args=args, result='{"error": "boom"}'
            )
            is None
        )

    def test_patch_tool_new_string_scanned(self):
        mod = _load_plugin_init()
        args = {
            "path": "/tmp/foo.py",
            "old_string": "x = 1",
            "new_string": "x = eval(user_input)",
        }
        result = mod._on_transform_tool_result(
            tool_name="patch", args=args, result='{"success": true}'
        )
        assert isinstance(result, str)
        assert "eval_injection" in result

    def test_untargeted_tool_skipped(self):
        mod = _load_plugin_init()
        # The plugin only scans write_file/patch/skill_manage. terminal output
        # should pass through untouched.
        args = {"command": "echo pickle.load"}
        assert (
            mod._on_transform_tool_result(
                tool_name="terminal", args=args, result='{"output": "pickle.load"}'
            )
            is None
        )

    def test_disable_kill_switch(self, monkeypatch):
        mod = _load_plugin_init()
        monkeypatch.setenv("SECURITY_GUIDANCE_DISABLE", "1")
        args = {"path": "/tmp/foo.py", "content": "pickle.load(f)\n"}
        assert (
            mod._on_transform_tool_result(
                tool_name="write_file", args=args, result='{"ok": true}'
            )
            is None
        )

    def test_block_mode_makes_transform_hook_quiet(self, monkeypatch):
        """In block mode, pre_tool_call handles the warning; the transform
        hook stays silent so we don't double-emit."""
        mod = _load_plugin_init()
        monkeypatch.setenv("SECURITY_GUIDANCE_BLOCK", "1")
        args = {"path": "/tmp/foo.py", "content": "pickle.load(f)\n"}
        assert (
            mod._on_transform_tool_result(
                tool_name="write_file", args=args, result='{"ok": true}'
            )
            is None
        )


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
        assert "SECURITY_GUIDANCE_BLOCK" in out["message"]  # tells user how to disable

    def test_no_block_in_block_mode_on_clean_content(self, monkeypatch):
        mod = _load_plugin_init()
        monkeypatch.setenv("SECURITY_GUIDANCE_BLOCK", "1")
        args = {"path": "/tmp/foo.py", "content": "import json\n"}
        assert mod._on_pre_tool_call(tool_name="write_file", args=args) is None

    def test_untargeted_tool_skipped(self, monkeypatch):
        mod = _load_plugin_init()
        monkeypatch.setenv("SECURITY_GUIDANCE_BLOCK", "1")
        args = {"command": "echo pickle.load(f)"}
        assert mod._on_pre_tool_call(tool_name="terminal", args=args) is None


# ---------------------------------------------------------------------------
# Bundled-plugin discovery
# ---------------------------------------------------------------------------

class TestPluginDiscovery:
    def test_loads_via_plugin_manager(self, _isolate_env, monkeypatch):
        """End-to-end: enable in config.yaml and verify the PluginManager
        picks it up via the standard discovery path."""
        import yaml

        config = {"plugins": {"enabled": ["security-guidance"]}}
        (_isolate_env / "config.yaml").write_text(yaml.safe_dump(config))

        # Wipe any cached plugin state from earlier tests in this worker.
        for k in list(sys.modules):
            if k.startswith(("hermes_plugins", "hermes_cli.plugins")):
                del sys.modules[k]

        from hermes_cli.plugins import _ensure_plugins_discovered

        mgr = _ensure_plugins_discovered(force=True)
        loaded = set()
        if hasattr(mgr, "_plugins"):
            loaded = set(mgr._plugins.keys())
        assert "security-guidance" in loaded
