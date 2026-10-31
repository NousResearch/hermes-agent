"""Regression tests: -z/--oneshot must honor agent.disabled_toolsets (#61184).

The oneshot path builds its AIAgent directly (bypassing HermesCLI), so the
global disable list has to be forwarded explicitly — otherwise the strict
resolved-tool subtraction in model_tools (#17309) never runs for -z sessions
and only the name-level pre-subtraction in _get_platform_tools applies.
These tests pin the config-read helper and the forwarding contract.
"""

import ast
from pathlib import Path

from hermes_cli.oneshot import _disabled_toolsets_from_config


class TestDisabledToolsetsFromConfig:
    def test_none_and_non_dict_cfg(self):
        assert _disabled_toolsets_from_config(None) is None
        assert _disabled_toolsets_from_config("not-a-dict") is None

    def test_missing_agent_section(self):
        assert _disabled_toolsets_from_config({}) is None
        assert _disabled_toolsets_from_config({"agent": None}) is None

    def test_empty_list_returns_none(self):
        assert _disabled_toolsets_from_config({"agent": {"disabled_toolsets": []}}) is None

    def test_plain_list_passthrough(self):
        cfg = {"agent": {"disabled_toolsets": ["memory", " browser "]}}
        assert _disabled_toolsets_from_config(cfg) == ["memory", "browser"]

    def test_scalar_string_is_one_name(self):
        cfg = {"agent": {"disabled_toolsets": "memory"}}
        assert _disabled_toolsets_from_config(cfg) == ["memory"]

    def test_json_array_string_form_parsed(self):
        # `hermes config set` / JSON-mode editor saves store lists as quoted
        # JSON strings (#86661); the helper must not treat that as one name.
        cfg = {"agent": {"disabled_toolsets": '["memory", "browser"]'}}
        assert _disabled_toolsets_from_config(cfg) == ["memory", "browser"]

    def test_whitespace_entries_filtered(self):
        cfg = {"agent": {"disabled_toolsets": ["  ", "memory"]}}
        assert _disabled_toolsets_from_config(cfg) == ["memory"]


class TestForwardingContract:
    def test_aiagent_call_passes_disabled_toolsets(self):
        """The AIAgent(...) construction in oneshot must forward the kwarg.

        Pinned via AST so a refactor that rebuilds the call without
        disabled_toolsets fails here instead of silently reverting #61184.
        """
        import hermes_cli.oneshot as oneshot_mod

        source = Path(oneshot_mod.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        aiagent_calls = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                (isinstance(node.func, ast.Name) and node.func.id == "AIAgent")
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "AIAgent")
            )
        ]
        assert aiagent_calls, "expected at least one AIAgent(...) call in oneshot"
        for call in aiagent_calls:
            kwarg_names = {kw.arg for kw in call.keywords}
            assert "disabled_toolsets" in kwarg_names, (
                "AIAgent call in oneshot no longer forwards disabled_toolsets "
                "(#61184 regression)"
            )
