"""Oneshot (`hermes -z`) per-turn iteration budget resolution.

Regression coverage for the budget-decapitation bug: oneshot used to build
its AIAgent without a max_iterations argument, so `hermes -z` silently ran
at the AIAgent default and ignored `agent.max_turns` in config / the
`--max-turns` flag. Long-running workers (e.g. MeshBoard dispatches) were cut
off mid-task with no way to raise the cap per invocation.
"""

import pytest

from hermes_cli._parser import build_top_level_parser
from hermes_cli.oneshot import _resolve_max_iterations, run_oneshot


class TestOneshotMaxTurnsParsing:
    def test_max_turns_pairs_with_oneshot(self):
        parser, _subparsers, _chat_parser = build_top_level_parser()
        args = parser.parse_args(["-z", "hi", "--max-turns", "5"])
        assert args.max_turns == 5

    def test_absent_max_turns_is_none(self):
        parser, _subparsers, _chat_parser = build_top_level_parser()
        args = parser.parse_args(["-z", "hi"])
        assert args.max_turns is None

    def test_top_level_value_survives_chat_subparser(self):
        parser, _subparsers, _chat_parser = build_top_level_parser()
        args = parser.parse_args(["--max-turns", "7", "chat"])
        assert args.max_turns == 7

    def test_chat_subparser_value_still_parses(self):
        parser, _subparsers, _chat_parser = build_top_level_parser()
        args = parser.parse_args(["chat", "--max-turns", "9"])
        assert args.max_turns == 9


class TestResolveMaxIterations:
    def test_cli_flag_wins(self):
        cfg = {"agent": {"max_turns": 120}}
        assert _resolve_max_iterations(cfg, 250) == 250

    def test_config_agent_max_turns_used_when_no_flag(self):
        cfg = {"agent": {"max_turns": 120}}
        assert _resolve_max_iterations(cfg, None) == 120

    def test_builtin_default_when_loaded_value_is_missing(self):
        assert _resolve_max_iterations({}, None) == 90

    def test_non_numeric_loaded_value_uses_default(self):
        cfg = {"agent": {"max_turns": "not-a-number"}}
        assert _resolve_max_iterations(cfg, None) == 90

    def test_zero_loaded_value_uses_default(self):
        assert _resolve_max_iterations({"agent": {"max_turns": 0}}, None) == 90


class TestRunOneshotThreadsMaxTurns:
    def test_max_turns_reaches_run_agent(self, monkeypatch):
        captured = {}

        def _fake_run_agent(prompt, **kwargs):
            captured.update(kwargs)
            return ("ok", {})

        import hermes_cli.oneshot as oneshot_mod

        monkeypatch.setattr(oneshot_mod, "_run_agent", _fake_run_agent)
        assert run_oneshot("hi", max_turns=250) == 0
        assert captured.get("max_turns") == 250

    def test_default_is_none_passthrough(self, monkeypatch):
        captured = {}

        def _fake_run_agent(prompt, **kwargs):
            captured.update(kwargs)
            return ("ok", {})

        import hermes_cli.oneshot as oneshot_mod

        monkeypatch.setattr(oneshot_mod, "_run_agent", _fake_run_agent)
        assert run_oneshot("hi") == 0
        assert captured.get("max_turns") is None


class TestRunAgentPassesMaxIterationsToAgent:
    """End-to-end through the real ``_run_agent`` resolver: only the AIAgent
    constructor and the network-touching provider resolver are stubbed, so
    the actual config-precedence code runs and we assert the resolved cap
    lands on ``AIAgent(max_iterations=...)`` — the crux line."""

    def _run(self, monkeypatch, tmp_path, *, cli_max_turns, config_yaml):
        import hermes_cli.oneshot as oneshot_mod
        import hermes_cli.config as config_mod
        import hermes_cli.runtime_provider as rp_mod
        import hermes_cli.tools_config as tc_mod
        import run_agent as ra_mod

        captured = {}

        class _FakeAgent:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            suppress_status_output = False
            stream_delta_callback = None
            tool_gen_callback = None

            def run_conversation(self, _prompt):
                return {"final_response": "ok"}

        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        monkeypatch.delenv("HERMES_MANAGED_DIR", raising=False)
        if config_yaml:
            (tmp_path / "config.yaml").write_text(config_yaml, encoding="utf-8")
        config_mod._LOAD_CONFIG_CACHE.clear()
        config_mod._RAW_CONFIG_CACHE.clear()
        monkeypatch.setattr(
            rp_mod,
            "resolve_runtime_provider",
            lambda **kwargs: {
                "api_key": "k",
                "base_url": "http://localhost:1/v1",
                "provider": "test",
                "api_mode": "chat",
                "credential_pool": None,
            },
        )
        monkeypatch.setattr(tc_mod, "_get_platform_tools", lambda *a, **k: set())
        monkeypatch.setattr(oneshot_mod, "get_fallback_chain", lambda *a, **k: [])
        monkeypatch.setattr(
            oneshot_mod, "_create_session_db_for_oneshot", lambda: None
        )
        monkeypatch.setattr(ra_mod, "AIAgent", _FakeAgent)

        oneshot_mod._run_agent("hi", max_turns=cli_max_turns)
        return captured

    def test_cli_max_turns_lands_on_agent(self, monkeypatch, tmp_path):
        captured = self._run(
            monkeypatch,
            tmp_path,
            cli_max_turns=250,
            config_yaml="agent:\n  max_turns: 120\n",
        )
        assert captured["max_iterations"] == 250

    def test_config_max_turns_lands_on_agent_when_no_flag(
        self, monkeypatch, tmp_path
    ):
        captured = self._run(
            monkeypatch,
            tmp_path,
            cli_max_turns=None,
            config_yaml="agent:\n  max_turns: 120\n",
        )
        assert captured["max_iterations"] == 120

    def test_legacy_root_config_is_normalized_before_resolution(
        self, monkeypatch, tmp_path
    ):
        captured = self._run(
            monkeypatch,
            tmp_path,
            cli_max_turns=None,
            config_yaml="max_turns: 55\n",
        )
        assert captured["max_iterations"] == 55


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
