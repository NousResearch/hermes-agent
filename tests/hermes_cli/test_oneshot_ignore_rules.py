"""Regression tests for ``hermes -z --ignore-rules``.

The ``--ignore-rules`` flag is documented as: "Skip auto-injection of
AGENTS.md, SOUL.md, .cursorrules, memory, and preloaded skills". It works
on the chat path (covered by ``test_ignore_user_config_flags.py``) but was
a silent no-op on the oneshot path before this fix because
``hermes_cli/oneshot.run_oneshot()`` did not read the flag or the
``HERMES_IGNORE_RULES`` env var, and did not pass ``skip_context_files`` /
``skip_memory`` to ``AIAgent``.

See #26633 for the full root-cause writeup. These tests cover the wiring
end-to-end: parser → main.run_oneshot → _run_agent → AIAgent kwargs.
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("HERMES_IGNORE_RULES", raising=False)
    yield
    os.environ.pop("HERMES_IGNORE_RULES", None)


class TestRunAgentForwardsIgnoreRules:
    """``_run_agent`` must translate ``ignore_rules`` into the two AIAgent
    kwargs that actually suppress AGENTS.md/SOUL.md/.cursorrules + memory.
    """

    def _build_agent_mock(self):
        agent_instance = MagicMock()
        agent_instance.chat.return_value = "ok"
        return agent_instance

    def _patch_dependencies(self, monkeypatch, agent_instance):
        from hermes_cli import oneshot

        fake_agent_cls = MagicMock(return_value=agent_instance)

        # Stub the heavy resolver dependencies so we can drive _run_agent
        # without needing config files / network / credentials.
        monkeypatch.setattr(
            "hermes_cli.config.load_config", lambda: {"model": {"default": "x"}}
        )
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **_: {
                "api_key": "k",
                "base_url": "https://example",
                "provider": "openrouter",
                "api_mode": "chat_completions",
                "credential_pool": None,
            },
        )
        monkeypatch.setattr(
            "hermes_cli.tools_config._get_platform_tools", lambda cfg, name: set()
        )
        monkeypatch.setattr("run_agent.AIAgent", fake_agent_cls)
        # Avoid touching ~/.hermes/session.sqlite during tests.
        monkeypatch.setattr(
            oneshot, "_create_session_db_for_oneshot", lambda: None
        )
        return fake_agent_cls

    def test_ignore_rules_true_sets_both_skip_kwargs(self, monkeypatch):
        from hermes_cli.oneshot import _run_agent

        agent_instance = self._build_agent_mock()
        fake_agent_cls = self._patch_dependencies(monkeypatch, agent_instance)

        _run_agent("hi", model="x", provider="openrouter", ignore_rules=True)

        kwargs = fake_agent_cls.call_args.kwargs
        assert kwargs.get("skip_context_files") is True
        assert kwargs.get("skip_memory") is True

    def test_ignore_rules_false_leaves_both_skip_kwargs_false(self, monkeypatch):
        from hermes_cli.oneshot import _run_agent

        agent_instance = self._build_agent_mock()
        fake_agent_cls = self._patch_dependencies(monkeypatch, agent_instance)

        _run_agent("hi", model="x", provider="openrouter", ignore_rules=False)

        kwargs = fake_agent_cls.call_args.kwargs
        assert kwargs.get("skip_context_files") is False
        assert kwargs.get("skip_memory") is False

    def test_default_is_false(self, monkeypatch):
        """Omitting the kwarg must match the pre-fix behavior (rules ON)."""
        from hermes_cli.oneshot import _run_agent

        agent_instance = self._build_agent_mock()
        fake_agent_cls = self._patch_dependencies(monkeypatch, agent_instance)

        _run_agent("hi", model="x", provider="openrouter")

        kwargs = fake_agent_cls.call_args.kwargs
        assert kwargs.get("skip_context_files") is False
        assert kwargs.get("skip_memory") is False


class TestRunOneshotForwardsToRunAgent:
    """``run_oneshot`` must propagate its ``ignore_rules`` argument and the
    ``HERMES_IGNORE_RULES`` env-var fallback through to ``_run_agent``.
    """

    def _captured_run_agent(self):
        captured: dict = {}

        def fake(*args, **kwargs):
            captured.update(kwargs)
            captured.setdefault("_args", args)
            return "ok"

        return captured, fake

    def test_explicit_param_propagates(self, monkeypatch):
        from hermes_cli import oneshot

        captured, fake = self._captured_run_agent()
        monkeypatch.setattr(oneshot, "_run_agent", fake)

        rc = oneshot.run_oneshot("hello", ignore_rules=True)
        assert rc == 0
        assert captured.get("ignore_rules") is True

    def test_env_var_propagates_when_param_false(self, monkeypatch):
        from hermes_cli import oneshot

        captured, fake = self._captured_run_agent()
        monkeypatch.setattr(oneshot, "_run_agent", fake)
        monkeypatch.setenv("HERMES_IGNORE_RULES", "1")

        rc = oneshot.run_oneshot("hello")
        assert rc == 0
        assert captured.get("ignore_rules") is True

    def test_neither_param_nor_env_means_false(self, monkeypatch):
        from hermes_cli import oneshot

        captured, fake = self._captured_run_agent()
        monkeypatch.setattr(oneshot, "_run_agent", fake)
        monkeypatch.delenv("HERMES_IGNORE_RULES", raising=False)

        rc = oneshot.run_oneshot("hello")
        assert rc == 0
        assert captured.get("ignore_rules") is False

    def test_env_var_other_value_is_falsy(self, monkeypatch):
        """Only the literal "1" activates the gate (same as the chat path)."""
        from hermes_cli import oneshot

        captured, fake = self._captured_run_agent()
        monkeypatch.setattr(oneshot, "_run_agent", fake)
        monkeypatch.setenv("HERMES_IGNORE_RULES", "true")

        rc = oneshot.run_oneshot("hello")
        assert rc == 0
        assert captured.get("ignore_rules") is False


class TestMainDispatchForwardsIgnoreRules:
    """The top-level ``--oneshot`` / ``-z`` dispatch in ``hermes_cli/main.py``
    must pass ``args.ignore_rules`` to ``run_oneshot``. Without this, the
    explicit param wiring above never gets exercised on the real CLI path.
    """

    def test_main_oneshot_path_forwards_ignore_rules(self):
        import inspect
        import hermes_cli.main as hm

        src = inspect.getsource(hm)
        assert "run_oneshot(" in src, "main.py must still dispatch to run_oneshot"
        # main.py has more than one oneshot dispatch site (fast-path vs full
        # parser path). Every call site must forward args.ignore_rules; if
        # even one site drops it, --ignore-rules is a silent no-op on whichever
        # path the bare `hermes -z` invocation takes.
        sites = []
        cursor = 0
        while True:
            idx = src.find("run_oneshot(", cursor)
            if idx == -1:
                break
            sites.append(src[idx : idx + 400])
            cursor = idx + 1
        assert sites, "expected at least one run_oneshot(...) dispatch in main.py"
        missing = [i for i, w in enumerate(sites) if "ignore_rules=" not in w]
        assert not missing, (
            f"main.py oneshot dispatch site(s) {missing} (of {len(sites)} total) "
            "must forward args.ignore_rules to run_oneshot; without this "
            "--ignore-rules is a silent no-op on -z (#26633)."
        )


class TestRegressionGuard:
    """Old behavior: before this fix, ``run_oneshot`` did not accept
    ``ignore_rules`` and ``_run_agent`` never set ``skip_context_files`` /
    ``skip_memory``. If anyone reverts the oneshot wiring, this guard fails.
    """

    def test_run_oneshot_signature_has_ignore_rules(self):
        import inspect
        from hermes_cli.oneshot import run_oneshot

        sig = inspect.signature(run_oneshot)
        assert "ignore_rules" in sig.parameters, (
            "run_oneshot must accept ignore_rules; without it the -z path "
            "cannot honor --ignore-rules (#26633)."
        )

    def test_run_agent_passes_both_skip_kwargs(self, monkeypatch):
        """If a future refactor drops either kwarg, both flag effects break."""
        from hermes_cli.oneshot import _run_agent

        agent_instance = MagicMock()
        agent_instance.chat.return_value = "ok"
        fake_cls = MagicMock(return_value=agent_instance)

        monkeypatch.setattr(
            "hermes_cli.config.load_config", lambda: {"model": {"default": "x"}}
        )
        monkeypatch.setattr(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            lambda **_: {"api_key": "k", "base_url": "u", "provider": "p", "api_mode": "chat_completions", "credential_pool": None},
        )
        monkeypatch.setattr(
            "hermes_cli.tools_config._get_platform_tools", lambda cfg, name: set()
        )
        monkeypatch.setattr("run_agent.AIAgent", fake_cls)

        from hermes_cli import oneshot
        monkeypatch.setattr(oneshot, "_create_session_db_for_oneshot", lambda: None)

        _run_agent("hi", model="x", provider="p", ignore_rules=True)

        kwargs = fake_cls.call_args.kwargs
        # Both kwargs must be wired — one without the other still leaks
        # either rules (AGENTS.md / SOUL.md / .cursorrules) or memory.
        assert "skip_context_files" in kwargs
        assert "skip_memory" in kwargs
        assert kwargs["skip_context_files"] is True
        assert kwargs["skip_memory"] is True
