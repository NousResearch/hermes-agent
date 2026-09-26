"""Auth classification and SDK availability — claude-agent-sdk runtime tests (#25267).

Split from ``tests/agent/test_claude_sdk_runtime.py``; the SDK message
stand-ins, fake clients and shared builders live in
``tests.agent.claude_sdk_fakes``.
"""


import pytest

from agent.transports.claude_agent_sdk_session import ClaudeAgentSdkSession
from agent.transports.claude_agent_sdk_session_availability import classify_auth_failure
from tests.agent.claude_sdk_fakes import (
    ResultMessage,
    _make_session,
    isolate_provider_config,
)


@pytest.fixture(autouse=True)
def _isolate_provider_config(monkeypatch):
    """Provider config, gateway contextvars and the CLI approval callback are
    reset around every test in this module — carried explicitly, never hoisted
    to a conftest (see ``isolate_provider_config``)."""
    yield from isolate_provider_config(monkeypatch)


# ---------- auth classifier (with negative control) ----------


class TestAuthClassifier:
    def test_auth_failure_produces_hint(self):
        hint = classify_auth_failure("HTTP 401 unauthorized: oauth token expired")
        assert hint is not None
        assert "setup-token" in hint

    def test_hint_redacts_underlying_error(self):
        hint = classify_auth_failure(
            "HTTP 401 unauthorized: oauth token expired; Authorization: Bearer sk-secret-test-token"
        )
        assert hint is not None
        assert "401 unauthorized" in hint
        assert "sk-secret-test-token" not in hint

    def test_negative_control_ordinary_error_no_hint(self):
        # RED-first: an unrelated failure must surface verbatim, never as a
        # re-auth redirect.
        assert classify_auth_failure("connection reset by peer") is None
        assert classify_auth_failure("") is None

    def test_negative_control_overbroad_substrings(self):
        # RED-first against the original hint list: codex's
        # _OAUTH_REFRESH_FAILURE_HINTS has "401 unauthorized", never bare
        # "401", and no bare "credentials" — a tool id or an MCP server's
        # own file complaint must not retire the session as an auth failure.
        assert classify_auth_failure("tool_use toolu_401abc failed at 4012") is None
        assert (
            classify_auth_failure(
                "mcp server hermes-tools: could not read credentials file"
            )
            is None
        )
        assert (
            classify_auth_failure(
                "mcp server weather: Unauthorized — invalid region scope"
            )
            is None
        )

    @pytest.mark.parametrize(
        "message",
        [
            "HTTP 401: Unauthorized",
            "request failed: status code 401, Unauthorized",
            "request failed: status code 401 - Unauthorized",
            "request failed: status code 401 (Unauthorized)",
            "request failed: status code 401\nUnauthorized",
            "SDK result error: Unauthorized (HTTP 401)",
        ],
    )
    def test_delimited_401_is_auth_failure(self, message):
        assert classify_auth_failure(message) is not None

    @pytest.mark.parametrize(
        "message",
        [
            "mcp__calendar failed: HTTP 401 Unauthorized",
            'mcp__calendar failed: {"type":"authentication_error"}',
            "mcp server failed: invalid api key",
            "mcp__tool failed; Anthropic OAuth error: HTTP 401 Unauthorized",
        ],
    )
    def test_mcp_auth_payload_is_not_claude_auth_failure(self, message):
        assert classify_auth_failure(message) is None

    def test_structured_mcp_provenance_suppresses_unlabeled_auth_payload(self):
        assert (
            classify_auth_failure(
                "HTTP 401 Unauthorized: authentication_error",
                mcp_attributed=True,
            )
            is None
        )


class TestSdkAvailabilityGate:
    @staticmethod
    def _hide_sdk(monkeypatch, tmp_path):
        # A lean install, crossing real PM admission (pm.extras.ensure_import):
        # the SDK is absent from sys.modules and unfindable, and no runtime
        # facts exist, so a successful sync needs no restart to activate.
        import importlib.abc
        import sys as _sys

        from pm import paths

        class MissingSDK(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "claude_agent_sdk":
                    raise ModuleNotFoundError("No module named 'claude_agent_sdk'")

        monkeypatch.delitem(_sys.modules, "claude_agent_sdk", raising=False)
        monkeypatch.setattr(_sys, "meta_path", [MissingSDK(), *_sys.meta_path])
        monkeypatch.setattr(paths, "runtime_facts_path", lambda: tmp_path / "unselected-facts")

    def test_check_routes_through_pm_lazy_install(self, monkeypatch, tmp_path):
        # F1 (deps): the SDK is a PM opt-in extra outside [all], so the
        # availability gate must offer PM's lazy-install lane first — the
        # exact pattern anthropic_adapter._get_anthropic_sdk uses for the
        # anthropic extra. A lean install otherwise dead-ends on
        # ImportError with no self-serve path. Only the venv sync is faked.
        import importlib.machinery
        import sys as _sys
        import types as _types

        import pm.client
        from agent.transports.claude_agent_sdk_session_availability import (
            check_claude_sdk_available,
        )

        self._hide_sdk(monkeypatch, tmp_path)
        sdk = _types.ModuleType("claude_agent_sdk")
        sdk.__spec__ = importlib.machinery.ModuleSpec("claude_agent_sdk", loader=None)
        synced = []

        def sync(extras):
            synced.append(extras)
            monkeypatch.setitem(_sys.modules, "claude_agent_sdk", sdk)

        monkeypatch.setattr(pm.client, "sync_venv", sync)
        assert check_claude_sdk_available() == (True, "ok")
        assert synced == [["claude-agent-sdk"]]

    def test_check_skips_lazy_lane_when_sdk_already_imports(self, monkeypatch):
        # An install rewrites the dependency environment. Running one
        # immediately before `import claude_agent_sdk -> mcp -> anyio` under a
        # live interpreter intermittently corrupted that very import
        # ("KeyError: 'anyio'" out of importlib._bootstrap._find_and_load).
        # When the extra is ALREADY importable PM must not be entered at all.
        import sys as _sys
        import types as _types

        from agent.transports.claude_agent_sdk_session_availability import (
            check_claude_sdk_available,
        )

        called = []
        monkeypatch.setattr("pm.ensure_import", lambda extra: called.append(extra))
        monkeypatch.setitem(
            _sys.modules, "claude_agent_sdk", _types.ModuleType("claude_agent_sdk")
        )
        assert check_claude_sdk_available() == (True, "ok")
        assert called == []

    def test_pm_extra_is_declared_opt_in_and_anchored(self):
        # The lane names the `claude-agent-sdk` pyproject extra. It keeps a
        # single exact SDK pin, stays a PM opt-in extra (each wheel bundles the
        # Claude Code CLI, so --all-extras bundles must never carry it), and PM
        # proves it installed by importing the SDK itself.
        import tomllib
        from pathlib import Path

        from packaging.requirements import Requirement

        from pm.extras import _anchors
        from pm.features import opt_in_extras

        root = Path(__file__).resolve().parents[2]
        extra = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))[
            "project"]["optional-dependencies"]["claude-agent-sdk"]
        sdk = [Requirement(spec) for spec in extra if Requirement(spec).name == "claude-agent-sdk"]
        assert len(sdk) == 1 and [s.operator for s in sdk[0].specifier] == ["=="]
        assert "claude-agent-sdk" in opt_in_extras(root)
        assert _anchors("claude-agent-sdk") == ("claude_agent_sdk",)

    def test_check_reports_missing_sdk(self, monkeypatch, tmp_path):
        # RED-first negative control: with the SDK absent and PM refusing the
        # install (lazy installs disabled / offline), the gate must fail with
        # PM's install hint and its refusal reason — never silently pass, and
        # never trigger a real ~100 MB SDK download on CI.
        import pm.client
        from pm import install_hint
        from pm.package import InstallError
        from agent.transports.claude_agent_sdk_session_availability import (
            check_claude_sdk_available,
        )

        self._hide_sdk(monkeypatch, tmp_path)

        def _refuse(extras):
            raise InstallError("venv", "lazy installs are disabled in this test")

        monkeypatch.setattr(pm.client, "sync_venv", _refuse)
        ok, msg = check_claude_sdk_available()
        assert ok is False
        assert install_hint("claude-agent-sdk") in msg
        assert "lazy installs are disabled in this test" in msg


class TestAnthropicTokenGuard:
    """F1 follow-up (#65982 independent verification): ANTHROPIC_TOKEN alone
    authenticates hermes' native metered lane, so an API-key-shaped value is
    the same fail-closed class as ANTHROPIC_API_KEY — while an OAuth-shaped
    value is the subscription lane itself and must keep working."""

    def test_anthropic_token_api_key_shaped_refuses_startup(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-api03-fake")
        session = ClaudeAgentSdkSession(cwd="/tmp")  # no factory → real path
        turn = session.run_turn("hi")
        assert turn.should_retire
        assert "ANTHROPIC_TOKEN" in (turn.error or "")

    def test_anthropic_token_oauth_shaped_starts_normally(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-oat01-fake")
        session, _holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            turn = session.run_turn("ping")
        finally:
            session.close()
        assert turn.error is None

    def test_allow_metered_key_admits_api_key_shaped_token(self, monkeypatch):
        import hermes_cli.config as cfg

        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
        monkeypatch.setenv("ANTHROPIC_TOKEN", "sk-ant-api03-fake")
        monkeypatch.setattr(
            cfg,
            "load_config_readonly",
            lambda *a, **k: {
                "agent": {"claude_agent_sdk": {"allow_metered_key": True}}
            },
            raising=False,
        )
        session, _holder = _make_session(script=[ResultMessage(result="ok")])
        try:
            turn = session.run_turn("ping")
        finally:
            session.close()
        assert turn.error is None
