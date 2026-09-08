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
    def test_check_routes_through_lazy_install_lane(self, monkeypatch):
        # F1 (deps): the SDK is an opt-in extra excluded from [all], so the
        # availability gate must offer the lazy-install lane first — the
        # exact pattern anthropic_adapter._get_anthropic_sdk uses for
        # provider.anthropic. A lean install otherwise dead-ends on
        # ImportError with no self-serve path.
        import tools.lazy_deps as lazy_deps
        from agent.transports.claude_agent_sdk_session_availability import (
            check_claude_sdk_available,
        )

        assert "provider.claude_agent_sdk" in lazy_deps.LAZY_DEPS
        called = {}

        def fake_ensure(feature, *, prompt=True):
            called["feature"] = feature
            called["prompt"] = prompt

        monkeypatch.setattr(lazy_deps, "ensure", fake_ensure)
        # Pin the LEAN install this lane exists for: a None entry in
        # sys.modules makes `import claude_agent_sdk` raise ImportError.
        import sys as _sys

        monkeypatch.setitem(_sys.modules, "claude_agent_sdk", None)
        check_claude_sdk_available()
        assert called == {"feature": "provider.claude_agent_sdk", "prompt": False}

    def test_check_skips_lazy_lane_when_sdk_already_imports(self, monkeypatch):
        # ensure() can shell out to `uv pip install` and calls
        # importlib.invalidate_caches(). Running it immediately before
        # `import claude_agent_sdk -> mcp -> anyio` rewrites site-packages and
        # drops import caches under a live interpreter, intermittently
        # corrupting that very import ("KeyError: 'anyio'" out of
        # importlib._bootstrap._find_and_load). When the extra is ALREADY
        # importable the installer must not run at all.
        import sys as _sys
        import types as _types

        import tools.lazy_deps as lazy_deps
        from agent.transports.claude_agent_sdk_session_availability import (
            check_claude_sdk_available,
        )

        called = {}

        def fake_ensure(feature, *, prompt=True):
            called["feature"] = feature

        monkeypatch.setattr(lazy_deps, "ensure", fake_ensure)
        monkeypatch.setitem(
            _sys.modules, "claude_agent_sdk", _types.ModuleType("claude_agent_sdk")
        )
        assert check_claude_sdk_available() == (True, "ok")
        assert called == {}

    def test_lazy_lane_pin_matches_pyproject_extra(self):
        # The LAZY_DEPS lane must mirror the pyproject extra in lockstep
        # (same contract test_pyproject_and_lazy_deps_pins_agree enforces
        # globally; pinned here so the SDK lane keeps a single exact spec).
        from tools.lazy_deps import LAZY_DEPS

        specs = LAZY_DEPS["provider.claude_agent_sdk"]
        assert len(specs) == 1
        assert specs[0].startswith("claude-agent-sdk==")

    def test_check_reports_missing_sdk(self, monkeypatch):
        # RED-first negative control: with the import broken, the gate must
        # fail with the install hint — never silently pass. The lazy lane is
        # stubbed to FeatureUnavailable (lazy installs disabled / offline) so
        # the test never triggers a real multi-MB SDK download on CI.
        import builtins

        import tools.lazy_deps as lazy_deps

        def _unavailable(feature, *, prompt=True):
            raise lazy_deps.FeatureUnavailable(feature, (), "disabled in test")

        monkeypatch.setattr(lazy_deps, "ensure", _unavailable)

        real_import = builtins.__import__

        def _broken(name, *args, **kwargs):
            if name == "claude_agent_sdk":
                raise ImportError("No module named 'claude_agent_sdk'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _broken)
        from agent.transports.claude_agent_sdk_session_availability import (
            check_claude_sdk_available,
        )

        ok, msg = check_claude_sdk_available()
        assert ok is False
        assert "hermes-agent[claude-agent-sdk]" in msg


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
