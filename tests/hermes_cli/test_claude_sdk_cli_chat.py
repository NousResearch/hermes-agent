"""CLI construction-path coverage for the claude-agent-sdk provider (#65982).

Clean-checkout E2E findings (jefftropeano): the documented
``hermes chat -Q --provider claude-agent-sdk -q "…"`` invocation died inside
``_ensure_runtime_credentials`` ("Provider resolver returned an empty base
URL.") before the SDK runtime was ever constructed — and a fatal
metered-billing refusal exited 0, so integrations recorded success. The
targeted suite passed without exercising the real ``hermes chat``
construction path; these tests run it in-process, end to end.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from hermes_cli.cli_agent_setup_mixin import CLIAgentSetupMixin


class _SdkHost(CLIAgentSetupMixin):
    """Minimal mixin host, following tests/hermes_cli/test_cli_custom_provider_vision.py."""

    def __init__(self):
        self.model = "claude-opus-4-8"
        self.requested_provider = "claude-agent-sdk"
        self.provider = "claude-agent-sdk"
        self.api_key = None
        self.base_url = None
        self.api_mode = "chat_completions"
        self.acp_command = None
        self.acp_args = []
        self.agent = None
        self._fallback_model = []
        self._explicit_api_key = None
        self._explicit_base_url = None
        self._credential_pool = None
        self.service_tier = None

    def _normalize_model_for_provider(self, _provider: str) -> bool:
        return False


def test_ensure_runtime_credentials_accepts_claude_agent_sdk():
    """The resolver's by-design empty base_url must not fail generic HTTP
    validation: this provider's runtime (the official Agent SDK) owns its own
    credentials and transport — there is no HTTP base URL on this path."""
    host = _SdkHost()
    assert host._ensure_runtime_credentials() is True
    assert host.api_mode == "claude_agent_sdk"
    assert host.provider == "claude-agent-sdk"
    assert host.base_url == ""
    assert host.api_key == "claude-subscription-oauth"


def _sdk_turn_success(agent, *, user_message, original_user_message, messages,
                      effective_task_id, should_review_memory=False):
    messages.append({"role": "assistant", "content": "OK"})
    return {
        "final_response": "OK",
        "messages": messages,
        "api_calls": 1,
        "completed": True,
        "partial": False,
        "error": None,
        "agent_persisted": True,
    }


def _stub_single_query_harness(monkeypatch, cli_mod):
    monkeypatch.setattr(cli_mod.atexit, "register", lambda *_a, **_k: None)
    monkeypatch.setattr(cli_mod, "_finalize_single_query", lambda _cli: None)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_GOAL_MODE", raising=False)


def test_quiet_single_query_reaches_sdk_runtime_and_exits_zero(monkeypatch, capsys):
    """The exact documented invocation shape: `hermes chat -Q --provider
    claude-agent-sdk -q "Reply with exactly: OK"`. Construction must get past
    credential setup, build a real agent routed at api_mode
    "claude_agent_sdk", reach the SDK runtime layer (faked here), and exit 0."""
    import cli as cli_mod

    seen = []

    def fake_turn(agent, **kwargs):
        seen.append((kwargs["user_message"], getattr(agent, "api_mode", None)))
        return _sdk_turn_success(agent, **kwargs)

    monkeypatch.setattr(
        "agent.claude_sdk_runtime.run_claude_agent_sdk_turn", fake_turn
    )
    _stub_single_query_harness(monkeypatch, cli_mod)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(
            query="Reply with exactly: OK",
            quiet=True,
            provider="claude-agent-sdk",
            model="claude-opus-4-8",
            toolsets="terminal",
        )

    assert exc_info.value.code == 0
    assert seen and seen[0][0] == "Reply with exactly: OK"
    assert seen[0][1] == "claude_agent_sdk"
    assert "OK" in capsys.readouterr().out


def test_quiet_single_query_metered_refusal_exits_nonzero(monkeypatch, capsys):
    """Fatal metered-billing refusal must not exit 0. Full path, nothing
    faked below the CLI: real credentials setup, real agent, real SDK
    runtime — the fail-closed guard fires before any subprocess spawn."""
    import cli as cli_mod

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fake")
    _stub_single_query_harness(monkeypatch, cli_mod)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(
            query="hi",
            quiet=True,
            provider="claude-agent-sdk",
            model="claude-opus-4-8",
            toolsets="terminal",
        )

    assert exc_info.value.code == 1
    # Pin the CAUSE, not just the code: pre-fix this invocation also exited 1,
    # but from the empty-base-URL death in credential setup — the refusal
    # guard's message proves the run got past construction to the runtime.
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err


def test_human_single_query_failed_turn_exits_nonzero(monkeypatch):
    """The human one-shot path (`-q` without `-Q`) mirrors the quiet path's
    exit contract: a turn that reports failure must not exit 0."""
    import cli as cli_mod

    class FakeCLI:
        def __init__(self, **_kwargs):
            self.console = SimpleNamespace(print=lambda *_a, **_k: None)
            self.session_id = "human-session"
            self.agent = SimpleNamespace(
                session_id="human-session", platform="cli"
            )

        def _claim_active_session(self, _surface, *, stderr=False):
            return True

        def _show_security_advisories(self):
            pass

        def chat(self, _query, images=None):
            # What the real chat() records for a fatal SDK refusal.
            self._last_turn_failed = True
            self._last_turn_failure_reason = "startup"
            return None

        def _print_exit_summary(self, clear_screen=True):
            pass

    monkeypatch.setattr(cli_mod, "HermesCLI", FakeCLI)
    _stub_single_query_harness(monkeypatch, cli_mod)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hi", quiet=False, toolsets="terminal")

    assert exc_info.value.code == 1


def test_human_single_query_metered_refusal_exits_nonzero(monkeypatch, capsys):
    """Same refusal as the quiet test, through the real human path: real
    chat(), real agent, real runtime — the failure must surface as a nonzero
    exit, not a polite panel followed by exit 0."""
    import cli as cli_mod

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-api03-fake")
    _stub_single_query_harness(monkeypatch, cli_mod)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(
            query="hi",
            quiet=False,
            provider="claude-agent-sdk",
            model="claude-opus-4-8",
            toolsets="terminal",
        )

    assert exc_info.value.code == 1
    # Cause-pinning (see the quiet twin): the refusal guard's message proves
    # the exit came from the runtime refusal, not a construction death.
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().out


def test_human_single_query_credentials_failure_exits_nonzero(monkeypatch):
    """Real chat() early-return coverage: a provider that cannot resolve
    makes _ensure_runtime_credentials return False; chat() records the
    failure and the one-shot branch must exit 1 (the -Q twin exits 1 via its
    credentials/init sys.exit — this pins the same contract for -q)."""
    import cli as cli_mod

    _stub_single_query_harness(monkeypatch, cli_mod)

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(
            query="hi",
            quiet=False,
            provider="no-such-provider-zzz",
            model="whatever",
            toolsets="terminal",
        )

    assert exc_info.value.code == 1


def _kanban_fake_cli(failure_reason):
    class FakeCLI:
        def __init__(self, **_kwargs):
            self.console = SimpleNamespace(print=lambda *_a, **_k: None)
            self.session_id = "kanban-session"
            self.agent = SimpleNamespace(
                session_id="kanban-session", platform="cli"
            )

        def _claim_active_session(self, _surface, *, stderr=False):
            return True

        def _show_security_advisories(self):
            pass

        def chat(self, _query, images=None):
            self._last_turn_failed = True
            self._last_turn_failure_reason = failure_reason
            return None

        def _print_exit_summary(self, clear_screen=True):
            pass

    return FakeCLI


@pytest.mark.parametrize(
    "failure_reason,kanban,expected",
    [
        # Quota walls on kanban workers exit EX_TEMPFAIL (75) so the reap
        # classifies rate_limited and requeues WITHOUT ticking the circuit
        # breaker — non-goal-mode workers run `chat -q` without -Q, so THIS
        # hook is their exit contract (mirrors the -Q branch's mapping).
        ("rate_limit", True, 75),
        ("billing", True, 75),
        # A config-shaped failure (e.g. the metered-key refusal's "startup")
        # is a real failure everywhere — plain 1 even under kanban.
        ("startup", True, 1),
        # Outside kanban, quota walls are plain failures for shell scripts.
        ("rate_limit", False, 1),
    ],
)
def test_human_single_query_kanban_exit_mapping(
    monkeypatch, failure_reason, kanban, expected
):
    import cli as cli_mod

    monkeypatch.setattr(cli_mod, "HermesCLI", _kanban_fake_cli(failure_reason))
    _stub_single_query_harness(monkeypatch, cli_mod)
    if kanban:
        monkeypatch.setenv("HERMES_KANBAN_TASK", "task-1")

    with pytest.raises(SystemExit) as exc_info:
        cli_mod.main(query="hi", quiet=False, toolsets="terminal")

    assert exc_info.value.code == expected


def test_chat_orchestration_exception_records_failure(monkeypatch):
    """chat()'s catch-all (post-thread orchestration dying, NOT
    run_conversation — those become failed result dicts inside run_agent)
    must record the failure so one-shot callers exit nonzero instead of
    printing "Error: ..." and exiting 0."""
    import cli as cli_mod

    monkeypatch.setattr(
        "agent.claude_sdk_runtime.run_claude_agent_sdk_turn",
        _sdk_turn_success,
    )
    shell = cli_mod.HermesCLI(
        model="claude-opus-4-8",
        provider="claude-agent-sdk",
        compact=True,
        max_turns=1,
    )
    monkeypatch.setattr(
        shell,
        "_flush_stream",
        lambda: (_ for _ in ()).throw(RuntimeError("display pipeline died")),
    )

    assert shell.chat("hi") is None
    assert shell._last_turn_failed is True
    assert shell._last_turn_failure_reason == "exception"
