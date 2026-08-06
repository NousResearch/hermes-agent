"""Contract for "this turn bills the user's Claude plan, or it does not run".

The Claude Code CLI resolves credentials in a fixed order and the user's
subscription is last, so an exported ``ANTHROPIC_API_KEY`` silently redirects
a "Claude subscription" turn onto a metered API account. Three guarantees are
pinned down here:

1. Every credential that outranks the subscription makes the runtime **refuse**,
   with a message that names that specific variable (never its value).
2. The environment the launcher builds for the CLI subprocess genuinely lacks
   those variables — while ``os.environ`` itself is never mutated, because
   Hermes is multi-threaded and other providers run concurrently.
3. The billing probe reads the CLI's own account report without ever issuing a
   model request.

No test here spawns ``claude`` or touches the network: the SDK client and its
transport are injected. The file runs with and without the optional
``claude-agent-sdk`` extra — everything that needs the real package is skipped
when it is absent, and the pure logic (precedence, sanitized env,
classification) is exercised either way.
"""

from __future__ import annotations

import os
import types
from dataclasses import dataclass, field
from importlib.util import find_spec
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agent import claude_billing
from agent.claude_billing import (
    BLOCKED_CHILD_ENV_VARS,
    CLAUDE_CREDENTIAL_PRECEDENCE,
    BillingSource,
    billing_source_refusal,
    blocking_credentials,
    classify_account,
    sanitized_child_env,
    static_billing_refusal,
    verify_claude_billing_source,
)
from agent.transports import claude_sanitized_transport

SDK_INSTALLED = find_spec("claude_agent_sdk") is not None
requires_sdk = pytest.mark.skipif(
    not SDK_INSTALLED, reason="claude-agent-sdk optional extra not installed"
)

# The env-var slots the CLI's documented precedence puts above the
# subscription. Derived from the module rather than restated, so adding a slot
# extends the coverage instead of silently leaving one untested.
ENV_CREDENTIAL_NAMES = [
    slot.name for slot in CLAUDE_CREDENTIAL_PRECEDENCE if slot.kind == "env"
]


@pytest.fixture(autouse=True)
def _clean_credential_env(monkeypatch):
    """No ambient credential decides the outcome of these tests."""
    for name in BLOCKED_CHILD_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


# ---------------------------------------------------------------------------
# Refusal: every higher-precedence credential
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", ENV_CREDENTIAL_NAMES)
def test_each_higher_precedence_credential_refuses_and_names_itself(name):
    refusal = static_billing_refusal({name: "some-value"})
    assert refusal is not None
    assert name in refusal
    # The user needs the action, not just the diagnosis.
    assert "unset" in refusal.lower()


@pytest.mark.parametrize("name", ENV_CREDENTIAL_NAMES)
def test_refusal_never_prints_the_credential_value(name):
    refusal = static_billing_refusal({name: "sk-ant-super-secret-value"})
    assert refusal is not None
    assert "sk-ant-super-secret-value" not in refusal


def test_the_documented_precedence_list_is_covered():
    """Every variable the decision record names has a slot."""
    documented = {
        "CLAUDE_CODE_USE_BEDROCK",
        "CLAUDE_CODE_USE_VERTEX",
        "CLAUDE_CODE_USE_FOUNDRY",
        "ANTHROPIC_AUTH_TOKEN",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_TOKEN",
        "CLAUDE_CODE_OAUTH" + "_TOKEN",
    }
    assert documented <= set(ENV_CREDENTIAL_NAMES)
    assert documented <= BLOCKED_CHILD_ENV_VARS


def test_several_conflicting_credentials_are_all_named():
    refusal = static_billing_refusal(
        {"ANTHROPIC_API_KEY": "a", "CLAUDE_CODE_USE_BEDROCK": "1"}
    )
    assert refusal is not None
    assert "ANTHROPIC_API_KEY" in refusal
    assert "CLAUDE_CODE_USE_BEDROCK" in refusal


def test_conflicts_are_reported_highest_precedence_first():
    slots = blocking_credentials(
        {"ANTHROPIC_API_KEY": "a", "CLAUDE_CODE_USE_VERTEX": "1"}
    )
    assert [slot.rank for slot in slots] == sorted(slot.rank for slot in slots)
    assert slots[0].name == "CLAUDE_CODE_USE_VERTEX"


def test_a_clean_environment_is_allowed():
    assert static_billing_refusal({"PATH": "/usr/bin", "HOME": "/home/x"}) is None


def test_a_blank_value_is_not_a_conflict():
    assert static_billing_refusal({"ANTHROPIC_API_KEY": "   "}) is None


def test_api_key_helper_is_a_setting_not_an_env_var():
    """`apiKeyHelper` lives in settings.json, so an env sweep cannot see it."""
    helper = next(
        slot for slot in CLAUDE_CREDENTIAL_PRECEDENCE if slot.name == "apiKeyHelper"
    )
    assert helper.kind == "setting"
    assert helper.name not in BLOCKED_CHILD_ENV_VARS
    assert blocking_credentials({"apiKeyHelper": "/bin/true"}) == []
    # It is still reachable when a caller can see the settings file.
    found = blocking_credentials({}, settings_keys={"apiKeyHelper": "/bin/true"})
    assert [slot.name for slot in found] == ["apiKeyHelper"]
    assert "apiKeyHelper" in claude_billing.credential_refusal_message(found)


# ---------------------------------------------------------------------------
# The sanitized child environment
# ---------------------------------------------------------------------------


def _dirty_env() -> dict:
    env = {name: f"value-for-{name}" for name in BLOCKED_CHILD_ENV_VARS}
    env.update(
        {
            "PATH": "/usr/bin:/bin",
            "HOME": "/home/tester",
            "CLAUDE_CONFIG_DIR": "/tmp/claude-config",
            "LANG": "C.UTF-8",
        }
    )
    return env


@pytest.mark.parametrize("name", sorted(BLOCKED_CHILD_ENV_VARS))
def test_the_launcher_env_lacks_every_blocked_key(name):
    child = sanitized_child_env(_dirty_env())
    assert name not in child


def test_the_launcher_env_keeps_everything_else():
    base = _dirty_env()
    child = sanitized_child_env(base)
    for name in ("PATH", "HOME", "LANG"):
        assert child[name] == base[name]


def test_claude_config_dir_passes_through():
    child = sanitized_child_env(_dirty_env())
    assert child["CLAUDE_CONFIG_DIR"] == "/tmp/claude-config"


def test_nothing_is_both_blocked_and_passed_through():
    from agent.claude_billing import PASS_THROUGH_ENV_VARS

    assert set(PASS_THROUGH_ENV_VARS) & BLOCKED_CHILD_ENV_VARS == set()


def test_home_is_never_overridden():
    """Rewriting HOME relocates the macOS keychain lookup and breaks login."""
    base = _dirty_env()
    child = sanitized_child_env(base)
    assert child["HOME"] == base["HOME"]


def test_building_the_child_env_does_not_mutate_os_environ(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-live")
    before = dict(os.environ)
    child = sanitized_child_env()
    assert "ANTHROPIC_API_KEY" not in child
    assert dict(os.environ) == before
    # ...which is what leaves every other api_mode's credential resolution
    # exactly as it was.
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-live"


# ---------------------------------------------------------------------------
# The transport's env builder
# ---------------------------------------------------------------------------


@dataclass
class _Options:
    env: dict = field(default_factory=dict)
    cwd: str | None = None
    enable_file_checkpointing: bool = False
    stderr: Any = None
    user: Any = None


def test_transport_env_strips_blocked_keys(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-live")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    child = claude_sanitized_transport.build_child_env(_Options(cwd="/work"))
    assert "ANTHROPIC_API_KEY" not in child
    assert "CLAUDE_CODE_USE_BEDROCK" not in child
    assert child["PWD"] == "/work"
    # The SDK's own required markers survive the rebuild.
    assert child["CLAUDE_CODE_ENTRYPOINT"] == "sdk-py"


def test_options_env_cannot_smuggle_a_credential_back_in(monkeypatch):
    """`options.env` is applied, then re-filtered — it is not an escape hatch."""
    child = claude_sanitized_transport.build_child_env(
        _Options(env={"ANTHROPIC_API_KEY": "sk-ant-sneaky", "FOO": "bar"})
    )
    assert "ANTHROPIC_API_KEY" not in child
    assert child["FOO"] == "bar"


def test_transport_env_leaves_home_and_config_dir_alone(monkeypatch):
    monkeypatch.setenv("HOME", "/home/tester")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/tmp/cfg")
    child = claude_sanitized_transport.build_child_env(_Options())
    assert child["HOME"] == "/home/tester"
    assert child["CLAUDE_CONFIG_DIR"] == "/tmp/cfg"


@requires_sdk
def test_sanitized_transport_subclasses_the_sdk_transport():
    from claude_agent_sdk._internal.transport.subprocess_cli import (
        SubprocessCLITransport,
    )

    assert issubclass(
        claude_sanitized_transport.sanitized_transport_class(), SubprocessCLITransport
    )


@requires_sdk
def test_sdk_options_env_alone_cannot_delete_an_inherited_key(monkeypatch):
    """Why a transport is required at all.

    The SDK merges ``options.env`` *over* a copy of ``os.environ``; there is no
    value that means "remove". This pins the SDK behavior our transport exists
    to work around.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-live")
    inherited = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    sdk_merge = {**inherited, "CLAUDE_CODE_ENTRYPOINT": "sdk-py", **{"ANTHROPIC_API_KEY": ""}}
    assert "ANTHROPIC_API_KEY" in sdk_merge  # still present, merely blank

    ours = claude_sanitized_transport.build_child_env(_Options())
    assert "ANTHROPIC_API_KEY" not in ours


# ---------------------------------------------------------------------------
# Classifying the CLI's own account report
# ---------------------------------------------------------------------------


def test_a_plan_login_is_a_subscription():
    source = classify_account(
        {
            "email": "user@example.com",
            "organization": "Example",
            "subscriptionType": "Claude Max",
            "apiProvider": "firstParty",
        }
    )
    assert source.is_subscription
    assert source.plan == "Claude Max"
    assert billing_source_refusal(source) is None


@pytest.mark.parametrize(
    "tier",
    [
        "Claude Max",
        "claudeProSubscription",
        "claudeMax20xSubscription",
        "claudeTeamSubscription",
        "claudeEnterpriseSubscription",
    ],
)
def test_every_plan_tier_shape_is_a_subscription(tier):
    source = classify_account({"subscriptionType": tier, "apiProvider": "firstParty"})
    assert source.is_subscription


def test_an_api_key_source_wins_over_a_claude_ai_token_source():
    """Observed on Claude Code 2.1.220 with ANTHROPIC_API_KEY exported."""
    source = classify_account(
        {
            "tokenSource": "claude.ai",
            "apiKeySource": "ANTHROPIC_API_KEY",
            "apiProvider": "firstParty",
        }
    )
    assert source.kind == "api_key"
    refusal = billing_source_refusal(source)
    assert refusal is not None
    assert "ANTHROPIC_API_KEY" in refusal


@pytest.mark.parametrize(
    "token_source", ["ANTHROPIC_AUTH_TOKEN", "anthropicApiKey", "apiKeyHelper"]
)
def test_api_shaped_token_sources_refuse_and_name_themselves(token_source):
    source = classify_account({"tokenSource": token_source})
    assert source.kind == "api_key"
    refusal = billing_source_refusal(source)
    assert refusal is not None
    assert token_source in refusal


def test_an_oauth_token_is_the_extra_usage_meter_not_the_plan():
    name = "CLAUDE_CODE_OAUTH" + "_TOKEN"
    source = classify_account({"tokenSource": name})
    assert source.kind == "oauth_token"
    refusal = billing_source_refusal(source)
    assert refusal is not None
    assert name in refusal


def test_a_cloud_provider_refuses_and_names_the_provider():
    source = classify_account({"apiProvider": "bedrock", "subscriptionType": "max"})
    assert source.kind == "cloud"
    refusal = billing_source_refusal(source)
    assert refusal is not None
    assert "bedrock" in refusal


def test_no_signed_in_account_refuses_with_the_login_command():
    source = classify_account({"tokenSource": "none"})
    assert source.kind == "unauthenticated"
    assert "claude auth login" in (billing_source_refusal(source) or "")


def test_an_unreadable_payload_refuses_rather_than_assuming():
    assert classify_account(None).kind == "unknown"
    assert billing_source_refusal(BillingSource(kind="unknown")) is not None


# ---------------------------------------------------------------------------
# The zero-cost probe
# ---------------------------------------------------------------------------


class _FakeTransport:
    """Records everything a real transport would be asked to do."""

    def __init__(self, options) -> None:
        self.options = options
        self.writes: list[str] = []
        self.connected = False

    async def connect(self) -> None:
        self.connected = True

    async def write(self, data: str) -> None:  # pragma: no cover - must not run
        self.writes.append(data)


class _FakeClient:
    """Fails loudly if the probe tries to run a turn."""

    def __init__(self, *, options, transport) -> None:
        self.options = options
        self.transport = transport
        self.calls: list[str] = []
        self.account: dict = {
            "email": "user@example.com",
            "subscriptionType": "Claude Max",
            "apiProvider": "firstParty",
        }

    async def connect(self, *_a, **_kw) -> None:
        self.calls.append("connect")

    async def get_server_info(self) -> dict:
        self.calls.append("get_server_info")
        return {"account": dict(self.account)}

    async def disconnect(self) -> None:
        self.calls.append("disconnect")

    async def query(self, *_a, **_kw):  # pragma: no cover - must not run
        raise AssertionError("the billing probe must not issue a model request")

    def receive_response(self):  # pragma: no cover - must not run
        raise AssertionError("the billing probe must not read a model response")


def _probe(account: dict | None = None):
    """Run the probe against injected fakes; return (source, client)."""
    made: dict = {}

    def _client_factory(*, options, transport):
        client = _FakeClient(options=options, transport=transport)
        if account is not None:
            client.account = account
        made["client"] = client
        return client

    source = claude_billing.probe_claude_billing_source(
        timeout=5.0,
        client_factory=_client_factory,
        options_factory=lambda: types.SimpleNamespace(tools=[], setting_sources=[]),
        transport_factory=_FakeTransport,
    )
    return source, made["client"]


def test_the_probe_reads_the_account_without_issuing_a_model_request():
    source, client = _probe()
    assert source.is_subscription
    assert source.plan == "Claude Max"
    # Connect, read the init report, hang up. Nothing else.
    assert client.calls == ["connect", "get_server_info", "disconnect"]
    assert client.transport.writes == []


@requires_sdk
def test_the_probe_session_loads_no_settings_and_no_tools():
    """`setting_sources=[]` is also what keeps an apiKeyHelper from running.

    Verified against Claude Code 2.1.220: with ``["user"]`` the helper script
    executes and the account reports ``tokenSource: apiKeyHelper``; with ``[]``
    it never runs.
    """
    options = claude_billing.probe_options()
    assert options.setting_sources == []
    assert options.tools == []
    assert options.allowed_tools == []


def test_the_probe_refuses_when_the_cli_resolved_an_api_key():
    source, _ = _probe({"apiKeySource": "ANTHROPIC_API_KEY", "tokenSource": "claude.ai"})
    assert source.kind == "api_key"
    assert "ANTHROPIC_API_KEY" in (billing_source_refusal(source) or "")


def test_the_probe_falls_back_to_claude_auth_status_when_the_sdk_path_fails():
    def _explode(**_kw):
        raise RuntimeError("no sdk here")

    with patch(
        "hermes_cli.claude_code.probe_claude_auth",
        return_value={
            "logged_in": True,
            "auth_method": "claude.ai",
            "subscription_type": "max",
            "account": "user@example.com",
        },
    ):
        source = claude_billing.probe_claude_billing_source(
            timeout=1.0,
            client_factory=_explode,
            options_factory=lambda: types.SimpleNamespace(),
            transport_factory=_FakeTransport,
        )
    assert source.is_subscription
    assert source.plan == "max"


def test_the_fallback_refuses_when_claude_auth_status_says_api_key():
    with patch(
        "hermes_cli.claude_code.probe_claude_auth",
        return_value={"logged_in": True, "auth_method": "api-key"},
    ):
        source = claude_billing.probe_claude_billing_source(
            timeout=1.0,
            client_factory=lambda **_kw: (_ for _ in ()).throw(RuntimeError("boom")),
            options_factory=lambda: types.SimpleNamespace(),
            transport_factory=_FakeTransport,
        )
    assert source.kind == "api_key"
    assert billing_source_refusal(source) is not None


# ---------------------------------------------------------------------------
# The combined gate
# ---------------------------------------------------------------------------


def test_the_static_check_short_circuits_before_spawning_anything():
    def _never(**_kw):  # pragma: no cover - must not run
        raise AssertionError("a refused environment must not spawn the CLI")

    refusal = verify_claude_billing_source(
        env={"ANTHROPIC_API_KEY": "sk-ant"}, client_factory=_never
    )
    assert refusal is not None
    assert "ANTHROPIC_API_KEY" in refusal


def test_a_clean_environment_and_a_plan_login_is_allowed():
    with patch.object(
        claude_billing,
        "probe_claude_billing_source",
        return_value=BillingSource(kind="subscription", plan="Claude Max"),
    ):
        assert verify_claude_billing_source(env={"PATH": "/usr/bin"}) is None


# ---------------------------------------------------------------------------
# The runtime seam
# ---------------------------------------------------------------------------


def _agent() -> Any:
    """The bare surface `verify_claude_billing_for_agent` touches."""
    return types.SimpleNamespace()


def test_the_billing_verdict_is_cached_for_the_life_of_the_session():
    from agent.claude_runtime import verify_claude_billing_for_agent

    agent = _agent()
    calls = []

    def _verify(**_kw):
        calls.append(1)
        return None

    with patch.object(claude_billing, "verify_claude_billing_source", _verify):
        assert verify_claude_billing_for_agent(agent) is None
        assert verify_claude_billing_for_agent(agent) is None
    assert len(calls) == 1


def test_retiring_a_session_forces_the_next_one_to_re_prove():
    from agent import claude_runtime

    closed = []
    agent = types.SimpleNamespace(
        _claude_session=types.SimpleNamespace(close=lambda: closed.append(1)),
        _claude_billing_refusal="stale verdict",
    )
    claude_runtime._retire_session(agent)
    assert closed == [1]
    assert agent._claude_billing_refusal is claude_runtime._UNSET


def test_a_refused_turn_returns_a_failure_that_names_the_variable(monkeypatch):
    from agent import claude_runtime

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-live")

    agent = types.SimpleNamespace(
        _interrupt_requested=False,
        _interrupt_message=None,
        clear_interrupt=lambda: None,
    )
    messages: list = []

    def _no_session(*_a, **_kw):  # pragma: no cover - must not run
        raise AssertionError("a refused turn must not build a session")

    with (
        patch.object(claude_runtime, "claude_runtime_preflight", return_value=None),
        patch.object(claude_runtime, "_ensure_session", _no_session),
    ):
        result = claude_runtime.run_claude_agent_sdk_turn(
            agent,
            user_message="hi",
            original_user_message="hi",
            messages=messages,
            effective_task_id="task-1",
        )

    assert result["failed"] is True
    assert result["completed"] is False
    assert "ANTHROPIC_API_KEY" in result["final_response"]
    assert "unset ANTHROPIC_API_KEY" in result["error"]


def test_preflight_refuses_a_conflicting_credential_before_the_login_probe(monkeypatch):
    from agent.claude_runtime import claude_runtime_preflight

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-live")

    def _never():  # pragma: no cover - must not run
        raise AssertionError("no subprocess before the free check has spoken")

    with (
        patch(
            "hermes_cli.claude_subscription.claude_agent_sdk_available",
            return_value=True,
        ),
        patch("hermes_cli.claude_code.probe_claude_auth", _never),
    ):
        message = claude_runtime_preflight({"claude_subscription": {"enabled": True}})
    assert message is not None
    assert "ANTHROPIC_API_KEY" in message


# ---------------------------------------------------------------------------
# Session start
# ---------------------------------------------------------------------------


class _StubClient:
    def __init__(self, *, options, transport=None) -> None:
        self.options = options
        self.transport = transport

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None


def test_starting_a_session_does_not_mutate_os_environ(monkeypatch):
    from agent.transports.claude_agent_session import ClaudeAgentSession

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-live")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/tmp/cfg")
    before = dict(os.environ)

    built: dict = {}

    def _transport_factory(options):
        built["env"] = claude_sanitized_transport.build_child_env(options)
        return object()

    session = ClaudeAgentSession(
        options_factory=lambda: _Options(),
        client_factory=_StubClient,
        transport_factory=_transport_factory,
    )
    try:
        session.ensure_started()
    finally:
        session.close()

    assert dict(os.environ) == before
    assert "ANTHROPIC_API_KEY" not in built["env"]
    assert built["env"]["CLAUDE_CONFIG_DIR"] == "/tmp/cfg"


def test_a_session_without_a_transport_factory_still_connects():
    """Existing callers keep working; the transport is opt-in at the seam."""
    from agent.transports.claude_agent_session import ClaudeAgentSession

    seen: dict = {}

    def _factory(**kwargs):
        seen.update(kwargs)
        return _StubClient(**kwargs)

    session = ClaudeAgentSession(
        options_factory=lambda: _Options(), client_factory=_factory
    )
    try:
        session.ensure_started()
    finally:
        session.close()
    assert "transport" not in seen


# ---------------------------------------------------------------------------
# stderr routing
# ---------------------------------------------------------------------------


def test_claude_stderr_goes_to_the_logs_not_the_users_screen(caplog):
    from agent.claude_runtime import _make_stderr_logger

    sink = _make_stderr_logger()
    with caplog.at_level("DEBUG", logger="agent.claude_runtime.claude_stderr"):
        sink("routine chatter")
        sink("")
        sink("Error: something broke")

    messages = [record.getMessage() for record in caplog.records]
    assert any("routine chatter" in m for m in messages)
    # The blank line is dropped, not logged as an empty record.
    assert not any(m.strip() == "claude:" for m in messages)
    levels = {
        record.levelname
        for record in caplog.records
        if "something broke" in record.getMessage()
    }
    assert "WARNING" in levels


def test_error_shaped_stderr_escalation_is_capped(caplog):
    from agent.claude_runtime import MAX_ESCALATED_STDERR_LINES, _make_stderr_logger

    sink = _make_stderr_logger()
    with caplog.at_level("DEBUG", logger="agent.claude_runtime.claude_stderr"):
        for i in range(MAX_ESCALATED_STDERR_LINES * 3):
            sink(f"error number {i}")

    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == MAX_ESCALATED_STDERR_LINES


@requires_sdk
def test_the_runtime_registers_a_stderr_callback():
    """Without one the SDK lets the child inherit the parent's stderr, which
    under Electron can be a closed handle."""
    from agent.claude_runtime import build_claude_agent_options

    agent = MagicMock()
    agent.model = "claude-sonnet-4-5"
    with patch(
        "agent.claude_runtime.build_hermes_sdk_mcp_server", return_value=MagicMock()
    ), patch("agent.claude_runtime.bridged_allowed_tools", return_value=[]):
        options = build_claude_agent_options(
            agent, system_prompt="prompt", effective_task_id=lambda: "t", cwd="/work"
        )
    assert callable(options.stderr)
    # The child env is built by the transport, not here.
    assert options.env == {}


# ---------------------------------------------------------------------------
# The spawn itself
# ---------------------------------------------------------------------------


@requires_sdk
def test_the_spawn_receives_a_sanitized_environment_and_a_stdin_pipe(monkeypatch):
    """End-to-end at the launcher: what actually reaches ``open_process``.

    Nothing is executed — ``anyio.open_process`` is replaced, so no ``claude``
    process is created and no network call is possible.
    """
    import anyio
    from subprocess import PIPE

    from claude_agent_sdk import ClaudeAgentOptions

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-live")
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "bearer-live")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", "/tmp/cfg")
    monkeypatch.setenv("HOME", "/home/tester")
    # Skip the CLI version probe, which would spawn the real binary.
    monkeypatch.setenv("CLAUDE_AGENT_SDK_SKIP_VERSION_CHECK", "1")

    captured: dict = {}

    class _FakeProcess:
        stdout = stderr = stdin = None

    async def _fake_open_process(cmd, **kwargs):
        captured["cmd"] = cmd
        captured.update(kwargs)
        return _FakeProcess()

    monkeypatch.setattr(anyio, "open_process", _fake_open_process)

    options = ClaudeAgentOptions(
        tools=[],
        setting_sources=[],
        cli_path="/usr/bin/claude-not-executed",
        cwd="/work",
        env={"ANTHROPIC_API_KEY": "sk-ant-sneaky"},
    )
    transport = claude_sanitized_transport.build_sanitized_transport(options)

    import asyncio

    asyncio.run(transport.connect())

    env = captured["env"]
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK"):
        assert name not in env
    assert env["CLAUDE_CONFIG_DIR"] == "/tmp/cfg"
    assert env["HOME"] == "/home/tester"
    assert env["PWD"] == "/work"
    # Claude Code probes stdin and blocks forever on an unusable inherited one.
    assert captured["stdin"] is PIPE
    assert dict(os.environ)["ANTHROPIC_API_KEY"] == "sk-ant-live"
