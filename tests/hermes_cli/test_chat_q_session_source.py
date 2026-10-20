"""``hermes chat -q`` must create its session even when the shell inherits an automation
``HERMES_SESSION_SOURCE`` (kanban worker, cron, ...).

WS-A regression: the create wire (gateway ``session_policy.SURFACES`` and the canonical create
contract's ``source`` Literal) only stores ``cli/tui/gui/acp/tool/oneshot``, but
``gateway_session_store.create`` emitted the inherited label verbatim (``kanban``), so every
finite ``chat -q`` run from a worker shell failed with ``invalid_params`` fields=['source'].
"""
import pytest

from agent import session_source as ss


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ("HERMES_SESSION_SOURCE", "HERMES_SESSION_SOURCE_EXPLICIT",
                 "HERMES_SINGLE_QUERY_SESSION"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TERMINAL_ENV", "local")


def test_inherited_automation_label_finite_run_becomes_oneshot(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "kanban")
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    assert ss.create_wire_source_for("cli") == "oneshot"


def test_inherited_automation_label_interactive_run_becomes_cli(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "kanban")
    assert ss.create_wire_source_for("cli") == "cli"


def test_explicit_source_is_kept_verbatim(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_SOURCE", "tool")
    monkeypatch.setenv("HERMES_SESSION_SOURCE_EXPLICIT", "1")
    assert ss.create_wire_source_for("cli") == "tool"


def test_plain_shell_finite_run_stays_oneshot(monkeypatch):
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    assert ss.create_wire_source_for("cli") == "oneshot"


def test_plain_shell_interactive_run_stays_cli():
    assert ss.create_wire_source_for("cli") == "cli"


def test_store_create_params_pass_the_canonical_gate(monkeypatch):
    """The exact WS-A failure path: store create params must clear the canonical dispatcher."""
    from hermes_cli.gateway_session_store import GatewaySessionStore
    from tui_gateway.contracts.registry import CANONICAL_METHODS, canonical_param_problems

    monkeypatch.setenv("HERMES_SESSION_SOURCE", "kanban")
    monkeypatch.setenv("HERMES_SINGLE_QUERY_SESSION", "1")
    captured = {}
    store = GatewaySessionStore.__new__(GatewaySessionStore)
    store.call = lambda method, **params: captured.update(params) or {"stored_session_id": "local-test"}
    store._adopt = lambda snapshot: "local-test"
    store.create()

    assert captured["source"] == "oneshot"
    contract = CANONICAL_METHODS["session.create"]
    assert canonical_param_problems(contract, captured) == []


def test_pre_fix_label_is_the_named_refusal():
    """Documents the failure shape the fix removes: fields=['source'] on the inherited label."""
    from tui_gateway.contracts.registry import CANONICAL_METHODS, canonical_param_problems

    contract = CANONICAL_METHODS["session.create"]
    params = {"request_id": "0" * 32, "source": "kanban", "cwd": "/tmp", "row_cwd": ""}
    assert canonical_param_problems(contract, params) == ["source"]


def test_create_vocabulary_matches_the_canonical_literal():
    """Skew guard: the client-side vocabulary mirror must match the wire contract exactly."""
    from tui_gateway.contracts.canonical_sessions import CanonicalCreateParams

    literal = CanonicalCreateParams.model_fields["source"].annotation
    assert set(literal.__args__) == set(ss.CREATE_SOURCES)
