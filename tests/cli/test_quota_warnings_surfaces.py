"""Tests for the quota-warning CLI surfaces (issue #6567, Task C).

Covers the three wired surfaces:

  * ``/quota`` command registration — ``CommandDef`` with ``cli_only=True``,
    resolvable via ``resolve_command`` and present in the ``COMMANDS`` dict.
  * Pre-turn warning — ``HermesCLI._maybe_emit_pre_turn_quota_warning`` honors
    ``quota.suppress_warnings`` (prints nothing when suppressed).
  * Startup warning — ``CLIAgentSetupMixin._emit_startup_quota_warning`` ALWAYS
    fires even when ``quota.suppress_warnings`` is set.
  * ``/quota`` (``HermesCLI._show_quota``) — ALWAYS renders the full
    account-usage block + warning lines, regardless of suppression.

The quota methods only touch ``self.agent`` / ``self.provider`` / ``self.config``
and the module-level ``CLI_CONFIG`` (on cli.py) / lazy ``agent.quota_warnings``
imports — so each is bound to a lightweight ``SimpleNamespace`` stub with
``types.MethodType`` instead of spinning up a real ``HermesCLI``.
"""

from __future__ import annotations

import types
from datetime import datetime, timezone
from types import SimpleNamespace

import cli as cli_mod
from agent.account_usage import AccountUsageSnapshot, AccountUsageWindow
from agent.quota_warnings import (
    quota_warning_lines,
    startup_warning_lines,
)
from hermes_cli import cli_agent_setup_mixin as mixin_mod
from hermes_cli.commands import COMMANDS, CommandDef, resolve_command


# ── fixtures / helpers ──────────────────────────────────────────────────────


def _critical_snapshot() -> AccountUsageSnapshot:
    """A snapshot sitting at 98% (trips the critical >= 95 threshold)."""
    return AccountUsageSnapshot(
        provider="openai",
        source="openai",
        fetched_at=datetime.now(timezone.utc),
        windows=(
            AccountUsageWindow(label="This month", used_percent=98.0),
        ),
    )


def _make_stub(provider: str | None = "openai", config=None) -> SimpleNamespace:
    """Minimal stand-in for ``HermesCLI`` carrying exactly what the quota
    methods read: ``agent``/``provider``/``base_url``/``api_key``/``config``.

    Defaults to explicit credentials (a real agent carries the resolved
    runtime creds) — the probe only fetches the ALREADY-RESOLVED active
    account and skips when creds are missing (regression:
    test_cli_provider_resolution.py::test_runtime_resolution_failure_is_not_sticky).
    """
    agent = SimpleNamespace(
        provider=provider, base_url="https://example.com", api_key="test-key"
    )
    return SimpleNamespace(
        provider=provider,
        base_url="https://example.com",
        api_key="test-key",
        agent=agent,
        config=config if config is not None else {},
    )


def _bind(cls, stub: SimpleNamespace, name: str) -> SimpleNamespace:
    """Bind a method from ``cls`` onto ``stub`` so it runs with the stub as self."""
    setattr(stub, name, types.MethodType(getattr(cls, name), stub))
    return stub


SUPPRESS = {"quota": {"suppress_warnings": True}}


# ── command registration ────────────────────────────────────────────────────


def test_quota_command_registered():
    cmd = resolve_command("quota")
    assert cmd is not None
    assert cmd.name == "quota"
    assert isinstance(cmd, CommandDef)
    # cli_only=True keeps it out of the gateway dispatch surface (issue #6565).
    assert cmd.cli_only is True
    assert cmd.gateway_only is False
    # Present in the backwards-compat COMMANDS dict (non-gateway entries only).
    assert "/quota" in COMMANDS


# ── engine semantics: the invariant the surfaces lean on ────────────────────


def test_engine_suppression_semantics():
    """``startup_warning_lines`` always warns; ``quota_warning_lines`` honors
    ``quota.suppress_warnings`` — the core invariant Task C's surfaces depend
    on."""
    snap = _critical_snapshot()
    # Suppressed config silences the pre-turn path...
    assert quota_warning_lines(snap, SUPPRESS) == []
    # ...but never the startup path.
    assert startup_warning_lines(snap, SUPPRESS)  # non-empty
    # And both fire when suppression is absent.
    assert quota_warning_lines(snap, {})
    assert startup_warning_lines(snap, {})


# ── pre-turn warning ────────────────────────────────────────────────────────


def test_pre_turn_warning_prints_when_not_suppressed(monkeypatch, capsys):
    monkeypatch.setattr(
        "agent.quota_warnings.fetch_quota_snapshot",
        lambda *a, **k: _critical_snapshot(),
    )
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", {})

    stub = _make_stub()
    _bind(cli_mod.HermesCLI, stub, "_maybe_emit_pre_turn_quota_warning")
    stub._maybe_emit_pre_turn_quota_warning()

    out = capsys.readouterr().out
    assert "Critical quota warning" in out


def test_pre_turn_warning_silenced_when_suppressed(monkeypatch, capsys):
    monkeypatch.setattr(
        "agent.quota_warnings.fetch_quota_snapshot",
        lambda *a, **k: _critical_snapshot(),
    )
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", SUPPRESS)

    stub = _make_stub()
    _bind(cli_mod.HermesCLI, stub, "_maybe_emit_pre_turn_quota_warning")
    stub._maybe_emit_pre_turn_quota_warning()

    out = capsys.readouterr().out
    assert "Critical quota warning" not in out
    assert out.strip() == ""


def test_pre_turn_warning_no_provider_is_silent(monkeypatch, capsys):
    """No provider → no probe, no output, no crash (fail-open)."""
    monkeypatch.setattr(
        "agent.quota_warnings.fetch_quota_snapshot",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not fetch")),
    )
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", {})

    stub = _make_stub(provider=None)
    stub.agent = SimpleNamespace(provider=None, base_url=None, api_key=None)
    _bind(cli_mod.HermesCLI, stub, "_maybe_emit_pre_turn_quota_warning")
    stub._maybe_emit_pre_turn_quota_warning()

    assert capsys.readouterr().out == ""


def test_pre_turn_warning_no_credentials_is_silent(monkeypatch, capsys):
    """Provider set but no resolved credentials → no probe, no output.

    The probe only fetches the ALREADY-RESOLVED active account; with missing
    creds a fetch would trigger runtime-provider credential resolution as a
    side effect, which an advisory probe must never do (regression:
    test_cli_provider_resolution.py::test_runtime_resolution_failure_is_not_sticky).
    """
    monkeypatch.setattr(
        "agent.quota_warnings.fetch_quota_snapshot",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not fetch")),
    )
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", {})

    stub = _make_stub(provider="openai")
    stub.agent = SimpleNamespace(
        provider="openai", base_url=None, api_key=None
    )
    stub.base_url = None
    stub.api_key = None
    _bind(cli_mod.HermesCLI, stub, "_maybe_emit_pre_turn_quota_warning")
    stub._maybe_emit_pre_turn_quota_warning()

    assert capsys.readouterr().out == ""


# ── startup warning ────────────────────────────────────────────────────────


def test_startup_warning_fires_even_when_suppressed(monkeypatch, capsys):
    """The startup probe must surface a critical warning regardless of
    ``quota.suppress_warnings`` (issue #6567 acceptance criterion)."""
    monkeypatch.setattr(
        "agent.quota_warnings.fetch_quota_snapshot",
        lambda *a, **k: _critical_snapshot(),
    )
    cleared = []
    monkeypatch.setattr(
        "agent.quota_warnings.clear_quota_cache",
        lambda: cleared.append(True),
    )

    stub = _make_stub(config=SUPPRESS)
    _bind(mixin_mod.CLIAgentSetupMixin, stub, "_emit_startup_quota_warning")
    stub._emit_startup_quota_warning()

    out = capsys.readouterr().out
    assert "Critical quota warning" in out
    # Fresh-cache-per-session requirement: clear must have run at session start.
    assert cleared == [True]


def test_startup_warning_no_provider_is_silent(monkeypatch, capsys):
    """No provider → startup still clears the cache but emits nothing."""
    cleared = []
    monkeypatch.setattr(
        "agent.quota_warnings.clear_quota_cache",
        lambda: cleared.append(True),
    )
    monkeypatch.setattr(
        "agent.quota_warnings.fetch_quota_snapshot",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not fetch")),
    )

    stub = _make_stub(provider=None, config={})
    stub.agent = SimpleNamespace(provider=None, base_url=None, api_key=None)
    _bind(mixin_mod.CLIAgentSetupMixin, stub, "_emit_startup_quota_warning")
    stub._emit_startup_quota_warning()

    assert capsys.readouterr().out == ""
    assert cleared == [True]


# ── /quota command surface ──────────────────────────────────────────────────


def test_show_quota_always_full_even_when_suppressed(monkeypatch, capsys):
    """``/quota`` renders the account-usage block + warning even when
    ``quota.suppress_warnings`` is set (uses ``startup_warning_lines``)."""
    monkeypatch.setattr(
        "agent.quota_warnings.fetch_quota_snapshot",
        lambda *a, **k: _critical_snapshot(),
    )
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", SUPPRESS)

    stub = _make_stub()
    _bind(cli_mod.HermesCLI, stub, "_show_quota")
    stub._show_quota()

    out = capsys.readouterr().out
    # Full account-usage block is always shown.
    assert "Account limits" in out
    # Warning is shown even though suppression is on.
    assert "Critical quota warning" in out


def test_show_quota_no_data_note_when_no_provider(monkeypatch, capsys):
    """Fail-open: no provider → friendly note, no crash."""
    monkeypatch.setattr(
        "agent.quota_warnings.fetch_quota_snapshot",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("must not fetch")),
    )
    monkeypatch.setattr(cli_mod, "CLI_CONFIG", {})

    stub = _make_stub(provider=None)
    stub.agent = SimpleNamespace(provider=None, base_url=None, api_key=None)
    _bind(cli_mod.HermesCLI, stub, "_show_quota")
    stub._show_quota()

    out = capsys.readouterr().out
    assert "No quota data for the current provider" in out


# ── /quota dispatch ──────────────────────────────────────────────────────────


def test_handle_quota_command_delegates_to_show_quota():
    """`/quota` is a thin dispatch: stray args are ignored and `_show_quota` is
    invoked exactly once (issue #6567 Task C — dead-code removal).

    The spy is installed as a plain instance attribute, which Python looks up
    at call time — so the bound `_handle_quota_command` calls it without ever
    touching the real (network-bound) `_show_quota`.
    """
    calls = []

    def _spy():
        calls.append(True)

    stub = _make_stub()
    _bind(cli_mod.HermesCLI, stub, "_handle_quota_command")
    stub._show_quota = _spy

    stub._handle_quota_command("/quota whatever")

    assert calls == [True]
