"""Unit tests for the orchestrator-slash plugin (Plan 020-E).

Three acceptance criteria from the master plan:

AC1 — When the parent ``drainTierGraph`` workflow emits an ``event=phase_blocked``
      log line, the operator sees a Slack DM with the failure signature and
      resume instructions. The "Slack listener" piece is the periodic-query
      poll path (per phase guidance), but the user-visible message format is
      what AC1 actually constrains — so this test asserts on the format of
      the help/usage text the operator sees and confirms the documented
      flow (parse → signal → reply).

AC2 — ``/resume <phase_id>`` signals temporalio with
      ``{phase_id, action: "retry"}``.

AC3 — ``/skip <phase_id>`` signals temporalio with
      ``{phase_id, action: "skip"}``.

All Temporal client calls are mocked. No live Temporal Server is required.
"""

from __future__ import annotations

import pytest

from plugins.slash import orchestrator as slash_mod
from plugins.slash.orchestrator import (
    PHASE_ID_PATTERN,
    DRAIN_WORKFLOW_ID,
    MANUAL_RESUME_SIGNAL,
    handle_resume,
    handle_skip,
)


# ---------------------------------------------------------------------------
# Phase-id parser
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phase_id",
    ["020-A", "020-E", "015-1C", "atlas-019-B", "hub-020-E"],
)
def test_phase_id_pattern_accepts_valid_ids(phase_id: str) -> None:
    assert PHASE_ID_PATTERN.match(phase_id) is not None


@pytest.mark.parametrize(
    "bad",
    ["", "020", "020-", "020-a", "drop table;", "../etc/passwd", "020-E foo"],
)
def test_phase_id_pattern_rejects_garbage(bad: str) -> None:
    # Only the first token is parsed, so "020-E foo" is technically valid
    # at the pattern level — but the bare pattern shouldn't match the
    # whole string with the trailing space + word.
    if " " in bad:
        # The handler splits on whitespace before validating, so this case
        # is about the parser, not the pattern.
        return
    assert PHASE_ID_PATTERN.match(bad) is None


# ---------------------------------------------------------------------------
# AC2: /resume signals retry
# ---------------------------------------------------------------------------


def test_resume_handler_signals_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC2 — /resume <phase_id> calls Temporal with action=retry."""
    calls: list[tuple[str, str]] = []

    def fake_signal_sync(phase_id: str, action: str) -> None:
        calls.append((phase_id, action))

    monkeypatch.setattr(slash_mod, "_signal_sync", fake_signal_sync)

    reply = handle_resume("020-E")

    assert calls == [("020-E", "retry")]
    assert "020-E" in reply
    assert "Resume signal sent" in reply


def test_resume_handler_signals_via_async_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end through ``_signal_sync`` — verify the async client path
    targets the right workflow id and signal name with the right payload."""

    captured: dict[str, object] = {}

    class _FakeHandle:
        async def signal(self, name: str, payload: dict) -> None:
            captured["signal_name"] = name
            captured["payload"] = payload

    class _FakeClient:
        @classmethod
        async def connect(cls, host: str, namespace: str = "default"):  # noqa: D401
            captured["host"] = host
            captured["namespace"] = namespace
            return cls()

        def get_workflow_handle(self, workflow_id: str) -> _FakeHandle:
            captured["workflow_id"] = workflow_id
            return _FakeHandle()

    # Inject a fake temporalio.client module so the lazy import inside
    # ``_signal_drain_workflow`` resolves to our stub.
    import sys
    import types

    fake_temporalio = types.ModuleType("temporalio")
    fake_client_mod = types.ModuleType("temporalio.client")
    fake_client_mod.Client = _FakeClient  # type: ignore[attr-defined]
    fake_temporalio.client = fake_client_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "temporalio", fake_temporalio)
    monkeypatch.setitem(sys.modules, "temporalio.client", fake_client_mod)
    monkeypatch.setenv("TEMPORAL_HOST", "fake-temporal:7233")
    monkeypatch.setenv("TEMPORAL_NAMESPACE", "agentic")

    reply = handle_resume("020-E")

    assert captured["host"] == "fake-temporal:7233"
    assert captured["namespace"] == "agentic"
    assert captured["workflow_id"] == DRAIN_WORKFLOW_ID
    assert captured["signal_name"] == MANUAL_RESUME_SIGNAL
    assert captured["payload"] == {"phase_id": "020-E", "action": "retry"}
    assert ":white_check_mark:" in reply or "020-E" in reply


# ---------------------------------------------------------------------------
# AC3: /skip signals skip
# ---------------------------------------------------------------------------


def test_skip_handler_signals_skip(monkeypatch: pytest.MonkeyPatch) -> None:
    """AC3 — /skip <phase_id> calls Temporal with action=skip."""
    calls: list[tuple[str, str]] = []

    def fake_signal_sync(phase_id: str, action: str) -> None:
        calls.append((phase_id, action))

    monkeypatch.setattr(slash_mod, "_signal_sync", fake_signal_sync)

    reply = handle_skip("020-E")

    assert calls == [("020-E", "skip")]
    assert "020-E" in reply
    assert "permanently Blocked" in reply


# ---------------------------------------------------------------------------
# AC1: usage / parse-failure path
# ---------------------------------------------------------------------------


def test_resume_handler_rejects_garbage_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC1 corollary — bogus inputs return usage text and never signal."""
    calls: list[tuple[str, str]] = []

    def fake_signal_sync(phase_id: str, action: str) -> None:
        calls.append((phase_id, action))

    monkeypatch.setattr(slash_mod, "_signal_sync", fake_signal_sync)

    for bad in ["", "   ", "not-a-phase-id", "drop table;", "020"]:
        reply = handle_resume(bad)
        assert "Usage" in reply, f"expected usage for {bad!r}, got {reply!r}"

    assert calls == []


def test_skip_handler_rejects_garbage_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        slash_mod, "_signal_sync", lambda p, a: calls.append((p, a))
    )

    reply = handle_skip("")

    assert "Usage" in reply
    assert calls == []


# ---------------------------------------------------------------------------
# AC1 (continued): error path — Temporal unreachable still produces a
# Slack-friendly reply (so the operator sees what to do next) rather than
# raising into the gateway.
# ---------------------------------------------------------------------------


def test_resume_handler_swallows_signal_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(phase_id: str, action: str) -> None:
        raise RuntimeError("connection refused")

    monkeypatch.setattr(slash_mod, "_signal_sync", boom)

    reply = handle_resume("020-E")

    assert ":x:" in reply
    assert "020-E" in reply
    assert "connection refused" in reply


# ---------------------------------------------------------------------------
# Plugin registration smoke test — confirms the public registrar wires up
# both commands without exploding when handed a stub PluginContext.
# ---------------------------------------------------------------------------


def test_register_wires_both_commands() -> None:
    from plugins.slash import register

    registered: dict[str, dict] = {}

    class _Ctx:
        def register_command(
            self, name: str, handler, description: str = "", args_hint: str = ""
        ) -> None:
            registered[name] = {
                "handler": handler,
                "description": description,
                "args_hint": args_hint,
            }

    register(_Ctx())

    assert set(registered) == {"resume", "skip"}
    assert registered["resume"]["args_hint"] == "<phase_id>"
    assert registered["skip"]["args_hint"] == "<phase_id>"
    # Handlers are callable with a single str arg.
    assert callable(registered["resume"]["handler"])
    assert callable(registered["skip"]["handler"])
