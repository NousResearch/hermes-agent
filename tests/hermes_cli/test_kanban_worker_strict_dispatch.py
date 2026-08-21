"""Tests for STRICT-READONLY Kanban worker dispatch semantics.

Regression matrix coverage:

S1   strict capability propagates objective -> task -> dispatcher env
     (dispatch + env component — companion field test covers persistence)
S4   non-Kanban CLI unchanged (worker-only env vars)
S15  terminal disabled for strict worker
S16  execute_code disabled while Kanban lifecycle tools remain available
S18  two workspace artifacts promote to attachments (trusted Kanban
     internal copy path — store_attachment_bytes — is unaffected by
     the strict gate)
S20  background review suppression unchanged via
     HERMES_SESSION_SOURCE=kanban / accepted Scope A
S21  provider/model/reasoning propagation unchanged

These tests use ONLY stdlib + hermes internals; no live subprocess spawn
and no live network calls. ``subprocess.Popen`` is monkey-patched so we
can observe ``cmd`` / ``env`` / ``cwd`` without booting a worker.

The S15/S16 behavioural assertions are expressed against the worker's
effective ``--toolsets`` payload (the supported CLI transport) — NOT
against the unsupported ``--disabled-toolsets`` flag the real CLI
rejects with ``invalid choice: 'terminal,code_execution'``. The same
boundary check that crashed the V2 real canary is replayed against the
real ``hermes_cli._parser`` so any future regression that re-introduces
the malformed flag (or breaks the supported path) fails here before
reaching production.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import hermes_state
from hermes_cli import kanban_db as kb


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Isolated HERMES_HOME with a kanban DB and a workspace directory."""
    home = tmp_path / ".hermes"
    home.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("TERMINAL_CWD", str(workspace))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture
def captured_spawn(monkeypatch):
    """Capture ``cmd``, ``env``, ``cwd`` passed to ``subprocess.Popen``."""
    captured: dict = {}

    class _FakePopen:
        def __init__(self, cmd, **kwargs):
            captured["cmd"] = list(cmd)
            captured["env"] = dict(kwargs.get("env") or {})
            captured["cwd"] = kwargs.get("cwd")
            captured["stdin"] = kwargs.get("stdin")
            captured["stdout"] = kwargs.get("stdout")
            captured["stderr"] = kwargs.get("stderr")
            self.pid = 99999

    # Patch the ``subprocess`` module that ``_default_spawn`` imports
    # inside its own function body (it does ``import subprocess``).
    import subprocess as _sp
    monkeypatch.setattr(_sp, "Popen", _FakePopen)
    return captured


def _create_and_get(conn, *, strict_readonly: bool, assignee: str = "coder"):
    task_id = kb.create_task(
        conn,
        title=f"task-strict={strict_readonly}",
        assignee=assignee,
        created_by="user",
        workspace_kind="scratch",
        initial_status="running",
        strict_readonly=strict_readonly,
    )
    task = kb.get_task(conn, task_id)
    assert task is not None
    return task


# ---------------------------------------------------------------------------
# S1 — env propagation
# ---------------------------------------------------------------------------


def test_strict_worker_receives_env_var(kanban_home, captured_spawn):
    """S1: dispatch of strict task sets HERMES_KANBAN_STRICT_READONLY=1."""
    workspace = str(kanban_home / "ws")
    with kb.connect_closing() as conn:
        task = _create_and_get(conn, strict_readonly=True)
        kb._default_spawn(task, workspace)  # noqa: SLF001
    assert captured_spawn["env"].get("HERMES_KANBAN_STRICT_READONLY") == "1"


def test_ordinary_worker_does_not_receive_env_var(kanban_home, captured_spawn):
    """S1 (negative half): ordinary writable task does NOT see the env var."""
    workspace = str(kanban_home / "ws")
    with kb.connect_closing() as conn:
        task = _create_and_get(conn, strict_readonly=False)
        kb._default_spawn(task, workspace)  # noqa: SLF001
    assert "HERMES_KANBAN_STRICT_READONLY" not in captured_spawn["env"]


# ---------------------------------------------------------------------------
# S4 — non-Kanban CLI unchanged
# ---------------------------------------------------------------------------


def test_strict_env_var_is_only_set_when_task_opted_in(kanban_home, captured_spawn):
    """S4: env var is opt-in per task, not ambient."""
    workspace = str(kanban_home / "ws")
    with kb.connect_closing() as conn:
        strict_task = _create_and_get(conn, strict_readonly=True)
        writable_task = _create_and_get(conn, strict_readonly=False)
        kb._default_spawn(strict_task, workspace)  # noqa: SLF001
        strict_env = captured_spawn["env"]
        kb._default_spawn(writable_task, workspace)  # noqa: SLF001
        writable_env = captured_spawn["env"]
    assert strict_env.get("HERMES_KANBAN_STRICT_READONLY") == "1"
    assert "HERMES_KANBAN_STRICT_READONLY" not in writable_env


# ---------------------------------------------------------------------------
# S15 / S16 — strict surface expressed via the supported --toolsets transport
# ---------------------------------------------------------------------------


def _toolsets_from_cmd(cmd):
    """Return the comma-separated value passed to ``--toolsets``, or None.

    Mirrors the dispatcher's single emission contract: ``--toolsets`` is
    the supported transport; ``--disabled-toolsets`` does not exist in the
    real CLI parser.
    """
    pairs = list(zip(cmd, cmd[1:]))
    for flag, value in pairs:
        if flag == "--toolsets":
            return value
    return None


def test_strict_worker_subtracts_terminal_and_code_execution(kanban_home, captured_spawn):
    """S15+S16: strict worker argv contains ``--toolsets`` whose payload has
    ``terminal`` and ``code_execution`` removed from the assignee's
    resolved CLI toolset surface. Uses the supported CLI transport, not
    the rejected ``--disabled-toolsets`` flag that crashed the V2 canary.
    """
    workspace = str(kanban_home / "ws")
    with kb.connect_closing() as conn:
        task = _create_and_get(conn, strict_readonly=True)
        kb._default_spawn(task, workspace)  # noqa: SLF001
    cmd = captured_spawn["cmd"]
    toolsets_value = _toolsets_from_cmd(cmd)
    assert toolsets_value is not None, (
        f"--toolsets missing from cmd={cmd}; cannot express STRICT surface"
    )
    emitted = {s.strip() for s in toolsets_value.split(",") if s.strip()}
    assert "terminal" not in emitted, (
        f"STRICT worker must not have terminal in --toolsets, got {emitted}"
    )
    assert "code_execution" not in emitted, (
        f"STRICT worker must not have code_execution in --toolsets, got {emitted}"
    )
    # The malformed runtime path that crashed the V2 canary must never
    # re-appear — see the real-CLI-boundary regression test below for the
    # parser-level guard.
    assert "--disabled-toolsets" not in cmd, (
        f"STRICT worker must not emit --disabled-toolsets "
        f"(unsupported in real Hermes CLI), got cmd={cmd}"
    )


def test_ordinary_worker_keeps_full_toolsets_payload(kanban_home, captured_spawn):
    """Ordinary (non-strict) workers emit ``--toolsets`` with the assignee's
    full resolved CLI toolset surface — ``terminal`` and ``code_execution``
    must remain when the strict capability is OFF."""
    workspace = str(kanban_home / "ws")
    with kb.connect_closing() as conn:
        task = _create_and_get(conn, strict_readonly=False)
        kb._default_spawn(task, workspace)  # noqa: SLF001
    cmd = captured_spawn["cmd"]
    toolsets_value = _toolsets_from_cmd(cmd)
    assert toolsets_value is not None, (
        f"ordinary worker missing --toolsets pin in cmd={cmd}"
    )
    emitted = {s.strip() for s in toolsets_value.split(",") if s.strip()}
    # The default ``kanban`` fixture resolves to a non-empty toolset that
    # includes the worker-relevant surface; the strict filter must NOT
    # have run, so the full set is preserved.
    assert "terminal" in emitted, (
        f"ordinary worker must keep terminal; got {emitted}"
    )
    assert "code_execution" in emitted, (
        f"ordinary worker must keep code_execution; got {emitted}"
    )
    assert "--disabled-toolsets" not in cmd, (
        f"ordinary worker must not emit --disabled-toolsets either, "
        f"got cmd={cmd}"
    )


def test_strict_worker_kanban_lifecycle_anchor_pinned(kanban_home, captured_spawn):
    """S16 (lifecycle preservation): HERMES_KANBAN_TASK remains pinned so
    ``model_tools._compute_tool_definitions`` re-appends the ``kanban``
    toolset regardless of profile config."""
    workspace = str(kanban_home / "ws")
    with kb.connect_closing() as conn:
        task = _create_and_get(conn, strict_readonly=True)
        task_id = task.id
        kb._default_spawn(task, workspace)  # noqa: SLF001
    env = captured_spawn["env"]
    assert env.get("HERMES_KANBAN_TASK") == task_id
    # Session source still tagged kanban so Scope-A skip_background_review
    # continues to apply.
    assert env.get("HERMES_SESSION_SOURCE") == "kanban"


# ---------------------------------------------------------------------------
# S20 — background review suppression unchanged (Scope A inheritance)
# ---------------------------------------------------------------------------


def test_strict_worker_inherits_scope_a_session_source(kanban_home, captured_spawn):
    """S20: HERMES_SESSION_SOURCE=kanban is still set, so the
    cli_agent_setup_mixin wires skip_background_review=True."""
    workspace = str(kanban_home / "ws")
    with kb.connect_closing() as conn:
        task = _create_and_get(conn, strict_readonly=True)
        kb._default_spawn(task, workspace)  # noqa: SLF001
    assert captured_spawn["env"].get("HERMES_SESSION_SOURCE") == "kanban"


# ---------------------------------------------------------------------------
# S21 — provider/model/reasoning propagation unchanged
# ---------------------------------------------------------------------------


def test_strict_worker_propagates_model_provider_reasoning(kanban_home, captured_spawn):
    """S21: strict worker still passes -m / --provider / --reasoning."""
    workspace = str(kanban_home / "ws")
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="strict-with-overrides",
            assignee="coder",
            created_by="user",
            workspace_kind="scratch",
            initial_status="running",
            strict_readonly=True,
            model_override="some-model",
            provider_override="some-provider",
            reasoning_effort="high",
        )
        task = kb.get_task(conn, task_id)
        assert task is not None
        kb._default_spawn(task, workspace)  # noqa: SLF001
    cmd = captured_spawn["cmd"]
    pairs = list(zip(cmd, cmd[1:]))

    def _pair(flag):
        for f, v in pairs:
            if f == flag:
                return v
        return None

    assert _pair("-m") == "some-model"
    assert _pair("--provider") == "some-provider"
    assert _pair("--reasoning") == "high"
    # Strict env still exported alongside overrides.
    assert captured_spawn["env"].get("HERMES_KANBAN_STRICT_READONLY") == "1"


# ---------------------------------------------------------------------------
# S18 — trusted artifact promotion path (unaffected by strict gate)
# ---------------------------------------------------------------------------


def test_two_artifacts_promote_to_task_attachments(kanban_home):
    """S18: ``store_attachment_bytes`` is the trusted Kanban internal-copy
    path used by ``kanban_attach`` / dashboard / CLI. It writes to
    ``task_attachments_dir(task_id)`` directly and is NOT a model-tool
    surface, so it is naturally unaffected by the strict gate. Two
    attachments persist as expected."""
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title="strict-attachment-test",
            assignee="coder",
            created_by="user",
            workspace_kind="scratch",
            initial_status="running",
            strict_readonly=True,
        )
        kb.store_attachment_bytes(
            conn,
            task_id,
            "out_a.txt",
            b"AAA",
            uploaded_by="strict-worker",
        )
        kb.store_attachment_bytes(
            conn,
            task_id,
            "out_b.txt",
            b"BBB",
            uploaded_by="strict-worker",
        )
        attachments = kb.list_attachments(conn, task_id)

    assert len(attachments) == 2
    filenames = sorted(a.filename for a in attachments)
    assert filenames == ["out_a.txt", "out_b.txt"]
    # Both attachment files exist on disk under task_attachments_dir.
    for a in attachments:
        full = kanban_home / "kanban" / a.stored_path
        assert full.is_file(), f"attachment missing: {full}"


# ---------------------------------------------------------------------------
# Real Hermes CLI parser / process-boundary regression
# ---------------------------------------------------------------------------
#
# The V2 real canary failed with::
#
#     hermes: error: argument command:
#     invalid choice: 'terminal,code_execution'
#
# because the dispatcher emitted ``--disabled-toolsets terminal,code_execution``
# and the real Hermes CLI does NOT accept that flag at any layer (top-level
# or ``chat`` subparser). This regression replays the exact boundary check
# the dispatcher crash tripped on: feed the actual generated argv into
# the real ``hermes_cli._parser.build_top_level_parser`` and assert that
# it parses cleanly. A second sub-test feeds the synthetic pre-repair
# malformed argv into the same parser and asserts it is rejected with
# ``invalid choice`` — exactly the production crash signature.
#
# The test is *behavioural*: ``--disabled-toolsets`` absent alone is not
# sufficient, because a future regression could drop the strict filter
# entirely and still satisfy that trivial assertion. What the boundary
# demands is that the generated argv actually crosses the real
# ``argparse`` boundary, and the negative half proves the parser does
# reject the malformed variant.

# Local alias for the profile-relative argv prefix the real CLI uses
# after ``_apply_profile_override`` resolves the worker profile.
_STRICT_FLAG_PREFIX_PROFILE = ["-p", "coder", "--cli", "--accept-hooks"]


def _strip_profile_bootstrap_flags(cmd):
    """Drop the leading argv elements the dispatcher injects to bootstrap
    the profile before the shared parser sees the flags.

    ``_apply_profile_override`` and the ``-p <assignee>`` selection live
    outside ``build_top_level_parser`` — they are wired up by
    ``hermes_cli.main``. The pattern below mirrors the assertion already
    used by ``test_kanban_worker_spawn_toolsets`` for the same reason.
    """
    # The dispatcher emits: ``hermes -p <profile> --cli --accept-hooks [per-task flags...] chat -q <prompt>``
    # ``-p`` and its value ride together; ``--cli`` and ``--accept-hooks``
    # are flag-only.
    assert cmd[1:3] == ["-p", "coder"], (
        f"unexpected profile bootstrap in cmd={cmd}"
    )
    assert "--cli" in cmd[:5]
    assert "--accept-hooks" in cmd[:6]
    return cmd[3:]


def test_strict_worker_argv_parses_through_real_cli_parser(
    kanban_home, captured_spawn
):
    """The exact argv ``_default_spawn`` generates for a STRICT task must
    cross the *real* ``hermes_cli._parser`` boundary without raising
    ``SystemExit``. This is the regression the V2 canary proved was
    missing: a unit test that observed ``--disabled-toolsets`` in the
    generated argv did not catch that the real CLI parser rejects the
    flag with ``invalid choice`` and crashes the worker before session
    init.

    Concretely:

    1. Generate the strict-worker argv by invoking ``_default_spawn``.
    2. Strip the profile-bootstrap prefix the dispatcher adds (mirrors
       how ``test_kanban_worker_spawn_toolsets`` exercises the parser).
    3. Hand the remainder to the real ``build_top_level_parser``.
    4. Assert ``parser.parse_args(...)`` returns cleanly and resolves
       ``args.command == "chat"``.
    5. Assert the parsed ``args.toolsets`` payload has neither
       ``terminal`` nor ``code_execution`` (the S15/S16 invariant,
       expressed against the real parser's view of the world).
    """
    from hermes_cli._parser import build_top_level_parser

    workspace = str(kanban_home / "ws")
    with kb.connect_closing() as conn:
        task = _create_and_get(conn, strict_readonly=True)
        kb._default_spawn(task, workspace)  # noqa: SLF001
    cmd = captured_spawn["cmd"]
    parser, _subparsers, _chat_parser = build_top_level_parser()

    # Crossing the real argparse boundary is the assertion. ``parse_args``
    # raises ``SystemExit(2)`` on an unknown flag or invalid value —
    # exactly the production crash shape the canary logged.
    tail = _strip_profile_bootstrap_flags(cmd)
    try:
        args = parser.parse_args(tail)
    except SystemExit as exc:
        raise AssertionError(
            f"strict-worker argv was REJECTED by the real CLI parser "
            f"(exit={exc.code!r}); the dispatcher emitted malformed flags. "
            f"cmd={cmd}"
        )

    assert args.command == "chat", (
        f"strict-worker argv did not resolve to chat subcommand; "
        f"got command={args.command!r}, cmd={cmd}"
    )

    # The S15/S16 behavioural contract, validated against the real parser.
    emitted = {
        s.strip() for s in (args.toolsets or "").split(",") if s.strip()
    }
    assert "terminal" not in emitted, (
        f"STRICT worker must not have terminal in parsed --toolsets, "
        f"got {emitted}"
    )
    assert "code_execution" not in emitted, (
        f"STRICT worker must not have code_execution in parsed --toolsets, "
        f"got {emitted}"
    )


def test_pre_repair_disabled_toolsets_argv_is_rejected_by_real_cli_parser(
    kanban_home, monkeypatch, captured_spawn
):
    """Negative companion to the boundary regression above.

    Construct the exact malformed argv the V2 dispatcher emitted before
    the repair — ``--disabled-toolsets terminal,code_execution`` between
    the strict flag prefix and ``chat -q <prompt>`` — and feed it to the
    real ``hermes_cli._parser``. Assert that ``parse_args`` raises
    ``SystemExit(2)`` with a usage message that names the bad choice.

    This proves two things at once:

    * The parser truly does reject ``--disabled-toolsets`` (so any future
      regression that re-introduces the flag trips this test, not a
      live canary).
    * The shape of the rejection matches the production crash signature
      observed in the canary log
      (``/home/jr-ubuntu/.hermes/kanban/boards/ai-os/logs/t_552bab31.log``):
      ``invalid choice: 'terminal,code_execution'``.

    Without this negative half the boundary regression would silently
    pass on a no-op refactor that strips ``--disabled-toolsets`` without
    restoring the strict tool surface. Both halves must hold.
    """
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, _chat_parser = build_top_level_parser()

    # Construct the synthetic pre-repair argv. The ``--disabled-toolsets``
    # flag is unknown to the parser, so argparse treats its value as a
    # positional candidate and rejects it on the ``command`` subparser
    # with the exact message shape the canary captured.
    synthetic_tail = [
        "--disabled-toolsets", "terminal,code_execution",
        "chat", "-q", "work kanban task t_synthetic",
    ]

    stderr_capture = []
    real_print_usage = parser.print_usage
    real_error = parser.error

    def _capture_error(message):
        stderr_capture.append(message)
        raise SystemExit(2)

    def _capture_usage(file=None):
        stderr_capture.append("__usage__")

    parser.error = _capture_error
    parser.print_usage = _capture_usage
    try:
        with pytest.raises(SystemExit) as exc_info:
            parser.parse_args(synthetic_tail)
    finally:
        parser.error = real_error
        parser.print_usage = real_print_usage

    assert exc_info.value.code == 2, (
        f"expected SystemExit(2) from real CLI parser on malformed argv; "
        f"got code={exc_info.value.code!r}"
    )
    combined = "\n".join(stderr_capture)
    assert "invalid choice" in combined, (
        f"parser error did not match the production canary signature; "
        f"got: {combined!r}"
    )
    assert "terminal,code_execution" in combined, (
        f"parser error did not name the rejected value; got: {combined!r}"
    )
