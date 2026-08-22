"""Compaction trigger-attribution coverage.

The contract: EVERY compaction names WHY it fired — in the log line and in
the user-facing lifecycle status — whether it was fired by the user or by an
automatic arm.

These are structural guards rather than snapshots. The vocabulary is derived
FROM THE PRODUCERS by scanning source for ``trigger_reason=`` literals, so a
new trigger added without a matching clause fails here instead of silently
rendering an empty reason. A hand-maintained list would drift the moment
someone adds an arm, which is exactly the failure being guarded against.
"""

import logging
import os
import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agent import conversation_compression
from agent.conversation_compression import (
    COMPACTION_STATUS,
    MANUAL_TRIGGER_REASON,
    compaction_reason_clause,
)

REPO = Path(__file__).resolve().parents[2]

# Files that may legitimately pass trigger_reason literals into the core.
# 🔴 RECURSIVE on purpose (review finding, PR #91158): `agent/*.py` alone lets a
# call site in an existing subpackage (agent/monitoring/, gateway/…) escape
# attribution coverage entirely while FEELING protected. A guard test below
# asserts the scan actually reaches at least one subdirectory file so this
# cannot silently regress back to flat globs.
_PRODUCER_GLOBS = [
    "agent/**/*.py",
    "gateway/**/*.py",
    "tui_gateway/**/*.py",
    "acp_adapter/**/*.py",
    "cli.py",
    "run_agent.py",
]

# A producer passes either a bare string ("threshold") or one of the module's
# exported constants (MANUAL_TRIGGER_REASON). Both forms are real call sites,
# so the scan resolves both — matching only quoted literals would silently
# miss every surface that uses the constant, and the coverage test would then
# pass without ever having checked them.
# NOTE (load-bearing convention): _LITERAL_RE matches lowercase snake_case only.
# trigger_reason literals MUST be lowercase snake_case ("overflow_413", never
# "Overflow413") — a differently-cased literal would slip this scan entirely.
_LITERAL_RE = re.compile(r"""trigger_reason=["']([a-z_0-9]+)["']""")
_CONSTANT_RE = re.compile(r"trigger_reason=([A-Z][A-Z_0-9]*)")

# The manual /compress surfaces, each of which must attribute its compaction.
_MANUAL_SURFACES = {
    "gateway slash": Path("gateway") / "slash_commands.py",
    "tui": Path("tui_gateway") / "server.py",
    "cli": Path("cli.py"),
    "acp": Path("acp_adapter") / "server.py",
}


def _resolve_constant(name: str) -> str | None:
    value = getattr(conversation_compression, name, None)
    return value if isinstance(value, str) else None


def _collect_producer_reasons() -> set[str]:
    reasons: set[str] = set()
    for pattern in _PRODUCER_GLOBS:
        for path in REPO.glob(pattern):
            text = path.read_text(errors="replace")
            reasons.update(_LITERAL_RE.findall(text))
            for name in _CONSTANT_RE.findall(text):
                resolved = _resolve_constant(name)
                assert resolved, (
                    f"{path.name} passes trigger_reason={name}, which is not a "
                    f"string constant in agent.conversation_compression — the "
                    f"coverage scan cannot verify it renders a clause"
                )
                reasons.add(resolved)
    return reasons


def test_scan_finds_the_known_producers():
    """Guard the guard: a broken glob must not green the coverage test below.

    Without this, a typo in _PRODUCER_GLOBS would make the scan return an
    empty set and every assertion over it would pass vacuously.
    """
    reasons = _collect_producer_reasons()
    assert "threshold" in reasons
    assert "session_hygiene" in reasons
    assert MANUAL_TRIGGER_REASON in reasons


def test_scan_reaches_subpackages():
    """The globs must be RECURSIVE: a producer in agent/<subpkg>/x.py has to be
    inside the scan, or a future call site there escapes attribution coverage
    while feeling protected (review finding on PR #91158). Assert the file walk
    actually visits at least one file below a subdirectory of a scanned root."""
    seen_subdir_file = False
    for pattern in _PRODUCER_GLOBS:
        for path in REPO.glob(pattern):
            rel = path.relative_to(REPO)
            if len(rel.parts) > 2:  # e.g. agent/monitoring/foo.py
                seen_subdir_file = True
                break
        if seen_subdir_file:
            break
    assert seen_subdir_file, (
        "_PRODUCER_GLOBS never reached a subpackage file — globs are no longer "
        "recursive, so subpackage call sites are invisible to this coverage test"
    )


def test_every_produced_reason_renders_a_clause():
    """Producer/renderer lockstep: no trigger may render an empty clause."""
    silent = {
        reason: compaction_reason_clause(reason)
        for reason in sorted(_collect_producer_reasons())
        if not compaction_reason_clause(reason).strip()
    }
    assert not silent, (
        f"trigger_reason values that render NO clause (the 'compacted with no "
        f"reason stated' bug): {sorted(silent)}. Add an arm to "
        f"_TRIGGER_REASON_CLAUSES in agent/conversation_compression.py."
    )


def test_every_produced_reason_reads_as_prose():
    """A clause is an explanation, not an echo of the internal label.

    ``(trigger: pre_api_pressure)`` is the never-silent fallback for an
    UNKNOWN reason; a reason a producer actually ships should have been given
    real wording.
    """
    echoed = [
        reason
        for reason in sorted(_collect_producer_reasons())
        if compaction_reason_clause(reason) == f" (trigger: {reason})"
    ]
    assert not echoed, (
        f"these shipped triggers fall through to the raw-label fallback and "
        f"read as internal jargon to a user: {echoed}"
    )


def test_unknown_reason_is_never_silent():
    """An unrecognized reason renders its raw label rather than nothing.

    Silence is the bug; a raw label is merely ugly. This keeps the failure
    mode of a future un-mapped trigger legible instead of invisible.
    """
    clause = compaction_reason_clause("some_future_trigger")
    assert clause.strip()
    assert "some_future_trigger" in clause


@pytest.mark.parametrize("empty", [None, "", "   "])
def test_absent_reason_renders_nothing(empty):
    """No reason at all is the one case that renders no clause.

    Callers that pass nothing are warned about in the log (UNATTRIBUTED);
    the status line must not grow a dangling empty parenthesis.
    """
    assert compaction_reason_clause(empty) == ""


def test_status_line_carries_the_reason():
    """The user-facing lifecycle status names the trigger.

    Asserts the relationship (status starts with the base wording and gains
    the clause), not a frozen string, so rewording either part stays free.
    """
    line = COMPACTION_STATUS + compaction_reason_clause("threshold")
    assert line.startswith(COMPACTION_STATUS)
    assert line != COMPACTION_STATUS
    assert compaction_reason_clause("threshold") in line


def test_every_manual_surface_passes_the_manual_label():
    """Each manual /compress path attributes its compaction.

    Source-pinned: a manual surface that calls _compress_context without a
    trigger_reason logs UNATTRIBUTED and renders no reason, which is the
    original reported symptom ("why did you compress here, there was no
    reason stated" about a /compress the user ran themselves).
    """
    for name, relative in _MANUAL_SURFACES.items():
        text = (REPO / relative).read_text(errors="replace")
        assert "trigger_reason=MANUAL_TRIGGER_REASON" in text, (
            f"{name} surface ({relative}) has a manual /compress path that "
            f"does not pass trigger_reason=MANUAL_TRIGGER_REASON — its "
            f"compressions log trigger=UNATTRIBUTED and name no reason"
        )


# ---------------------------------------------------------------------------
# E2E: the real compress_context path, not a stand-in.
# ---------------------------------------------------------------------------

# Distinguishes "caller passed nothing" from "caller passed None" — the former
# is the wiring-defect case the UNATTRIBUTED warning exists to catch.
_ABSENT = object()


def _build_agent(session_db, session_id: str):
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        agent = AIAgent(
            api_key="test-key",
            base_url="https://openrouter.ai/api/v1",
            model="test/model",
            quiet_mode=True,
            session_db=session_db,
            session_id=session_id,
            skip_context_files=True,
            skip_memory=True,
        )

    compressor = MagicMock()
    compressor.compress.return_value = [
        {"role": "user", "content": "[CONTEXT COMPACTION] summary"},
        {"role": "user", "content": "tail"},
    ]
    compressor.compression_count = 1
    compressor.last_prompt_tokens = 0
    compressor.last_completion_tokens = 0
    compressor._last_summary_error = None
    compressor._last_compress_aborted = False
    compressor._last_aux_model_failure_model = None
    compressor._last_aux_model_failure_error = None
    # No custom formatter and no opt-out: this agent takes the default
    # lifecycle status so the test observes what a real user would see.
    del compressor.get_automatic_compaction_status_message
    compressor.emit_automatic_compaction_status = True
    agent.context_compressor = compressor
    agent.compression_enabled = False  # skip the aux-provider feasibility probe
    return agent


def _run_compression(tmp_path, caplog, *, trigger_reason):
    """Drive a real compression and return (statuses, log_text)."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    sid = "TRIGGER_ATTRIBUTION_SESSION"
    db.create_session(sid, source="cli")
    agent = _build_agent(db, sid)

    statuses: list[str] = []
    agent.status_callback = lambda _kind, text: statuses.append(text)

    messages = [{"role": "user", "content": f"m{i}"} for i in range(20)]
    kwargs = {} if trigger_reason is _ABSENT else {"trigger_reason": trigger_reason}
    with caplog.at_level(logging.INFO, logger="agent.conversation_compression"):
        agent._compress_context(messages, "sys", approx_tokens=120_000, **kwargs)
    return statuses, caplog.text


def test_e2e_automatic_compaction_names_its_trigger(tmp_path, caplog):
    """A real automatic compression attributes itself in log AND status."""
    statuses, log_text = _run_compression(
        tmp_path, caplog, trigger_reason="threshold"
    )

    assert "context compression started:" in log_text
    assert "trigger=threshold" in log_text
    assert "UNATTRIBUTED" not in log_text

    compaction_statuses = [s for s in statuses if COMPACTION_STATUS in s]
    assert compaction_statuses, f"no compaction status emitted; saw {statuses}"
    assert compaction_reason_clause("threshold") in compaction_statuses[0]


def test_e2e_manual_compaction_names_itself_as_manual(tmp_path, caplog):
    """A user-fired /compress says so — the originally reported gap."""
    statuses, log_text = _run_compression(
        tmp_path, caplog, trigger_reason=MANUAL_TRIGGER_REASON
    )

    assert f"trigger={MANUAL_TRIGGER_REASON}" in log_text
    compaction_statuses = [s for s in statuses if COMPACTION_STATUS in s]
    assert compaction_statuses, f"no compaction status emitted; saw {statuses}"
    # The category word itself must appear: "(you ran /compress)" alone
    # requires the reader to infer "therefore manual".
    assert "manual" in compaction_statuses[0]


def test_e2e_unattributed_caller_is_logged_loudly(tmp_path, caplog):
    """A call site that passes no reason is a wiring defect — say so.

    Silence here is what let an unattributed arm ship unnoticed; the warning
    is the signal that a new call site was added without attribution.
    """
    statuses, log_text = _run_compression(tmp_path, caplog, trigger_reason=_ABSENT)

    assert "trigger=UNATTRIBUTED" in log_text
    assert "every compaction must name its arm" in log_text
    # No reason known => no clause, and specifically no dangling parenthesis.
    compaction_statuses = [s for s in statuses if COMPACTION_STATUS in s]
    assert compaction_statuses, f"no compaction status emitted; saw {statuses}"
    assert compaction_statuses[0] == COMPACTION_STATUS

