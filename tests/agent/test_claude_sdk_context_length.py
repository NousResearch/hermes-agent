"""The footer must size context against the CLI's window, not the model's.

Hermes resolves context_length from model metadata -- 1,000,000 for
claude-opus-5. On this lane the real window is whatever the spawned CLI runs
with, and `agent.claude_agent_sdk.env.CLAUDE_CODE_AUTO_COMPACT_WINDOW` can cut
it to a fraction. Measured 2026-08-16 with the window at 300,000: the runtime
footer read 16% while the CLI sat at 53% of its real window and one turn from
autocompacting. Worse than a wrong number, the scale was structurally squashed
-- compaction fires at 267,000, which against a denominator of 1,000,000 is
27%, so the gauge could never have read above roughly a quarter full.

The same value sizes gateway session hygiene, so this is not display-only.
"""

from __future__ import annotations

import pytest

from agent import claude_sdk_runtime as R


class _Compressor:
    """Mirrors the real setter's contract: assignment re-derives the budgets."""

    def __init__(self, context_length=1_000_000):
        self.context_length = context_length
        self.threshold_invalidated = 0

    def __setattr__(self, name, value):
        if name == "context_length" and "context_length" in self.__dict__:
            if value != self.__dict__["context_length"]:
                self.__dict__["threshold_invalidated"] = (
                    self.__dict__.get("threshold_invalidated", 0) + 1
                )
        super().__setattr__(name, value)


class _Session:
    def __init__(self, usage, raises=False):
        self._usage = usage
        self._raises = raises
        self.calls = 0

    def context_usage(self):
        self.calls += 1
        if self._raises:
            raise RuntimeError("CLI gone")
        return self._usage


class _Agent:
    def __init__(self, session, compressor=None):
        self._claude_sdk_session = session
        self.context_compressor = (
            compressor if compressor is not None else _Compressor()
        )


def test_cli_max_tokens_replaces_the_metadata_window():
    agent = _Agent(_Session({"maxTokens": 300_000, "totalTokens": 42}))

    R._sync_context_length_from_cli(agent)

    assert agent.context_compressor.context_length == 300_000


def test_assignment_invalidates_the_derived_threshold():
    """Correcting the window is pointless if the threshold keeps the old one."""
    agent = _Agent(_Session({"maxTokens": 300_000}))

    R._sync_context_length_from_cli(agent)

    assert agent.context_compressor.threshold_invalidated == 1


def test_queried_once_per_session_then_cached():
    """context_usage() is a real round-trip to the child, on the turn path."""
    session = _Session({"maxTokens": 300_000})
    agent = _Agent(session)

    for _ in range(5):
        R._sync_context_length_from_cli(agent)

    assert session.calls == 1
    assert agent.context_compressor.context_length == 300_000


def test_a_new_session_is_resynced():
    """A respawned CLI can carry a different window — the cache is per session."""
    agent = _Agent(_Session({"maxTokens": 300_000}))
    R._sync_context_length_from_cli(agent)

    agent._claude_sdk_session = _Session({"maxTokens": 500_000})
    R._sync_context_length_from_cli(agent)

    assert agent.context_compressor.context_length == 500_000


@pytest.mark.parametrize(
    "usage",
    [
        None,
        {},
        {"maxTokens": 0},
        {"maxTokens": 1},
        {"maxTokens": 63_999},
        {"maxTokens": -1},
        {"maxTokens": 300_000.5},
        {"maxTokens": "300000.5"},
        {"maxTokens": "not-a-number"},
        {"totalTokens": 500},
    ],
)
def test_unusable_reports_keep_the_metadata_value(usage):
    """Degrade to today's behaviour rather than zeroing the denominator.

    A 0 here would make the footer divide by zero (it guards, and renders
    nothing) and would collapse every hygiene threshold to 0.
    """
    agent = _Agent(_Session(usage))

    R._sync_context_length_from_cli(agent)

    assert agent.context_compressor.context_length == 1_000_000


def test_a_failing_query_is_contained():
    """This runs on the turn path — a dead CLI must not raise into it."""
    agent = _Agent(_Session(None, raises=True))

    R._sync_context_length_from_cli(agent)

    assert agent.context_compressor.context_length == 1_000_000


def test_a_failed_attempt_is_not_retried_every_turn():
    """Retrying costs the query timeout per turn to relearn the same answer."""
    session = _Session(None, raises=True)
    agent = _Agent(session)

    for _ in range(4):
        R._sync_context_length_from_cli(agent)

    assert session.calls == 1


def test_matching_window_is_a_no_op():
    """No spurious invalidation when metadata already agrees with the CLI."""
    agent = _Agent(_Session({"maxTokens": 1_000_000}))

    R._sync_context_length_from_cli(agent)

    assert agent.context_compressor.threshold_invalidated == 0


def test_missing_session_or_compressor_is_safe():
    class _Bare:
        pass

    R._sync_context_length_from_cli(_Bare())
    R._sync_context_length_from_cli(_Agent(None))

    agent = _Agent(_Session({"maxTokens": 300_000}))
    agent.context_compressor = None
    R._sync_context_length_from_cli(agent)


def test_against_the_real_compressor():
    """The stand-in above encodes an ASSUMPTION about the setter — verify it.

    A mock that agrees with itself proves nothing about production. The real
    setter invalidates the derived budgets AND re-floors threshold_percent for
    the new window (0.5 -> 0.75 at 300k), so both the window and the threshold
    genuinely track the CLI.
    """
    from agent.context_compressor import ContextCompressor

    compressor = ContextCompressor(
        model="claude-opus-5",
        config_context_length=1_000_000,
        api_mode="claude_agent_sdk",
        quiet_mode=True,
    )
    before = compressor.threshold_tokens
    agent = _Agent(_Session({"maxTokens": 300_000}), compressor=compressor)

    R._sync_context_length_from_cli(agent)

    assert compressor.context_length == 300_000
    assert compressor.threshold_tokens != before
    # The threshold must land inside the new window, not the old one — that is
    # the bug this fixes: a threshold above maxTokens is unreachable.
    assert compressor.threshold_tokens < 300_000


def test_the_correction_is_logged(caplog):
    """Silent denominator swaps are how the wrong one survived this long."""
    agent = _Agent(_Session({"maxTokens": 300_000}))

    with caplog.at_level("INFO", logger="agent.claude_sdk_runtime"):
        R._sync_context_length_from_cli(agent)

    assert "300000" in caplog.text.replace(",", "")
    assert "1000000" in caplog.text.replace(",", "")
