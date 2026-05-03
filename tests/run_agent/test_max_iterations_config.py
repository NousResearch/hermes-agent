"""Tests for agent.max_turns discoverability in iteration-exhaustion messages.

When the agent exhausts its iteration budget, the messages it shows to users
(both on console and in the returned final_response) should tell users HOW
to raise the limit — i.e., point at the agent.max_turns config key, the
HERMES_MAX_ITERATIONS env var, or the --max-turns CLI flag — not just
report that the limit was reached. Otherwise users hit the message and have
no path forward except trial-and-error in the docs.

The hint is centralized in `run_agent._MAX_ITERATIONS_HINT` so message-site
edits can't drift apart over time. These tests enforce that contract.

Issue context: https://github.com/NousResearch/hermes-agent/issues/18601
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import run_agent


HINT_KEY = "agent.max_turns"
HINT_FILE_PATH = "~/.hermes/config.yaml"
HINT_ENV_VAR = "HERMES_MAX_ITERATIONS"
HINT_CLI_FLAG = "--max-turns"
HINT_CONSTANT_NAME = "_MAX_ITERATIONS_HINT"

# Number of message sites we expect to use the hint constant in run_agent.py.
# Bumped from a flexible floor to an exact count after Flatline review pointed
# out that a floor lets future edits silently drop hints from message sites.
EXPECTED_USE_SITES = 8


def _source() -> str:
    """Return the run_agent.py source as a string."""
    return Path(run_agent.__file__).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Constant correctness
# ---------------------------------------------------------------------------

class TestHintConstant:
    """The centralized hint constant should mention all three precedence
    paths so a user who set the limit via --max-turns or
    HERMES_MAX_ITERATIONS isn't told to edit a config file that won't take
    effect (per `cli.py:2122-2131` precedence chain).
    """

    def test_constant_is_defined(self):
        assert hasattr(run_agent, HINT_CONSTANT_NAME), (
            f"expected run_agent.{HINT_CONSTANT_NAME} to be defined"
        )

    def test_constant_mentions_config_key(self):
        hint = getattr(run_agent, HINT_CONSTANT_NAME)
        assert HINT_KEY in hint, (
            f"hint constant should mention '{HINT_KEY}': {hint!r}"
        )

    def test_constant_mentions_config_file_path(self):
        hint = getattr(run_agent, HINT_CONSTANT_NAME)
        assert HINT_FILE_PATH in hint, (
            f"hint constant should mention '{HINT_FILE_PATH}': {hint!r}"
        )

    def test_constant_mentions_env_var(self):
        """A user who set HERMES_MAX_ITERATIONS shouldn't be told to edit
        the config file (env var has higher precedence)."""
        hint = getattr(run_agent, HINT_CONSTANT_NAME)
        assert HINT_ENV_VAR in hint, (
            f"hint constant should mention '{HINT_ENV_VAR}': {hint!r}"
        )

    def test_constant_mentions_cli_flag(self):
        """A user who passed --max-turns shouldn't be told to edit the
        config file (CLI flag has highest precedence)."""
        hint = getattr(run_agent, HINT_CONSTANT_NAME)
        assert HINT_CLI_FLAG in hint, (
            f"hint constant should mention '{HINT_CLI_FLAG}': {hint!r}"
        )


# ---------------------------------------------------------------------------
# Source-presence tests — every exhaustion site uses the constant
# ---------------------------------------------------------------------------

class TestExhaustionMessagesUseConstant:
    """Every user-facing iteration-exhaustion message should reference the
    hint constant. Counted via static source analysis so a future edit
    can't silently drop the hint from a message site.
    """

    def test_every_exhaustion_marker_has_hint_within_window(self):
        """For every iteration-exhaustion marker, the hint constant should
        appear within the next 6 lines (covers single-line and multi-line
        message-site formatting)."""
        src = _source()

        exhaustion_marker_re = re.compile(
            r"reached the iteration limit"
            r"|reached the maximum iterations"
            r"|Reached maximum iterations"
            r"|Iteration budget exhausted",
            re.IGNORECASE,
        )

        lines = src.splitlines()
        marker_positions = [
            i for i, line in enumerate(lines) if exhaustion_marker_re.search(line)
        ]

        assert marker_positions, (
            "expected at least one iteration-exhaustion marker in run_agent.py"
        )

        WINDOW = 6
        missing = []
        for idx in marker_positions:
            window = "\n".join(lines[idx:min(len(lines), idx + WINDOW + 1)])
            if HINT_CONSTANT_NAME not in window:
                missing.append((idx + 1, lines[idx].strip()))

        assert not missing, (
            f"the following iteration-exhaustion messages do not reference "
            f"'{HINT_CONSTANT_NAME}' within {WINDOW} following lines:\n"
            + "\n".join(f"  line {ln}: {text}" for ln, text in missing)
        )

    def test_constant_use_site_count_is_exact(self):
        """The constant should be referenced exactly at the known-good number
        of use sites (plus 1 definition line). Set as exact rather than a
        floor so a future edit that drops a use site fails this test —
        which is the whole point of having the test.

        If a legitimate new exhaustion message site is added, bump
        EXPECTED_USE_SITES at the top of this file and the test will pass.
        """
        src = _source()
        # +1 for the definition line itself.
        expected_total = EXPECTED_USE_SITES + 1
        actual_total = src.count(HINT_CONSTANT_NAME)
        assert actual_total == expected_total, (
            f"expected {expected_total} occurrences of '{HINT_CONSTANT_NAME}' "
            f"({EXPECTED_USE_SITES} use sites + 1 definition); found {actual_total}. "
            f"If you added or removed an exhaustion message site, update "
            f"EXPECTED_USE_SITES at the top of this file."
        )


# ---------------------------------------------------------------------------
# Behavioral tests — _handle_max_iterations exception path
# ---------------------------------------------------------------------------

class TestHandleMaxIterationsExceptionPath:
    """Exercise the _handle_max_iterations exception path: when the inner
    summary API call raises, the function should return a final_response
    string that includes the hint. We use a deliberately under-specified
    stub for `self` so the very first attribute access inside the `try:`
    block raises AttributeError, triggering the exception path without
    needing to mock the full API surface.
    """

    def test_exception_final_response_contains_hint(self, capsys):
        stub = type("Stub", (), {})()
        stub.max_iterations = 1

        result = run_agent.AIAgent._handle_max_iterations(
            stub, messages=[], api_call_count=1
        )

        # Exception-path final_response should embed the canonical hint.
        for marker in (HINT_KEY, HINT_FILE_PATH, HINT_ENV_VAR, HINT_CLI_FLAG):
            assert marker in result, (
                f"expected exception-path final_response to mention {marker!r}; "
                f"got: {result!r}"
            )

        # The console banner at the top of _handle_max_iterations should
        # also include the hint (printed before the API setup runs).
        captured = capsys.readouterr()
        assert HINT_KEY in captured.out, (
            f"expected console banner to mention '{HINT_KEY}'; "
            f"got stdout: {captured.out!r}"
        )
