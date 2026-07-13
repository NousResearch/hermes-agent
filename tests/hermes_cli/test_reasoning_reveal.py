"""Tests for the CLI `/reasoning reveal` post-hoc reasoning display.

`/reasoning reveal` shows the reasoning text captured from the last turn.
`/reasoning reveal --limit N` shows only the last N reasoning blocks.

These assert:
- Empty state when no reasoning is stored.
- Full replay when reasoning is stored and no --limit is given.
- Truncation to the last N blocks when --limit N is given.
- Rejection of invalid --limit values (0, negative, non-numeric, missing N).
- Rejection of unknown arguments after `reveal`.
- Registration in the command registry.
"""

from unittest.mock import patch

from hermes_cli.cli_commands_mixin import CLICommandsMixin


class _Stub(CLICommandsMixin):
    """Minimal carrier for the attributes _handle_reasoning_command reads."""

    def __init__(self):
        self.reasoning_config = None
        self.show_reasoning = True
        self.reasoning_full = False
        self.agent = None
        self._last_reasoning = None

    def _current_reasoning_callback(self):
        return None


def _run_reveal(stub, cmd="/reasoning reveal"):
    """Run the reveal command and return captured _cprint output as a string."""
    captured = []

    def fake_cprint(text):
        captured.append(text)

    # The handler imports _cprint from cli at call time
    with patch("cli._cprint", fake_cprint):
        stub._handle_reasoning_command(cmd)
    return "\n".join(captured)


# --- Empty state ---

def test_reveal_no_reasoning_stored():
    s = _Stub()
    s._last_reasoning = None
    out = _run_reveal(s)
    assert "No reasoning stored" in out


def test_reveal_empty_string_reasoning():
    s = _Stub()
    s._last_reasoning = ""
    out = _run_reveal(s)
    assert "No reasoning stored" in out


# --- Full replay ---

def test_reveal_shows_full_reasoning():
    s = _Stub()
    s._last_reasoning = "Step 1: Analyze\nStep 2: Plan\nStep 3: Execute"
    out = _run_reveal(s)
    assert "Step 1: Analyze" in out
    assert "Step 2: Plan" in out
    assert "Step 3: Execute" in out
    assert "showing last" not in out


def test_reveal_shows_multi_block_reasoning():
    s = _Stub()
    s._last_reasoning = (
        "Block 1 line 1\nBlock 1 line 2\n\n"
        "Block 2 line 1\n\n"
        "Block 3 line 1\nBlock 3 line 2"
    )
    out = _run_reveal(s)
    assert "Block 1" in out
    assert "Block 2" in out
    assert "Block 3" in out
    assert "showing last" not in out


# --- --limit N truncation ---

def test_reveal_limit_shows_last_n_blocks():
    s = _Stub()
    s._last_reasoning = "Block 1\n\nBlock 2\n\nBlock 3\n\nBlock 4\n\nBlock 5"
    out = _run_reveal(s, "/reasoning reveal --limit 2")
    assert "Block 4" in out
    assert "Block 5" in out
    assert "Block 1" not in out
    assert "Block 2" not in out
    assert "Block 3" not in out
    assert "showing last 2 of 5" in out


def test_reveal_limit_one_shows_single_block():
    s = _Stub()
    s._last_reasoning = "AAA\n\nBBB\n\nCCC"
    out = _run_reveal(s, "/reasoning reveal --limit 1")
    assert "CCC" in out
    assert "AAA" not in out
    assert "BBB" not in out
    assert "showing last 1 of 3" in out


def test_reveal_limit_equal_to_total():
    s = _Stub()
    s._last_reasoning = "AAA\n\nBBB\n\nCCC"
    out = _run_reveal(s, "/reasoning reveal --limit 3")
    assert "AAA" in out
    assert "BBB" in out
    assert "CCC" in out
    assert "showing last" not in out


def test_reveal_limit_exceeds_total():
    s = _Stub()
    s._last_reasoning = "AAA\n\nBBB"
    out = _run_reveal(s, "/reasoning reveal --limit 10")
    assert "AAA" in out
    assert "BBB" in out
    assert "showing last" not in out


# --- Separator handling ---

def test_reveal_limit_separator_dash_dash_dash():
    s = _Stub()
    s._last_reasoning = "AAA\n---\nBBB\n---\nCCC"
    out = _run_reveal(s, "/reasoning reveal --limit 1")
    assert "CCC" in out
    assert "AAA" not in out
    assert "BBB" not in out


# --- --limit validation ---

def test_reveal_limit_zero_rejected():
    s = _Stub()
    s._last_reasoning = "Some reasoning"
    out = _run_reveal(s, "/reasoning reveal --limit 0")
    # regex \d+ matches "0", then the <=0 check rejects it
    assert "positive" in out.lower() or "Invalid" in out


def test_reveal_limit_negative_rejected():
    s = _Stub()
    s._last_reasoning = "Some reasoning"
    # regex ^--limit\s+(\d+)$ won't match "-1" as a digit
    out = _run_reveal(s, "/reasoning reveal --limit -1")
    assert "Invalid" in out


def test_reveal_limit_non_numeric_rejected():
    s = _Stub()
    s._last_reasoning = "Some reasoning"
    out = _run_reveal(s, "/reasoning reveal --limit nope")
    assert "Invalid" in out


def test_reveal_limit_missing_n_rejected():
    s = _Stub()
    s._last_reasoning = "Some reasoning"
    out = _run_reveal(s, "/reasoning reveal --limit")
    assert "Invalid" in out


def test_reveal_unknown_arg_rejected():
    s = _Stub()
    s._last_reasoning = "Some reasoning"
    out = _run_reveal(s, "/reasoning reveal --foo")
    assert "Unknown argument" in out


# --- Registration in command registry ---

def test_reveal_not_in_shared_command_registry():
    """reveal is a CLI-only extension; it must NOT appear in the shared
    COMMAND_REGISTRY subcommands so the gateway and TUI don't advertise it
    on surfaces that don't implement post-hoc reasoning replay."""
    from hermes_cli.commands import COMMAND_REGISTRY
    reasoning_cmd = next(
        (c for c in COMMAND_REGISTRY if c.name == "reasoning"), None
    )
    assert reasoning_cmd is not None
    assert "reveal" not in reasoning_cmd.subcommands
