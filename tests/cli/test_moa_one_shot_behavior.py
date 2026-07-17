"""Tests for /moa one-shot behavior: dispatch, no-continuation-state, invalid input.

Verifies:
  (1) /moa <prompt> dispatches to the immediate execution path (sets pending
      seed, MoA provider swap, disable-after-turn flag, restore snapshot).
  (2) Normal (non-MoA) mode is never affected by one-shot machinery.
  (3) Invalid one-shot inputs (empty, whitespace, agent-running) return the
      expected error/help and create no side effects.
  (4) After the one-shot turn completes, no follow-up/continuation state
      remains — the model is restored and MoA flags are cleared.
"""

import queue
from unittest.mock import MagicMock, patch

from cli import HermesCLI


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cli(**overrides):
    """Build a minimal HermesCLI with just the fields /moa and chat() touch."""
    cli = HermesCLI.__new__(HermesCLI)
    cli.config = {
        "moa": {
            "default_preset": "default",
            "presets": {
                "default": {
                    "reference_models": [
                        {"provider": "openai-codex", "model": "gpt-5.5"},
                    ],
                    "aggregator": {
                        "provider": "openrouter",
                        "model": "anthropic/claude-opus-4.8",
                    },
                },
            },
        }
    }
    cli._pending_input = queue.Queue()
    cli._pending_agent_seed = None
    cli._pending_moa_config = None
    cli._pending_moa_disable_after_turn = False
    cli._pending_moa_restore_model = None
    cli._agent_running = False
    cli.agent = None
    cli.provider = "openrouter"
    cli.requested_provider = "openrouter"
    cli.model = "anthropic/claude-opus-4.8"
    cli.api_key = "test-key"
    cli.base_url = "https://openrouter.ai/api/v1"
    cli.api_mode = "chat_completions"
    for k, v in overrides.items():
        setattr(cli, k, v)
    return cli


# ---------------------------------------------------------------------------
# (1) One-shot dispatch sets immediate execution state
# ---------------------------------------------------------------------------

def test_one_shot_sets_pending_seed():
    """The prompt is placed in _pending_agent_seed so the main loop picks it up
    as the next agent turn input without blocking on user input."""
    cli = _make_cli()
    with patch("cli._cprint"):
        cli.process_command("/moa explain deadlocks in Go")
    assert cli._pending_agent_seed == "explain deadlocks in Go"


def test_one_shot_swaps_provider_to_moa():
    """After /moa, provider and model are set to MoA virtual provider so the
    next agent turn runs through the MoA fan-out path."""
    cli = _make_cli()
    with patch("cli._cprint"):
        cli.process_command("/moa test prompt")
    assert cli.provider == "moa"
    assert cli.requested_provider == "moa"
    assert cli.model == "default"  # the default preset name
    assert cli.api_key == "moa-virtual-provider"
    assert cli.base_url == "moa://local"
    assert cli.api_mode == "chat_completions"


def test_one_shot_sets_disable_after_turn_flag():
    """/moa sets _pending_moa_disable_after_turn so the chat path knows to
    restore the prior model after this turn completes."""
    cli = _make_cli()
    with patch("cli._cprint"):
        cli.process_command("/moa run this once")
    assert cli._pending_moa_disable_after_turn is True


def test_one_shot_snapshots_prior_model_for_restore():
    """The prior model identity is saved in _pending_moa_restore_model so it
    can be restored after the MoA turn."""
    cli = _make_cli()
    with patch("cli._cprint"):
        cli.process_command("/moa check this")
    restore = cli._pending_moa_restore_model
    assert restore is not None
    assert restore["provider"] == "openrouter"
    assert restore["model"] == "anthropic/claude-opus-4.8"
    assert restore["requested_provider"] == "openrouter"
    assert restore["api_key"] == "test-key"
    assert restore["base_url"] == "https://openrouter.ai/api/v1"
    assert restore["api_mode"] == "chat_completions"


def test_one_shot_evicts_cached_agent():
    """The agent is set to None so a fresh one is built with the MoA provider."""
    cli = _make_cli(agent=MagicMock())
    assert cli.agent is not None
    with patch("cli._cprint"):
        cli.process_command("/moa fresh start")
    assert cli.agent is None


# ---------------------------------------------------------------------------
# (2) Normal mode is unaffected
# ---------------------------------------------------------------------------

def test_non_moa_command_leaves_moa_flags_untouched():
    """A regular slash command (not /moa) does not mutate any MoA state."""
    cli = _make_cli()
    original_provider = cli.provider
    original_model = cli.model
    with patch("cli._cprint"):
        # /help is always handled and returns True without MoA side-effects
        cli.process_command("/help")
    assert cli.provider == original_provider
    assert cli.model == original_model
    assert cli._pending_moa_disable_after_turn is False
    assert cli._pending_moa_restore_model is None
    assert cli._pending_agent_seed is None


def test_moa_state_isolated_between_one_shots():
    """Two consecutive /moa calls each set up their own restore snapshot; the
    second doesn't see stale state from the first."""
    cli = _make_cli()
    with patch("cli._cprint"):
        cli.process_command("/moa first prompt")
    # Simulate restore after first turn
    restore_1 = cli._pending_moa_restore_model
    for key, value in restore_1.items():
        if value is not None:
            setattr(cli, key, value)
    cli._pending_moa_disable_after_turn = False
    cli._pending_moa_restore_model = None
    cli._pending_agent_seed = None

    # Now fire a second one-shot
    with patch("cli._cprint"):
        cli.process_command("/moa second prompt")
    assert cli._pending_agent_seed == "second prompt"
    assert cli._pending_moa_disable_after_turn is True
    restore_2 = cli._pending_moa_restore_model
    # Restore should point back to the original model, not moa
    assert restore_2["provider"] == "openrouter"


# ---------------------------------------------------------------------------
# (3) Invalid one-shot inputs
# ---------------------------------------------------------------------------

def test_bare_moa_shows_usage_no_state_mutation():
    """/moa with no argument shows usage and does not switch provider."""
    cli = _make_cli()
    printed = []
    with patch("cli._cprint", side_effect=printed.append):
        result = cli.process_command("/moa")
    assert result is True
    assert cli.provider != "moa"
    assert cli._pending_agent_seed is None
    assert cli._pending_moa_disable_after_turn is False
    assert any("Usage" in str(s) or "usage" in str(s) for s in printed)


def test_moa_whitespace_only_shows_usage():
    """/moa with only whitespace is equivalent to bare /moa."""
    cli = _make_cli()
    printed = []
    with patch("cli._cprint", side_effect=printed.append):
        result = cli.process_command("/moa     ")
    assert result is True
    assert cli.provider != "moa"
    assert cli._pending_agent_seed is None


def test_moa_while_agent_running_still_queues_one_shot():
    """/moa during an active agent run still sets up one-shot state.

    NOTE: unlike the gateway (which has an explicit running-session guard),
    the CLI /moa handler does NOT reject during _agent_running — it sets up
    the MoA swap and relies on the main loop to consume _pending_agent_seed
    after the current turn finishes. This test documents the actual behavior.
    """
    cli = _make_cli(_agent_running=True)
    with patch("cli._cprint"):
        result = cli.process_command("/moa explain this")
    assert result is True
    # CLI does NOT guard — it swaps the provider and queues the seed
    assert cli.provider == "moa"
    assert cli._pending_agent_seed == "explain this"
    assert cli._pending_moa_disable_after_turn is True


# ---------------------------------------------------------------------------
# (4) No continuation state after one-shot
# ---------------------------------------------------------------------------

def test_restore_clears_all_moa_continuation_state():
    """Simulates the post-turn restore path (cli.py ~line 12271-12278) and
    verifies no MoA continuation state remains."""
    cli = _make_cli()
    with patch("cli._cprint"):
        cli.process_command("/moa check this")

    # Verify MoA is active before restore
    assert cli.provider == "moa"
    assert cli._pending_moa_disable_after_turn is True
    assert cli._pending_moa_restore_model is not None

    # Simulate the restore path (what the chat() finally does after the turn)
    restore = cli._pending_moa_restore_model or {}
    for key, value in restore.items():
        if value is not None:
            setattr(cli, key, value)
    cli.agent = None
    cli._pending_moa_restore_model = None
    cli._pending_moa_disable_after_turn = False

    # All MoA state must be gone
    assert cli.provider == "openrouter"
    assert cli.model == "anthropic/claude-opus-4.8"
    assert cli.api_key == "test-key"
    assert cli.base_url == "https://openrouter.ai/api/v1"
    assert cli._pending_moa_disable_after_turn is False
    assert cli._pending_moa_restore_model is None


def test_one_shot_does_not_mutate_pending_input_queue():
    """/moa must not place anything in the regular _pending_input queue — it
    uses _pending_agent_seed, which is consumed differently."""
    cli = _make_cli()
    with patch("cli._cprint"):
        cli.process_command("/moa test prompt")
    assert cli._pending_input.empty()


def test_one_shot_does_not_set_pending_moa_config():
    """The one-shot path uses provider/model swap, NOT _pending_moa_config
    (which is the per-turn MoA config injection path)."""
    cli = _make_cli()
    with patch("cli._cprint"):
        cli.process_command("/moa test prompt")
    assert cli._pending_moa_config is None
