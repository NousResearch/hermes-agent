"""Tests for _coalesce_session_name_args — multi-word session name merging."""

from hermes_cli.main import _coalesce_session_name_args


def test_the_active_memory_providers_command_ends_a_session_name(monkeypatch):
    """``hermes <provider>`` is registered by the active memory provider, not hardcoded in core."""
    import plugins.memory as memory_pkg

    monkeypatch.setattr(memory_pkg, "_get_active_memory_provider", lambda: "somemem")
    assert _coalesce_session_name_args(["chat", "somemem"]) == ["chat", "somemem"]  # no -c: untouched
    assert _coalesce_session_name_args(["-c", "my", "session", "somemem", "sync"]) == [
        "-c", "my session", "somemem", "sync"]


class TestCoalesceSessionNameArgs:
    """Ensure unquoted multi-word session names are merged into one token."""

    # ── -c / --continue ──────────────────────────────────────────────────

    def test_continue_multiword_unquoted(self):
        """hermes -c Pokemon Agent Dev → -c 'Pokemon Agent Dev'"""
        assert _coalesce_session_name_args(
            ["-c", "Pokemon", "Agent", "Dev"]
        ) == ["-c", "Pokemon Agent Dev"]


    # ── -r / --resume ────────────────────────────────────────────────────


    # ── combined flags ───────────────────────────────────────────────────


    # ── passthrough (no session flags) ───────────────────────────────────

    def test_no_session_flags_passthrough(self):
        """hermes -w chat -q hello (nothing to merge)"""
        result = _coalesce_session_name_args(["-w", "chat", "-q", "hello"])
        assert result == ["-w", "chat", "-q", "hello"]


    # ── subcommand boundary ──────────────────────────────────────────────


    def test_stops_at_setup_subcommand(self):
        """hermes -c my setup → 'setup' is a subcommand, not part of name"""
        assert _coalesce_session_name_args(
            ["-c", "my", "setup"]
        ) == ["-c", "my", "setup"]
