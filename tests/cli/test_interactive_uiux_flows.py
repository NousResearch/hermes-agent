"""UI/UX flow tests for the interactive slash-command engine.

Simulates end-to-end user journeys through `_run_interactive_spec` with
mocked I/O, verifying:
- navigation to a final option completes the action and returns to chat
  (one-shot), printing only the intended result (no artifact/footprint);
- Esc at the root exits cleanly with no output;
- an invalid free-text entry (empty) cancels cleanly;
- repeated invocations behave consistently.
"""

import types


def _bare_cli():
    from cli import HermesCLI

    cli = HermesCLI.__new__(HermesCLI)
    return cli


def _wire(cli, cli_mod, picks, prompts, printed):
    def fake_picker(self, title, items, default_index=0):
        return picks.pop(0) if picks else None

    def fake_free_text(self, title, prompt):
        return prompts.pop(0) if prompts else None

    cli._run_curses_picker = types.MethodType(fake_picker, cli)
    cli._prompt_free_text_modal = types.MethodType(fake_free_text, cli)
    cli_mod._cprint = printed.append
    return cli


def _peers_spec():
    """Mimic what cmd_peers produces for a 2-peer host."""
    def send(value, text=None):
        return f"Sent to {value}: {text}"

    return {
        "title": "Peers · 2 live · 0 working · 2 idle",
        "items": [
            {
                "label": "○ kensei-a  cli  ○idle  —  -",
                "value": "peer-a",
                "actions": [
                    {"key": "s", "label": "Send message", "handler": send,
                     "prompt": "Message to kensei-a:"},
                ],
            },
            {
                "label": "○ kensei-b  cli  ○idle  —  -",
                "value": "peer-b",
                "actions": [
                    {"key": "s", "label": "Send message", "handler": send,
                     "prompt": "Message to kensei-b:"},
                ],
            },
        ],
    }


def test_full_send_flow_returns_to_chat_with_result():
    """Select peer -> Send -> type -> result printed, then engine exits (one-shot)."""
    import cli as cli_mod

    cli = _bare_cli()
    printed = []
    # picks: item 0 (peer-a), action 0 (Send)
    _wire(cli, cli_mod, [0, 0], ["hello"], printed)

    cli._run_interactive_spec(_peers_spec(), "peers")

    # Exactly the intended result is printed — nothing else (no footprint).
    assert printed == ["Sent to peer-a: hello"], printed


def test_esc_at_root_exits_cleanly():
    """Esc at the root picker produces no output at all."""
    import cli as cli_mod

    cli = _bare_cli()
    printed = []
    _wire(cli, cli_mod, [None], [], printed)

    cli._run_interactive_spec(_peers_spec(), "peers")
    assert printed == [], printed


def test_empty_free_text_cancels_cleanly():
    """Empty free-text input cancels the action without printing garbage."""
    import cli as cli_mod

    cli = _bare_cli()
    printed = []
    # picks: item 0, action 0; prompt returns None (cancel)
    _wire(cli, cli_mod, [0, 0], [None], printed)

    cli._run_interactive_spec(_peers_spec(), "peers")
    assert printed == [], printed


def test_repeated_invocations_are_consistent():
    """The same flow twice yields identical results (no state leakage)."""
    import cli as cli_mod

    cli = _bare_cli()
    printed = []
    _wire(cli, cli_mod, [0, 0], ["first"], printed)
    cli._run_interactive_spec(_peers_spec(), "peers")
    assert printed == ["Sent to peer-a: first"], printed

    # Second run with fresh I/O state (new cli_mod patch), same spec.
    printed2 = []
    _wire(cli, cli_mod, [1, 0], ["second"], printed2)
    cli._run_interactive_spec(_peers_spec(), "peers")
    assert printed2 == ["Sent to peer-b: second"], printed2
