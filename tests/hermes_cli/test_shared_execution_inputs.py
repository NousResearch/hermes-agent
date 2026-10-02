"""CLI presentation stays at the terminal boundary after shared execution moves."""
def test_profile_terminal_output_uses_canonical_identity(monkeypatch):
    import hermes_cli.cli_commands_mixin as cli_commands
    import hermes_cli.profiles as profiles
    lines = []
    monkeypatch.setattr(profiles, "profile_command_details", lambda: {
        "profile_name": "research", "home_display": "/profiles/research",
        "profile_label": "Research display name"})
    monkeypatch.setattr(cli_commands, "_say_block", lambda *values: lines.extend(values))
    cli_commands.CLICommandsMixin._handle_profile_command(object())
    assert lines == ["  Profile: research", "  Home:    /profiles/research"]


def test_egress_terminal_output_preserves_literal_status(monkeypatch):
    import hermes_cli.cli_loops_mixin as loops
    import hermes_cli.proxy_cli as proxy
    status = "Egress proxy status\nHosts: [example.invalid]"
    monkeypatch.setattr(proxy, "format_status_text", lambda: status)
    output = []
    class Terminal:
        def _console_print(self, text, **kwargs):
            output.append((text, kwargs))
    loops.CLILoopsMixin._cmd_egress(Terminal(), "/egress")
    assert output == [(status, {"highlight": False, "markup": False})]
