"""Generated messaging help must teach the unambiguous denial grammar."""
from hermes_cli.commands import resolve_command, gateway_help_lines


def test_generated_deny_help_requires_explicit_reason_marker():
    command = resolve_command('deny')
    assert command.args_hint == '[all|exact-request-id] [--reason text]'
    line = next(line for line in gateway_help_lines() if '/deny' in line)
    assert '--reason text' in line and 'exact-request-id' in line
