from hermes_cli.commands import resolve_command


def test_phase_two_adds_only_dryrun_parallel_review_gateway_command():
    assert resolve_command("parallel") is None

    cmd = resolve_command("parallel_review")
    assert cmd is not None
    assert cmd.gateway_only is True
    assert cmd.args_hint == "dryrun [--save] <request>"
    assert "run" not in cmd.subcommands
