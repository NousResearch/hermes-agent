"""`hermes backup`: -o/--output and --quick are mutually exclusive (#98369).

--quick stores a state snapshot into HERMES_HOME/state-snapshots/ and never
writes an archive, so accepting -o with --quick silently dropped the requested
path (exit 0, nothing created). The combination is now rejected at parse time.
"""
import argparse

import pytest

from hermes_cli.subcommands.backup import build_backup_parser


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="hermes")
    sub = p.add_subparsers()
    build_backup_parser(sub, cmd_backup=lambda args: None)
    return p


def test_quick_with_output_is_rejected():
    with pytest.raises(SystemExit) as ei:
        _parser().parse_args(["backup", "--quick", "-o", "/tmp/x.zip"])
    assert ei.value.code == 2  # argparse usage error, not a silent exit 0


@pytest.mark.parametrize(
    "argv,expect",
    [
        (["backup", "-o", "/tmp/x.zip"], {"output": "/tmp/x.zip", "quick": False}),
        (["backup", "--quick"], {"output": None, "quick": True}),
        (["backup", "--quick", "-l", "nightly"], {"output": None, "quick": True, "label": "nightly"}),
    ],
)
def test_valid_backup_arg_combos_still_parse(argv, expect):
    ns = _parser().parse_args(argv)
    for key, value in expect.items():
        assert getattr(ns, key) == value
