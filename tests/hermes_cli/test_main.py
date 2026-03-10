import sys
from unittest.mock import patch

from hermes_cli import main as main_module


def test_setup_accepts_migrate_from_openclaw_and_dispatches_to_wizard():
    captured = {}

    def fake_run_setup_wizard(args):
        captured["args"] = args

    with (
        patch.object(sys, "argv", ["hermes", "setup", "--migrate-from", "openclaw"]),
        patch(
            "hermes_cli.setup.run_setup_wizard",
            side_effect=fake_run_setup_wizard,
        ),
    ):
        main_module.main()

    assert captured["args"].command == "setup"
    assert captured["args"].migrate_from == "openclaw"
