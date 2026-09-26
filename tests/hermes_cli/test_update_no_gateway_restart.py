"""The PM completion child honors --no-gateway-restart without skipping preparation (#93649)."""
import argparse
import json
import shutil

from hermes_cli import update_completion
from hermes_cli.subcommands.update import build_update_parser
from tests.hermes_cli.test_update_completion_process import transition  # noqa: F401


def test_restart_deferral_crosses_real_completion_process(transition):
    root, _git, _old, _new, request = transition
    parser = argparse.ArgumentParser()
    build_update_parser(parser.add_subparsers(), cmd_update=lambda args: None)
    request["no_gateway_restart"] = parser.parse_args(["update", "--no-gateway-restart"]).no_gateway_restart
    shutil.copy2(update_completion.__file__, root / "hermes_cli/update_completion.py")
    receipt = root / "hermes_cli/update_receipt.py"
    with receipt.open("a", encoding="utf-8") as stream:
        stream.write("\nfrom hermes_cli.probe import event\ndef record_skip(*args): event('deferred')\n")
    marker = root / "fleet_restart_pending"
    marker.write_text("pending", encoding="utf-8")

    result = update_completion.run_completion(request)

    assert result["exit_code"] == 0
    assert result["receipt"]["outcome"] == "success"
    assert result["windows_resume"]["resume_needed"] is False
    events = [json.loads(line)["name"] for line in (root / "events.jsonl").read_text().splitlines()]
    assert {"prepare", "build", "maintenance", "deferred", "emergency_resume"} <= set(events)
    assert "restart" not in events and "verify" not in events
    assert marker.read_text() == "pending"
