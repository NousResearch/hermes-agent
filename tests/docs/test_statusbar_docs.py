from pathlib import Path


def test_cli_statusbar_docs_describe_runtime_segments():
    text = Path("website/docs/user-guide/cli.md").read_text(encoding="utf-8")

    for expected in [
        "run:<mode>",
        "phase:<phase>",
        "target:<name>",
        "main:<agent>",
        "sub:<label>",
        "tool:<name>✓",
        "skill:<name>",
        "task:<done>/<total>",
        "bg:N",
        "wait:<reason>",
    ]:
        assert expected in text


def test_statusbar_command_registry_mentions_runtime_status():
    from hermes_cli.commands import COMMAND_REGISTRY

    statusbar = next(cmd for cmd in COMMAND_REGISTRY if cmd.name == "statusbar")

    assert "runtime" in statusbar.description.lower()
    assert "context" in statusbar.description.lower()
