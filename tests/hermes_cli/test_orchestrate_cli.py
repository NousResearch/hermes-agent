from types import SimpleNamespace

from hermes_cli.subcommands.orchestrate import (
    _restart_launch_agent,
    build_orchestrate_parser,
)


def _handler(_):
    return None


def test_orchestrate_start_parser():
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_orchestrate_parser(subparsers, cmd_orchestrate=_handler)
    args = parser.parse_args(["orchestrate", "start", "CYCLE_ONE"])
    assert args.command == "orchestrate"
    assert args.orchestrate_action == "start"
    assert args.cycle_id == "CYCLE_ONE"
    assert args.func is _handler


def test_orchestrate_status_requires_cycle():
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_orchestrate_parser(subparsers, cmd_orchestrate=_handler)
    args = parser.parse_args(
        ["orchestrate", "status", "dispatch-one", "--cycle", "CYCLE_ONE"]
    )
    assert args.dispatch_id == "dispatch-one"
    assert args.cycle_id == "CYCLE_ONE"


def test_orchestrate_prepare_collects_repeated_scope_fields():
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_orchestrate_parser(subparsers, cmd_orchestrate=_handler)
    args = parser.parse_args(
        [
            "orchestrate", "prepare", "--repo", "/srv/project",
            "--repository-id", "my-project", "--cycle", "FEATURE_EXAMPLE_001",
            "--contract", "FEATURE-EXAMPLE-001", "--goal", "Build it",
            "--accept", "Tests pass", "--allow", "src/example.py",
            "--branch", "feat/example", "--worktree", "/srv/worktrees/example",
        ]
    )
    assert args.acceptance == ["Tests pass"]
    assert args.allowed_paths == ["src/example.py"]


def test_orchestrate_activate_parser():
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_orchestrate_parser(subparsers, cmd_orchestrate=_handler)
    args = parser.parse_args(["orchestrate", "activate", "/tmp/proposal.json"])
    assert args.orchestrate_action == "activate"
    assert args.proposal == "/tmp/proposal.json"


def test_orchestrate_restart_parser():
    import argparse

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_orchestrate_parser(subparsers, cmd_orchestrate=_handler)
    args = parser.parse_args(["orchestrate", "restart"])
    assert args.orchestrate_action == "restart"
    assert args.label == "ai.hermes.builder-adapter"
    assert args.timeout == 30.0


def test_restart_launch_agent_waits_for_exact_registry(monkeypatch):
    import hashlib
    import json

    calls = []
    registry = {"CYCLE": {"revision": 1}}
    registry_sha256 = hashlib.sha256(
        json.dumps(registry, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    def run(argv, **_kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0)

    class Client:
        responses = iter(
            [
                {"operational": True, "cycle_registry_sha256": "old"},
                {
                    "operational": True,
                    "cycle_registry_sha256": registry_sha256,
                    "process_id": 42,
                    "registered_cycles": 1,
                },
            ]
        )

        def health(self):
            return next(self.responses)

    monkeypatch.setattr("hermes_cli.subcommands.orchestrate.sys.platform", "darwin")
    monkeypatch.setattr(
        "hermes_cli.subcommands.orchestrate.os.getuid", lambda: 501, raising=False
    )
    monkeypatch.setattr("hermes_cli.subcommands.orchestrate.subprocess.run", run)
    monkeypatch.setattr("hermes_cli.subcommands.orchestrate.time.sleep", lambda _: None)
    settings = SimpleNamespace(cycle_registry=registry)
    args = SimpleNamespace(
        label="ai.hermes.builder-adapter",
        timeout=1.0,
    )

    result = _restart_launch_agent(args, settings, Client())

    assert result["process_id"] == 42
    assert calls == [
        ["/bin/launchctl", "print", "gui/501/ai.hermes.builder-adapter"],
        [
            "/bin/launchctl",
            "kill",
            "SIGTERM",
            "gui/501/ai.hermes.builder-adapter",
        ],
    ]
