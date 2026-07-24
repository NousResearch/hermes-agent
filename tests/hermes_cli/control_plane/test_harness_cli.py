from __future__ import annotations

import json

from hermes_cli._parser import build_top_level_parser
from hermes_cli.control_plane.nodes import NodeRegistry
from hermes_cli.subcommands import harness
from hermes_cli.subcommands.harness import build_harness_parser


def parser():
    top, subparsers, _ = build_top_level_parser()
    build_harness_parser(subparsers)
    return top


def test_parser_accepts_enrollment_contract():
    args = parser().parse_args([
        "harness",
        "nodes",
        "enroll",
        "--enrollment-key",
        "request-1",
        "--node-id",
        "node-1",
        "--role",
        "worker",
        "--owner",
        "ops",
        "--actor",
        "operator:alice",
        "--capabilities",
        '{"os":"linux"}',
    ])

    assert args.func is harness.cmd_nodes_enroll
    assert args.capabilities == {"os": "linux"}


def test_cli_enroll_list_transition_history_and_audit(tmp_path, monkeypatch, capsys):
    registry = NodeRegistry(tmp_path / "control-plane.db", clock=lambda: 1_000)
    monkeypatch.setattr(harness, "_registry", lambda: registry)
    top = parser()

    commands = [
        [
            "harness",
            "nodes",
            "enroll",
            "--enrollment-key",
            "request-1",
            "--node-id",
            "node-1",
            "--role",
            "worker",
            "--owner",
            "ops",
            "--actor",
            "operator:alice",
        ],
        ["harness", "nodes", "list", "--state", "enrolled"],
        [
            "harness",
            "nodes",
            "transition",
            "node-1",
            "active",
            "--actor",
            "service:reconciler",
            "--expected-revision",
            "1",
            "--reason",
            "ready",
        ],
        ["harness", "nodes", "history", "node-1"],
        ["harness", "nodes", "audit"],
    ]
    results = []
    for command in commands:
        args = top.parse_args(command)
        args.func(args)
        results.append(json.loads(capsys.readouterr().out))

    assert results[0]["id"] == "node-1"
    assert results[1][0]["state"] == "enrolled"
    assert results[2]["state"] == "active"
    assert [event["event_type"] for event in results[3]] == [
        "node.enrolled",
        "node.active",
    ]
    assert results[4] == {"valid": True}
