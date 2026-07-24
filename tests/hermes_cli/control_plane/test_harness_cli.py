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

    assert results[0]["node"]["id"] == "node-1"
    raw = results[0]["credential"]
    assert raw
    assert results[1][0]["state"] == "enrolled"
    assert results[2]["state"] == "active"
    assert [event["event_type"] for event in results[3]] == [
        "node.enrolled",
        "node.active",
    ]
    assert results[4] == {"valid": True}
    public_output = json.dumps(results[1:])
    assert raw not in public_output


def test_cli_rotation_and_revocation_share_registry(tmp_path, monkeypatch, capsys):
    registry = NodeRegistry(tmp_path / "control-plane.db", clock=lambda: 1_000)
    monkeypatch.setattr(harness, "_registry", lambda: registry)
    top = parser()
    enrolled = registry.enroll(
        enrollment_key="request-1",
        node_id="node-1",
        role="worker",
        owner="ops",
        actor="operator:alice",
    )

    args = top.parse_args([
        "harness",
        "nodes",
        "rotate-credential",
        "node-1",
        "--actor",
        "operator:bob",
        "--expected-credential-revision",
        "1",
    ])
    args.func(args)
    rotated = json.loads(capsys.readouterr().out)
    assert rotated["credential"]
    assert not registry.authenticate("node-1", enrolled.credential)
    assert registry.authenticate("node-1", rotated["credential"])

    args = top.parse_args([
        "harness",
        "nodes",
        "revoke-credential",
        "node-1",
        "--actor",
        "operator:bob",
        "--expected-credential-revision",
        "2",
    ])
    args.func(args)
    revoked = json.loads(capsys.readouterr().out)
    assert revoked["credential_status"] == "revoked"
    assert "credential" not in revoked


def test_cli_report_policy_and_reconciliation_views(tmp_path, monkeypatch, capsys):
    registry = NodeRegistry(tmp_path / "control-plane.db", clock=lambda: 1_000)
    monkeypatch.setattr(harness, "_registry", lambda: registry)
    top = parser()
    issuance = registry.enroll(
        enrollment_key="request-1",
        node_id="node-1",
        role="worker",
        owner="ops",
        actor="operator:alice",
    )
    credential = issuance.credential
    assert credential
    monkeypatch.setenv("HERMES_NODE_CREDENTIAL", credential)

    commands = [
        [
            "harness",
            "nodes",
            "report",
            "node-1",
            "--report-sequence",
            "1",
            "--observed-at",
            "900",
            "--health-state",
            "healthy",
            "--capabilities",
            '{"os":"linux"}',
        ],
        [
            "harness",
            "nodes",
            "policy",
            "set",
            "node-1",
            "--actor",
            "operator:alice",
            "--health-state",
            "healthy",
            "--capabilities",
            '{"os":"linux"}',
            "--expected-revision",
            "0",
        ],
        ["harness", "nodes", "reconcile", "node-1"],
    ]
    results = []
    for command in commands:
        assert credential not in command
        args = top.parse_args(command)
        args.func(args)
        results.append(json.loads(capsys.readouterr().out))

    assert results[0]["report_sequence"] == 1
    assert results[1]["revision"] == 1
    assert results[2]["in_sync"] is True
    assert credential not in json.dumps(results)


def test_cli_report_help_and_output_do_not_expose_credential(
    tmp_path, monkeypatch, capsys
):
    registry = NodeRegistry(tmp_path / "control-plane.db", clock=lambda: 1_000)
    monkeypatch.setattr(harness, "_registry", lambda: registry)
    issuance = registry.enroll(
        enrollment_key="request-1",
        node_id="node-1",
        role="worker",
        owner="ops",
        actor="operator:alice",
    )
    credential = issuance.credential
    assert credential
    monkeypatch.setenv("HERMES_NODE_CREDENTIAL", credential)
    top = parser()
    report = top.parse_args([
        "harness",
        "nodes",
        "report",
        "node-1",
        "--report-sequence",
        "1",
        "--observed-at",
        "900",
        "--health-state",
        "healthy",
    ])

    assert not hasattr(report, "credential")
    assert credential not in [value for _, value in report._get_kwargs()]
    report.func(report)
    assert credential not in capsys.readouterr().out

    report_parser = parser()
    try:
        report_parser.parse_args(["harness", "nodes", "report", "--help"])
    except SystemExit as exc:
        assert exc.code == 0
    help_output = capsys.readouterr().out
    assert "HERMES_NODE_CREDENTIAL" in help_output
    assert "--credential" not in help_output
    assert credential not in help_output


def test_cli_report_requires_credential_environment_variable(monkeypatch):
    monkeypatch.delenv("HERMES_NODE_CREDENTIAL", raising=False)
    args = parser().parse_args([
        "harness",
        "nodes",
        "report",
        "node-1",
        "--report-sequence",
        "1",
        "--observed-at",
        "900",
        "--health-state",
        "healthy",
    ])

    try:
        args.func(args)
    except SystemExit as exc:
        assert str(exc) == (
            "HERMES_NODE_CREDENTIAL must contain the managed-node credential"
        )
    else:
        raise AssertionError("missing managed-node credential should fail closed")
