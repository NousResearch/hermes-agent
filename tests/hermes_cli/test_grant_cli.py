"""CLI wiring tests for ``hermes grant`` (process-boundary contract).

``hermes grant issue`` is REFUSED: a short-lived CLI process cannot mint a
grant the running Hermes process would accept (different authority
generation).  Issuance happens exclusively inside the long-lived Hermes
process via the governed tool.  ``list`` / ``revoke`` / ``audit`` remain
available for operational visibility.
"""

import argparse
import json
import time
import uuid

import pytest

from hermes_cli.subcommands.grant import (
    build_grant_parser,
    run_grant_command,
)

_IDENTITY = {"hostname": "storage-guest", "boot_id": "boot-A", "product_uuid": "uuid-A"}


def _make_parser():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    build_grant_parser(subparsers, cmd_grant=lambda a: run_grant_command(a))
    return parser


class TestGrantParser:
    def test_issue_parses_full_tuple(self):
        parser = _make_parser()
        args = parser.parse_args(
            [
                "grant", "issue",
                "--operation", "CREATE_FILESYSTEM",
                "--vm", "101",
                "--hostname", "storage-guest",
                "--device", "/dev/sdb1",
                "--fs", "ext4",
                "--label", "DATA",
                "--subject", "operator",
                "--session", "sess-1",
                "--ttl", "300",
            ]
        )
        assert args.grant_command == "issue"
        assert args.vm_id == "101"
        assert args.device == "/dev/sdb1"
        assert args.fs_type == "ext4"
        assert args.label == "DATA"
        assert args.subject == "operator"
        assert args.session_id == "sess-1"
        assert args.ttl == 300
        assert hasattr(args, "func")

    def test_list_and_audit_parse(self):
        parser = _make_parser()
        for sub in ("list", "audit"):
            args = parser.parse_args(["grant", sub])
            assert args.grant_command == sub
            assert hasattr(args, "func")

    def test_revoke_parses_grant_id(self):
        parser = _make_parser()
        args = parser.parse_args(["grant", "revoke", "abc-123"])
        assert args.grant_command == "revoke"
        assert args.grant_id == "abc-123"


class TestGrantCommand:
    def test_issue_is_refused_no_grant_created(self, capsys):
        """Track A: the CLI must NOT mint a grant.  ``hermes grant issue``
        exits nonzero with an explicit message and creates nothing."""
        from tools import destructive_grants as dg

        before = len(dg.list_grants())
        args = _make_parser().parse_args(
            [
                "grant", "issue",
                "--operation", "CREATE_FILESYSTEM",
                "--vm", "101",
                "--hostname", "storage-guest",
                "--device", "/dev/sdb1",
                "--fs", "ext4",
                "--label", "DATA",
                "--subject", "operator",
                "--session", "sess-1",
            ]
        )
        rc = run_grant_command(args)
        assert rc == 2
        err = capsys.readouterr().err
        assert "REFUSED" in err
        assert "running Hermes process" in err
        # No grant was created by the CLI.
        assert len(dg.list_grants()) == before

    def test_issue_json_also_refused(self, capsys):
        args = _make_parser().parse_args(
            [
                "grant", "issue",
                "--operation", "CREATE_FILESYSTEM",
                "--vm", "101",
                "--hostname", "storage-guest",
                "--device", "/dev/sdb1",
                "--fs", "ext4",
                "--label", "DATA",
                "--subject", "operator",
                "--session", "sess-1",
                "--json",
            ]
        )
        rc = run_grant_command(args)
        assert rc == 2
        err = capsys.readouterr().err
        assert "REFUSED" in err

    def test_list_audit_revoke_workflow(self, capsys):
        """list/audit/revoke remain operational (no issuance)."""
        # list on an empty pool
        list_args = _make_parser().parse_args(["grant", "list"])
        rc = run_grant_command(list_args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "No live grants" in out

        # audit on an empty trail
        audit_args = _make_parser().parse_args(["grant", "audit"])
        rc = run_grant_command(audit_args)
        assert rc == 0
        out = capsys.readouterr().out
        assert "No audit entries" in out

        # revoke of an unknown (but well-formed) id
        revoke_args = _make_parser().parse_args(
            ["grant", "revoke", "99999999-9999-4999-8999-999999999999"]
        )
        rc = run_grant_command(revoke_args)
        assert rc == 1
        err = capsys.readouterr().err
        assert "not_found" in err

        # revoke of a malformed id -> clean error, no crash
        revoke_args = _make_parser().parse_args(["grant", "revoke", "not-a-uuid"])
        rc = run_grant_command(revoke_args)
        assert rc == 2
        err = capsys.readouterr().err
        assert "invalid grant id" in err
