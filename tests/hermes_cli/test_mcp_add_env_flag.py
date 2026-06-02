"""Regression test for ``hermes mcp add --env`` repeated flag support.

Verifies that multiple ``--env KEY=VALUE`` flags accumulate correctly,
rather than only keeping the last one (the ``nargs=\"*\"`` bug).
See: https://github.com/NousResearch/hermes-agent/issues/37501
"""

import argparse


def _build_mcp_add_parser() -> argparse.ArgumentParser:
    """Reproduce the relevant subset of the ``hermes mcp add`` parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--command", dest="mcp_command")
    parser.add_argument("--args", nargs="*", default=[])
    parser.add_argument("--env", action="append", default=[],
                        help="Environment variables (KEY=VALUE)")
    return parser


class TestMcpAddEnvFlag:
    """Tests for ``--env`` flag parsing in ``hermes mcp add``."""

    def test_single_env(self):
        parser = _build_mcp_add_parser()
        args = parser.parse_args(["myserver", "--command", "npx",
                                  "--env", "KEY=VALUE"])
        assert args.env == ["KEY=VALUE"]

    def test_multiple_env_flags(self):
        """The core regression: repeated --env flags must accumulate."""
        parser = _build_mcp_add_parser()
        args = parser.parse_args([
            "myserver", "--command", "npx",
            "--env", "ALPACA_API_KEY=sk-test",
            "--env", "ALPACA_SECRET_KEY=secret123",
        ])
        assert args.env == ["ALPACA_API_KEY=sk-test", "ALPACA_SECRET_KEY=secret123"]

    def test_three_env_flags(self):
        parser = _build_mcp_add_parser()
        args = parser.parse_args([
            "myserver", "--command", "npx",
            "--env", "A=1", "--env", "B=2", "--env", "C=3",
        ])
        assert args.env == ["A=1", "B=2", "C=3"]

    def test_no_env_flag(self):
        parser = _build_mcp_add_parser()
        args = parser.parse_args(["myserver", "--command", "npx"])
        assert args.env == []

    def test_env_value_with_equals(self):
        """Values containing '=' must be preserved (split on first '=' only)."""
        parser = _build_mcp_add_parser()
        args = parser.parse_args([
            "myserver", "--command", "npx",
            "--env", "CONNECTION_STRING=host=localhost;port=5432",
        ])
        assert args.env == ["CONNECTION_STRING=host=localhost;port=5432"]


class TestMcpAddEnvParsing:
    """Tests for ``_parse_env_assignments`` with the new list format."""

    def test_parse_accumulated_env(self):
        """Verify _parse_env_assignments handles the accumulated list."""
        from hermes_cli.mcp_config import _parse_env_assignments

        result = _parse_env_assignments([
            "ALPACA_API_KEY=sk-test",
            "ALPACA_SECRET_KEY=secret123",
        ])
        assert result == {
            "ALPACA_API_KEY": "sk-test",
            "ALPACA_SECRET_KEY": "secret123",
        }

    def test_parse_empty_list(self):
        from hermes_cli.mcp_config import _parse_env_assignments
        assert _parse_env_assignments([]) == {}

    def test_parse_none(self):
        from hermes_cli.mcp_config import _parse_env_assignments
        assert _parse_env_assignments(None) == {}

    def test_parse_value_with_equals(self):
        from hermes_cli.mcp_config import _parse_env_assignments
        result = _parse_env_assignments(["DB_URL=host=localhost;port=5432"])
        assert result == {"DB_URL": "host=localhost;port=5432"}
