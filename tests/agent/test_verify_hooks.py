"""Unit tests for the verification-loop policy (agent/verify_hooks.py).

The `pre_verify` user-hook aggregation lives in `hermes_cli.plugins`
(`get_pre_verify_continue_message`) and is tested in
`tests/hermes_cli/test_plugins.py`, alongside `get_pre_tool_call_block_message`.
"""

from __future__ import annotations

import logging

from agent import verify_hooks


class TestMaxVerifyNudges:
    def test_default_when_unset(self):
        assert (
            verify_hooks.max_verify_nudges({})
            == verify_hooks.DEFAULT_MAX_VERIFY_NUDGES
        )
        assert (
            verify_hooks.max_verify_nudges({"agent": {}})
            == verify_hooks.DEFAULT_MAX_VERIFY_NUDGES
        )

    def test_reads_and_coerces(self):
        assert verify_hooks.max_verify_nudges({"agent": {"max_verify_nudges": 5}}) == 5
        assert verify_hooks.max_verify_nudges({"agent": {"max_verify_nudges": "2"}}) == 2
        assert verify_hooks.max_verify_nudges({"agent": {"max_verify_nudges": -1}}) == 0

    def test_bad_value_falls_back(self):
        assert (
            verify_hooks.max_verify_nudges({"agent": {"max_verify_nudges": "x"}})
            == verify_hooks.DEFAULT_MAX_VERIFY_NUDGES
        )

    def test_config_load_failure_falls_back_and_logs_debug(self, monkeypatch, caplog):
        import hermes_cli.config as config_mod

        def fail_load_config():
            raise RuntimeError("c2 verify-hooks fixture")

        monkeypatch.setattr(config_mod, "load_config", fail_load_config)

        with caplog.at_level(logging.DEBUG, logger="agent.verify_hooks"):
            assert verify_hooks.max_verify_nudges(None) == verify_hooks.DEFAULT_MAX_VERIFY_NUDGES

        records = [
            record
            for record in caplog.records
            if record.name == "agent.verify_hooks"
        ]
        assert len(records) == 1
        assert records[0].levelno == logging.DEBUG
        assert "using verification defaults" in records[0].getMessage()


class TestCodingVerifyGuidance:
    def test_enabled_by_default(self):
        assert (
            verify_hooks.coding_verify_guidance({})
            == verify_hooks.CODING_VERIFY_GUIDANCE
        )
        assert (
            verify_hooks.coding_verify_guidance({"agent": {}})
            == verify_hooks.CODING_VERIFY_GUIDANCE
        )

    def test_reads_truthy_config(self):
        cfg = {"agent": {"verify_guidance": "yes"}}
        assert verify_hooks.coding_verify_guidance(cfg) == verify_hooks.CODING_VERIFY_GUIDANCE

    def test_opt_out_via_config(self):
        off = {"agent": {"verify_guidance": False}}
        assert verify_hooks.coding_verify_guidance(off) is None
