"""Startup validation of the boundary config (education design 5)."""
import logging

from agent.boundary_config import boundary_config_errors, log_boundary_config_errors

IDS = {"release_kits": {"data_sources": ["1" * 32]}, "prds": {"data_sources": ["2" * 32]},
       "ideas": {"data_sources": ["3" * 32]}}


def test_clean_and_unconfigured():
    assert boundary_config_errors({}) == []
    cfg = {"notion": {"write_guard": {"enabled": True, **IDS}}, "agent_write_guard": True,
           "autonomy": {"announce_channel": "C1", "actions": {"post_release_update": "veto",
                                                               "zendesk_create_draft": "auto"}},
           "slack": {"post_allowed_channels": ["C1", "C2"], "post_gated_channels": ["C2"]}}
    assert boundary_config_errors(cfg) == []


def test_reports_each_problem(caplog):
    cfg = {"notion": {"write_guard": {"enabled": True}},
           "autonomy": {"actions": {"zendesk_publish": "auto", "advance_asset_status": "signoff"},
                        "veto_wait_working_hours": "soon"},
           "slack": {"post_gated_channels": ["C2"]}}
    errors = boundary_config_errors(cfg)
    joined = " | ".join(errors)
    assert "write_guard" in joined and "zendesk_publish" in joined
    assert "advance_asset_status" in joined and "veto_wait_working_hours" in joined
    assert "post_allowed_channels" in joined
    with caplog.at_level(logging.ERROR):
        assert log_boundary_config_errors("educator", cfg) == errors
    assert "educator" in caplog.text


def test_gated_outside_allowlist_reported():
    cfg = {"slack": {"post_allowed_channels": ["C1"], "post_gated_channels": ["C2"]}}
    assert "C2" in boundary_config_errors(cfg)[0]
