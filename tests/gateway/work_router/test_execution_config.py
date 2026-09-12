from dataclasses import replace

import pytest

from gateway.work_router.config import DEFAULT_PROFILES, RouterConfig
from gateway.work_router.models import CanonicalEvent
from gateway.work_router.rules import DirectiveRejection, ParsedWorkDirective, parse_work_directive


def _mapping(tmp_path):
    return {
        "enabled": True,
        "channel_allowlist": ["CEXEC"],
        "bot_registry": {name: f"UBOT{i}" for i, name in enumerate(DEFAULT_PROFILES)},
        "db_path": str(tmp_path / "router.db"),
        "work_execution": {
            "enabled": True,
            "channel_allowlist": ["CEXEC"],
            "slack_team_id": "TTEST",
            "max_body_bytes": 1024,
        },
    }


def _event():
    return CanonicalEvent(
        event_id="EvDirective", channel_id="CEXEC", thread_ts="1700000000.1",
        text="[WORK] <@UBOT1> :: Review\nEvidence",
        author_kind="bot", author_profile="Demian", author_user_id="UBOT0",
        metadata={"slack_team_id": "TTEST"},
    )


def test_execution_mapping_reaches_real_directive_parser(tmp_path):
    raw = _mapping(tmp_path)
    config = RouterConfig.from_mapping(raw).require_ready()
    parsed = parse_work_directive(_event(), config)
    assert isinstance(parsed, ParsedWorkDirective)
    assert parsed.owner_profile == "Hans"
    assert parse_work_directive(
        replace(_event(), metadata={"slack_team_id": "TOTHER"}), config
    ) == DirectiveRejection("workspace_mismatch")
    assert parse_work_directive(
        replace(_event(), author_user_id="UOTHER"), config
    ) == DirectiveRejection("issuer_identity_mismatch")
    raw.pop("work_execution")
    assert parse_work_directive(
        _event(), RouterConfig.from_mapping(raw).require_ready()
    ) == DirectiveRejection("feature_disabled")
    assert parse_work_directive(_event(), RouterConfig()) == DirectiveRejection("feature_disabled")


@pytest.mark.parametrize("change", [
    {"enabled": "true"}, {"channel_allowlist": ["COTHER"]},
    {"channel_allowlist": []}, {"slack_team_id": ""},
    {"max_body_bytes": True}, {"max_body_bytes": -1},
    {"max_body_bytes": 1.5}, {"unknown": True},
])
def test_invalid_execution_config_cannot_become_ready(tmp_path, change):
    raw = _mapping(tmp_path)
    raw["work_execution"].update(change)
    config = RouterConfig.from_mapping(raw)
    assert not config.ready
    assert config.error is not None and "work_execution" in config.error
