"""Tests for Codex gpt-5.5 autoraise notice deduplication."""

from __future__ import annotations

import yaml

from agent.agent_init import _codex_gpt55_autoraise_notice_if_unseen
from agent.onboarding import CODEX_GPT55_AUTORAISE_NOTICE_FLAG


def test_codex_gpt55_autoraise_notice_marks_seen(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("compression:\n  threshold: 0.5\n", encoding="utf-8")
    config: dict[str, object] = {"compression": {"threshold": 0.5}}

    notice = _codex_gpt55_autoraise_notice_if_unseen(
        {"from": 0.5, "to": 0.85},
        config,
        config_path=config_path,
    )

    assert notice is not None
    assert "auto-compaction was raised to 85%" in notice
    onboarding = config["onboarding"]
    assert isinstance(onboarding, dict)
    seen = onboarding["seen"]
    assert isinstance(seen, dict)
    assert seen[CODEX_GPT55_AUTORAISE_NOTICE_FLAG] is True

    persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert persisted["onboarding"]["seen"][CODEX_GPT55_AUTORAISE_NOTICE_FLAG] is True

    assert (
        _codex_gpt55_autoraise_notice_if_unseen(
            {"from": 0.5, "to": 0.85},
            config,
            config_path=config_path,
        )
        is None
    )


def test_codex_gpt55_autoraise_notice_respects_existing_seen_flag(tmp_path):
    config_path = tmp_path / "config.yaml"
    config = {
        "onboarding": {
            "seen": {
                CODEX_GPT55_AUTORAISE_NOTICE_FLAG: True,
            }
        }
    }

    notice = _codex_gpt55_autoraise_notice_if_unseen(
        {"from": 0.5, "to": 0.85},
        config,
        config_path=config_path,
    )

    assert notice is None
    assert not config_path.exists()
