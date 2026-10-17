from __future__ import annotations

import json
from pathlib import Path

from hermes_constants import set_hermes_home_override, reset_hermes_home_override
from agent import wiki


def test_record_event_redacts_secrets_and_writes_profile_local_raw(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        path = wiki.record_review_event(
            actions=["Skill 'proxy' updated with token=sk-secret-value"],
            session_id="session-1",
            outcome="reviewed",
            evidence=["Authorization: Bearer abc123"],
            now="2026-09-08T10:00:00+00:00",
        )
        assert path == tmp_path / "wiki" / "raw" / "2026-09" / "20260908T100000Z-session-1.json"
        payload = json.loads(path.read_text())
        text = json.dumps(payload)
        assert "sk-secret-value" not in text
        assert "abc123" not in text
        assert "[REDACTED]" in text
    finally:
        reset_hermes_home_override(token)


def test_maintain_wiki_creates_pending_proposal_without_touching_skills(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        wiki.record_review_event(
            actions=["Skill 'proxy-network-adaptive' updated after a verified recovery"],
            session_id="session-2",
            outcome="reviewed",
            evidence=["session-2"],
            now="2026-09-08T11:00:00+00:00",
        )
        result = wiki.maintain_wiki()
        proposal = next((tmp_path / "wiki" / "proposals").glob("*.md"))
        body = proposal.read_text()
        assert result["proposals_created"] == 1
        assert "status: pending" in body
        assert "proxy-network-adaptive" in body
        assert not (tmp_path / "skills").exists()
    finally:
        reset_hermes_home_override(token)


def test_maintain_wiki_preserves_existing_index_and_proposals(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        root = tmp_path / "wiki"
        (root / "patterns").mkdir(parents=True)
        (root / "proposals").mkdir(parents=True)
        (root / "patterns" / "existing.md").write_text("# Existing", encoding="utf-8")
        existing_proposal = root / "proposals" / "existing.md"
        existing_proposal.write_text("# Keep my wording", encoding="utf-8")
        (root / "index.md").write_text(
            "# Hermes Experience Wiki\n\n## Patterns\n\n- Existing custom description\n\n## Pending proposals\n\n- Existing proposal custom description\n",
            encoding="utf-8",
        )
        wiki.record_review_event(
            actions=["Skill 'new-skill' updated"],
            session_id="session-4",
            outcome="reviewed",
            evidence=[],
            now="2026-09-08T13:00:00+00:00",
        )
        wiki.maintain_wiki()
        index = (root / "index.md").read_text(encoding="utf-8")
        assert "Existing custom description" in index
        assert "Existing proposal custom description" in index
        assert "new-skill" in index
    finally:
        reset_hermes_home_override(token)


def test_maintain_wiki_is_idempotent(tmp_path):
    token = set_hermes_home_override(tmp_path)
    try:
        wiki.record_review_event(
            actions=["Skill 'example' updated"],
            session_id="session-3",
            outcome="reviewed",
            evidence=[],
            now="2026-09-08T12:00:00+00:00",
        )
        first = wiki.maintain_wiki()
        second = wiki.maintain_wiki()
        assert first["proposals_created"] == 1
        assert second["proposals_created"] == 0
        assert len(list((tmp_path / "wiki" / "proposals").glob("*.md"))) == 1
    finally:
        reset_hermes_home_override(token)
