"""Pending write-approval digest in background review summaries (#58834)."""

from __future__ import annotations

import pytest

from agent import background_review as br
from agent import i18n
from tools import write_approval as wa


@pytest.fixture(autouse=True)
def _english(monkeypatch):
    monkeypatch.setenv("HERMES_LANGUAGE", "en")
    i18n.reset_language_cache()
    yield
    i18n.reset_language_cache()


def _record(pid: str, subsystem: str, action: str, summary: str, created_at: float = 1.0) -> dict:
    return {
        "id": pid,
        "subsystem": subsystem,
        "action": action,
        "summary": summary,
        "created_at": created_at,
        "payload": {"action": action},
    }


def test_no_pending_returns_no_lines(monkeypatch):
    monkeypatch.setattr(wa, "list_pending", lambda subsystem: [])
    assert br._pending_write_lines() == []


def test_default_notice_is_count_only(monkeypatch):
    records = [
        _record(
            "m1",
            wa.MEMORY,
            "patch",
            "background review consolidation (patch on memory): - patch span 'old' -> 'new'",
        ),
        _record("s1", wa.SKILLS, "patch", "patch: skill-x - a -> b"),
    ]
    monkeypatch.setattr(
        wa, "list_pending", lambda subsystem: [r for r in records if r["subsystem"] == subsystem]
    )

    lines = br._pending_write_lines()

    assert len(lines) == 1
    assert "2 write approval(s) pending (1 memory, 1 skill(s))" in lines[0]
    assert "/memory pending" in lines[0] and "/skills pending" in lines[0]
    # Default mode must not leak pending ids or entry content.
    assert "m1" not in lines[0] and "s1" not in lines[0]
    assert "patch span" not in lines[0]


def test_verbose_digest_lists_memory_and_skills_newest_first(monkeypatch):
    memory = [
        _record(
            "m-old",
            wa.MEMORY,
            "patch",
            "background review consolidation (patch on memory): - patch span 'old' -> 'new'",
            1.0,
        ),
        _record(
            "m-new",
            wa.MEMORY,
            "remove",
            "background review consolidation (remove on memory): - remove entry 'x'",
            3.0,
        ),
    ]
    skills = [_record("s1", wa.SKILLS, "patch", "patch: skill-x - a -> b", 2.0)]

    monkeypatch.setattr(
        wa, "list_pending", lambda subsystem: memory if subsystem == wa.MEMORY else skills
    )

    lines = br._pending_write_lines(verbose=True)
    text = "\n".join(lines)

    assert "3 write approval(s) pending (2 memory, 1 skill(s))" in text
    assert text.index("m-new") < text.index("s1") < text.index("m-old")
    assert "[Memory/remove]" in text
    assert "[Skill/patch]" in text
    assert "patch span 'old' -> 'new'" in text
    assert "background review consolidation" not in text
    assert "Approve: /memory approve <id>" in text
    assert "Reject: /memory reject <id>" in text


def test_verbose_digest_caps_items_and_reports_remainder(monkeypatch):
    records = [_record(f"m{i}", wa.MEMORY, "patch", f"- patch {i}", float(i)) for i in range(7)]
    monkeypatch.setattr(
        wa, "list_pending", lambda subsystem: records if subsystem == wa.MEMORY else []
    )

    lines = br._pending_write_lines(verbose=True, limit=5)
    text = "\n".join(lines)

    assert "7 write approval(s) pending" in text
    assert "…and 2 more" in text
    assert text.count("• ") == 5


def test_batch_action_shows_operation_count(monkeypatch):
    record = _record("m1", wa.MEMORY, "batch", "- patch a; - remove b")
    record["payload"] = {"action": "batch", "operations": [{}, {}, {}]}
    monkeypatch.setattr(
        wa, "list_pending", lambda subsystem: [record] if subsystem == wa.MEMORY else []
    )

    text = "\n".join(br._pending_write_lines(verbose=True))

    assert "[Memory/batch/3]" in text


def test_bad_created_at_does_not_crash(monkeypatch):
    records = [
        _record("bad", wa.MEMORY, "patch", "- patch a", "not-a-number"),
        _record("none", wa.MEMORY, "patch", "- patch b", None),
    ]
    monkeypatch.setattr(
        wa, "list_pending", lambda subsystem: records if subsystem == wa.MEMORY else []
    )

    text = "\n".join(br._pending_write_lines(verbose=True))

    assert "2 write approval(s) pending" in text
    assert "bad [Memory/patch]" in text


def test_verbose_summary_strips_control_characters(monkeypatch):
    record = _record("m1", wa.MEMORY, "patch", "- patch \x1b[31mred\x1b[0m")
    monkeypatch.setattr(
        wa, "list_pending", lambda subsystem: [record] if subsystem == wa.MEMORY else []
    )

    text = "\n".join(br._pending_write_lines(verbose=True))

    assert "\x1b" not in text
    assert "red" in text


def test_notice_reads_real_pending_store(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    record = wa.stage_write(
        wa.MEMORY,
        {"action": "patch", "target": "memory", "old_text": "a", "content": "b"},
        summary="background review consolidation (patch on memory): - patch span 'a' -> 'b'",
        origin="background_review",
    )

    text = "\n".join(br._pending_write_lines(verbose=True))

    assert record["id"] in text
    assert "[Memory/patch]" in text
    assert "patch span 'a' -> 'b'" in text


class _Agent:
    def __init__(self):
        self.printed: list[str] = []
        self.background_review_callback = None

    def _safe_print(self, text: str):
        self.printed.append(text)


def test_publish_emits_notice_when_only_pending(monkeypatch):
    monkeypatch.setattr(
        br,
        "_pending_write_lines",
        lambda **kwargs: [
            "📥 1 write approval(s) pending (1 memory, 0 skill(s)) — review with /memory pending or /skills pending"
        ],
    )
    agent = _Agent()

    br._publish_review_summary(agent, [])

    assert len(agent.printed) == 1
    assert agent.printed[0].startswith("  💾 Self-improvement review: ")
    assert "📥 1 write approval" in agent.printed[0]


def test_publish_passes_verbose_to_notice(monkeypatch):
    calls = []

    def fake_pending(*, verbose=False):
        calls.append(verbose)
        return ["📥 1 write approval(s) pending"]

    monkeypatch.setattr(br, "_pending_write_lines", fake_pending)
    agent = _Agent()

    br._publish_review_summary(agent, [], verbose=True)

    assert calls == [True]


def test_publish_combines_actions_and_notice(monkeypatch):
    monkeypatch.setattr(
        br,
        "_pending_write_lines",
        lambda **kwargs: ["📥 1 write approval(s) pending (1 memory, 0 skill(s))"],
    )
    agent = _Agent()

    br._publish_review_summary(agent, ["Memory updated"])

    assert "Memory updated\n📥 1 write approval" in agent.printed[0]


def test_publish_is_silent_when_nothing_to_report(monkeypatch):
    monkeypatch.setattr(br, "_pending_write_lines", lambda **kwargs: [])
    agent = _Agent()

    br._publish_review_summary(agent, [])

    assert agent.printed == []


def test_staged_memory_message_still_surfaces_without_notice():
    data = {"staged": True, "proposal_staged": True, "message": "staged for approval"}

    assert br._action_lines(data, {"tool": "memory"}, False) == ["staged for approval"]
    # A staged skill write has no proposal_staged flag; the pending notice is its notification.
    assert br._action_lines({"staged": True, "message": "staged"}, {"tool": "skill_manage"}, False) == []
