import json
from types import SimpleNamespace

import pytest

from plugins.platforms.discord.adapter import DISCORD_AVAILABLE, NewsArticleFeedbackView


@pytest.mark.skipif(not DISCORD_AVAILABLE, reason="discord.py is not installed")
@pytest.mark.asyncio
async def test_intelligence_article_read_button_defers_and_persists_article_feedback(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "hermes_constants.get_hermes_home",
        lambda: tmp_path,
    )
    view = NewsArticleFeedbackView(allowed_user_ids=set())
    calls = []

    async def defer(**kwargs):
        calls.append(("defer", kwargs))

    async def add_reaction(emoji):
        calls.append(("add_reaction", emoji))

    async def followup_send(content, **kwargs):
        calls.append(("followup", content, kwargs))

    interaction = SimpleNamespace(
        guild=SimpleNamespace(id=1),
        user=SimpleNamespace(id=123, display_name="hiro"),
        channel=SimpleNamespace(id=456),
        message=SimpleNamespace(
            id=789,
            content="## #ai-tools 編集部記事\n\n### Useful article\n\n本文",
            add_reaction=add_reaction,
            embeds=[],
        ),
        response=SimpleNamespace(defer=defer),
        followup=SimpleNamespace(send=followup_send),
    )

    await view._mark(interaction, "read", "読んだ")

    assert calls[0] == ("defer", {"ephemeral": True, "thinking": False})
    assert calls[1] == ("add_reaction", "✅")
    assert calls[2] == ("followup", "✅ 読んだ として記録しました〜", {"ephemeral": True})

    lines = (tmp_path / "state" / "news_feedback.jsonl").read_text(encoding="utf-8").splitlines()
    record = json.loads(lines[0])
    assert record["reaction"] == "read"
    assert record["reaction_label"] == "読んだ"
    assert record["custom_id"] == "intelligence_article_read"
    assert record["feedback_scope"] == "article"
