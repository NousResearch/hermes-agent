"""``pre_llm_call`` must carry the turn's trusted author (``agent._turn_author``) so a plugin can
tell a Bot-Mode DM turn from an ordinary human turn BEFORE inference, without parsing message text.
See ``agent/turn_author.py`` (identity) and ``agent/turn_context.py::_collect_pre_llm_call_context``
(the hook site)."""
from types import SimpleNamespace


def _run(monkeypatch, *, turn_author=None, user_id="", platform=""):
    from agent.turn_context import _collect_pre_llm_call_context

    captured: dict = {}

    def fake_invoke_hook(name, **kwargs):
        captured["name"] = name
        captured["kwargs"] = kwargs
        return []

    monkeypatch.setattr("hermes_cli.lifecycle.invoke_hook", fake_invoke_hook)
    agent = SimpleNamespace(
        session_id="sess-1", model="test-model", platform=platform, _user_id=user_id,
        _turn_author=turn_author, _parent_session_id=None, _persist_disabled=False,
    )
    _collect_pre_llm_call_context(
        agent, effective_task_id="task-1", turn_id="turn-1", original_user_message="hi",
        messages=[], conversation_history=[],
    )
    return captured["kwargs"]


class TestPreLLMCallTurnAuthor:
    def test_ordinary_human_turn_has_no_turn_author(self, monkeypatch):
        kwargs = _run(monkeypatch, turn_author=None, user_id="12345", platform="telegram")
        assert kwargs["turn_author"] is None
        assert kwargs["sender_id"] == "12345"
        assert kwargs["platform"] == "telegram"

    def test_local_bot_dm_turn_carries_a_structured_bot_author(self, monkeypatch):
        author = {"id": "bot:watch_tracker", "name": "watch_tracker", "is_bot": True}
        kwargs = _run(monkeypatch, turn_author=author)
        assert kwargs["turn_author"] == author
        assert kwargs["turn_author"]["is_bot"] is True

    def test_remote_bot_dm_turn_author_carries_its_origin(self, monkeypatch):
        author = {"id": "bot:mini/watch_tracker", "name": "watch_tracker", "is_bot": True}
        kwargs = _run(monkeypatch, turn_author=author)
        from agent.turn_author import bot_origin

        assert kwargs["turn_author"] == author
        assert bot_origin(kwargs["turn_author"]) == "mini"

    def test_local_bot_origin_classifies_as_local(self, monkeypatch):
        author = {"id": "bot:watch_tracker", "name": "watch_tracker", "is_bot": True}
        kwargs = _run(monkeypatch, turn_author=author)
        from agent.turn_author import bot_origin

        assert bot_origin(kwargs["turn_author"]) == "local"

    def test_unverified_sender_has_no_bot_author(self, monkeypatch):
        # No door stamped a turn author: an unknown/unverified sender must not be reported as a bot.
        kwargs = _run(monkeypatch, turn_author=None)
        from agent.turn_author import bot_origin

        assert kwargs["turn_author"] is None
        assert bot_origin(kwargs["turn_author"]) is None
