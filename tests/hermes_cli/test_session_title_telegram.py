"""``hermes sessions rename`` / console ``sessions rename`` Telegram topic propagation.

The CLI rename surfaces historically wrote only the session database — the Telegram topic
kept its old name unless the user renamed it by hand (the gateway's ``/title`` lane already
propagates). These tests pin the shared helper's gating rules so both CLI surfaces keep
agreeing with the gateway lane:

- no binding / topic mode off → no Telegram call;
- operator kill-switch ``disable_topic_auto_rename`` → no Telegram call;
- operator-declared ``dm_topics`` → no Telegram call (their names are operator config);
- happy path → ``editForumTopic`` fires with the sanitized (≤120 char) name.
"""

from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli.session_title_telegram import (
    _telegram_topic_name,
    propagate_session_title_to_telegram_topic,
)


class _FakeDB:
    """Minimal SessionDB double: only the two topic APIs the helper needs."""

    def __init__(self, binding=None, topic_mode=True):
        self._binding = binding
        self._topic_mode = topic_mode

    def get_telegram_topic_binding_by_session(self, *, session_id):
        return self._binding if session_id == "s1" else None

    def is_telegram_topic_mode_enabled(self, *, chat_id, user_id, profile_name):
        return self._topic_mode


_BINDING = {
    "chat_id": "111111111",
    "thread_id": "222222",
    "user_id": "111111111",
    "profile_name": "default",
    "session_id": "s1",
}


def _run(db, title="New Title"):
    """Call the helper with everything after the gating steps stubbed out."""
    with (
        patch(
            "hermes_cli.session_title_telegram._telegram_extra_config", return_value={}
        ),
        patch("hermes_cli.session_title_telegram._telegram_token", return_value="tok"),
        patch("hermes_cli.session_title_telegram._rename_telegram_topic") as rename,
        patch("model_tools._run_async", side_effect=lambda coro: coro.close()),
    ):
        result = propagate_session_title_to_telegram_topic(db, "s1", title)
    return result, rename


def test_no_binding_is_a_noop():
    result, rename = _run(_FakeDB(binding=None))
    assert result is False
    rename.assert_not_called()


def test_topic_mode_disabled_is_a_noop():
    result, rename = _run(_FakeDB(binding=_BINDING, topic_mode=False))
    assert result is False
    rename.assert_not_called()


def test_operator_kill_switch_is_a_noop():
    db = _FakeDB(binding=_BINDING)
    with (
        patch(
            "hermes_cli.session_title_telegram._telegram_extra_config",
            return_value={"disable_topic_auto_rename": True},
        ),
        patch("hermes_cli.session_title_telegram._telegram_token", return_value="tok"),
        patch("hermes_cli.session_title_telegram._rename_telegram_topic") as rename,
    ):
        result = propagate_session_title_to_telegram_topic(db, "s1", "New Title")
    assert result is False
    rename.assert_not_called()


def test_operator_declared_topics_are_skipped():
    db = _FakeDB(binding=_BINDING)
    with (
        patch(
            "hermes_cli.session_title_telegram._telegram_extra_config",
            return_value={
                "dm_topics": [{"chat_id": "111111111", "topics": [{"name": "Pinned"}]}]
            },
        ),
        patch("hermes_cli.session_title_telegram._telegram_token", return_value="tok"),
        patch("hermes_cli.session_title_telegram._rename_telegram_topic") as rename,
    ):
        result = propagate_session_title_to_telegram_topic(db, "s1", "New Title")
    assert result is False
    rename.assert_not_called()


def test_missing_token_is_a_noop():
    db = _FakeDB(binding=_BINDING)
    with (
        patch(
            "hermes_cli.session_title_telegram._telegram_extra_config", return_value={}
        ),
        patch("hermes_cli.session_title_telegram._telegram_token", return_value=None),
        patch("hermes_cli.session_title_telegram._rename_telegram_topic") as rename,
    ):
        result = propagate_session_title_to_telegram_topic(db, "s1", "New Title")
    assert result is False
    rename.assert_not_called()


def test_bound_session_renames_the_topic():
    result, rename = _run(_FakeDB(binding=_BINDING))
    assert result is True
    rename.assert_called_once()
    assert rename.call_args.args[:2] == ("111111111", "222222")
    assert rename.call_args.args[2] == "New Title"


def test_db_lookup_failure_never_raises():
    class _BrokenDB(_FakeDB):
        def get_telegram_topic_binding_by_session(self, *, session_id):
            raise RuntimeError("boom")

    result, rename = _run(_BrokenDB(binding=_BINDING))
    assert result is False
    rename.assert_not_called()


def test_telegram_failure_never_raises():
    db = _FakeDB(binding=_BINDING)

    def _boom(coro):
        coro.close()
        raise RuntimeError("telegram down")

    with (
        patch(
            "hermes_cli.session_title_telegram._telegram_extra_config", return_value={}
        ),
        patch("hermes_cli.session_title_telegram._telegram_token", return_value="tok"),
        patch("model_tools._run_async", side_effect=_boom),
    ):
        assert propagate_session_title_to_telegram_topic(db, "s1", "New Title") is False


def test_title_sanitization_mirrors_gateway_lane():
    assert _telegram_topic_name("  a\n  b  ") == "a b"
    assert _telegram_topic_name("x" * 200) == "x" * 117 + "..."
    assert _telegram_topic_name("") == ""


def test_simple_namespace_import_shape():
    # The helper must stay importable without a live gateway (CLI-only entry point).
    import hermes_cli.session_title_telegram as mod

    assert callable(mod.propagate_session_title_to_telegram_topic)
    assert isinstance(SimpleNamespace(), SimpleNamespace)
