"""``/suggestions accept``'s origin capture must match cron's canonical shape.

Same bug class as ``tests/hermes_cli/test_blueprint_cmd_origin.py``: the
module's own docstring claimed to "mirror cron's ``_origin_from_env``" but
actually mirrored its pre-``8b6cf434cb`` shape (no ``scope_id``/``user_id``,
no Slack synthetic-thread guard). ``accept_suggestion`` merges this origin
straight into the persisted job spec (``cron/suggestions.py``), so the gap
baked a scope_id-less origin into every job created via ``/suggestions
accept`` in a scoped Slack workspace.
"""

from unittest.mock import patch

from hermes_cli.suggestions_cmd import _resolve_origin


def _session_env(env: dict):
    return patch(
        "gateway.session_context.get_session_env",
        side_effect=lambda name, default="": env.get(name, default),
    )


class TestSuggestionsResolveOrigin:
    def test_captures_scope_id_and_user_id(self):
        env = {
            "HERMES_SESSION_PLATFORM": "slack",
            "HERMES_SESSION_CHAT_ID": "D0BJTDCSR7C",
            "HERMES_SESSION_SCOPE_ID": "T0WORKSPACE",
            "HERMES_SESSION_USER_ID": "U0USER",
        }
        with _session_env(env):
            origin = _resolve_origin()
        assert origin is not None
        assert origin["platform"] == "slack"
        assert origin["chat_id"] == "D0BJTDCSR7C"
        assert origin["scope_id"] == "T0WORKSPACE"
        assert origin["user_id"] == "U0USER"

    def test_drops_synthetic_slack_per_message_thread(self):
        env = {
            "HERMES_SESSION_PLATFORM": "slack",
            "HERMES_SESSION_CHAT_ID": "D0BJTDCSR7C",
            "HERMES_SESSION_THREAD_ID": "1755043010.123456",
            "HERMES_SESSION_MESSAGE_ID": "1755043010.123456",
        }
        with _session_env(env):
            origin = _resolve_origin()
        assert origin is not None
        assert origin["thread_id"] is None

    def test_no_session_env_returns_none(self):
        with _session_env({}):
            assert _resolve_origin() is None
