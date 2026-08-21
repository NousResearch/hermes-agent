"""``/blueprint``'s origin capture must match cron's canonical shape.

``_resolve_origin`` previously hand-rolled a truncated origin dict (platform,
chat_id, chat_name, thread_id only) instead of delegating to
``tools.cronjob_tools._origin_from_env`` — the function ``8b6cf434cb`` fixed to
carry ``scope_id`` (embedded in every scoped Slack session key) and drop the
Slack synthetic per-message thread stamp. A ``/blueprint`` job created via the
deterministic ``slot=val`` shortcut baked the truncated shape permanently into
the job, reproducing the exact "continuation amnesia" bug in a scoped Slack
workspace that ``8b6cf434cb`` fixed for the main creation path.
"""

from unittest.mock import patch

from hermes_cli.blueprint_cmd import _resolve_origin


def _session_env(env: dict):
    return patch(
        "gateway.session_context.get_session_env",
        side_effect=lambda name, default="": env.get(name, default),
    )


class TestBlueprintResolveOrigin:
    def test_explicit_origin_passthrough(self):
        """An explicit origin (dashboard/API caller) is returned untouched."""
        explicit = {"platform": "slack", "chat_id": "C1"}
        assert _resolve_origin(explicit) is explicit

    def test_captures_scope_id_and_user_id(self):
        env = {
            "HERMES_SESSION_PLATFORM": "slack",
            "HERMES_SESSION_CHAT_ID": "D0BJTDCSR7C",
            "HERMES_SESSION_SCOPE_ID": "T0WORKSPACE",
            "HERMES_SESSION_USER_ID": "U0USER",
        }
        with _session_env(env):
            origin = _resolve_origin(None)
        assert origin is not None
        assert origin["platform"] == "slack"
        assert origin["chat_id"] == "D0BJTDCSR7C"
        assert origin["scope_id"] == "T0WORKSPACE"
        assert origin["user_id"] == "U0USER"

    def test_drops_synthetic_slack_per_message_thread(self):
        """thread_id == message_id on Slack is a session-keying stamp, not a
        durable thread — must be dropped, same as cron's own creation path."""
        env = {
            "HERMES_SESSION_PLATFORM": "slack",
            "HERMES_SESSION_CHAT_ID": "D0BJTDCSR7C",
            "HERMES_SESSION_THREAD_ID": "1755043010.123456",
            "HERMES_SESSION_MESSAGE_ID": "1755043010.123456",
        }
        with _session_env(env):
            origin = _resolve_origin(None)
        assert origin is not None
        assert origin["thread_id"] is None

    def test_no_session_env_returns_none(self):
        with _session_env({}):
            assert _resolve_origin(None) is None
