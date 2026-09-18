"""Behaviour contract for the dashboard main-model change notifier.

Switching the main model from the dashboard (``POST /api/model/set``) is otherwise silent: no
gateway turn is involved, so none of the in-chat notifications (manual ``/model``, automatic
fallback, gateway restart) cover it. A user who leaves a dashboard tab on the model picker can
move a running session onto a model they did not consciously choose and never be told.

What is pinned here is the contract, not the message text:

- a switch that actually changes provider or model notifies exactly once;
- a no-op switch (same provider AND same model) notifies nobody -- the dashboard re-saves the
  current selection on unrelated edits, so firing on every call would train the user to ignore it;
- a profile with no Telegram credentials is a silent skip, never an error;
- a failing sender never propagates: the notification is best-effort and must not fail the model
  switch that triggered it.
"""

from types import SimpleNamespace
from unittest.mock import patch

from hermes_cli import web_server_config as wsc
from hermes_cli.web_server_config import _notify_model_change_via_telegram

_SENDER_SEAM = "tools.send_message_tool._send_telegram"


def _recording_sender():
    """An async stand-in for the Telegram sender that records its calls."""
    calls = []

    async def _sender(token, chat_id, message, *args, **kwargs):
        calls.append({"token": token, "chat_id": chat_id, "message": message})

    return calls, _sender


def _creds(token="bot-token", chat_id="12345"):
    return patch(
        "hermes_cli.web_server_config._read_profile_telegram_creds",
        return_value=(token, chat_id),
    )


class TestNotifiesOnRealChange:
    def test_provider_change_notifies_once(self):
        calls, sender = _recording_sender()
        with _creds(), patch(_SENDER_SEAM, sender):
            _notify_model_change_via_telegram("nous", "upstage/solar-pro4:free", "openrouter", "deepseek/deepseek-v4.1-flash")

        assert len(calls) == 1
        message = calls[0]["message"]
        # Both ends of the switch must be visible, or the notice cannot be acted on.
        assert "upstage/solar-pro4:free" in message
        assert "deepseek/deepseek-v4.1-flash" in message

    def test_model_change_on_same_provider_notifies(self):
        calls, sender = _recording_sender()
        with _creds(), patch(_SENDER_SEAM, sender):
            _notify_model_change_via_telegram("openrouter", "deepseek/deepseek-v4.1-flash", "openrouter", "deepseek/deepseek-v4-flash-0731:free")

        assert len(calls) == 1

    def test_uses_the_profiles_own_credentials(self):
        calls, sender = _recording_sender()
        with _creds(token="merry-bot-token", chat_id="555"), patch(_SENDER_SEAM, sender):
            _notify_model_change_via_telegram("nous", "a/b", "openrouter", "c/d")

        assert calls[0]["token"] == "merry-bot-token"
        assert calls[0]["chat_id"] == "555"


class TestStaysSilent:
    def test_identical_assignment_notifies_nobody(self):
        calls, sender = _recording_sender()
        with _creds(), patch(_SENDER_SEAM, sender):
            _notify_model_change_via_telegram("openrouter", "deepseek/deepseek-v4.1-flash", "openrouter", "deepseek/deepseek-v4.1-flash")

        assert calls == []

    def test_missing_credentials_is_a_silent_skip(self):
        calls, sender = _recording_sender()
        with _creds(token="", chat_id=""), patch(_SENDER_SEAM, sender):
            _notify_model_change_via_telegram("nous", "a/b", "openrouter", "c/d")

        assert calls == []

    def test_missing_chat_id_is_a_silent_skip(self):
        calls, sender = _recording_sender()
        with _creds(token="bot-token", chat_id=""), patch(_SENDER_SEAM, sender):
            _notify_model_change_via_telegram("nous", "a/b", "openrouter", "c/d")

        assert calls == []


class TestNeverBlocksTheSwitch:
    def test_failing_sender_does_not_propagate(self):
        async def _exploding_sender(*args, **kwargs):
            raise RuntimeError("telegram is down")

        with _creds(), patch(_SENDER_SEAM, _exploding_sender):
            # Must return normally: the model switch itself already succeeded by this point.
            _notify_model_change_via_telegram("nous", "a/b", "openrouter", "c/d")

    def test_unreadable_credentials_file_does_not_propagate(self):
        with patch(
            "hermes_cli.web_server_config._read_profile_telegram_creds",
            side_effect=OSError("no such file"),
        ):
            _notify_model_change_via_telegram("nous", "a/b", "openrouter", "c/d")


class TestAssignmentPathInvokesTheNotifier:
    """Pins the WIRING, which the unit tests above cannot.

    They call the notifier directly, so they stay green even with the call site deleted from
    ``_apply_main_assignment_sync`` -- verified by disabling it. This class exercises the real
    assignment function instead, so it fails the moment the dashboard path stops notifying.
    """

    def _run_assignment(self, cfg, result, recorder):
        with patch.object(wsc, "_prepare_main_assignment", return_value=("", result)), \
             patch.object(wsc, "_apply_main_model_assignment", return_value={"base_url": ""}), \
             patch.object(wsc, "_resolve_assignment_credentials"), \
             patch.object(wsc, "_notify_model_change_via_telegram", recorder), \
             patch("hermes_cli.config.save_config"):
            wsc._apply_main_assignment_sync(
                cfg, result.target_provider, result.new_model, "", ""
            )

    def test_assignment_reports_the_model_it_replaced(self):
        cfg = {"model": {"provider": "nous", "default": "upstage/solar-pro4:free"}}
        result = SimpleNamespace(
            target_provider="openrouter", new_model="deepseek/deepseek-v4.1-flash"
        )
        seen = []
        self._run_assignment(cfg, result, lambda *args: seen.append(args))

        # Previous pair must come from the config being replaced -- reading the post-switch
        # config here would report the model as having replaced itself.
        assert seen == [
            ("nous", "upstage/solar-pro4:free", "openrouter", "deepseek/deepseek-v4.1-flash")
        ]
