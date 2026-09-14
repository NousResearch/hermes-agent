"""api_server cron transcript delivery (ITEM-32).

An api_server-origin cron job's output was never deliverable: the platform's
``send()`` is a permanent stub ("API server uses HTTP request/response, not
send()"), so both the live-adapter and standalone lanes dead-ended with
"live adapter send failed". Worse, preflight blocked job creation outright
because api_server is never in the gateway's connected-platform set, pushing
operators toward channels that also cannot reach an api_server session.

The fix delivers api_server targets by appending to the target session's
transcript — the same visibility model gateway/wake.py uses for this
platform — and admits the platform at preflight without a credential gate.

These tests pin: transcript append on a live session, honest error on a
missing one, rotation-lineage follow, preflight admission, and bare-token
origin resolution.
"""

from unittest.mock import MagicMock, patch

from cron import scheduler_delivery as sd
from cron.scheduler_preflight import _preflight_check_delivery


def _job(deliver="api_server", origin=None):
    job = {"id": "j-item32", "name": "item32", "deliver": deliver}
    if origin is not None:
        job["origin"] = {"platform": origin[0], "chat_id": origin[1]}
    return job


class TestTranscriptDelivery:
    def test_appends_to_existing_session(self):
        db = MagicMock()
        db.get_session.return_value = {"session_id": "sess-1"}
        with patch("hermes_state.SessionDB", return_value=db):
            err = sd._deliver_to_api_server_transcript(
                _job(), "sess-1", "hello")
        assert err is None
        db.append_message.assert_called_once()
        kwargs = db.append_message.call_args.kwargs
        assert kwargs["session_id"] == "sess-1"
        assert kwargs["role"] == "user"
        assert "hello" in kwargs["content"]
        db.close.assert_called_once()

    def test_missing_session_is_an_honest_error(self):
        db = MagicMock()
        db.get_session.return_value = None
        db.resolve_resume_session_id.return_value = None
        with patch("hermes_state.SessionDB", return_value=db):
            err = sd._deliver_to_api_server_transcript(
                _job(), "sess-gone", "hello")
        assert err is not None
        assert "sess-gone does not exist" in err
        assert "send failed" not in err
        db.append_message.assert_not_called()

    def test_follows_rotation_lineage(self):
        db = MagicMock()
        db.get_session.side_effect = [None, {"session_id": "sess-new"}]
        db.resolve_resume_session_id.return_value = "sess-new"
        with patch("hermes_state.SessionDB", return_value=db):
            err = sd._deliver_to_api_server_transcript(
                _job(), "sess-old", "hello")
        assert err is None
        assert db.append_message.call_args.kwargs["session_id"] == "sess-new"

    def test_empty_content_skips_db(self):
        err = sd._deliver_to_api_server_transcript(_job(), "sess-1", "  ")
        assert err is None

    def test_append_failure_is_reported(self):
        db = MagicMock()
        db.get_session.return_value = {"session_id": "sess-1"}
        db.append_message.side_effect = RuntimeError("db locked")
        with patch("hermes_state.SessionDB", return_value=db):
            err = sd._deliver_to_api_server_transcript(
                _job(), "sess-1", "hello")
        assert err is not None
        assert "append" in err


class TestDeliveryLoopLane:
    def test_api_server_target_uses_transcript_not_adapter(self):
        """The loop must route api_server targets to the transcript helper,
        never to _prepare_target_delivery (whose lanes dead-end in send())."""
        calls = {}

        def fake_transcript(job, chat_id, content):
            calls["args"] = (chat_id, content)
            return None

        job = _job()
        with patch.object(sd, "_deliver_to_api_server_transcript",
                          side_effect=fake_transcript), \
             patch.object(sd, "_prepare_target_delivery") as prep, \
             patch.object(sd, "_resolve_delivery_targets",
                          return_value=[{"platform": "api_server",
                                         "chat_id": "sess-9",
                                         "thread_id": None}]):
            err = sd._deliver_result(job, "payload")
        assert err is None
        assert calls["args"] == ("sess-9", "payload")
        prep.assert_not_called()


class TestPreflightAdmission:
    def test_api_server_never_hits_credential_gate(self):
        """api_server must be admitted without gateway credentials, even when
        the gateway reports it unconnected."""
        with patch("gateway.config.load_gateway_config") as load:
            cfg = MagicMock()
            cfg.get_connected_platforms.return_value = set()
            load.return_value = cfg
            assert _preflight_check_delivery(_job()) is None

    def test_unknown_platform_still_blocks(self):
        assert _preflight_check_delivery(_job(deliver="nosuchplat")) is not None


class TestBareTokenResolution:
    def _resolve(self, job, value):
        return sd._resolve_single_delivery_target(job, value)

    def test_bare_token_uses_origin_session(self):
        target = self._resolve(
            _job(origin=("api_server", "sess-origin")), "api_server")
        assert target is not None
        assert target["platform"] == "api_server"
        assert target["chat_id"] == "sess-origin"

    def test_bare_token_without_origin_does_not_resolve(self):
        assert self._resolve(_job(), "api_server") is None

    def test_explicit_session_target_passes_through(self):
        """api_server:<session-id> is handed to the resolver unchanged (the
        transcript helper validates existence at delivery time)."""
        with patch("tools.send_message_tool.prepare_send_message_platforms"), \
             patch("tools.send_message_tool.resolve_send_target",
                   return_value=("sess-x", None, None)) as rst:
            target = self._resolve(_job(), "api_server:sess-x")
        assert target is not None
        assert target["chat_id"] == "sess-x"
        assert rst.call_args.args[1] == "sess-x"
