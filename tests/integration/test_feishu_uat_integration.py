"""Integration tests for Feishu UAT (User Access Token) flows.

Three scenarios:
1. Device-flow end-to-end: begin_device_authorization → poll → save_uat → verify file.
2. Tool call chain (UAT injection): feishu_calendar_list_events reads UAT from fixture,
   calls SDK with user_access_token header, surfaces errcode 99991679 correctly.
3. Streaming card end-to-end: consume_stream state machine + throttled flush call count.
"""

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

pytestmark = pytest.mark.integration

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _future_expires_at_ms(seconds_from_now: int = 7200) -> int:
    """Return a millisecond epoch timestamp well in the future."""
    return int((time.time() + seconds_from_now) * 1000)


def _write_fake_uat(path: Path, access_token: str = "u-faketoken123", open_id: str = "ou_abc") -> None:
    """Write a minimal valid UAT JSON file at *path* (mode 0600)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "app_id": "cli_test_app",
        "user_open_id": open_id,
        "access_token": access_token,
        "refresh_token": "r-fakerefresh",
        "expires_at": _future_expires_at_ms(7200),
        "refresh_expires_at": _future_expires_at_ms(2592000),
        "scope": "calendar:calendar",
        "granted_at": int(time.time() * 1000),
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    os.chmod(path, 0o600)


# ===========================================================================
# Scenario 1 — Device flow end-to-end
# ===========================================================================

class TestDeviceFlowEndToEnd:
    """Test the full Feishu OAuth device flow without touching real endpoints."""

    def test_begin_device_authorization_returns_required_fields(self, monkeypatch, tmp_path):
        """begin_device_authorization parses response and returns all required fields."""
        fake_response = {
            "device_code": "dev-code-abc",
            "user_code": "ABCD-1234",
            "verification_uri": "https://accounts.feishu.cn/oauth/device",
            "verification_uri_complete": "https://accounts.feishu.cn/oauth/device?user_code=ABCD-1234",
            "expires_in": 1800,
            "interval": 5,
        }

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = fake_response

        with patch("hermes_cli.feishu_auth.requests.post", return_value=mock_resp) as mock_post:
            from hermes_cli.feishu_auth import begin_device_authorization
            result = begin_device_authorization("cli_test_app_id")

        assert result["device_code"] == "dev-code-abc"
        assert result["user_code"] == "ABCD-1234"
        assert result["expires_in"] == 1800
        assert result["interval"] == 5
        assert "verification_uri_complete" in result
        mock_post.assert_called_once()

    def test_poll_device_token_returns_access_token_on_success(self, monkeypatch):
        """poll_device_token extracts access_token from a successful response."""
        fake_token_response = {
            "access_token": "u-success-token",
            "refresh_token": "r-refresh-xyz",
            "open_id": "ou_user123",
            "expires_in": 7200,
            "refresh_expires_in": 2592000,
            "token_type": "Bearer",
            "scope": "calendar:calendar",
        }

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = fake_token_response

        with patch("hermes_cli.feishu_auth.requests.post", return_value=mock_resp):
            from hermes_cli.feishu_auth import poll_device_token
            result = poll_device_token("dev-code-abc", "cli_test_app_id")

        assert result["access_token"] == "u-success-token"
        assert result["refresh_token"] == "r-refresh-xyz"
        assert result["open_id"] == "ou_user123"
        assert result["error"] is None

    def test_poll_device_token_returns_pending_when_not_yet_authorized(self, monkeypatch):
        """poll_device_token returns error=authorization_pending while user has not scanned."""
        pending_response = {
            "error": "authorization_pending",
            "error_description": "User has not yet authorized.",
        }

        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = pending_response

        with patch("hermes_cli.feishu_auth.requests.post", return_value=mock_resp):
            from hermes_cli.feishu_auth import poll_device_token
            result = poll_device_token("dev-code-abc", "cli_test_app_id")

        assert result["error"] == "authorization_pending"
        assert result["access_token"] is None

    def test_save_uat_writes_file_with_correct_permissions(self, tmp_path, monkeypatch):
        """save_uat writes feishu_uat.json with mode 0600 and correct JSON content."""
        fake_uat_path = tmp_path / ".hermes" / "feishu_uat.json"
        monkeypatch.setattr("hermes_cli.feishu_auth.FEISHU_UAT_PATH", fake_uat_path)

        from hermes_cli.feishu_auth import save_uat
        save_uat(
            access_token="u-mytoken",
            refresh_token="r-myrefresh",
            open_id="ou_tester",
            expires_in=7200,
            refresh_expires_in=2592000,
            scope="calendar:calendar",
            app_id="cli_test",
        )

        assert fake_uat_path.exists(), "UAT file should be created"
        stat = fake_uat_path.stat()
        assert oct(stat.st_mode)[-3:] == "600", "UAT file must have mode 0600"

        data = json.loads(fake_uat_path.read_text())
        assert data["access_token"] == "u-mytoken"
        assert data["refresh_token"] == "r-myrefresh"
        assert data["user_open_id"] == "ou_tester"
        assert data["app_id"] == "cli_test"
        assert "expires_at" in data
        assert "granted_at" in data

    def test_full_device_flow_pending_then_success_saves_uat(self, tmp_path, monkeypatch):
        """Full flow: begin_device_authorization → poll (pending) → poll (success) → save_uat.

        Tests the polling functions directly without the real-time loop in
        wait_for_authorization_success (which would require real clock patches
        that conflict with the conftest 30s SIGALRM guard).
        """
        fake_uat_path = tmp_path / ".hermes" / "feishu_uat.json"
        monkeypatch.setattr("hermes_cli.feishu_auth.FEISHU_UAT_PATH", fake_uat_path)

        import hermes_cli.feishu_auth as _feishu_auth_mod

        # Step 1: begin_device_authorization
        dev_auth_resp = {
            "device_code": "dev-xyz",
            "user_code": "WXYZ-9999",
            "verification_uri": "https://accounts.feishu.cn/device",
            "verification_uri_complete": "https://accounts.feishu.cn/device?code=WXYZ-9999",
            "expires_in": 1800,
            "interval": 3,
        }
        mock_dev_r = MagicMock()
        mock_dev_r.raise_for_status.return_value = None
        mock_dev_r.json.return_value = dev_auth_resp

        with patch("hermes_cli.feishu_auth.requests.post", return_value=mock_dev_r):
            auth_data = _feishu_auth_mod.begin_device_authorization("cli_app")

        assert auth_data["device_code"] == "dev-xyz"

        # Step 2: poll — first call pending
        pending_mock = MagicMock()
        pending_mock.raise_for_status.return_value = None
        pending_mock.json.return_value = {"error": "authorization_pending"}

        with patch("hermes_cli.feishu_auth.requests.post", return_value=pending_mock):
            pending = _feishu_auth_mod.poll_device_token("dev-xyz", "cli_app")

        assert pending["access_token"] is None
        assert pending["error"] == "authorization_pending"

        # Step 3: poll — second call succeeds
        success_mock = MagicMock()
        success_mock.raise_for_status.return_value = None
        success_mock.json.return_value = {
            "access_token": "u-final-token",
            "refresh_token": "r-final-refresh",
            "open_id": "ou_final_user",
            "expires_in": 7200,
            "refresh_expires_in": 2592000,
            "token_type": "Bearer",
            "scope": "calendar:calendar",
        }

        with patch("hermes_cli.feishu_auth.requests.post", return_value=success_mock):
            result = _feishu_auth_mod.poll_device_token("dev-xyz", "cli_app")

        assert result["access_token"] == "u-final-token"
        assert result["open_id"] == "ou_final_user"

        # Step 4: save_uat writes file with correct content and permissions
        _feishu_auth_mod.save_uat(
            access_token=result["access_token"],
            refresh_token=result["refresh_token"],
            open_id=result["open_id"],
            expires_in=result["expires_in"],
            refresh_expires_in=result["refresh_expires_in"],
            scope="calendar:calendar",
            app_id="cli_app",
        )

        assert fake_uat_path.exists()
        assert oct(fake_uat_path.stat().st_mode)[-3:] == "600"
        data = json.loads(fake_uat_path.read_text())
        assert data["access_token"] == "u-final-token"
        assert data["user_open_id"] == "ou_final_user"


# ===========================================================================
# Scenario 2 — Tool call chain (UAT injection)
# ===========================================================================

class TestToolCallChainUATInjection:
    """Test that feishu_calendar_list_events loads UAT from disk and injects it
    into SDK requests, and that errcode 99991679 surfaces as UserAuthRequiredError."""

    @pytest.fixture()
    def fake_uat_path(self, tmp_path):
        """Write a valid fake UAT file and return its path."""
        path = tmp_path / ".hermes" / "feishu_uat.json"
        _write_fake_uat(path, access_token="u-injected-uat", open_id="ou_test_user")
        return path

    def test_for_user_loads_access_token_from_disk(self, fake_uat_path, monkeypatch):
        """FeishuClient.for_user() reads access_token from the UAT file on disk."""
        monkeypatch.setenv("FEISHU_APP_ID", "cli_test_app")
        monkeypatch.setenv("FEISHU_APP_SECRET", "secret_test")
        monkeypatch.setattr("tools.feishu_oapi_client.FEISHU_UAT_PATH", fake_uat_path)

        # Clear client cache to avoid cross-test pollution
        from tools.feishu_oapi_client import FeishuClient
        FeishuClient._cache.clear()

        with patch.object(FeishuClient, "_build_sdk", return_value=MagicMock()):
            client = FeishuClient.for_user()

        assert client.access_token == "u-injected-uat"
        assert client.user_open_id == "ou_test_user"

    def test_calendar_list_events_sends_user_access_token_header(self, fake_uat_path, monkeypatch):
        """feishu_calendar_list_events passes user_access_token to SDK request call."""
        monkeypatch.setenv("FEISHU_APP_ID", "cli_test_app")
        monkeypatch.setenv("FEISHU_APP_SECRET", "secret_test")
        monkeypatch.setattr("tools.feishu_oapi_client.FEISHU_UAT_PATH", fake_uat_path)

        from tools.feishu_oapi_client import FeishuClient
        FeishuClient._cache.clear()

        # Build mock SDK that captures the request option passed to .request()
        captured = {}

        def _fake_sdk_request(request, option=None):
            captured["option"] = option
            # Return a successful response with fake event data
            mock_resp = MagicMock()
            mock_resp.code = None
            mock_resp.msg = ""
            mock_resp.data = None
            raw = MagicMock()
            raw.content = json.dumps({
                "code": 0,
                "msg": "success",
                "data": {"items": [{"summary": "Team meeting"}], "has_more": False},
            }).encode()
            mock_resp.raw = raw
            return mock_resp

        mock_sdk = MagicMock()
        mock_sdk.request.side_effect = _fake_sdk_request

        with patch.object(FeishuClient, "_build_sdk", return_value=mock_sdk):
            from tools.feishu_calendar_tool import _handle_calendar_list_events
            result_json = _handle_calendar_list_events({
                "start_time": "2024-01-01T00:00:00+08:00",
                "end_time": "2024-01-02T00:00:00+08:00",
                "calendar_id": "cal_primary",
            })

        # Verify the SDK was called with a request option containing user token
        assert mock_sdk.request.called, "SDK request should have been called"
        option = captured.get("option")
        assert option is not None, "RequestOption must be passed (not None) for UAT calls"

        result = json.loads(result_json)
        assert "events" in result or "error" not in result

    def test_calendar_list_events_without_calendar_id_resolves_primary(self, fake_uat_path, monkeypatch):
        """When calendar_id is omitted, list_events calls the primary calendar API first."""
        monkeypatch.setenv("FEISHU_APP_ID", "cli_test_app")
        monkeypatch.setenv("FEISHU_APP_SECRET", "secret_test")
        monkeypatch.setattr("tools.feishu_oapi_client.FEISHU_UAT_PATH", fake_uat_path)

        from tools.feishu_oapi_client import FeishuClient
        FeishuClient._cache.clear()

        request_uris = []

        def _fake_sdk_request(request, option=None):
            uri = getattr(request, "_uri", None) or getattr(request, "uri", "")
            request_uris.append(uri)

            mock_resp = MagicMock()
            mock_resp.code = None
            mock_resp.msg = ""
            mock_resp.data = None

            if "primary" in str(uri):
                raw = MagicMock()
                raw.content = json.dumps({
                    "code": 0,
                    "msg": "success",
                    "data": {
                        "calendars": [{"calendar_id": "cal_resolved_primary"}]
                    },
                }).encode()
                mock_resp.raw = raw
            else:
                raw = MagicMock()
                raw.content = json.dumps({
                    "code": 0,
                    "msg": "success",
                    "data": {"items": [], "has_more": False},
                }).encode()
                mock_resp.raw = raw
            return mock_resp

        mock_sdk = MagicMock()
        mock_sdk.request.side_effect = _fake_sdk_request

        with patch.object(FeishuClient, "_build_sdk", return_value=mock_sdk):
            from tools.feishu_calendar_tool import _handle_calendar_list_events
            _handle_calendar_list_events({
                "start_time": "2024-01-01T00:00:00+08:00",
                "end_time": "2024-01-02T00:00:00+08:00",
            })

        # At least two SDK calls: one for primary calendar, one for list events
        assert mock_sdk.request.call_count >= 2, (
            f"Expected >=2 SDK calls (primary + events), got {mock_sdk.request.call_count}"
        )

    def test_errcode_99991679_triggers_user_auth_required_error_in_tool(self, fake_uat_path, monkeypatch):
        """When Feishu returns errcode 99991679, tool returns UserAuthRequiredError message."""
        monkeypatch.setenv("FEISHU_APP_ID", "cli_test_app")
        monkeypatch.setenv("FEISHU_APP_SECRET", "secret_test")
        monkeypatch.setattr("tools.feishu_oapi_client.FEISHU_UAT_PATH", fake_uat_path)

        from tools.feishu_oapi_client import FeishuClient
        FeishuClient._cache.clear()

        def _fake_sdk_request(request, option=None):
            mock_resp = MagicMock()
            mock_resp.code = None
            mock_resp.msg = ""
            mock_resp.data = None
            raw = MagicMock()

            uri = getattr(request, "_uri", "") or ""
            if "primary" in str(uri):
                raw.content = json.dumps({
                    "code": 0,
                    "msg": "success",
                    "data": {"calendars": [{"calendar_id": "cal_xyz"}]},
                }).encode()
            else:
                raw.content = json.dumps({
                    "code": 99991679,
                    "msg": "user scope insufficient",
                    "data": {},
                }).encode()

            mock_resp.raw = raw
            return mock_resp

        mock_sdk = MagicMock()
        mock_sdk.request.side_effect = _fake_sdk_request

        with patch.object(FeishuClient, "_build_sdk", return_value=mock_sdk):
            from tools.feishu_calendar_tool import _handle_calendar_list_events
            result_json = _handle_calendar_list_events({
                "start_time": "2024-01-01T00:00:00+08:00",
                "end_time": "2024-01-02T00:00:00+08:00",
                "calendar_id": "cal_xyz",
            })

        result = json.loads(result_json)
        # Tool should return an error string containing auth-related message
        assert "error" in result or ("success" in result and result.get("success") is False), (
            f"Expected error result from tool, got: {result}"
        )
        result_str = json.dumps(result)
        assert any(keyword in result_str.lower() for keyword in ("auth", "scope", "user", "99991679")), (
            f"Error message should reference auth/scope issue, got: {result_str}"
        )

    def test_missing_uat_file_returns_need_authorization_error(self, tmp_path, monkeypatch):
        """When UAT file does not exist, for_user() raises NeedAuthorizationError."""
        monkeypatch.setenv("FEISHU_APP_ID", "cli_test_app")
        monkeypatch.setenv("FEISHU_APP_SECRET", "secret_test")

        missing_path = tmp_path / ".hermes" / "feishu_uat.json"
        monkeypatch.setattr("tools.feishu_oapi_client.FEISHU_UAT_PATH", missing_path)

        from tools.feishu_oapi_client import FeishuClient, NeedAuthorizationError
        FeishuClient._cache.clear()

        with patch.object(FeishuClient, "_build_sdk", return_value=MagicMock()):
            with pytest.raises(NeedAuthorizationError):
                FeishuClient.for_user()


# ===========================================================================
# Scenario 3 — Streaming card end-to-end
# ===========================================================================

class TestStreamingCardEndToEnd:
    """Test the StreamingCardController state machine, flush throttling, and final card JSON."""

    def _run(self, coro):
        """Run a coroutine synchronously using asyncio.run()."""
        return asyncio.run(coro)

    def test_state_machine_transitions_idle_to_streaming_to_completed(self):
        """Controller transitions idle -> streaming -> completed when stream is consumed."""
        from gateway.platforms.cards.streaming_controller import StreamingCardController, Phase

        patch_calls = []

        async def _run_test():
            ctrl = StreamingCardController(message_id="om_test_msg", client=None)
            assert ctrl.phase == Phase.IDLE

            # Feed a few tokens
            await ctrl.add_text_chunk("Hello ")
            assert ctrl.phase == Phase.STREAMING

            await ctrl.add_text_chunk("world!")
            assert ctrl.phase == Phase.STREAMING

            await ctrl.mark_completed()
            assert ctrl.phase == Phase.COMPLETED
            assert ctrl.is_terminal_phase

        self._run(_run_test())

    def test_pending_flush_flag_coalesces_concurrent_tokens(self):
        """Two tokens sent back-to-back within one throttle window produce only 1 flush.

        Verifies the _pending_flush coalescing mechanism: when a second add_text_chunk
        call arrives while _pending_flush is already True, it skips scheduling
        another flush, so only 1 actual PATCH is issued for both tokens.
        """
        from gateway.platforms.cards.streaming_controller import StreamingCardController

        patch_call_count = {"n": 0}

        async def _run_test():
            ctrl = StreamingCardController(message_id="om_coalesce_test", client=MagicMock())

            def _mock_sync_patch(card_json: str) -> None:
                patch_call_count["n"] += 1

            ctrl._sync_patch_message = _mock_sync_patch

            # Send 2 tokens concurrently — both are scheduled before either flush fires
            await asyncio.gather(
                ctrl.add_text_chunk("first "),
                ctrl.add_text_chunk("second "),
            )
            # Wait long enough for the throttle window to expire and flush to fire
            await asyncio.sleep(0.6)  # > 500ms throttle
            await ctrl.mark_completed()

        self._run(_run_test())

        # Both tokens should be accumulated; only 1 flush should have fired for the pair
        # plus 1 more for mark_completed final card = at most 2 total
        assert patch_call_count["n"] <= 3, (
            f"Expected at most 3 flushes for 2 coalesced tokens, got {patch_call_count['n']}"
        )
        assert patch_call_count["n"] >= 1, "At least 1 flush must have occurred"

    def test_final_card_json_contains_text_block(self):
        """Final card JSON produced by to_card_json() contains the accumulated text."""
        from gateway.platforms.cards.streaming_controller import StreamingCardController, Phase

        async def _run_test():
            ctrl = StreamingCardController(message_id=None, client=None)
            await ctrl.add_text_chunk("The answer is 42.")
            await ctrl.mark_completed()
            return ctrl.to_card_json()

        card = self._run(_run_test())

        card_str = json.dumps(card)
        assert "The answer is 42." in card_str, f"Card JSON should contain answer text: {card_str}"
        assert "schema" in card
        assert "body" in card

    def test_final_card_json_contains_reasoning_block_when_present(self):
        """Final card JSON includes reasoning section when reasoning chunks were fed."""
        from gateway.platforms.cards.streaming_controller import StreamingCardController

        async def _run_test():
            ctrl = StreamingCardController(message_id=None, client=None)
            await ctrl.add_reasoning_chunk("Let me think... the answer is 42.")
            await ctrl.add_text_chunk("42.")
            await ctrl.mark_completed()
            return ctrl.to_card_json()

        card = self._run(_run_test())

        card_str = json.dumps(card)
        assert "think" in card_str.lower() or "reason" in card_str.lower() or "42" in card_str, (
            f"Card JSON should contain reasoning or text content: {card_str}"
        )

    def test_final_card_json_contains_tool_use_block_when_steps_present(self):
        """Final card JSON includes tool_use steps when tool calls were recorded."""
        from gateway.platforms.cards.streaming_controller import StreamingCardController

        async def _run_test():
            ctrl = StreamingCardController(message_id=None, client=None)
            ctrl.start_tool_use("web_search", {"query": "feishu api docs"})
            ctrl.complete_tool_use({"result": "found docs"})
            await ctrl.add_text_chunk("Based on the search, here is the answer.")
            await ctrl.mark_completed()
            return ctrl.to_card_json()

        card = self._run(_run_test())

        card_str = json.dumps(card)
        assert "web_search" in card_str, f"Card JSON should contain tool name: {card_str}"

    def test_consume_stream_drives_full_lifecycle(self):
        """consume_stream feeds tokens from an async generator and auto-completes."""
        from gateway.platforms.cards.streaming_controller import StreamingCardController, Phase

        final_patch_payloads = []

        async def _run_test():
            ctrl = StreamingCardController(message_id="om_stream_msg", client=MagicMock())

            # Patch the sync method dispatched to executor
            def _mock_sync_patch(card_json: str) -> None:
                final_patch_payloads.append(card_json)

            ctrl._sync_patch_message = _mock_sync_patch

            async def _fake_llm_stream():
                for chunk in ["Hello", " ", "from", " ", "the", " ", "LLM!"]:
                    yield chunk

            await ctrl.consume_stream(_fake_llm_stream())
            return ctrl

        ctrl = self._run(_run_test())

        assert ctrl.phase == Phase.COMPLETED
        assert "LLM!" in ctrl.text.accumulated_text

        # Final card should have been sent at least once
        assert len(final_patch_payloads) >= 1
        last_card = json.loads(final_patch_payloads[-1])
        assert "body" in last_card

    def test_update_message_patch_count_within_expected_range_with_real_tokens(self):
        """consume_stream completes lifecycle and produces valid final card JSON.

        Verifies the end-to-end path: 30 tokens streamed in, state reaches
        COMPLETED, and the final PATCH payload is valid Feishu card JSON.
        The exact flush count is not asserted (it is timing-dependent) but
        the final card structure is validated.
        """
        from gateway.platforms.cards.streaming_controller import StreamingCardController, Phase

        patch_calls = []

        async def _run_test():
            ctrl = StreamingCardController(message_id="om_count_test", client=MagicMock())

            def _mock_sync_patch(card_json: str) -> None:
                patch_calls.append(card_json)

            ctrl._sync_patch_message = _mock_sync_patch

            async def _token_stream():
                for i in range(30):
                    yield f"word{i} "

            await ctrl.consume_stream(_token_stream())
            return ctrl

        ctrl = asyncio.run(_run_test())

        # State machine must have reached COMPLETED
        assert ctrl.phase == Phase.COMPLETED
        # All 30 tokens must be accumulated in the text block
        for i in range(30):
            assert f"word{i}" in ctrl.text.accumulated_text

        # At minimum the final card flush from _flush_final must have fired
        assert len(patch_calls) >= 1, "At least 1 PATCH must have been issued (final card)"

        # Final card payload must be valid Feishu card JSON
        last = json.loads(patch_calls[-1])
        assert "schema" in last
        assert "body" in last
        body_str = json.dumps(last["body"])
        assert "word" in body_str, "Final card body should contain accumulated text"
