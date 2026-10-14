"""Tests for the Telegram consent-request inline-keyboard button (``cr:`` prefix).

Mirrors test_telegram_clarify_buttons.py's mock-Update/CallbackQuery pattern. Points
CONSENT_REQUEST_SCRIPT at the REAL consent_request.py script (so this exercises the real
resolve_via_button logic, not a fake) but always with CONSENT_LEDGER redirected to a
per-test tmp_path — never the production
/etc/hermes-agent/resident-guard-state/consent_ledger.jsonl, and never calls `hermes send` / a real chat_id.
"""
import importlib.util
import os
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.platforms.base import unauthorized_action_notice

# This test file intentionally loads the real consent_request.py from
# /etc/hermes-agent/resident-guard/ to exercise the
# real resolve_via_button logic.  That read hits the home-IO guard, which is
# correct behaviour — this opt-out is intentional and documented.
pytestmark = pytest.mark.allow_real_home_io

_repo = str(Path(__file__).resolve().parents[2])
if _repo not in sys.path:
    sys.path.insert(0, _repo)

from plugins.platforms.telegram.adapter import TelegramAdapter
from gateway.config import PlatformConfig

_CONSENT_SCRIPT = "/etc/hermes-agent/resident-guard/consent_request.py"


def _make_adapter():
    config = PlatformConfig(enabled=True, token="test-token", extra={})
    adapter = TelegramAdapter(config)
    adapter._bot = AsyncMock()
    adapter._app = MagicMock()
    return adapter


def _load_consent_module():
    spec = importlib.util.spec_from_file_location("consent_request_test_helper", _CONSENT_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def consent_env(tmp_path, monkeypatch):
    """Redirect BOTH the adapter's dynamic import and the ledger to tmp_path; reset the
    adapter module's import cache so each test gets a fresh (still-tmp) module."""
    ledger = tmp_path / "consent_ledger.jsonl"
    monkeypatch.setenv("CONSENT_LEDGER", str(ledger))
    monkeypatch.setenv("CONSENT_REQUEST_SCRIPT", _CONSENT_SCRIPT)
    import plugins.platforms.telegram.adapter as adapter_mod
    monkeypatch.setattr(adapter_mod, "_consent_request_module", None)
    helper = _load_consent_module()
    helper_ledger_path = tmp_path / "consent_ledger.jsonl"
    monkeypatch.setattr(helper, "ledger_path", lambda: helper_ledger_path)
    return {"ledger": ledger, "helper": helper}


def _seed_request(helper, request_id, *, principal="testuser", chat_id="12345678"):
    ev = {
        "event": "requested", "request_id": request_id, "created_at": time.time(),
        "requested_by": "test", "principal": principal, "chat_id": chat_id,
        "description": "test", "purpose": "", "ttl_seconds": 1800, "pid": os.getpid(),
    }
    helper._with_lock(helper.ledger_path(), lambda events: (None, [ev]))


def _make_query(data, *, chat_id="12345678", user_id="12345678", first_name="Testy", text="Consent request"):
    query = AsyncMock()
    query.data = data
    query.message = MagicMock()
    query.message.chat_id = chat_id
    query.message.chat.type = "private"
    query.message.text = text
    query.from_user = MagicMock()
    query.from_user.id = user_id
    query.from_user.first_name = first_name
    query.answer = AsyncMock()
    query.edit_message_text = AsyncMock()
    return query


async def _dispatch(adapter, query):
    update = MagicMock()
    update.callback_query = query
    context = MagicMock()
    with patch.dict(os.environ, {"TELEGRAM_ALLOWED_USERS": "*"}, clear=False):
        await adapter._handle_callback_query(update, context)


class TestConsentButtonApprove:
    @pytest.mark.asyncio
    async def test_yes_tap_resolves_approved_and_answers(self, consent_env):
        helper = consent_env["helper"]
        _seed_request(helper, "CR-AAAA0001")
        adapter = _make_adapter()
        query = _make_query("cr:yes:CR-AAAA0001")

        await _dispatch(adapter, query)

        query.answer.assert_called_once()
        assert "Approved" in query.answer.call_args[1]["text"]
        query.edit_message_text.assert_called_once()
        edited_text = query.edit_message_text.call_args[1]["text"]
        assert "Approved" in edited_text
        assert query.edit_message_text.call_args[1]["reply_markup"] is None

        events = helper._read_events(helper.ledger_path())
        resolved = [e for e in events if e["event"] == "resolved" and e["request_id"] == "CR-AAAA0001"]
        assert len(resolved) == 1
        assert resolved[0]["decision"] == "approved"
        assert resolved[0]["resolved_by"] == "telegram_button"

    @pytest.mark.asyncio
    async def test_no_tap_resolves_denied(self, consent_env):
        helper = consent_env["helper"]
        _seed_request(helper, "CR-AAAA0002")
        adapter = _make_adapter()
        query = _make_query("cr:no:CR-AAAA0002")

        await _dispatch(adapter, query)

        assert "Denied" in query.answer.call_args[1]["text"]
        events = helper._read_events(helper.ledger_path())
        resolved = [e for e in events if e["event"] == "resolved" and e["request_id"] == "CR-AAAA0002"]
        assert resolved[0]["decision"] == "denied"


class TestConsentButtonIdempotency:
    @pytest.mark.asyncio
    async def test_double_tap_does_not_double_resolve(self, consent_env):
        helper = consent_env["helper"]
        _seed_request(helper, "CR-AAAA0003")
        adapter = _make_adapter()

        await _dispatch(adapter, _make_query("cr:yes:CR-AAAA0003"))
        # Second tap (double-click / retried callback) with a conflicting choice
        query2 = _make_query("cr:no:CR-AAAA0003")
        await _dispatch(adapter, query2)

        assert "Already approved" in query2.answer.call_args[1]["text"]
        events = helper._read_events(helper.ledger_path())
        resolved = [e for e in events if e["event"] == "resolved" and e["request_id"] == "CR-AAAA0003"]
        assert len(resolved) == 1  # never double-written
        assert resolved[0]["decision"] == "approved"  # first tap wins, second can't flip it


class TestConsentButtonUnknownAndErrors:
    @pytest.mark.asyncio
    async def test_unknown_request_id_answers_without_crashing(self, consent_env):
        adapter = _make_adapter()
        query = _make_query("cr:yes:CR-NOSUCHID")

        await _dispatch(adapter, query)

        assert "Unknown" in query.answer.call_args[1]["text"]
        query.edit_message_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_malformed_callback_data_ignored(self, consent_env):
        adapter = _make_adapter()
        query = _make_query("cr:yes")  # missing request_id segment

        await _dispatch(adapter, query)

        query.answer.assert_not_called()
        query.edit_message_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_consent_request_script_fails_soft(self, consent_env, monkeypatch):
        monkeypatch.setenv("CONSENT_REQUEST_SCRIPT", "/nonexistent/consent_request.py")
        adapter = _make_adapter()
        query = _make_query("cr:yes:CR-WHATEVER")

        await _dispatch(adapter, query)  # must not raise out of the dispatcher

        query.answer.assert_called_once()
        assert "Could not resolve" in query.answer.call_args[1]["text"]
        query.edit_message_text.assert_not_called()


class TestConsentButtonAuthorization:
    @pytest.mark.asyncio
    async def test_unauthorized_user_rejected_and_not_resolved(self, consent_env):
        helper = consent_env["helper"]
        _seed_request(helper, "CR-AAAA0004")

        adapter = _make_adapter()

        class _DenyRunner:
            async def _handle_message(self, event):
                return None

            def _is_user_authorized(self, source):
                return False

        adapter._message_handler = _DenyRunner()._handle_message

        query = _make_query("cr:yes:CR-AAAA0004", user_id="666", first_name="Mallory")
        update = MagicMock()
        update.callback_query = query
        context = MagicMock()
        await adapter._handle_callback_query(update, context)  # no TELEGRAM_ALLOWED_USERS=* override

        query.answer.assert_called_once()
        assert query.answer.call_args[1]["text"] == unauthorized_action_notice("telegram")
        query.edit_message_text.assert_not_called()

        events = helper._read_events(helper.ledger_path())
        resolved = [e for e in events if e["event"] == "resolved" and e["request_id"] == "CR-AAAA0004"]
        assert resolved == []  # unauthorized tap must not consume the request


class TestConsentButtonChatMismatch:
    @pytest.mark.asyncio
    async def test_chat_mismatch_answers_unauthorized_and_does_not_resolve(self, consent_env):
        """A request_id token surfacing in the wrong chat: belt-and-suspenders guard in
        resolve_via_button (Telegram itself already scopes callback delivery to the chat
        holding the message, so this path is defense-in-depth, not the primary gate)."""
        helper = consent_env["helper"]
        _seed_request(helper, "CR-AAAA0005", chat_id="1111111")
        adapter = _make_adapter()
        query = _make_query("cr:yes:CR-AAAA0005", chat_id="2222222")

        await _dispatch(adapter, query)

        query.answer.assert_called_once()
        assert query.answer.call_args[1]["text"] == unauthorized_action_notice("telegram")
        query.edit_message_text.assert_not_called()

        events = helper._read_events(helper.ledger_path())
        resolved = [e for e in events if e["event"] == "resolved" and e["request_id"] == "CR-AAAA0005"]
        assert resolved == []  # mismatch must not consume the request
