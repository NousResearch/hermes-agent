"""Signal can edit a sent message, so the gateway updates one bubble instead of posting a new one.

signal-cli identifies a sent message by its send timestamp and edits it with ``editTimestamp``
(CLI: ``--edit-timestamp``). Before this, the adapter declared ``SUPPORTS_MESSAGE_EDITING = False``
and ``send()`` returned ``message_id=None``, so every long-running heartbeat tick and every
streaming update fell through to a fresh ``send`` — a new Signal message every interval.

Only signal-cli's JSON-RPC is stubbed; the real adapter, chunker and markdown→Signal conversion run.
``send()``'s own message_id contract is covered by TestSignalSendReturnsMessageId in test_signal.py.
"""
import pytest

from gateway.config import Platform, PlatformConfig

GROUP = "group:dGVzdC1ncm91cA=="      # a group chat_id needs no recipient resolution
ORIGINAL_TS = 1795400000000           # the timestamp of the message being edited
EDIT_TS = 1795400999999               # signal-cli's timestamp for the edit itself


def _adapter(monkeypatch):
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", "")
    from plugins.platforms.signal.adapter import SignalAdapter
    config = PlatformConfig()
    config.enabled = True
    config.extra = {"http_url": "http://localhost:8080", "account": "+15551234567"}
    adapter = SignalAdapter(config)
    adapter.set_authorization_check(lambda *a, **k: True)
    return adapter


def _stub_rpc(adapter, monkeypatch, result):
    """Capture every JSON-RPC call and return *result* for each."""
    calls = []

    async def mock_rpc(method, params, rpc_id=None, **kwargs):
        calls.append({"method": method, "params": dict(params)})
        return result

    monkeypatch.setattr(adapter, "_rpc", mock_rpc)
    return calls


def _sends(calls):
    return [c for c in calls if c["method"] == "send"]


def _ok(timestamp):
    return {"timestamp": timestamp, "results": [{"type": "SUCCESS"}]}


# --------------------------------------------------------------------------- capability contract

def test_gateway_now_detects_signal_edit_support(monkeypatch):
    """The regression that caused the spam: the gateway probes this before trying to edit."""
    from gateway.plugin_messages import _supports_edit
    assert _supports_edit(_adapter(monkeypatch)) is True


# --------------------------------------------------------------------------- edit_message

@pytest.mark.asyncio
async def test_edit_issues_a_send_carrying_edit_timestamp(monkeypatch):
    adapter = _adapter(monkeypatch)
    calls = _stub_rpc(adapter, monkeypatch, _ok(EDIT_TS))

    result = await adapter.edit_message(GROUP, str(ORIGINAL_TS), "Working — 9 min")

    assert result.success is True
    sends = _sends(calls)
    assert len(sends) == 1
    assert sends[0]["params"]["editTimestamp"] == ORIGINAL_TS
    assert sends[0]["params"]["message"] == "Working — 9 min"
    assert sends[0]["params"]["groupId"] == GROUP[len("group:"):]


@pytest.mark.asyncio
async def test_edit_keeps_reporting_the_original_timestamp(monkeypatch):
    """Signal anchors an edit chain on the first send; chaining off each edit walks off target."""
    adapter = _adapter(monkeypatch)
    _stub_rpc(adapter, monkeypatch, _ok(EDIT_TS))

    result = await adapter.edit_message(GROUP, str(ORIGINAL_TS), "second revision")

    assert result.message_id == str(ORIGINAL_TS)
    assert result.message_id != str(EDIT_TS)


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_id", ["", "not-a-timestamp", "17954e5", None])
async def test_edit_refuses_a_non_timestamp_id_without_calling_signal(monkeypatch, bad_id):
    adapter = _adapter(monkeypatch)
    calls = _stub_rpc(adapter, monkeypatch, _ok(EDIT_TS))

    result = await adapter.edit_message(GROUP, bad_id, "text")

    assert result.success is False
    assert _sends(calls) == [], "a malformed id must not reach signal-cli"


@pytest.mark.asyncio
async def test_edit_refuses_content_needing_several_messages(monkeypatch):
    """One edit replaces one message; the caller must send anew rather than show a fragment."""
    adapter = _adapter(monkeypatch)
    calls = _stub_rpc(adapter, monkeypatch, _ok(EDIT_TS))

    result = await adapter.edit_message(GROUP, str(ORIGINAL_TS), "y" * (adapter.MAX_MESSAGE_LENGTH * 2))

    assert result.success is False
    assert _sends(calls) == []


@pytest.mark.asyncio
async def test_edit_refuses_empty_content(monkeypatch):
    adapter = _adapter(monkeypatch)
    calls = _stub_rpc(adapter, monkeypatch, _ok(EDIT_TS))

    result = await adapter.edit_message(GROUP, str(ORIGINAL_TS), "   ")

    assert result.success is False
    assert _sends(calls) == []


@pytest.mark.asyncio
async def test_a_rejected_edit_reports_failure_so_the_caller_sends_anew(monkeypatch):
    """An old signal-cli without editTimestamp, or a vanished target, must degrade not raise."""
    adapter = _adapter(monkeypatch)
    _stub_rpc(adapter, monkeypatch, {"timestamp": EDIT_TS,
                                     "results": [{"type": "UNREGISTERED_FAILURE"}]})

    result = await adapter.edit_message(GROUP, str(ORIGINAL_TS), "text")

    assert result.success is False


@pytest.mark.asyncio
async def test_edit_accepts_and_ignores_finalize(monkeypatch):
    """Signal has no 'in progress' message state to close, unlike DingTalk AI Cards."""
    adapter = _adapter(monkeypatch)
    calls = _stub_rpc(adapter, monkeypatch, _ok(EDIT_TS))

    result = await adapter.edit_message(GROUP, str(ORIGINAL_TS), "done", finalize=True)

    assert result.success is True
    assert len(_sends(calls)) == 1
