"""signal.note_to_self: YAML → PlatformConfig.extra → SignalAdapter → _handle_envelope (issue #121970)."""

import pytest

from gateway.config import Platform, load_gateway_config
from gateway.platforms.signal import SignalAdapter

ACCOUNT = "+15551234567"
GROUP_ID = "abc123=="


@pytest.fixture(autouse=True)
def _signal_env(monkeypatch):
    monkeypatch.setenv("SIGNAL_HTTP_URL", "http://localhost:8080")
    monkeypatch.setenv("SIGNAL_ACCOUNT", ACCOUNT)
    monkeypatch.setenv("SIGNAL_GROUP_ALLOWED_USERS", GROUP_ID)


def _write_profile(tmp_path, name, note_to_self_line):
    home = tmp_path / name
    home.mkdir()
    body = "signal:\n  enabled: true\n"
    if note_to_self_line is not None:
        body += f"  {note_to_self_line}\n"
    (home / "config.yaml").write_text(body)
    return home


def _load_adapter(monkeypatch, home):
    monkeypatch.setenv("HERMES_HOME", str(home))
    pc = load_gateway_config().platforms[Platform.SIGNAL]
    adapter = SignalAdapter(pc)
    captured = []

    async def fake_handle(event):
        captured.append(event)

    adapter.handle_message = fake_handle
    return adapter, captured


def _self_sync(ts, text="note to self", attachments=None):
    sent = {"destinationNumber": ACCOUNT, "destination": ACCOUNT, "timestamp": ts, "message": text}
    if attachments:
        sent["attachments"] = attachments
    return {"envelope": {"sourceNumber": ACCOUNT, "sourceUuid": "uuid-self", "timestamp": ts,
                         "syncMessage": {"sentMessage": sent}}}


def _group_sync(ts, group_id=GROUP_ID, text="ping the group", destination=None):
    return {"envelope": {"sourceNumber": ACCOUNT, "sourceUuid": "uuid-self", "timestamp": ts,
                         "syncMessage": {"sentMessage": {
                             "destinationNumber": destination, "destination": destination, "timestamp": ts,
                             "message": text, "groupInfo": {"groupId": group_id, "type": "DELIVER"}}}}}


@pytest.mark.parametrize("line, expected", [
    (None, True),
    ("note_to_self: true", True),
    ("note_to_self: false", False),
    ('note_to_self: "false"', False),
    ("note_to_self: 'no'", False),
])
def test_yaml_value_reaches_adapter(monkeypatch, tmp_path, line, expected):
    adapter, _ = _load_adapter(monkeypatch, _write_profile(tmp_path, "p", line))
    assert adapter.note_to_self is expected


@pytest.mark.asyncio
async def test_disabled_drops_self_chat_before_attachment_fetch(monkeypatch, tmp_path):
    adapter, captured = _load_adapter(monkeypatch, _write_profile(tmp_path, "p", 'note_to_self: "false"'))
    fetched = []

    async def fake_collect(attachments_data):
        fetched.append(attachments_data)
        return [], []

    monkeypatch.setattr(adapter, "_collect_attachments", fake_collect)
    await adapter._handle_envelope(_self_sync(1000, attachments=[{"id": "att1", "contentType": "image/png"}]))
    assert captured == []
    assert fetched == []


@pytest.mark.asyncio
@pytest.mark.parametrize("destination", [None, ACCOUNT])
async def test_disabled_still_dispatches_allowed_group(monkeypatch, tmp_path, destination):
    adapter, captured = _load_adapter(monkeypatch, _write_profile(tmp_path, "p", "note_to_self: false"))
    await adapter._handle_envelope(_group_sync(2000, destination=destination))
    assert len(captured) == 1
    assert captured[0].source.chat_id == f"group:{GROUP_ID}"


@pytest.mark.asyncio
async def test_disabled_still_filters_disallowed_group(monkeypatch, tmp_path):
    adapter, captured = _load_adapter(monkeypatch, _write_profile(tmp_path, "p", "note_to_self: false"))
    await adapter._handle_envelope(_group_sync(2500, group_id="other=="))
    assert captured == []


@pytest.mark.asyncio
async def test_disabled_still_applies_group_mention_rules(monkeypatch, tmp_path):
    adapter, captured = _load_adapter(monkeypatch, _write_profile(tmp_path, "p", "note_to_self: false"))
    adapter.require_mention = True
    await adapter._handle_envelope(_group_sync(2750, text="unmentioned group message"))
    assert captured == []


@pytest.mark.asyncio
async def test_disabled_still_suppresses_group_echo(monkeypatch, tmp_path):
    adapter, captured = _load_adapter(monkeypatch, _write_profile(tmp_path, "p", "note_to_self: false"))
    adapter._track_sent_timestamp({"timestamp": 3000})
    await adapter._handle_envelope(_group_sync(3000, text="bot's own reply"))
    assert captured == []
    assert 3000 not in adapter._recent_sent_timestamps


@pytest.mark.asyncio
async def test_default_keeps_self_chat(monkeypatch, tmp_path):
    adapter, captured = _load_adapter(monkeypatch, _write_profile(tmp_path, "p", None))
    await adapter._handle_envelope(_self_sync(4000))
    assert len(captured) == 1


@pytest.mark.asyncio
async def test_profile_isolation_a_b_a(monkeypatch, tmp_path):
    a = _write_profile(tmp_path, "a", "note_to_self: false")
    b = _write_profile(tmp_path, "b", None)
    results = []
    for i, home in enumerate((a, b, a)):
        adapter, captured = _load_adapter(monkeypatch, home)
        await adapter._handle_envelope(_self_sync(5000 + i))
        results.append((adapter.note_to_self, len(captured)))
    assert results == [(False, 0), (True, 1), (False, 0)]
