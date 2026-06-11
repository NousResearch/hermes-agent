from types import SimpleNamespace


def _runner():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._stt_echo_keys = {}
    return runner


def test_stt_echo_dedupe_suppresses_immediate_duplicate(monkeypatch):
    import gateway.run as gateway_run

    runner = _runner()
    source = SimpleNamespace(platform="telegram", chat_id="chat-1", thread_id=None, message_id="msg-1")
    event = SimpleNamespace(message_id="msg-1")

    monkeypatch.setattr(gateway_run.time, "time", lambda: 1000.0)

    assert runner._mark_stt_echo_sent_once(source, event, "/tmp/a.ogg", "Проверка") is True
    assert runner._mark_stt_echo_sent_once(source, event, "/tmp/b.ogg", "Проверка") is False


def test_stt_echo_dedupe_allows_same_transcript_after_ttl(monkeypatch):
    import gateway.run as gateway_run

    runner = _runner()
    source = SimpleNamespace(platform="telegram", chat_id="chat-1", thread_id=None, message_id="msg-1")
    event = SimpleNamespace(message_id="msg-1")

    now = {"value": 1000.0}
    monkeypatch.setattr(gateway_run.time, "time", lambda: now["value"])

    assert runner._mark_stt_echo_sent_once(source, event, "/tmp/a.ogg", "Проверка") is True
    now["value"] = 1021.0
    assert runner._mark_stt_echo_sent_once(source, event, "/tmp/c.ogg", "Проверка") is True
