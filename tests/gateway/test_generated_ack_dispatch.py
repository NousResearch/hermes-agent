from types import SimpleNamespace


def test_generated_ack_dispatch_saturation_drops_without_queueing(monkeypatch):
    import gateway.run as gateway_run

    class SaturatedSlots:
        def acquire(self, *, blocking):
            assert blocking is False
            return False

        def release(self):
            raise AssertionError("saturated dispatch must not release without acquire")

    class ExecutorMustNotRun:
        def submit(self, fn):
            raise AssertionError("saturated dispatch must not submit work")

    monkeypatch.setattr(gateway_run, "_GENERATED_ACK_DISPATCH_SLOTS", SaturatedSlots())
    monkeypatch.setattr(gateway_run, "_GENERATED_ACK_DISPATCH_EXECUTOR", ExecutorMustNotRun())

    assert gateway_run._submit_generated_ack_voice_out("voice request") is False


def test_generated_ack_dispatch_releases_slot_after_publish(monkeypatch):
    import gateway.run as gateway_run

    calls = []
    releases = []

    class Slots:
        def acquire(self, *, blocking):
            assert blocking is False
            return True

        def release(self):
            releases.append(True)

    class ImmediateFuture:
        def add_done_callback(self, callback):
            callback(self)

    class ImmediateExecutor:
        def submit(self, fn):
            fn()
            return ImmediateFuture()

    def fake_publish(message_text, **kwargs):
        calls.append((message_text, kwargs))

    monkeypatch.setattr(gateway_run, "_GENERATED_ACK_DISPATCH_SLOTS", Slots())
    monkeypatch.setattr(gateway_run, "_GENERATED_ACK_DISPATCH_EXECUTOR", ImmediateExecutor())
    monkeypatch.setitem(
        __import__("sys").modules,
        "gateway.pulse_voice_events",
        SimpleNamespace(publish_generated_ack_voice_out=fake_publish),
    )

    assert gateway_run._submit_generated_ack_voice_out("voice request", session_id="s1") is True
    assert calls == [("voice request", {"session_id": "s1"})]
    assert releases == [True]
