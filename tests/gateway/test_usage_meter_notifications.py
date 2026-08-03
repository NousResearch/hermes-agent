from gateway.usage_meter_notifications import UsageMeterNotifier


def _reading(claude=10, codex=20):
    return {
        "claude": {
            "available": claude is not None,
            "windows": [] if claude is None else [{"label": "5-hour", "used_percent": claude}],
        },
        "codex": {
            "available": codex is not None,
            "windows": [] if codex is None else [{"label": "weekly", "used_percent": codex}],
        },
    }


def test_notifier_sends_once_for_claude_only_change_with_kanban_summary():
    sent: list[str] = []
    notifier = UsageMeterNotifier(
        send=lambda message: sent.append(message) or {"success": True, "message_id": "m1"},
        kanban_summary=lambda: {"running": 2, "ready": 1, "todo": 3, "blocked": 4},
    )

    assert notifier.observe(_reading()) is None
    assert notifier.observe(_reading(claude=11)) == {"success": True, "message_id": "m1"}

    assert len(sent) == 1
    assert "Claude" in sent[0]
    assert "5-hour 11%" in sent[0]
    assert "running 2" in sent[0]
    assert "ready 1" in sent[0]
    assert "todo 3" in sent[0]
    assert "blocked 4" in sent[0]


def test_notifier_sends_once_for_codex_only_change():
    sent: list[str] = []
    notifier = UsageMeterNotifier(send=lambda message: sent.append(message) or {"success": True})

    notifier.observe(_reading())
    notifier.observe(_reading(codex=21))

    assert len(sent) == 1
    assert "Codex" in sent[0]
    assert "weekly 21%" in sent[0]


def test_notifier_combines_provider_changes_into_one_message():
    sent: list[str] = []
    notifier = UsageMeterNotifier(send=lambda message: sent.append(message) or {"success": True})

    notifier.observe(_reading())
    notifier.observe(_reading(claude=11, codex=21))

    assert len(sent) == 1
    assert "Claude" in sent[0]
    assert "Codex" in sent[0]


def test_notifier_suppresses_identical_reading_and_unavailable_provider():
    sent: list[str] = []
    notifier = UsageMeterNotifier(send=lambda message: sent.append(message) or {"success": True})

    notifier.observe(_reading(claude=10, codex=None))
    assert notifier.observe(_reading(claude=10, codex=None)) is None
    assert notifier.observe(_reading(claude=10, codex=20)) is None

    assert sent == []


def test_notifier_retries_a_changed_reading_after_delivery_failure():
    attempts: list[str] = []

    def send(message: str):
        attempts.append(message)
        return {"error": "temporary Discord failure"} if len(attempts) == 1 else {"success": True}

    notifier = UsageMeterNotifier(send=send)
    notifier.observe(_reading())

    assert notifier.observe(_reading(claude=11)) == {"error": "temporary Discord failure"}
    assert notifier.observe(_reading(claude=11)) == {"success": True}
    assert len(attempts) == 2
