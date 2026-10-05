from plugins.platforms.discord.background_activity import (
    PublishGate,
    render_dashboard,
    render_presence,
)


def _item(key="one", elapsed=1, **overrides):
    item = {
        "key": key, "title": "Build safe thing", "profile": "coder", "model": "gpt-safe",
        "provider": "openai-codex", "worker": "kanban", "started_at": 100.0,
        "elapsed_seconds": elapsed, "elapsed": f"{elapsed}s", "state": "running",
    }
    item.update(overrides)
    return item


def test_presence_is_compact_and_aggregates_workers():
    text = render_presence([_item("one"), _item("two")])
    assert text.startswith("2 workers · Build safe thing")
    assert len(text) <= 96


def test_dashboard_has_one_idle_or_active_document():
    assert "Idle" in render_dashboard([], now=120)
    active = render_dashboard([_item()], now=120)
    assert "1 active worker" in active
    assert "Build safe thing" in active
    assert "coder" in active
    assert "gpt-safe" in active


def test_dashboard_lists_each_worker_separately_with_identity_and_state():
    dashboard = render_dashboard(
        [
            _item("one", title="First card", worker="kanban", profile="coder"),
            _item(
                "two", title="Second card", worker="delegate", profile="default",
                model="claude-x", provider="anthropic",
            ),
        ],
        now=120,
    )
    assert "2 active workers" in dashboard
    assert "• **First card**" in dashboard
    assert "• **Second card**" in dashboard
    # Every row carries its own identity line — no aggregate-only rendering.
    assert "worker=kanban · profile=coder · model=gpt-safe · provider=openai-codex · state=running" in dashboard
    assert "worker=delegate · profile=default · model=claude-x · provider=anthropic · state=running" in dashboard
    assert dashboard.count("state=running") == 2


def test_dashboard_omits_absent_model_and_provider_but_keeps_state():
    row = render_dashboard([_item(model="", provider="")], now=120)
    assert "model=" not in row
    assert "provider=" not in row
    assert "worker=kanban · profile=coder · state=running" in row


def test_dashboard_order_is_deterministic_and_independent_of_input_order():
    early = _item("early", title="Early card", started_at=10.0)
    late = _item("late", title="Late card", started_at=900.0)
    forward = render_dashboard([early, late], now=1000)
    reversed_input = render_dashboard([late, early], now=1000)
    assert forward == reversed_input
    assert forward.index("Early card") < forward.index("Late card")


def test_publish_gate_flushes_transitions_but_throttles_elapsed_ticks():
    gate = PublishGate(periodic_seconds=15, minimum_interval_seconds=1)
    active = [_item(elapsed=1)]
    assert gate.should_publish(active, now=100)
    gate.mark_published(active, now=100)
    assert not gate.should_publish([_item(elapsed=2)], now=101)
    assert gate.should_publish([_item(elapsed=16)], now=116)
    gate.mark_published([_item(elapsed=16)], now=116)
    assert not gate.should_publish([], now=116.1)
    assert gate.should_publish([], now=117)
    gate.mark_published([], now=117)
    assert not gate.should_publish([], now=118)


def test_publish_gate_retries_failed_idle_transition():
    gate = PublishGate(minimum_interval_seconds=0)
    active = [_item()]
    gate.mark_published(active, now=100)
    assert gate.should_publish([], now=101)
    # No mark_published: the failed terminal transition remains pending.
    assert gate.should_publish([], now=101.5)
