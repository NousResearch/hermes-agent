from __future__ import annotations

from gateway.config import GatewayConfig, HomeChannel, Platform, PlatformConfig
from hermes_cli import kanban_db as kb


def _runner(monkeypatch, checkpoint_config=None, keywords=None):
    from gateway import run as gateway_run
    from gateway.run import GatewayRunner

    cfg = checkpoint_config or {
        "enabled": True,
        "platforms": ["discord"],
        "keywords": keywords
        or ["checkpoint", "review needed", "approval needed", "decision:"],
    }
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda: {"kanban": {"checkpoint_notifications": cfg}},
    )
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={
            Platform.DISCORD: PlatformConfig(
                enabled=True,
                home_channel=HomeChannel(
                    platform=Platform.DISCORD,
                    chat_id="1501780210578755715",
                    name="#hermes",
                ),
            )
        }
    )
    return runner


def _task(**overrides):
    data = dict(
        id="t_review",
        title="Decision: Gabriel approve Phase 1 visual direction",
        body="Review: /tmp/preview.png\nSafety: local-only/no-send",
        assignee="reviewer",
        status="blocked",
        priority=0,
        created_by="worker",
        created_at=1,
        started_at=None,
        completed_at=None,
        workspace_kind="scratch",
        workspace_path=None,
        claim_lock=None,
        claim_expires=None,
        tenant=None,
    )
    data.update(overrides)
    return kb.Task(**data)


def _event(kind="blocked", payload=None):
    return kb.Event(
        id=1,
        task_id="t_review",
        kind=kind,
        payload=payload if payload is not None else {"reason": "checkpoint reached: approve the preview"},
        created_at=2,
    )


def test_checkpoint_event_detection_matches_blocked_review_language(monkeypatch):
    runner = _runner(monkeypatch)

    assert runner._is_kanban_checkpoint_event(_task(), _event()) is True


def test_checkpoint_event_detection_matches_promoted_ready_checkpoint(monkeypatch):
    runner = _runner(monkeypatch)

    assert runner._is_kanban_checkpoint_event(
        _task(status="ready", title="Gabriel checkpoint: approve GHL Manager UI"),
        _event(kind="promoted", payload=None),
    ) is True


def test_checkpoint_event_detection_matches_gabriel_assignee_without_keyword(monkeypatch):
    runner = _runner(monkeypatch, keywords=["checkpoint"])

    assert runner._is_kanban_checkpoint_event(
        _task(assignee="gabriel", title="Approve launch"),
        _event(kind="promoted", payload=None),
    ) is True


def test_checkpoint_targets_prefer_board_specific_discord_channel(monkeypatch):
    runner = _runner(
        monkeypatch,
        checkpoint_config={
            "enabled": True,
            "default_target": "discord:1501780210578755715",
            "board_targets": {
                "ghl-manager-ui": "discord:1502593547122249758",
            },
        },
    )

    targets = runner._kanban_checkpoint_targets(Platform, board="ghl-manager-ui")

    assert len(targets) == 1
    assert targets[0]["platform"] is Platform.DISCORD
    assert targets[0]["chat_id"] == "1502593547122249758"
    assert targets[0]["name"] == "ghl-manager-ui"


def test_checkpoint_targets_fall_back_to_default_target_then_home(monkeypatch):
    runner = _runner(
        monkeypatch,
        checkpoint_config={
            "enabled": True,
            "default_target": "discord:1502593547122249758",
            "board_targets": {},
        },
    )

    configured = runner._kanban_checkpoint_targets(Platform, board="unknown-board")
    assert [t["chat_id"] for t in configured] == ["1502593547122249758"]

    runner = _runner(monkeypatch, checkpoint_config={"enabled": True, "platforms": ["discord"]})
    home = runner._kanban_checkpoint_targets(Platform, board="unknown-board")
    assert [t["chat_id"] for t in home] == ["1501780210578755715"]


def test_checkpoint_message_uses_standard_review_packet(monkeypatch):
    runner = _runner(monkeypatch)

    msg = runner._format_kanban_checkpoint_message(board="ghl-manager-ui", task=_task(), event=_event())

    assert "Gabriel review needed: Decision: Gabriel approve Phase 1 visual direction" in msg
    assert "Review: /tmp/preview.png" in msg
    assert "Card: ghl-manager-ui/t_review" in msg
    assert "Decision needed: checkpoint reached: approve the preview" in msg
    assert "Safety: local-only/no-send" in msg
    assert "Assignee: @reviewer" in msg


def test_checkpoint_collection_routes_promoted_once_to_board_target(monkeypatch, tmp_path):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    for key in ("HERMES_KANBAN_DB", "HERMES_KANBAN_BOARD", "HERMES_KANBAN_WORKSPACES_ROOT"):
        monkeypatch.delenv(key, raising=False)

    kb.create_board("ghl-manager-ui", name="GHL Manager UI")
    kb.init_db(board="ghl-manager-ui")
    with kb.connect(board="ghl-manager-ui") as conn:
        parent = kb.create_task(conn, title="review parent")
        checkpoint = kb.create_task(
            conn,
            title="Gabriel checkpoint: approve GHL Manager UI",
            body="Review: /tmp/preview.png\nSafety: local-only/no-send",
            assignee="gabriel",
            parents=[parent],
        )
        kb.complete_task(conn, parent, summary="ready for Gabriel checkpoint")

    runner = _runner(
        monkeypatch,
        checkpoint_config={
            "enabled": True,
            "default_target": "discord:#inbox",
            "board_targets": {"ghl-manager-ui": "discord:1502593547122249758"},
        },
    )

    collected = runner._collect_kanban_notifier_deliveries(Platform, kb)
    deliveries = [d for d in collected["checkpoints"] if d["board"] == "ghl-manager-ui"]

    assert len(deliveries) == 1
    assert deliveries[0]["target"]["chat_id"] == "1502593547122249758"
    assert deliveries[0]["items"][0]["event"].kind == "promoted"
    assert deliveries[0]["items"][0]["event"].task_id == checkpoint

    runner._kanban_checkpoint_advance(
        deliveries[0]["target_key"],
        deliveries[0]["cursor"],
        "ghl-manager-ui",
    )

    restarted = runner._collect_kanban_notifier_deliveries(Platform, kb)
    assert [d for d in restarted["checkpoints"] if d["board"] == "ghl-manager-ui"] == []
