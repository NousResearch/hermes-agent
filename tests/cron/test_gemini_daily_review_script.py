from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

import agent.gemini_daily_review_runtime as daily_review_runtime
from agent.gemini_daily_review import AlertDeliveryIndeterminateError
from agent.gemini_daily_review_runtime import (
    _make_default_slack_sender,
    _hard_deadline,
    run_configured_review,
)
from agent.gemini_route_receipts import GeminiReceiptStore


@pytest.fixture(autouse=True)
def _profile_local_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))


def _config(db: Path) -> dict:
    return {
        "delegation": {
            "gemini_routing": {
                "enabled": True,
                "profiles": ["default"],
                "receipt_db": db.name if db.is_absolute() else str(db),
                "retention": {"raw_days": 30, "aggregate_days": 180},
                "review": {
                    "enabled": True,
                    "sample_size": 5,
                    "timezone": "America/Los_Angeles",
                    "review_provider": "openai-codex",
                    "review_model": "gpt-5.6-sol",
                    "review_reasoning_effort": "medium",
                    "alert_target": "slack:C0A12345678",
                    "alert_workspace_id": "T_EXPECTED",
                },
            }
        }
    }


def _seed_previous_day(db: Path) -> None:
    store = GeminiReceiptStore(db)
    receipt_id = store.prepare_attempt(
        parent_session_id="parent",
        parent_turn_id="turn",
        child_session_id="child",
        task_index=0,
        route_requested="auto",
        route_decision="gemini",
        route_reason="eligible",
        data_classification="standard",
        output_contract="text",
        goal_text="Summarize",
        context_text="Context",
        requested_provider="antigravity-subscription",
        requested_model="gemini-3.8-flash-low",
        requested_effort="low",
        started_at=datetime(2026, 9, 10, 19, tzinfo=timezone.utc),
    )
    store.mark_process_started(receipt_id, when=datetime(2026, 9, 10, 19, tzinfo=timezone.utc))
    store.complete_attempt(
        receipt_id,
        worker_status="completed",
        response_text="Summary",
        duration_ms=1,
        completed_at=datetime(2026, 9, 10, 19, 1, tzinfo=timezone.utc),
    )


def test_configured_review_runs_previous_local_day_and_stays_quiet_on_pass(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    db = tmp_path / "routing.sqlite3"
    _seed_previous_day(db)
    created = []
    alerts = []

    def reviewer_factory():
        created.append(object())
        return lambda prompt: {"verdict": "pass", "reason": "faithful", "failure_kind": "none"}

    result = run_configured_review(
        config=_config(db),
        now=datetime(2026, 9, 11, 16, tzinfo=timezone.utc),
        reviewer_factory=reviewer_factory,
        alert_sender=alerts.append,
    )

    assert result["status"] == "passed"
    assert result["routing_day"] == "2026-09-10"
    assert len(created) == 1
    assert alerts == []
    assert capsys.readouterr() == ("", "")


def test_configured_review_alerts_only_the_exact_configured_channel_on_failure(tmp_path: Path):
    db = tmp_path / "routing.sqlite3"
    _seed_previous_day(db)
    alerts = []

    def sender(message: str) -> dict:
        alerts.append(message)
        return {
            "success": True,
            "chat_id": "C0A12345678",
            "message_id": "1720000000.000003",
        }

    result = run_configured_review(
        config=_config(db),
        now=datetime(2026, 9, 11, 16, tzinfo=timezone.utc),
        reviewer_factory=lambda: (
            lambda prompt: {"verdict": "fail", "reason": "missed the requested evidence", "failure_kind": "correctness"}
        ),
        alert_sender=sender,
    )

    assert result["status"] == "failed"
    assert result["alert_status"] == "sent"
    assert len(alerts) == 1
    assert "missed the requested evidence" not in alerts[0]


def test_configured_review_retry_binds_sender_to_persisted_channel(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db = tmp_path / "routing.sqlite3"
    store = GeminiReceiptStore(db)
    batch = store.create_or_get_review_batch(
        routing_day="2026-09-11",
        timezone_name="America/Los_Angeles",
        sample_size_requested=0,
        eligible_count=0,
        sample_seed_hex="51" * 32,
        sample_receipt_ids=[],
    )
    review_lease = "review-lease"
    assert store.claim_review_batch(
        batch["batch_id"], lease_token=review_lease, now=datetime.now(timezone.utc)
    )
    message = "persisted alert bytes"
    store.update_review_batch(
        batch["batch_id"],
        lease_token=review_lease,
        status="pipeline_failed",
        pipeline_error="reviewer_unavailable",
        alert_status="pending",
        alert_message=message,
        alert_delivery_key=(
            f"gemini-daily-review:{batch['batch_id']}:"
            f"{hashlib.sha256(message.encode()).hexdigest()}"
        ),
        slack_channel_id="C_PERSISTED",
        slack_workspace_id="T_PERSISTED",
    )
    config = _config(db)
    config["delegation"]["gemini_routing"]["review"]["alert_target"] = (
        "slack:C_CHANGED"
    )
    config["delegation"]["gemini_routing"]["review"]["alert_workspace_id"] = (
        "T_CHANGED"
    )
    bound_channels: list[str] = []
    bound_workspaces: list[str] = []
    sent_messages: list[str] = []

    def build_sender(channel_id: str, *, expected_workspace_id: str, **_kwargs):
        bound_channels.append(channel_id)
        bound_workspaces.append(expected_workspace_id)

        def send(payload: str) -> dict:
            sent_messages.append(payload)
            return {"success": True, "chat_id": channel_id, "message_id": "1.51"}

        return send

    monkeypatch.setattr(daily_review_runtime, "_make_default_slack_sender", build_sender)

    result = run_configured_review(
        config=config,
        now=datetime(2026, 9, 12, 16, tzinfo=timezone.utc),
        reviewer_factory=lambda: pytest.fail("terminal batch must not be reviewed"),
    )

    assert result["alert_status"] == "sent"
    assert bound_channels == ["C_PERSISTED"]
    assert bound_workspaces == ["T_PERSISTED"]
    assert sent_messages == [message]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("review_provider", "arbitrary-provider"),
        ("review_model", "arbitrary-model"),
    ],
)
def test_configured_reviewer_identity_mismatch_is_rejected_before_batch(
    tmp_path: Path, field: str, value: str
):
    db = tmp_path / "routing.sqlite3"
    config = _config(db)
    config["delegation"]["gemini_routing"]["review"][field] = value
    with pytest.raises(ValueError, match=field):
        run_configured_review(
            config=config,
            now=datetime(2026, 9, 11, 16, tzinfo=timezone.utc),
            reviewer_factory=lambda: pytest.fail("mismatched reviewer must not be constructed"),
            alert_sender=lambda _message: pytest.fail("mismatched sender must not be constructed"),
        )

    assert not db.exists()


def test_disabled_runtime_does_not_construct_reviewer_or_sender(tmp_path: Path):
    calls = []
    result = run_configured_review(
        config={"delegation": {"gemini_routing": {"enabled": False}}},
        reviewer_factory=lambda: calls.append("reviewer"),
        alert_sender=lambda message: calls.append(message),
    )
    assert result == {"status": "disabled"}
    assert calls == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda config: config["delegation"]["gemini_routing"].update(enabled="false"),
        lambda config: config["delegation"]["gemini_routing"].update(profiles="default"),
        lambda config: config["delegation"]["gemini_routing"].update(profiles=[""]),
        lambda config: config["delegation"]["gemini_routing"].update(command=""),
        lambda config: config["delegation"]["gemini_routing"].update(extra_args=None),
        lambda config: config["delegation"]["gemini_routing"].update(receipt_db=""),
        lambda config: config["delegation"]["gemini_routing"].update(retention=None),
        lambda config: config["delegation"]["gemini_routing"].update(
            retention={"raw_days": 0, "aggregate_days": 180}
        ),
        lambda config: config["delegation"]["gemini_routing"].update(review=None),
        lambda config: config["delegation"]["gemini_routing"].update(review={}),
        lambda config: config["delegation"]["gemini_routing"]["review"].update(enabled="false"),
        lambda config: config["delegation"]["gemini_routing"]["review"].update(
            alert_workspace_id=""
        ),
    ],
    ids=[
        "truthy-routing-enabled",
        "string-profiles",
        "empty-profile",
        "empty-command",
        "null-extra-args",
        "empty-receipt-db",
        "null-retention",
        "nonpositive-retention",
        "null-review",
        "empty-review",
        "truthy-review-enabled",
        "empty-alert-workspace",
    ],
)
def test_runtime_fails_closed_on_malformed_security_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutate
):
    config = _config(tmp_path / "routing.sqlite3")
    mutate(config)
    calls: list[str] = []

    def fail_store(*_args, **_kwargs):
        calls.append("store")
        pytest.fail("invalid contract must not construct the receipt store")

    monkeypatch.setattr(daily_review_runtime, "GeminiReceiptStore", fail_store)

    with pytest.raises(ValueError, match="must be"):
        run_configured_review(
            config=config,
            reviewer_factory=lambda: calls.append("reviewer"),
            alert_sender=lambda message: calls.append(message),
        )

    assert calls == []


@pytest.mark.parametrize(
    "field",
    ["sample_size", "timezone", "not_before_local", "review_provider", "review_model"],
)
def test_runtime_uses_defaults_only_when_daily_review_keys_are_absent(
    tmp_path: Path, field: str
):
    db = tmp_path / "routing.sqlite3"
    config = _config(db)
    config["delegation"]["gemini_routing"]["review"].pop(field, None)

    result = run_configured_review(
        config=config,
        now=datetime(2026, 9, 11, 7, 14, tzinfo=timezone.utc),
        reviewer_factory=lambda: pytest.fail("not-before tick must not construct reviewer"),
        alert_sender=lambda _message: pytest.fail("not-before tick must not construct sender"),
    )

    assert result == {"status": "not_before"}
    assert not db.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sample_size", 4),
        ("timezone", "UTC"),
        ("timezone", ""),
        ("timezone", None),
        ("timezone", False),
        ("timezone", 0),
        ("not_before_local", ""),
        ("not_before_local", None),
        ("not_before_local", False),
        ("not_before_local", 0),
        ("review_provider", ""),
        ("review_provider", None),
        ("review_provider", False),
        ("review_provider", 0),
        ("review_model", ""),
        ("review_model", None),
        ("review_model", False),
        ("review_model", 0),
    ],
)
def test_runtime_rejects_noncanonical_daily_review_contract(
    tmp_path: Path, field: str, value: object
):
    config = _config(tmp_path / "routing.sqlite3")
    config["delegation"]["gemini_routing"]["review"][field] = value

    with pytest.raises(ValueError, match=field):
        run_configured_review(
            config=config,
            now=datetime(2026, 9, 11, 16, tzinfo=timezone.utc),
            reviewer_factory=lambda: pytest.fail("invalid contract must not construct reviewer"),
            alert_sender=lambda _message: pytest.fail("invalid contract must not construct sender"),
        )


def test_configured_review_stays_idle_before_local_not_before(tmp_path: Path):
    config = _config(tmp_path / "routing.sqlite3")
    config["delegation"]["gemini_routing"]["review"]["not_before_local"] = "00:15"
    calls = []

    result = run_configured_review(
        config=config,
        now=datetime(2026, 9, 11, 7, 14, tzinfo=timezone.utc),
        reviewer_factory=lambda: calls.append("reviewer"),
        alert_sender=lambda message: calls.append(message),
    )

    assert result == {"status": "not_before"}
    assert calls == []
    assert not (tmp_path / "routing.sqlite3").exists()


@pytest.mark.parametrize(
    "alert_target",
    ["all", "slack:", "slack:C:extra", "slack: C0A12345678", " slack:C0A12345678"],
)
def test_alert_target_must_be_one_exact_slack_channel(
    tmp_path: Path, alert_target: str
):
    config = _config(tmp_path / "routing.sqlite3")
    config["delegation"]["gemini_routing"]["review"]["alert_target"] = alert_target
    with pytest.raises(ValueError, match="exact slack"):
        run_configured_review(
            config=config,
            now=datetime(2026, 9, 11, 16, tzinfo=timezone.utc),
            reviewer_factory=lambda: lambda prompt: {"verdict": "pass", "reason": "ok", "failure_kind": "none"},
            alert_sender=lambda message: None,
        )


def test_no_agent_entrypoint_emits_empty_stdout_on_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    script = Path(__file__).parents[2] / "scripts" / "gemini_daily_review.py"
    spec = importlib.util.spec_from_file_location("run_gemini_daily_review_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "run_configured_review", lambda: {"status": "passed"})

    assert module.main() == 0
    assert capsys.readouterr() == ("", "")
    assert os.access(script, os.X_OK)


def test_no_agent_entrypoint_writes_fatal_errors_only_to_stderr(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    script = Path(__file__).parents[2] / "scripts" / "gemini_daily_review.py"
    spec = importlib.util.spec_from_file_location("run_gemini_daily_review_error_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def fail():
        raise RuntimeError("boom")

    monkeypatch.setattr(module, "run_configured_review", fail)
    assert module.main() == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "RuntimeError: boom" in captured.err


def test_no_agent_entrypoint_keeps_handled_pending_alert_quiet(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    script = Path(__file__).parents[2] / "scripts" / "gemini_daily_review.py"
    spec = importlib.util.spec_from_file_location("gemini_daily_review_pending_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(
        module,
        "run_configured_review",
        lambda: {"status": "failed", "alert_status": "pending"},
    )

    assert module.main() == 0
    assert capsys.readouterr() == ("", "")


def test_configured_review_requires_active_profile_membership(tmp_path: Path):
    config = _config(tmp_path / "routing.sqlite3")
    config["delegation"]["gemini_routing"]["profiles"] = ["other"]

    result = run_configured_review(
        config=config,
        now=datetime(2026, 9, 11, 16, tzinfo=timezone.utc),
        active_profile="default",
        reviewer_factory=lambda: pytest.fail("reviewer must not be constructed"),
        alert_sender=lambda _message: pytest.fail("sender must not be called"),
    )

    assert result == {"status": "disabled"}
    assert not (tmp_path / "routing.sqlite3").exists()


@pytest.mark.parametrize(
    "configured_path",
    ["../outside.sqlite3", "/tmp/outside-gemini-review.sqlite3"],
    ids=["parent-traversal", "absolute-outside-profile"],
)
def test_configured_review_rejects_receipt_paths_outside_active_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    configured_path: str,
):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    config = _config(Path(configured_path))
    config["delegation"]["gemini_routing"]["receipt_db"] = configured_path

    with pytest.raises(ValueError, match="HERMES_HOME"):
        run_configured_review(
            config=config,
            now=datetime(2026, 9, 11, 16, tzinfo=timezone.utc),
            active_profile="default",
            reviewer_factory=lambda: pytest.fail("reviewer must not be constructed"),
            alert_sender=lambda _message: pytest.fail("sender must not be called"),
        )

    assert not (tmp_path / "outside.sqlite3").exists()


def test_configured_review_rejects_absolute_receipt_path_inside_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    home = tmp_path / "profile"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    absolute = home / "routing.sqlite3"
    config = _config(absolute)
    config["delegation"]["gemini_routing"]["receipt_db"] = str(absolute)

    with pytest.raises(ValueError, match="relative to HERMES_HOME"):
        run_configured_review(
            config=config,
            now=datetime(2026, 9, 11, 16, tzinfo=timezone.utc),
            reviewer_factory=lambda: pytest.fail("reviewer must not be constructed"),
            alert_sender=lambda _message: pytest.fail("sender must not be called"),
        )

    assert not absolute.exists()


def test_no_agent_entrypoint_subprocess_honors_custom_profile_home_and_is_silent(
    tmp_path: Path,
):
    script = Path(__file__).parents[2] / "scripts" / "gemini_daily_review.py"
    home = tmp_path / "profile"
    home.mkdir(mode=0o700)
    (home / "config.yaml").write_text(
        """delegation:
  gemini_routing:
    enabled: false
""",
        encoding="utf-8",
    )
    env = {
        "HOME": os.environ.get("HOME", ""),
        "PATH": os.environ.get("PATH", ""),
        "HERMES_HOME": str(home),
    }

    completed = subprocess.run(
        [sys.executable, str(script)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert completed.stderr == ""


def test_default_slack_sender_verifies_workspace_membership_and_history():
    calls: list[tuple[str, dict]] = []
    message = "Gemini daily review: Receipt: ~/.hermes/routing.sqlite3 batch grb_abcdef"

    class FakeClient:
        async def auth_test(self):
            calls.append(("auth_test", {}))
            return {"ok": True, "team_id": "T_KIZUKI"}

        async def conversations_info(self, **kwargs):
            calls.append(("conversations_info", kwargs))
            return {
                "ok": True,
                "channel": {
                    "id": "C0A12345678",
                    "is_member": True,
                    "context_team_id": "T_KIZUKI",
                },
            }

        async def conversations_history(self, **kwargs):
            calls.append(("conversations_history", kwargs))
            if sum(name == "conversations_history" for name, _ in calls) == 1:
                return {"ok": True, "messages": []}
            return {
                "ok": True,
                "messages": [
                    {
                        "ts": "1720000000.000004",
                        "text": message,
                    }
                ],
            }

    async def send(_config, channel, message):
        calls.append(("send", {"channel": channel, "message": message}))
        return {
            "success": True,
            "chat_id": channel,
            "message_id": "1720000000.000004",
        }

    result = _make_default_slack_sender(
        "C0A12345678",
        expected_workspace_id="T_KIZUKI",
        standalone_send=send,
        client_factory=lambda token: FakeClient(),
        token="xoxb-test-only",
    )(message)

    assert result == {
        "success": True,
        "platform": "slack",
        "team_id": "T_KIZUKI",
        "chat_id": "C0A12345678",
        "message_id": "1720000000.000004",
    }
    assert [name for name, _ in calls] == [
        "auth_test",
        "conversations_info",
        "conversations_history",
        "send",
        "conversations_history",
    ]


def test_default_slack_sender_reconciles_unknown_send_from_channel_history():
    message = "Gemini daily review: Receipt: x batch grb_deadbeef"

    class FakeClient:
        async def auth_test(self):
            return {"ok": True, "team_id": "T_KIZUKI"}

        async def conversations_info(self, **_kwargs):
            return {
                "ok": True,
                "channel": {"is_member": True, "context_team_id": "T_KIZUKI"},
            }

        async def conversations_history(self, **kwargs):
            assert "oldest" not in kwargs
            return {
                "ok": True,
                "messages": [
                    {"ts": "1720000000.000005", "text": message}
                ],
            }

    sends: list[str] = []

    async def uncertain_send(*_args):
        sends.append("sent")
        raise TimeoutError("response lost after possible delivery")

    result = _make_default_slack_sender(
        "C0A12345678",
        expected_workspace_id="T_KIZUKI",
        standalone_send=uncertain_send,
        client_factory=lambda token: FakeClient(),
        token="xoxb-test-only",
    )(message)

    assert result["message_id"] == "1720000000.000005"
    assert sends == []


def test_default_slack_sender_classifies_unreconciled_send_as_indeterminate():
    message = "Gemini daily review: Receipt: x batch grb_unreconciled"
    history_calls = 0

    class FakeClient:
        async def auth_test(self):
            return {"ok": True, "team_id": "T_KIZUKI"}

        async def conversations_info(self, **_kwargs):
            return {
                "ok": True,
                "channel": {"is_member": True, "context_team_id": "T_KIZUKI"},
            }

        async def conversations_history(self, **_kwargs):
            nonlocal history_calls
            history_calls += 1
            return {"ok": True, "messages": []}

    async def uncertain_send(*_args):
        raise TimeoutError("response lost after possible delivery")

    sender = _make_default_slack_sender(
        "C0A12345678",
        expected_workspace_id="T_KIZUKI",
        standalone_send=uncertain_send,
        client_factory=lambda token: FakeClient(),
        token="xoxb-test-only",
    )

    with pytest.raises(AlertDeliveryIndeterminateError):
        sender(message)
    assert history_calls == 2


def test_default_slack_sender_classifies_transport_error_mapping_as_indeterminate():
    message = "Gemini daily review: Receipt: x batch grb_transport_error"

    class FakeClient:
        async def auth_test(self):
            return {"ok": True, "team_id": "T_KIZUKI"}

        async def conversations_info(self, **_kwargs):
            return {
                "ok": True,
                "channel": {"is_member": True, "context_team_id": "T_KIZUKI"},
            }

        async def conversations_history(self, **_kwargs):
            return {"ok": True, "messages": []}

    async def uncertain_send(*_args):
        return {"error": "Slack send failed: response lost after possible delivery"}

    sender = _make_default_slack_sender(
        "C0A12345678",
        expected_workspace_id="T_KIZUKI",
        standalone_send=uncertain_send,
        client_factory=lambda token: FakeClient(),
        token="xoxb-test-only",
    )

    with pytest.raises(AlertDeliveryIndeterminateError):
        sender(message)


def test_default_slack_sender_keeps_known_api_rejection_retryable():
    message = "Gemini daily review: Receipt: x batch grb_api_rejection"

    class FakeClient:
        async def auth_test(self):
            return {"ok": True, "team_id": "T_KIZUKI"}

        async def conversations_info(self, **_kwargs):
            return {
                "ok": True,
                "channel": {"is_member": True, "context_team_id": "T_KIZUKI"},
            }

        async def conversations_history(self, **_kwargs):
            return {"ok": True, "messages": []}

    async def rejected_send(*_args):
        return {"error": "Slack API error: channel_not_found"}

    sender = _make_default_slack_sender(
        "C0A12345678",
        expected_workspace_id="T_KIZUKI",
        standalone_send=rejected_send,
        client_factory=lambda token: FakeClient(),
        token="xoxb-test-only",
    )

    with pytest.raises(RuntimeError, match="not visible"):
        sender(message)


def test_default_slack_sender_paginates_history_and_matches_exact_payload():
    message = "Gemini daily review: Receipt: x batch grb_deadbeef"
    cursors: list[str | None] = []

    class FakeClient:
        async def auth_test(self):
            return {"ok": True, "team_id": "T_KIZUKI"}

        async def conversations_info(self, **_kwargs):
            return {
                "ok": True,
                "channel": {"is_member": True, "context_team_id": "T_KIZUKI"},
            }

        async def conversations_history(self, **kwargs):
            cursors.append(kwargs.get("cursor"))
            if kwargs.get("cursor") is None:
                return {
                    "ok": True,
                    "messages": [{"ts": "1.0", "text": f"{message} changed"}],
                    "response_metadata": {"next_cursor": "page-2"},
                }
            return {
                "ok": True,
                "messages": [{"ts": "2.0", "text": message}],
                "response_metadata": {"next_cursor": ""},
            }

    async def must_not_send(*_args):
        pytest.fail("exact payload already exists on a later history page")

    result = _make_default_slack_sender(
        "C0A12345678",
        expected_workspace_id="T_KIZUKI",
        standalone_send=must_not_send,
        client_factory=lambda token: FakeClient(),
        token="xoxb-test-only",
    )(message)

    assert result["message_id"] == "2.0"
    assert cursors == [None, "page-2"]


def test_default_slack_sender_refuses_workspace_channel_identity_mismatch():
    sent: list[str] = []

    class FakeClient:
        async def auth_test(self):
            return {"ok": True, "team_id": "T_WRONG"}

        async def conversations_info(self, **_kwargs):
            return {
                "ok": True,
                "channel": {"is_member": True, "context_team_id": "T_WRONG"},
            }

    async def send(*_args):
        sent.append("sent")
        return {}

    sender = _make_default_slack_sender(
        "C0A12345678",
        expected_workspace_id="T_KIZUKI",
        standalone_send=send,
        client_factory=lambda token: FakeClient(),
        token="xoxb-test-only",
    )

    with pytest.raises(RuntimeError, match="workspace"):
        sender("Gemini daily review: batch grb_deadbeef")
    assert sent == []


def test_internal_deadline_expires_before_cron_outer_timeout():
    with pytest.raises(TimeoutError, match="internal deadline"):
        with _hard_deadline(0.01):
            time.sleep(0.1)
