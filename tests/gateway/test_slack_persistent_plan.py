"""Behavior contracts for persistent native Slack thinking-plan messages."""
import asyncio
from copy import deepcopy
from types import SimpleNamespace

import pytest
from slack_sdk.errors import SlackApiError
from slack_sdk.web.slack_response import SlackResponse

from gateway.slack_task_stream import SlackTaskStream


class Wire:
    def __init__(self):
        self.calls = []
        self.messages = {}
        self.opens = 0
        self.update_failures = []
        self.expire_append = False
        self.live = True

    def is_live(self):
        return self.live

    async def chat_startStream(self, **payload):
        self.opens += 1
        ts = f"stream-{self.opens}"
        self.messages[ts] = {"tasks": {}, "blocks": None}
        self.calls.append(("start", deepcopy(payload)))
        return {"ts": ts}

    async def chat_appendStream(self, **payload):
        self.calls.append(("append", deepcopy(payload)))
        if self.expire_append:
            self.expire_append = False
            raise RuntimeError("message_not_in_streaming_state")
        state = self.messages[payload["ts"]]["tasks"]
        for chunk in payload["chunks"]:
            if chunk["type"] != "task_update":
                continue
            row = state.setdefault(chunk["id"], {})
            details = row.get("details", "") + chunk.get("details", "")
            row.update(chunk)
            row["details"] = details
        return {"ok": True}

    async def chat_stopStream(self, **payload):
        self.calls.append(("stop", deepcopy(payload)))
        return {"ok": True}

    async def chat_update(self, **payload):
        self.calls.append(("update", deepcopy(payload)))
        if self.update_failures:
            raise self.update_failures.pop(0)
        self.messages[payload["ts"]]["blocks"] = deepcopy(payload["blocks"])
        return {"ok": True}


def text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(text(v) for v in value)
    if isinstance(value, dict):
        return value.get("text", "") + text(value.get("elements", []))
    return ""


def rows(wire, ts="stream-1"):
    return wire.messages[ts]["blocks"][0]["tasks"]


@pytest.fixture
def clock(monkeypatch):
    from gateway import slack_task_stream, slack_plan_snapshot
    now = [100.0]
    clock = SimpleNamespace(monotonic=lambda: now[0])
    monkeypatch.setattr(slack_task_stream, "time", clock)
    monkeypatch.setattr(slack_plan_snapshot, "time", clock)
    return now


def stream(wire, **kwargs):
    return SlackTaskStream(wire, "C123", "123.4", recipient_user_id="U123", recipient_team_id="T123", **kwargs)


@pytest.mark.asyncio
async def test_age_transition_preserves_exact_details_once_and_same_message(clock):
    wire = Wire()
    card = stream(wire, rollover_age_s=240)
    await card.task_started("one", "write_file", details="First paragraph.\n\nSecond paragraph.")
    clock[0] += 301
    await card._append_raw_task("one", "Updated", status="in_progress", details=" More.")
    await card.task_finished("one", "write_file", summary="Done")
    await card.stop()
    assert wire.opens == 1
    assert rows(wire)[0]["status"] == "complete"
    assert text(rows(wire)[0]["details"]) == "First paragraph.\n\nSecond paragraph. More."


@pytest.mark.asyncio
async def test_reactive_expiry_does_not_duplicate_rejected_chunk(clock):
    wire = Wire()
    card = stream(wire)
    await card.task_started("one", "read_file", details="A")
    wire.expire_append = True
    await card._append_raw_task("one", "Read", status="in_progress", details="B")
    await card.stop()
    assert wire.opens == 1
    assert text(rows(wire)[0]["details"]) == "AB"


@pytest.mark.asyncio
async def test_retry_after_keeps_message_and_flushes_latest_on_idle_tick(clock):
    wire = Wire()
    card = stream(wire, rollover_age_s=1)
    await card.task_started("one", "read_file", details="A")
    clock[0] += 2
    response = SlackResponse(client=None, http_verb="POST", api_url="https://slack.com/api/chat.update", req_args={},
                             data={"ok": False, "error": "ratelimited"}, headers={"Retry-After": "7"}, status_code=429)
    wire.update_failures.append(SlackApiError("fixture", response))
    await card._append_raw_task("one", "Read", status="in_progress", details="B")
    await card.task_finished("one", "read_file", summary="Newest")
    count = sum(method == "update" for method, _ in wire.calls)
    clock[0] += 6
    await card.flush_pending_plan()
    assert sum(method == "update" for method, _ in wire.calls) == count
    clock[0] += 1
    await card.flush_pending_plan()
    assert wire.opens == 1
    assert rows(wire)[0]["status"] == "complete"
    assert text(rows(wire)[0]["details"]) == "AB"
    await card.stop()


@pytest.mark.asyncio
async def test_capacity_splits_only_at_real_limit_and_keeps_all_rows(clock):
    wire = Wire()
    card = stream(wire, rollover_age_s=1)
    await card.task_started("t0", "read_file", details="0")
    clock[0] += 2
    for i in range(1, 51):
        await card.task_started(f"t{i}", "read_file", details=str(i))
        await card.task_finished(f"t{i}", "read_file")
    await card.stop()
    assert wire.opens == 2
    plans = [m["blocks"][0] for m in wire.messages.values()]
    assert all(len(p["tasks"]) <= 50 for p in plans)
    actual = {r["task_id"]: text(r.get("details", "")) for p in plans for r in p["tasks"]}
    assert actual == {f"t{i}": str(i) for i in range(51)}


@pytest.mark.asyncio
async def test_parent_child_same_message_ids_are_independent(clock):
    wire = Wire()
    parent, child = stream(wire, rollover_age_s=1), stream(wire, rollover_age_s=1)
    await parent.task_started("parent", "terminal")
    await child.task_started("child", "read_file")
    clock[0] += 500
    await parent.task_finished("parent", "terminal")
    await child.task_finished("child", "read_file")
    await parent.stop()
    await child.stop()
    assert wire.opens == 2
    assert [r["task_id"] for r in rows(wire, "stream-1")] == ["parent"]
    assert [r["task_id"] for r in rows(wire, "stream-2")] == ["child"]


@pytest.mark.asyncio
async def test_force_stop_never_flushes_pending_plan_after_egress_death(clock):
    wire = Wire()
    card = stream(wire, rollover_age_s=1)
    await card.task_started("one", "read_file")
    clock[0] += 2
    await card._append_raw_task("one", "Read", status="in_progress", details="safe")
    await card._append_raw_task("one", "Read", status="in_progress", details="pending")
    before = sum(method == "update" for method, _ in wire.calls)
    wire.live = False
    await card.stop(flush_reasoning=False)
    clock[0] += 10
    await card.flush_pending_plan()
    assert sum(method == "update" for method, _ in wire.calls) == before


@pytest.mark.asyncio
async def test_payload_overflow_preserves_large_single_task_details(clock):
    wire = Wire()
    card = stream(wire, rollover_age_s=1)
    await card.task_started("one", "read_file", details="A" * 20_000)
    clock[0] += 2
    await card._append_raw_task("one", "Read", status="in_progress", details="B" * 20_000)
    await card._append_raw_task("one", "Read", status="complete", details="C" * 20_000)
    await card.stop()
    assert wire.opens > 1
    details = "".join(text(r.get("details", "")) for m in wire.messages.values()
                      for r in m["blocks"][0]["tasks"])
    assert details == "A" * 20_000 + "B" * 20_000 + "C" * 20_000


@pytest.mark.asyncio
async def test_transient_plan_edit_keeps_pending_snapshot_for_retry(clock):
    wire = Wire()
    card = stream(wire, rollover_age_s=1)
    await card.task_started("one", "read_file", details="A")
    clock[0] += 2
    wire.update_failures.append(TimeoutError("ambiguous idempotent update"))
    await card._append_raw_task("one", "Read", status="complete", details="B")
    assert not card.disabled
    clock[0] += 3
    await card.flush_pending_plan()
    assert wire.opens == 1
    assert text(rows(wire)[0]["details"]) == "AB"
    await card.stop()
