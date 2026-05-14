from __future__ import annotations

import pytest

from gateway import mcp_bridge
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key


def _valid_payload(title: str = "002BW Discord output mirror implementation") -> dict:
    return {
        "title": title,
        "project": "hermes-agent",
        "mode": "local",
        "worktree_scope": {"path": "/tmp/hermes-worktree"},
        "task_contract": {
            "objective": "Implement focused Discord mirror behavior without executing tasks.",
            "acceptance_criteria": ["records mirrored response"],
        },
        "allowed_actions": ["read files", "edit scoped files", "run targeted tests"],
        "forbidden_actions": ["run_shell", "git_push", "docker_run"],
        "return_format": {"sections": ["summary", "tests"]},
    }


class _DiscordAdapter(BasePlatformAdapter):
    platform = Platform.DISCORD

    async def connect(self):  # pragma: no cover - abstract implementation
        pass

    async def disconnect(self):  # pragma: no cover - abstract implementation
        pass

    async def send(self, chat_id, content, **kwargs):  # pragma: no cover
        pass

    async def get_chat_info(self, chat_id):  # pragma: no cover
        return {}


def _runner(adapter: _DiscordAdapter) -> GatewayRunner:
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.DISCORD: PlatformConfig(enabled=True, token="***")}
    )
    runner.adapters = {Platform.DISCORD: adapter}
    return runner


def _source(platform: Platform = Platform.DISCORD) -> SessionSource:
    return SessionSource(
        platform=platform,
        chat_id="channel-1",
        chat_type="channel",
        user_id="user-1",
    )


def _event(text: str, platform: Platform = Platform.DISCORD) -> MessageEvent:
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=_source(platform),
        message_id="message-1",
    )


def test_discord_mirror_registers_after_delivery_callback_and_waits(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    submitted = mcp_bridge.submit_task(_valid_payload())
    adapter = _DiscordAdapter(PlatformConfig(enabled=True, token="***"), Platform.DISCORD)
    runner = _runner(adapter)
    session_key = build_session_key(_source())

    registered = runner._register_discord_mcp_bridge_mirror(
        source=_source(),
        event=_event(f"Task {submitted['task_id']}"),
        response="Final Discord response.",
        run_generation=7,
    )

    assert registered is True
    assert mcp_bridge.get_task_result(submitted["task_id"])["result"] is None
    callback = adapter.pop_post_delivery_callback(session_key, generation=7)
    assert callable(callback)
    callback()
    assert (
        mcp_bridge.get_task_result(submitted["task_id"])["result"]["response"]
        == "Final Discord response."
    )


def test_discord_mirror_callback_infrastructure_error_does_not_mutate_record(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    submitted = mcp_bridge.submit_task(_valid_payload())
    assert mcp_bridge.mirror_task_result(
        submitted["task_id"],
        "SUCCESS: meaningful completed task report.",
        platform="discord",
    )
    before = mcp_bridge.get_task_result(submitted["task_id"])["record"]
    adapter = _DiscordAdapter(PlatformConfig(enabled=True, token="***"), Platform.DISCORD)
    runner = _runner(adapter)
    session_key = build_session_key(_source())

    registered = runner._register_discord_mcp_bridge_mirror(
        source=_source(),
        event=_event(f"Task {submitted['task_id']}"),
        response="provider authentication failed: token_invalidated",
        run_generation=8,
    )

    assert registered is True
    callback = adapter.pop_post_delivery_callback(session_key, generation=8)
    assert callable(callback)
    callback()
    assert mcp_bridge.get_task_result(submitted["task_id"])["record"] == before


def test_discord_mirror_chains_existing_post_delivery_callbacks(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    submitted = mcp_bridge.submit_task(_valid_payload())
    adapter = _DiscordAdapter(PlatformConfig(enabled=True, token="***"), Platform.DISCORD)
    runner = _runner(adapter)
    session_key = build_session_key(_source())
    fired: list[str] = []
    adapter.register_post_delivery_callback(
        session_key,
        lambda: fired.append("existing"),
        generation=3,
    )

    runner._register_discord_mcp_bridge_mirror(
        source=_source(),
        event=_event(f"Task {submitted['task_id']}"),
        response="Mirrored after existing callback.",
        run_generation=3,
    )

    callback = adapter.pop_post_delivery_callback(session_key, generation=3)
    callback()
    assert fired == ["existing"]
    assert (
        mcp_bridge.get_task_result(submitted["task_id"])["result"]["response"]
        == "Mirrored after existing callback."
    )


def test_discord_mirror_unique_prefix_resolves(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    submitted = mcp_bridge.submit_task(_valid_payload("002BW Discord mirror task"))
    adapter = _DiscordAdapter(PlatformConfig(enabled=True, token="***"), Platform.DISCORD)
    runner = _runner(adapter)

    runner._register_discord_mcp_bridge_mirror(
        source=_source(),
        event=_event("Done with 002BW."),
        response="Unique prefix response.",
        run_generation=1,
    )
    adapter.pop_post_delivery_callback(build_session_key(_source()), generation=1)()

    assert (
        mcp_bridge.get_task_result(submitted["task_id"])["result"]["response"]
        == "Unique prefix response."
    )


def test_discord_mirror_full_code_resolves_approval_continuation(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    first = mcp_bridge.submit_task(_valid_payload("002DV-A lifecycle helpers"))
    mcp_bridge.submit_task(_valid_payload("002DV-B1 private runner plan"))
    b2 = mcp_bridge.submit_task(
        _valid_payload("002DV-B2 accepted read-only private runner source implementation tests")
    )
    adapter = _DiscordAdapter(PlatformConfig(enabled=True, token="***"), Platform.DISCORD)
    runner = _runner(adapter)

    registered = runner._register_discord_mcp_bridge_mirror(
        source=_source(),
        event=_event("Одобрявам 002DV-B2 accepted read-only private runner source implementation tests"),
        response="002DV-B2 SUCCESS final Discord response.",
        run_generation=2,
    )

    assert registered is True
    adapter.pop_post_delivery_callback(build_session_key(_source()), generation=2)()
    assert (
        mcp_bridge.get_task_result(b2["task_id"])["result"]["response"]
        == "002DV-B2 SUCCESS final Discord response."
    )
    assert mcp_bridge.get_task_result(first["task_id"])["result"] is None


def test_discord_mirror_full_code_b2a_resolves_more_specific_task(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    b2 = mcp_bridge.submit_task(_valid_payload("002DV-B2 accepted read-only private runner"))
    b2a = mcp_bridge.submit_task(
        _valid_payload("002DV-B2-A commit accepted read-only private runner source/tests")
    )
    adapter = _DiscordAdapter(PlatformConfig(enabled=True, token="***"), Platform.DISCORD)
    runner = _runner(adapter)

    registered = runner._register_discord_mcp_bridge_mirror(
        source=_source(),
        event=_event("002DV-B2-A commit accepted read-only private runner source/tests"),
        response="002DV-B2-A committed.",
        run_generation=3,
    )

    assert registered is True
    adapter.pop_post_delivery_callback(build_session_key(_source()), generation=3)()
    assert mcp_bridge.get_task_result(b2a["task_id"])["result"]["response"] == "002DV-B2-A committed."
    assert mcp_bridge.get_task_result(b2["task_id"])["result"] is None


@pytest.mark.parametrize(
    "event_text",
    ["No bridge marker here.", "Please finish 002BW."],
)
def test_discord_mirror_no_match_or_ambiguous_prefix_writes_nothing(
    tmp_path, monkeypatch, event_text
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    first = mcp_bridge.submit_task(_valid_payload("002BW first task"))
    second = mcp_bridge.submit_task(_valid_payload("002BW second task"))
    if "No bridge marker" in event_text:
        event_text = "No bridge marker here."
    adapter = _DiscordAdapter(PlatformConfig(enabled=True, token="***"), Platform.DISCORD)
    runner = _runner(adapter)

    registered = runner._register_discord_mcp_bridge_mirror(
        source=_source(),
        event=_event(event_text),
        response="Should not mirror.",
        run_generation=1,
    )

    assert registered is False
    assert mcp_bridge.get_task_result(first["task_id"])["result"] is None
    assert mcp_bridge.get_task_result(second["task_id"])["result"] is None


def test_discord_mirror_duplicate_refused_plus_active_approval_required_resolves_active(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    title = "002EA define accepted read-only diagnostic task classes"
    refused_payload = _valid_payload(title)
    refused_payload.pop("allowed_actions")
    refused = mcp_bridge.submit_task(refused_payload)
    active_payload = _valid_payload(title)
    active_payload["task_contract"] = {
        "objective": "Use a direct write_file operation for a scoped local bridge test fixture.",
        "acceptance_criteria": ["task is routed for approval and remains unexecuted"],
    }
    active_payload["allowed_actions"] = ["direct write_file in scoped test fixture"]
    active = mcp_bridge.submit_task(active_payload)
    adapter = _DiscordAdapter(PlatformConfig(enabled=True, token="***"), Platform.DISCORD)
    runner = _runner(adapter)

    registered = runner._register_discord_mcp_bridge_mirror(
        source=_source(),
        event=_event("Ю Одобрявам 002EA define accepted read-only diagnostic task classes"),
        response="002EA final Discord report.",
        run_generation=4,
    )

    assert refused["status"] == "refused"
    assert active["status"] == "approval_required"
    assert registered is True
    adapter.pop_post_delivery_callback(build_session_key(_source()), generation=4)()
    assert mcp_bridge.get_task_result(refused["task_id"])["result"] is None
    active_record = mcp_bridge.get_task_result(active["task_id"])["record"]
    assert active_record["status"] == "completed"
    assert active_record["execution"]["state"] == "completed"
    assert active_record["result"]["response"] == "002EA final Discord report."


def test_discord_mirror_multiple_non_refused_code_matches_fail_closed(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    first = mcp_bridge.submit_task(_valid_payload("002EA first active task"))
    second = mcp_bridge.submit_task(_valid_payload("002EA second active task"))
    refused_payload = _valid_payload("002EA refused duplicate")
    refused_payload.pop("allowed_actions")
    refused = mcp_bridge.submit_task(refused_payload)
    adapter = _DiscordAdapter(PlatformConfig(enabled=True, token="***"), Platform.DISCORD)
    runner = _runner(adapter)

    registered = runner._register_discord_mcp_bridge_mirror(
        source=_source(),
        event=_event("Одобрявам 002EA ambiguous active tasks"),
        response="Should not mirror.",
        run_generation=5,
    )

    assert refused["status"] == "refused"
    assert registered is False
    assert mcp_bridge.get_task_result(first["task_id"])["result"] is None
    assert mcp_bridge.get_task_result(second["task_id"])["result"] is None


def test_discord_mirror_refused_exact_id_does_not_complete_record(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    refused_payload = _valid_payload("002EA refused exact task")
    refused_payload.pop("allowed_actions")
    refused = mcp_bridge.submit_task(refused_payload)
    adapter = _DiscordAdapter(PlatformConfig(enabled=True, token="***"), Platform.DISCORD)
    runner = _runner(adapter)

    registered = runner._register_discord_mcp_bridge_mirror(
        source=_source(),
        event=_event(f"Exact refused {refused['task_id']}"),
        response="Should not complete refused.",
        run_generation=6,
    )

    assert refused["status"] == "refused"
    assert registered is True
    adapter.pop_post_delivery_callback(build_session_key(_source()), generation=6)()
    record = mcp_bridge.get_task_result(refused["task_id"])["record"]
    assert record["status"] == "refused"
    assert record["result"] is None


def test_discord_mirror_approval_required_exact_id_can_complete_record(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    payload = _valid_payload("002EA approval exact task")
    payload["task_contract"] = {
        "objective": "Use a direct write_file operation for a scoped local bridge test fixture.",
        "acceptance_criteria": ["task is routed for approval and remains unexecuted"],
    }
    payload["allowed_actions"] = ["direct write_file in scoped test fixture"]
    active = mcp_bridge.submit_task(payload)
    adapter = _DiscordAdapter(PlatformConfig(enabled=True, token="***"), Platform.DISCORD)
    runner = _runner(adapter)

    registered = runner._register_discord_mcp_bridge_mirror(
        source=_source(),
        event=_event(f"Exact approval {active['task_id']}"),
        response="Approval-required final report.",
        run_generation=7,
    )

    assert active["status"] == "approval_required"
    assert registered is True
    adapter.pop_post_delivery_callback(build_session_key(_source()), generation=7)()
    record = mcp_bridge.get_task_result(active["task_id"])["record"]
    assert record["status"] == "completed"
    assert record["execution"]["state"] == "completed"
    assert record["result"]["response"] == "Approval-required final report."


def test_discord_mirror_ignores_non_discord_sources(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    submitted = mcp_bridge.submit_task(_valid_payload())
    adapter = _DiscordAdapter(PlatformConfig(enabled=True, token="***"), Platform.DISCORD)
    runner = _runner(adapter)

    registered = runner._register_discord_mcp_bridge_mirror(
        source=_source(Platform.TELEGRAM),
        event=_event(f"Task {submitted['task_id']}", Platform.TELEGRAM),
        response="Should not mirror.",
        run_generation=1,
    )

    assert registered is False
    assert mcp_bridge.get_task_result(submitted["task_id"])["result"] is None
