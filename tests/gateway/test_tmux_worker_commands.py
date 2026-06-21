"""Discord tmux worker command tests.

These tests cover the Discord-only MVP for launching Claude/Codex workers into
one shared tmux session and routing Discord thread follow-ups back to the
mapped tmux pane.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock
import asyncio
import sys

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.session import SessionSource


def test_tmux_worker_commands_are_gateway_commands():
    from hermes_cli.commands import is_gateway_known_command, resolve_command

    for name in ("claude", "codex", "tmux_claude", "tmux_codex"):
        cmd = resolve_command(name)
        assert cmd is not None
        assert is_gateway_known_command(name)

    assert resolve_command("claude").args_hint == "<question>"
    assert resolve_command("codex").args_hint == "<question>"
    assert resolve_command("tmux_claude").args_hint == "<title>"
    assert resolve_command("tmux_codex").args_hint == "<title>"


def test_parse_worker_args_defaults_and_short_flags():
    from gateway.tmux_workers import parse_worker_args

    claude = parse_worker_args("claude", "설계 검토해")
    assert claude.model == "opus"
    assert claude.effort == "xhigh"
    assert claude.task == "설계 검토해"

    claude_override = parse_worker_args("claude", "--sonnet --max 빠르게 봐줘")
    assert claude_override.model == "sonnet"
    assert claude_override.effort == "max"
    assert claude_override.task == "빠르게 봐줘"

    codex = parse_worker_args("codex", "--high --model gpt-5.4 테스트 봐")
    assert codex.model == "gpt-5.4"
    assert codex.effort == "high"
    assert codex.task == "테스트 봐"


def test_worker_report_block_extraction_prefers_last_complete_block():
    from gateway.tmux_workers import extract_worker_report_block

    raw = """
startup noise
<HERMES_REPORT>
상태: running
결론: 첫 번째 보고
</HERMES_REPORT>
spinner noise
<HERMES_REPORT>
상태: done
결론: 최종 보고
근거: 테스트 통과
다음: 없음
</HERMES_REPORT>
trailing noise
"""

    assert extract_worker_report_block(raw) == "상태: done\n결론: 최종 보고\n근거: 테스트 통과\n다음: 없음"


def test_worker_prompt_wraps_discord_followup_with_report_contract():
    from gateway.tmux_workers import build_worker_prompt

    prompt = build_worker_prompt("테스트도 돌려", kind="discord_followup")

    assert "Discord thread follow-up" in prompt
    assert "테스트도 돌려" in prompt
    assert "<HERMES_REPORT>" in prompt
    assert "상태: running | needs_input | done | blocked" in prompt


def test_tmux_worker_manager_collects_stable_relay_delta(monkeypatch, tmp_path):
    from gateway.tmux_workers import TmuxWorkerManager, WorkerRecord

    now = 1000.0
    monkeypatch.setattr("gateway.tmux_workers.time.time", lambda: now)

    manager = TmuxWorkerManager(home=tmp_path)
    record = WorkerRecord(
        id="c1",
        tool="codex",
        mode="persistent",
        model="gpt-5.5",
        effort="xhigh",
        task="로그인 테스트 실패 고쳐",
        tmux_session="hermes",
        tmux_window="codex-c1-login-test",
        pane_id="%42",
        platform="discord",
        thread_id="thread-123",
        chat_id="thread-123",
        chat_name="codex · login-test-fix",
        user_id="user-1",
        status="running",
        request_path=tmp_path / "requests" / "c1.md",
        log_path=tmp_path / "logs" / "c1.log",
    )
    record.log_path.parent.mkdir(parents=True, exist_ok=True)
    record.log_path.write_text("\x1b[31m결론:\x1b[0m 테스트를 추가했습니다.\n", encoding="utf-8")
    manager.save(record)

    assert manager.collect_relay_updates(idle_seconds=2.0) == []

    now += 3.0
    updates = manager.collect_relay_updates(idle_seconds=2.0)

    assert len(updates) == 1
    update = updates[0]
    assert update.worker.id == "c1"
    assert update.next_offset == record.log_path.stat().st_size
    assert "\x1b" not in update.text
    assert "결론: 테스트를 추가했습니다." in update.text

    manager.mark_relay_sent(update)
    sent = manager.load("c1")
    assert sent.relay_offset == update.next_offset
    assert sent.relay_last_hash == update.digest
    assert manager.collect_relay_updates(idle_seconds=0.0) == []


def test_tmux_worker_manager_starts_persistent_worker_without_shell_injection(monkeypatch, tmp_path):
    from gateway.tmux_workers import TmuxWorkerManager, parse_worker_args

    calls: list[list[str]] = []

    class Result:
        def __init__(self, returncode=0, stdout="%7\n", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[:3] == ["tmux", "has-session", "-t"]:
            return Result(returncode=1, stdout="")
        if cmd[:3] == ["tmux", "new-window", "-t"]:
            return Result(stdout="%42\n")
        return Result(stdout="")

    monkeypatch.setattr("gateway.tmux_workers.subprocess.run", fake_run)
    monkeypatch.setattr("gateway.tmux_workers.time.sleep", lambda _seconds: None)

    manager = TmuxWorkerManager(home=tmp_path)
    spec = parse_worker_args("codex", "--xhigh 로그인 테스트 실패 고쳐")
    worker = manager.start_persistent(
        tool="codex",
        spec=spec,
        platform="discord",
        thread_id="thread-123",
        chat_id="thread-123",
        chat_name="codex · login-test-fix",
        user_id="user-1",
    )

    assert worker.tool == "codex"
    assert worker.model == "gpt-5.5"
    assert worker.effort == "xhigh"
    assert worker.thread_id == "thread-123"
    assert worker.pane_id == "%42"
    assert worker.log_path.exists() or worker.log_path.parent.exists()
    assert manager.find_by_thread("discord", "thread-123").id == worker.id

    flat = [" ".join(cmd) for cmd in calls]
    assert any("new-session" in item and "hermes" in item for item in flat)
    assert any("new-window" in item for item in flat)
    assert any("pipe-pane" in item for item in flat)
    assert any("load-buffer" in item for item in flat)
    assert any("paste-buffer" in item and " -p " in f" {item} " for item in flat)
    assert any("send-keys" in item and "Enter" in item for item in flat)
    assert not any("로그인 테스트 실패 고쳐" in item for item in flat), "task text must go through a prompt file/buffer, not shell args"


def test_tmux_worker_manager_can_start_persistent_worker_without_initial_prompt(monkeypatch, tmp_path):
    from gateway.tmux_workers import TmuxWorkerManager, parse_worker_args

    calls: list[list[str]] = []

    class Result:
        def __init__(self, returncode=0, stdout="%7\n", stderr=""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[:3] == ["tmux", "has-session", "-t"]:
            return Result(returncode=1, stdout="")
        if cmd[:3] == ["tmux", "new-window", "-t"]:
            return Result(stdout="%42\n")
        return Result(stdout="")

    monkeypatch.setattr("gateway.tmux_workers.subprocess.run", fake_run)
    monkeypatch.setattr("gateway.tmux_workers.time.sleep", lambda _seconds: None)

    manager = TmuxWorkerManager(home=tmp_path)
    spec = parse_worker_args("codex", "--high 소통 테스트")
    worker = manager.start_persistent(
        tool="codex",
        spec=spec,
        platform="discord",
        thread_id="thread-123",
        chat_id="thread-123",
        chat_name="codex · 소통-테스트",
        user_id="user-1",
        send_initial=False,
    )

    assert worker.task == "소통 테스트"
    assert worker.effort == "high"
    assert manager.find_by_thread("discord", "thread-123").id == worker.id

    flat = [" ".join(cmd) for cmd in calls]
    assert any("new-window" in item for item in flat)
    assert any("pipe-pane" in item for item in flat)
    assert not any("load-buffer" in item for item in flat)
    assert not any("paste-buffer" in item for item in flat)
    assert not any("send-keys" in item and "Enter" in item for item in flat)


@pytest.mark.asyncio
async def test_gateway_routes_plain_discord_thread_message_to_tmux_worker(monkeypatch, tmp_path):
    import gateway.run as gateway_run
    from gateway.tmux_workers import TmuxWorkerManager, WorkerRecord

    manager = TmuxWorkerManager(home=tmp_path)
    record = WorkerRecord(
        id="c1",
        tool="codex",
        mode="persistent",
        model="gpt-5.5",
        effort="xhigh",
        task="로그인 테스트 실패 고쳐",
        tmux_session="hermes",
        tmux_window="codex-c1-login-test",
        pane_id="%42",
        platform="discord",
        thread_id="thread-123",
        chat_id="thread-123",
        chat_name="codex · login-test-fix",
        user_id="user-1",
        status="running",
        request_path=tmp_path / "requests" / "c1.md",
        log_path=tmp_path / "logs" / "c1.log",
    )
    manager.save(record)

    sent: list[tuple[str, str]] = []

    def fake_send_followup(worker, text):
        sent.append((worker.id, text))
        return "queued"

    monkeypatch.setattr(manager, "send_followup", fake_send_followup)

    runner = object.__new__(gateway_run.GatewayRunner)
    runner._tmux_worker_manager = manager
    runner._run_agent = AsyncMock(side_effect=AssertionError("Hermes agent should not handle mapped worker thread follow-up"))
    runner.config = SimpleNamespace()

    event = MessageEvent(
        text="테스트도 돌려",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="thread-123",
            chat_type="thread",
            thread_id="thread-123",
            user_id="user-1",
        ),
    )

    result = await runner._maybe_route_tmux_worker_followup(event)

    assert result == "queued"
    assert len(sent) == 1
    worker_id, routed_text = sent[0]
    assert worker_id == "c1"
    assert "Discord thread follow-up" in routed_text
    assert "테스트도 돌려" in routed_text
    assert "<HERMES_REPORT>" in routed_text
    runner._run_agent.assert_not_called()


@pytest.mark.asyncio
async def test_gateway_routes_first_deferred_thread_message_as_initial_worker_request(monkeypatch, tmp_path):
    import gateway.run as gateway_run
    from gateway.tmux_workers import TmuxWorkerManager, WorkerRecord

    manager = TmuxWorkerManager(home=tmp_path)
    record = WorkerRecord(
        id="c2",
        tool="codex",
        mode="persistent",
        model="gpt-5.5",
        effort="high",
        task="소통 테스트",
        tmux_session="hermes",
        tmux_window="codex-c2-소통-테스트",
        pane_id="%43",
        platform="discord",
        thread_id="thread-456",
        chat_id="thread-456",
        chat_name="codex · 소통-테스트",
        user_id="user-1",
        status="running",
        request_path=tmp_path / "requests" / "c2.md",
        log_path=tmp_path / "logs" / "c2.log",
        initial_sent=False,
    )
    manager.save(record)

    sent: list[tuple[str, str]] = []

    def fake_send_followup(worker, text):
        sent.append((worker.id, text))
        worker.initial_sent = True
        manager.save(worker)
        return "queued"

    monkeypatch.setattr(manager, "send_followup", fake_send_followup)

    runner = object.__new__(gateway_run.GatewayRunner)
    runner._tmux_worker_manager = manager
    runner.config = SimpleNamespace()

    event = MessageEvent(
        text="이제 왕복 테스트해봐",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="thread-456",
            chat_type="thread",
            thread_id="thread-456",
            user_id="user-1",
        ),
    )

    result = await runner._maybe_route_tmux_worker_followup(event)

    assert result == "queued"
    assert sent[0][0] == "c2"
    assert "Initial Discord tmux worker request" in sent[0][1]
    assert "이제 왕복 테스트해봐" in sent[0][1]
    assert manager.load("c2").initial_sent is True


@pytest.mark.asyncio
async def test_gateway_tmux_persistent_command_defers_title_as_initial_prompt(monkeypatch):
    import gateway.run as gateway_run

    captured: dict[str, object] = {}

    class Manager:
        def start_persistent(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                id="c2",
                model=kwargs["spec"].model,
                effort=kwargs["spec"].effort,
                tmux_session="hermes",
                tmux_window="codex-c2-소통-테스트",
                log_path="/tmp/c2.log",
            )

    runner = object.__new__(gateway_run.GatewayRunner)
    runner._tmux_worker_manager = Manager()

    event = MessageEvent(
        text="/tmux_codex --high 소통 테스트",
        message_type=MessageType.COMMAND,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="thread-123",
            chat_type="thread",
            thread_id="thread-123",
            user_id="user-1",
        ),
    )

    result = await runner._handle_tmux_worker_command(event, "tmux_codex")

    spec = captured["spec"]
    assert spec.task == "소통 테스트"
    assert spec.effort == "high"
    assert captured["send_initial"] is False
    assert "첫 지시" in result


@pytest.mark.asyncio
async def test_gateway_summarizer_uses_hermes_report_without_aux_llm(monkeypatch):
    import gateway.run as gateway_run

    async def fail_if_called(*_args, **_kwargs):
        raise AssertionError("report blocks must bypass auxiliary LLM summarization")

    monkeypatch.setitem(
        sys.modules,
        "agent.auxiliary_client",
        SimpleNamespace(async_call_llm=fail_if_called),
    )

    runner = object.__new__(gateway_run.GatewayRunner)
    worker = SimpleNamespace(id="c1", tool="codex", model="gpt-5.5", effort="xhigh", log_path="/tmp/c1.log")
    raw = """
OpenAI Codex startup noise
<HERMES_REPORT>
상태: needs_input
결론: 구현 경로 선택 필요
근거: plugin과 built-in command 둘 다 가능
다음: plugin으로 갈지 확인 필요
</HERMES_REPORT>
screen noise
"""

    summary = await runner._summarize_tmux_worker_update(worker, raw)

    assert "Codex 작업 보고 · c1" in summary
    assert "상태: needs_input" in summary
    assert "구현 경로 선택 필요" in summary
    assert "OpenAI Codex startup noise" not in summary
    assert "screen noise" not in summary


@pytest.mark.asyncio
async def test_gateway_relay_posts_summary_to_discord_thread(monkeypatch, tmp_path):
    import gateway.run as gateway_run
    from gateway.tmux_workers import TmuxWorkerManager, WorkerRecord, WorkerRelayUpdate

    manager = TmuxWorkerManager(home=tmp_path)
    worker = WorkerRecord(
        id="c1",
        tool="codex",
        mode="persistent",
        model="gpt-5.5",
        effort="xhigh",
        task="로그인 테스트 실패 고쳐",
        tmux_session="hermes",
        tmux_window="codex-c1-login-test",
        pane_id="%42",
        platform="discord",
        thread_id="thread-123",
        chat_id="thread-123",
        chat_name="codex · login-test-fix",
        user_id="user-1",
        status="running",
        request_path=tmp_path / "requests" / "c1.md",
        log_path=tmp_path / "logs" / "c1.log",
    )
    update = WorkerRelayUpdate(worker=worker, text="긴 raw 출력", next_offset=123, digest="abc")
    monkeypatch.setattr(manager, "collect_relay_updates", lambda **_kwargs: [update])
    marked: list[WorkerRelayUpdate] = []
    monkeypatch.setattr(manager, "mark_relay_sent", lambda sent: marked.append(sent))

    adapter = SimpleNamespace(send=AsyncMock())
    runner = object.__new__(gateway_run.GatewayRunner)
    runner._tmux_worker_manager = manager
    runner.adapters = {Platform.DISCORD: adapter}
    runner._running = True

    async def fake_summary(_worker, _text):
        return "Codex 작업 업데이트 · c1\n\n결론:\n- 테스트를 추가했습니다."

    monkeypatch.setattr(runner, "_summarize_tmux_worker_update", fake_summary)

    sent_count = await runner._run_tmux_worker_relay_once(idle_seconds=0.0)

    assert sent_count == 1
    adapter.send.assert_awaited_once_with(
        "thread-123",
        "Codex 작업 업데이트 · c1\n\n결론:\n- 테스트를 추가했습니다.",
        metadata={"thread_id": "thread-123"},
    )
    assert marked == [update]


@pytest.mark.asyncio
async def test_gateway_tmux_followups_are_serialized_for_initial_prompt(monkeypatch):
    import gateway.run as gateway_run

    worker = SimpleNamespace(
        id="c1",
        tool="codex",
        initial_sent=False,
        pane_id="%42",
        status="running",
        tmux_window="codex-c1-test",
    )
    prompts: list[str] = []

    class Manager:
        def find_by_thread(self, platform, thread_id):
            assert platform == "discord"
            assert thread_id == "thread-123"
            return worker

        def load(self, worker_id):
            assert worker_id == "c1"
            return worker

        def send_followup(self, sent_worker, prompt):
            prompts.append(prompt)
            sent_worker.initial_sent = True
            return f"sent-{len(prompts)}"

    async def fake_to_thread(func, *args, **kwargs):
        await asyncio.sleep(0)
        return func(*args, **kwargs)

    monkeypatch.setattr(gateway_run.asyncio, "to_thread", fake_to_thread)

    runner = object.__new__(gateway_run.GatewayRunner)
    runner._tmux_worker_manager = Manager()

    event = MessageEvent(
        text="연속 지시",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="thread-123",
            chat_type="thread",
            thread_id="thread-123",
            user_id="user-1",
        ),
    )

    results = await asyncio.gather(
        runner._maybe_route_tmux_worker_followup(event),
        runner._maybe_route_tmux_worker_followup(event),
    )

    assert results == ["sent-1", "sent-2"]
    assert "Initial Discord tmux worker request" in prompts[0]
    assert "Discord thread follow-up" in prompts[1]


@pytest.mark.asyncio
async def test_gateway_relay_redacts_provider_keys_and_hides_host_log_paths():
    import gateway.run as gateway_run

    raw = (
        "완료: provider keys sk-ant-" + "a" * 24 + " "
        "AIza" + "B" * 35 + " AKIA" + "1234567890ABCDEF"
    )
    worker = SimpleNamespace(
        id="c1",
        tool="codex",
        log_path="/home/idhoons/.hermes/tmux-agents/logs/c1.log",
    )

    summary = gateway_run.GatewayRunner._fallback_tmux_worker_summary(worker, raw)

    assert "sk-ant" not in summary
    assert "AIza" not in summary
    assert "AKIA" not in summary
    assert summary.count("[REDACTED]") >= 3
    assert "/home/" not in summary
    assert ".hermes" not in summary
    assert "c1.log" in summary


@pytest.mark.asyncio
async def test_gateway_one_shot_tmux_command_redacts_worker_output(monkeypatch):
    import gateway.run as gateway_run

    class Manager:
        def run_once(self, **_kwargs):
            return "result sk-ant-" + "a" * 24

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(gateway_run.asyncio, "to_thread", fake_to_thread)

    runner = object.__new__(gateway_run.GatewayRunner)
    runner._tmux_worker_manager = Manager()

    event = MessageEvent(
        text="/codex 확인해",
        message_type=MessageType.COMMAND,
        source=SessionSource(
            platform=Platform.DISCORD,
            chat_id="thread-123",
            chat_type="thread",
            thread_id="thread-123",
            user_id="user-1",
        ),
    )

    result = await runner._handle_tmux_worker_command(event, "codex")

    assert "sk-ant" not in result
    assert "[REDACTED]" in result


@pytest.mark.asyncio
async def test_gateway_rejects_tmux_worker_commands_outside_discord(monkeypatch):
    import gateway.run as gateway_run

    class Manager:
        def run_once(self, **_kwargs):  # pragma: no cover - should not be reached
            raise AssertionError("non-Discord /claude must not launch a worker")

    runner = object.__new__(gateway_run.GatewayRunner)
    runner._tmux_worker_manager = Manager()

    event = MessageEvent(
        text="/claude 텔레그램에서는 실행되면 안 됨",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id="telegram-dm",
            chat_type="dm",
            user_id="user-1",
        ),
    )

    result = await runner._handle_tmux_worker_command(event, "claude")

    assert "Discord" in result
