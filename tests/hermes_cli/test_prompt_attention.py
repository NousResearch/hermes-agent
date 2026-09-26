import json
import multiprocessing
import os
import subprocess
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from hermes_cli.cli_modal_mixin import CLIModalMixin
from hermes_cli.prompt_attention import (
    PromptAttentionArbiter,
    PromptAttentionController,
    PromptAttentionLease,
    PromptAttentionRequest,
    capture_prompt_attention_request,
)


def _request(name: str, *, pid: int | None = None) -> PromptAttentionRequest:
    return PromptAttentionRequest(
        request_id=name,
        pid=pid or os.getpid(),
        process_start_time=1.0,
        session_id=f"hermes-{name}",
        tmux_session=f"session-{name}",
        tmux_pane=f"%{name[-1]}",
        tmux_client_tty="/dev/pts/9",
        sway_container_id=42,
    )


def _acquire_in_child(runtime_dir: str, result_queue) -> None:
    request = _request("child", pid=os.getpid())
    request = PromptAttentionRequest(**{
        **request.to_json(),
        "process_start_time": None,
    })
    lease = PromptAttentionArbiter(runtime_dir, poll_interval=0.01).acquire(
        request, timeout=5
    )
    result_queue.put(lease.request.request_id)
    lease.release()


def test_fifo_lease_serializes_two_sessions(tmp_path):
    live = lambda _pid, _start: True
    first = PromptAttentionArbiter(tmp_path, process_alive=live, poll_interval=0.01)
    second = PromptAttentionArbiter(tmp_path, process_alive=live, poll_interval=0.01)
    lease_one = first.acquire(_request("r1"), timeout=1)
    acquired = []

    thread = threading.Thread(
        target=lambda: acquired.append(second.acquire(_request("r2"), timeout=1)),
        daemon=True,
    )
    thread.start()
    time.sleep(0.05)
    assert acquired == []

    lease_one.release()
    thread.join(timeout=1)
    assert len(acquired) == 1
    assert acquired[0].request.request_id == "r2"
    acquired[0].release()


def test_fifo_lease_serializes_across_processes(tmp_path):
    parent = PromptAttentionArbiter(tmp_path, poll_interval=0.01)
    request = _request("parent", pid=os.getpid())
    request = PromptAttentionRequest(**{
        **request.to_json(),
        "process_start_time": None,
    })
    lease = parent.acquire(request, timeout=1)
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_acquire_in_child, args=(str(tmp_path), result_queue)
    )
    process.start()
    time.sleep(0.1)
    assert result_queue.empty()

    lease.release()
    assert result_queue.get(timeout=5) == "child"
    process.join(timeout=5)
    assert process.exitcode == 0


def test_dead_owner_is_reclaimed_before_next_request(tmp_path):
    live_pids = {101}
    arbiter = PromptAttentionArbiter(
        tmp_path,
        process_alive=lambda pid, _start: pid in live_pids,
        poll_interval=0.01,
    )
    stale = _request("stale", pid=100)
    state = {
        "next_sequence": 2,
        "owner": stale.request_id,
        "requests": [{**stale.to_json(), "sequence": 1}],
    }
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "prompt_attention.json").write_text(json.dumps(state))

    lease = arbiter.acquire(_request("r1", pid=101), timeout=1)
    assert lease.request.request_id == "r1"
    lease.release()


def test_failed_release_can_be_retried():
    arbiter = MagicMock()
    arbiter.release.side_effect = [OSError("busy"), None]
    lease = PromptAttentionLease(arbiter, _request("r1"))

    with pytest.raises(OSError, match="busy"):
        lease.release()
    lease.release()

    assert arbiter.release.call_count == 2


def test_modal_release_retries_and_never_replaces_prompt_result():
    lease = MagicMock()
    lease.release.side_effect = [OSError("busy"), None]

    CLIModalMixin._release_prompt_attention(lease)

    assert lease.release.call_count == 2


def test_connection_wait_phase_releases_attention_lease_after_disabling_form():
    modal = CLIModalMixin()
    lease = MagicMock()
    modal._connection_state = {"phase": "form", "prompt_attention_lease": lease}

    def assert_form_is_already_disabled(_lease):
        assert modal._connection_state["phase"] == "waiting"
        assert modal._connection_state["prompt_attention_lease"] is None

    modal._release_prompt_attention = MagicMock(
        side_effect=assert_form_is_already_disabled
    )

    modal._connection_transition("waiting")

    modal._release_prompt_attention.assert_called_once_with(lease)


def test_connection_actionable_phase_reacquires_and_alerts():
    modal = CLIModalMixin()
    lease = MagicMock()
    modal._connection_state = {
        "phase": "waiting",
        "prompt_attention_lease": None,
        "fields": [{"type": "plain"}],
        "field_index": 0,
    }
    modal._acquire_prompt_attention = MagicMock(return_value=lease)
    modal._ring_bell = MagicMock()

    modal._connection_transition("form")

    assert modal._connection_state["prompt_attention_lease"] is lease
    modal._ring_bell.assert_called_once_with(
        prompt=True,
        context="connection setup",
        voice_allowed=True,
        attention_lease=lease,
        acquire_attention=False,
    )


def test_corrupt_state_fails_closed_without_focus(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "prompt_attention.json").write_text("not-json")
    arbiter = PromptAttentionArbiter(tmp_path, process_alive=lambda *_: True)

    with pytest.raises(RuntimeError, match="coordination state"):
        arbiter.acquire(_request("r1"), timeout=0.05)


def test_modal_acquisition_failure_does_not_publish_unserialized_prompt(monkeypatch):
    modal = CLIModalMixin()
    modal.prompt_attention_config = {"enabled": True}
    modal.session_id = "session"
    monkeypatch.setattr(
        "hermes_cli.prompt_attention.PromptAttentionArbiter.acquire",
        MagicMock(side_effect=RuntimeError("coordination state unreadable")),
    )

    with pytest.raises(RuntimeError, match="coordination state unreadable"):
        modal._acquire_prompt_attention()


def test_controller_targets_exact_tmux_client_and_sway_container(tmp_path):
    calls = []

    def run(argv, **_kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    controller = PromptAttentionController(
        runtime_dir=tmp_path,
        runner=run,
        popen=lambda argv, **_kwargs: calls.append(argv),
    )
    controller.activate(
        _request("r1"),
        {
            "focus": True,
            "sound": False,
        },
    )

    assert ["tmux", "switch-client", "-c", "/dev/pts/9", "-t", "%1"] in calls
    assert not any(argv[:2] == ["tmux", "select-window"] for argv in calls)
    assert not any(argv[:2] == ["tmux", "select-pane"] for argv in calls)
    assert ["swaymsg", "[con_id=42]", "focus"] in calls


def test_controller_refuses_unscoped_tmux_mutation(tmp_path):
    calls = []
    request = _request("r1")
    request = PromptAttentionRequest(**{
        **request.to_json(),
        "tmux_client_tty": "",
        "sway_container_id": None,
    })
    controller = PromptAttentionController(
        runtime_dir=tmp_path,
        runner=lambda argv, **_kwargs: calls.append(argv),
        popen=MagicMock(),
    )

    controller.activate(request, {"focus": True, "sound": False})

    assert calls == []


def test_capture_routes_direct_sway_terminal_from_process_ancestry(monkeypatch):
    monkeypatch.delenv("TMUX_PANE", raising=False)
    monkeypatch.setenv("SWAYSOCK", "/run/user/1000/sway.sock")
    monkeypatch.setattr(
        "hermes_cli.prompt_attention._ancestor_pids", lambda _pid: {111}
    )

    def run(argv, **_kwargs):
        assert argv == ["swaymsg", "-t", "get_tree", "-r"]
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "type": "root",
                "nodes": [{"type": "con", "id": 42, "pid": 111}],
            }),
            stderr="",
        )

    request = capture_prompt_attention_request("session", runner=run)

    assert request.sway_container_id == 42


def test_capture_refuses_ambiguous_sway_terminal_process(monkeypatch):
    monkeypatch.delenv("TMUX_PANE", raising=False)
    monkeypatch.setenv("SWAYSOCK", "/run/user/1000/sway.sock")
    monkeypatch.setattr(
        "hermes_cli.prompt_attention._ancestor_pids", lambda _pid: {111}
    )

    def run(_argv, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps({
                "type": "root",
                "nodes": [
                    {"type": "con", "id": 42, "pid": 111},
                    {"type": "con", "id": 43, "pid": 111},
                ],
            }),
            stderr="",
        )

    request = capture_prompt_attention_request("session", runner=run)

    assert request.sway_container_id is None


def test_controller_runs_voice_callback_only_inside_recency_window(tmp_path):
    voice = MagicMock()
    controller = PromptAttentionController(
        runtime_dir=tmp_path,
        runner=MagicMock(
            return_value=SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
        popen=MagicMock(),
        monotonic=lambda: 1000.0,
    )
    request = _request("r1")

    controller.activate(
        request,
        {"focus": False, "sound": False, "voice_recent_seconds": 300},
        voice_enabled_at=700.001,
        voice_callback=voice,
    )
    voice.assert_called_once_with()

    voice.reset_mock()
    controller.activate(
        request,
        {"focus": False, "sound": False, "voice_recent_seconds": 300},
        voice_enabled_at=700.0,
        voice_callback=voice,
    )
    voice.assert_not_called()

    controller.activate(
        request,
        {"focus": False, "sound": False, "voice_recent_seconds": 999},
        voice_enabled_at=699.999,
        voice_callback=voice,
    )
    voice.assert_not_called()


def test_controller_does_not_start_voice_for_masked_or_selection_prompt(tmp_path):
    voice = MagicMock()
    controller = PromptAttentionController(
        runtime_dir=tmp_path,
        runner=MagicMock(
            return_value=SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
        popen=MagicMock(),
        monotonic=lambda: 1000.0,
    )

    controller.activate(
        _request("r1"),
        {"focus": False, "sound": False, "voice_recent_seconds": 300},
        voice_enabled_at=999.0,
        voice_callback=voice,
        voice_allowed=False,
    )

    voice.assert_not_called()


def test_thunder_is_generated_once_and_played_without_shell(tmp_path):
    spawned = []
    controller = PromptAttentionController(
        runtime_dir=tmp_path,
        runner=MagicMock(
            return_value=SimpleNamespace(returncode=0, stdout="", stderr="")
        ),
        popen=lambda argv, **kwargs: spawned.append((argv, kwargs)),
    )

    controller.activate(_request("r1"), {"focus": False, "sound": True})
    wav = tmp_path / "prompt-attention-thunder.wav"
    assert wav.read_bytes().startswith(b"RIFF")
    assert spawned[0][0] == ["pw-play", str(wav)]
    assert spawned[0][1]["stdin"] is not None
    assert spawned[0][1].get("shell", False) is False


def test_sound_process_is_killed_and_reaped_after_timeout(tmp_path):
    reaped = threading.Event()

    class StubbornProcess:
        def __init__(self):
            self.wait_calls = 0
            self.terminated = False
            self.killed = False

        def wait(self, timeout=None):
            self.wait_calls += 1
            if self.wait_calls < 3:
                raise subprocess.TimeoutExpired("stubborn", timeout)
            reaped.set()

        def terminate(self):
            self.terminated = True

        def kill(self):
            self.killed = True

    process = StubbornProcess()
    controller = PromptAttentionController(
        runtime_dir=tmp_path, popen=lambda *_a, **_k: process
    )

    controller.activate(
        _request("r1"),
        {
            "focus": False,
            "sound": True,
            "sound_command": ["stubborn"],
            "command_timeout": 0.1,
        },
    )

    assert reaped.wait(timeout=1)
    assert process.terminated is True
    assert process.killed is True
    assert process.wait_calls == 3


def test_custom_focus_and_sound_commands_use_argv_without_shell(tmp_path):
    ran = []
    spawned = []

    def run(argv, **kwargs):
        ran.append((argv, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    controller = PromptAttentionController(
        runtime_dir=tmp_path,
        runner=run,
        popen=lambda argv, **kwargs: spawned.append((argv, kwargs)),
    )
    controller.activate(
        _request("r1"),
        {
            "focus": True,
            "focus_command": ["focus-helper", "{tmux_session}", "{tmux_pane}"],
            "sound": True,
            "sound_command": ["sound-helper", "--loud"],
            "command_timeout": 1.5,
        },
    )

    assert ran[0][0] == ["focus-helper", "session-r1", "%1"]
    assert ran[0][1]["timeout"] == 1.5
    assert spawned[0][0] == ["sound-helper", "--loud"]
    assert spawned[0][1]["shell"] is False
