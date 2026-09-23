"""Synthetic native-scope and socket fixtures; no real OMP/tmux/model calls."""

import contextlib
import json
import os
import shlex
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import venv
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "linux", reason="Linux-only skill")
pytest.importorskip("fcntl")
SCRIPTS = (
    Path(__file__).resolve().parents[2]
    / "optional-skills/autonomous-ai-agents/omp-session-supervision/scripts"
)
sys.path.insert(0, str(SCRIPTS))
from omp_supervisor import tui  # noqa: E402


def test_skill_runs_after_isolated_installation(run):
    from tools.skills_hub_official import OptionalSkillSource

    installed = run.parent / "installed-skill"
    source = OptionalSkillSource()
    source._optional_dir = Path(__file__).resolve().parents[2] / "optional-skills"
    bundle = source.fetch("official/autonomous-ai-agents/omp-session-supervision")
    assert bundle is not None
    assert "scripts/omp_supervisor/tui_extension.ts" in bundle.files
    assert not any("__pycache__" in name for name in bundle.files)
    for name, content in bundle.files.items():
        destination = installed / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(
            content if isinstance(content, bytes) else content.encode()
        )
    runtime = run.parent / "python"
    venv.EnvBuilder(with_pip=False).create(runtime)
    command = [str(runtime / "bin/python"), str(installed / "scripts/omp_supervise.py")]
    env = {
        name: value
        for name, value in os.environ.items()
        if name.startswith("HERMES_SESSION_")
    }
    env.update(HERMES_HOME=os.environ["HERMES_HOME"], PATH=os.defpath)
    prepared = subprocess.run(
        [
            *command,
            "prepare",
            "--workspace",
            str(run.parent / "w"),
            "--tmux-session",
            "installed-test",
            "--state-root",
            str(run.parent),
        ],
        cwd=runtime,
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    enrolled = json.loads(prepared.stdout)
    observed = subprocess.run(
        [*command, "status", "--run-dir", enrolled["run_dir"]],
        cwd=runtime,
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    )
    assert json.loads(observed.stdout)["run_id"] == enrolled["run_id"]
    assert json.loads(observed.stdout)["observer_active"] is False
    env["HERMES_SESSION_ID"] = "different-generation"
    rejected = subprocess.run(
        [*command, "status", "--run-dir", enrolled["run_dir"]],
        cwd=runtime,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    assert rejected.returncode == 2
    assert json.loads(rejected.stderr)["error"] == "scope_denied"


@pytest.fixture
def run(monkeypatch):
    # Short paths are necessary even when pytest's generated test names are long.
    with tempfile.TemporaryDirectory(
        prefix="t", dir=os.environ.get("TMPDIR")
    ) as directory:
        root = Path(directory)
        home = root / "h"
        workspace = root / "w"
        home.mkdir()
        workspace.mkdir()
        for name, value in {
            "HERMES_HOME": str(home),
            "HERMES_SESSION_PLATFORM": "discord",
            "HERMES_SESSION_KEY": "synthetic:discord:chat:thread",
            "HERMES_SESSION_ID": "synthetic-native-generation",
            "HERMES_SESSION_CHAT_ID": "111",
            "HERMES_SESSION_THREAD_ID": "222",
            "HERMES_SESSION_PROFILE": "",
        }.items():
            monkeypatch.setenv(name, value)
        result = tui.prepare(workspace, "synthetic-test", root, "a" * 32)
        yield Path(result["run_dir"])


def event(run, seq, kind="turn_settled", epoch="b" * 32, session="synthetic-omp"):
    return {
        "version": 1,
        "type": "event",
        "run_id": run.name,
        "epoch": epoch,
        "omp_session_id": session,
        "seq": seq,
        "kind": kind,
        "at_ms": 1000,
    }


def test_prepare_rejects_a_name_the_extension_cannot_accept(run):
    with pytest.raises(tui.TUIError, match="invalid_tmux_session"):
        tui.prepare(run.parent, "a" * 65, run.parent)


def test_default_root_follows_the_current_real_profile_home(run, monkeypatch):
    for name in ("h", "profile"):
        home = run.parent / name
        home.mkdir(exist_ok=True)
        monkeypatch.setenv("HERMES_HOME", str(home))
        monkeypatch.setenv("HERMES_SESSION_PROFILE", "ignored-profile-label")
        root = home / "omp-supervisor"
        expected_run = root / ("b" * 32)
        if len(os.fsencode(expected_run / "bridge.sock")) > 107:
            with pytest.raises(tui.TUIError, match="socket_path_too_long"):
                tui.prepare(run.parent, "profile-test", run_id=expected_run.name)
            assert not expected_run.exists()
        else:
            result = tui.prepare(run.parent, "profile-test", run_id=expected_run.name)
            assert Path(result["run_dir"]) == expected_run
            binding = json.loads((expected_run / "binding.json").read_text())
            assert binding["owner"]["hermes_home"] == str(home)
        assert root.is_dir()
        assert stat.S_IMODE(root.stat().st_mode) == 0o700


@pytest.mark.parametrize(
    "platform", ["slack", "telegram", "matrix", "custom.v2-bridge"]
)
@pytest.mark.parametrize(
    "field,replacement",
    [("HERMES_SESSION_PLATFORM", "discord"), ("HERMES_SESSION_ID", "next-generation")],
)
def test_non_discord_owner_is_accepted_but_mismatches_are_denied(
    run, monkeypatch, platform, field, replacement
):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", platform)
    result = tui.prepare(run.parent, "other-platform", run.parent, "c" * 32)
    other_run = Path(result["run_dir"])
    assert tui.status(other_run)["run_id"] == other_run.name
    binding = json.loads((other_run / "binding.json").read_text())
    assert binding["owner"]["platform"] == platform
    monkeypatch.setenv(field, replacement)
    for operation in (tui.status, tui.watch):
        with pytest.raises(tui.TUIError, match="scope_denied"):
            operation(other_run)
    with pytest.raises(tui.TUIError, match="scope_denied"):
        tui.launch(other_run, run.parent / "unused-prompt")
    assert not (other_run / "cursor.json").exists()
    assert not (other_run / "launch.json").exists()


@pytest.mark.parametrize("platform", ["", " ", "a/b", "a\n", "a\x7f", "a" * 65])
def test_malformed_platform_denied(run, monkeypatch, platform):
    monkeypatch.setenv("HERMES_SESSION_PLATFORM", platform)
    with pytest.raises(tui.TUIError):
        tui.current_owner()


def test_default_root_rejects_unsafe_permissions(run):
    root = Path(tui.current_owner()["hermes_home"]) / "omp-supervisor"
    root.mkdir(mode=0o755)
    root.chmod(0o755)
    with pytest.raises(tui.TUIError, match="unsafe_directory"):
        tui.prepare(run.parent, "profile-test")


def test_default_root_rejects_symlinks(run):
    root = Path(tui.current_owner()["hermes_home"]) / "omp-supervisor"
    root.symlink_to(run.parent, target_is_directory=True)
    with pytest.raises(tui.TUIError, match="unsafe_directory"):
        tui.prepare(run.parent, "profile-test")


def hello(run, seq=0, epoch="b" * 32, session="synthetic-omp", state="idle"):
    return {
        "version": 1,
        "type": "hello",
        "run_id": run.name,
        "epoch": epoch,
        "omp_session_id": session,
        "seq": seq,
        "pid": os.getpid(),
        "state": state,
    }


def journal(run, events, state="idle", **kwargs):
    value = hello(run, events[-1]["seq"] if events else 0, state=state, **kwargs)
    value.pop("type")
    value["events"] = events
    tui._write_json(run / "journal.json", value)


@contextlib.contextmanager
def server(run, connections):
    requests = []
    errors = []
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(str(run / "bridge.sock"))
    os.chmod(run / "bridge.sock", 0o600)
    listener.listen()
    listener.settimeout(3)

    def serve():
        try:
            for frames in connections:
                with listener.accept()[0] as client:
                    client.settimeout(3)
                    request = b""
                    while not request.endswith(b"\n"):
                        request += client.recv(4096)
                    requests.append(json.loads(request))
                    for frame in frames:
                        payload = (
                            frame
                            if isinstance(frame, bytes)
                            else (json.dumps(frame) + "\n").encode()
                        )
                        client.sendall(payload)
        except Exception as exc:
            errors.append(exc)

    worker = threading.Thread(target=serve, daemon=True)
    worker.start()
    try:
        yield requests
    finally:
        worker.join(4)
        listener.close()
        assert not worker.is_alive()
        assert not errors


@pytest.mark.parametrize(
    "name",
    [
        "HERMES_HOME",
        "HERMES_SESSION_PLATFORM",
        "HERMES_SESSION_KEY",
        "HERMES_SESSION_ID",
        "HERMES_SESSION_CHAT_ID",
        "HERMES_SESSION_THREAD_ID",
    ],
)
def test_missing_scope_denied(run, monkeypatch, name):
    monkeypatch.delenv(name)
    with pytest.raises(tui.TUIError):
        tui.status(run)


@pytest.mark.parametrize(
    "name,value",
    [
        ("HERMES_SESSION_ID", "other"),
        ("HERMES_SESSION_KEY", "other"),
        ("HERMES_SESSION_CHAT_ID", "other"),
        ("HERMES_SESSION_THREAD_ID", "other"),
        ("HERMES_SESSION_PLATFORM", "cli"),
    ],
)
def test_scope_mismatch_all_actions(run, monkeypatch, name, value):
    monkeypatch.setenv(name, value)
    for action in (
        lambda: tui.status(run),
        lambda: tui.watch(run, 1),
        lambda: tui.launch(run, "absent", "absent", "absent"),
    ):
        with pytest.raises(tui.TUIError):
            action()
    assert not (run / "launch.json").exists()


def test_home_is_profile_identity(run, monkeypatch):
    other = run.parent / "other-home"
    other.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(other))
    with pytest.raises(tui.TUIError, match="scope_denied"):
        tui.status(run)


def test_prepare_private_immutable_and_no_profile_dependency(run, monkeypatch):
    original = (run / "binding.json").read_bytes()
    binding = json.loads(original)
    assert binding["mode"] == "tui"
    assert "profile" not in binding["owner"]
    assert stat.S_IMODE(run.stat().st_mode) == 0o700
    assert stat.S_IMODE((run / "binding.json").stat().st_mode) == 0o600
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: pytest.fail("prepare must not execute commands"),
    )
    with pytest.raises(tui.TUIError, match="binding_already_exists"):
        tui.prepare(binding["workspace"], "different", run.parent, run.name)
    assert (run / "binding.json").read_bytes() == original
    assert tui.status(run)["run_id"] == run.name


@pytest.mark.parametrize(
    "bad", ["A" * 32, "x" * 32, "a" * 31, "a" * 33, "../bad", " a"]
)
def test_bad_run_ids_preserved_and_rejected(run, bad):
    with pytest.raises(tui.TUIError, match="invalid_run_id"):
        tui.prepare(run.parent / "w", "safe", run.parent, bad)


def test_private_directory_and_file_checks(run, monkeypatch):
    os.chmod(run, 0o755)
    with pytest.raises(tui.TUIError, match="unsafe_directory"):
        tui.status(run)
    os.chmod(run, 0o700)
    os.chmod(run / "binding.json", 0o644)
    with pytest.raises(tui.TUIError, match="unsafe_file"):
        tui.status(run)
    os.chmod(run / "binding.json", 0o600)
    uid = os.getuid()
    monkeypatch.setattr(tui.os, "getuid", lambda: uid + 1)
    with pytest.raises(tui.TUIError, match="unsafe_directory"):
        tui.status(run)


def test_symlink_binding_and_lock_denied(run):
    binding = run / "binding.json"
    real = run / "original.json"
    binding.rename(real)
    binding.symlink_to(real)
    with pytest.raises(OSError):
        tui.status(run)
    binding.unlink()
    real.rename(binding)
    (run / "observation.lock").symlink_to(binding)
    with pytest.raises(OSError):
        tui.watch(run, 1)
    assert json.loads(binding.read_bytes())["mode"] == "tui"


def test_single_watcher_lock(run):
    with tui.observation_lock(run):
        assert tui.status(run)["observer_active"]
        with pytest.raises(tui.TUIError, match="observer_already_running"):
            tui.watch(run, 1)
    assert not tui.status(run)["observer_active"]


def test_cursor_commit_precedes_notification_and_deduplicates(run, capsys):
    journal(run, [event(run, 1, "started"), event(run, 2)])
    assert tui.main(["watch", "--run-dir", str(run), "--timeout", "1"]) == 0
    notice = json.loads(capsys.readouterr().out)
    cursor = json.loads((run / "cursor.json").read_bytes())
    assert cursor["receipt"] == notice
    assert cursor["status"] == "ready_for_native_notification"
    journal(
        run, [event(run, 1, "started"), event(run, 2), event(run, 3, "needs_input")]
    )
    assert tui.watch(run, 1)["seq"] == 3
    assert tui.status(run)["cursor"]["seq"] == 3


@pytest.mark.parametrize(
    "changed,reason",
    [
        ({"epoch": "c" * 32}, "epoch_changed"),
        ({"session": "different"}, "session_changed"),
    ],
)
def test_changed_epoch_or_omp_session_fails_closed(run, changed, reason):
    journal(run, [event(run, 1)])
    tui.watch(run, 1)
    journal(run, [event(run, 2, **changed)], **changed)
    receipt = tui.watch(run, 1)
    assert receipt["kind"] == "observation_lost"
    assert receipt["reason"] == reason
    assert receipt["seq"] == 1
    with pytest.raises(tui.TUIError, match="observation_closed"):
        tui.watch(run, 1)


def test_gap_and_bounded_journal(run):
    journal(run, [event(run, 5)])
    assert tui.watch(run, 1)["reason"] == "event_gap"
    journal(run, [event(run, i) for i in range(1, 130)])
    with pytest.raises(tui.TUIError, match="invalid_journal"):
        tui.status(run)


@pytest.mark.parametrize(
    "kind,state",
    [
        ("shutdown", "closed"),
        ("session_revoked", "revoked"),
        ("observation_lost", "closed"),
    ],
)
def test_terminal_journal_is_observed_without_socket_then_not_rearmed(run, kind, state):
    journal(run, [event(run, 1, kind)], state=state)
    assert tui.watch(run, 1)["kind"] == kind
    with pytest.raises(tui.TUIError, match="observation_closed"):
        tui.watch(run, 1)


def test_socket_reconnect_preserves_epoch_and_cursor(run):
    with server(
        run,
        [
            [hello(run), event(run, 1, "started")],
            [hello(run, 2), event(run, 1, "started"), event(run, 2)],
        ],
    ) as requests:
        receipt = tui.watch(run, 2)
    assert receipt["kind"] == "turn_settled"
    assert requests[0]["after_seq"] == 0
    assert requests[1]["after_seq"] == 1
    assert requests[1]["epoch"] == "b" * 32


@pytest.mark.parametrize(
    "bad", [b"x" * 4097, b'{"version":1,"version":1}\n', b"[]\n", b"\xff\n"]
)
def test_socket_frame_limits_and_malformed_json(run, bad):
    with server(run, [[bad]]):
        receipt = tui.watch(run, 1)
    assert receipt["kind"] == "observation_lost"
    assert receipt["reason"] == "invalid_protocol"


def test_unknown_fields_and_wrong_run_are_rejected(run):
    frame = hello(run)
    frame["raw_output"] = "must-not-escape"
    with server(run, [[frame]]):
        receipt = tui.watch(run, 1)
    assert receipt["reason"] == "invalid_protocol"
    assert "must-not-escape" not in (run / "cursor.json").read_text()


def test_timeout_stops_observation_not_omp(run, monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: pytest.fail("watch must not execute commands"),
    )
    monkeypatch.setattr(os, "kill", lambda *a, **k: pytest.fail("must not signal OMP"))
    receipt = tui.watch(run, 0.01)
    assert receipt["kind"] == "observation_lost"
    assert receipt["reason"] == "timeout"
    assert not tui.status(run)["observer_active"]


def test_timeout_preserves_owner_and_cursor_for_authorized_rearm(run, monkeypatch):
    journal(run, [event(run, 1, "started")], state="busy")
    with monkeypatch.context() as deadline:
        ticks = iter([0])
        deadline.setattr(tui.time, "monotonic", lambda: next(ticks, 2))
        receipt = tui.watch(run, timeout=1)
    assert (receipt["kind"], receipt["reason"]) == ("observation_lost", "timeout")
    expired = tui.status(run)
    assert not expired["observer_active"]
    assert expired["cursor"]["terminal"] is False
    assert (expired["cursor"]["epoch"], expired["cursor"]["seq"]) == ("b" * 32, 1)
    with monkeypatch.context() as other_owner:
        other_owner.setenv("HERMES_SESSION_ID", "different-generation")
        with pytest.raises(tui.TUIError, match="scope_denied"):
            tui.watch(run, timeout=1)
    journal(run, [event(run, 1, "started"), event(run, 2)])
    later = tui.watch(run, timeout=1)
    assert (later["kind"], later["seq"], later["epoch"]) == (
        "turn_settled",
        2,
        "b" * 32,
    )


def test_reconnect_is_bounded(run, monkeypatch):
    monkeypatch.setattr(tui, "MAX_RECONNECTS", 1)
    assert tui.watch(run, 2)["reason"] == "reconnect_exhausted"


@pytest.fixture
def launch_inputs(run, monkeypatch):
    prompt = run.parent / "prompt with ' quotes.txt"
    prompt.write_text("SYNTHETIC SECRET MUST NOT BE PRINTED")

    extension = run.parent / "extension.ts"
    extension.write_text("export default () => {}")
    executable = run.parent / "mock-executable"
    executable.write_text("not executed")
    executable.chmod(0o700)

    monkeypatch.setattr(tui, "EXTENSION", extension)
    return dict(
        run_dir=run,
        prompt_file=prompt,
        omp_executable=executable,
        tmux_executable=executable,
    )


def test_launch_requires_observer(run, launch_inputs):
    with pytest.raises(tui.TUIError, match="native_observer_required"):
        tui.launch(**launch_inputs)
    assert not (run / "launch.json").exists()


def test_launch_shell_quoting_canary_and_durable_at_most_once(
    run, launch_inputs, monkeypatch
):
    calls = []

    def invoke(argv, **kwargs):
        calls.append(argv)
        assert (run / "launch.json").exists()
        assert kwargs["stdout"] == subprocess.DEVNULL
        return SimpleNamespace(returncode=1 if len(calls) == 1 else 0)

    monkeypatch.setattr(subprocess, "run", invoke)
    with tui.observation_lock(run):
        result = tui.launch(**launch_inputs, canary=True)
        with pytest.raises(tui.TUIError, match="launch_already_attempted"):
            tui.launch(**launch_inputs, canary=True)
    assert result["status"] == "tmux_session_present"
    assert [call[1] for call in calls] == ["has-session", "new-session", "has-session"]
    command = shlex.split(calls[1][-1])
    assert command[:2] == [
        "env",
        "OMP_HERMES_BINDING_FILE=" + str(run / "binding.json"),
    ]
    assert command[-1] == "@" + str(launch_inputs["prompt_file"])
    for flag in (
        "--no-tools",
        "--no-skills",
        "--no-rules",
        "--no-extensions",
        "--no-title",
        "--no-session",
        "--no-lsp",
        "-e",
    ):
        assert flag in command
    assert "-e" not in calls[1][1:-1]  # tmux 2.7 has no -e environment option.
    assert "--mode" not in command and "--print" not in command
    assert not any(
        arg.startswith(("--model", "--thinking", "--append-system-prompt"))
        for arg in command
    )
    assert "SYNTHETIC SECRET" not in json.dumps(result)
    assert "SYNTHETIC SECRET" not in (run / "launch.json").read_text()


@pytest.mark.parametrize("thinking", tui.THINKING_LEVELS)
def test_custom_launch_options_are_quoted_and_path_defaults_work(
    run, launch_inputs, monkeypatch, capsys, thinking
):
    for name in ("omp", "tmux"):
        (run.parent / name).symlink_to(launch_inputs["omp_executable"])
    monkeypatch.setenv("PATH", str(run.parent))
    system_prompt = run.parent / "policy with ' quotes; $(not-a-command).md"
    system_prompt.write_text("SYNTHETIC SYSTEM PROMPT MUST NOT BE PRINTED")
    model = "provider/model with ' quotes; $(not-a-command)"
    calls = []

    def invoke(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=1 if len(calls) == 1 else 0)

    monkeypatch.setattr(subprocess, "run", invoke)
    with tui.observation_lock(run):
        args = [
            "launch",
            "--run-dir",
            str(run),
            "--prompt-file",
            str(launch_inputs["prompt_file"]),
            "--model",
            model,
            "--thinking",
            thinking,
            "--append-system-prompt",
            str(system_prompt),
        ]
        assert tui.main(args) == 0
    command = shlex.split(calls[1][-1])
    assert command[2] == str(run.parent / "omp")
    assert calls[0][0] == str(run.parent / "tmux")
    assert command[3:7] == [
        "--model=" + model,
        "--thinking=" + thinking,
        "--append-system-prompt",
        str(system_prompt),
    ]
    assert command[-1] == "@" + str(launch_inputs["prompt_file"])
    assert "--no-tools" not in command
    output = capsys.readouterr()
    assert not output.err
    assert "SYNTHETIC" not in output.out
    assert "SYNTHETIC" not in (run / "launch.json").read_text()


@pytest.mark.parametrize(
    "options,reason",
    [
        ({"model": ""}, "invalid_model"),
        ({"model": " "}, "invalid_model"),
        ({"model": "bad\nmodel"}, "invalid_model"),
        ({"model": []}, "invalid_model"),
        ({"model": "a" * 1025}, "invalid_model"),
        ({"thinking": "unknown"}, "invalid_thinking"),
        ({"thinking": []}, "invalid_thinking"),
        ({"append_system_prompt": ""}, "invalid_system_prompt"),
        ({"append_system_prompt": "bad\x00path"}, "invalid_system_prompt"),
        ({"append_system_prompt": []}, "invalid_system_prompt"),
    ],
)
def test_malformed_launch_options_fail_before_intent(
    run, launch_inputs, monkeypatch, options, reason
):
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: pytest.fail("must not launch")
    )
    with pytest.raises(tui.TUIError, match=reason):
        tui.launch(**launch_inputs, **options)
    assert not (run / "launch.json").exists()


@pytest.mark.parametrize(
    "kind", ["empty", "symlink", "directory", "oversized", "missing"]
)
def test_system_prompt_file_validation_precedes_launch(
    run, launch_inputs, monkeypatch, kind
):
    path = run.parent / "system-prompt"
    if kind == "empty":
        path.write_text(" \n")
    elif kind == "symlink":
        path.symlink_to(launch_inputs["prompt_file"])
    elif kind == "directory":
        path.mkdir()
    elif kind == "oversized":
        path.write_bytes(b"x" * (1024 * 1024 + 1))
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **kw: pytest.fail("must not launch")
    )
    with pytest.raises((tui.TUIError, OSError)):
        tui.launch(**launch_inputs, append_system_prompt=path)
    assert not (run / "launch.json").exists()


@pytest.mark.parametrize("name", ["omp_executable", "tmux_executable"])
def test_missing_path_executable_fails_before_intent(
    run, launch_inputs, monkeypatch, name
):
    monkeypatch.setenv("PATH", str(run.parent))
    launch_inputs[name] = "not-installed"
    with pytest.raises(tui.TUIError, match="invalid_executable"):
        tui.launch(**launch_inputs)
    assert not (run / "launch.json").exists()


@pytest.mark.parametrize("failure", ["existing", "timeout", "nonzero"])
def test_failed_or_ambiguous_launch_never_retries_or_kills(
    run, launch_inputs, monkeypatch, failure
):
    calls = []

    def invoke(argv, **kwargs):
        calls.append(argv)
        if len(calls) == 1:
            return SimpleNamespace(returncode=0 if failure == "existing" else 1)
        if failure == "timeout":
            raise subprocess.TimeoutExpired(argv, 15)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(subprocess, "run", invoke)
    with tui.observation_lock(run):
        with pytest.raises(tui.TUIError):
            tui.launch(**launch_inputs)
        count = len(calls)
        with pytest.raises(tui.TUIError, match="launch_already_attempted"):
            tui.launch(**launch_inputs)
    assert len(calls) == count
    assert all(call[1] in {"has-session", "new-session"} for call in calls)


def test_json_size_limit_and_sanitized_cli_failure(run, capsys):
    target = run / "journal.json"
    target.write_bytes(b" " * (tui.MAX_JSON + 1))
    target.chmod(0o600)
    assert tui.main(["status", "--run-dir", str(run)]) == 2
    output = capsys.readouterr()
    assert not output.out
    assert json.loads(output.err) == {"error": "file_too_large"}


@pytest.mark.parametrize(
    "field,value",
    [
        ("state", []),
        ("seq", True),
        ("version", True),
        ("epoch", {}),
        ("run_id", "c" * 32),
    ],
)
def test_invalid_wire_types_fail_closed(run, field, value):
    frame = hello(run)
    frame[field] = value
    with server(run, [[frame]]):
        assert tui.watch(run, 1)["reason"] == "invalid_protocol"


def test_interrupted_watcher_only_closes_observation(run, monkeypatch):
    def interrupt(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(tui, "_connect", interrupt)
    monkeypatch.setattr(os, "kill", lambda *a: pytest.fail("must never terminate OMP"))
    assert tui.watch(run, 1)["reason"] == "observer_interrupted"
    assert not tui.status(run)["observer_active"]
