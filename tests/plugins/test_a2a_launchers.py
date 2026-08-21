"""Phase 1 contracts for validated A2A launcher routing."""

from __future__ import annotations

from plugins.platforms.a2a import protocol
from plugins.platforms.a2a.launchers import (
    LaunchOutcome,
    LaunchRequest,
    parse_launcher_spec,
)


def _adapter(extra):
    from gateway.config import PlatformConfig
    from plugins.platforms.a2a.adapter import A2AAdapter

    return A2AAdapter(PlatformConfig(enabled=True, extra={"agents": extra}))


def test_launcher_specs_are_immutable_and_validate_transport_shapes():
    process = parse_launcher_spec({
        "transport": "process",
        "start": ["tool", "{prompt}"],
        "timeout": 3,
    })
    assert process.start == ("tool", "{prompt}")
    try:
        process.timeout = 4
    except AttributeError:
        pass
    else:
        raise AssertionError("launcher spec must be immutable")

    for invalid in (
        {"transport": "process", "start": "tool"},
        {
            "transport": "pi_rpc",
            "protocol_profile": "unknown",
            "command": ["tool", "--mode", "rpc"],
        },
        {
            "transport": "pi_rpc",
            "protocol_profile": "omp",
            "command": ["tool", "--mode=rpc"],
        },
    ):
        try:
            parse_launcher_spec(invalid)
        except ValueError:
            continue
        raise AssertionError("invalid launcher specification accepted")


def test_root_route_stays_live_when_served_launcher_is_configured():
    adapter = _adapter({
        "external": {"launcher": {"transport": "process", "start": ["tool"]}}
    })
    assert adapter._route_for_path("/")["agent"]["slug"] == ""
    assert adapter._route_for_path("/external/")["agent"]["slug"] == "external"
    assert adapter._agents[""]["local"] is True


def test_invalid_launcher_route_is_omitted_from_path_and_tenant_routing():
    adapter = _adapter({
        "bad": {
            "tenant": "bad",
            "launcher": {"transport": "process", "start": "not-argv"},
        }
    })
    assert "bad" not in adapter._agents
    assert adapter._route_for_request("/", {"tenant": "bad"})["agent"]["slug"] == ""
    assert {entry["slug"] for entry in adapter._served_agent_summary()} == {"default"}


def test_explicit_launcher_overrides_profile_on_its_route():
    adapter = _adapter({
        "mixed": {
            "profile": "other",
            "launcher": {"transport": "process", "start": ["tool"]},
        }
    })
    route = adapter._route_for_path("/mixed/")["agent"]
    assert route["local"] is False
    assert route["launcher_spec"] is not None


def test_external_input_required_uses_common_finalizer():
    adapter = _adapter({"profile": {"profile": "profile"}})
    adapter.launchers.send = lambda request: LaunchOutcome(
        protocol.STATE_COMPLETED, "[INPUT_REQUIRED] Need detail"
    )
    terminal, pending = adapter._prepare_task(
        {
            "message": protocol.text_message(
                protocol.ROLE_USER, "hello", context_id="ctx"
            )
        },
        "peer",
        adapter._agents["profile"],
    )
    assert pending is None
    assert terminal["status"]["state"] == protocol.STATE_INPUT_REQUIRED
    assert protocol.extract_text(terminal["status"]["message"]) == "Need detail"


def test_launch_request_preserves_route_attribution_and_context():
    request = LaunchRequest("task", "research", "peer", "context/unsafe", "prompt")
    assert (request.task_id, request.agent_slug, request.context_id) == (
        "task",
        "research",
        "context/unsafe",
    )


# These use CPython helpers rather than mocks so pipe, argv, and process-tree
# ownership are exercised by the OS implementation.
import json
import sys
import threading
import time

from plugins.platforms.a2a.launchers import ProcessLauncher, _session_key


def _process_launcher(tmp_path, script, **settings):
    raw = {
        "transport": "process",
        "start": [sys.executable, "-c", script],
        "cwd": str(tmp_path),
        "timeout": 2,
        **settings,
    }
    return ProcessLauncher(parse_launcher_spec(raw), str(tmp_path / "home"))


def _request(task_id="task", slug="route", context="context", prompt="hello"):
    return LaunchRequest(task_id, slug, "peer", context, prompt)


def test_process_spec_rejects_nonfinite_or_unknown_placeholders(tmp_path):
    for raw in (
        {"transport": "process", "start": ["tool", "{unknown}"]},
        {"transport": "process", "start": ["tool", "{prompt!r}"]},
        {"transport": "process", "start": ["tool"], "timeout": float("inf")},
        {
            "transport": "process",
            "start": ["tool"],
            "cwd": str(tmp_path),
            "resume": ["tool", "{session_key}"],
        },
    ):
        try:
            parse_launcher_spec(raw)
        except ValueError:
            continue
        raise AssertionError("invalid process launcher configuration accepted")


def test_process_keeps_prompt_as_one_shell_free_argv_element(tmp_path):
    launcher = _process_launcher(
        tmp_path,
        "import sys; print(json.dumps(sys.argv[1:]))",
        start=[
            sys.executable,
            "-c",
            "import sys; print(repr(sys.argv[1]))",
            "{prompt}",
        ],
    )
    prompt = "x; touch SHOULD_NOT_EXIST $(echo nope)"
    outcome = launcher.send(_request(prompt=prompt))
    assert outcome.state == protocol.STATE_COMPLETED
    assert outcome.reply == repr(prompt)
    assert not (tmp_path / "SHOULD_NOT_EXIST").exists()


def test_process_uses_explicit_cwd_and_minimal_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("A2A_KEEP", "allowed")
    monkeypatch.setenv("A2A_SECRET", "not-for-child")
    script = "import json, os; print(json.dumps([os.getcwd(), os.getenv('A2A_KEEP'), os.getenv('A2A_SECRET'), os.getenv('LITERAL')]))"
    launcher = _process_launcher(
        tmp_path, script, pass_env=["A2A_KEEP"], env={"LITERAL": "fixed"}
    )
    outcome = launcher.send(_request())
    assert json.loads(outcome.reply) == [str(tmp_path), "allowed", None, "fixed"]


def test_process_text_stream_selection_and_session_marker_stripping(tmp_path):
    script = "import sys; print('session: opaque-1\\nanswer'); print('diagnostic', file=sys.stderr)"
    launcher = _process_launcher(
        tmp_path,
        script,
        resume=[sys.executable, "-c", script, "{session_id}"],
        output={
            "format": "text",
            "reply_from": "stdout",
            "session_id_from": "stdout",
            "session_id_regex": r"session:\s*(\S+)",
            "strip_session_match": True,
        },
    )
    outcome = launcher.send(_request())
    assert outcome.state == protocol.STATE_COMPLETED
    assert outcome.reply == "answer"
    assert outcome.session_id == "opaque-1"


def test_process_can_select_stderr_reply(tmp_path):
    launcher = _process_launcher(
        tmp_path,
        "import sys; print('noise'); print('answer', file=sys.stderr)",
        output={"format": "text", "reply_from": "stderr"},
    )
    assert launcher.send(_request()).reply == "answer"


def test_process_json_reply_and_optional_session_fields(tmp_path):
    script = "import json; print(json.dumps({'result': {'text': 'answer'}, 'meta': {'id': 'opaque-2'}}))"
    launcher = _process_launcher(
        tmp_path,
        script,
        resume=[sys.executable, "-c", script, "{session_id}"],
        output={
            "format": "json",
            "reply_field": "result.text",
            "session_id_field": "meta.id",
        },
    )
    outcome = launcher.send(_request())
    assert (outcome.state, outcome.reply, outcome.session_id) == (
        protocol.STATE_COMPLETED,
        "answer",
        "opaque-2",
    )


def test_process_fails_for_empty_nonzero_missing_invalid_and_overflow_output(tmp_path):
    cases = [
        (_process_launcher(tmp_path, "pass"), "empty"),
        (
            _process_launcher(
                tmp_path, "import sys; print('bad', file=sys.stderr); sys.exit(4)"
            ),
            "bad",
        ),
        (
            _process_launcher(
                tmp_path,
                "print('{bad json')",
                output={"format": "json", "reply_field": "answer"},
            ),
            "JSON",
        ),
        (
            _process_launcher(
                tmp_path, "import sys; sys.stdout.write('x' * (4 * 1024 * 1024 + 1))"
            ),
            "capture limit",
        ),
    ]
    for launcher, expected in cases:
        outcome = launcher.send(_request())
        assert outcome.state == protocol.STATE_FAILED
        assert expected in outcome.reply
    missing = ProcessLauncher(
        parse_launcher_spec({
            "transport": "process",
            "start": ["definitely-not-an-a2a-executable"],
            "cwd": str(tmp_path),
        }),
        str(tmp_path / "home"),
    )
    assert missing.send(_request()).state == protocol.STATE_FAILED


def test_process_timeout_terminates_child_tree(tmp_path):
    marker = tmp_path / "descendant-marker"
    script = (
        "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(0.4); open(r\""
        + str(marker)
        + '", "w").write("alive")\']); time.sleep(10)'
    )
    launcher = _process_launcher(tmp_path, script, timeout=0.1)
    assert launcher.send(_request()).state == protocol.STATE_FAILED
    time.sleep(0.6)
    assert not marker.exists()


def test_process_cancel_and_close_terminate_descendants(tmp_path):
    marker = tmp_path / "cancel-marker"
    script = (
        "import subprocess, sys, time; subprocess.Popen([sys.executable, '-c', "
        "'import time; time.sleep(0.4); open(r\""
        + str(marker)
        + '", "w").write("alive")\']); time.sleep(10)'
    )
    launcher = _process_launcher(tmp_path, script, timeout=5)
    thread = threading.Thread(target=launcher.send, args=(_request("cancel-task"),))
    thread.start()
    for _ in range(100):
        if launcher.cancel("cancel-task"):
            break
        time.sleep(0.01)
    else:
        raise AssertionError("process was not registered for cancellation")
    thread.join(3)
    launcher.close()
    time.sleep(0.6)
    assert not thread.is_alive()
    assert not marker.exists()


def test_launcher_manager_tracks_only_the_active_task(tmp_path):
    from plugins.platforms.a2a.launchers import LauncherManager

    launcher = _process_launcher(tmp_path, "import time; time.sleep(10)", timeout=20)
    manager = LauncherManager(str(tmp_path / "home"))
    manager._launchers["route"] = launcher
    thread = threading.Thread(target=manager.send, args=(_request("managed-cancel"),))
    thread.start()
    for _ in range(100):
        if manager.cancel("managed-cancel"):
            break
        time.sleep(0.01)
    else:
        raise AssertionError("manager did not register active launcher")
    thread.join(3)
    assert not thread.is_alive()
    assert manager.active_task_ids() == ()


def test_process_deterministic_session_key_is_safe_and_context_scoped(tmp_path):
    first = _session_key("route", "context/unsafe")
    assert first == _session_key("route", "context/unsafe")
    assert first != _session_key("route", "other")
    assert len(first) == 64 and all(char in "0123456789abcdef" for char in first)
    launcher = _process_launcher(
        tmp_path,
        "import sys; print(sys.argv[-1])",
        start=[
            sys.executable,
            "-c",
            "import sys; print(sys.argv[-1])",
            "{session_key}",
        ],
    )
    assert launcher.send(_request(context="context/unsafe")).reply == first


def test_process_opaque_mapping_survives_recreation_and_is_scoped_and_private(tmp_path):
    script = "import sys; print('answer'); print('session: generated', file=sys.stderr)"
    raw = {
        "transport": "process",
        "start": [sys.executable, "-c", script],
        "resume": [
            sys.executable,
            "-c",
            "import sys; print(sys.argv[-1])",
            "{session_id}",
        ],
        "cwd": str(tmp_path),
        "output": {
            "format": "text",
            "reply_from": "stdout",
            "session_id_from": "stderr",
            "session_id_regex": r"session:\s*(\S+)",
        },
    }
    spec = parse_launcher_spec(raw)
    home = tmp_path / "home"
    assert ProcessLauncher(spec, str(home)).send(_request()).session_id == "generated"
    assert ProcessLauncher(spec, str(home)).send(_request()).reply == "generated"
    assert (
        ProcessLauncher(spec, str(home)).send(_request(slug="other")).reply == "answer"
    )
    assert (
        ProcessLauncher(spec, str(tmp_path / "other-home")).send(_request()).reply
        == "answer"
    )
    state = home / "a2a_launchers" / "sessions.json"
    assert oct(state.stat().st_mode & 0o777) == "0o600"


def test_process_missing_optional_session_and_corrupt_mapping_start_fresh(tmp_path):
    script = "print('answer')"
    raw = {
        "transport": "process",
        "start": [sys.executable, "-c", script],
        "resume": [
            sys.executable,
            "-c",
            "import sys; print(sys.argv[-1])",
            "{session_id}",
        ],
        "cwd": str(tmp_path),
        "output": {
            "format": "text",
            "reply_from": "stdout",
            "session_id_from": "stderr",
            "session_id_regex": r"session:\s*(\S+)",
        },
    }
    home = tmp_path / "home"
    launcher = ProcessLauncher(parse_launcher_spec(raw), str(home))
    assert launcher.send(_request()).reply == "answer"
    state = home / "a2a_launchers" / "sessions.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text("not-json", encoding="utf-8")
    assert (
        ProcessLauncher(parse_launcher_spec(raw), str(home)).send(_request()).reply
        == "answer"
    )


# The RPC child is deliberately stdlib-only: every lifecycle frame is emitted
# through the same pipe contract as the real versioned workers.
from plugins.platforms.a2a.pi_rpc import PiRpcLauncher


_RPC_CHILD = r"""
import json, sys, time
profile = sys.argv[1]
if profile == 'omp':
    print(json.dumps({'type': 'ready', 'maxFrameBytes': 1048576}), flush=True)
for raw in sys.stdin:
    if raw == 'MALFORMED\n':
        print('{bad-json', flush=True); continue
    request = json.loads(raw)
    rid = request.get('id')
    if request['type'] == 'extension_ui_response':
        continue
    if request['type'] == 'abort':
        print(json.dumps({'type': 'response', 'id': rid, 'success': True}), flush=True)
        print(json.dumps({'type': 'agent_end' if profile == 'omp' else 'agent_settled'}), flush=True)
        continue
    prompt = request['message']
    if prompt == 'crash': sys.exit(3)
    if prompt == 'malformed': print('{bad-json', flush=True); continue
    if prompt == 'oversize': print('x' * (1024 * 1024 + 1), flush=True); continue
    if prompt == 'wait': time.sleep(10); continue
    if prompt == 'reject':
        print(json.dumps({'type': 'response', 'id': rid, 'success': False}), flush=True); continue
    if prompt == 'ui':
        print(json.dumps({'type': 'extension_ui_request', 'id': 'dialog', 'method': 'input'}), flush=True)
        for followup_raw in sys.stdin:
            followup = json.loads(followup_raw)
            if followup['type'] == 'extension_ui_response':
                assert followup == {'type': 'extension_ui_response', 'id': 'dialog', 'cancelled': True}
                continue
            if followup['type'] == 'abort':
                print(json.dumps({'type': 'response', 'id': followup['id'], 'success': True}), flush=True)
                print(json.dumps({'type': 'agent_end' if profile == 'omp' else 'agent_settled'}), flush=True)
                break
        continue
    print(json.dumps({'type': 'response', 'id': rid, 'success': True}), flush=True)
    print(json.dumps({'type': 'response', 'id': rid, 'success': True}), flush=True)
    if prompt == 'early': print(json.dumps({'type': 'available_commands_update'}), flush=True)
    if profile == 'feynman':
        print(json.dumps({'type': 'message_end', 'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': 'intermediate'}]}}), flush=True)
    print(json.dumps({'type': 'message_end', 'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': prompt.upper()}]}}), flush=True)
    print(json.dumps({'type': 'agent_end'}), flush=True)
    if profile == 'feynman': print(json.dumps({'type': 'agent_settled'}), flush=True)
"""


def _rpc_launcher(tmp_path, profile, **settings):
    raw = {
        "transport": "pi_rpc",
        "protocol_profile": profile,
        "command": [sys.executable, "-u", "-c", _RPC_CHILD, profile, "--mode", "rpc"],
        "cwd": str(tmp_path),
        "timeout": 1,
        "startup_timeout": 0.3,
        "idle_timeout": 1,
        **settings,
    }
    return PiRpcLauncher(parse_launcher_spec(raw))


def test_rpc_versioned_profiles_select_authoritative_terminal_and_assistant(tmp_path):
    omp = _rpc_launcher(tmp_path, "omp")
    feynman = _rpc_launcher(tmp_path, "feynman")
    assert omp.send(_request(prompt="omp")).reply == "OMP"
    assert feynman.send(_request(prompt="feynman")).reply == "FEYNMAN"
    omp.close()
    feynman.close()


def test_rpc_retains_early_unmatched_and_multiple_correlated_responses(tmp_path):
    launcher = _rpc_launcher(tmp_path, "omp")
    assert launcher.send(_request(prompt="early")).reply == "EARLY"
    assert launcher.send(_request(task_id="next", prompt="next")).reply == "NEXT"
    launcher.close()


def test_rpc_reuses_exact_context_and_isolates_distinct_contexts(tmp_path):
    launcher = _rpc_launcher(tmp_path, "omp")
    assert launcher.send(_request(context="same", prompt="one")).reply == "ONE"
    first = launcher._workers[("route", "same")]
    assert (
        launcher.send(_request(task_id="two", context="same", prompt="two")).reply
        == "TWO"
    )
    assert launcher._workers[("route", "same")] is first
    assert (
        launcher.send(_request(task_id="three", context="other", prompt="three")).reply
        == "THREE"
    )
    assert launcher._workers[("route", "other")] is not first
    launcher.close()


def test_rpc_serializes_turns_and_cancels_blocking_ui(tmp_path):
    launcher = _rpc_launcher(tmp_path, "omp")
    result = []
    thread = threading.Thread(
        target=lambda: result.append(launcher.send(_request("ui-task", prompt="ui")))
    )
    thread.start()
    for _ in range(100):
        if launcher.cancel("ui-task"):
            break
        time.sleep(0.01)
    thread.join(2)
    assert not thread.is_alive()
    assert result and result[0].state == protocol.STATE_FAILED
    assert launcher.send(_request("recover", prompt="recover")).reply == "RECOVER"
    launcher.close()


def test_rpc_evicts_rejection_crash_timeout_and_idle_workers(tmp_path):
    launcher = _rpc_launcher(tmp_path, "omp", timeout=0.1, idle_timeout=0.01)
    for prompt in ("reject", "crash", "wait", "malformed", "oversize"):
        assert (
            launcher.send(_request(task_id=prompt, context=prompt, prompt=prompt)).state
            == protocol.STATE_FAILED
        )
        assert ("route", prompt) not in launcher._workers
    assert launcher.send(_request(context="idle", prompt="idle")).reply == "IDLE"
    launcher.reap_idle(time.monotonic() + 1)
    assert not launcher._workers
    launcher.close()


def test_rpc_close_terminates_persistent_worker(tmp_path):
    launcher = _rpc_launcher(tmp_path, "omp")
    assert launcher.send(_request(prompt="live")).state == protocol.STATE_COMPLETED
    worker = launcher._workers[("route", "context")]
    launcher.close()
    assert worker.process is None


def test_rpc_cancel_queued_same_context_never_emits_second_prompt(tmp_path):
    launcher = _rpc_launcher(tmp_path, "omp")
    first = threading.Thread(
        target=lambda: launcher.send(_request("first", context="shared", prompt="ui"))
    )
    first.start()
    for _ in range(100):
        if ("route", "shared") in launcher._workers:
            break
        time.sleep(0.01)
    queued = []
    second = threading.Thread(
        target=lambda: queued.append(
            launcher.send(_request("queued", context="shared", prompt="second"))
        )
    )
    second.start()
    assert launcher.cancel("queued") is True
    assert launcher.cancel("first") is True
    first.join(2)
    second.join(2)
    assert queued and queued[0].state == protocol.STATE_FAILED
    launcher.close()


def test_opaque_process_serializes_first_turn_before_resume(tmp_path):
    log = tmp_path / "calls"
    script = "import pathlib, sys, time; pathlib.Path(sys.argv[1]).open('a').write(sys.argv[2] + '\\n'); time.sleep(.1); print('answer'); print('session: opaque')"
    raw = {
        "transport": "process",
        "start": [sys.executable, "-c", script, str(log), "start", "{prompt}"],
        "resume": [
            sys.executable,
            "-c",
            script,
            str(log),
            "resume",
            "{prompt}",
            "{session_id}",
        ],
        "output": {
            "format": "text",
            "reply_from": "stdout",
            "session_id_from": "stdout",
            "session_id_regex": r"session:\s*(\S+)",
        },
    }
    launcher = ProcessLauncher(parse_launcher_spec(raw), str(tmp_path / "home"))
    threads = [
        threading.Thread(
            target=launcher.send, args=(_request(f"task-{i}", context="same"),)
        )
        for i in range(2)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(2)
    assert log.read_text(encoding="utf-8").splitlines() == ["start", "resume"]


def test_manager_close_refuses_future_work_and_cancels_registered_work(tmp_path):
    from plugins.platforms.a2a.launchers import LauncherManager

    manager = LauncherManager(str(tmp_path / "home"))
    manager._launchers["route"] = _process_launcher(
        tmp_path, "import time; time.sleep(10)", timeout=20
    )
    thread = threading.Thread(target=manager.send, args=(_request("closing"),))
    thread.start()
    for _ in range(100):
        if manager.active_task_ids():
            break
        time.sleep(0.01)
    manager.close()
    thread.join(3)
    assert not thread.is_alive()
    assert manager.active_task_ids() == ()
    assert manager.send(_request("after-close")).state == protocol.STATE_FAILED


def test_placeholder_typos_reject_while_literal_code_braces_remain_valid():
    for value in (
        "{session-id}",
        "{prompt.foo}",
        "{}",
        "{prompt!r}",
        "{prompt:>10}",
        "{prompt",
    ):
        try:
            parse_launcher_spec({"transport": "process", "start": ["tool", value]})
        except ValueError:
            pass
        else:
            raise AssertionError(f"malformed placeholder accepted: {value}")
    parse_launcher_spec({
        "transport": "process",
        "start": ["python", "-c", "print({'key': 1})"],
    })
