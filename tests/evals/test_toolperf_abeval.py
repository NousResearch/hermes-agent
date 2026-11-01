"""ToolPerf must never score missing, mixed, or incomplete trial evidence."""

import json
import os
import shutil
import subprocess
import tomllib

import pytest

from evals.toolperf_abeval import ab_eval


def trace_events(work, root="session-a", tool_count=1):
    def scope(name, category, scope_id, phase, data=None):
        return {
            "kind": "scope",
            "name": name,
            "category": category,
            "scope_category": phase,
            "uuid": scope_id,
            "propagation_root_uuid": root,
            "data": data or {},
        }

    events = [scope("hermes.session", "agent", root, "start", {"cwd": str(work)})]
    events += [scope("completion", "llm", "llm-1", phase) for phase in ("start", "end")]
    for i in range(tool_count):
        events += [
            scope("terminal", "tool", f"tool-{i}", phase, "ok")
            for phase in ("start", "end")
        ]
    events.append(scope("hermes.session", "agent", root, "end"))
    return events


def write_trace(path, events):
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )
    return path


@pytest.mark.parametrize("contents", [None, "", "not json\n", "[]\n"])
def test_unusable_trace_is_an_error(tmp_path, contents):
    path = tmp_path / "trace.jsonl"
    if contents is not None:
        path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match="trace"):
        ab_eval.score_run(path)


def test_scores_distinct_complete_trials_and_records_identity(tmp_path):
    scores = []
    for i, count in enumerate((1, 3)):
        work = tmp_path / f"trial-{i}"
        root = f"session-{i}"
        trace = write_trace(tmp_path / f"{i}.jsonl", trace_events(work, root, count))
        score = ab_eval.score_run(trace, work=work, session_uuid=root)
        assert score["session_uuid"] == root
        scores.append(score["tools"])
    assert scores == [1, 3]


@pytest.mark.parametrize(
    "damage",
    [
        "mixed",
        "unclosed_session",
        "unclosed_tool",
        "wrong_work",
        "wrong_session",
        "unclosed_turn",
        "missing_kind",
        "missing_uuid",
        "missing_tool_kind",
    ],
)
def test_rejects_incomplete_or_misattributed_trace(tmp_path, damage):
    events = trace_events(tmp_path)
    damage_trace = {
        "mixed": lambda: events.extend(trace_events(tmp_path, "session-b")),
        "unclosed_session": lambda: events.pop(),
        "unclosed_tool": lambda: events.pop(-2),
        "wrong_work": lambda: events[0]["data"].update(cwd=str(tmp_path / "other")),
        "wrong_session": lambda: None,
        "unclosed_turn": lambda: events.append(
            {**events[0], "name": "hermes.turn", "category": "function", "uuid": "turn"}
        ),
        "missing_kind": lambda: (events[0].pop("kind"), events.pop()),
        "missing_uuid": lambda: (events[1].pop("uuid"), events[2].pop("uuid")),
        "missing_tool_kind": lambda: (events[3].pop("kind"), events[4].pop("kind")),
    }
    damage_trace[damage]()
    path = write_trace(tmp_path / "trace.jsonl", events)
    with pytest.raises(ValueError, match="trace"):
        ab_eval.score_run(
            path,
            work=tmp_path,
            session_uuid="other" if damage == "wrong_session" else "session-a",
        )


def test_report_does_not_turn_missing_evidence_into_zero_cost(tmp_path, monkeypatch):
    monkeypatch.setattr(ab_eval, "ROOT", tmp_path)
    result_dir = tmp_path / "results" / "model" / "baseline"
    result_dir.mkdir(parents=True)
    meta = {
        "run_id": "err_case_search-r0",
        "task": "err_case_search",
        "wall_s": 1,
        "session_uuid": "session-a",
        "tail": "client.go settings.ini",
    }
    (result_dir / "meta.jsonl").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="trace"):
        ab_eval.report(["model"])


@pytest.fixture
def fake_cli(tmp_path):
    """A real child process whose exporter, like Relay, is fixed at gateway boot."""
    tree = tmp_path / "tree"
    package = tree / "hermes_cli"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (package / "main.py").write_text(
        """
import json, os, signal, sys, tomllib, uuid
from pathlib import Path
home = Path(os.environ["HERMES_HOME"])
state = home / "gateway_state.json"
if "gateway" in sys.argv:
    lock = home / "stub-owner"
    try:
        lock.touch(exist_ok=False)
    except FileExistsError:
        sys.exit(0)
    config = tomllib.loads(Path(os.environ["HERMES_NEMO_RELAY_PLUGINS_TOML"]).read_text())
    component = config["components"][0]["config"]
    if component["version"] != 4:
        sys.exit(2)
    sink = component["atof"]["sinks"][0]
    trace = Path(sink["output_directory"]) / sink["filename"]
    with (home / "starts.jsonl").open("a") as output:
        output.write(json.dumps({"pid": os.getpid(), "tree": str(Path(__file__).parent.parent)}) + "\\n")
    def stop(*args):
        pending = home / "pending.json"
        if pending.exists():
            with trace.open("a") as output:
                output.write(pending.read_text())
            pending.unlink()
        lock.unlink()
        sys.exit(0)
    signal.signal(signal.SIGTERM, stop)
    state.write_text(json.dumps({"pid": os.getpid(), "gateway_state": "running", "trace": str(trace), "tree": str(Path(__file__).parent.parent)}))
    while True:
        signal.pause()
else:
    info = json.loads(state.read_text())
    root = str(uuid.uuid4())
    def event(name, category, uid, phase, data=None):
        return json.dumps({"kind": "scope", "name": name, "category": category,
            "uuid": uid, "propagation_root_uuid": root, "scope_category": phase,
            "data": data or {}}) + "\\n"
    events = event("hermes.session", "agent", root, "start", {"cwd": os.getcwd(), "gateway_pid": info["pid"], "tree": info["tree"]})
    for phase in ("start", "end"):
        events += event("completion", "llm", "llm", phase)
    events += event("hermes.session", "agent", root, "end")
    with Path(info["trace"]).open("a") as output:
        output.write(events)
    # Buffered exporter events become visible only after graceful shutdown.
    with (home / "pending.json").open("a") as output:
        for phase in ("start", "end"):
            output.write(event("auxiliary", "llm", "aux", phase))
    print("client.go settings.ini")
""",
        encoding="utf-8",
    )
    gateway = tree / "gateway"
    gateway.mkdir()
    (gateway / "__init__.py").touch()
    (gateway / "status.py").write_text(
        "import os, signal\ndef write_planned_stop_marker(pid):\n"
        "    os.kill(pid, signal.SIGTERM)\n    return True\n",
        encoding="utf-8",
    )
    return tree


@pytest.mark.spawns_gateway_lookalike
@pytest.mark.platforms("posix")  # the fixture gateway uses POSIX signals
def test_batch_reuses_gateway_switches_code_arms_and_flushes_before_recording(
    tmp_path, monkeypatch, fake_cli
):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(ab_eval, "ROOT", tmp_path)
    monkeypatch.setattr(ab_eval, "HOME", home)
    monkeypatch.setattr(ab_eval, "make_sandbox", lambda work: None)
    ab_eval.run("baseline", "model", 2, str(fake_cli), ["err_case_search"])
    resdir = tmp_path / "results" / "model" / "baseline"
    records = [
        json.loads(line) for line in (resdir / "meta.jsonl").read_text().splitlines()
    ]
    assert len(records) == 2
    assert len({m["session_uuid"] for m in records}) == 2
    starts = [
        json.loads(line) for line in (home / "starts.jsonl").read_text().splitlines()
    ]
    assert len(starts) == 1
    for record in records:
        path = resdir / f"{record['run_id']}.atof.jsonl"
        score = ab_eval.score_run(path)
        assert score["llm"] == 2
        session = json.loads(path.read_text().splitlines()[0])
        assert session["data"]["gateway_pid"] == starts[0]["pid"]
        assert session["data"]["tree"] == str(fake_cli)
    ab_eval.run("baseline", "model", 2, str(fake_cli), ["err_case_search"])
    assert len((resdir / "meta.jsonl").read_text().splitlines()) == 2
    assert len((home / "starts.jsonl").read_text().splitlines()) == 1
    fixes_tree = tmp_path / "fixes-tree"
    shutil.copytree(fake_cli, fixes_tree)
    ab_eval.run("fixes", "model", 1, str(fixes_tree), ["err_case_search"])
    starts = [
        json.loads(line) for line in (home / "starts.jsonl").read_text().splitlines()
    ]
    assert len(starts) == 2
    assert starts[1]["tree"] == str(fixes_tree)
    fixed_trace = tmp_path / "results/model/fixes/err_case_search-r0.atof.jsonl"
    session = json.loads(fixed_trace.read_text().splitlines()[0])
    assert session["data"]["gateway_pid"] == starts[1]["pid"]
    assert session["data"]["tree"] == str(fixes_tree)
    (resdir / f"{records[0]['run_id']}.atof.jsonl").unlink()
    with pytest.raises(ValueError, match="trace"):
        ab_eval.run("baseline", "model", 2, str(fake_cli), ["err_case_search"])


def test_shared_trace_selects_one_trial_even_when_events_interleave(tmp_path):
    first = trace_events(tmp_path / "first", "first", tool_count=1)
    second = trace_events(tmp_path / "second", "second", tool_count=3)
    interleaved = [event for pair in zip(first, second) for event in pair]
    interleaved.extend(second[len(first) :])
    path = write_trace(tmp_path / "batch.jsonl", interleaved)
    for work, expected in (("first", 1), ("second", 3)):
        events, root = ab_eval.read_trace(path, tmp_path / work, None, shared=True)
        split = write_trace(tmp_path / f"{work}.jsonl", events)
        score = ab_eval.score_run(split, work=tmp_path / work, session_uuid=root)
        assert score["tools"] == expected
        assert root == work


@pytest.mark.parametrize(
    "damage", ["missing", "ambiguous", "unassociated", "incomplete"]
)
def test_shared_trace_requires_unambiguous_complete_trial(tmp_path, damage):
    work = tmp_path / "trial"
    events = trace_events(work, "first")
    if damage == "missing":
        work = tmp_path / "missing"
    elif damage == "ambiguous":
        events.extend(trace_events(work, "second"))
    elif damage == "unassociated":
        events[3]["propagation_root_uuid"] = "unknown"
        events[4]["propagation_root_uuid"] = "unknown"
    else:
        events.pop(-2)
    path = write_trace(tmp_path / "batch.jsonl", events)
    with pytest.raises(ValueError, match="trace"):
        ab_eval.read_trace(path, work, None, shared=True)


@pytest.mark.spawns_gateway_lookalike
@pytest.mark.platforms("posix")
def test_failed_chat_keeps_only_completed_trials_after_batch_flush(
    tmp_path, monkeypatch, fake_cli
):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(ab_eval, "ROOT", tmp_path)
    monkeypatch.setattr(ab_eval, "HOME", home)
    monkeypatch.setattr(ab_eval, "make_sandbox", lambda work: None)
    real_run = subprocess.run

    def fail_second_chat(command, **kwargs):
        if "chat" in command and kwargs["cwd"].name.endswith("-r1"):
            return subprocess.CompletedProcess(command, 1, "partial answer", "failed")
        return real_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", fail_second_chat)
    with pytest.raises(RuntimeError, match="Chat failed"):
        ab_eval.run("baseline", "model", 2, str(fake_cli), ["err_case_search"])
    result = tmp_path / "results/model/baseline"
    records = [
        json.loads(line) for line in (result / "meta.jsonl").read_text().splitlines()
    ]
    assert [record["run_id"] for record in records] == ["err_case_search-r0"]
    assert len((home / "starts.jsonl").read_text().splitlines()) == 1
    assert not (home / "stub-owner").exists()
    assert ab_eval.score_run(result / "err_case_search-r0.atof.jsonl")["llm"] == 2


@pytest.mark.spawns_gateway_lookalike
@pytest.mark.platforms("posix")  # the fixture gateway uses POSIX signals
@pytest.mark.parametrize("fail_chat", [False, True])
def test_cleanup_owns_only_its_child_and_failed_trials_are_not_recorded(
    tmp_path, monkeypatch, fake_cli, fail_chat
):
    home = tmp_path / "home"
    home.mkdir()
    work = tmp_path / "owned"
    work.mkdir()
    config = work / "relay.toml"
    ab_eval.write_relay_config(config, work / "trace.jsonl")
    env = {
        **os.environ,
        "PYTHONPATH": str(fake_cli),
        "HERMES_HOME": str(home),
        "HERMES_NEMO_RELAY_PLUGINS_TOML": str(config),
    }
    if fail_chat:
        monkeypatch.setattr(ab_eval, "ROOT", tmp_path)
        monkeypatch.setattr(ab_eval, "HOME", home)
        monkeypatch.setattr(ab_eval, "make_sandbox", lambda work: None)
        real_run = subprocess.run

        def timeout_chat(command, **kwargs):
            if "chat" in command:
                raise subprocess.TimeoutExpired(command, 600)
            return real_run(command, **kwargs)

        monkeypatch.setattr(subprocess, "run", timeout_chat)
        with pytest.raises(subprocess.TimeoutExpired):
            ab_eval.run("baseline", "model", 1, str(fake_cli), ["err_case_search"])
        assert not (tmp_path / "results/model/baseline/meta.jsonl").exists()
    else:
        with ab_eval.batch_gateway(env, work):
            owner = json.loads((home / "gateway_state.json").read_text())["pid"]
            with pytest.raises(RuntimeError, match="dedicated, idle home"):
                with ab_eval.batch_gateway(env, tmp_path):
                    pytest.fail("must not attach to an existing gateway")
            os.kill(
                owner, 0
            )  # the rejected contender did not terminate the existing owner
    assert not (home / "stub-owner").exists()


def test_generated_config_is_accepted_by_relay(tmp_path):
    plugin = pytest.importorskip("nemo_relay.plugin")
    config = tmp_path / "relay.toml"
    ab_eval.write_relay_config(config, tmp_path / "trace.jsonl")
    report = plugin.validate_exact(tomllib.loads(config.read_text()))
    assert not [
        item for item in report["config"]["diagnostics"] if item["level"] == "error"
    ]


def test_waits_for_pending_trace_writes_and_bounds_missing_evidence(tmp_path):
    trace = tmp_path / "trace.jsonl"

    class Gateway:
        def wait(self, timeout):
            write_trace(trace, trace_events(tmp_path))
            raise subprocess.TimeoutExpired("fixture-gateway", timeout)

    ab_eval.wait_for_trace(trace, tmp_path, Gateway())
    trace.unlink()
    with pytest.raises(ValueError, match="trace"):
        ab_eval.wait_for_trace(trace, tmp_path, Gateway(), timeout=0)


@pytest.mark.platforms("windows")
def test_windows_launcher_is_refused_before_spawning(tmp_path):
    with pytest.raises(RuntimeError, match="ownership currently requires"):
        with ab_eval.batch_gateway({}, tmp_path):
            pytest.fail("unsupported launcher must not be started")
