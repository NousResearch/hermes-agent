"""ToolPerf must never score missing, mixed, or incomplete trial evidence."""

import json
import os
import shutil
import subprocess
import sys
import tomllib
import venv
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import psutil
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
import json, os, sys, tomllib, uuid
from pathlib import Path
from threading import Event
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
    def stop():
        pending = home / "pending.json"
        if pending.exists():
            with trace.open("a") as output:
                output.write(pending.read_text())
            pending.unlink()
        lock.unlink()
        sys.exit(1 if (home / "fail-stop").exists() else 0)
    state.write_text(json.dumps({"pid": os.getpid(), "gateway_state": "running", "trace": str(trace), "tree": str(Path(__file__).parent.parent)}))
    stop_file = home / "stub-stop.json"
    while True:
        if stop_file.exists():
            try:
                target = json.loads(stop_file.read_text())["pid"]
            except (OSError, ValueError):
                target = None
            if target == os.getpid():
                stop_file.unlink()
                stop()
        Event().wait(0.05)
else:
    behavior_file = home / "chat-behavior.json"
    behaviors = json.loads(behavior_file.read_text()) if behavior_file.exists() else {}
    behavior = behaviors.get("/".join(Path.cwd().parts[-3:]), behaviors.get(Path.cwd().name, {}))
    if behavior.get("startup_failure"):
        print("fixture startup failure", file=sys.stderr)
        sys.exit(1)
    info = json.loads(state.read_text())
    root = str(uuid.uuid4())
    def event(name, category, uid, phase, data=None):
        return json.dumps({"kind": "scope", "name": name, "category": category,
            "uuid": uid, "propagation_root_uuid": root, "scope_category": phase,
            "data": data or {}}) + "\\n"
    events = event("hermes.session", "agent", root, "start", {"cwd": os.getcwd(), "gateway_pid": info["pid"], "tree": info["tree"]})
    for phase in ("start", "end"):
        events += event("completion", "llm", "llm", phase)
    end = event("hermes.session", "agent", root, "end")
    if behavior.get("trace") not in ("incomplete", "flush_only"):
        events += end
    with Path(info["trace"]).open("a") as output:
        output.write(events)
    # Buffered exporter events become visible only after graceful shutdown.
    with (home / "pending.json").open("a") as output:
        if behavior.get("trace") == "flush_only":
            output.write(end)
        for phase in ("start", "end"):
            output.write(event("auxiliary", "llm", "aux", phase))
    print(behavior.get("stdout", "client.go settings.ini"))
    sys.exit(behavior.get("exit", 0))
""",
        encoding="utf-8",
    )
    gateway = tree / "gateway"
    gateway.mkdir()
    (gateway / "__init__.py").touch()
    (gateway / "status.py").write_text(
        "import json, os\nfrom pathlib import Path\n"
        "def write_planned_stop_marker(pid):\n"
        "    (Path(os.environ['HERMES_HOME']) / 'stub-stop.json').write_text(json.dumps({'pid': pid}))\n"
        "    return True\n",
        encoding="utf-8",
    )
    return tree


@pytest.mark.spawns_gateway_lookalike
@pytest.mark.platforms("linux", "macos", "windows")
@pytest.mark.parametrize("python_kind", ["current", "venv"])
def test_batch_reuses_gateway_switches_code_arms_and_flushes_before_recording(
    tmp_path, monkeypatch, fake_cli, python_kind
):
    if python_kind == "venv":
        venv.EnvBuilder(with_pip=False).create(tmp_path / "venv")
        python = (
            tmp_path
            / "venv"
            / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        )
        monkeypatch.setattr(sys, "executable", str(python))
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
@pytest.mark.platforms("linux", "macos", "windows")
@pytest.mark.parametrize("exit_code", [1, 130, 0xC000013A])
def test_failed_chat_keeps_only_completed_trials_after_batch_flush(
    tmp_path, monkeypatch, fake_cli, exit_code
):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(ab_eval, "ROOT", tmp_path)
    monkeypatch.setattr(ab_eval, "HOME", home)
    monkeypatch.setattr(ab_eval, "make_sandbox", lambda work: None)
    real_wait = ab_eval.wait_for_trace
    monkeypatch.setattr(
        ab_eval,
        "wait_for_trace",
        lambda path, work, owner, timeout=60: real_wait(path, work, owner, timeout=0),
    )
    real_run = subprocess.run

    def fail_second_chat(command, **kwargs):
        if "chat" in command and kwargs["cwd"].name.endswith("-r1"):
            return subprocess.CompletedProcess(
                command, exit_code, "partial answer", "failed"
            )
        return real_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", fail_second_chat)
    error_type = (
        ab_eval.IncompleteBatchError
        if exit_code == 1
        else subprocess.CalledProcessError
    )
    with pytest.raises(error_type):
        ab_eval.run("baseline", "model", 3, str(fake_cli), ["err_case_search"])
    result = tmp_path / "results/model/baseline"
    records = [
        json.loads(line) for line in (result / "meta.jsonl").read_text().splitlines()
    ]
    expected = (
        ["err_case_search-r0", "err_case_search-r2"]
        if exit_code == 1
        else ["err_case_search-r0"]
    )
    assert [record["run_id"] for record in records] == expected
    assert len((home / "starts.jsonl").read_text().splitlines()) == 1
    assert not (home / "stub-owner").exists()
    assert ab_eval.score_run(result / "err_case_search-r0.atof.jsonl")["llm"] == 2


@pytest.mark.spawns_gateway_lookalike
@pytest.mark.platforms("linux", "macos", "windows")
def test_failed_task_with_valid_trace_is_measured_and_not_retried(
    tmp_path, monkeypatch, fake_cli, capsys
):
    home = tmp_path / "home"
    home.mkdir()
    (home / "chat-behavior.json").write_text(
        json.dumps(
            {"err_case_search-r1": {"exit": 1, "stdout": "Reached the turn limit"}}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ab_eval, "ROOT", tmp_path)
    monkeypatch.setattr(ab_eval, "HOME", home)
    monkeypatch.setattr(ab_eval, "make_sandbox", lambda work: None)
    ab_eval.run("baseline", "model", 2, str(fake_cli), ["err_case_search"])
    resdir = tmp_path / "results/model/baseline"
    records = [
        json.loads(line) for line in (resdir / "meta.jsonl").read_text().splitlines()
    ]
    assert [record["exit"] for record in records] == [0, 1]
    assert ab_eval.score_run(resdir / "err_case_search-r1.atof.jsonl")["llm"] == 2
    capsys.readouterr()
    ab_eval.report(["model"])
    rows = capsys.readouterr().out
    assert "| baseline |  2  50%" in rows
    ab_eval.run("baseline", "model", 2, str(fake_cli), ["err_case_search"])
    assert len((home / "starts.jsonl").read_text().splitlines()) == 1


@pytest.mark.spawns_gateway_lookalike
@pytest.mark.platforms("linux", "macos", "windows")
@pytest.mark.parametrize("trace_state", ["incomplete", "flush_only"])
def test_trace_failure_keeps_other_trials_and_final_flush_decides_validity(
    tmp_path, monkeypatch, fake_cli, trace_state
):
    home = tmp_path / "home"
    home.mkdir()
    (home / "chat-behavior.json").write_text(
        json.dumps({"err_case_search-r1": {"trace": trace_state}}), encoding="utf-8"
    )
    monkeypatch.setattr(ab_eval, "ROOT", tmp_path)
    monkeypatch.setattr(ab_eval, "HOME", home)
    monkeypatch.setattr(ab_eval, "make_sandbox", lambda work: None)
    real_wait = ab_eval.wait_for_trace
    monkeypatch.setattr(
        ab_eval,
        "wait_for_trace",
        lambda path, work, owner, timeout=60: real_wait(path, work, owner, timeout=0),
    )
    if trace_state == "incomplete":
        with pytest.raises(ab_eval.IncompleteBatchError, match="incomplete"):
            ab_eval.run("baseline", "model", 3, str(fake_cli), ["err_case_search"])
    else:
        ab_eval.run("baseline", "model", 3, str(fake_cli), ["err_case_search"])
    resdir = tmp_path / "results/model/baseline"
    records = [
        json.loads(line) for line in (resdir / "meta.jsonl").read_text().splitlines()
    ]
    expected = [0, 2] if trace_state == "incomplete" else [0, 1, 2]
    assert [record["rep"] for record in records] == expected
    assert not (home / "stub-owner").exists()
    for record in records:
        assert ab_eval.score_run(resdir / f"{record['run_id']}.atof.jsonl")["llm"] == 2
    if trace_state == "incomplete":
        (home / "chat-behavior.json").unlink()
        ab_eval.run("baseline", "model", 3, str(fake_cli), ["err_case_search"])
        resumed = [
            json.loads(line)
            for line in (resdir / "meta.jsonl").read_text().splitlines()
        ]
        assert [record["rep"] for record in resumed] == [0, 2, 1]
        assert resumed[:2] == records
        assert len((home / "starts.jsonl").read_text().splitlines()) == 2


@pytest.mark.spawns_gateway_lookalike
@pytest.mark.platforms("linux", "macos")
@pytest.mark.parametrize("failure", ["startup", "shutdown", None])
def test_run_all_continues_only_after_clean_recoverable_batch_failure(
    tmp_path, fake_cli, failure
):
    home = tmp_path / "home"
    home.mkdir()
    if failure == "startup":
        (home / "chat-behavior.json").write_text(
            json.dumps(
                {"model-a/baseline/err_case_search-r0": {"startup_failure": True}}
            ),
            encoding="utf-8",
        )
    elif failure == "shutdown":
        (home / "fail-stop").touch()
    # Run the real driver/CLI entry point, limiting the battery to one existing task.
    python = tmp_path / "eval-python"
    python.write_text(
        f"#!{sys.executable}\nimport os, sys\n"
        "args = sys.argv[1:]\n"
        "if args[1] == 'run': args += ['--only', 'err_case_search']\n"
        "os.execv(sys.executable, [sys.executable, *args])\n",
        encoding="utf-8",
    )
    python.chmod(0o700)
    result = subprocess.run(
        [
            "bash",
            str(Path(ab_eval.__file__).with_name("run_all.sh")),
            str(fake_cli),
            str(fake_cli),
            "1",
            "model-a",
            "model-b",
        ],
        env={
            **os.environ,
            "PYTHON": str(python),
            "ABEVAL_ROOT": str(tmp_path),
            "ABEVAL_HOME": str(home),
        },
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    starts = (home / "starts.jsonl").read_text().splitlines()
    if failure == "shutdown":
        assert result.returncode not in (0, 3)
        assert len(starts) == 1
        assert "MODEL:" not in result.stdout
        assert not list((tmp_path / "results").glob("*/*/meta.jsonl"))
    else:
        assert result.returncode == (3 if failure else 0), result.stderr
        assert len(starts) == 4
        assert "MODEL: model-a" in result.stdout and "MODEL: model-b" in result.stdout
        assert len(list((tmp_path / "results").glob("*/*/meta.jsonl"))) == (
            3 if failure else 4
        )
    assert not (home / "stub-owner").exists()


@pytest.mark.spawns_gateway_lookalike
@pytest.mark.platforms("linux", "macos", "windows")
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
        attempted = []

        def timeout_chat(command, **kwargs):
            if "chat" in command:
                attempted.append(kwargs["cwd"])
                raise subprocess.TimeoutExpired(command, 600)
            return real_run(command, **kwargs)

        monkeypatch.setattr(subprocess, "run", timeout_chat)
        with pytest.raises(subprocess.TimeoutExpired):
            ab_eval.run("baseline", "model", 3, str(fake_cli), ["err_case_search"])
        assert len(attempted) == 1
        assert not (tmp_path / "results/model/baseline/meta.jsonl").exists()
    else:
        with ab_eval.batch_gateway(env, work):
            owner = json.loads((home / "gateway_state.json").read_text())["pid"]
            with pytest.raises(RuntimeError, match="dedicated, idle home"):
                with ab_eval.batch_gateway(env, tmp_path):
                    pytest.fail("must not attach to an existing gateway")
            assert psutil.Process(owner).is_running()
    assert not (home / "stub-owner").exists()


def test_generated_config_is_accepted_by_relay(tmp_path):
    plugin = pytest.importorskip("nemo_relay.plugin")
    config = tmp_path / "relay.toml"
    ab_eval.write_relay_config(config, tmp_path / "trace.jsonl")
    report = plugin.validate_exact(tomllib.loads(config.read_text()))
    assert not [
        item for item in report["config"]["diagnostics"] if item["level"] == "error"
    ]


@pytest.mark.parametrize("flush_last_trial", [False, True])
def test_missing_traces_share_one_batch_wait_budget(
    tmp_path, monkeypatch, flush_last_trial
):
    clock = SimpleNamespace(elapsed=0.0, stopped=False)
    last_work = tmp_path / "runs/model/baseline/err_case_search-r2"

    class Gateway:
        def poll(self):
            return None

        def wait(self, timeout):
            clock.elapsed += timeout
            raise subprocess.TimeoutExpired("fixture-gateway", timeout)

    @contextmanager
    def gateway(env, work):
        yield Gateway()
        clock.stopped = True
        if flush_last_trial:
            write_trace(work / "trace.atof.jsonl", trace_events(last_work))

    monkeypatch.setattr(ab_eval, "ROOT", tmp_path)
    monkeypatch.setattr(ab_eval, "HOME", tmp_path / "home")
    monkeypatch.setattr(ab_eval, "batch_gateway", gateway)
    monkeypatch.setattr(ab_eval, "make_sandbox", lambda work: None)
    monkeypatch.setattr(ab_eval, "run_trial", lambda *args: ("answer", 0, 0))
    monkeypatch.setattr(
        ab_eval, "time", SimpleNamespace(monotonic=lambda: clock.elapsed)
    )
    with pytest.raises(ab_eval.IncompleteBatchError):
        ab_eval.run("baseline", "model", 3, str(tmp_path), ["err_case_search"])

    assert clock.stopped
    meta = tmp_path / "results/model/baseline/meta.jsonl"
    if flush_last_trial:
        records = [json.loads(line) for line in meta.read_text().splitlines()]
        assert [record["run_id"] for record in records] == [last_work.name]
        assert (
            ab_eval.score_run(meta.parent / f"{last_work.name}.atof.jsonl")["llm"] == 1
        )
    else:
        assert not meta.exists()
    # Allow one polling interval of overshoot, independent of the trial count.
    assert 60 <= clock.elapsed <= 60.2


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


@pytest.mark.spawns_gateway_lookalike
@pytest.mark.platforms("linux", "macos", "windows")
def test_gateway_worker_can_be_a_launcher_descendant(tmp_path):
    # Exercise a real launcher/worker chain on every host, including Windows venv shims.
    worker = (
        "import json, os, sys; from pathlib import Path; from threading import Event; "
        "Path(sys.argv[1]).write_text(json.dumps({'pid': os.getpid(), 'gateway_state': 'running'})); "
        "Event().wait(30)"
    )
    launcher = "import subprocess, sys; sys.exit(subprocess.call([sys.executable, '-c', *sys.argv[1:]]))"
    process = subprocess.Popen(
        [sys.executable, "-c", launcher, worker, str(tmp_path / "gateway_state.json")]
    )
    try:
        owner = ab_eval.wait_for_gateway(process, tmp_path, timeout=3)
        state = json.loads((tmp_path / "gateway_state.json").read_text())
        assert owner.pid == state["pid"]
        assert owner.pid != process.pid
    finally:
        children = psutil.Process(process.pid).children(recursive=True)
        for child in reversed(children):
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        process.kill()
        process.wait(timeout=10)
        psutil.wait_procs(children, timeout=10)


@pytest.mark.spawns_gateway_lookalike
@pytest.mark.platforms("linux", "macos", "windows")
@pytest.mark.parametrize("named", [False, True])
def test_batch_and_chat_preserve_selected_home_with_real_cli_bootstrap(
    tmp_path, monkeypatch, fake_cli, named
):
    root = tmp_path / "home"
    root.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in ("eval", "other"):
        profile = root / "profiles" / name
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text("model: test\n", encoding="utf-8")
    (root / "active_profile").write_text("other\n", encoding="utf-8")
    home = root / "profiles" / "eval" if named else root
    monkeypatch.setattr(ab_eval, "ROOT", tmp_path)
    monkeypatch.setattr(ab_eval, "HOME", home)
    monkeypatch.setattr(ab_eval, "make_sandbox", lambda work: None)
    real_popen = subprocess.Popen
    checked = []

    def check_bootstrap(command, **kwargs):
        if command[1:3] == ["-m", "hermes_cli.main"]:
            probe = (
                "import os, sys; sys.argv = ['hermes', *sys.argv[1:]]; "
                "import hermes_cli.main; print('SELECTED_HOME=' + os.environ['HERMES_HOME'])"
            )
            env = {
                **kwargs["env"],
                "PYTHONPATH": str(Path(ab_eval.__file__).parents[2]),
            }
            with real_popen(
                [sys.executable, "-c", probe, *command[3:]],
                env=env,
                cwd=tmp_path,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            ) as child:
                stdout, stderr = child.communicate(timeout=30)
                assert child.returncode == 0, stderr
                assert f"SELECTED_HOME={home}" in stdout.splitlines()
            checked.append("gateway" if "gateway" in command else "chat")
        return real_popen(command, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", check_bootstrap)
    ab_eval.run("baseline", "model", 1, str(fake_cli), ["err_case_search"])
    assert checked == ["gateway", "chat"]
    assert (home / "starts.jsonl").exists()
    assert not (root / "profiles" / "other" / "gateway_state.json").exists()


@pytest.mark.spawns_gateway_lookalike
@pytest.mark.platforms("linux", "macos", "windows")
@pytest.mark.parametrize("startup_failure", [False, True])
def test_batch_stops_owned_worker_behind_launcher(
    tmp_path, monkeypatch, fake_cli, startup_failure
):
    home = tmp_path / "home"
    home.mkdir()
    work = tmp_path / "batch"
    work.mkdir()
    config = work / "relay.toml"
    ab_eval.write_relay_config(config, work / "trace.jsonl")
    env = {
        **os.environ,
        "PYTHONPATH": str(fake_cli),
        "HERMES_HOME": str(home),
        "HERMES_NEMO_RELAY_PLUGINS_TOML": str(config),
    }
    real_popen, real_wait = subprocess.Popen, ab_eval.wait_for_gateway
    workers = []

    def launch(command, **kwargs):
        if "gateway" in command:
            command = [
                sys.executable,
                "-c",
                "import subprocess, sys; sys.exit(subprocess.call(sys.argv[1:]))",
                *command,
            ]
        return real_popen(command, **kwargs)

    def wait(process, home):
        owner = real_wait(process, home, timeout=3)
        workers.extend(psutil.Process(process.pid).children(recursive=True))
        if startup_failure:
            raise RuntimeError("fixture startup failure")
        return owner

    monkeypatch.setattr(subprocess, "Popen", launch)
    monkeypatch.setattr(ab_eval, "wait_for_gateway", wait)
    try:
        if startup_failure:
            with pytest.raises(RuntimeError, match="fixture startup failure"):
                with ab_eval.batch_gateway(env, work):
                    pytest.fail("startup must fail")
        else:
            with ab_eval.batch_gateway(env, work):
                assert workers
            assert not (home / "stub-owner").exists()
        assert not psutil.wait_procs(workers, timeout=3)[1]
    finally:
        for worker in workers:
            if worker.is_running():
                worker.kill()
