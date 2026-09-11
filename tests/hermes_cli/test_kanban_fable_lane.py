"""Tests for the selective Fable worker lane (``hermes_cli.kanban_fable``).

Covers the three properties the lane exists for:

1. **Selective / default-disabled** — a profile only takes the launcher path
   when its own config opts in AND the assignee matches the lane name. Every
   other task keeps the normal ``hermes -p <assignee>`` spawn.
2. **Fail closed** — missing launcher, missing ``claude``, nonzero exit,
   timeout, disabled launcher, or output that doesn't prove the Fable model
   ran all raise / block. Nothing ever falls back to another model.
3. **Lifecycle preserved** — the supervisor completes or blocks the task, so a
   lane run leaves the board in the same shapes a normal worker would.

Everything here is offline: the "launcher" is a stub script that prints a
canned JSON payload, and ``claude`` resolution is stubbed. The one test that
would touch the real launcher/model is skipped unless
``HERMES_FABLE_CANARY=1`` is set, and even then it is read-only.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_fable as kf


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


def _make_launcher(tmp_path: Path, payload: dict, *, exit_code: int = 0) -> Path:
    """Write a stub launcher that echoes ``payload`` and exits ``exit_code``."""
    path = tmp_path / "stub_launcher.py"
    path.write_text(
        "import json, sys\n"
        f"print(json.dumps({payload!r}))\n"
        f"raise SystemExit({exit_code})\n",
        encoding="utf-8",
    )
    return path


def _write_lane_config(home: Path, profile: str, **overrides) -> Path:
    """Enable ``kanban.fable_lane`` in ``profiles/<profile>/config.yaml``."""
    profile_dir = home / "profiles" / profile
    profile_dir.mkdir(parents=True, exist_ok=True)
    settings = {"enabled": True, **overrides}
    lines = ["kanban:", "  fable_lane:"]
    for key, value in settings.items():
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        elif isinstance(value, int):
            rendered = str(value)
        else:
            rendered = json.dumps(str(value))
        lines.append(f"    {key}: {rendered}")
    profile_dir.joinpath("config.yaml").write_text(
        "\n".join(lines) + "\n", encoding="utf-8",
    )
    return profile_dir


def _ok_payload(answer: str = "done") -> dict:
    return {
        "model_requested": kf.FABLE_MODEL,
        "oauth_only": True,
        "usage_credits_enabled": False,
        "checks": [
            {
                "name": "read_only_task",
                "exit_code": 0,
                "elapsed_ms": 12,
                "stderr": "",
                "result": {"result": answer, "modelUsage": {kf.FABLE_MODEL: {}}},
            }
        ],
    }


def _stub_claude(monkeypatch, path: str | None = "/usr/local/bin/claude"):
    monkeypatch.setattr(kf.shutil, "which", lambda *a, **k: path)


def _cfg(launcher: Path, **overrides) -> kf.FableLaneConfig:
    return kf.FableLaneConfig(
        assignee=kf.DEFAULT_LANE_ASSIGNEE, launcher=launcher, **overrides,
    )


# ---------------------------------------------------------------------------
# 1. Selective opt-in
# ---------------------------------------------------------------------------

def test_lane_disabled_by_default(kanban_home):
    """A `fable` profile with no lane config keeps the normal worker path."""
    (kanban_home / "profiles" / "fable").mkdir(parents=True)
    assert kf.load_lane_config("fable") is None
    assert kf.lane_for_assignee("fable") is None


def test_lane_engages_only_for_the_configured_assignee(kanban_home, tmp_path):
    launcher = _make_launcher(tmp_path, _ok_payload())
    _write_lane_config(kanban_home, "fable", launcher=str(launcher))
    (kanban_home / "profiles" / "elias").mkdir(parents=True)

    assert kf.lane_for_assignee("fable") is not None
    assert kf.lane_for_assignee("Fable") is not None  # display-cased assignee
    assert kf.lane_for_assignee("elias") is None


def test_lane_ignores_config_whose_lane_name_differs(kanban_home, tmp_path):
    """`assignee:` names the lane; a profile that isn't it stays on the normal path."""
    launcher = _make_launcher(tmp_path, _ok_payload())
    _write_lane_config(
        kanban_home, "elias", launcher=str(launcher), assignee="fable",
    )
    assert kf.lane_for_assignee("elias") is None


def test_default_spawn_routes_fable_task_to_the_launcher(
    kanban_home, tmp_path, monkeypatch,
):
    launcher = _make_launcher(tmp_path, _ok_payload())
    _write_lane_config(kanban_home, "fable", launcher=str(launcher))
    _stub_claude(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-should-not-be-copied")

    captured = {}

    class _FakeProc:
        pid = 5150

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        return _FakeProc()

    monkeypatch.setattr("subprocess.Popen", _fake_popen)

    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="read the repo", assignee="fable", classification="deep",
        )
        task = kb.get_task(conn, tid)

    pid = kb._default_spawn(task, str(kanban_home))

    assert pid == 5150
    assert captured["cmd"][:4] == [
        sys.executable, "-m", "hermes_cli.kanban_fable", "run",
    ]
    assert "--task" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--task") + 1] == tid
    # Kanban lifecycle pins survive the swap.
    assert captured["env"]["HERMES_KANBAN_TASK"] == tid
    assert captured["env"]["HERMES_KANBAN_DB"]
    assert captured["env"]["HERMES_KANBAN_BOARD"]
    assert captured["env"]["HERMES_PROFILE"] == "fable"
    # No credential copying into the OAuth-only child.
    assert "ANTHROPIC_API_KEY" not in captured["env"]


def test_default_spawn_leaves_other_profiles_alone(
    kanban_home, tmp_path, monkeypatch,
):
    launcher = _make_launcher(tmp_path, _ok_payload())
    _write_lane_config(kanban_home, "fable", launcher=str(launcher))
    (kanban_home / "profiles" / "elias").mkdir(parents=True)

    captured = {}

    class _FakeProc:
        pid = 4242

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return _FakeProc()

    monkeypatch.setattr("subprocess.Popen", _fake_popen)

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="normal work", assignee="elias")
        task = kb.get_task(conn, tid)

    kb._default_spawn(task, str(kanban_home))

    assert "hermes_cli.kanban_fable" not in captured["cmd"]
    assert "-p" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("-p") + 1] == "elias"


# ---------------------------------------------------------------------------
# 2. Fail closed
# ---------------------------------------------------------------------------

def test_preflight_rejects_missing_launcher(tmp_path, monkeypatch):
    _stub_claude(monkeypatch)
    with pytest.raises(kf.FableLaneUnavailable, match="launcher not found"):
        kf.preflight(_cfg(tmp_path / "nope.py"))


def test_preflight_rejects_unset_launcher(monkeypatch):
    _stub_claude(monkeypatch)
    with pytest.raises(kf.FableLaneUnavailable, match="launcher is unset"):
        kf.preflight(kf.FableLaneConfig(assignee="fable", launcher=None))


def test_preflight_rejects_missing_claude(tmp_path, monkeypatch):
    launcher = _make_launcher(tmp_path, _ok_payload())
    _stub_claude(monkeypatch, None)
    with pytest.raises(kf.FableLaneUnavailable, match="claude"):
        kf.preflight(_cfg(launcher))


def test_preflight_rejects_a_different_model(tmp_path, monkeypatch):
    launcher = _make_launcher(tmp_path, _ok_payload())
    _stub_claude(monkeypatch)
    with pytest.raises(kf.FableLaneUnavailable, match="lane model must be"):
        kf.preflight(_cfg(launcher, model="claude-opus-5"))


def test_default_spawn_raises_instead_of_falling_back(
    kanban_home, tmp_path, monkeypatch,
):
    """An unavailable lane must be a spawn failure, never a normal worker.

    The dispatcher records the raised error via ``_record_spawn_failure`` and
    eventually auto-blocks. What must NOT happen is a silent
    ``hermes -p fable`` spawn answering the task with the profile's Luna model.
    """
    _write_lane_config(kanban_home, "fable", launcher=str(tmp_path / "gone.py"))
    _stub_claude(monkeypatch)

    def _explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("no worker may be spawned when the lane is broken")

    monkeypatch.setattr("subprocess.Popen", _explode)

    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="read the repo", assignee="fable", classification="deep",
        )
        task = kb.get_task(conn, tid)

    with pytest.raises(kf.FableLaneUnavailable):
        kb._default_spawn(task, str(kanban_home))


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda p: p.update(model_requested="claude-opus-5"), "expected"),
        (lambda p: p.update(usage_credits_enabled=True), "usage credits"),
        (lambda p: p.pop("oauth_only"), "OAuth-only"),
        (lambda p: p.update(oauth_only=False), "OAuth-only"),
        (lambda p: p.pop("usage_credits_enabled"), "paid usage credits"),
        (lambda p: p.update(checks=[]), "no checks"),
        (lambda p: p["checks"][0].update(exit_code=2, stderr="auth failed"), "exited"),
        (lambda p: p["checks"][0].update(timeout=True), "timed out"),
        (
            lambda p: p["checks"][0]["result"].update(
                modelUsage={kf.FABLE_MODEL: {}, "claude-haiku-4-5-20251001": {}}
            ),
            "non-Fable",
        ),
        (
            lambda p: p["checks"][0]["result"].update(
                modelUsage={"claude-opus-5": {}}
            ),
            "non-Fable",
        ),
        (
            lambda p: p["checks"][0]["result"].update(modelUsage={}),
            "exact model",
        ),
        (
            lambda p: p["checks"][0]["result"].update(
                modelUsage={"claude-fable-5[1m]-fallback": {}}
            ),
            "exact model",
        ),
    ],
)
def test_verify_payload_fails_closed(tmp_path, mutate, match):
    payload = _ok_payload()
    mutate(payload)
    with pytest.raises(kf.FableLaneUnavailable, match=match):
        kf.verify_payload(payload, _cfg(tmp_path / "l.py"))


def test_verify_payload_accepts_a_clean_fable_run(tmp_path):
    kf.verify_payload(_ok_payload(), _cfg(tmp_path / "l.py"))


def test_run_launcher_rejects_disabled_launcher(tmp_path, monkeypatch):
    """The launcher's own default-off answer is a failure, not a result."""
    launcher = _make_launcher(
        tmp_path,
        {"status": "disabled", "reason": "explicit local canary flag required"},
    )
    _stub_claude(monkeypatch)
    with pytest.raises(kf.FableLaneUnavailable, match="disabled"):
        kf.run_launcher(_cfg(launcher), "hi", str(tmp_path))


def test_run_launcher_rejects_nonzero_exit(tmp_path, monkeypatch):
    launcher = _make_launcher(tmp_path, _ok_payload(), exit_code=2)
    _stub_claude(monkeypatch)
    with pytest.raises(kf.FableLaneUnavailable, match="exited 2"):
        kf.run_launcher(_cfg(launcher), "hi", str(tmp_path))


def test_run_launcher_reports_timeout_as_transient(tmp_path, monkeypatch):
    launcher = _make_launcher(tmp_path, _ok_payload())
    _stub_claude(monkeypatch)

    def _timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="stub", timeout=1)

    monkeypatch.setattr(kf.subprocess, "run", _timeout)
    with pytest.raises(kf.FableLaneUnavailable) as excinfo:
        kf.run_launcher(_cfg(launcher, timeout_seconds=1), "hi", str(tmp_path))
    assert excinfo.value.kind == "transient"


def test_run_launcher_passes_opt_in_flag_and_strips_credentials(
    tmp_path, monkeypatch,
):
    launcher = _make_launcher(tmp_path, _ok_payload())
    _stub_claude(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-nope")
    monkeypatch.setenv("CLAUDE_CODE_USE_BEDROCK", "1")

    payload = kf.run_launcher(_cfg(launcher), "summarize", str(tmp_path))
    assert kf.result_text(payload) == "done"

    argv = kf.launcher_argv(_cfg(launcher), "summarize", str(tmp_path))
    assert kf.LAUNCHER_OPT_IN_FLAG in argv
    assert str(launcher) in argv
    clean = kf.sanitize_env()
    assert "ANTHROPIC_API_KEY" not in clean
    assert "CLAUDE_CODE_USE_BEDROCK" not in clean
    restricted = kf.sanitize_env({
        "PATH": "/bin", "HOME": "/tmp/home", "CLAUDE_CODE_OAUTH_TOKEN": "secret",
        "AWS_ACCESS_KEY_ID": "key", "AWS_SECRET_ACCESS_KEY": "secret",
        "OPENAI_API_KEY": "secret", "RANDOM_TOKEN": "secret",
    })
    assert restricted == {"PATH": "/bin", "HOME": "/tmp/home"}


def test_prompt_carries_the_read_only_notice(tmp_path, monkeypatch):
    launcher = _make_launcher(tmp_path, _ok_payload())
    _stub_claude(monkeypatch)
    captured = {}

    class _Done:
        returncode = 0
        stdout = json.dumps(_ok_payload())
        stderr = ""

    def _run(argv, **kwargs):
        captured["argv"] = list(argv)
        return _Done()

    monkeypatch.setattr(kf.subprocess, "run", _run)
    kf.run_launcher(_cfg(launcher), "analyse the repo", str(tmp_path))

    prompt = captured["argv"][captured["argv"].index("--prompt") + 1]
    assert prompt.startswith("analyse the repo")
    assert "READ-ONLY LANE" in prompt


# ---------------------------------------------------------------------------
# 3. Kanban lifecycle
# ---------------------------------------------------------------------------

def _claimed_fable_task(
    title: str = "read the repo",
    classification: str = "deep",
) -> str:
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title=title, assignee="fable", classification=classification,
        )
        assert kb.claim_task(conn, tid) is not None
    return tid


def test_run_task_completes_the_task(kanban_home, tmp_path, monkeypatch):
    launcher = _make_launcher(tmp_path, _ok_payload("the answer"))
    _write_lane_config(kanban_home, "fable", launcher=str(launcher))
    _stub_claude(monkeypatch)
    tid = _claimed_fable_task()

    assert kf.run_task(tid, str(tmp_path)) == 0

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        summary = kb.latest_summary(conn, tid)
    assert task.status == "done"
    assert task.result == "the answer"
    assert summary == "the answer"


def test_run_task_refuses_stale_supervisor_without_running_launcher(
    kanban_home, tmp_path, monkeypatch,
):
    launcher = _make_launcher(tmp_path, _ok_payload("must not run"))
    _write_lane_config(kanban_home, "fable", launcher=str(launcher))
    _stub_claude(monkeypatch)
    tid = _claimed_fable_task()

    def _explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("stale supervisor must not invoke launcher")

    monkeypatch.setattr(kf.subprocess, "run", _explode)
    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        current = task.current_run_id
    assert current is not None
    assert kf.run_task(tid, str(tmp_path), expected_run_id=int(current) + 1) == 2

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "running"


def test_run_task_blocks_when_the_lane_is_unavailable(
    kanban_home, tmp_path, monkeypatch,
):
    _write_lane_config(kanban_home, "fable", launcher=str(tmp_path / "gone.py"))
    _stub_claude(monkeypatch)
    tid = _claimed_fable_task()

    assert kf.run_task(tid, str(tmp_path)) == 1

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        reason = kb.latest_summary(conn, tid)
    assert task.status == "blocked"
    assert "launcher not found" in (reason or "")


def test_run_task_blocks_when_the_lane_was_turned_off(
    kanban_home, tmp_path, monkeypatch,
):
    """Config flipped off between spawn and run: block, don't run something else."""
    (kanban_home / "profiles" / "fable").mkdir(parents=True)
    tid = _claimed_fable_task()

    assert kf.run_task(tid, str(tmp_path)) == 1

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        reason = kb.latest_summary(conn, tid)
    assert task.status == "blocked"
    assert "not enabled" in (reason or "")


def test_run_task_blocks_a_task_that_pins_another_model(
    kanban_home, tmp_path, monkeypatch,
):
    launcher = _make_launcher(tmp_path, _ok_payload())
    _write_lane_config(kanban_home, "fable", launcher=str(launcher))
    _stub_claude(monkeypatch)

    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="pinned", assignee="fable", classification="deep",
        )
        conn.execute(
            "UPDATE tasks SET model_override = ? WHERE id = ?",
            ("claude-opus-5", tid),
        )
        conn.commit()
        assert kb.claim_task(conn, tid) is not None

    assert kf.run_task(tid, str(tmp_path)) == 1

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        reason = kb.latest_summary(conn, tid)
    assert task.status == "blocked"
    assert "model_override" in (reason or "")


def test_run_task_blocks_on_an_empty_answer(kanban_home, tmp_path, monkeypatch):
    launcher = _make_launcher(tmp_path, _ok_payload(""))
    _write_lane_config(kanban_home, "fable", launcher=str(launcher))
    _stub_claude(monkeypatch)
    tid = _claimed_fable_task()

    assert kf.run_task(tid, str(tmp_path)) == 1

    with kb.connect() as conn:
        assert kb.get_task(conn, tid).status == "blocked"


# ---------------------------------------------------------------------------
# Canary harness
# ---------------------------------------------------------------------------

def test_canary_is_offline_safe_with_a_stub_launcher(
    kanban_home, tmp_path, monkeypatch, capsys,
):
    launcher = _make_launcher(tmp_path, _ok_payload("FABLE_LANE_CANARY_OK"))
    _write_lane_config(kanban_home, "fable", launcher=str(launcher))
    _stub_claude(monkeypatch)

    assert kf.main(["canary", "--workspace", str(tmp_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    assert report["model"] == kf.FABLE_MODEL
    assert report["answer"] == "FABLE_LANE_CANARY_OK"


def test_canary_reports_a_disabled_lane(kanban_home, tmp_path, capsys):
    (kanban_home / "profiles" / "fable").mkdir(parents=True)
    assert kf.main(["canary", "--workspace", str(tmp_path)]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert "not enabled" in report["reason"]


# ---------------------------------------------------------------------------
# 4. Classification-gated dispatch
# ---------------------------------------------------------------------------

def test_task_has_fable_classification():
    """``task_has_fable_classification`` normalizes and matches only the allowed tags."""
    for tag in kf.FABLE_ALLOWED_CLASSIFICATIONS:
        assert kf.task_has_fable_classification(tag) is True
        assert kf.task_has_fable_classification(tag.upper()) is True
        assert kf.task_has_fable_classification(f"  {tag}  ") is True
    assert kf.task_has_fable_classification(None) is False
    assert kf.task_has_fable_classification("") is False
    assert kf.task_has_fable_classification("random-tag") is False
    # The deep-review board uses an explicit composite classification. It must
    # engage the lane, while lookalikes remain ordinary task metadata.
    assert kf.task_has_fable_classification("deep/fable") is True
    assert kf.task_has_fable_classification(" DEEP/FABLE ") is True
    assert kf.task_has_fable_classification("deep/fable-ish") is False


def test_create_task_persists_classification(kanban_home):
    """``classification`` round-trips through create → get."""
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="classified", assignee="fable", classification="Deep")
        task = kb.get_task(conn, tid)
    assert task.classification == "deep"  # normalized


def test_create_task_persists_none_classification(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="plain", assignee="fable")
        task = kb.get_task(conn, tid)
    assert task.classification is None


def test_unclassified_fable_task_uses_normal_worker(
    kanban_home, tmp_path, monkeypatch,
):
    """A task assigned to fable but with no classification keeps the normal worker path."""
    launcher = _make_launcher(tmp_path, _ok_payload())
    _write_lane_config(kanban_home, "fable", launcher=str(launcher))
    _stub_claude(monkeypatch)

    captured = {}

    class _FakeProc:
        pid = 4242

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        return _FakeProc()

    monkeypatch.setattr("subprocess.Popen", _fake_popen)

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="unclassified work", assignee="fable")
        task = kb.get_task(conn, tid)

    kb._default_spawn(task, str(kanban_home))

    # Normal hermes worker, NOT the fable supervisor.
    assert "hermes_cli.kanban_fable" not in captured["cmd"]
    assert "-p" in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("-p") + 1] == "fable"


@pytest.mark.parametrize("tag", sorted(kf.FABLE_ALLOWED_CLASSIFICATIONS))
def test_classified_fable_task_routes_to_launcher(
    kanban_home, tmp_path, monkeypatch, tag,
):
    """Each allowed classification tag engages the Fable lane."""
    launcher = _make_launcher(tmp_path, _ok_payload())
    _write_lane_config(kanban_home, "fable", launcher=str(launcher))
    _stub_claude(monkeypatch)

    captured = {}

    class _FakeProc:
        pid = 5150

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        return _FakeProc()

    monkeypatch.setattr("subprocess.Popen", _fake_popen)

    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title=f"task-{tag}", assignee="fable", classification=tag,
        )
        task = kb.get_task(conn, tid)

    pid = kb._default_spawn(task, str(kanban_home))

    assert pid == 5150
    assert captured["cmd"][:4] == [
        sys.executable, "-m", "hermes_cli.kanban_fable", "run",
    ]
    assert "ANTHROPIC_API_KEY" not in captured["env"]


def test_run_task_blocks_when_classification_missing(
    kanban_home, tmp_path, monkeypatch,
):
    """Supervisor fails closed when the task has no Fable classification."""
    launcher = _make_launcher(tmp_path, _ok_payload("must not run"))
    _write_lane_config(kanban_home, "fable", launcher=str(launcher))
    _stub_claude(monkeypatch)

    def _explode(*args, **kwargs):  # pragma: no cover
        raise AssertionError("supervisor must not invoke launcher without classification")

    monkeypatch.setattr(kf.subprocess, "run", _explode)

    # Create WITHOUT classification, then claim so the supervisor can run.
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="unclassified", assignee="fable")
        assert kb.claim_task(conn, tid) is not None

    assert kf.run_task(tid, str(tmp_path)) == 1

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
        reason = kb.latest_summary(conn, tid)
    assert task.status == "blocked"
    assert "classification" in (reason or "").lower()


def test_run_task_blocks_when_classification_is_invalid_tag(
    kanban_home, tmp_path, monkeypatch,
):
    """Supervisor fails closed when the task has an unrecognised classification."""
    launcher = _make_launcher(tmp_path, _ok_payload("must not run"))
    _write_lane_config(kanban_home, "fable", launcher=str(launcher))
    _stub_claude(monkeypatch)

    def _explode(*args, **kwargs):  # pragma: no cover
        raise AssertionError("supervisor must not invoke launcher with wrong tag")

    monkeypatch.setattr(kf.subprocess, "run", _explode)

    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="bad tag", assignee="fable", classification="random-tag",
        )
        assert kb.claim_task(conn, tid) is not None

    assert kf.run_task(tid, str(tmp_path)) == 1

    with kb.connect() as conn:
        task = kb.get_task(conn, tid)
    assert task.status == "blocked"


@pytest.mark.skipif(
    os.environ.get("HERMES_FABLE_CANARY") != "1",
    reason="live canary: needs a real OAuth Claude Code install; set HERMES_FABLE_CANARY=1",
)
def test_live_canary_runs_the_real_launcher(tmp_path):
    """Read-only end-to-end canary against the configured launcher.

    Opt-in only. Reads the operator's real profile config, runs one read-only
    prompt in a throwaway directory, and asserts the Fable model answered.
    """
    profile = os.environ.get("HERMES_FABLE_CANARY_PROFILE", kf.DEFAULT_LANE_ASSIGNEE)
    cfg = kf.load_lane_config(profile)
    assert cfg is not None, f"kanban.fable_lane not enabled for profile {profile!r}"
    payload = kf.run_launcher(cfg, kf.CANARY_PROMPT, str(tmp_path))
    assert payload["model_requested"] == kf.FABLE_MODEL
    assert "FABLE_LANE_CANARY_OK" in kf.result_text(payload)
