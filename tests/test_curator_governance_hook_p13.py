"""P13 isolation proof for scripts/curator-governance-hook.py.

Verifies:
- HERMES_HOME parameterisation: BASE and derived paths (SKILLS_DIR,
  PROFILES_DIR, STATE_FILE, GOVERNANCE_LOG, LOCK_FILE) resolve under
  HERMES_HOME, not /home/kensei/.hermes.
- --dry-run suppresses every write path: no `hermes curator pin` CLI
  calls, no add_skill_to_enabled config mutation, no set_adoption_status
  SKILL.md mutation, no log_event logboard write, no lockfile create.
  Read paths (load_profile_skills, read_curator_report) run unchanged.
- import-safe: importing the module does not create the lockfile or
  logboard dir.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "curator-governance-hook.py"


def _load_module(monkeypatch, fake_home: Path):
    monkeypatch.setenv("HERMES_HOME", str(fake_home))
    scripts_dir = REPO_ROOT / "scripts"
    for pth in (str(scripts_dir), str(REPO_ROOT)):
        if pth not in sys.path:
            sys.path.insert(0, pth)
    spec = importlib.util.spec_from_file_location(
        "curator_governance_hook_under_test", str(SCRIPT)
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_home(tmp_path):
    fake = tmp_path / "fake_hermes"
    fake.mkdir()
    return fake


def test_paths_resolve_under_hermes_home(monkeypatch, fake_home):
    mod = _load_module(monkeypatch, fake_home)
    assert str(mod.BASE).startswith(str(fake_home))
    assert mod.SKILLS_DIR == fake_home / "skills"
    assert mod.PROFILES_DIR == fake_home / "profiles"
    assert mod.STATE_FILE == fake_home / "skills" / ".curator_state"
    assert mod.GOVERNANCE_LOG == fake_home / "governance" / "logboard"
    assert mod.LOCK_FILE == fake_home / ".curator_governance_hook.lock"


def test_import_is_side_effect_free(monkeypatch, fake_home):
    assert not (fake_home / ".curator_governance_hook.lock").exists()
    _load_module(monkeypatch, fake_home)
    assert not (fake_home / ".curator_governance_hook.lock").exists(), (
        "import created the lockfile"
    )


def test_dry_run_exits_zero_without_cli(monkeypatch, fake_home, tmp_path):
    """--dry-run must exit 0 with no curator report present and no `hermes`
    CLI on PATH. No lockfile, no logboard dir created."""
    env = dict(os.environ)
    env["HERMES_HOME"] = str(fake_home)
    env["PATH"] = "/usr/bin:/bin"  # no hermes CLI
    env["PYTHONPATH"] = str(REPO_ROOT)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--dry-run"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"dry-run failed: rc={proc.returncode} stderr={proc.stderr!r} stdout={proc.stdout!r}"
    )
    assert not (fake_home / ".curator_governance_hook.lock").exists()
    assert not (fake_home / "governance").exists() or not any(
        (fake_home / "governance").iterdir()
    )


def test_dry_run_does_not_pin_or_mutate(monkeypatch, fake_home, tmp_path):
    """With a curator report present proposing an archival of a
    profile-referenced skill, --dry-run must NOT call `hermes curator pin`,
    NOT mutate the profile config, NOT write the logboard, and NOT create
    the lockfile. The re-pin decision is still computed (archival override
    detected) but the write is suppressed."""
    mod = _load_module(monkeypatch, fake_home)
    # Build a fake HERMES_HOME with a profile config referencing a skill,
    # and a curator report proposing to archive that skill.
    (fake_home / "profiles").mkdir(parents=True)
    (fake_home / "skills").mkdir(parents=True)
    profile_cfg = fake_home / "profiles" / "octacon" / "config.yaml"
    profile_cfg.parent.mkdir(parents=True, exist_ok=True)
    profile_cfg.write_text(
        "skills:\n  enabled_skills:\n    - my-skill\n"
    )
    # curator state pointing at a report dir
    report_dir = tmp_path / "report"
    report_dir.mkdir()
    (report_dir / "run.json").write_text(
        '{"started_at":"2026-01-01T00:00:00Z","counts":{},'
        '"archived":["my-skill"],"added":[]}'
    )
    state_file = fake_home / "skills" / ".curator_state"
    state_file.write_text(f'{{"last_report_path": "{report_dir}"}}')

    mod._DRY_RUN = True
    # Stub subprocess.run to detect any `hermes` call (must not happen).
    called = []
    import types
    real_run = subprocess.run

    def spy_run(cmd, *a, **k):
        if cmd and cmd[0] == "hermes":
            called.append(cmd)
            raise AssertionError(f"dry-run called hermes CLI: {cmd}")
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(subprocess, "run", spy_run)
    # Run main (argv already parsed would set the flag; we set it directly).
    monkeypatch.setattr(sys, "argv", ["curator-governance-hook.py", "--dry-run"])
    mod.main()
    assert called == [], "dry-run invoked the hermes CLI"
    # Profile config must be unchanged.
    assert "my-skill" in profile_cfg.read_text()
    assert "dry-run" not in profile_cfg.read_text()
    # No logboard dir/file created.
    assert not (fake_home / "governance" / "logboard").exists() or not any(
        (fake_home / "governance" / "logboard").glob("*.mdl")
    )
    # No lockfile.
    assert not (fake_home / ".curator_governance_hook.lock").exists()


# ===========================================================================
# STEP 6 (Behaviour A) — direct post-curator governance + weekly replay.
# Does not alter the P13 isolation tests above.
#
# Contract:
#   - gateway passes --direct: governance runs once per curator report and
#     actionable output is stored in a runtime state file under HERMES_HOME
#     (NOT the governance GitOps tree) keyed by the report's started_at with
#     pending_delivery.
#   - weekly cron (no --direct): if the same report was already processed
#     and pending_delivery exists → print it EXACTLY ONCE, clear it, and
#     repeat no mutations/log events. If the direct run was missed → process
#     normally. No-action reports are marked processed and stay silent.
#   - --dry-run writes nothing (no marker, no lock, no mutations).
# ===========================================================================

import json as _json2  # noqa: E402


DELIVERY_STATE_NAME = ".curator_governance_delivery.json"


def _seed_report(fake_home: Path, tmp_path: Path, *,
                 archived: list | None = None,
                 added: list | None = None,
                 started_at: str = "2026-08-29T00:00:00Z") -> dict:
    """Create a curator report + .curator_state pointing at it."""
    (fake_home / "profiles").mkdir(parents=True, exist_ok=True)
    (fake_home / "skills").mkdir(parents=True, exist_ok=True)
    profile_cfg = fake_home / "profiles" / "octacon" / "config.yaml"
    profile_cfg.parent.mkdir(parents=True, exist_ok=True)
    profile_cfg.write_text("skills:\n  enabled_skills:\n    - my-skill\n")
    report_dir = tmp_path / (
        "report-" + started_at.replace(":", "").replace("-", ""))
    report_dir.mkdir()
    (report_dir / "run.json").write_text(_json2.dumps({
        "started_at": started_at,
        "counts": {"checked": 2, "archived_this_run": 0, "added_this_run": 0},
        "archived": archived or [],
        "added": added or [],
    }))
    state_file = fake_home / "skills" / ".curator_state"
    state_file.write_text(_json2.dumps({"last_report_path": str(report_dir)}))
    return {"started_at": started_at, "report_dir": report_dir}


def _read_delivery_state(fake_home: Path) -> dict:
    path = fake_home / DELIVERY_STATE_NAME
    if not path.exists():
        return {}
    return _json2.loads(path.read_text())


def test_delivery_state_file_under_hermes_home(monkeypatch, fake_home):
    """The marker lives under HERMES_HOME root, not the governance GitOps
    tree (governance/logboard/)."""
    mod = _load_module(monkeypatch, fake_home)
    assert mod.DELIVERY_STATE_FILE == fake_home / DELIVERY_STATE_NAME
    assert "governance/logboard" not in str(mod.DELIVERY_STATE_FILE)
    assert mod.DELIVERY_STATE_FILE.parent == fake_home


def test_direct_fresh_report_processes_once_and_stores_pending(
        monkeypatch, fake_home, tmp_path):
    mod = _load_module(monkeypatch, fake_home)
    info = _seed_report(fake_home, tmp_path, archived=["my-skill"])

    pin_calls = []
    real_run = subprocess.run

    def spy_run(cmd, *a, **k):
        if cmd and cmd[0] == "hermes":
            pin_calls.append(cmd)
            class R: returncode = 0; stderr = ""; stdout = ""
            return R()
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(subprocess, "run", spy_run)
    import io, contextlib
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        mod._DIRECT = True
        mod.main()

    # Direct run itself stays silent on stdout (weekly cron delivers).
    assert buf_out.getvalue() == ""
    # Governance ran: the re-pin happened exactly once.
    assert pin_calls == [["hermes", "curator", "pin", "my-skill"]]
    # State file: processed marker + pending delivery present.
    state = _read_delivery_state(fake_home)
    assert state.get("processed_started_at") == info["started_at"]
    pending = state.get("pending_delivery")
    assert isinstance(pending, str) and "my-skill" in pending, (
        f"actionable output not stored for delivery: {state!r}")
    # The runtime state file sits under HERMES_HOME, not governance/.
    assert (fake_home / DELIVERY_STATE_NAME).is_file()
    assert not (fake_home / "governance" / DELIVERY_STATE_NAME).exists()


def test_duplicate_direct_run_is_noop(monkeypatch, fake_home, tmp_path):
    mod = _load_module(monkeypatch, fake_home)
    info = _seed_report(fake_home, tmp_path, archived=["my-skill"])

    real_run = subprocess.run

    def mk_spy(counter):

        def spy_run(cmd, *a, **k):
            if cmd and cmd[0] == "hermes":
                counter.append(cmd)
                class R: returncode = 0; stderr = ""; stdout = ""
                return R()
            return real_run(cmd, *a, **k)
        return spy_run

    counter1: list = []
    monkeypatch.setattr(subprocess, "run", mk_spy(counter1))
    mod._DIRECT = True
    mod.main()
    state_before = (fake_home / DELIVERY_STATE_NAME).read_bytes()

    # Second direct run on the SAME report: pure no-op.
    counter2: list = []
    monkeypatch.setattr(subprocess, "run", mk_spy(counter2))
    import io, contextlib
    buf_out = io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(io.StringIO()):
        mod.main()
    assert counter2 == [], "duplicate direct run repeated governance mutations"
    assert (fake_home / DELIVERY_STATE_NAME).read_bytes() == state_before, (
        "duplicate direct run rewrote the delivery state")


def test_two_direct_reports_before_weekly_preserve_both_outputs(
        monkeypatch, fake_home, tmp_path):
    """A new weekly curator pass must not overwrite an older undelivered alert."""
    mod = _load_module(monkeypatch, fake_home)

    class R:
        returncode = 0
        stderr = ""
        stdout = ""

    real_run = subprocess.run

    def spy_run(cmd, *args, **kwargs):
        return R() if cmd and cmd[0] == "hermes" else real_run(
            cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", spy_run)
    mod._DIRECT = True

    _seed_report(
        fake_home, tmp_path, archived=["my-skill"],
        started_at="2026-08-22T00:00:00Z",
    )
    mod.main()

    _seed_report(
        fake_home, tmp_path, archived=["other-skill"],
        started_at="2026-08-29T00:00:00Z",
    )
    profile_cfg = fake_home / "profiles" / "octacon" / "config.yaml"
    profile_cfg.write_text(
        "skills:\n  enabled_skills:\n    - my-skill\n    - other-skill\n"
    )
    mod.main()

    state = _read_delivery_state(fake_home)
    pending = state.get("pending_delivery") or ""
    assert "my-skill" in pending
    assert "other-skill" in pending

    import contextlib
    import io

    mod._DIRECT = False
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        mod.main()
    assert "my-skill" in out.getvalue()
    assert "other-skill" in out.getvalue()
    assert not _read_delivery_state(fake_home).get("pending_delivery")


def test_weekly_replays_pending_exactly_once(monkeypatch, fake_home, tmp_path):
    mod = _load_module(monkeypatch, fake_home)
    info = _seed_report(fake_home, tmp_path, archived=["my-skill"])
    pending_text = ("🚫 Curator governance - 1 blocked archival(s), "
                    "0 skill(s) need review\n"
                    "  my-skill - referenced by: octacon\n")
    (fake_home / DELIVERY_STATE_NAME).write_text(_json2.dumps({
        "processed_started_at": info["started_at"],
        "pending_delivery": pending_text,
    }))

    pin_calls: list = []
    real_run = subprocess.run

    def spy_run(cmd, *a, **k):
        if cmd and cmd[0] == "hermes":
            pin_calls.append(cmd)
            raise AssertionError("weekly replay repeated governance mutations")
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(subprocess, "run", spy_run)
    import io, contextlib
    buf_out = io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(io.StringIO()):
        mod._DIRECT = False
        mod.main()

    # Output delivered exactly once, byte-identical to the stored payload.
    assert buf_out.getvalue() == pending_text, repr(buf_out.getvalue())
    # pending_delivery cleared, processed marker kept.
    state = _read_delivery_state(fake_home)
    assert state.get("processed_started_at") == info["started_at"]
    assert not state.get("pending_delivery")

    # A second weekly run must deliver nothing further.
    buf_out2 = io.StringIO()
    with contextlib.redirect_stdout(buf_out2), contextlib.redirect_stderr(io.StringIO()):
        mod.main()
    assert buf_out2.getvalue() == "", "weekly replay printed twice"


def test_weekly_processes_fresh_when_direct_missed(
        monkeypatch, fake_home, tmp_path):
    """Direct run missed/failed → weekly falls back to normal processing:
    prints actionable output AND marks processed."""
    mod = _load_module(monkeypatch, fake_home)
    info = _seed_report(fake_home, tmp_path, archived=["my-skill"])
    # No delivery-state file at all.

    pin_calls: list = []
    real_run = subprocess.run

    def spy_run(cmd, *a, **k):
        if cmd and cmd[0] == "hermes":
            pin_calls.append(cmd)
            class R: returncode = 0; stderr = ""; stdout = ""
            return R()
        return real_run(cmd, *a, **k)

    monkeypatch.setattr(subprocess, "run", spy_run)
    import io, contextlib
    buf_out = io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(io.StringIO()):
        mod._DIRECT = False
        mod.main()

    assert "my-skill" in buf_out.getvalue(), (
        f"weekly fallback did not deliver actionable output: {buf_out.getvalue()!r}")
    state = _read_delivery_state(fake_home)
    assert state.get("processed_started_at") == info["started_at"]
    assert not state.get("pending_delivery"), (
        "fallback marked pending for output it already delivered")
    # Governance mutations ran (re-pin), exactly once this run.
    assert pin_calls == [["hermes", "curator", "pin", "my-skill"]]


def test_no_action_report_marks_processed_and_stays_silent(
        monkeypatch, fake_home, tmp_path):
    mod = _load_module(monkeypatch, fake_home)
    info = _seed_report(fake_home, tmp_path, archived=[], added=[])
    import io, contextlib

    # Direct: processed, nothing pending, stdout silent.
    mod._DIRECT = True
    with contextlib.redirect_stdout(io.StringIO()) as out, \
            contextlib.redirect_stderr(io.StringIO()):
        mod.main()
    state = _read_delivery_state(fake_home)
    assert state.get("processed_started_at") == info["started_at"]
    assert not state.get("pending_delivery")

    # Weekly: silent too (no replay of nothing).
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        mod._DIRECT = False
        mod.main()
    assert buf.getvalue() == ""


def test_dry_run_never_writes_delivery_state(monkeypatch, fake_home, tmp_path):
    mod = _load_module(monkeypatch, fake_home)
    _seed_report(fake_home, tmp_path, archived=["my-skill"])
    mod._DIRECT = True
    mod._DRY_RUN = True
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()), \
            contextlib.redirect_stderr(io.StringIO()):
        mod.main()
    assert not (fake_home / DELIVERY_STATE_NAME).exists(), (
        "--dry-run wrote the delivery state marker")
    assert not (fake_home / ".curator_governance_hook.lock").exists(), (
        "--dry-run created the lockfile")
    # No config mutation either.
    cfg = fake_home / "profiles" / "octacon" / "config.yaml"
    assert "my-skill" in cfg.read_text()


def test_direct_run_without_report_is_silent_noop(
        monkeypatch, fake_home, tmp_path):
    """No curator report at all: both modes are silent no-ops that create
    nothing."""
    mod = _load_module(monkeypatch, fake_home)
    (fake_home / "skills").mkdir(parents=True)
    import io, contextlib
    for direct in (True, False):
        mod._DIRECT = direct
        buf_out = io.StringIO()
        with contextlib.redirect_stdout(buf_out), \
                contextlib.redirect_stderr(io.StringIO()):
            mod.main()
        assert buf_out.getvalue() == ""
    assert not (fake_home / DELIVERY_STATE_NAME).exists()
