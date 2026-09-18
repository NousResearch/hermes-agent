"""``hermes doctor`` Installation section (hermes_cli/doctor_install.py)."""

import subprocess
from pathlib import Path

from hermes_cli import doctor_install


def _git_repo(tmp_path: Path) -> None:
    for argv in (["init"], ["config", "user.email", "t@example.com"], ["config", "user.name", "T"]):
        subprocess.run(["git", *argv], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "tracked.txt").write_text("one", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)


def test_source_state_reports_dirty_tracked_and_untracked_and_renders(tmp_path, monkeypatch):
    assert doctor_install.collect_source_tree_state(tmp_path) == []  # not a checkout: nothing to say
    tmp_path = tmp_path / "checkout"  # the shared conftest seeds tmp_path with a fake HERMES_HOME dir
    tmp_path.mkdir()
    _git_repo(tmp_path)
    (tmp_path / "tracked.txt").write_text("two", encoding="utf-8")
    (tmp_path / "scratch.txt").write_text("scratch", encoding="utf-8")

    rendered = []
    monkeypatch.setattr(doctor_install, "check_ok", lambda text, detail="": rendered.append(("ok", text, detail)))
    monkeypatch.setattr(doctor_install, "check_warn", lambda text, detail="": rendered.append(("warn", text, detail)))
    monkeypatch.setattr(doctor_install, "check_info", lambda text: rendered.append(("info", text, "")))
    doctor_install.report_source_tree_state(tmp_path)

    assert any(level == "warn" and "1 tracked file(s) changed" in detail for level, _, detail in rendered)
    # info rows carry their detail inline: check_info takes one argument (the first version passed two).
    assert any(level == "info" and text.startswith("Untracked source files: 1 file(s); first: scratch.txt")
               for level, text, _ in rendered)


def test_source_state_shares_one_latency_budget_across_git_probes(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    time_values = iter([0.0, 0.0, 3.1, 3.1, 3.1, 3.1])
    monkeypatch.setattr(doctor_install.time, "monotonic", lambda: next(time_values))
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs["timeout"])
        return subprocess.CompletedProcess(args[0], 0, stdout="main\n", stderr="")

    monkeypatch.setattr(doctor_install.subprocess, "run", fake_run)
    doctor_install.collect_source_tree_state(tmp_path)
    assert calls == [3.0]  # the second probe found the budget spent and never spawned


def test_installation_state_names_version_method_upstream_distance_and_update_command(tmp_path, monkeypatch):
    import hermes_cli.banner as banner
    import hermes_cli.config as config

    monkeypatch.setattr(config, "detect_install_method", lambda root: "git")
    monkeypatch.setattr(config, "load_config", lambda: {})
    monkeypatch.setattr(config, "recommended_update_command", lambda: "hermes update")
    monkeypatch.setattr(banner, "get_git_banner_state", lambda: {"ahead": 2})
    seen = {}
    monkeypatch.setattr(banner, "check_for_updates", lambda passive: seen.setdefault("passive", passive) or 3)

    rows = doctor_install.collect_installation_state(tmp_path)

    assert rows[0][0] == "ok" and rows[0][1].startswith("Hermes Agent v")
    assert rows[1] == ("info", "Install method", "git")
    assert rows[2] == ("warn", "3 commit(s) behind upstream main, 2 local commit(s) carried", "")
    assert rows[-1] == ("info", "Update", "hermes update")
    assert seen == {"passive": False}  # doctor is explicit: never silently skipped by the banner opt-out

    # Docker has no working tree to compare and must not hit GitHub at all.
    monkeypatch.setattr(config, "detect_install_method", lambda root: "docker")
    monkeypatch.setattr(banner, "check_for_updates", lambda passive: (_ for _ in ()).throw(AssertionError("probed")))
    rows = doctor_install.collect_installation_state(tmp_path)
    assert rows[2][0] == "info" and rows[2][1] == "Upstream check not applicable"
