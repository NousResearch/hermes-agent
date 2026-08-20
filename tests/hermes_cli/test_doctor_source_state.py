from pathlib import Path

from hermes_cli.doctor import collect_source_tree_state


def test_collect_source_tree_state_empty_for_non_git_directory(tmp_path):
    assert collect_source_tree_state(tmp_path) == []


def test_collect_source_tree_state_reports_dirty_tracked_and_untracked(tmp_path):
    import subprocess

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("one", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)

    tracked.write_text("two", encoding="utf-8")
    (tmp_path / "scratch.txt").write_text("scratch", encoding="utf-8")

    rows = collect_source_tree_state(tmp_path)
    assert any(level == "warn" and "local modifications" in text for level, text, _ in rows)
    assert any(level == "info" and "Untracked source files" in text and "scratch.txt" in detail for level, text, detail in rows)



def test_report_source_tree_state_renders_info_rows_without_type_error(tmp_path, monkeypatch):
    import subprocess
    from hermes_cli import doctor

    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=tmp_path, check=True)
    (tmp_path / "tracked.txt").write_text("one", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-m", "init"], cwd=tmp_path, check=True, capture_output=True)
    (tmp_path / "scratch.txt").write_text("scratch", encoding="utf-8")

    rendered = []
    monkeypatch.setattr(doctor, "check_ok", lambda text, detail="": rendered.append(("ok", text, detail)))
    monkeypatch.setattr(doctor, "check_warn", lambda text, detail="": rendered.append(("warn", text, detail)))
    monkeypatch.setattr(doctor, "check_info", lambda text: rendered.append(("info", text, "")))

    doctor.report_source_tree_state(tmp_path)

    assert any(level == "info" and "Source checkout" in text for level, text, _ in rendered)
    assert any(level == "info" and "scratch.txt" in text for level, text, _ in rendered)


def test_collect_source_tree_state_uses_single_shared_latency_budget(tmp_path, monkeypatch):
    import subprocess

    from hermes_cli import doctor

    (tmp_path / ".git").mkdir()
    time_values = iter([0.0, 0.0, 3.1, 3.1, 3.1, 3.1])
    monkeypatch.setattr(doctor.time, "monotonic", lambda: next(time_values))
    calls = []

    def fake_run(*args, **kwargs):
        calls.append(kwargs["timeout"])
        return subprocess.CompletedProcess(args[0], 0, stdout="main\n", stderr="")

    monkeypatch.setattr(doctor.subprocess, "run", fake_run)

    doctor.collect_source_tree_state(tmp_path)

    assert calls == [3.0]
