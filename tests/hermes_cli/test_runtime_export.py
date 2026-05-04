from __future__ import annotations

import json
import stat
from datetime import datetime, timezone
from pathlib import Path

from hermes_cli import runtime_export


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _stub_run_command(argv):
    if argv[:2] == ["kael", "--version"]:
        return {"returncode": 0, "stdout": "Hermes Agent v0.12.0\n", "stderr": ""}
    if argv[:2] == ["hermes", "--version"]:
        return {"returncode": 0, "stdout": "Hermes Agent v0.12.0\n", "stderr": ""}
    if argv[:3] == ["systemctl", "--user", "show"]:
        return {
            "returncode": 0,
            "stdout": (
                "ActiveState=active\n"
                "SubState=running\n"
                "MainPID=1234\n"
                "UnitFileState=enabled\n"
                "ExecMainStartTimestamp=Fri 2026-05-02 00:00:00 UTC\n"
            ),
            "stderr": "",
        }
    if argv and argv[0] == "ps":
        return {
            "returncode": 0,
            "stdout": (
                "123 ubuntu 1 120 0.1 0.2 /home/ubuntu/.hermes/hermes-agent/venv/bin/python -m hermes_cli.main gateway run --replace\n"
                "456 ubuntu 123 60 0.0 0.1 npm exec @stripe/mcp --api-key=rk_liv_abc123\n"
            ),
            "stderr": "",
        }
    if argv and argv[0] == "git":
        return {"returncode": 1, "stdout": "", "stderr": "not a git repository"}
    return {"returncode": 0, "stdout": "", "stderr": ""}


def test_export_runtime_snapshot_excludes_secrets_and_exports_safe_metadata(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    output_root = tmp_path / "export-root"

    _write(hermes_home / "SOUL.md", "# Soul\noperator doctrine\n")
    _write(hermes_home / "BOOT.md", "# Boot\nchecklist\n")
    _write(
        hermes_home / ".skills_prompt_snapshot.json",
        json.dumps({"version": 1, "manifest": {"ops/a/SKILL.md": [1, 2], "ops/b/SKILL.md": [3, 4]}}),
    )
    _write(hermes_home / "config.yaml", "model:\n  provider: openai-codex\n")
    _write(hermes_home / ".env", "OPENAI_API_KEY=super-secret\n")
    _write(hermes_home / "auth.json", '{"token": "top-secret"}\n')
    _write(hermes_home / "x_credentials.env", "X_PASSWORD=very-secret\n")

    _write(hermes_home / "skills" / "alpha" / "SKILL.md", "# Alpha\n")
    _write_bytes(hermes_home / "heapdumps" / "heap-1.heapsnapshot", b"heap-bytes")
    _write_bytes(hermes_home / "archives" / "sessions" / "archive-1.zip", b"archive-bytes")
    _write(hermes_home / "checkpoints" / "cp-1" / "HERMES_WORKDIR", "/tmp/work\n")
    _write(hermes_home / "checkpoints" / "cp-1" / "index", "index\n")
    _write(hermes_home / "logs" / "disk-monitor.log", "a\nb\nc\nd\n")
    _write(hermes_home / "logs" / "update.log", "u1\nu2\n")
    _write(hermes_home / "logs" / "agent.log", "raw agent log should not be tailed\n")
    _write(hermes_home / "sessions" / "session_1.json", '{"secret": "TOPSECRET"}\n')
    _write(hermes_home / "memories" / "USER.md", "private memory\n")
    _write(
        hermes_home / "cron" / "jobs.json",
        json.dumps(
            {
                "jobs": [
                    {
                        "id": "job-1",
                        "schedule": "every 30m",
                        "schedule_display": "every 30m",
                        "enabled": True,
                        "state": "scheduled",
                        "last_status": "ok",
                        "next_run_at": "2026-05-02T01:00:00Z",
                        "prompt": "VERY SECRET PROMPT",
                        "deliver": "origin",
                        "skills": ["hermes-agent"],
                    }
                ]
            }
        ),
    )

    _write(hermes_home / "profiles" / "canary" / "config.yaml", "model:\n  provider: openai-codex\n")
    _write(hermes_home / "profiles" / "canary" / "sessions" / "session_2.json", '{}\n')
    _write(hermes_home / "profiles" / "canary" / "memories" / "MEMORY.md", "x\n")
    _write(hermes_home / "profiles" / "canary" / "skills" / "beta" / "SKILL.md", "# Beta\n")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(runtime_export, "_run_command", _stub_run_command)

    summary = runtime_export.export_runtime_snapshot(output_root=output_root, keep_releases=2, log_tail_lines=3)

    release_dir = output_root / "releases" / summary["metadata"]["release_name"]
    assert release_dir.is_dir()
    assert not (output_root / "current").exists()
    assert (output_root / "summary.json").is_file()
    assert (output_root / "exported_at.txt").is_file()
    assert (output_root / "current_release.txt").is_file()
    assert stat.S_IMODE(output_root.stat().st_mode) == 0o2750
    assert stat.S_IMODE((output_root / "releases").stat().st_mode) == 0o2750
    assert stat.S_IMODE((output_root / "summary.json").stat().st_mode) == 0o640
    assert (output_root / "inventory").is_dir()
    assert (output_root / "ops").is_dir()
    assert (output_root / "tails").is_dir()
    assert (release_dir / "summary.json").exists()
    assert (release_dir / "SOUL.md").exists()
    assert (release_dir / "BOOT.md").exists()
    assert (release_dir / "skills_snapshot.json").exists()

    summary_text = (output_root / "summary.json").read_text(encoding="utf-8")
    exported = json.loads(summary_text)

    assert exported["direct_files"]["skills_snapshot.json"]["manifest_count"] == 2
    assert exported["inventories"]["skills"]["skill_count"] == 1
    assert exported["inventories"]["heapdumps"]["file_count"] == 1
    assert exported["inventories"]["archives"]["file_count"] == 1
    assert exported["inventories"]["sessions"]["file_count"] == 1
    assert exported["ops"]["cron"]["job_count"] == 1
    assert exported["ops"]["cron"]["jobs"][0]["name"] == "[auto-redacted-from-prompt]"
    assert "prompt" not in exported["ops"]["cron"]["jobs"][0]
    assert exported["ops"]["profiles"]["profile_count"] == 2
    assert exported["tails"]["disk-monitor.log"]["lines"] == ["b", "c", "d"]
    assert "heapdumps_present" in exported["health_flags"]

    assert not (release_dir / ".env").exists()
    assert not (release_dir / "auth.json").exists()
    assert not (release_dir / "x_credentials.env").exists()
    assert "TOPSECRET" not in summary_text
    assert "VERY SECRET PROMPT" not in summary_text
    assert "rk_liv_abc123" not in summary_text
    assert '"label": "stripe mcp"' in summary_text

    assert summary["metadata"]["release_name"] == exported["metadata"]["release_name"]


def test_export_runtime_snapshot_skips_unparseable_skills_snapshot_and_prunes_old_releases(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    output_root = tmp_path / "export-root"

    _write(hermes_home / "SOUL.md", "# Soul\n")
    _write(hermes_home / "BOOT.md", "# Boot\n")
    _write(hermes_home / ".skills_prompt_snapshot.json", "{not-json\n")
    _write(hermes_home / "config.yaml", "model:\n  provider: openai-codex\n")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(runtime_export, "_run_command", _stub_run_command)

    first_now = datetime(2026, 5, 2, 1, 0, 0, tzinfo=timezone.utc)
    second_now = datetime(2026, 5, 2, 1, 30, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(runtime_export, "_now_utc", lambda: first_now)
    runtime_export.export_runtime_snapshot(output_root=output_root, keep_releases=1, log_tail_lines=2)

    monkeypatch.setattr(runtime_export, "_now_utc", lambda: second_now)
    summary = runtime_export.export_runtime_snapshot(output_root=output_root, keep_releases=1, log_tail_lines=2)

    releases = sorted([p.name for p in (output_root / "releases").iterdir() if p.is_dir()])
    assert releases == [summary["metadata"]["release_name"]]

    exported = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert exported["direct_files"]["skills_snapshot.json"]["copied"] is False
    assert any("not parseable" in warning for warning in exported["warnings"])
    assert not (output_root / "skills_snapshot.json").exists()


def test_export_runtime_snapshot_rejects_symlink_sources(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    output_root = tmp_path / "export-root"

    _write(hermes_home / ".env", "OPENAI_API_KEY=super-secret\n")
    _write(hermes_home / "BOOT.md", "# Boot\n")
    _write(hermes_home / ".skills_prompt_snapshot.json", json.dumps({"version": 1, "manifest": {}}))
    _write(hermes_home / "config.yaml", "model:\n  provider: openai-codex\n")
    (hermes_home / "SOUL.md").parent.mkdir(parents=True, exist_ok=True)
    (hermes_home / "SOUL.md").symlink_to(hermes_home / ".env")
    (hermes_home / "logs").mkdir(parents=True, exist_ok=True)
    (hermes_home / "logs" / "update.log").symlink_to(hermes_home / ".env")

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(runtime_export, "_run_command", _stub_run_command)

    exported = runtime_export.export_runtime_snapshot(output_root=output_root, keep_releases=1, log_tail_lines=2)
    summary_text = (output_root / "summary.json").read_text(encoding="utf-8")

    assert exported["direct_files"]["SOUL.md"]["copied"] is False
    assert exported["direct_files"]["SOUL.md"]["symlink_rejected"] is True
    assert exported["tails"]["update.log"]["symlink_rejected"] is True
    assert "super-secret" not in summary_text
    assert not (output_root / "SOUL.md").exists()


def test_export_runtime_snapshot_does_not_rechmod_existing_output_root(tmp_path, monkeypatch):
    hermes_home = tmp_path / ".hermes"
    output_root = tmp_path / "export-root"

    _write(hermes_home / "SOUL.md", "# Soul\n")
    _write(hermes_home / "BOOT.md", "# Boot\n")
    _write(hermes_home / "config.yaml", "model:\n  provider: openai-codex\n")

    output_root.mkdir(parents=True)
    output_root.chmod(0o2770)

    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    monkeypatch.setattr(runtime_export, "_run_command", _stub_run_command)

    runtime_export.export_runtime_snapshot(output_root=output_root, keep_releases=1, log_tail_lines=2)

    assert stat.S_IMODE(output_root.stat().st_mode) == 0o2770
    assert stat.S_IMODE((output_root / "releases").stat().st_mode) == 0o2750
    assert stat.S_IMODE((output_root / "summary.json").stat().st_mode) == 0o640
    assert stat.S_IMODE((output_root / "inventory").stat().st_mode) == 0o2750
