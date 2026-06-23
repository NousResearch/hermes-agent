from __future__ import annotations

import json
from pathlib import Path

from hermes_cli.agent_readiness import (
    FIXED,
    READY,
    WARN,
    check_agent_readiness,
)
from hermes_cli.profiles import create_profile


def _profile(root: Path) -> Path:
    profile = root / "profiles" / "jeeves"
    for name in ["skills", "plugins", "cron", "memories", "sessions", "logs", "workspace"]:
        (profile / name).mkdir(parents=True, exist_ok=True)
    (profile / ".env").write_text("# secrets\n", encoding="utf-8")
    return profile


def test_ready_profile_has_no_issues(tmp_path: Path) -> None:
    profile = _profile(tmp_path)

    result = check_agent_readiness(
        profile_name="jeeves",
        profile_dir=profile,
        runtime_home=profile,
        persist=True,
    )

    assert result.status == READY
    assert result.issues == []
    assert json.loads((profile / ".readiness" / "issues.json").read_text()) == []


def test_profile_home_drift_is_warned_and_persisted(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    wrong_home = tmp_path / "profiles" / "alfred"
    wrong_home.mkdir(parents=True)

    result = check_agent_readiness(
        profile_name="jeeves",
        profile_dir=profile,
        runtime_home=wrong_home,
        persist=True,
    )

    assert result.status == WARN
    assert [issue.check for issue in result.issues] == ["profile_home"]
    assert result.issues[0].expected == str(profile)
    assert result.issues[0].actual == str(wrong_home)
    persisted = json.loads((profile / ".readiness" / "issues.json").read_text())
    assert persisted[0]["check"] == "profile_home"
    assert persisted[0]["status"] == WARN


def test_missing_assigned_credential_file_is_warned(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    config = {
        "agent_readiness": {
            "credential_files": [
                {"name": "CodeWalnut Google Workspace", "path": "google_token.json"}
            ]
        }
    }

    result = check_agent_readiness(
        profile_name="jeeves",
        profile_dir=profile,
        runtime_home=profile,
        config=config,
        persist=False,
    )

    assert result.status == WARN
    assert result.issues[0].check == "credential:CodeWalnut Google Workspace"
    assert result.issues[0].repairable is False


def test_repair_recreates_safe_profile_paths_and_logs(tmp_path: Path) -> None:
    profile = _profile(tmp_path)
    missing = profile / "cron"
    missing.rmdir()

    result = check_agent_readiness(
        profile_name="jeeves",
        profile_dir=profile,
        runtime_home=profile,
        repair=True,
        persist=True,
    )

    assert result.status == READY
    assert missing.is_dir()
    assert result.issues == []
    log_text = (profile / ".readiness" / "repair.log.jsonl").read_text(encoding="utf-8")
    assert '"check": "path_exists:cron"' in log_text
    assert f'"status": "{FIXED}"' in log_text


def test_profile_create_runs_commissioning_readiness(monkeypatch, tmp_path: Path) -> None:
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))

    profile = create_profile("field-agent", no_alias=True, no_skills=True)

    assert profile == home / "profiles" / "field-agent"
    issues_path = profile / ".readiness" / "issues.json"
    assert issues_path.exists()
    assert json.loads(issues_path.read_text(encoding="utf-8")) == []
    assert (profile / ".env").exists()
