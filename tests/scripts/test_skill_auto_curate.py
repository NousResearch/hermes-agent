"""Tests for skill-auto-curate.py."""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

SCRIPT = Path(r"C:\Users\bbask\AppData\Local\hermes\scripts\skill-auto-curate.py")
sys.path.insert(0, str(SCRIPT.parent))
import importlib.util
spec = importlib.util.spec_from_file_location("sac", SCRIPT)
sac = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sac)


# --- parse_issue_count ---

def test_parse_issue_count_finds_number():
    output = "Found 5 issues across 3 skills"
    assert sac.parse_issue_count(output) == 5


def test_parse_issue_count_zero():
    output = "All skills healthy"
    assert sac.parse_issue_count(output) == 0


def test_parse_issue_count_handles_no_match():
    output = "random garbage with no numbers"
    assert sac.parse_issue_count(output) == 0


def test_parse_issue_count_total_issues():
    output = "Audit summary: 12 total issues"
    assert sac.parse_issue_count(output) == 12


# --- parse_skill_names ---

def test_parse_skill_names_extracts_dashed_names():
    output = """- hermes-agent
* verify-before-complete
  pr-reviewer
"""
    names = sac.parse_skill_names(output)
    assert "hermes-agent" in names
    assert "verify-before-complete" in names
    assert "pr-reviewer" in names


def test_parse_skill_names_skips_keywords():
    output = """- skill
- skills
- found
- issue
- issues
- real-skill
"""
    names = sac.parse_skill_names(output)
    # "real-skill" should be in; "skill", "skills", "found", "issue", "issues" should NOT
    assert "real-skill" in names
    assert "skill" not in names
    assert "skills" not in names
    assert "found" not in names


def test_parse_skill_names_caps_at_20():
    output = "\n".join(f"- skill-{i}" for i in range(30))
    names = sac.parse_skill_names(output)
    assert len(names) == 20


def test_parse_skill_names_handles_empty():
    assert sac.parse_skill_names("") == []


# --- run_hygiene_audit ---

def test_run_hygiene_audit_success(tmp_path, monkeypatch):
    """When the script exists and runs successfully, return (0, output)."""
    audit_script = tmp_path / "skills-hygiene-audit.py"
    audit_script.write_text("# stub")
    monkeypatch.setattr(sac, "HYGIENE_SCRIPT", audit_script)
    with patch.object(sac.subprocess, "run") as mock_run:
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout="Found 5 issues",
            stderr="",
        )
        rc, output = sac.run_hygiene_audit()
    assert rc == 0
    assert "5 issues" in output


def test_run_hygiene_audit_missing_script(tmp_path, monkeypatch):
    monkeypatch.setattr(sac, "HYGIENE_SCRIPT", tmp_path / "missing.py")
    rc, output = sac.run_hygiene_audit()
    assert rc == 1
    assert "missing" in output


# --- main flow ---

def test_main_no_issues_exits_0(tmp_path, monkeypatch):
    monkeypatch.setattr(sac, "LOG_FILE", tmp_path / "log.txt")
    monkeypatch.setattr(sac, "HYGIENE_SCRIPT", tmp_path / "audit.py")
    with patch.object(sac, "run_hygiene_audit", return_value=(0, "All skills healthy")):
        with patch.object(sac, "send_telegram", return_value=True):
            r = sac.main()
    assert r == 0
    log_text = (tmp_path / "log.txt").read_text()
    assert "no issues" in log_text


def test_main_with_issues_alerts(tmp_path, monkeypatch):
    """When audit returns issues (rc=1 with issue count in output), alert."""
    monkeypatch.setattr(sac, "LOG_FILE", tmp_path / "log.txt")
    monkeypatch.setattr(sac, "HYGIENE_SCRIPT", tmp_path / "audit.py")
    output = """Found 5 issues across 3 skills
- hermes-agent
- verify-before-complete
- pr-reviewer
"""
    # Audit script returns rc=1 when issues are found
    with patch.object(sac, "run_hygiene_audit", return_value=(1, output)):
        with patch.object(sac, "send_telegram", return_value=True) as mock_tg:
            r = sac.main()
    assert r == 0  # not a failure to auto-curate, just issues to alert on
    assert mock_tg.called
    msg = mock_tg.call_args[0][0]
    assert "5 issue" in msg
    assert "hermes-agent" in msg


def test_main_audit_crash_alerts(tmp_path, monkeypatch):
    """When audit returns rc=1 with traceback in output, alert failure."""
    monkeypatch.setattr(sac, "LOG_FILE", tmp_path / "log.txt")
    monkeypatch.setattr(sac, "HYGIENE_SCRIPT", tmp_path / "audit.py")
    output = "Traceback (most recent call last):\n  File x, line 1\nRuntimeError: bad"
    with patch.object(sac, "run_hygiene_audit", return_value=(1, output)):
        with patch.object(sac, "send_telegram", return_value=True) as mock_tg:
            r = sac.main()
    assert r == 1
    assert mock_tg.called
    msg = mock_tg.call_args[0][0]
    assert "FAILED" in msg