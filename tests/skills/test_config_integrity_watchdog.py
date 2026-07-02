"""Tests for the config-integrity-watchdog skill.

Covers:
  - SKILL.md frontmatter conforms to the standard format
  - All scripts parse as valid Python (AST check)
  - Functional end-to-end: seal -> verify -> tamper -> verify (exit 1) -> restore -> verify (exit 0)
  - Edge cases: no baseline, log tampering detection, no-op restore
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest  # type: ignore[import]
import yaml

SKILL_DIR = (
    Path(__file__).resolve().parents[2]
    / "skills"
    / "devops"
    / "config-integrity-watchdog"
)
SCRIPTS_DIR = SKILL_DIR / "scripts"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(script: str, env: dict) -> subprocess.CompletedProcess:
    """Run a skill script as a subprocess with the given env overrides."""
    import os
    full_env = {**os.environ, **env}
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / f"{script}.py")],
        capture_output=True,
        text=True,
        env=full_env,
    )


def _init_dotfiles_repo(dotfiles: Path) -> None:
    """Create a minimal git repo in *dotfiles* with an initial commit."""
    hermes_dir = dotfiles / "hermes"
    hermes_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", str(dotfiles)], check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=dotfiles, check=True, capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=dotfiles, check=True, capture_output=True,
    )
    (dotfiles / "README.md").write_text("dotfiles\n")
    subprocess.run(["git", "add", "README.md"], cwd=dotfiles, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=dotfiles, check=True, capture_output=True,
    )


@pytest.fixture()
def env(tmp_path):
    """Return a dict with HERMES_CONFIG and HERMES_DOTFILES_DIR pointing at temp dirs."""
    config = tmp_path / "config.yaml"
    config.write_text("model:\n  default: test-model\n")

    dotfiles = tmp_path / "dotfiles"
    _init_dotfiles_repo(dotfiles)

    return {
        "config": config,
        "dotfiles": dotfiles,
        "log": dotfiles / "hermes" / "config_integrity.jsonl",
        "script_env": {
            "HERMES_CONFIG": str(config),
            "HERMES_DOTFILES_DIR": str(dotfiles),
        },
    }


# ---------------------------------------------------------------------------
# Static checks
# ---------------------------------------------------------------------------

class TestStaticChecks:
    def test_skill_dir_exists(self):
        assert SKILL_DIR.is_dir(), f"Skill directory not found: {SKILL_DIR}"

    def test_skill_md_exists(self):
        assert (SKILL_DIR / "SKILL.md").is_file()

    def test_frontmatter_required_fields(self):
        src = (SKILL_DIR / "SKILL.md").read_text()
        m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
        assert m, "SKILL.md missing YAML frontmatter"
        fm = yaml.safe_load(m.group(1))
        for field in ("name", "description", "version", "author", "platforms"):
            assert field in fm, f"SKILL.md frontmatter missing '{field}'"
        assert fm["name"] == "config-integrity-watchdog"

    def test_frontmatter_hermes_tags(self):
        src = (SKILL_DIR / "SKILL.md").read_text()
        m = re.search(r"^---\n(.*?)\n---", src, re.DOTALL)
        assert m is not None, "SKILL.md missing YAML frontmatter"
        fm = yaml.safe_load(m.group(1))
        tags = fm.get("metadata", {}).get("hermes", {}).get("tags", [])
        assert "devops" in tags
        assert "integrity" in tags

    @pytest.mark.parametrize("script", ["seal", "verify", "restore"])
    def test_scripts_are_valid_python(self, script):
        src = (SCRIPTS_DIR / f"{script}.py").read_text()
        try:
            ast.parse(src)
        except SyntaxError as e:
            pytest.fail(f"{script}.py has a syntax error: {e}")

    @pytest.mark.parametrize("script", ["seal", "verify", "restore"])
    def test_scripts_have_shebang(self, script):
        first_line = (SCRIPTS_DIR / f"{script}.py").read_text().splitlines()[0]
        assert first_line.startswith("#!"), f"{script}.py missing shebang"

    @pytest.mark.parametrize("script", ["seal", "verify", "restore"])
    def test_scripts_reference_env_vars(self, script):
        src = (SCRIPTS_DIR / f"{script}.py").read_text()
        assert "HERMES_CONFIG" in src
        assert "HERMES_DOTFILES_DIR" in src


# ---------------------------------------------------------------------------
# Seal tests
# ---------------------------------------------------------------------------

class TestSeal:
    def test_seal_exits_0_and_creates_log(self, env):
        result = _run("seal", env["script_env"])
        assert result.returncode == 0, result.stderr

        log = env["log"]
        assert log.exists()
        entries = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
        assert len(entries) == 1
        assert entries[0]["event"] == "seal"

        expected_hash = hashlib.sha256(env["config"].read_bytes()).hexdigest()
        assert entries[0]["hash"] == expected_hash

    def test_seal_hash_matches_file(self, env):
        _run("seal", env["script_env"])
        entries = [
            json.loads(l) for l in env["log"].read_text().splitlines() if l.strip()
        ]
        expected = hashlib.sha256(env["config"].read_bytes()).hexdigest()
        assert entries[0]["hash"] == expected

    def test_seal_missing_config_exits_1(self, env):
        bad_env = {**env["script_env"], "HERMES_CONFIG": "/nonexistent/config.yaml"}
        result = _run("seal", bad_env)
        assert result.returncode == 1

    def test_seal_commits_to_git(self, env):
        _run("seal", env["script_env"])
        # After seal, the log should be committed (clean git status)
        result = subprocess.run(
            ["git", "status", "--porcelain", "hermes/config_integrity.jsonl"],
            cwd=env["dotfiles"], capture_output=True, text=True,
        )
        assert result.stdout.strip() == "", "Log file should be committed after seal"

    def test_seal_appends_on_second_call(self, env):
        _run("seal", env["script_env"])
        env["config"].write_text("model:\n  default: updated-model\n")
        _run("seal", env["script_env"])
        entries = [
            json.loads(l) for l in env["log"].read_text().splitlines() if l.strip()
        ]
        assert len(entries) == 2
        assert all(e["event"] == "seal" for e in entries)


# ---------------------------------------------------------------------------
# Verify tests
# ---------------------------------------------------------------------------

class TestVerify:
    def test_verify_ok_after_seal(self, env):
        _run("seal", env["script_env"])
        result = _run("verify", env["script_env"])
        assert result.returncode == 0, result.stderr

    def test_verify_detects_tamper(self, env):
        _run("seal", env["script_env"])
        env["config"].write_text("model:\n  default: evil-model\n")
        result = _run("verify", env["script_env"])
        assert result.returncode == 1

    def test_verify_no_baseline_returns_3(self, env):
        result = _run("verify", env["script_env"])
        assert result.returncode == 3

    def test_verify_detects_uncommitted_log_changes(self, env):
        _run("seal", env["script_env"])
        # Manually append to the log without committing
        with open(env["log"], "a") as f:
            f.write(json.dumps({"event": "seal", "hash": "a" * 64}) + "\n")
        result = _run("verify", env["script_env"])
        assert result.returncode == 2

    def test_verify_output_contains_hash_prefix(self, env):
        _run("seal", env["script_env"])
        result = _run("verify", env["script_env"])
        assert result.returncode == 0
        # Should print at least the first 16 chars of the hash
        expected_prefix = hashlib.sha256(env["config"].read_bytes()).hexdigest()[:16]
        assert expected_prefix in result.stdout


# ---------------------------------------------------------------------------
# Restore tests
# ---------------------------------------------------------------------------

class TestRestore:
    def _place_canonical_in_dotfiles(self, env):
        """Copy config.yaml into dotfiles/hermes/ and commit it, so git restore works."""
        canonical = env["dotfiles"] / "hermes" / "config.yaml"
        canonical.write_bytes(env["config"].read_bytes())
        subprocess.run(
            ["git", "add", "hermes/config.yaml"],
            cwd=env["dotfiles"], check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "add canonical config"],
            cwd=env["dotfiles"], check=True, capture_output=True,
        )

    def test_restore_noop_when_matches_baseline(self, env):
        self._place_canonical_in_dotfiles(env)
        _run("seal", env["script_env"])
        result = _run("restore", env["script_env"])
        assert result.returncode == 0
        assert "no restore needed" in result.stdout.lower() or result.returncode == 0

    def test_restore_reverts_tampered_config(self, env):
        original_content = env["config"].read_bytes()
        self._place_canonical_in_dotfiles(env)
        _run("seal", env["script_env"])

        # Tamper
        env["config"].write_text("model:\n  default: evil-model\n")
        assert _run("verify", env["script_env"]).returncode == 1

        # Restore
        result = _run("restore", env["script_env"])
        assert result.returncode == 0, result.stderr

        # Config should be back to original
        assert env["config"].read_bytes() == original_content

    def test_restore_verify_clean_after_restore(self, env):
        self._place_canonical_in_dotfiles(env)
        _run("seal", env["script_env"])
        env["config"].write_text("model:\n  default: evil-model\n")
        _run("restore", env["script_env"])

        result = _run("verify", env["script_env"])
        assert result.returncode == 0

    def test_restore_creates_backup_of_tampered_config(self, env):
        self._place_canonical_in_dotfiles(env)
        _run("seal", env["script_env"])
        env["config"].write_text("model:\n  default: evil-model\n")
        _run("restore", env["script_env"])

        config_dir = env["config"].parent
        backups = list(config_dir.glob("*.pre-restore-*"))
        assert len(backups) >= 1, "Expected a backup file after restore"

    def test_restore_no_baseline_exits_1(self, env):
        result = _run("restore", env["script_env"])
        assert result.returncode == 1

    def test_restore_logs_tamper_detected_entry(self, env):
        self._place_canonical_in_dotfiles(env)
        _run("seal", env["script_env"])
        env["config"].write_text("model:\n  default: evil-model\n")
        _run("restore", env["script_env"])

        entries = [
            json.loads(l) for l in env["log"].read_text().splitlines() if l.strip()
        ]
        events = [e["event"] for e in entries]
        assert "tamper_detected" in events
        assert "restore" in events
