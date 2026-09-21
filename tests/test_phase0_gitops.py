"""
KENSEI Phase 0 GitOps — unit + integration tests.

Covers:
  - backfill-from-ledger: parses both ledger sections, dedupes, dry-run is safe
  - .gitignore template: forbidden patterns are present
  - watcher: idempotent no-op when no changes
  - pre-commit lint: blocks forbidden paths, blocks malformed YAML, passes clean
  - install.sh: dry-runnable on a throwaway HERMES_HOME

Run with:  pytest tests/test_phase0_gitops.py -v
"""

from __future__ import annotations

import os
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GITOPS = REPO / "scripts" / "gitops"
TEMPLATE = GITOPS / "templates" / ".gitignore.template"
WATCHER = GITOPS / "hermes-auto-commit.sh"
LINT = GITOPS / "pre-commit-lint.sh"
BACKFILL = GITOPS / "backfill-from-ledger.py"
INSTALL = GITOPS / "install.sh"


# ---- Fixtures -------------------------------------------------------------

@pytest.fixture
def tmp_hermes(tmp_path: Path) -> Path:
    """Build a tiny ~/.hermes-shaped sandbox."""
    home = tmp_path / "hermes"
    for sub in ["profiles", "cron", "governance", "scripts", "skills", "runbooks"]:
        (home / sub).mkdir(parents=True)
    (home / "config.yaml").write_text("version: 1\n")
    (home / "profiles" / "kensei").mkdir()
    (home / "profiles" / "kensei" / "config.yaml").write_text("name: kensei\n")
    (home / "governance" / "profile-change-ledger.md").write_text(
        textwrap.dedent(
            """
            # Ledger

            ## Format

            | Date | Profile | Trigger | Change | Follow-up | Outcome |
            |------|---------|---------|--------|-----------|---------|
            | 18/05/26 | Misa-Misa | intake | added toolsets | 01/06/26 | Pending |
            | 17/05/26 | Denji | new | created profile | 31/05/26 | Pending |

            ## Register (chronological, newest first)

            | 03/06/2026 17:06 | remii | dashboard | enabled skill 'dogfood' | pending | |
            """
        ).strip()
    )
    return home


# ---- Unit: watcher script --------------------------------------------------

class TestWatcherScript:
    def test_track_paths_contains_feature_artifacts(self):
        text = WATCHER.read_text()
        assert '"feature-artifacts"' in text, "TRACK_PATHS in hermes-auto-commit.sh missing feature-artifacts"

    def test_track_paths_contains_core_paths(self):
        text = WATCHER.read_text()
        for path in ["config.yaml", "profiles", "cron", "governance", "scripts", "skills", "runbooks"]:
            assert f'"{path}"' in text, f"TRACK_PATHS missing {path!r}"

    def test_unstages_cron_jobs_json(self):
        """Watcher must unstage cron/jobs.json after git add to prevent
        commits on scheduler metadata changes."""
        text = WATCHER.read_text()
        assert "git reset -- cron/jobs.json" in text, (
            "watcher missing `git reset -- cron/jobs.json` — "
            "scheduler metadata (last_run_at, next_run_at, etc.) would cause "
            "a commit every 5 minutes"
        )


# ---- Unit: template --------------------------------------------------------

class TestGitignoreTemplate:
    def test_exists(self):
        assert TEMPLATE.is_file(), f"template missing: {TEMPLATE}"

    def test_uses_default_deny(self):
        text = TEMPLATE.read_text()
        assert "/*" in text, "template missing default-deny at top level"
        # Each curated path must have a `!`-allow line.
        for path in ["config.yaml", "profiles", "cron", "governance", "scripts", "skills", "runbooks", "feature-artifacts"]:
            assert f"!/{path}" in text, f"template missing allow for {path!r}"

    def test_feature_artifacts_md_only(self):
        """feature-artifacts must allow *.md and deny everything else (no secret exposure)."""
        text = TEMPLATE.read_text()
        assert "!/feature-artifacts" in text, "template missing whitelist for feature-artifacts"
        assert "!feature-artifacts/**/*.md" in text, "template missing *.md allow inside feature-artifacts"
        assert "feature-artifacts/**\n" in text, "template missing deny-all inside feature-artifacts"

    def test_blocks_python_junk(self):
        text = TEMPLATE.read_text()
        for pattern in ["**/__pycache__/", "**/*.pyc", "**/*.pyo"]:
            assert pattern in text, f"template missing {pattern!r}"

    def test_blocks_db_artifacts(self):
        text = TEMPLATE.read_text()
        for pattern in ["**/*.db", "**/*.db-wal", "**/*.db-shm", "**/*.sqlite", "**/*.sqlite3"]:
            assert pattern in text, f"template missing db pattern {pattern!r}"

    def test_blocks_locks_and_backups(self):
        text = TEMPLATE.read_text()
        for pattern in ["**/*.lock", "**/*.tmp", "**/*.bak", "**/*.bak-*", "**/*.superseded-*"]:
            assert pattern in text, f"template missing {pattern!r}"

    def test_blocks_cron_output(self):
        text = TEMPLATE.read_text()
        assert "cron/output/" in text, "template missing cron/output/ exclusion"

    def test_always_allows_gitignore(self):
        text = TEMPLATE.read_text()
        assert "!.gitignore" in text, "template must always allow .gitignore"

    def test_blocks_nested_auth_temp_files(self):
        """Nested profiles/**/auth.json.* and auth.tmp.* must be excluded."""
        text = TEMPLATE.read_text()
        for pattern in [
            "**/auth.json",
            "**/auth.lock",
            "**/credentials.json",
            "**/auth.json.*",
            "**/auth.tmp.*",
            "**/auth.lock.*",
            "**/auth.lock.tmp.*",
            "**/credentials.json.*",
            "**/credentials.tmp.*",
        ]:
            assert pattern in text, f"template missing nested auth pattern {pattern!r}"

    def test_blocks_vault_and_private_keys(self):
        """Vault keys and private keys must be excluded."""
        text = TEMPLATE.read_text()
        for pattern in [
            "**/vault.key",
            "**/*.pem",
            "**/id_rsa*",
            "**/id_ed25519*",
            "**/id_ecdsa*",
        ]:
            assert pattern in text, f"template missing key pattern {pattern!r}"

    def test_blocks_cron_jobs_json(self):
        """cron/jobs.json must be excluded (runtime metadata, not config)."""
        text = TEMPLATE.read_text()
        assert "cron/jobs.json" in text, "template missing cron/jobs.json exclusion"


# ---- Unit: backfill parser -------------------------------------------------

class TestBackfillParser:
    def test_parses_main_table(self, tmp_hermes: Path):
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location("backfill", BACKFILL)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["backfill"] = mod
        spec.loader.exec_module(mod)

        entries = mod.parse_ledger(tmp_hermes / "governance" / "profile-change-ledger.md")
        assert len(entries) >= 3, f"expected ≥3 entries, got {len(entries)}"

        # Main table entries
        misa = next((e for e in entries if e.profile == "Misa-Misa"), None)
        assert misa is not None
        assert misa.iso_date == "2026-05-18"
        assert "intake" in misa.trigger.lower() or "intake" in misa.change.lower()

        # Register entry
        remii = next((e for e in entries if e.profile == "remii"), None)
        assert remii is not None
        assert remii.iso_date == "2026-06-03"

    def test_to_iso_validates_dates(self):
        import importlib.util
        import sys

        spec = importlib.util.spec_from_file_location("backfill", BACKFILL)
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules["backfill"] = mod
        spec.loader.exec_module(mod)

        assert mod._to_iso("18/05/26") == "2026-05-18"
        assert mod._to_iso("01/06/2026") == "2026-06-01"
        assert mod._to_iso("nope") is None
        assert mod._to_iso("32/13/26") is None  # invalid day/month


# ---- Integration: install + watcher (sandbox) ------------------------------

@pytest.mark.skipif(not shutil.which("git"), reason="git not available")
class TestInstallSandbox:
    def test_install_initialises_repo(self, tmp_hermes: Path, monkeypatch):
        # Run install.sh against the sandbox, but redirect KENSEI_REPO to a fake
        # that still has the actual scripts (so the install can copy them).
        # We rely on the real source tree; the install copies files INTO the sandbox.
        env = os.environ.copy()
        env["HERMES_HOME"] = str(tmp_hermes)
        env["KENSEI_REPO"] = str(REPO)
        res = subprocess.run(
            ["bash", str(INSTALL)],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert (tmp_hermes / ".git").is_dir(), f"git not initialised: {res.stderr}"
        assert (tmp_hermes / ".gitignore").is_file()
        assert (tmp_hermes / ".git" / "hooks" / "pre-commit").is_file()
        assert (tmp_hermes / "scripts" / "gitops" / "hermes-auto-commit.sh").is_file()
        assert (tmp_hermes / "scripts" / "gitops" / "backfill-from-ledger.py").is_file()
        # Initial commit happened
        log = subprocess.run(
            ["git", "log", "--oneline"], cwd=tmp_hermes, capture_output=True, text=True
        )
        assert "init:" in log.stdout, f"no init commit; log: {log.stdout}"

    def test_install_does_not_call_crontab(self):
        """install.sh must NOT invoke crontab. Hermes cron owns the watcher."""
        text = INSTALL.read_text()
        # The word "crontab" may appear in comments (e.g. "no user-crontab
        # mutation"), but the script must not invoke the crontab command.
        assert "crontab -" not in text, (
            "install.sh must not invoke crontab — Hermes cron owns the watcher"
        )
        assert "crontab -l" not in text, (
            "install.sh must not read crontab — Hermes cron owns the watcher"
        )
        assert "Hermes cron owns the watcher" in text, (
            "install.sh must log that Hermes cron owns the watcher"
        )
        assert "no user-crontab row installed" in text, (
            "install.sh must state no user-crontab row is installed"
        )

    def test_watcher_is_idempotent(self, tmp_hermes: Path, monkeypatch):
        env = os.environ.copy()
        env["HERMES_HOME"] = str(tmp_hermes)
        # First, init the repo and PRE-COMMIT the curated paths so the watcher
        # starts from a clean state. Copy the .gitignore template so the
        # watcher's own state files do not get committed.
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_hermes, check=True)
        subprocess.run(
            ["git", "config", "user.email", "kensei@local"], cwd=tmp_hermes, check=True
        )
        subprocess.run(
            ["git", "config", "user.name", "KENSEI"], cwd=tmp_hermes, check=True
        )
        shutil.copy(TEMPLATE, tmp_hermes / ".gitignore")
        # Stage and commit the seeded curated paths + .gitignore as a baseline.
        subprocess.run(
            ["git", "add", ".gitignore", "config.yaml", "profiles", "cron", "governance", "scripts", "skills", "runbooks"],
            cwd=tmp_hermes,
            check=True,
        )
        subprocess.run(
            ["git", "commit", "-q", "-m", "baseline"],
            cwd=tmp_hermes,
            check=True,
        )
        before = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"], cwd=tmp_hermes, capture_output=True, text=True
        ).stdout.strip()

        # Invoke watcher twice. Both should exit 0 and NOT add new commits.
        for _ in range(2):
            res = subprocess.run(
                ["bash", str(WATCHER)], env=env, capture_output=True, text=True, timeout=30
            )
            assert res.returncode == 0, f"watcher failed: {res.stderr}"
        after = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"], cwd=tmp_hermes, capture_output=True, text=True
        ).stdout.strip()
        assert before == after, f"watcher created commit on no-op: before={before} after={after}"


# ---- Integration: pre-commit lint ------------------------------------------

class TestPreCommitLint:
    def test_blocks_forbidden_paths(self, tmp_hermes: Path):
        # Init repo, install the hook, attempt to add a forbidden path
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_hermes, check=True)
        subprocess.run(
            ["git", "config", "user.email", "kensei@local"], cwd=tmp_hermes, check=True
        )
        subprocess.run(
            ["git", "config", "user.name", "KENSEI"], cwd=tmp_hermes, check=True
        )
        (tmp_hermes / ".git" / "hooks").mkdir(exist_ok=True)
        shutil.copy(LINT, tmp_hermes / ".git" / "hooks" / "pre-commit")
        os.chmod(tmp_hermes / ".git" / "hooks" / "pre-commit", 0o755)

        (tmp_hermes / "auth.json").write_text("{}")
        subprocess.run(["git", "add", "auth.json"], cwd=tmp_hermes, check=True)
        res = subprocess.run(
            ["git", "commit", "-m", "test"], cwd=tmp_hermes, capture_output=True, text=True
        )
        assert res.returncode != 0, "lint should have blocked auth.json"
        assert "forbidden" in res.stderr.lower() or "refusing" in res.stderr.lower()

    def test_blocks_malformed_yaml(self, tmp_hermes: Path):
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_hermes, check=True)
        subprocess.run(
            ["git", "config", "user.email", "kensei@local"], cwd=tmp_hermes, check=True
        )
        subprocess.run(
            ["git", "config", "user.name", "KENSEI"], cwd=tmp_hermes, check=True
        )
        (tmp_hermes / ".git" / "hooks").mkdir(exist_ok=True)
        shutil.copy(LINT, tmp_hermes / ".git" / "hooks" / "pre-commit")
        os.chmod(tmp_hermes / ".git" / "hooks" / "pre-commit", 0o755)

        # Malformed YAML (bad indentation, mixed tabs)
        (tmp_hermes / "bad.yaml").write_text("key: value\n  bad:\n- item1\n - item2\n")
        subprocess.run(["git", "add", "bad.yaml"], cwd=tmp_hermes, check=True)
        res = subprocess.run(
            ["git", "commit", "-m", "test"], cwd=tmp_hermes, capture_output=True, text=True
        )
        assert res.returncode != 0, "lint should have blocked malformed YAML"
        assert "yaml" in res.stderr.lower() or "parse" in res.stderr.lower()

    def test_passes_clean_yaml(self, tmp_hermes: Path):
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_hermes, check=True)
        subprocess.run(
            ["git", "config", "user.email", "kensei@local"], cwd=tmp_hermes, check=True
        )
        subprocess.run(
            ["git", "config", "user.name", "KENSEI"], cwd=tmp_hermes, check=True
        )
        (tmp_hermes / ".git" / "hooks").mkdir(exist_ok=True)
        shutil.copy(LINT, tmp_hermes / ".git" / "hooks" / "pre-commit")
        os.chmod(tmp_hermes / ".git" / "hooks" / "pre-commit", 0o755)

        (tmp_hermes / "good.yaml").write_text("key: value\nlist:\n  - a\n  - b\n")
        subprocess.run(["git", "add", "good.yaml"], cwd=tmp_hermes, check=True)
        res = subprocess.run(
            ["git", "commit", "-m", "test"], cwd=tmp_hermes, capture_output=True, text=True
        )
        assert res.returncode == 0, f"clean YAML should pass: {res.stderr}"

    def test_blocks_nested_auth_temp(self, tmp_hermes: Path):
        """Pre-commit must block nested profiles/**/auth.json.* paths."""
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_hermes, check=True)
        subprocess.run(
            ["git", "config", "user.email", "kensei@local"], cwd=tmp_hermes, check=True
        )
        subprocess.run(
            ["git", "config", "user.name", "KENSEI"], cwd=tmp_hermes, check=True
        )
        (tmp_hermes / ".git" / "hooks").mkdir(exist_ok=True)
        shutil.copy(LINT, tmp_hermes / ".git" / "hooks" / "pre-commit")
        os.chmod(tmp_hermes / ".git" / "hooks" / "pre-commit", 0o755)

        # Simulate a nested auth temp file inside profiles/
        (tmp_hermes / "profiles" / "kensei").mkdir(parents=True, exist_ok=True)
        (tmp_hermes / "profiles" / "kensei" / "auth.json.bak").write_text("{}")
        subprocess.run(
            ["git", "add", "profiles/kensei/auth.json.bak"], cwd=tmp_hermes, check=True
        )
        res = subprocess.run(
            ["git", "commit", "-m", "test"], cwd=tmp_hermes, capture_output=True, text=True
        )
        assert res.returncode != 0, "lint should have blocked nested auth.json.bak"
        assert "forbidden" in res.stderr.lower() or "refusing" in res.stderr.lower()

    def test_blocks_private_key(self, tmp_hermes: Path):
        """Pre-commit must block private key files."""
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_hermes, check=True)
        subprocess.run(
            ["git", "config", "user.email", "kensei@local"], cwd=tmp_hermes, check=True
        )
        subprocess.run(
            ["git", "config", "user.name", "KENSEI"], cwd=tmp_hermes, check=True
        )
        (tmp_hermes / ".git" / "hooks").mkdir(exist_ok=True)
        shutil.copy(LINT, tmp_hermes / ".git" / "hooks" / "pre-commit")
        os.chmod(tmp_hermes / ".git" / "hooks" / "pre-commit", 0o755)

        (tmp_hermes / "id_rsa").write_text("PRIVATE KEY")
        subprocess.run(["git", "add", "id_rsa"], cwd=tmp_hermes, check=True)
        res = subprocess.run(
            ["git", "commit", "-m", "test"], cwd=tmp_hermes, capture_output=True, text=True
        )
        assert res.returncode != 0, "lint should have blocked id_rsa"
        assert "forbidden" in res.stderr.lower() or "refusing" in res.stderr.lower()

    def test_blocks_nested_exact_auth(self, tmp_hermes: Path):
        """Pre-commit must block exact nested auth.json, auth.lock, credentials.json."""
        subprocess.run(["git", "init", "-q", "-b", "main"], cwd=tmp_hermes, check=True)
        subprocess.run(
            ["git", "config", "user.email", "kensei@local"], cwd=tmp_hermes, check=True
        )
        subprocess.run(
            ["git", "config", "user.name", "KENSEI"], cwd=tmp_hermes, check=True
        )
        (tmp_hermes / ".git" / "hooks").mkdir(exist_ok=True)
        shutil.copy(LINT, tmp_hermes / ".git" / "hooks" / "pre-commit")
        os.chmod(tmp_hermes / ".git" / "hooks" / "pre-commit", 0o755)

        # Test nested auth.json (exact, not temp variant)
        (tmp_hermes / "profiles" / "kensei").mkdir(parents=True, exist_ok=True)
        (tmp_hermes / "profiles" / "kensei" / "auth.json").write_text("{}")
        subprocess.run(
            ["git", "add", "profiles/kensei/auth.json"], cwd=tmp_hermes, check=True
        )
        res = subprocess.run(
            ["git", "commit", "-m", "test"], cwd=tmp_hermes, capture_output=True, text=True
        )
        assert res.returncode != 0, "lint should have blocked nested auth.json"
        assert "forbidden" in res.stderr.lower() or "refusing" in res.stderr.lower()
