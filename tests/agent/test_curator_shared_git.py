"""Shared-tree git safety contract + e2e (agent/curator_shared.py).

Covers spec Phase 3 + Invariant 3/AC4/AC5 against a REAL git repo in a temp
HERMES_HOME (real imports, no mocks for the resolution chain):

- git_precheck_shared clean/dirty
- snapshot_shared writes a separate shared tar + manifest with the
  intended-write set recorded BEFORE mutation
- commit_shared: explicit pathspec only; a concurrent NON-lock-taking
  sibling write is never absorbed (drift-abort or pathspec confinement)
- the shared pass end-to-end: split fires with consolidate:false, commit is
  tagged `curator:` and confined to skills-shared/, dirty tree skips with
  zero edits, dry-run makes zero writes and zero commits
- concurrent-write e2e (AC4): a second process writes into skills-shared/
  DURING the pass; the sibling file is absent from the curator commit
- curator-vs-curator: second lock acquisition fails -> skip
- crash recovery: exact-set match restores self-inflicted dirt; a sibling
  (un-manifested) dirty path -> skip-and-report, never clobbered
- archive_shared_skill: in-tree <group>/.archive/<name>/ move
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True,
    )


def _mk_big_skill(root: Path, group: str, name: str, kb: int = 140) -> Path:
    d = root / group / name
    d.mkdir(parents=True, exist_ok=True)
    filler = ("filler content line for the oversized skill body " * 20).strip()
    parts = [f"---\nname: {name}\ndescription: big\n---\n\nIntro.\n\n"]
    i = 0
    while sum(len(p) for p in parts) < kb * 1024:
        parts.append(f"## Section {i:03d}\n\n" + (filler + "\n") * 12 + "\n")
        i += 1
    (d / "SKILL.md").write_text("".join(parts), encoding="utf-8")
    return d


@pytest.fixture
def git_env(tmp_path, monkeypatch):
    """Temp HERMES_HOME that IS a real git repo with a skills-shared/ tree."""
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    shared = home / "skills-shared"
    (shared / "smart-home").mkdir(parents=True)
    (shared / "devops").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    _git(home, "init", "-q")
    _git(home, "config", "user.email", "curator-test@example.com")
    _git(home, "config", "user.name", "Curator Test")

    big = _mk_big_skill(shared, "smart-home", "clanker-e2e")
    small = shared / "devops" / "cron-alert-discipline"
    small.mkdir(parents=True)
    (small / "SKILL.md").write_text(
        "---\nname: cron-alert-discipline\ndescription: small\n---\nsmall\n",
        encoding="utf-8",
    )

    (home / "config.yaml").write_text(textwrap.dedent(f"""\
        skills:
          external_dirs:
            - {shared / 'smart-home'}
            - {shared / 'devops'}
        curator:
          include_shared_dirs: true
          split_over_kb: 100
          consolidate: false
        """), encoding="utf-8")

    _git(home, "add", "-A")
    _git(home, "commit", "-q", "-m", "baseline")

    import agent.skill_utils as su
    importlib.reload(su)
    import agent.curator_shared as cs
    importlib.reload(cs)

    yield {"home": home, "shared": shared, "big": big, "small": small, "cs": cs}


class TestPrecheckAndSnapshot:
    def test_precheck_clean(self, git_env):
        cs = git_env["cs"]
        ok, reason, dirty = cs.git_precheck_shared(git_env["shared"])
        assert ok and reason == "clean" and dirty == []

    def test_precheck_dirty(self, git_env):
        cs = git_env["cs"]
        (git_env["big"] / "SKILL.md").write_text("dirty!", encoding="utf-8")
        ok, reason, dirty = cs.git_precheck_shared(git_env["shared"])
        assert not ok and reason == "dirty working tree"
        assert any("clanker-e2e" in p for p in dirty)

    def test_snapshot_writes_separate_tar_and_manifest(self, git_env):
        cs = git_env["cs"]
        snap = cs.snapshot_shared(
            [git_env["big"]],
            ["skills-shared/smart-home/clanker-e2e/SKILL.md"],
            shared_root=git_env["shared"],
        )
        assert snap is not None
        tars = list(snap.glob("shared-*.tar.gz"))
        assert len(tars) == 1
        manifest = json.loads(
            (snap / cs.SHARED_MANIFEST_NAME).read_text(encoding="utf-8")
        )
        assert manifest["intended_writes"] == [
            "skills-shared/smart-home/clanker-e2e/SKILL.md"
        ]
        assert manifest["baseline_rev"]
        # the pre-mutation SKILL.md is inside the tar
        import tarfile
        with tarfile.open(tars[0]) as tf:
            names = tf.getnames()
        assert any(n.endswith("SKILL.md") for n in names)


class TestCommitShared:
    def test_explicit_pathspec_commit(self, git_env):
        cs = git_env["cs"]
        home = git_env["home"]
        target = git_env["small"] / "SKILL.md"
        target.write_text(target.read_text() + "\nedit\n", encoding="utf-8")
        ok, msg = cs.commit_shared(
            "test edit", [target],
            ["skills-shared/devops/cron-alert-discipline/SKILL.md"],
            shared_root=git_env["shared"],
        )
        assert ok, msg
        log = _git(home, "log", "-1", "--pretty=%s").stdout.strip()
        assert log.startswith("curator: ")
        show = _git(home, "show", "--stat", "--pretty=", "HEAD").stdout
        assert "cron-alert-discipline" in show
        assert "clanker-e2e" not in show

    def test_drift_abort_on_unexpected_sibling_path(self, git_env):
        """A NON-lock-taking sibling write during the pass -> abort, nothing
        staged, sibling file untouched."""
        cs = git_env["cs"]
        home = git_env["home"]
        target = git_env["small"] / "SKILL.md"
        target.write_text(target.read_text() + "\ncurator edit\n",
                          encoding="utf-8")
        # sibling writes DURING the pass (after precheck, before commit)
        sibling = git_env["shared"] / "smart-home" / "sibling-new"
        sibling.mkdir()
        (sibling / "SKILL.md").write_text("sibling work — hands off",
                                          encoding="utf-8")
        ok, msg = cs.commit_shared(
            "test edit", [target], [], shared_root=git_env["shared"],
        )
        assert not ok
        assert "drifted" in msg
        # nothing was committed; sibling intact
        log = _git(home, "log", "-1", "--pretty=%s").stdout.strip()
        assert log == "baseline"
        assert (sibling / "SKILL.md").read_text(encoding="utf-8") == \
            "sibling work — hands off"

    def test_refuses_paths_outside_repo(self, git_env, tmp_path):
        cs = git_env["cs"]
        outside = tmp_path / "outside.txt"
        outside.write_text("x", encoding="utf-8")
        ok, msg = cs.commit_shared(
            "bad", [outside], [], shared_root=git_env["shared"],
        )
        assert not ok and "outside the repo" in msg


class TestSharedPassE2E:
    def _run_pass(self, monkeypatch, dry_run=False, threshold=100,
                  cfg_extra=None):
        import agent.curator as curator
        importlib.reload(curator)
        cfg = {"include_shared_dirs": True, "split_over_kb": threshold}
        cfg.update(cfg_extra or {})
        monkeypatch.setattr(curator, "_load_config", lambda: cfg)
        return curator._run_shared_pass(dry_run, threshold)

    def test_split_fires_with_consolidate_false_and_commits(
        self, git_env, monkeypatch,
    ):
        """AC3+AC4 core: the oversized shared skill splits (consolidate is
        false in config), the commit is tagged curator:, confined to
        skills-shared/, and SKILL.md drops under the cap."""
        report = self._run_pass(monkeypatch)
        assert report["skipped"] is None, report
        assert "clanker-e2e" in report["split"]
        assert report["commit"], report
        home = git_env["home"]
        log = _git(home, "log", "-1", "--pretty=%s").stdout.strip()
        assert log.startswith("curator: split")
        # diff confined to skills-shared/
        files = _git(home, "show", "--name-only", "--pretty=",
                     "HEAD").stdout.strip().splitlines()
        assert files and all(f.startswith("skills-shared/") for f in files)
        post = (git_env["big"] / "SKILL.md").stat().st_size
        assert post < 100 * 1024
        # revertable: git revert restores the original
        r = _git(home, "revert", "--no-edit", "HEAD")
        assert r.returncode == 0, r.stderr
        assert (git_env["big"] / "SKILL.md").stat().st_size > 100 * 1024

    def test_dirty_tree_skips_with_zero_edits(self, git_env, monkeypatch):
        # make the tree dirty with a SIBLING edit (not in any manifest)
        marker = git_env["shared"] / "devops" / "sibling-file.md"
        marker.write_text("sibling", encoding="utf-8")
        before = (git_env["big"] / "SKILL.md").read_bytes()
        report = self._run_pass(monkeypatch)
        assert report["skipped"] is not None
        assert "precheck failed" in report["skipped"]
        assert (git_env["big"] / "SKILL.md").read_bytes() == before
        assert marker.read_text(encoding="utf-8") == "sibling"
        log = _git(git_env["home"], "log", "-1", "--pretty=%s").stdout.strip()
        assert log == "baseline"

    def test_dry_run_zero_writes_zero_commits(self, git_env, monkeypatch):
        before = (git_env["big"] / "SKILL.md").read_bytes()
        report = self._run_pass(monkeypatch, dry_run=True)
        assert report["skipped"] == "dry-run (no mutations)"
        assert any("clanker-e2e" in s for s in report["split"])
        assert (git_env["big"] / "SKILL.md").read_bytes() == before
        st = _git(git_env["home"], "status", "--porcelain").stdout.strip()
        assert st == ""
        log = _git(git_env["home"], "log", "-1", "--pretty=%s").stdout.strip()
        assert log == "baseline"

    def test_concurrent_sibling_write_never_absorbed(self, git_env, monkeypatch):
        """AC4 concurrent-write e2e: a second PROCESS (not taking the lock)
        writes into skills-shared/ DURING the pass. The curator commit must
        not contain the sibling file, and the sibling file must survive
        un-reverted. We interpose on commit_shared to spawn the writer at
        the worst possible moment (post-split, pre-commit)."""
        import agent.curator_shared as cs
        sibling_path = git_env["shared"] / "devops" / "concurrent-sibling.md"
        real_commit = cs.commit_shared

        def commit_with_concurrent_writer(summary, written, precheck=None, **kw):
            # a real second process writes DURING the pass
            code = (
                "import pathlib,sys; "
                f"pathlib.Path({str(sibling_path)!r}).write_text('sibling mid-pass')"
            )
            subprocess.run([sys.executable, "-c", code], check=True)
            return real_commit(summary, written, precheck, **kw)

        monkeypatch.setattr(cs, "commit_shared", commit_with_concurrent_writer)
        report = self._run_pass(monkeypatch)

        # Either the drift-abort fired (commit None) or the pathspec commit
        # excluded the sibling — both satisfy AC4. In BOTH cases the sibling
        # file survives untouched and is absent from any curator commit.
        assert sibling_path.exists()
        assert sibling_path.read_text(encoding="utf-8") == "sibling mid-pass"
        home = git_env["home"]
        if report.get("commit"):
            files = _git(home, "show", "--name-only", "--pretty=",
                         "HEAD").stdout
            assert "concurrent-sibling" not in files
        else:
            assert any("drifted" in e for e in report["errors"]) or \
                report["skipped"]

    def test_second_curator_lock_contention_skips(self, git_env, monkeypatch):
        """Curator-vs-curator: while one process holds the fcntl lock, the
        pass skips with a lock-contention report."""
        import agent.curator_shared as cs
        lock_file = git_env["shared"] / cs.LOCK_NAME
        # hold the lock from a second real process
        holder = subprocess.Popen(
            [sys.executable, "-c", (
                "import fcntl,sys,time; "
                f"f=open({str(lock_file)!r},'a+'); "
                "fcntl.flock(f, fcntl.LOCK_EX); "
                "print('locked',flush=True); time.sleep(20)"
            )],
            stdout=subprocess.PIPE, text=True,
        )
        try:
            assert holder.stdout.readline().strip() == "locked"
            report = self._run_pass(monkeypatch)
            assert report["skipped"] is not None
            assert "lock contention" in report["skipped"]
        finally:
            holder.kill()
            holder.wait()

    def test_snapshot_failure_hard_gates(self, git_env, monkeypatch):
        import agent.curator_shared as cs
        monkeypatch.setattr(cs, "snapshot_shared", lambda *a, **k: None)
        before = (git_env["big"] / "SKILL.md").read_bytes()
        report = self._run_pass(monkeypatch)
        assert report["skipped"] == "shared snapshot failed (hard gate)"
        assert (git_env["big"] / "SKILL.md").read_bytes() == before


class TestCrashRecovery:
    def test_self_inflicted_dirt_restored(self, git_env):
        """Every dirty path manifested as an intended write -> auto-restore."""
        cs = git_env["cs"]
        target = git_env["small"] / "SKILL.md"
        rel = "skills-shared/devops/cron-alert-discipline/SKILL.md"
        snap = cs.snapshot_shared([git_env["small"]], [rel],
                                  shared_root=git_env["shared"])
        assert snap is not None
        original = target.read_text(encoding="utf-8")
        target.write_text("half-applied crash leftovers", encoding="utf-8")
        recovered, why = cs.attempt_crash_recovery(git_env["shared"])
        assert recovered, why
        assert target.read_text(encoding="utf-8") == original
        ok, reason, _ = cs.git_precheck_shared(git_env["shared"])
        assert ok

    def test_unmanifested_sibling_dirt_skips(self, git_env):
        """Any un-manifested dirty path -> skip-and-report, NEVER clobber."""
        cs = git_env["cs"]
        rel = "skills-shared/devops/cron-alert-discipline/SKILL.md"
        snap = cs.snapshot_shared([git_env["small"]], [rel],
                                  shared_root=git_env["shared"])
        assert snap is not None
        # sibling dirt in a path the manifest never recorded
        sibling = git_env["shared"] / "smart-home" / "clanker-e2e" / "SKILL.md"
        sibling_text = sibling.read_text(encoding="utf-8") + "\nsibling edit\n"
        sibling.write_text(sibling_text, encoding="utf-8")
        recovered, why = cs.attempt_crash_recovery(git_env["shared"])
        assert not recovered
        assert "sibling edit present" in why
        assert sibling.read_text(encoding="utf-8") == sibling_text

    def test_superset_dirt_skips(self, git_env):
        """Manifested dirt PLUS an extra un-manifested path -> skip."""
        cs = git_env["cs"]
        target = git_env["small"] / "SKILL.md"
        rel = "skills-shared/devops/cron-alert-discipline/SKILL.md"
        snap = cs.snapshot_shared([git_env["small"]], [rel],
                                  shared_root=git_env["shared"])
        assert snap is not None
        target.write_text("crash leftovers", encoding="utf-8")
        extra = git_env["shared"] / "devops" / "extra-sibling.md"
        extra.write_text("sibling", encoding="utf-8")
        recovered, why = cs.attempt_crash_recovery(git_env["shared"])
        assert not recovered
        assert extra.read_text(encoding="utf-8") == "sibling"

    def test_no_snapshot_skips(self, git_env):
        cs = git_env["cs"]
        (git_env["small"] / "SKILL.md").write_text("dirt", encoding="utf-8")
        recovered, why = cs.attempt_crash_recovery(git_env["shared"])
        assert not recovered
        assert "no shared snapshot" in why


class TestSharedArchive:
    @pytest.mark.live_system_guard_bypass
    def test_archive_moves_in_tree(self, git_env):
        cs = git_env["cs"]
        ok, msg, touched = cs.archive_shared_skill(
            git_env["small"], shared_root=git_env["shared"],
        )
        assert ok, msg
        dest = git_env["shared"] / "devops" / ".archive" / "cron-alert-discipline"
        assert dest.is_dir()
        assert (dest / "SKILL.md").exists()
        assert not git_env["small"].exists()

    def test_archive_rejects_outside_shared(self, git_env, tmp_path):
        cs = git_env["cs"]
        stray = tmp_path / "stray-skill"
        stray.mkdir()
        ok, msg, _ = cs.archive_shared_skill(
            stray, shared_root=git_env["shared"],
        )
        assert not ok

    @pytest.mark.live_system_guard_bypass
    def test_archive_skill_routes_shared_to_in_tree(self, git_env, monkeypatch):
        """tools.skill_usage.archive_skill routes an in-scope shared skill to
        the IN-TREE archive, not the local skills/.archive/."""
        import agent.skill_utils as su
        importlib.reload(su)
        import tools.skill_usage as usage
        importlib.reload(usage)
        ok, msg = usage.archive_skill("cron-alert-discipline")
        assert ok, msg
        dest = git_env["shared"] / "devops" / ".archive" / "cron-alert-discipline"
        assert dest.is_dir()
        local_archive = git_env["home"] / "skills" / ".archive" / "cron-alert-discipline"
        assert not local_archive.exists()

    def test_archive_skill_still_readonly_flag_off(self, git_env, monkeypatch):
        (git_env["home"] / "config.yaml").write_text(
            (git_env["home"] / "config.yaml").read_text().replace(
                "include_shared_dirs: true", "include_shared_dirs: false"
            ),
            encoding="utf-8",
        )
        import agent.skill_utils as su
        importlib.reload(su)
        import tools.skill_usage as usage
        importlib.reload(usage)
        ok, msg = usage.archive_skill("cron-alert-discipline")
        assert not ok
        assert "read-only" in msg
        assert git_env["small"].exists()
