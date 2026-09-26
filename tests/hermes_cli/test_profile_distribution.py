"""Tests for hermes_cli.profile_distribution — git-based profile installs.

Covers manifest parsing, version requirement checks, install / update / describe
on local-directory sources, and guards on what can and can't be installed.

Transport-layer tests (git clone, URL handling) are exercised through live
E2E runs, not unit tests — git itself is tested upstream, and subprocess-
mocking git would just test the mock.
"""

from __future__ import annotations

import json
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli.profile_distribution import (
    DEFAULT_DIST_OWNED,
    DistributionError,
    DistributionManifest,
    EnvRequirement,
    MANIFEST_FILENAME,
    _env_template_from_manifest,
    _looks_like_git_url,
    _parse_semver,
    _stage_source,
    check_hermes_requires,
    describe_distribution,
    install_distribution,
    plan_install,
    read_manifest,
    update_distribution,
    write_manifest,
)


# ---------------------------------------------------------------------------
# Isolated profile env (matches tests/hermes_cli/test_profiles.py)
# ---------------------------------------------------------------------------


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir(exist_ok=True)
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


def _make_staging_dir(root: Path, name: str = "src", *, manifest: DistributionManifest = None) -> Path:
    """Build a local distribution staging directory (what a git clone would
    contain after .git is removed).

    Lays down a minimal but representative tree: SOUL.md, config.yaml,
    mcp.json, one skill, one cron job, plus the distribution.yaml manifest.
    """
    staged = root / f"staging_{name}"
    staged.mkdir(parents=True, exist_ok=True)
    (staged / "SOUL.md").write_text("I am Source.\n")
    (staged / "config.yaml").write_text("model:\n  model: gpt-4\n")
    (staged / "mcp.json").write_text('{"servers": {}}\n')
    (staged / "skills").mkdir(exist_ok=True)
    (staged / "skills" / "demo").mkdir(exist_ok=True)
    (staged / "skills" / "demo" / "SKILL.md").write_text(
        "---\nname: demo\ndescription: test\n---\n# Demo skill\n"
    )
    from cron.jobs import create_job, use_cron_store
    with use_cron_store(staged):
        create_job("daily task", "0 9 * * *", name="daily")

    mf = manifest or DistributionManifest(name=name, version="0.1.0")
    write_manifest(staged, mf)
    return staged


def _symlink_file_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable in test environment: {exc}")


# ===========================================================================
# Manifest parsing
# ===========================================================================


class TestManifestParsing:


    def test_full_manifest(self, tmp_path):
        (tmp_path / MANIFEST_FILENAME).write_text(
            "name: telem\n"
            "version: 1.2.3\n"
            "description: Telem monitor\n"
            "hermes_requires: '>=0.12.0'\n"
            "author: Kyle\n"
            "license: MIT\n"
            "env_requires:\n"
            "  - name: OPENAI_API_KEY\n"
            "    description: OpenAI key\n"
            "  - name: GRAPH_URL\n"
            "    required: false\n"
            "    default: http://127.0.0.1:8000\n"
            "distribution_owned:\n"
            "  - SOUL.md\n"
            "  - skills/\n"
        )
        m = read_manifest(tmp_path)
        assert m.name == "telem"
        assert m.version == "1.2.3"
        assert m.author == "Kyle"
        assert m.license == "MIT"
        assert len(m.env_requires) == 2
        assert m.env_requires[0].name == "OPENAI_API_KEY"
        assert m.env_requires[0].required is True
        assert m.env_requires[1].required is False
        assert m.env_requires[1].default == "http://127.0.0.1:8000"
        assert m.distribution_owned == ["SOUL.md", "skills"]






    def test_roundtrip_write_read(self, tmp_path):
        original = DistributionManifest(
            name="rt",
            version="1.0.0",
            description="roundtrip",
            env_requires=[EnvRequirement(name="FOO", description="foo")],
        )
        write_manifest(tmp_path, original)
        parsed = read_manifest(tmp_path)
        assert parsed.name == "rt"
        assert parsed.env_requires[0].name == "FOO"


# ===========================================================================
# Version requirement checks
# ===========================================================================


class TestVersionRequires:

    @pytest.mark.parametrize("spec,cur,ok", [
        ("", "0.1.0", True),
        (">=0.12.0", "0.12.0", True),
        (">=0.12.0", "0.13.0", True),
        (">=0.12.0", "0.11.9", False),
        ("==0.12.0", "0.12.0", True),
        ("==0.12.0", "0.13.0", False),
        ("!=0.12.0", "0.13.0", True),
        (">0.12.0", "0.12.1", True),
        (">0.12.0", "0.12.0", False),
        ("<0.13.0", "0.12.9", True),
        ("<=0.12.0", "0.12.0", True),
        ("0.12.0", "0.13.0", True),     # Bare = >=
        ("0.12.0", "0.11.0", False),    # Bare = >=
    ])
    def test_check_matrix(self, spec, cur, ok):
        if ok:
            check_hermes_requires(spec, cur)
        else:
            with pytest.raises(DistributionError, match="requires Hermes"):
                check_hermes_requires(spec, cur)

    def test_parse_semver_handles_prerelease(self):
        assert _parse_semver("0.12.0-rc1") == (0, 12, 0)
        assert _parse_semver("v0.12.0+abc") == (0, 12, 0)


# ===========================================================================
# Env template
# ===========================================================================


class TestEnvTemplate:

    def test_required_is_uncommented(self):
        m = DistributionManifest(
            name="x",
            env_requires=[EnvRequirement(name="FOO", description="foo key")],
        )
        out = _env_template_from_manifest(m)
        assert "# foo key" in out
        assert "# (required)" in out
        assert "FOO=" in out
        # No leading `# ` before FOO=
        assert "\nFOO=" in out or out.startswith("FOO=") or "\nFOO=\n" in out or "FOO=\n" in out


# ===========================================================================
# Source URL detection
# ===========================================================================


class TestLooksLikeGitUrl:

    @pytest.mark.parametrize("src", [
        "github.com/user/repo",
        "https://github.com/user/repo",
        "https://github.com/user/repo.git",
        "http://example.com/repo",
        "git@github.com:user/repo.git",
        "ssh://git@example.com/repo.git",
        "git://example.com/repo.git",
    ])
    def test_accepts_git_sources(self, src):
        assert _looks_like_git_url(src)

    @pytest.mark.platforms("windows")
    def test_git_source_removes_read_only_git_metadata(self, tmp_path, monkeypatch):
        origin = tmp_path / "origin"
        subprocess.run(["git", "init", "--quiet", str(origin)], check=True)
        (origin / MANIFEST_FILENAME).write_text("name: demo\nversion: 1.0.0\n", encoding="utf-8")
        subprocess.run(["git", "-C", str(origin), "add", "."], check=True)
        subprocess.run([
            "git", "-C", str(origin), "-c", "user.name=Test", "-c",
            "user.email=test@example.invalid", "commit", "--quiet", "-m", "init",
        ], check=True)
        saw_read_only_objects = []

        def clone_local(_url, dest):
            subprocess.run(["git", "clone", "--quiet", str(origin), str(dest)], check=True)
            objects = [path for path in (dest / ".git" / "objects").rglob("*") if path.is_file()]
            saw_read_only_objects.append(
                bool(objects) and any(
                    path.stat().st_file_attributes & stat.FILE_ATTRIBUTE_READONLY for path in objects
                )
            )

        monkeypatch.setattr("hermes_cli.profile_distribution._git_clone", clone_local)

        workdir = tmp_path / "work"
        workdir.mkdir()
        staged, _provenance = _stage_source("https://example.invalid/demo.git", workdir)

        assert saw_read_only_objects == [True]
        assert not (staged / ".git").exists()


# ===========================================================================
# Install — fresh and force (from a local-directory source)
# ===========================================================================


class TestInstall:

    def test_install_from_directory(self, profile_env):
        staged = _make_staging_dir(profile_env, "src")
        plan = install_distribution(str(staged), name="installed")
        assert plan.target_dir.is_dir()
        assert (plan.target_dir / "SOUL.md").read_text() == "I am Source.\n"
        assert (plan.target_dir / "skills" / "demo" / "SKILL.md").exists()
        assert (plan.target_dir / "mcp.json").exists()
        # Manifest on disk records canonical name + provenance
        m = read_manifest(plan.target_dir)
        assert m.name == "installed"
        assert m.source == str(staged)

    def test_install_respects_distribution_owned_allowlist(self, profile_env):
        """Install must only copy paths listed in distribution_owned."""
        mf = DistributionManifest(
            name="restricted",
            version="0.1.0",
            distribution_owned=["SOUL.md", "skills"],
        )
        staged = _make_staging_dir(profile_env, "restricted", manifest=mf)
        # Confirm extra files exist in staging
        assert (staged / "mcp.json").exists(), "mcp.json should exist in staged for this test"
        assert (staged / "cron").is_dir(), "cron/ should exist in staged for this test"

        plan = install_distribution(str(staged), name="restricted")
        # Owned paths must be present
        assert (plan.target_dir / "SOUL.md").read_text() == "I am Source.\n"
        assert (plan.target_dir / "skills").is_dir()
        assert (plan.target_dir / "skills" / "demo" / "SKILL.md").exists()
        # NOT-owned paths must NOT be copied from staging
        assert not (plan.target_dir / "mcp.json").exists(), \
            "mcp.json should NOT be copied (not in distribution_owned)"
        # cron/ is created by _bootstrap_user_dirs, but the staged cron/ content
        # must NOT leak through
        if (plan.target_dir / "cron").exists():
            cron_content = list((plan.target_dir / "cron").iterdir())
            assert not cron_content, \
                f"cron/ should be empty (staged content skipped): {cron_content}"
        # distribution.yaml is always written by write_manifest
        assert (plan.target_dir / "distribution.yaml").exists()


    def test_install_omitted_allowlist_copies_everything(self, profile_env):
        """Legacy contract: when distribution_owned is OMITTED, every staged
        entry outside USER_OWNED_EXCLUDE is copied — the omitted list must NOT
        silently narrow to DEFAULT_DIST_OWNED."""
        staged = _make_staging_dir(profile_env, "legacy_all")
        # Extra top-level payload not covered by DEFAULT_DIST_OWNED
        (staged / "extra.txt").write_text("bonus\n")
        (staged / "assets").mkdir()
        (staged / "assets" / "helper.py").write_text("# helper\n")

        plan = install_distribution(str(staged), name="legacy_all")
        assert (plan.target_dir / "extra.txt").read_text() == "bonus\n", \
            "omitted distribution_owned must keep copying undeclared files"
        assert (plan.target_dir / "assets" / "helper.py").exists(), \
            "omitted distribution_owned must keep copying undeclared dirs"

    def test_install_allowlist_supports_nested_paths(self, profile_env):
        """Nested skill paths and the canonical cron store can be allowlisted exactly."""
        mf = DistributionManifest(
            name="nested",
            version="0.1.0",
            distribution_owned=["SOUL.md", "skills/research/", "cron/jobs.json"],
        )
        staged = _make_staging_dir(profile_env, "nested", manifest=mf)
        (staged / "skills" / "research").mkdir()
        (staged / "skills" / "research" / "SKILL.md").write_text(
            "---\nname: research\ndescription: r\n---\n# R\n"
        )

        plan = install_distribution(str(staged), name="nested")
        assert (plan.target_dir / "skills" / "research" / "SKILL.md").exists()
        assert (plan.target_dir / "cron" / "jobs.json").exists()
        assert not (plan.target_dir / "skills" / "demo").exists(), \
            "skills/demo is not allowlisted and must not be copied"
        assert not (plan.target_dir / "mcp.json").exists()

    def test_update_respects_distribution_owned_allowlist(self, profile_env):
        """Update must only copy paths listed in distribution_owned."""
        # 1. Install with full default distribution_owned
        staged = _make_staging_dir(profile_env, "up_src")
        plan = install_distribution(str(staged), name="up_restricted")
        assert (plan.target_dir / "mcp.json").exists(), "baseline: mcp.json should exist"

        # 2. Write a new manifest with restricted distribution_owned
        restricted_mf = DistributionManifest(
            name="up_restricted",
            version="0.2.0",
            distribution_owned=["SOUL.md", "skills"],
        )
        write_manifest(staged, restricted_mf)
        # Also add a NEW file in the staged dir that is NOT in distribution_owned
        (staged / "new_config.toml").write_text("[extra]\n")
        # The manifest on disk needs the new source to match
        from hermes_cli.profile_distribution import read_manifest as _read
        m_on_disk = _read(plan.target_dir)
        m_on_disk.source = str(staged)
        write_manifest(plan.target_dir, m_on_disk)

        # 3. Update
        update_distribution("up_restricted", force_config=True)

        # 4. Owned paths should be updated
        assert (plan.target_dir / "SOUL.md").read_text() == "I am Source.\n"
        assert (plan.target_dir / "skills").is_dir()
        # 5. Formerly-owned paths (mcp.json, cron/) should NOT be copied on update
        #    Note: mcp.json existed before so it stays (not removed). The guard is
        #    about what gets COPIED, not what's cleaned up.
        assert not (plan.target_dir / "new_config.toml").exists(), \
            "new_config.toml should not be copied (not in distribution_owned)"

    def test_install_pauses_shipped_cron_jobs_and_skips_runtime_state(self, profile_env):
        """Distribution cron definitions are inert until the installer explicitly resumes them."""
        from cron.jobs import get_due_jobs, is_job_runnable, list_jobs, update_job, use_cron_store

        staged = _make_staging_dir(profile_env, "cron_src")
        with use_cron_store(staged):
            shipped = list_jobs(include_disabled=True)[0]
            update_job(shipped["id"], {
                "next_run_at": (datetime.now(timezone.utc) - timedelta(days=2)).isoformat()
            })
        (staged / "cron" / "future-runtime.bin").write_text("runtime", encoding="utf-8")
        (staged / "skills" / ".future-runtime").write_text("runtime", encoding="utf-8")
        (staged / "skills" / "demo" / ".authored-hidden").write_text("keep", encoding="utf-8")

        plan = install_distribution(str(staged), name="cron_paused")

        with use_cron_store(plan.target_dir):
            installed = {job["id"]: job for job in list_jobs(include_disabled=True)}
            due = get_due_jobs()
        job = installed[shipped["id"]]
        assert not is_job_runnable(job)
        assert job["state"] == "paused"
        assert due == []
        assert not (plan.target_dir / "cron" / "future-runtime.bin").exists()
        assert not (plan.target_dir / "skills" / ".future-runtime").exists()
        assert (plan.target_dir / "skills" / "demo" / ".authored-hidden").read_text() == "keep"


    def test_install_rejects_non_distribution_directory(self, profile_env, tmp_path):
        bogus = tmp_path / "bogus_dir"
        bogus.mkdir()
        (bogus / "some_file").write_text("hi")
        with pytest.raises(DistributionError, match="No distribution.yaml"):
            plan_install(str(bogus), tmp_path / "work", override_name="x")


    def test_install_enforces_hermes_requires(self, profile_env, monkeypatch):
        # Pin current Hermes version to something well below the requirement
        monkeypatch.setattr(
            "hermes_cli.version_info.get_version_info",
            lambda: SimpleNamespace(base_version="0.1.0"),
        )

        mf = DistributionManifest(
            name="future",
            version="1.0.0",
            hermes_requires=">=99.0.0",
        )
        staged = _make_staging_dir(profile_env, "future", manifest=mf)
        with pytest.raises(DistributionError, match="requires Hermes"):
            install_distribution(str(staged), name="future")


# ===========================================================================
# Update — preserves user data, preserves config by default
# ===========================================================================


class TestUpdate:

    def test_update_and_force_install_merge_owned_dirs_per_root(self, profile_env):
        """skills/ and cron/ are containers of roots: roots the payload ships are replaced
        wholesale (retired files disappear), roots the user added survive both paths."""
        staged = _make_staging_dir(profile_env, "src")
        plan = install_distribution(str(staged), name="skills_safe")

        custom = plan.target_dir / "skills" / "custom"
        custom.mkdir()
        (custom / "SKILL.md").write_text("custom skill\n", encoding="utf-8")
        (plan.target_dir / "cron" / "mine.json").write_text('{"schedule": "* * * * *"}\n', encoding="utf-8")
        (plan.target_dir / "skills" / "demo" / "stale.txt").write_text("old file\n", encoding="utf-8")
        (staged / "skills" / "demo" / "SKILL.md").write_text("updated demo\n", encoding="utf-8")
        (staged / "skills" / "new").mkdir()
        (staged / "skills" / "new" / "SKILL.md").write_text("new skill\n", encoding="utf-8")
        # Categorised skill (skills/<category>/<skill>): the category is a container too,
        # so a sibling the user added inside it survives (issue #25120's literal repro).
        (staged / "skills" / "devops" / "team-deploy").mkdir(parents=True)
        (staged / "skills" / "devops" / "team-deploy" / "SKILL.md").write_text("team deploy\n", encoding="utf-8")
        mine = plan.target_dir / "skills" / "devops" / "my-custom-skill"
        mine.mkdir(parents=True)
        (mine / "SKILL.md").write_text("my custom skill\n", encoding="utf-8")

        update_distribution("skills_safe")

        assert (custom / "SKILL.md").read_text(encoding="utf-8") == "custom skill\n"
        assert (mine / "SKILL.md").read_text(encoding="utf-8") == "my custom skill\n"
        assert (plan.target_dir / "skills" / "devops" / "team-deploy" / "SKILL.md").exists()
        assert (plan.target_dir / "cron" / "mine.json").read_text(encoding="utf-8") == '{"schedule": "* * * * *"}\n'
        assert (plan.target_dir / "skills" / "demo" / "SKILL.md").read_text(encoding="utf-8") == "updated demo\n"
        assert (plan.target_dir / "skills" / "new" / "SKILL.md").read_text(encoding="utf-8") == "new skill\n"
        assert not (plan.target_dir / "skills" / "demo" / "stale.txt").exists()

        install_distribution(str(staged), name="skills_safe", force=True)

        assert (custom / "SKILL.md").read_text(encoding="utf-8") == "custom skill\n"
        assert (plan.target_dir / "cron" / "mine.json").exists()

    def test_update_merges_cron_jobs_without_losing_local_state(self, profile_env):
        """Updating one shipped definition cannot replace the profile's whole cron store."""
        from cron.jobs import create_job, list_jobs, pause_job, resume_job, update_job, use_cron_store

        staged = _make_staging_dir(profile_env, "cron_update")
        with use_cron_store(staged):
            shipped = list_jobs(include_disabled=True)[0]
        plan = install_distribution(str(staged), name="cron_merge")

        with use_cron_store(plan.target_dir):
            resume_job(shipped["id"])
            mine = create_job("water the plants", "0 9 * * *", name="mine")
            pause_job(mine["id"], reason="my choice")
            old_next_run = {
                job["id"]: job for job in list_jobs(include_disabled=True)
            }[shipped["id"]]["next_run_at"]

        with use_cron_store(staged):
            update_job(shipped["id"], {
                "prompt": "updated upstream definition",
                "schedule": "every 7d",
            })

        update_distribution("cron_merge")

        with use_cron_store(plan.target_dir):
            jobs = {job["id"]: job for job in list_jobs(include_disabled=True)}
        assert mine["id"] in jobs
        assert jobs[mine["id"]]["state"] == "paused"
        assert jobs[mine["id"]]["paused_reason"] == "my choice"
        assert jobs[shipped["id"]]["prompt"] == "updated upstream definition"
        assert jobs[shipped["id"]]["schedule"]["minutes"] == 7 * 24 * 60
        assert jobs[shipped["id"]]["next_run_at"] != old_next_run
        assert jobs[shipped["id"]]["enabled"] is True

        # An authored schedule the live job cannot take (past one-shot) is rejected as a
        # job-labelled DistributionError before any other file is replaced.
        (staged / "SOUL.md").write_text("upstream v3\n")
        store = staged / "cron" / "jobs.json"
        data = json.loads(store.read_text(encoding="utf-8"))
        data["jobs"][0]["schedule"] = {"kind": "once", "run_at": "2020-01-01T00:00:00+00:00", "display": "once"}
        store.write_text(json.dumps(data), encoding="utf-8")
        with pytest.raises(DistributionError, match="cron job 'daily'"):
            update_distribution("cron_merge")
        assert (plan.target_dir / "SOUL.md").read_text() != "upstream v3\n"
        with use_cron_store(plan.target_dir):
            assert {j["id"]: j for j in list_jobs(include_disabled=True)}[shipped["id"]]["schedule"] == \
                jobs[shipped["id"]]["schedule"]

        # A corrupt target store surfaces as DistributionError too, not a raw RuntimeError.
        (plan.target_dir / "cron" / "jobs.json").write_text("42", encoding="utf-8")
        with pytest.raises(DistributionError, match="Could not merge cron jobs"):
            update_distribution("cron_merge")


    def test_update_refuses_symlinked_owned_container(self, profile_env):
        staged = _make_staging_dir(profile_env, "src")
        plan = install_distribution(str(staged), name="link_safe")

        shared = profile_env / "shared-skills"
        (shared / "mine").mkdir(parents=True)
        (shared / "mine" / "SKILL.md").write_text("shared skill\n", encoding="utf-8")
        before = sorted((p.relative_to(shared), p.read_bytes()) for p in shared.rglob("*") if p.is_file())

        # A symlinked category (skills/devops -> shared dir) is a container too and is refused
        # rather than unlinked and replaced by the shipped copy.
        (staged / "skills" / "devops" / "team-deploy").mkdir(parents=True)
        (staged / "skills" / "devops" / "team-deploy" / "SKILL.md").write_text("team deploy\n", encoding="utf-8")
        _symlink_file_or_skip(plan.target_dir / "skills" / "devops", shared)
        with pytest.raises(DistributionError, match="symlink"):
            update_distribution("link_safe")
        assert (plan.target_dir / "skills" / "devops").is_symlink()
        (plan.target_dir / "skills" / "devops").unlink()

        skills = plan.target_dir / "skills"
        shutil.rmtree(skills)
        _symlink_file_or_skip(skills, shared)
        (staged / "skills" / "demo" / "SKILL.md").write_text("updated demo\n", encoding="utf-8")
        # Every other shipped entry changes upstream too: the refusal must fire before the
        # first write, or the profile is left half-updated and every retry fails the same way.
        (staged / "SOUL.md").write_text("updated soul\n", encoding="utf-8")
        (staged / "mcp.json").write_text('{"servers": {"new": {}}}\n', encoding="utf-8")
        (staged / "cron" / "daily.json").write_text('{"schedule": "0 10 * * *"}', encoding="utf-8")
        untouched = {p: p.read_bytes() for p in plan.target_dir.rglob("*") if p.is_file()}

        with pytest.raises(DistributionError, match="symlink"):
            update_distribution("link_safe")

        assert skills.is_symlink() and skills.resolve() == shared.resolve()
        after = sorted((p.relative_to(shared), p.read_bytes()) for p in shared.rglob("*") if p.is_file())
        assert after == before
        assert {p: p.read_bytes() for p in plan.target_dir.rglob("*") if p.is_file()} == untouched

    def test_update_preserves_user_data(self, profile_env):
        # 1. Build staging dir, install
        staged = _make_staging_dir(profile_env, "src")
        plan = install_distribution(str(staged), name="telem")

        # 2. Add user-owned data to the installed profile
        (plan.target_dir / "memories").mkdir(exist_ok=True)
        (plan.target_dir / "memories" / "MEMORY.md").write_text("# USER MEMORY\n")
        (plan.target_dir / ".env").write_text("OPENAI_API_KEY=sk-user\n")
        (plan.target_dir / "auth.json").write_text('{"user": "auth"}')
        (plan.target_dir / "sessions").mkdir(exist_ok=True)
        (plan.target_dir / "sessions" / "chat.json").write_text('{"s": 1}')

        # 3. Bump source in the staging dir
        (staged / "SOUL.md").write_text("I am Source v2.\n")

        # 4. Update
        update_distribution("telem", force_config=False)

        # 5. Dist-owned changed
        assert (plan.target_dir / "SOUL.md").read_text() == "I am Source v2.\n"
        # 6. User-owned preserved
        assert (plan.target_dir / "memories" / "MEMORY.md").read_text() == "# USER MEMORY\n"
        assert (plan.target_dir / ".env").read_text() == "OPENAI_API_KEY=sk-user\n"
        assert (plan.target_dir / "auth.json").read_text() == '{"user": "auth"}'
        assert (plan.target_dir / "sessions" / "chat.json").read_text() == '{"s": 1}'

    def test_update_preserves_config_by_default(self, profile_env):
        staged = _make_staging_dir(profile_env, "src")
        plan = install_distribution(str(staged), name="t2")

        # User edits config
        (plan.target_dir / "config.yaml").write_text(
            "model:\n  model: gpt-5\n# user override\n"
        )

        # Bump source config
        (staged / "config.yaml").write_text("model:\n  model: claude\n")

        update_distribution("t2", force_config=False)
        assert "gpt-5" in (plan.target_dir / "config.yaml").read_text()
        assert "user override" in (plan.target_dir / "config.yaml").read_text()


    def test_update_missing_manifest_errors(self, profile_env):
        # Make a profile without a manifest; update must refuse
        from hermes_cli.profiles import create_profile
        create_profile(name="plain", no_alias=True)
        with pytest.raises(DistributionError, match="not a distribution"):
            update_distribution("plain")


# ===========================================================================
# describe_distribution — info subcommand
# ===========================================================================


class TestDescribe:

    def test_describe_existing_distribution(self, profile_env):
        mf = DistributionManifest(
            name="telem",
            version="1.0.0",
            description="compliance monitor",
            env_requires=[EnvRequirement(name="API", description="api key")],
        )
        staged = _make_staging_dir(profile_env, "telem", manifest=mf)
        install_distribution(str(staged), name="telem")
        data = describe_distribution("telem")
        assert data["name"] == "telem"
        assert data["version"] == "1.0.0"
        assert data["env_requires"][0]["name"] == "API"


    def test_describe_missing_profile_raises(self, profile_env):
        with pytest.raises(DistributionError, match="No profile named .*hermes profile list"):
            describe_distribution("nonexistent")


# ===========================================================================
# Security — USER_OWNED_EXCLUDE covers the right paths
# ===========================================================================


class TestSecurity:


    def test_install_does_not_import_credentials_from_staging(self, profile_env):
        """If an author accidentally ships auth.json or .env in their
        staging dir, the installer must NOT copy them to the target profile."""
        staged = _make_staging_dir(profile_env, "src")
        # Author leaks credentials into the staging tree (shouldn't happen, but...)
        (staged / "auth.json").write_text('{"leaked": true}')
        (staged / ".env").write_text("LEAKED=1")

        plan = install_distribution(str(staged), name="clean")
        assert not (plan.target_dir / "auth.json").exists(), "auth.json leaked"
        # Fresh profile may have its own .env via the bootstrap; what we care
        # about is that the leaked content didn't land in the target.
        if (plan.target_dir / ".env").exists():
            assert "LEAKED" not in (plan.target_dir / ".env").read_text()

    def test_install_rejects_symlinked_distribution_files(self, profile_env, tmp_path):
        """Distribution install must not follow symlinks to local files."""
        staged = _make_staging_dir(profile_env, "src")
        local_secret = tmp_path / "local-secret.txt"
        local_secret.write_text("outside secret\n")
        _symlink_file_or_skip(
            staged / "skills" / "demo" / "leak.txt",
            local_secret,
        )

        with pytest.raises(DistributionError, match="symlink"):
            install_distribution(str(staged), name="clean")

        from hermes_cli.profiles import get_profile_dir
        target = get_profile_dir("clean")
        assert not (target / "skills" / "demo" / "leak.txt").exists()


# ===========================================================================
# Nested directories whose names match USER_OWNED_EXCLUDE must survive install
# ===========================================================================


class TestNestedUserOwnedExcludeNotFiltered:

    def test_nested_bin_dir_is_preserved(self, profile_env):
        """A distribution shipping assets/bin/ must not have assets/bin/ dropped
        during install even though 'bin' is in USER_OWNED_EXCLUDE."""
        mf = DistributionManifest(
            name="nested_bin",
            version="0.1.0",
            distribution_owned=list(DEFAULT_DIST_OWNED) + ["assets"],
        )
        staged = _make_staging_dir(profile_env, "src", manifest=mf)
        (staged / "assets" / "bin").mkdir(parents=True)
        (staged / "assets" / "bin" / "tool.py").write_text("# tool\n")

        plan = install_distribution(str(staged), name="nested_bin")
        assert (plan.target_dir / "assets" / "bin").is_dir(), "nested bin/ was dropped"
        assert (plan.target_dir / "assets" / "bin" / "tool.py").exists()

    def test_top_level_user_owned_still_skipped(self, profile_env):
        """Top-level entries in USER_OWNED_EXCLUDE must still be skipped —
        only nested (deeper) directories should be preserved.

        Note: _bootstrap_user_dirs creates some of these (logs/, sessions/,
        memories/) in every fresh profile, so we check that the *staged content*
        did not leak through rather than asserting the directory doesn't exist."""
        staged = _make_staging_dir(profile_env, "src")
        # Add top-level excluded entries alongside the legit ones
        (staged / "bin").mkdir(exist_ok=True)
        (staged / "bin" / "shipped_binary").write_text("x")
        (staged / "logs").mkdir(exist_ok=True)
        (staged / "logs" / "shipped.log").write_text("y\n")

        plan = install_distribution(str(staged), name="top_filter")
        # bin/ is not created by _bootstrap_user_dirs so absence means filtered
        assert not (plan.target_dir / "bin").exists(), "top-level bin/ should be filtered"
        # logs/ is created by _bootstrap_user_dirs even on a clean profile,
        # so check that the staged file did NOT land there.
        assert not (plan.target_dir / "logs" / "shipped.log").exists(), \
            "staged logs/ content should not leak into target"

    def test_both_nested_and_top_level_coexist(self, profile_env):
        """Top-level bin/ filtered, but assets/bin/ kept."""
        mf = DistributionManifest(
            name="coexist",
            version="0.1.0",
            distribution_owned=list(DEFAULT_DIST_OWNED) + ["assets"],
        )
        staged = _make_staging_dir(profile_env, "src", manifest=mf)
        (staged / "bin").mkdir(exist_ok=True)
        (staged / "bin" / "top.sh").write_text("# top\n")
        (staged / "assets" / "bin").mkdir(parents=True)
        (staged / "assets" / "bin" / "helper.py").write_text("# helper\n")

        plan = install_distribution(str(staged), name="coexist")
        assert not (plan.target_dir / "bin").exists()
        assert (plan.target_dir / "assets" / "bin" / "helper.py").exists()



# ===========================================================================
# Install-time metadata (installed_at stamp)
# ===========================================================================


class TestInstalledAtStamp:

    def test_install_stamps_installed_at(self, profile_env):
        staged = _make_staging_dir(profile_env, "src")
        plan = install_distribution(str(staged), name="stamped")
        mf = read_manifest(plan.target_dir)
        assert mf.installed_at, "installed_at should be set after install"
        # ISO-8601 UTC sanity: starts with 4-digit year, contains 'T', ends with '+00:00'.
        assert mf.installed_at[:4].isdigit()
        assert "T" in mf.installed_at
        assert mf.installed_at.endswith("+00:00")

    def test_update_refreshes_installed_at(self, profile_env, monkeypatch):
        staged = _make_staging_dir(profile_env, "src")
        install_distribution(str(staged), name="demo")
        from hermes_cli.profiles import get_profile_dir
        first = read_manifest(get_profile_dir("demo")).installed_at

        # Freeze `datetime.now()` to a fixed future time so we can observe that
        # update writes a NEW stamp (installs within the same second otherwise
        # collide at iso-8601 seconds resolution).
        import datetime as _dt
        class _FakeDT(_dt.datetime):
            @classmethod
            def now(cls, tz=None):
                return _dt.datetime(2099, 1, 1, 0, 0, 0, tzinfo=tz or _dt.timezone.utc)
        monkeypatch.setattr(
            "hermes_cli.profile_distribution.datetime", _FakeDT, raising=True
        )

        from hermes_cli.profile_distribution import update_distribution
        update_distribution("demo")
        refreshed = read_manifest(get_profile_dir("demo")).installed_at
        assert refreshed != first, "installed_at should change on update"
        assert refreshed.startswith("2099-01-01"), refreshed


# ===========================================================================
# ProfileInfo exposes distribution metadata
# ===========================================================================


class TestProfileInfoDistribution:

    def test_installed_distribution_shows_in_list(self, profile_env):
        staged = _make_staging_dir(
            profile_env, "src",
            manifest=DistributionManifest(name="telem", version="1.2.3"),
        )
        install_distribution(str(staged), name="telem")

        from hermes_cli.profiles import list_profiles
        rows = {p.name: p for p in list_profiles()}
        assert "telem" in rows
        row = rows["telem"]
        assert row.distribution_name == "telem"
        assert row.distribution_version == "1.2.3"
        assert row.distribution_source  # path populated, exact value depends on fixture


    def test_malformed_manifest_does_not_break_list(self, profile_env):
        from hermes_cli.profiles import create_profile, list_profiles, get_profile_dir
        create_profile(name="brokenmeta", no_alias=True)
        # Write a distribution.yaml that isn't a valid mapping
        (get_profile_dir("brokenmeta") / "distribution.yaml").write_text(
            "not: [a, valid, mapping\n"  # broken YAML
        )
        # list_profiles must NOT raise; distribution_* stay None for this row.
        rows = {p.name: p for p in list_profiles()}
        assert rows["brokenmeta"].distribution_name is None


# ===========================================================================
# Error surfaces: validation failures should propagate as DistributionError
# or ValueError (both caught and rendered cleanly by the CLI handler)
# ===========================================================================


class TestErrorSurfaces:

    def test_bad_profile_name_raises_valueerror_not_traceback(self, profile_env, tmp_path):
        """A manifest whose 'name' can't be used as a profile identifier
        should raise ValueError from validate_profile_name — the CLI handler
        catches both DistributionError and ValueError so users see a clean
        'Error: ...' line instead of a Python traceback.
        """
        mf = DistributionManifest(name="Invalid Name With Spaces", version="0.1.0")
        staged = _make_staging_dir(profile_env, "bad", manifest=mf)
        with pytest.raises((ValueError, DistributionError)):
            plan_install(str(staged), tmp_path / "work")


# ===========================================================================
# Crash durability: write_manifest rewrites distribution.yaml in place
# ===========================================================================


class TestManifestCrashDurability:
    """``write_manifest`` runs on every install and update of a shared profile.

    ``read_manifest`` reports a missing-or-unparseable manifest as "this isn't
    a distribution", so a truncated distribution.yaml silently demotes the
    profile — update tracking and ``env_requires`` just stop existing, with no
    error surfaced anywhere.
    """

    def test_previous_manifest_survives_an_interrupted_write(self, tmp_path):
        import os

        original = DistributionManifest(
            name="keepme",
            version="1.0.0",
            description="the manifest already on disk",
            env_requires=[EnvRequirement(name="FOO", description="foo")],
        )
        write_manifest(tmp_path, original)
        on_disk = (tmp_path / "distribution.yaml").read_bytes()

        def boom(fd):
            raise OSError("simulated crash mid-write")

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(os, "fsync", boom)
            with pytest.raises(OSError):
                write_manifest(
                    tmp_path,
                    DistributionManifest(name="replacement", version="2.0.0"),
                )

        # The old manifest must still be byte-identical and still parse.
        assert (tmp_path / "distribution.yaml").read_bytes() == on_disk
        parsed = read_manifest(tmp_path)
        assert parsed is not None, "profile silently stopped being a distribution"
        assert parsed.name == "keepme"
        assert parsed.env_requires[0].name == "FOO"

        # No temp file left behind next to the manifest.
        assert list(tmp_path.glob("*.tmp")) == []

    @pytest.mark.platforms("posix")  # POSIX permission bits
    def test_existing_file_mode_is_preserved(self, tmp_path):
        import os
        import stat

        write_manifest(tmp_path, DistributionManifest(name="modes", version="1.0.0"))
        mf = tmp_path / "distribution.yaml"
        os.chmod(mf, 0o644)

        write_manifest(tmp_path, DistributionManifest(name="modes", version="2.0.0"))

        mode = stat.S_IMODE(mf.stat().st_mode)
        assert mode == 0o644, f"mode changed to {oct(mode)}"

    @pytest.mark.platforms("posix")  # POSIX permission bits
    def test_created_file_mode_is_not_tightened(self, tmp_path):
        """A manifest this function *creates* must not land owner-only.

        ``atomic_yaml_write`` only re-applies a mode it captured from an
        existing file, so a fresh distribution.yaml would otherwise keep
        ``tempfile.mkstemp``'s 0600. ``_materialize`` hits that path whenever a
        distribution's explicit ``distribution_owned`` allowlist omits
        distribution.yaml, so the staged copy never lands in the profile.
        """
        import stat

        mf = tmp_path / "distribution.yaml"
        assert not mf.exists()

        write_manifest(tmp_path, DistributionManifest(name="fresh", version="1.0.0"))

        mode = stat.S_IMODE(mf.stat().st_mode)
        assert mode == 0o644, f"new manifest created as {oct(mode)}"
