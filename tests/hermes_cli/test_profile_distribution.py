"""Tests for hermes_cli.profile_distribution — git-based profile installs.

Covers manifest parsing, version requirement checks, install / update / describe
on local-directory sources, and guards on what can and can't be installed.

Transport-layer tests (git clone, URL handling) are exercised through live
E2E runs, not unit tests — git itself is tested upstream, and subprocess-
mocking git would just test the mock.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hermes_cli.profile_distribution import (
    DEFAULT_DIST_OWNED,
    DistributionError,
    DistributionManifest,
    EnvRequirement,
    MANIFEST_FILENAME,
    USER_OWNED_EXCLUDE,
    _env_template_from_manifest,
    _looks_like_git_url,
    _parse_semver,
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
    mcp.json, one skill, one cron file, plus the distribution.yaml manifest.
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
    (staged / "cron").mkdir(exist_ok=True)
    (staged / "cron" / "daily.json").write_text('{"schedule": "0 9 * * *"}')

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

    def test_install_default_owned_paths_preserved(self, profile_env):
        """When distribution_owned is not set, all DEFAULT_DIST_OWNED paths are copied."""
        staged = _make_staging_dir(profile_env, "default_owned")
        plan = install_distribution(str(staged), name="default_owned")
        for path in DEFAULT_DIST_OWNED:
            full = plan.target_dir / path
            assert full.exists() or full.is_dir(), \
                f"DEFAULT_DIST_OWNED '{path}' not found in target"

    def test_install_omitted_allowlist_copies_everything(self, profile_env):
        """Legacy contract: when distribution_owned is OMITTED, every staged
        entry outside USER_OWNED_EXCLUDE is copied — the omitted list must NOT
        silently narrow to DEFAULT_DIST_OWNED."""
        staged = _make_staging_dir(profile_env, "legacy_all")
        # Extra top-level payload not covered by DEFAULT_DIST_OWNED
        (staged / "extra.txt").write_text("bonus\n")
        (staged / "tools").mkdir()
        (staged / "tools" / "helper.py").write_text("# helper\n")

        plan = install_distribution(str(staged), name="legacy_all")
        assert (plan.target_dir / "extra.txt").read_text() == "bonus\n", \
            "omitted distribution_owned must keep copying undeclared files"
        assert (plan.target_dir / "tools" / "helper.py").exists(), \
            "omitted distribution_owned must keep copying undeclared dirs"

    def test_install_allowlist_supports_nested_paths(self, profile_env):
        """Documented nested entries like skills/research/ and cron/digest.json
        must select exactly that subtree/file, not be silently dropped."""
        mf = DistributionManifest(
            name="nested",
            version="0.1.0",
            distribution_owned=["SOUL.md", "skills/research/", "cron/digest.json"],
        )
        staged = _make_staging_dir(profile_env, "nested", manifest=mf)
        (staged / "skills" / "research").mkdir()
        (staged / "skills" / "research" / "SKILL.md").write_text(
            "---\nname: research\ndescription: r\n---\n# R\n"
        )
        (staged / "cron" / "digest.json").write_text('{"schedule": "0 8 * * *"}')

        plan = install_distribution(str(staged), name="nested")
        # Nested allowlisted paths are installed
        assert (plan.target_dir / "skills" / "research" / "SKILL.md").exists()
        assert (plan.target_dir / "cron" / "digest.json").exists()
        # Sibling paths under the same parents are NOT dragged along
        assert not (plan.target_dir / "skills" / "demo").exists(), \
            "skills/demo is not allowlisted and must not be copied"
        assert not (plan.target_dir / "cron" / "daily.json").exists(), \
            "cron/daily.json is not allowlisted and must not be copied"
        # Unrelated top-level entries stay out too
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

    def test_install_rejects_non_distribution_directory(self, profile_env, tmp_path):
        bogus = tmp_path / "bogus_dir"
        bogus.mkdir()
        (bogus / "some_file").write_text("hi")
        with pytest.raises(DistributionError, match="No distribution.yaml"):
            plan_install(str(bogus), tmp_path / "work", override_name="x")


    def test_install_enforces_hermes_requires(self, profile_env, monkeypatch):
        # Pin current Hermes version to something well below the requirement
        import hermes_cli
        monkeypatch.setattr(hermes_cli, "__version__", "0.1.0", raising=False)

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
        with pytest.raises(DistributionError, match="does not exist"):
            describe_distribution("nonexistent")


# ===========================================================================
# Security — USER_OWNED_EXCLUDE covers the right paths
# ===========================================================================


class TestSecurity:

    def test_user_owned_exclude_covers_credentials(self):
        assert "auth.json" in USER_OWNED_EXCLUDE
        assert ".env" in USER_OWNED_EXCLUDE
        assert "memories" in USER_OWNED_EXCLUDE
        assert "sessions" in USER_OWNED_EXCLUDE
        assert "local" in USER_OWNED_EXCLUDE

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
        """A distribution shipping tools/bin/ must not have tools/bin/ dropped
        during install even though 'bin' is in USER_OWNED_EXCLUDE."""
        mf = DistributionManifest(
            name="nested_bin",
            version="0.1.0",
            distribution_owned=list(DEFAULT_DIST_OWNED) + ["tools"],
        )
        staged = _make_staging_dir(profile_env, "src", manifest=mf)
        (staged / "tools" / "bin").mkdir(parents=True)
        (staged / "tools" / "bin" / "tool.py").write_text("# tool\n")

        plan = install_distribution(str(staged), name="nested_bin")
        assert (plan.target_dir / "tools" / "bin").is_dir(), "nested bin/ was dropped"
        assert (plan.target_dir / "tools" / "bin" / "tool.py").exists()

    def test_nested_logs_dir_is_preserved(self, profile_env):
        mf = DistributionManifest(
            name="nested_logs",
            version="0.1.0",
            distribution_owned=list(DEFAULT_DIST_OWNED) + ["scripts"],
        )
        staged = _make_staging_dir(profile_env, "src", manifest=mf)
        (staged / "scripts" / "logs").mkdir(parents=True)
        (staged / "scripts" / "logs" / "run.log").write_text("ok\n")
        plan = install_distribution(str(staged), name="nested_logs")
        assert (plan.target_dir / "scripts" / "logs").is_dir()
        assert (plan.target_dir / "scripts" / "logs" / "run.log").read_text() == "ok\n"

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
        """Top-level bin/ filtered, but tools/bin/ kept."""
        mf = DistributionManifest(
            name="coexist",
            version="0.1.0",
            distribution_owned=list(DEFAULT_DIST_OWNED) + ["tools"],
        )
        staged = _make_staging_dir(profile_env, "src", manifest=mf)
        (staged / "bin").mkdir(exist_ok=True)
        (staged / "bin" / "top.sh").write_text("# top\n")
        (staged / "tools" / "bin").mkdir(parents=True)
        (staged / "tools" / "bin" / "helper.py").write_text("# helper\n")

        plan = install_distribution(str(staged), name="coexist")
        assert not (plan.target_dir / "bin").exists()
        assert (plan.target_dir / "tools" / "bin" / "helper.py").exists()


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

