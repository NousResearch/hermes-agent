"""Tests for hermes_cli.profile_distribution — git-based profile installs.

Covers manifest parsing, version requirement checks, install / update / describe
on local-directory sources, and guards on what can and can't be installed.

Transport-layer tests (git clone, URL handling) are exercised through live
E2E runs, not unit tests — git itself is tested upstream, and subprocess-
mocking git would just test the mock.
"""

from __future__ import annotations

import os
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


# ===========================================================================
# Manifest parsing
# ===========================================================================


class TestManifestParsing:

    def test_minimal_manifest(self, tmp_path):
        (tmp_path / MANIFEST_FILENAME).write_text("name: minimal\n")
        m = read_manifest(tmp_path)
        assert m.name == "minimal"
        assert m.version == "0.1.0"
        assert m.env_requires == []
        assert m.distribution_owned == []

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

    def test_missing_name_rejected(self, tmp_path):
        (tmp_path / MANIFEST_FILENAME).write_text("version: 1.0\n")
        with pytest.raises(DistributionError, match="missing 'name'"):
            read_manifest(tmp_path)

    def test_env_requires_not_list_rejected(self, tmp_path):
        (tmp_path / MANIFEST_FILENAME).write_text(
            "name: bad\nenv_requires:\n  name: FOO\n"
        )
        with pytest.raises(DistributionError, match="env_requires must be a list"):
            read_manifest(tmp_path)

    def test_read_manifest_returns_none_when_absent(self, tmp_path):
        assert read_manifest(tmp_path) is None

    def test_owned_paths_default(self):
        m = DistributionManifest(name="x")
        assert m.owned_paths() == list(DEFAULT_DIST_OWNED)

    def test_owned_paths_explicit(self):
        m = DistributionManifest(name="x", distribution_owned=["SOUL.md", "skills"])
        assert m.owned_paths() == ["SOUL.md", "skills"]

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

    def test_parse_semver_pads(self):
        assert _parse_semver("1") == (1, 0, 0)
        assert _parse_semver("1.2") == (1, 2, 0)

    def test_parse_semver_rejects_garbage(self):
        with pytest.raises(DistributionError, match="Unparseable"):
            _parse_semver("not-a-version")


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

    def test_optional_is_commented(self):
        m = DistributionManifest(
            name="x",
            env_requires=[EnvRequirement(name="BAR", required=False, default="http://x")],
        )
        out = _env_template_from_manifest(m)
        assert "# (optional)" in out
        assert "# BAR=http://x" in out

    def test_empty_env_requires_is_header_only(self):
        m = DistributionManifest(name="x")
        out = _env_template_from_manifest(m)
        assert "Hermes distribution" in out
        assert "FOO" not in out


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

    @pytest.mark.parametrize("src", [
        "/tmp/local/path",
        "./relative/dir",
        "~/profile",
        "some-random-string",
    ])
    def test_rejects_non_git(self, src):
        assert not _looks_like_git_url(src)


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

    def test_install_uses_manifest_name_when_no_override(self, profile_env):
        mf = DistributionManifest(name="telem", version="1.0.0")
        staged = _make_staging_dir(profile_env, "telem", manifest=mf)
        plan = install_distribution(str(staged))
        assert plan.manifest.name == "telem"
        assert plan.target_dir.name == "telem"

    def test_install_rejects_existing_without_force(self, profile_env):
        staged = _make_staging_dir(profile_env, "src")
        install_distribution(str(staged), name="existing")
        with pytest.raises(DistributionError, match="already exists"):
            install_distribution(str(staged), name="existing")

    def test_install_with_force_overwrites(self, profile_env):
        staged = _make_staging_dir(profile_env, "src")
        install_distribution(str(staged), name="target")
        # Install again with --force succeeds
        plan = install_distribution(str(staged), name="target", force=True)
        assert plan.target_dir.is_dir()

    def test_install_rejects_default_name(self, profile_env):
        staged = _make_staging_dir(profile_env, "src")
        with pytest.raises(DistributionError, match="Cannot install"):
            install_distribution(str(staged), name="default")

    def test_install_rejects_non_distribution_directory(self, profile_env, tmp_path):
        bogus = tmp_path / "bogus_dir"
        bogus.mkdir()
        (bogus / "some_file").write_text("hi")
        with pytest.raises(DistributionError, match="No distribution.yaml"):
            plan_install(str(bogus), tmp_path / "work", override_name="x")

    def test_install_rejects_unknown_source(self, profile_env, tmp_path):
        with pytest.raises(DistributionError, match="Cannot resolve"):
            plan_install("definitely-not-a-thing", tmp_path / "work", override_name="x")

    def test_install_emits_env_example_when_manifest_has_env(self, profile_env):
        mf = DistributionManifest(
            name="needs_env",
            version="0.1.0",
            env_requires=[EnvRequirement(name="OPENAI_API_KEY", description="key")],
        )
        staged = _make_staging_dir(profile_env, "needs_env", manifest=mf)
        plan = install_distribution(str(staged), name="needs_env")
        example = plan.target_dir / ".env.EXAMPLE"
        assert example.is_file()
        assert "OPENAI_API_KEY" in example.read_text()

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

    def test_update_force_config_overwrites(self, profile_env):
        staged = _make_staging_dir(profile_env, "src")
        plan = install_distribution(str(staged), name="t3")

        (plan.target_dir / "config.yaml").write_text("model:\n  model: gpt-5\n")

        (staged / "config.yaml").write_text("model:\n  model: claude\n")

        update_distribution("t3", force_config=True)
        assert "claude" in (plan.target_dir / "config.yaml").read_text()
        assert "gpt-5" not in (plan.target_dir / "config.yaml").read_text()

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

    def test_describe_non_distribution_returns_empty(self, profile_env):
        from hermes_cli.profiles import create_profile
        create_profile(name="plain", no_alias=True)
        assert describe_distribution("plain") == {}

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

    def test_plain_profile_has_no_distribution_fields(self, profile_env):
        from hermes_cli.profiles import create_profile, list_profiles
        create_profile(name="plain", no_alias=True)
        rows = {p.name: p for p in list_profiles()}
        assert rows["plain"].distribution_name is None
        assert rows["plain"].distribution_version is None

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

    def test_path_traversal_name_rejected(self, profile_env, tmp_path):
        mf = DistributionManifest(name="../../etc/passwd", version="0.1.0")
        staged = _make_staging_dir(profile_env, "bad", manifest=mf)
        with pytest.raises((ValueError, DistributionError)):
            plan_install(str(staged), tmp_path / "work")

