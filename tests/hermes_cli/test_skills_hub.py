from io import StringIO

import pytest
from rich.console import Console

from hermes_cli.skills_hub import do_check, do_install, do_list, do_update, handle_skills_slash


class _DummyLockFile:
    def __init__(self, installed):
        self._installed = installed

    def list_installed(self):
        return self._installed


@pytest.fixture()
def hub_env(monkeypatch, tmp_path):
    """Set up isolated hub directory paths and return (monkeypatch, tmp_path)."""
    import tools.skills_hub as hub

    hub_dir = tmp_path / "skills" / ".hub"
    monkeypatch.setattr(hub, "SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(hub, "HUB_DIR", hub_dir)
    monkeypatch.setattr(hub, "LOCK_FILE", hub_dir / "lock.json")
    monkeypatch.setattr(hub, "QUARANTINE_DIR", hub_dir / "quarantine")
    monkeypatch.setattr(hub, "AUDIT_LOG", hub_dir / "audit.log")
    monkeypatch.setattr(hub, "TAPS_FILE", hub_dir / "taps.json")
    monkeypatch.setattr(hub, "INDEX_CACHE_DIR", hub_dir / "index-cache")

    return hub_dir


# ---------------------------------------------------------------------------
# Fixtures for common skill setups
# ---------------------------------------------------------------------------

_HUB_ENTRY = {"name": "hub-skill", "source": "github", "trust_level": "community"}

_ALL_THREE_SKILLS = [
    {"name": "hub-skill", "category": "x", "description": "hub"},
    {"name": "builtin-skill", "category": "x", "description": "builtin"},
    {"name": "local-skill", "category": "x", "description": "local"},
]

_BUILTIN_MANIFEST = {"builtin-skill": "abc123"}


@pytest.fixture()
def three_source_env(monkeypatch, hub_env):
    """Populate hub/builtin/local skills for source-classification tests."""
    import tools.skills_hub as hub
    import tools.skills_sync as skills_sync
    import tools.skills_tool as skills_tool

    monkeypatch.setattr(hub, "HubLockFile", lambda path=None: _DummyLockFile([_HUB_ENTRY] if path is None else []))
    monkeypatch.setattr(skills_tool, "_find_all_skills", lambda: list(_ALL_THREE_SKILLS))
    monkeypatch.setattr(skills_sync, "_read_manifest", lambda: dict(_BUILTIN_MANIFEST))

    return hub_env


def _capture(source_filter: str = "all") -> str:
    """Run do_list into a string buffer and return the output."""
    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_list(source_filter=source_filter, console=console)
    return sink.getvalue()


def _capture_check(monkeypatch, results, name=None) -> str:
    import tools.skills_hub as hub

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    monkeypatch.setattr(hub, "check_for_skill_updates", lambda **_kwargs: results)
    do_check(name=name, console=console)
    return sink.getvalue()


def _capture_update(monkeypatch, results) -> tuple[str, list[tuple[str, str, bool]]]:
    import tools.skills_hub as hub
    import hermes_cli.skills_hub as cli_hub

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    installs = []

    monkeypatch.setattr(hub, "check_for_skill_updates", lambda **_kwargs: results)
    monkeypatch.setattr(hub, "HubLockFile", lambda: type("L", (), {
        "get_installed": lambda self, name: {"install_path": "category/" + name}
    })())
    monkeypatch.setattr(cli_hub, "do_install", lambda identifier, category="", force=False, console=None: installs.append((identifier, category, force)))

    do_update(console=console)
    return sink.getvalue(), installs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_do_list_initializes_hub_dir(monkeypatch, hub_env):
    import tools.skills_sync as skills_sync
    import tools.skills_tool as skills_tool

    monkeypatch.setattr(skills_tool, "_find_all_skills", lambda: [])
    monkeypatch.setattr(skills_sync, "_read_manifest", lambda: {})

    hub_dir = hub_env
    assert not hub_dir.exists()

    _capture()

    assert hub_dir.exists()
    assert (hub_dir / "lock.json").exists()
    assert (hub_dir / "quarantine").is_dir()
    assert (hub_dir / "index-cache").is_dir()


def test_do_list_distinguishes_hub_builtin_and_local(three_source_env):
    output = _capture()

    assert "hub-skill" in output
    assert "builtin-skill" in output
    assert "local-skill" in output
    assert "1 hub-installed" in output
    assert "1 builtin" in output
    assert "1 local" in output


def test_do_list_filter_local(three_source_env):
    output = _capture(source_filter="local")

    assert "local-skill" in output
    assert "builtin-skill" not in output
    assert "hub-skill" not in output


def test_do_list_filter_hub(three_source_env):
    output = _capture(source_filter="hub")

    assert "hub-skill" in output
    assert "builtin-skill" not in output
    assert "local-skill" not in output


def test_do_list_filter_builtin(three_source_env):
    output = _capture(source_filter="builtin")

    assert "builtin-skill" in output
    assert "hub-skill" not in output
    assert "local-skill" not in output


def test_do_check_reports_available_updates(monkeypatch):
    output = _capture_check(monkeypatch, [
        {"name": "hub-skill", "source": "skills.sh", "status": "update_available"},
        {"name": "other-skill", "source": "github", "status": "up_to_date"},
    ])

    assert "hub-skill" in output
    assert "update_available" in output
    assert "up_to_date" in output


def test_do_check_handles_no_installed_updates(monkeypatch):
    output = _capture_check(monkeypatch, [])

    assert "No hub-installed skills to check" in output


def test_do_update_reinstalls_outdated_skills(monkeypatch):
    output, installs = _capture_update(monkeypatch, [
        {"name": "hub-skill", "identifier": "skills-sh/example/repo/hub-skill", "status": "update_available"},
        {"name": "other-skill", "identifier": "github/example/other-skill", "status": "up_to_date"},
    ])

    assert installs == [("skills-sh/example/repo/hub-skill", "category", True)]
    assert "Updated 1 skill" in output


def test_do_install_scans_with_resolved_identifier(monkeypatch, tmp_path, hub_env):
    import tools.skills_guard as guard
    import tools.skills_hub as hub

    canonical_identifier = "skills-sh/anthropics/skills/frontend-design"

    class _ResolvedSource:
        def inspect(self, identifier):
            return type("Meta", (), {
                "extra": {},
                "identifier": canonical_identifier,
            })()

        def fetch(self, identifier):
            return type("Bundle", (), {
                "name": "frontend-design",
                "files": {"SKILL.md": "# Frontend Design"},
                "source": "skills.sh",
                "identifier": canonical_identifier,
                "trust_level": "trusted",
                "metadata": {},
            })()

    q_path = tmp_path / "skills" / ".hub" / "quarantine" / "frontend-design"
    q_path.mkdir(parents=True)
    (q_path / "SKILL.md").write_text("# Frontend Design")

    scanned = {}

    def _scan_skill(skill_path, source="community"):
        scanned["source"] = source
        return guard.ScanResult(
            skill_name="frontend-design",
            source=source,
            trust_level="trusted",
            verdict="safe",
        )

    monkeypatch.setattr(hub, "ensure_hub_dirs", lambda: None)
    monkeypatch.setattr(hub, "create_source_router", lambda auth: [_ResolvedSource()])
    monkeypatch.setattr(hub, "quarantine_bundle", lambda bundle: q_path)
    # HubLockFile is called with both HubLockFile() and HubLockFile(path=...) —
    # the mock must accept an optional path kwarg.
    monkeypatch.setattr(hub, "HubLockFile",
                        lambda path=None: type("Lock", (), {
                            "get_installed": lambda self, name: None,
                            "is_hub_installed": lambda self, name: False,
                        })())
    monkeypatch.setattr(guard, "scan_skill", _scan_skill)
    monkeypatch.setattr(guard, "format_scan_report", lambda result: "scan ok")
    monkeypatch.setattr(guard, "should_allow_install", lambda result, force=False: (False, "stop after scan"))

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)

    do_install("skils-sh/anthropics/skills/frontend-design", console=console, skip_confirm=True)

    assert scanned["source"] == canonical_identifier


# ---------------------------------------------------------------------------
# Shared-scope helpers
# ---------------------------------------------------------------------------


class TestProfileSharedConfigHelpers:
    """Unit tests for _add_skill_to_profile_shared_config and _remove_skill_from_profile_shared_config."""

    def test_add_creates_entry_in_new_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.skills_hub import _add_skill_to_profile_shared_config
        _add_skill_to_profile_shared_config("my-skill")
        import yaml
        parsed = yaml.safe_load((tmp_path / "config.yaml").read_text())
        assert parsed["skills"]["shared"] == ["my-skill"]

    def test_add_appends_to_existing_list(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import yaml
        (tmp_path / "config.yaml").write_text(yaml.dump({"skills": {"shared": ["existing"]}}))
        from hermes_cli.skills_hub import _add_skill_to_profile_shared_config
        _add_skill_to_profile_shared_config("new-skill")
        parsed = yaml.safe_load((tmp_path / "config.yaml").read_text())
        assert "existing" in parsed["skills"]["shared"]
        assert "new-skill" in parsed["skills"]["shared"]

    def test_add_is_idempotent(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import yaml
        (tmp_path / "config.yaml").write_text(yaml.dump({"skills": {"shared": ["dup"]}}))
        from hermes_cli.skills_hub import _add_skill_to_profile_shared_config
        _add_skill_to_profile_shared_config("dup")
        parsed = yaml.safe_load((tmp_path / "config.yaml").read_text())
        assert parsed["skills"]["shared"].count("dup") == 1

    def test_remove_skill_from_list(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import yaml
        (tmp_path / "config.yaml").write_text(
            yaml.dump({"skills": {"shared": ["keep", "remove-me"]}})
        )
        from hermes_cli.skills_hub import _remove_skill_from_profile_shared_config
        _remove_skill_from_profile_shared_config("remove-me")
        parsed = yaml.safe_load((tmp_path / "config.yaml").read_text())
        assert "remove-me" not in parsed["skills"]["shared"]
        assert "keep" in parsed["skills"]["shared"]

    def test_remove_noop_when_not_present(self, tmp_path, monkeypatch):
        """remove should not raise when the skill isn't in the list."""
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        import yaml
        (tmp_path / "config.yaml").write_text(yaml.dump({"skills": {"shared": ["other"]}}))
        from hermes_cli.skills_hub import _remove_skill_from_profile_shared_config
        _remove_skill_from_profile_shared_config("absent")  # must not raise

    def test_remove_noop_when_no_config(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_HOME", str(tmp_path))
        from hermes_cli.skills_hub import _remove_skill_from_profile_shared_config
        _remove_skill_from_profile_shared_config("absent")  # must not raise


# ---------------------------------------------------------------------------
# Shared install flow
# ---------------------------------------------------------------------------


@pytest.fixture()
def shared_env(monkeypatch, tmp_path, hub_env):
    """Extend hub_env with isolated shared-skills paths."""
    import tools.skills_hub as hub

    shared_root = tmp_path / "shared-skills"
    shared_root.mkdir(parents=True)
    shared_lockfile = shared_root / ".hub-lock.json"

    monkeypatch.setattr(hub, "DEFAULT_SHARED_SKILLS", shared_root)
    monkeypatch.setattr(hub, "SHARED_HUB_LOCKFILE", shared_lockfile)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes_home"))
    (tmp_path / "hermes_home").mkdir(parents=True)

    return {
        "hub_dir": hub_env,
        "shared_root": shared_root,
        "shared_lockfile": shared_lockfile,
        "hermes_home": tmp_path / "hermes_home",
    }


def _make_install_mocks(monkeypatch, hub_env_dir, skill_name="test-skill"):
    """Return a (q_path, bundle) and wire up mocks for a successful do_install run."""
    import tools.skills_guard as guard
    import tools.skills_hub as hub

    q_path = hub_env_dir / "quarantine" / skill_name
    q_path.mkdir(parents=True)
    (q_path / "SKILL.md").write_text(
        f"---\nname: {skill_name}\ndescription: A test skill\n---\n\n# {skill_name}\n"
    )

    class _Source:
        def inspect(self, identifier):
            return type("Meta", (), {"extra": {}, "identifier": identifier})()

        def fetch(self, identifier):
            return type("Bundle", (), {
                "name": skill_name,
                "files": {"SKILL.md": "# Test"},
                "source": "github",
                "identifier": identifier,
                "trust_level": "community",
                "metadata": {},
            })()

    monkeypatch.setattr(hub, "ensure_hub_dirs", lambda: None)
    monkeypatch.setattr(hub, "create_source_router", lambda auth: [_Source()])
    monkeypatch.setattr(hub, "quarantine_bundle", lambda bundle: q_path)
    monkeypatch.setattr(guard, "format_scan_report", lambda result: "scan ok")
    monkeypatch.setattr(guard, "should_allow_install", lambda result, force=False: (True, ""))

    def _scan(path, source="community"):
        return guard.ScanResult(
            skill_name=skill_name, source=source,
            trust_level="community", verdict="safe",
        )

    monkeypatch.setattr(guard, "scan_skill", _scan)
    return q_path


class TestInstallSharedScope:
    def test_install_local_is_default_when_skip_confirm(self, monkeypatch, tmp_path, shared_env):
        """With skip_confirm=True and no explicit scope, skill installs profile-local."""
        import tools.skills_hub as hub

        q_path = _make_install_mocks(monkeypatch, shared_env["hub_dir"])
        monkeypatch.setattr(hub, "HubLockFile",
                            lambda path=None: type("Lock", (), {
                                "get_installed": lambda self, n: None,
                                "is_hub_installed": lambda self, n: False,
                                "record_install": lambda self, **kw: None,
                            })())

        sink = StringIO()
        console = Console(file=sink, force_terminal=False, color_system=None)
        do_install("github/owner/test-skill", console=console, skip_confirm=True)

        # Skill should land in profile-local skills dir, NOT shared root
        assert (tmp_path / "skills" / "test-skill").exists()
        assert not (shared_env["shared_root"] / "test-skill").exists()

    def test_install_shared_writes_to_shared_root(self, monkeypatch, tmp_path, shared_env):
        """shared=True routes the skill to DEFAULT_SHARED_SKILLS."""
        import tools.skills_hub as hub

        q_path = _make_install_mocks(monkeypatch, shared_env["hub_dir"])
        monkeypatch.setattr(hub, "HubLockFile",
                            lambda path=None: type("Lock", (), {
                                "get_installed": lambda self, n: None,
                                "is_hub_installed": lambda self, n: False,
                                "record_install": lambda self, **kw: None,
                            })())

        sink = StringIO()
        console = Console(file=sink, force_terminal=False, color_system=None)
        do_install("github/owner/test-skill", console=console, skip_confirm=True, shared=True)

        assert (shared_env["shared_root"] / "test-skill").exists()
        # Must NOT be in profile-local dir
        assert not (tmp_path / "skills" / "test-skill").exists()

    def test_install_shared_updates_profile_config(self, monkeypatch, tmp_path, shared_env):
        """Shared install adds the skill name to skills.shared in config.yaml."""
        import tools.skills_hub as hub
        import yaml

        _make_install_mocks(monkeypatch, shared_env["hub_dir"])
        monkeypatch.setattr(hub, "HubLockFile",
                            lambda path=None: type("Lock", (), {
                                "get_installed": lambda self, n: None,
                                "is_hub_installed": lambda self, n: False,
                                "record_install": lambda self, **kw: None,
                            })())

        sink = StringIO()
        console = Console(file=sink, force_terminal=False, color_system=None)
        do_install("github/owner/test-skill", console=console, skip_confirm=True, shared=True)

        config = yaml.safe_load((shared_env["hermes_home"] / "config.yaml").read_text())
        assert "test-skill" in config["skills"]["shared"]

    def test_install_shared_records_in_shared_lockfile(self, monkeypatch, tmp_path, shared_env):
        """Shared install writes provenance to SHARED_HUB_LOCKFILE."""
        import tools.skills_hub as hub
        import json

        q_path = _make_install_mocks(monkeypatch, shared_env["hub_dir"])
        # Use real HubLockFile so we can inspect the lockfile contents
        monkeypatch.setattr(hub, "HubLockFile",
                            lambda path=None: hub.HubLockFile.__wrapped__(path=path)
                            if hasattr(hub.HubLockFile, "__wrapped__")
                            else type("Lock", (), {
                                "get_installed": lambda self, n: None,
                                "is_hub_installed": lambda self, n: False,
                                "record_install": lambda self, **kw: None,
                            })())

        recorded = {}

        class _RealLikeLocFile:
            def __init__(self, path=None):
                self._path = path
            def get_installed(self, name):
                return None
            def is_hub_installed(self, name):
                return False
            def record_install(self, *, name, **kwargs):
                recorded["name"] = name
                recorded["path"] = self._path

        monkeypatch.setattr(hub, "HubLockFile", _RealLikeLocFile)

        sink = StringIO()
        console = Console(file=sink, force_terminal=False, color_system=None)
        do_install("github/owner/test-skill", console=console, skip_confirm=True, shared=True)

        assert recorded.get("name") == "test-skill"
        assert recorded.get("path") == shared_env["shared_lockfile"]


# ---------------------------------------------------------------------------
# Uninstall shared scope
# ---------------------------------------------------------------------------


class TestUninstallSharedScope:
    def test_uninstall_shared_skill_removes_dir_and_config(
        self, monkeypatch, tmp_path, shared_env,
    ):
        """Uninstalling a shared skill removes its dir and config entry."""
        import tools.skills_hub as hub
        import yaml
        from hermes_cli.skills_hub import do_uninstall

        # Create the skill dir in the shared root
        skill_dir = shared_env["shared_root"] / "shared-skill"
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text("# shared\n")

        # Write a real shared lockfile entry
        from tools.skills_hub import HubLockFile as _RealHubLockFile
        lock = _RealHubLockFile(path=shared_env["shared_lockfile"])
        lock.record_install(
            name="shared-skill",
            source="github",
            identifier="github/owner/shared-skill",
            trust_level="community",
            scan_verdict="safe",
            skill_hash="abc",
            install_path="shared-skill",
            files=["SKILL.md"],
        )

        # Add to profile config
        (shared_env["hermes_home"] / "config.yaml").write_text(
            yaml.dump({"skills": {"shared": ["shared-skill"]}})
        )

        # Patch hub paths so uninstall_skill uses the right roots
        monkeypatch.setattr(hub, "DEFAULT_SHARED_SKILLS", shared_env["shared_root"])
        monkeypatch.setattr(hub, "SHARED_HUB_LOCKFILE", shared_env["shared_lockfile"])

        sink = StringIO()
        console = Console(file=sink, force_terminal=False, color_system=None)
        do_uninstall("shared-skill", skip_confirm=True, console=console)

        # Skill directory should be gone
        assert not skill_dir.exists()
        # Lockfile entry should be removed
        remaining = lock.list_installed()
        assert not any(e["name"] == "shared-skill" for e in remaining)
        # Profile config entry should be removed
        config = yaml.safe_load((shared_env["hermes_home"] / "config.yaml").read_text())
        shared_list = (config.get("skills") or {}).get("shared", [])
        assert "shared-skill" not in shared_list

    def test_uninstall_local_skill_unchanged_shared_lockfile(
        self, monkeypatch, tmp_path, shared_env,
    ):
        """Uninstalling a profile-local hub skill does not touch the shared lockfile."""
        import tools.skills_hub as hub
        from hermes_cli.skills_hub import do_uninstall

        # Create local skill
        local_skill = tmp_path / "skills" / "local-hub-skill"
        local_skill.mkdir(parents=True)

        # Write to profile-local lockfile via real HubLockFile
        monkeypatch.setattr(hub, "LOCK_FILE", tmp_path / "skills" / ".hub" / "lock.json")
        (tmp_path / "skills" / ".hub").mkdir(parents=True)
        from tools.skills_hub import HubLockFile as _Real
        local_lock = _Real(path=tmp_path / "skills" / ".hub" / "lock.json")
        local_lock.record_install(
            name="local-hub-skill", source="github",
            identifier="github/o/local-hub-skill", trust_level="community",
            scan_verdict="safe", skill_hash="xyz", install_path="local-hub-skill",
            files=["SKILL.md"],
        )

        monkeypatch.setattr(hub, "DEFAULT_SHARED_SKILLS", shared_env["shared_root"])
        monkeypatch.setattr(hub, "SHARED_HUB_LOCKFILE", shared_env["shared_lockfile"])

        sink = StringIO()
        console = Console(file=sink, force_terminal=False, color_system=None)
        do_uninstall("local-hub-skill", skip_confirm=True, console=console)

        # Shared lockfile should remain empty
        from tools.skills_hub import HubLockFile as _Real2
        shared = _Real2(path=shared_env["shared_lockfile"])
        assert shared.list_installed() == []


# ---------------------------------------------------------------------------
# do_list scope column
# ---------------------------------------------------------------------------


class TestDoListScopeColumn:
    def test_scope_column_present(self, three_source_env, monkeypatch):
        """do_list always emits a Scope column."""
        import tools.skills_hub as hub

        monkeypatch.setattr(hub, "SHARED_HUB_LOCKFILE",
                            three_source_env / "shared-lock.json")
        output = _capture()
        assert "Scope" in output

    def test_shared_skill_shows_shared_scope(self, monkeypatch, tmp_path):
        """A skill in the shared lockfile is listed with scope='shared'."""
        import tools.skills_hub as hub
        import tools.skills_sync as skills_sync
        import tools.skills_tool as skills_tool

        hub_dir = tmp_path / "skills" / ".hub"
        monkeypatch.setattr(hub, "SKILLS_DIR", tmp_path / "skills")
        monkeypatch.setattr(hub, "HUB_DIR", hub_dir)
        monkeypatch.setattr(hub, "LOCK_FILE", hub_dir / "lock.json")
        monkeypatch.setattr(hub, "QUARANTINE_DIR", hub_dir / "quarantine")
        monkeypatch.setattr(hub, "AUDIT_LOG", hub_dir / "audit.log")
        monkeypatch.setattr(hub, "TAPS_FILE", hub_dir / "taps.json")
        monkeypatch.setattr(hub, "INDEX_CACHE_DIR", hub_dir / "index-cache")

        shared_lockfile = tmp_path / "shared-lock.json"
        monkeypatch.setattr(hub, "SHARED_HUB_LOCKFILE", shared_lockfile)

        # local lockfile has no entries; shared has one
        _shared_entry = {"name": "shared-skill", "source": "github", "trust_level": "community"}

        class _EmptyLock:
            def list_installed(self): return []

        class _SharedLock:
            def list_installed(self): return [_shared_entry]

        monkeypatch.setattr(hub, "HubLockFile",
                            lambda path=None: _SharedLock() if path == shared_lockfile else _EmptyLock())
        monkeypatch.setattr(skills_tool, "_find_all_skills",
                            lambda: [{"name": "shared-skill", "category": "", "description": "x"}])
        monkeypatch.setattr(skills_sync, "_read_manifest", lambda: {})

        sink = StringIO()
        console = Console(file=sink, force_terminal=False, color_system=None)
        from hermes_cli.skills_hub import do_list
        do_list(console=console)
        output = sink.getvalue()

        assert "shared-skill" in output
        assert "shared" in output

    def test_do_list_filter_shared(self, monkeypatch, tmp_path):
        """--source shared shows only shared-scope skills."""
        import tools.skills_hub as hub
        import tools.skills_sync as skills_sync
        import tools.skills_tool as skills_tool

        hub_dir = tmp_path / "skills" / ".hub"
        monkeypatch.setattr(hub, "SKILLS_DIR", tmp_path / "skills")
        monkeypatch.setattr(hub, "HUB_DIR", hub_dir)
        monkeypatch.setattr(hub, "LOCK_FILE", hub_dir / "lock.json")
        monkeypatch.setattr(hub, "QUARANTINE_DIR", hub_dir / "quarantine")
        monkeypatch.setattr(hub, "AUDIT_LOG", hub_dir / "audit.log")
        monkeypatch.setattr(hub, "TAPS_FILE", hub_dir / "taps.json")
        monkeypatch.setattr(hub, "INDEX_CACHE_DIR", hub_dir / "index-cache")

        shared_lockfile = tmp_path / "shared-lock.json"
        monkeypatch.setattr(hub, "SHARED_HUB_LOCKFILE", shared_lockfile)

        class _EmptyLock:
            def list_installed(self): return []

        class _SharedLock:
            def list_installed(self):
                return [{"name": "shared-only", "source": "github", "trust_level": "community"}]

        monkeypatch.setattr(hub, "HubLockFile",
                            lambda path=None: _SharedLock() if path == shared_lockfile else _EmptyLock())
        monkeypatch.setattr(skills_tool, "_find_all_skills", lambda: [
            {"name": "shared-only", "category": "", "description": "x"},
            {"name": "local-skill", "category": "", "description": "y"},
        ])
        monkeypatch.setattr(skills_sync, "_read_manifest", lambda: {})

        sink = StringIO()
        console = Console(file=sink, force_terminal=False, color_system=None)
        from hermes_cli.skills_hub import do_list
        do_list(source_filter="shared", console=console)
        output = sink.getvalue()

        assert "shared-only" in output
        assert "local-skill" not in output
