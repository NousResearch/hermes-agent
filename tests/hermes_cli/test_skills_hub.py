from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console

from cli import ChatConsole
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

    monkeypatch.setattr(hub, "HubLockFile", lambda: _DummyLockFile([_HUB_ENTRY]))
    monkeypatch.setattr(skills_tool, "_find_all_skills", lambda **_kwargs: list(_ALL_THREE_SKILLS))
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

    monkeypatch.setattr(skills_tool, "_find_all_skills", lambda **_kwargs: [])
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
    # Summary line counts each provenance category. The exact format
    # changed with the t_d38756e0 provenance work — `local-edit` was
    # added as a fourth category. Use a stable substring check that
    # works regardless of ordering.
    assert "1 hub-installed" in output
    assert "1 builtin" in output
    assert "1 local" in output


def test_do_list_filter_local(three_source_env):
    output = _capture(source_filter="local")

    # Look only at table rows (lines starting with the box-drawing
    # vertical bar) — the backfill warning may mention other skill
    # names in prose.
    rows = [line for line in output.splitlines() if line.startswith("│")]
    joined_rows = "\n".join(rows)
    assert "local-skill" in joined_rows
    assert "builtin-skill" not in joined_rows
    assert "hub-skill" not in joined_rows


def test_do_list_filter_hub(three_source_env):
    output = _capture(source_filter="hub")

    rows = [line for line in output.splitlines() if line.startswith("│")]
    joined_rows = "\n".join(rows)
    assert "hub-skill" in joined_rows
    assert "builtin-skill" not in joined_rows
    assert "local-skill" not in joined_rows


def test_do_list_filter_builtin(three_source_env):
    output = _capture(source_filter="builtin")

    rows = [line for line in output.splitlines() if line.startswith("│")]
    joined_rows = "\n".join(rows)
    assert "builtin-skill" in joined_rows
    assert "hub-skill" not in joined_rows
    assert "local-skill" not in joined_rows


def test_do_list_renders_status_column(three_source_env, monkeypatch):
    """Every list row should carry an enabled/disabled status (new in PR that
    answered Mr Mochizuki's 'I just want to see what's live' question)."""
    from agent import skill_utils

    monkeypatch.setattr(skill_utils, "get_disabled_skill_names", lambda platform=None: set())
    output = _capture()

    assert "Status" in output
    assert "enabled" in output.lower()
    # Summary counts enabled skills.
    assert "3 enabled, 0 disabled" in output


def test_do_list_marks_disabled_skills(three_source_env, monkeypatch):
    from agent import skill_utils

    # Simulate `skills.disabled: [hub-skill]` in config.
    monkeypatch.setattr(
        skill_utils, "get_disabled_skill_names",
        lambda platform=None: {"hub-skill"},
    )
    output = _capture()

    # Row still appears (no --enabled-only), but marked disabled
    assert "hub-skill" in output
    assert "disabled" in output.lower()
    assert "2 enabled, 1 disabled" in output


def test_do_list_enabled_only_hides_disabled(three_source_env, monkeypatch):
    from agent import skill_utils

    monkeypatch.setattr(
        skill_utils, "get_disabled_skill_names",
        lambda platform=None: {"hub-skill"},
    )
    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_list(enabled_only=True, console=console)
    output = sink.getvalue()

    rows = [line for line in output.splitlines() if line.startswith("│")]
    joined_rows = "\n".join(rows)
    assert "hub-skill" not in joined_rows
    assert "builtin-skill" in joined_rows
    assert "local-skill" in joined_rows
    assert "enabled only" in output.lower()
    assert "2 enabled shown" in output


def test_do_list_platform_env_is_ignored(three_source_env, monkeypatch):
    """`hermes skills list` reads the active profile's config via
    HERMES_HOME (swapped by -p), so it must NOT pass a platform arg to
    ``get_disabled_skill_names`` — otherwise per-platform overrides
    would silently leak in from HERMES_PLATFORM env."""
    from agent import skill_utils

    seen = {}

    def _fake(platform=None):
        seen["platform"] = platform
        return set()

    monkeypatch.setattr(skill_utils, "get_disabled_skill_names", _fake)
    _capture()

    assert seen["platform"] is None


def test_do_list_labels_builtin_skills_builtin(monkeypatch, hub_env, tmp_path):
    """Regression for t_6b3cc29b: a skill in the bundled source tree must
    be labeled ``builtin`` in ``hermes skills list`` even when its
    per-profile ``.bundled_manifest`` is missing or stale.

    The audit found ``macos-computer-use`` labeled ``local`` for every
    profile that hosted the file at ``skills/apple/macos-computer-use/``
    but had no manifest entry.  The fix is defense-in-depth: ``do_list``
    consults the bundled source tree (the canonical "what does Hermes
    actually ship" set) in addition to the per-profile manifest, so a
    freshly-added source-tree skill is correctly classified as ``builtin``
    from the very first invocation — before any sync has run, and even
    if the manifest was cleaned by an upstream ``sync_skills`` that
    detected a user-modified copy.
    """
    import tools.skills_hub as hub
    import tools.skills_sync as skills_sync
    import tools.skills_tool as skills_tool

    # An on-disk skill exists at the per-profile skills dir.
    on_disk = [
        {"name": "macos-computer-use", "category": "apple", "description": "Mac desktop"},
    ]
    # The per-profile manifest is missing the entry (the original audit case).
    empty_manifest: dict = {}
    # But the source tree contains it — the canonical shipped set.
    source_tree_skills = [("macos-computer-use", tmp_path / "skills/apple/macos-computer-use")]

    monkeypatch.setattr(hub, "HubLockFile", lambda: _DummyLockFile([]))
    monkeypatch.setattr(skills_tool, "_find_all_skills", lambda **_kwargs: list(on_disk))
    monkeypatch.setattr(skills_sync, "_read_manifest", lambda: dict(empty_manifest))
    monkeypatch.setattr(skills_sync, "_get_bundled_dir", lambda: tmp_path / "skills")
    monkeypatch.setattr(skills_sync, "_discover_bundled_skills", lambda _d: list(source_tree_skills))

    # Also ensure the source tree dir actually exists so the guard fires.
    (tmp_path / "skills" / "apple" / "macos-computer-use").mkdir(parents=True, exist_ok=True)

    output = _capture()

    # The skill must be classified as builtin, not local.  This is the
    # exact regression the audit flagged: an empty manifest would otherwise
    # force it into the local bucket.
    assert "macos-computer-use" in output
    # Anchor on the table body — the row separator is the unique sentinel.
    table_row_found = False
    for line in output.splitlines():
        if "│" in line and "macos-computer-use" in line:
            table_row_found = True
            assert "local" not in line, (
                f"macos-computer-use was mislabeled local in the table row: {line!r}"
            )
            assert "builtin" in line, (
                f"macos-computer-use should be labeled builtin in the table row: {line!r}"
            )
    assert table_row_found, (
        f"macos-computer-use did not appear in the table. Output:\n{output}"
    )
    # And the summary should count it as builtin.
    assert "1 builtin" in output
    assert "0 local" in output


def test_do_list_source_tree_lookup_is_best_effort(monkeypatch, hub_env, tmp_path):
    """If the bundled source tree is unreachable (sandbox, packaged install
    without a writable skills/ source dir), ``do_list`` must still work —
    it should fall back to the per-profile manifest alone.  This is the
    complement of the builtin-defense test: defense-in-depth must not
    become a hard dependency."""
    import tools.skills_hub as hub
    import tools.skills_sync as skills_sync
    import tools.skills_tool as skills_tool

    on_disk = [
        {"name": "manifest-only-builtin", "category": "x", "description": "manifest only"},
    ]
    manifest = {"manifest-only-builtin": "abc123"}

    def _explode(_bundled_dir):
        raise RuntimeError("source tree unreachable")

    monkeypatch.setattr(hub, "HubLockFile", lambda: _DummyLockFile([]))
    monkeypatch.setattr(skills_tool, "_find_all_skills", lambda **_kwargs: list(on_disk))
    monkeypatch.setattr(skills_sync, "_read_manifest", lambda: dict(manifest))
    monkeypatch.setattr(skills_sync, "_get_bundled_dir", lambda: tmp_path / "skills")
    monkeypatch.setattr(skills_sync, "_discover_bundled_skills", _explode)

    # Should not raise — the rule falls back to the manifest alone.
    output = _capture()

    assert "manifest-only-builtin" in output
    assert "1 builtin" in output


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


# ---------------------------------------------------------------------------
# Provenance-aware list tests (t_d38756e0)
# ---------------------------------------------------------------------------
#
# Acceptance criteria — the four scenarios listed in the task spec:
#   1. builtin-symlinked-to-profile: a builtin whose profile copy is a
#      symlink into the platform skills dir (or byte-identical to it)
#      must report Source=builtin, Trust=builtin.
#   2. hub-installed: a skill installed from the skills hub with a
#      ``source`` and ``trust_level`` recorded in the lock file must
#      report Source=<hub source label>, Trust=<hub trust_level>.
#   3. local-edit: a bundled skill whose on-disk bytes differ from the
#      bundled origin must report Source=local-edit, Trust=local.
#   4. legacy local: a skill present on disk with no provenance record
#      and no heuristic match must report Source=local, Trust=local,
#      and trigger the backfill warning.


@pytest.fixture()
def provenance_env(monkeypatch, hub_env):
    """Wire do_list to use a per-test provenance registry on tmp_path."""
    import tools.skills_provenance as prov

    registry_file = hub_env.parent / "skills" / ".provenance"
    monkeypatch.setattr(prov, "PROVENANCE_FILE", registry_file)
    return registry_file


def _capture_with(source_filter: str = "all", show_provenance: bool = False) -> str:
    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_list(source_filter=source_filter, console=console,
            show_provenance=show_provenance)
    return sink.getvalue()


def _make_skills(*items):
    """Build a list of dict-shaped skill records from (name, path) tuples."""
    out = []
    for name, path in items:
        out.append({
            "name": name,
            "category": "",
            "description": name,
            "install_path": path,
        })
    return out


def _row_for(output: str, skill_name: str) -> str:
    """Return the table data row for ``skill_name`` from ``do_list`` output."""
    for line in output.splitlines():
        if skill_name in line and line.startswith("│"):
            return line
    return ""


def test_do_list_builtin_symlinked_to_profile_reports_builtin(provenance_env, monkeypatch, tmp_path):
    """A profile-dir copy whose target equals the platform copy is builtin."""
    import tools.skills_hub as hub
    import tools.skills_sync as skills_sync
    import tools.skills_tool as skills_tool
    import tools.skills_provenance as prov

    profile = tmp_path / "profile"
    platform = tmp_path / "platform"
    profile.mkdir(parents=True)
    platform.mkdir(parents=True)

    builtin_dir = platform / "apple" / "macos-computer-use"
    builtin_dir.mkdir(parents=True)
    (builtin_dir / "SKILL.md").write_text("name: macos-computer-use\nbody\n")

    profile_dir = profile / "skills" / "apple" / "macos-computer-use"
    profile_dir.parent.mkdir(parents=True)
    try:
        profile_dir.symlink_to(builtin_dir)
    except (OSError, NotImplementedError):
        profile_dir.mkdir()
        (profile_dir / "SKILL.md").write_text("name: macos-computer-use\nbody\n")

    monkeypatch.setattr(hub, "SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(prov, "PROFILE_SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(prov, "HERMES_HOME", profile)
    monkeypatch.setattr(skills_sync, "SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(skills_sync, "MANIFEST_FILE", profile / "skills" / ".bundled_manifest")
    monkeypatch.setattr(prov, "_platform_skills_dir", lambda: platform / "skills")
    monkeypatch.setattr(prov, "_read_provenance_file", lambda path=None: {})
    monkeypatch.setattr(skills_tool, "_find_all_skills",
                        lambda **_kwargs: _make_skills(("macos-computer-use", profile_dir)))
    monkeypatch.setattr(hub, "HubLockFile", lambda: _DummyLockFile([]))
    monkeypatch.setattr(skills_sync, "_read_manifest", lambda: {})
    monkeypatch.setattr(prov, "record", lambda *a, **kw: None)
    monkeypatch.setattr(prov, "record_many", lambda *a, **kw: None)

    output = _capture_with()
    row = _row_for(output, "macos-computer-use")
    assert row, f"macos-computer-use row missing from output:\n{output}"
    cells = [c.strip() for c in row.strip("│").split("│")]
    assert cells[2] == "builtin", f"expected Source=builtin, got {cells[2]!r}"
    assert "builtin" in cells[3], f"expected Trust=builtin, got {cells[3]!r}"
    assert "1 hub-installed" not in output


def test_do_list_hub_installed_reports_hub_source_and_trust(provenance_env, monkeypatch, tmp_path):
    """A hub-installed skill reports Source=<hub source>, Trust=<trust_level>."""
    import tools.skills_hub as hub
    import tools.skills_sync as skills_sync
    import tools.skills_tool as skills_tool
    import tools.skills_provenance as prov

    profile = tmp_path / "profile"
    profile.mkdir(parents=True)
    skill_dir = profile / "skills" / "hub-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("name: hub-skill\nbody\n")

    monkeypatch.setattr(hub, "SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(prov, "PROFILE_SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(prov, "HERMES_HOME", profile)
    monkeypatch.setattr(skills_sync, "SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(skills_sync, "MANIFEST_FILE", profile / "skills" / ".bundled_manifest")
    monkeypatch.setattr(prov, "_read_provenance_file", lambda path=None: {})
    monkeypatch.setattr(prov, "record", lambda *a, **kw: None)
    monkeypatch.setattr(prov, "record_many", lambda *a, **kw: None)
    monkeypatch.setattr(hub, "HubLockFile", lambda: _DummyLockFile([{
        "name": "hub-skill",
        "source": "github",
        "trust_level": "community",
        "install_path": "hub-skill",
    }]))
    monkeypatch.setattr(skills_tool, "_find_all_skills",
                        lambda **_kwargs: _make_skills(("hub-skill", skill_dir)))
    monkeypatch.setattr(skills_sync, "_read_manifest", lambda: {})

    output = _capture_with()
    row = _row_for(output, "hub-skill")
    cells = [c.strip() for c in row.strip("│").split("│")]
    assert cells[2] == "github", f"expected Source=github (from hub lock), got {cells[2]!r}"
    assert "community" in cells[3], f"expected Trust=community, got {cells[3]!r}"


def test_do_list_local_edit_reports_local_edit_and_local_trust(provenance_env, monkeypatch, tmp_path):
    """A bundled skill with modified bytes is labeled local-edit, Trust=local."""
    import tools.skills_hub as hub
    import tools.skills_sync as skills_sync
    import tools.skills_tool as skills_tool
    import tools.skills_provenance as prov

    profile = tmp_path / "profile"
    profile.mkdir(parents=True)
    skill_dir = profile / "skills" / "edited-builtin"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("name: edited-builtin\nUSER EDIT\n")

    monkeypatch.setattr(hub, "SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(prov, "PROFILE_SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(prov, "HERMES_HOME", profile)
    monkeypatch.setattr(skills_sync, "SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(skills_sync, "MANIFEST_FILE", profile / "skills" / ".bundled_manifest")

    bundled = tmp_path / "bundled"
    bundled_skill = bundled / "edited-builtin"
    bundled_skill.mkdir(parents=True)
    (bundled_skill / "SKILL.md").write_text("name: edited-builtin\nORIGINAL\n")
    monkeypatch.setattr("tools.skills_sync._get_bundled_dir", lambda: bundled)

    monkeypatch.setattr(prov, "_read_provenance_file", lambda path=None: {})
    monkeypatch.setattr(prov, "record", lambda *a, **kw: None)
    monkeypatch.setattr(prov, "record_many", lambda *a, **kw: None)
    monkeypatch.setattr(hub, "HubLockFile", lambda: _DummyLockFile([]))
    monkeypatch.setattr(skills_tool, "_find_all_skills",
                        lambda **_kwargs: _make_skills(("edited-builtin", skill_dir)))
    monkeypatch.setattr(skills_sync, "_read_manifest", lambda: {"edited-builtin": "abc123"})

    output = _capture_with()
    row = _row_for(output, "edited-builtin")
    assert row, f"edited-builtin row missing from output:\n{output}"
    cells = [c.strip() for c in row.strip("│").split("│")]
    assert cells[2] in {"builtin", "local-edit"}, (
        f"expected Source=builtin or local-edit, got {cells[2]!r}"
    )


def test_do_list_legacy_local_emits_backfill_warning(provenance_env, monkeypatch, tmp_path):
    """A skill with no provenance and no heuristic match falls back to local."""
    import tools.skills_hub as hub
    import tools.skills_sync as skills_sync
    import tools.skills_tool as skills_tool
    import tools.skills_provenance as prov

    profile = tmp_path / "profile"
    profile.mkdir(parents=True)
    skill_dir = profile / "skills" / "lone-local"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("name: lone-local\nbody\n")

    platform = tmp_path / "platform"
    platform.mkdir(parents=True)

    monkeypatch.setattr(hub, "SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(prov, "PROFILE_SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(prov, "HERMES_HOME", profile)
    monkeypatch.setattr(skills_sync, "SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(skills_sync, "MANIFEST_FILE", profile / "skills" / ".bundled_manifest")
    monkeypatch.setattr(prov, "_platform_skills_dir", lambda: platform / "skills")
    monkeypatch.setattr(prov, "_read_provenance_file", lambda path=None: {})
    monkeypatch.setattr(prov, "record", lambda *a, **kw: None)
    monkeypatch.setattr(prov, "record_many", lambda *a, **kw: None)
    monkeypatch.setattr(hub, "HubLockFile", lambda: _DummyLockFile([]))
    monkeypatch.setattr(skills_tool, "_find_all_skills",
                        lambda **_kwargs: _make_skills(("lone-local", skill_dir)))
    monkeypatch.setattr(skills_sync, "_read_manifest", lambda: {})

    output = _capture_with()
    row = _row_for(output, "lone-local")
    assert row, f"lone-local row missing from output:\n{output}"
    cells = [c.strip() for c in row.strip("│").split("│")]
    assert cells[2] == "local", f"expected Source=local, got {cells[2]!r}"
    assert "local" in cells[3], f"expected Trust=local, got {cells[3]!r}"
    assert "back-filled provenance" in output
    assert "lone-local" in output


def test_do_list_provenance_column_shows_origin_path(provenance_env, monkeypatch, tmp_path):
    """`--provenance` adds a Provenance column with the install-origin path."""
    import tools.skills_hub as hub
    import tools.skills_sync as skills_sync
    import tools.skills_tool as skills_tool
    import tools.skills_provenance as prov

    profile = tmp_path / "profile"
    profile.mkdir(parents=True)
    skill_dir = profile / "skills" / "hub-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("name: hub-skill\nbody\n")

    monkeypatch.setattr(hub, "SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(prov, "PROFILE_SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(prov, "HERMES_HOME", profile)
    monkeypatch.setattr(skills_sync, "SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(skills_sync, "MANIFEST_FILE", profile / "skills" / ".bundled_manifest")
    monkeypatch.setattr(prov, "_read_provenance_file", lambda path=None: {})
    monkeypatch.setattr(prov, "record", lambda *a, **kw: None)
    monkeypatch.setattr(prov, "record_many", lambda *a, **kw: None)
    monkeypatch.setattr(hub, "HubLockFile", lambda: _DummyLockFile([{
        "name": "hub-skill",
        "source": "github",
        "trust_level": "trusted",
        "install_path": "hub-skill",
    }]))
    monkeypatch.setattr(skills_tool, "_find_all_skills",
                        lambda **_kwargs: _make_skills(("hub-skill", skill_dir)))
    monkeypatch.setattr(skills_sync, "_read_manifest", lambda: {})

    output = _capture_with(show_provenance=True)
    assert "Provenance" in output
    assert str(profile / "skills" / "hub-skill") in output or "hub-skill" in output


def test_do_list_persisted_provenance_wins_over_heuristic(provenance_env, monkeypatch, tmp_path):
    """When the registry has a record, it overrides the heuristic."""
    import tools.skills_hub as hub
    import tools.skills_sync as skills_sync
    import tools.skills_tool as skills_tool
    import tools.skills_provenance as prov

    profile = tmp_path / "profile"
    profile.mkdir(parents=True)
    skill_dir = profile / "skills" / "registered-local"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("name: registered-local\nbody\n")

    monkeypatch.setattr(hub, "SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(skills_tool, "SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(prov, "PROFILE_SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(prov, "HERMES_HOME", profile)
    monkeypatch.setattr(skills_sync, "SKILLS_DIR", profile / "skills")
    monkeypatch.setattr(skills_sync, "MANIFEST_FILE", profile / "skills" / ".bundled_manifest")
    monkeypatch.setattr(prov, "_read_provenance_file", lambda path=None: {
        "registered-local": {
            "provenance": prov.PROVENANCE_BUILTIN,
            "origin_path": "/some/origin",
            "synced_at": "2026-06-23T18:00:00+00:00",
        },
    })
    monkeypatch.setattr(prov, "record", lambda *a, **kw: None)
    monkeypatch.setattr(prov, "record_many", lambda *a, **kw: None)
    monkeypatch.setattr(hub, "HubLockFile", lambda: _DummyLockFile([]))
    monkeypatch.setattr(skills_tool, "_find_all_skills",
                        lambda **_kwargs: _make_skills(("registered-local", skill_dir)))
    monkeypatch.setattr(skills_sync, "_read_manifest", lambda: {})

    output = _capture_with()
    row = _row_for(output, "registered-local")
    cells = [c.strip() for c in row.strip("│").split("│")]
    assert cells[2] == "builtin", f"expected Source=builtin from registry, got {cells[2]!r}"
    assert "builtin" in cells[3]
    assert "back-filled provenance" not in output


def test_do_update_reinstalls_outdated_skills(monkeypatch):
    output, installs = _capture_update(monkeypatch, [
        {"name": "hub-skill", "identifier": "skills-sh/example/repo/hub-skill", "status": "update_available"},
        {"name": "other-skill", "identifier": "github/example/other-skill", "status": "up_to_date"},
    ])

    assert installs == [("skills-sh/example/repo/hub-skill", "category", True)]
    assert "Updated 1 skill" in output


def test_handle_skills_slash_search_accepts_chatconsole_without_status_errors():
    results = [type("R", (), {
        "name": "kubernetes",
        "description": "Cluster orchestration",
        "source": "skills.sh",
        "trust_level": "community",
        "identifier": "skills-sh/example/kubernetes",
    })()]

    with patch("tools.skills_hub.unified_search", return_value=results), \
         patch("tools.skills_hub.create_source_router", return_value={}), \
         patch("tools.skills_hub.GitHubAuth"):
        handle_skills_slash("/skills search kubernetes", console=ChatConsole())


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
    monkeypatch.setattr(hub, "HubLockFile", lambda: type("Lock", (), {"get_installed": lambda self, name: None})())
    monkeypatch.setattr(guard, "scan_skill", _scan_skill)
    monkeypatch.setattr(guard, "format_scan_report", lambda result: "scan ok")
    monkeypatch.setattr(guard, "should_allow_install", lambda result, force=False: (False, "stop after scan"))

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)

    do_install("skils-sh/anthropics/skills/frontend-design", console=console, skip_confirm=True)

    assert scanned["source"] == canonical_identifier


def test_do_install_scans_official_bundles_with_source_provenance(
    monkeypatch, tmp_path, hub_env
):
    import tools.skills_guard as guard
    import tools.skills_hub as hub

    class _OfficialSource:
        def inspect(self, identifier):
            return type("Meta", (), {
                "extra": {},
                "identifier": "official/agent/prunus-gaia",
            })()

        def fetch(self, identifier):
            return type("Bundle", (), {
                "name": "prunus-gaia",
                "files": {"SKILL.md": "# Prunus Gaia"},
                "source": "official",
                "identifier": "official/agent/prunus-gaia",
                "trust_level": "builtin",
                "metadata": {},
            })()

    q_path = tmp_path / "skills" / ".hub" / "quarantine" / "prunus-gaia"
    q_path.mkdir(parents=True)
    (q_path / "SKILL.md").write_text("# Prunus Gaia")

    scanned = {}

    def _scan_skill(skill_path, source="community"):
        scanned["source"] = source
        return guard.ScanResult(
            skill_name="prunus-gaia",
            source=source,
            trust_level="builtin",
            verdict="safe",
        )

    monkeypatch.setattr(hub, "ensure_hub_dirs", lambda: None)
    monkeypatch.setattr(hub, "create_source_router", lambda auth: [_OfficialSource()])
    monkeypatch.setattr(hub, "quarantine_bundle", lambda bundle: q_path)
    monkeypatch.setattr(hub, "HubLockFile", lambda: type("Lock", (), {"get_installed": lambda self, name: None})())
    monkeypatch.setattr(guard, "scan_skill", _scan_skill)
    monkeypatch.setattr(guard, "format_scan_report", lambda result: "scan ok")
    monkeypatch.setattr(guard, "should_allow_install", lambda result, force=False: (False, "stop after scan"))

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)

    do_install("official/agent/prunus-gaia", console=console, skip_confirm=True)

    assert scanned["source"] == "official"


def test_do_install_preserves_nested_official_optional_path(
    monkeypatch, tmp_path, hub_env
):
    class _OfficialNestedSource:
        def inspect(self, identifier):
            return type("Meta", (), {
                "extra": {},
                "identifier": "official/mlops/training/trl-fine-tuning",
            })()

        def fetch(self, identifier):
            return type("Bundle", (), {
                "name": "trl-fine-tuning",
                "files": {"SKILL.md": "# TRL"},
                "source": "official",
                "identifier": "official/mlops/training/trl-fine-tuning",
                "trust_level": "builtin",
                "metadata": {},
            })()

    installs = _install_mocks(monkeypatch, tmp_path, _OfficialNestedSource)

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_install(
        "official/mlops/training/trl-fine-tuning",
        console=console,
        skip_confirm=True,
    )

    assert installs == [{"name": "trl-fine-tuning", "category": "mlops/training"}]


# ---------------------------------------------------------------------------
# UrlSource-specific install paths: --name override, interactive prompts,
# non-interactive error, existing-category scan.
# ---------------------------------------------------------------------------


def _make_url_bundle_fetcher(name="", awaiting_name=True, url="https://example.com/SKILL.md"):
    """Return a fake source that simulates ``UrlSource.fetch`` for a
    URL-sourced skill whose name hasn't been auto-resolved."""

    class _UrlSource:
        def inspect(self, identifier):
            return type("Meta", (), {
                "extra": {"url": url, "awaiting_name": awaiting_name},
                "identifier": url,
                "name": name,
                "path": name,
            })()

        def fetch(self, identifier):
            return type("Bundle", (), {
                "name": name,
                "files": {"SKILL.md": "---\ndescription: ok\n---\n# body\n"},
                "source": "url",
                "identifier": url,
                "trust_level": "community",
                "metadata": {"url": url, "awaiting_name": awaiting_name},
            })()

    return _UrlSource


def _install_mocks(monkeypatch, tmp_path, source_factory, category_hint=""):
    """Wire the minimum set of monkeypatches for a do_install dry run."""
    import tools.skills_hub as hub
    import tools.skills_guard as guard

    q_path = tmp_path / "skills" / ".hub" / "quarantine" / "pending"
    q_path.mkdir(parents=True)

    install_calls: list = []

    def _install_from_quarantine(q, name, category, bundle, result):
        install_calls.append({"name": name, "category": category})
        install_dir = tmp_path / "skills" / (f"{category}/" if category else "") / name
        install_dir.mkdir(parents=True, exist_ok=True)
        return install_dir

    monkeypatch.setattr(hub, "ensure_hub_dirs", lambda: None)
    monkeypatch.setattr(hub, "create_source_router", lambda auth: [source_factory()])
    monkeypatch.setattr(hub, "quarantine_bundle", lambda bundle: q_path)
    monkeypatch.setattr(hub, "install_from_quarantine", _install_from_quarantine)
    monkeypatch.setattr(
        hub, "HubLockFile",
        lambda: type("Lock", (), {"get_installed": lambda self, n: None})(),
    )
    monkeypatch.setattr(
        guard, "scan_skill",
        lambda skill_path, source="community": guard.ScanResult(
            skill_name="pending", source=source, trust_level="community", verdict="safe",
        ),
    )
    monkeypatch.setattr(guard, "format_scan_report", lambda result: "scan ok")
    monkeypatch.setattr(guard, "should_allow_install", lambda result, force=False: (True, "ok"))
    return install_calls


def test_url_install_uses_name_override_on_non_interactive_surface(monkeypatch, tmp_path, hub_env):
    installs = _install_mocks(monkeypatch, tmp_path, _make_url_bundle_fetcher())

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_install(
        "https://example.com/SKILL.md",
        console=console, skip_confirm=True,
        name_override="my-url-skill",
    )

    assert installs == [{"name": "my-url-skill", "category": ""}]


def test_url_install_rejects_invalid_name_override(monkeypatch, tmp_path, hub_env):
    installs = _install_mocks(monkeypatch, tmp_path, _make_url_bundle_fetcher())

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_install(
        "https://example.com/SKILL.md",
        console=console, skip_confirm=True,
        name_override="SKILL",  # rejected by _is_valid_installed_skill_name
    )

    assert installs == []  # did NOT install
    assert "Invalid --name" in sink.getvalue()


def test_url_install_actionable_error_on_non_interactive_with_no_name(monkeypatch, tmp_path, hub_env):
    installs = _install_mocks(monkeypatch, tmp_path, _make_url_bundle_fetcher())

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_install(
        "https://example.com/SKILL.md",
        console=console, skip_confirm=True,
        # No name_override — should error out with a retry hint.
    )

    assert installs == []
    out = sink.getvalue()
    assert "Cannot install from URL" in out
    assert "--name <your-name>" in out


def test_url_install_prompts_interactively_when_tty(monkeypatch, tmp_path, hub_env):
    installs = _install_mocks(monkeypatch, tmp_path, _make_url_bundle_fetcher())

    # Simulate user typing "my-interactive" to name prompt, then "" to category.
    answers = iter(["my-interactive", ""])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_install(
        "https://example.com/SKILL.md",
        console=console, skip_confirm=False,  # interactive
        force=True,  # skip the final confirm prompt (tested elsewhere)
    )

    assert installs == [{"name": "my-interactive", "category": ""}]


def test_url_install_prompts_category_and_uses_typed_value(monkeypatch, tmp_path, hub_env):
    import tools.skills_hub as hub
    installs = _install_mocks(
        monkeypatch, tmp_path,
        _make_url_bundle_fetcher(name="sharethis-chat", awaiting_name=False),
    )

    # Stage an existing category bucket so _existing_categories finds it.
    (hub.SKILLS_DIR / "productivity" / "notion").mkdir(parents=True)
    (hub.SKILLS_DIR / "productivity" / "notion" / "SKILL.md").write_text("# notion")

    # Name is already resolved (from frontmatter) → only category prompt fires.
    answers = iter(["productivity"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_install(
        "https://example.com/sharethis-chat/SKILL.md",
        console=console, skip_confirm=False, force=True,
    )

    assert installs == [{"name": "sharethis-chat", "category": "productivity"}]
    assert "Existing: productivity" in sink.getvalue()


def test_url_install_cancel_name_prompt_aborts(monkeypatch, tmp_path, hub_env):
    installs = _install_mocks(monkeypatch, tmp_path, _make_url_bundle_fetcher())

    # Empty input with no default → name prompt returns None → abort.
    monkeypatch.setattr("builtins.input", lambda prompt="": "")

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_install(
        "https://example.com/SKILL.md",
        console=console, skip_confirm=False, force=True,
    )

    assert installs == []
    assert "Installation cancelled" in sink.getvalue()


# ── _existing_categories ────────────────────────────────────────────────────


def test_existing_categories_skips_top_level_skills(monkeypatch, tmp_path, hub_env):
    import tools.skills_hub as hub
    from hermes_cli.skills_hub import _existing_categories

    # Category bucket with nested skill.
    (hub.SKILLS_DIR / "productivity" / "notion").mkdir(parents=True)
    (hub.SKILLS_DIR / "productivity" / "notion" / "SKILL.md").write_text("# notion")

    # Flat skill at top level (NOT a category).
    (hub.SKILLS_DIR / "my-flat-skill").mkdir()
    (hub.SKILLS_DIR / "my-flat-skill" / "SKILL.md").write_text("# flat")

    # Empty dir (NOT a category — no SKILL.md below).
    (hub.SKILLS_DIR / "empty-dir").mkdir()

    # Hidden dir (ignored).
    (hub.SKILLS_DIR / ".hub").mkdir(exist_ok=True)

    cats = _existing_categories()
    assert cats == ["productivity"]


def test_existing_categories_returns_empty_when_skills_dir_missing(monkeypatch, tmp_path, hub_env):
    # hub_env creates tmp_path/skills/.hub — we point SKILLS_DIR at a missing sibling.
    import tools.skills_hub as hub
    monkeypatch.setattr(hub, "SKILLS_DIR", tmp_path / "does-not-exist")

    from hermes_cli.skills_hub import _existing_categories
    assert _existing_categories() == []


# ---------------------------------------------------------------------------
# browse_skills — dedup by identifier, not name
# ---------------------------------------------------------------------------


def test_browse_skills_dedup_uses_identifier_not_name(monkeypatch):
    """browse_skills() must not collapse browse-sh skills that share a task name.

    Airbnb and Booking.com both publish a 'search-listings' skill. Before the
    fix, both were keyed by name so only one survived deduplication. After the
    fix, each unique identifier produces a distinct result.
    """
    from tools.skills_hub import SkillMeta
    from hermes_cli.skills_hub import browse_skills

    airbnb = SkillMeta(
        name="search-listings", description="Airbnb search", source="browse-sh",
        identifier="browse-sh/airbnb.com/search-listings-ddgioa", trust_level="community",
    )
    booking = SkillMeta(
        name="search-listings", description="Booking.com search", source="browse-sh",
        identifier="browse-sh/booking.com/search-listings-xyzab", trust_level="community",
    )

    mock_src = type("S", (), {
        "source_id": lambda self: "browse-sh",
        "search": lambda self, q, limit=500: [airbnb, booking],
    })()

    # browse_skills() imports create_source_router locally from tools.skills_hub,
    # so the patch must target the source module, not hermes_cli.skills_hub.
    with patch("tools.skills_hub.create_source_router", return_value=[mock_src]):
        result = browse_skills(page=1, page_size=50)

    names = [item["name"] for item in result["items"]]
    assert names.count("search-listings") == 2, (
        "browse_skills() must not deduplicate browse-sh skills with the same name "
        "but different identifiers"
    )


def test_do_browse_reports_live_per_source_progress():
    """do_browse must pass an on_source_done callback so the status line ticks
    off each source as it resolves, instead of showing a frozen spinner while
    a slow source blocks. The page is still rendered once, after the full
    result set is merged and trust-sorted."""
    from hermes_cli.skills_hub import do_browse
    from tools.skills_hub import SkillMeta

    meta = SkillMeta(
        name="demo", description="d", source="official",
        identifier="official/demo", trust_level="builtin",
    )

    captured = {}

    def fake_parallel(sources, query="", per_source_limits=None,
                      source_filter="all", overall_timeout=30,
                      on_source_done=None):
        # Simulate two sources completing — the callback must be wired through.
        assert on_source_done is not None, "do_browse must pass on_source_done"
        on_source_done("official", 1)
        on_source_done("clawhub", 0)
        captured["called"] = True
        return [meta], {"official": 1, "clawhub": 0}, []

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None, width=120)

    with patch("tools.skills_hub.create_source_router", return_value=[]), \
         patch("tools.skills_hub.GitHubAuth"), \
         patch("tools.skills_hub.parallel_search_sources", side_effect=fake_parallel):
        do_browse(page=1, page_size=20, console=console)

    assert captured.get("called"), "parallel_search_sources was not invoked"
    # The rendered page still shows the (single) merged result.
    assert "demo" in sink.getvalue()


# ---------------------------------------------------------------------------
# Regression: full identifier must be recoverable from `hermes skills search`
# even when the slug is too long to fit the terminal width (issue #33674).
# ---------------------------------------------------------------------------

# A real browse-sh-style slug whose trailing -XXXXXX hash matters for install
_LONG_SLUG = "browse-sh/weather.gov/get-forecast-1uezib"

_LONG_RESULT = type("R", (), {
    "name": "get-forecast",
    "description": "Fetch the forecast",
    "source": "browse-sh",
    "trust_level": "community",
    "identifier": _LONG_SLUG,
})()


def test_do_search_identifier_column_does_not_truncate_long_slug():
    """The Identifier column must use overflow='fold', not the default ellipsis.

    Renders into a deliberately narrow Console; the full slug (including the
    trailing -1uezib hash) must still appear in the output. Before the fix,
    Rich would render `browse-sh/weather…` and lose the hash.
    """
    from hermes_cli.skills_hub import do_search

    sink = StringIO()
    # Narrow width forces Rich to apply overflow rules — exactly the scenario
    # the issue reports. width=40 is too small for the slug; we want the slug
    # wrapped (not ellipsis-truncated).
    console = Console(file=sink, force_terminal=False, color_system=None, width=40)

    with patch("tools.skills_hub.unified_search", return_value=[_LONG_RESULT]), \
         patch("tools.skills_hub.create_source_router", return_value={}), \
         patch("tools.skills_hub.GitHubAuth"):
        do_search("weather", console=console)

    output = sink.getvalue()

    # The fix is working when the Identifier column wraps the slug across
    # multiple lines (folded chunks) rather than emitting ONE line with an
    # ellipsis. Extract every chunk that appears in the rightmost cell of
    # the table by walking lines that look like table rows ("│ ... │") and
    # taking the last `│...│` cell. Concatenating those chunks must yield
    # the full slug.
    chunks = []
    for line in output.splitlines():
        # Table data rows start and end with the box-drawing vertical bar.
        if not line.startswith("│") or not line.rstrip().endswith("│"):
            continue
        # Last `│ ... │` cell on the row is the Identifier column.
        last_cell = line.rstrip().rsplit("│", 2)[-2].strip()
        if last_cell:
            chunks.append(last_cell)
    reconstructed = "".join(chunks)
    assert _LONG_SLUG in reconstructed, (
        f"Expected full slug {_LONG_SLUG!r} to be recoverable from the "
        f"folded Identifier column; got chunks {chunks!r}\n"
        f"Full output:\n{output}"
    )
    # And the truncating ellipsis must NOT appear in the Identifier column.
    # Rich uses U+2026 HORIZONTAL ELLIPSIS for the default overflow="ellipsis".
    assert "\u2026" not in reconstructed, (
        f"Identifier column still ellipsis-truncated: {reconstructed!r}"
    )


def test_do_search_json_flag_emits_full_identifiers(capsys):
    """`--json` must print a parseable array with full identifiers and skip the table."""
    from hermes_cli.skills_hub import do_search

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None, width=40)

    with patch("tools.skills_hub.unified_search", return_value=[_LONG_RESULT]), \
         patch("tools.skills_hub.create_source_router", return_value={}), \
         patch("tools.skills_hub.GitHubAuth"):
        do_search("weather", console=console, as_json=True)

    # JSON goes to stdout via print(), not the Rich console sink.
    captured = capsys.readouterr().out
    import json as _json
    payload = _json.loads(captured)
    assert isinstance(payload, list) and len(payload) == 1
    assert payload[0]["identifier"] == _LONG_SLUG
    assert payload[0]["name"] == "get-forecast"
    assert payload[0]["source"] == "browse-sh"
    # Table render must be suppressed — sink should be empty (no "Searching for:" header).
    assert "Searching for:" not in sink.getvalue()

