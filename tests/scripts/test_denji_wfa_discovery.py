"""Tests for WFA discover_dbs() — canonical 7-DB topology enumeration.

P05 Batch 1: proves discover_dbs() enumerates all 7 canonical boards
(default→core, apps, content, core, kensei-rebuild, research, security-ops)
plus profile-scoped boards from a fixture HERMES_HOME.  Uses a temporary
HERMES_HOME — never touches live ``~/.hermes``.
"""
from __future__ import annotations

import importlib.util
import sqlite3
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "scripts"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(SCRIPTS))


def _make_board(home: Path, slug: str) -> Path:
    """Create a board directory + board.json + minimal kanban.db."""
    d = home / "kanban" / "boards" / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "board.json").write_text(
        f'{{"slug":"{slug}","name":"{slug}","archived":false}}'
    )
    db = d / "kanban.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE IF NOT EXISTS task_runs (id TEXT, task_id TEXT, profile TEXT, status TEXT, outcome TEXT, started_at TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS task_events (task_id TEXT, kind TEXT, payload TEXT, created_at TEXT)")
    conn.close()
    return db


def _make_profile_board(home: Path, profile: str, slug: str) -> Path:
    """Create a profile-scoped board under profiles/<profile>/kanban/boards/."""
    d = home / "profiles" / profile / "kanban" / "boards" / slug
    d.mkdir(parents=True, exist_ok=True)
    db = d / "kanban.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY)")
    conn.execute("CREATE TABLE IF NOT EXISTS task_runs (id TEXT, task_id TEXT, profile TEXT, status TEXT, outcome TEXT, started_at TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS task_events (task_id TEXT, kind TEXT, payload TEXT, created_at TEXT)")
    conn.close()
    return db


def _load_wfa(home: Path, monkeypatch):
    """Import denji-wfa.py with HERMES_HOME pointed at the fixture."""
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(home))
    # Also clear HERMES_KANBAN_DB so it doesn't override
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    spec = importlib.util.spec_from_file_location(
        "denji_wfa_under_test", str(SCRIPTS / "denji-wfa.py")
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_home(tmp_path, monkeypatch):
    h = tmp_path / "hermes"
    h.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(h))
    monkeypatch.delenv("HERMES_KANBAN_DB", raising=False)
    return h


class TestCanonicalDiscovery:
    """discover_dbs must enumerate the full canonical 7-DB topology."""

    def test_all_seven_canonical_boards_discovered(self, fake_home, monkeypatch):
        """Create all 7 canonical board DBs; discover_dbs must find all 7."""
        # The canonical slugs (post-compat): core, apps, content,
        # kensei-rebuild, research, security-ops.  "default" maps to core.
        for slug in (
            "core", "apps", "content", "kensei-rebuild", "research", "security-ops"
        ):
            _make_board(fake_home, slug)

        mod = _load_wfa(fake_home, monkeypatch)
        dbs = mod.discover_dbs()

        # We created 6 distinct board DBs (default resolves to core, so
        # the "default" entry and any "core" board resolve to the same path
        # and get deduplicated).  Expect 6 unique DBs.
        assert len(dbs) == 6
        boards = {db["board"] for db in dbs}
        # The canonical board labels (legacy keys preserved):
        assert "default" in boards or "core" in boards
        assert "apps" in boards
        assert "research" in boards
        assert "kensei-rebuild" in boards
        # ops → security-ops, content-lead → content
        assert "ops" in boards or "security-ops" in boards
        assert "content-lead" in boards or "content" in boards

    def test_kensei_rebuild_included(self, fake_home, monkeypatch):
        """kensei-rebuild was missing from the legacy REQUIRED_BOARDS; it
        must now be discovered."""
        _make_board(fake_home, "kensei-rebuild")
        mod = _load_wfa(fake_home, monkeypatch)
        dbs = mod.discover_dbs()
        boards = {db["board"] for db in dbs}
        assert "kensei-rebuild" in boards

    def test_legacy_slug_resolves_to_canonical(self, fake_home, monkeypatch):
        """When only the canonical board exists (no legacy), the legacy slug
        resolves to it."""
        _make_board(fake_home, "security-ops")
        mod = _load_wfa(fake_home, monkeypatch)
        dbs = mod.discover_dbs()
        paths = {db["path"] for db in dbs}
        assert any("security-ops" in p for p in paths)

    def test_default_resolves_to_core(self, fake_home, monkeypatch):
        """When no legacy kanban.db exists, default resolves to core."""
        _make_board(fake_home, "core")
        mod = _load_wfa(fake_home, monkeypatch)
        dbs = mod.discover_dbs()
        paths = [db["path"] for db in dbs]
        assert any("boards" in p and "core" in p for p in paths)

    def test_content_lead_resolves_to_content(self, fake_home, monkeypatch):
        _make_board(fake_home, "content")
        mod = _load_wfa(fake_home, monkeypatch)
        dbs = mod.discover_dbs()
        paths = {db["path"] for db in dbs}
        assert any("content" in p and "content-lead" not in p for p in paths)


class TestProfileScopedDiscovery:
    """discover_dbs must also enumerate profile-scoped boards."""

    def test_profile_scoped_board_discovered(self, fake_home, monkeypatch):
        _make_board(fake_home, "core")
        _make_profile_board(fake_home, "dezzy", "ops")
        mod = _load_wfa(fake_home, monkeypatch)
        dbs = mod.discover_dbs()
        boards = {db["board"] for db in dbs}
        assert "dezzy/ops" in boards

    def test_multiple_profile_scoped_boards(self, fake_home, monkeypatch):
        _make_board(fake_home, "core")
        _make_profile_board(fake_home, "dezzy", "ops")
        _make_profile_board(fake_home, "wesker", "default")
        _make_profile_board(fake_home, "octacon", "apps")
        mod = _load_wfa(fake_home, monkeypatch)
        dbs = mod.discover_dbs()
        boards = {db["board"] for db in dbs}
        assert "dezzy/ops" in boards
        assert "wesker/default" in boards
        assert "octacon/apps" in boards

    def test_no_profiles_dir_does_not_crash(self, fake_home, monkeypatch):
        """If profiles/ doesn't exist, discover_dbs still works."""
        _make_board(fake_home, "core")
        mod = _load_wfa(fake_home, monkeypatch)
        dbs = mod.discover_dbs()
        assert len(dbs) >= 1


class TestDedupAndExistence:
    """discover_dbs must deduplicate by resolved path and skip non-existent."""

    def test_deduplication_by_resolved_path(self, fake_home, monkeypatch):
        """If default and core resolve to the same path, only one entry."""
        _make_board(fake_home, "core")
        mod = _load_wfa(fake_home, monkeypatch)
        dbs = mod.discover_dbs()
        paths = [db["path"] for db in dbs]
        assert len(paths) == len(set(paths))  # no duplicates

    def test_nonexistent_boards_skipped(self, fake_home, monkeypatch):
        """Boards that don't exist on disk are skipped."""
        mod = _load_wfa(fake_home, monkeypatch)
        dbs = mod.discover_dbs()
        assert len(dbs) == 0


class TestNoStaticLabelReliance:
    """discover_dbs must not rely on the legacy REQUIRED_BOARDS list alone."""

    def test_canonical_boards_constant_exists(self, fake_home, monkeypatch):
        mod = _load_wfa(fake_home, monkeypatch)
        assert hasattr(mod, "CANONICAL_BOARDS")
        assert "kensei-rebuild" in mod.CANONICAL_BOARDS

    def test_required_boards_is_alias(self, fake_home, monkeypatch):
        """REQUIRED_BOARDS is kept as a back-compat alias."""
        mod = _load_wfa(fake_home, monkeypatch)
        assert mod.REQUIRED_BOARDS == mod.CANONICAL_BOARDS


# ═══════════════════════════════════════════════════════════════════════════
# P1.1 defect witnesses: shared-skill visibility, path normalisation,
# and duplicate finding behaviour in analyse_db()/profile_skill_names().
# ═══════════════════════════════════════════════════════════════════════════

def _make_full_board(home: Path, slug: str) -> Path:
    """Board DB with the full tasks/task_runs/task_events schema WFA needs."""
    d = home / "kanban" / "boards" / slug
    d.mkdir(parents=True, exist_ok=True)
    (d / "board.json").write_text(
        f'{{"slug":"{slug}","name":"{slug}","archived":false}}'
    )
    db = d / "kanban.db"
    conn = sqlite3.connect(str(db))
    conn.execute("CREATE TABLE IF NOT EXISTS tasks (id TEXT PRIMARY KEY, title TEXT, body TEXT, assignee TEXT, status TEXT, created_at TEXT, updated_at TEXT, completed_at TEXT, result TEXT, skills TEXT, status_reason TEXT, last_failure_error TEXT, current_run_id TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS task_runs (id TEXT PRIMARY KEY, task_id TEXT, profile TEXT, status TEXT, outcome TEXT, summary TEXT, error TEXT, metadata TEXT, started_at TEXT, ended_at TEXT)")
    conn.execute("CREATE TABLE IF NOT EXISTS task_events (task_id TEXT, kind TEXT, payload TEXT, created_at TEXT, id INTEGER PRIMARY KEY AUTOINCREMENT)")
    conn.close()
    return db


def _insert_task(db: Path, task_id: str, assignee: str, skills: str) -> None:
    """Insert a task with a forced-skill list into the board DB."""
    conn = sqlite3.connect(str(db))
    conn.execute(
        "INSERT INTO tasks (id, title, assignee, status, skills) VALUES (?, ?, ?, ?, ?)",
        (task_id, "P1.1 witness", assignee, "in_progress", skills),
    )
    conn.commit()
    conn.close()


def _shared_skill_dir(home: Path, name: str) -> Path:
    """Create a shared skill dir with a SKILL.md under home/shared-skills/."""
    d = home / "shared-skills" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\nname: {name}\n---\nbody")
    return d


class TestSharedSkillVisibilityViaExternalDirs:
    """profile_skill_names() must accept skills resolved via skills.external_dirs.

    Defect witness: profiles that resolve shared skills through
    skills.external_dirs in their config.yaml currently get
    'forced_skill_not_visible_to_assignee_profile' findings even though the
    forced skill is visible to the profile at runtime.
    """

    def test_profile_with_external_dirs_sees_shared_skill(
        self, fake_home, monkeypatch
    ):
        # The shared skill lives under <home>/shared-skills, which the
        # profile's config.yaml declares as an external_dir. The profile has
        # NO local skills directory at all — external_dirs is the only source.
        shared = _shared_skill_dir(fake_home, "governance")
        assert shared.exists()
        profile_dir = fake_home / "profiles" / "denji-reviewer"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "config.yaml").write_text(
            "skills:\n"
            "  external_dirs:\n"
            f"    - {fake_home / 'shared-skills'}\n"
        )

        mod = _load_wfa(fake_home, monkeypatch)
        visible = mod.profile_skill_names("denji-reviewer")
        assert "governance" in visible, (
            "profile_skill_names must count skills resolved via "
            "skills.external_dirs as visible to the profile"
        )

    def test_profile_with_broken_external_dir_does_not_crash(
        self, fake_home, monkeypatch
    ):
        """A dangling external_dir path in config.yaml must not break the scan."""
        profile_dir = fake_home / "profiles" / "wesker"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "config.yaml").write_text(
            "skills:\n"
            "  external_dirs:\n"
            f"    - {fake_home / 'does-not-exist'}\n"
        )
        mod = _load_wfa(fake_home, monkeypatch)
        # Must not raise; local scan still works.
        assert isinstance(mod.profile_skill_names("wesker"), set)

    def test_profile_without_config_yaml_degrades_to_local_scan(
        self, fake_home, monkeypatch
    ):
        """profiles/<name>/skills still counts even when no config.yaml exists."""
        skill_dir = fake_home / "profiles" / "local-only" / "skills" / "cats"
        skill_dir.mkdir(parents=True, exist_ok=True)
        (skill_dir / "SKILL.md").write_text("---\nname: cats\n---\nbody")
        mod = _load_wfa(fake_home, monkeypatch)
        assert "cats" in mod.profile_skill_names("local-only")

    def test_analyse_db_external_dirs_skill_not_flagged(
        self, fake_home, monkeypatch
    ):
        """End to end: a task forcing an external_dirs-resolved skill on a
        profile must NOT yield forced_skill_not_visible_to_assignee_profile."""
        _make_full_board(fake_home, "core")
        _shared_skill_dir(fake_home, "governance")
        profile_dir = fake_home / "profiles" / "governance-lead"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "config.yaml").write_text(
            "skills:\n"
            "  external_dirs:\n"
            f"    - {fake_home / 'shared-skills'}\n"
        )
        db = fake_home / "kanban" / "boards" / "core" / "kanban.db"
        _insert_task(db, "t_ext_dirs", "governance-lead", '["governance"]')

        mod = _load_wfa(fake_home, monkeypatch)
        result = mod.analyse_db(
            {"board": "core", "path": str(db)}, set(), {}
        )
        kinds = [f["kind"] for f in result["findings"]]
        assert "forced_skill_not_visible_to_assignee_profile" not in kinds, (
            "external_dirs-resolved skill must be treated as visible; "
            f"got findings: {result['findings']}"
        )


class TestPathQualifiedSkillNames:
    """Forced-skill entries recorded as category-qualified paths or absolute
    paths must be normalised to their skill identity before comparison."""

    def _wfa_with_forced_skill(self, monkeypatch, fake_home, raw_skill: str):
        _make_full_board(fake_home, "core")
        _shared_skill_dir(fake_home, "governance")
        profile_dir = fake_home / "profiles" / "governance-lead"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "config.yaml").write_text(
            "skills:\n"
            "  external_dirs:\n"
            f"    - {fake_home / 'shared-skills'}\n"
        )
        db = fake_home / "kanban" / "boards" / "core" / "kanban.db"
        _insert_task(db, "t_pathqual", "governance-lead", f'["{raw_skill}"]')
        mod = _load_wfa(fake_home, monkeypatch)
        return mod.analyse_db({"board": "core", "path": str(db)}, set(), {})

    def test_category_qualified_name_matches_bare_skill(
        self, fake_home, monkeypatch
    ):
        """'devops/governance' under an external_dir whose layout includes a
        devops/ category directory must match the bare skill identity."""
        # give the shared dir a category layout: shared-skills/devops/governance/
        cat = fake_home / "shared-skills" / "devops" / "governance"
        cat.mkdir(parents=True, exist_ok=True)
        (cat / "SKILL.md").write_text("---\nname: governance\n---\nbody")
        result = self._wfa_with_forced_skill(monkeypatch, fake_home, "devops/governance")
        kinds = [f["kind"] for f in result["findings"]]
        assert "forced_skill_not_visible_to_assignee_profile" not in kinds

    def test_absolute_path_skill_name_matches_bare_skill(
        self, fake_home, monkeypatch
    ):
        """A task skill recorded as an absolute path to a visible skill must
        not be flagged."""
        result = self._wfa_with_forced_skill(
            monkeypatch, fake_home,
            str(fake_home / "shared-skills" / "governance" / "SKILL.md"),
        )
        kinds = [f["kind"] for f in result["findings"]]
        assert "forced_skill_not_visible_to_assignee_profile" not in kinds

    def test_absolute_dir_path_skill_name_matches_bare_skill(
        self, fake_home, monkeypatch
    ):
        """A task skill recorded as an absolute dir path (no SKILL.md suffix)."""
        result = self._wfa_with_forced_skill(
            monkeypatch, fake_home,
            str(fake_home / "shared-skills" / "governance"),
        )
        kinds = [f["kind"] for f in result["findings"]]
        assert "forced_skill_not_visible_to_assignee_profile" not in kinds

    def test_genuinely_unknown_skill_still_flagged(
        self, fake_home, monkeypatch
    ):
        """Normalisation must not swallow real defects: a skill that exists
        nowhere is still missing."""
        result = self._wfa_with_forced_skill(
            monkeypatch, fake_home, "totally-bogus-skill"
        )
        kinds = [f["kind"] for f in result["findings"]]
        assert "forced_skill_not_visible_to_assignee_profile" in kinds
        # The raw string is preserved in evidence for humans.
        skill_findings = [
            f for f in result["findings"]
            if f["kind"] == "forced_skill_not_visible_to_assignee_profile"
        ]
        assert skill_findings, "expected at least one missing-skill finding"
        all_reported = [s for f in skill_findings for s in f["evidence"]["missing_skills"]]
        assert "totally-bogus-skill" in all_reported


class TestDuplicateFindingDedup:
    """The same board+task_id+kind must appear once per finding, with all
    source evidence merged into that single finding."""

    def _wfa_with_dup_sources(self, monkeypatch, fake_home):
        _make_full_board(fake_home, "core")
        profile_dir = fake_home / "profiles" / "solo"
        profile_dir.mkdir(parents=True, exist_ok=True)
        # No config.yaml, no skills dir: everything is invisible.
        db = fake_home / "kanban" / "boards" / "core" / "kanban.db"
        _insert_task(db, "t_dupsources", "solo", '["alpha-skill"]')
        # Also seed a created event forcing the same skill.
        conn = sqlite3.connect(str(db))
        conn.execute(
            "INSERT INTO task_events (task_id, kind, payload, created_at) VALUES (?, ?, ?, ?)",
            (
                "t_dupsources",
                "created",
                '{"skills": ["alpha-skill"]}',
                "2026-08-31T00:00:00Z",
            ),
        )
        conn.commit()
        conn.close()
        mod = _load_wfa(fake_home, monkeypatch)
        return mod.analyse_db({"board": "core", "path": str(db)}, set(), {})

    def test_same_kind_task_reported_once(self, fake_home, monkeypatch):
        """task.skills and created_event.skills forcing the same skill must
        produce ONE forced_skill_not_visible finding, not two."""
        result = self._wfa_with_dup_sources(monkeypatch, fake_home)
        dup_findings = [
            f for f in result["findings"]
            if f["kind"] == "forced_skill_not_visible_to_assignee_profile"
        ]
        assert len(dup_findings) == 1, (
            f"expected exactly one deduplicated finding, got {len(dup_findings)}: "
            f"{[f['evidence'].get('source') for f in dup_findings]}"
        )

    def test_merged_finding_preserves_all_source_evidence(
        self, fake_home, monkeypatch
    ):
        """The surviving finding must name every source that requested the
        skill, so deduplicate does not destroy evidence."""
        result = self._wfa_with_dup_sources(monkeypatch, fake_home)
        dup_findings = [
            f for f in result["findings"]
            if f["kind"] == "forced_skill_not_visible_to_assignee_profile"
        ]
        assert len(dup_findings) == 1
        evidence = dup_findings[0]["evidence"]
        sources = evidence.get("sources") or ([evidence["source"]] if evidence.get("source") else [])
        assert set(sources) == {"task.skills", "created_event.skills"}, (
            f"merged finding must name both sources; got {sources}"
        )

    def test_finding_preserves_missing_skills_list(self, fake_home, monkeypatch):
        """missing_skills must still be present on the merged finding."""
        result = self._wfa_with_dup_sources(monkeypatch, fake_home)
        dup_findings = [
            f for f in result["findings"]
            if f["kind"] == "forced_skill_not_visible_to_assignee_profile"
        ]
        assert dup_findings[0]["evidence"]["missing_skills"] == ["alpha-skill"]
