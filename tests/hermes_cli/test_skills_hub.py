from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console

from cli import ChatConsole
from hermes_cli.skills_hub import do_check, do_install, do_list, do_update, do_audit, handle_skills_slash


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
    monkeypatch.setattr(cli_hub, "do_install", lambda identifier, category="", force=False, console=None, source_id=None: installs.append((identifier, category, force)))

    do_update(console=console)
    return sink.getvalue(), installs


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------




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


# ---------------------------------------------------------------------------
# Cross-registry hijack regression tests
#
# An update must never change a skill's source registry. Skill names are not
# namespaced across registries, so an unconstrained name resolve can install a
# different author's same-named skill over the user's files.
# ---------------------------------------------------------------------------




def test_check_for_skill_updates_does_not_fall_back_across_registries():
    """An entry whose source has no adapter reports `unavailable`.

    Previously `candidate_sources ... or sources` fell back to every source, so
    a same-named skill in another registry could satisfy the fetch and be
    reported as this entry's update -- the step that preceded the overwrite.
    The foreign source here returns a *valid* bundle with a different hash, so
    the old code reports `update_available` (sourced from the wrong registry)
    while the fixed code reports `unavailable`.
    """
    from tools.skills_hub import check_for_skill_updates

    class _ForeignBundle:
        name = "reddit"
        files = {"SKILL.md": "# a different author's reddit skill"}
        source = "skills.sh"
        identifier = "skills-sh/someone-else/reddit"
        trust_level = "community"
        metadata: dict = {}

    class _ForeignSource:
        """skills-sh adapter; must NOT be consulted for a clawhub-locked entry."""

        def source_id(self):
            return "skills-sh"

        def fetch(self, identifier):
            return _ForeignBundle()

        def inspect(self, identifier):
            return _ForeignBundle()

    lock = _DummyLockFile([
        {"name": "reddit", "identifier": "reddit", "source": "clawhub",
         "content_hash": "hash-of-the-clawhub-copy"},
    ])

    results = check_for_skill_updates(
        lock=lock,  # type: ignore[arg-type]  # duck-typed double, matches _DummyLockFile usage above
        sources=[_ForeignSource()],  # type: ignore[list-item]
    )

    assert len(results) == 1
    assert results[0]["source"] == "clawhub", "provenance must be preserved"
    assert results[0]["status"] == "unavailable", (
        "a clawhub-locked skill must not be matched against a skills-sh bundle; "
        "reporting update_available here is the cross-registry hijack"
    )
    assert "bundle" not in results[0], "must not carry a foreign registry's bundle"


def test_resolve_does_not_pair_catalog_meta_with_foreign_same_name_bundle():
    """Inspect metadata from one registry must not ship with another registry's files.

    skills.sh can inspect ``owner/repo/skills/skillopt`` while ClawHub used to
    fetch by last path segment ``skillopt`` and return a different author's
    SKILL.md. The header then showed the requested identifier and the preview
    showed the wrong skill.
    """
    from hermes_cli.skills_hub import _resolve_source_meta_and_bundle
    from tools.skills_hub import SkillBundle, SkillMeta

    class CatalogSource:
        def inspect(self, identifier):
            return SkillMeta(
                name="skillopt",
                description="kanban-based pipelines",
                source="skills.sh",
                identifier="skills-sh/latipun7/agent-skill-collections/skills/skillopt",
                trust_level="community",
            )

        def fetch(self, identifier):
            return None

    class ForeignSlugSource:
        def inspect(self, identifier):
            return SkillMeta(
                name="skillopt",
                description="Train, evaluate, and improve Agent skill files",
                source="clawhub",
                identifier="skillopt",
                trust_level="community",
            )

        def fetch(self, identifier):
            return SkillBundle(
                name="skillopt",
                files={"SKILL.md": "# Train, evaluate, and improve Agent skill files\n"},
                source="clawhub",
                identifier="skillopt",
                trust_level="community",
            )

    meta, bundle, matched = _resolve_source_meta_and_bundle(
        "latipun7/agent-skill-collections/skills/skillopt",
        [CatalogSource(), ForeignSlugSource()],
    )

    assert bundle is not None
    assert bundle.source == "clawhub"
    assert meta is not None
    assert meta.source == "clawhub"
    assert meta.identifier == "skillopt"
    assert "kanban-based" not in (meta.description or "")
    assert matched is not None
    assert matched.__class__ is ForeignSlugSource


def test_resolve_keeps_catalog_meta_when_later_sources_do_not_fetch():
    from hermes_cli.skills_hub import _resolve_source_meta_and_bundle
    from tools.skills_hub import SkillMeta

    class CatalogSource:
        def inspect(self, identifier):
            return SkillMeta(
                name="skillopt",
                description="kanban-based pipelines",
                source="skills.sh",
                identifier="skills-sh/latipun7/agent-skill-collections/skills/skillopt",
                trust_level="community",
            )

        def fetch(self, identifier):
            return None

    class QuietSource:
        def inspect(self, identifier):
            return None

        def fetch(self, identifier):
            return None

    meta, bundle, matched = _resolve_source_meta_and_bundle(
        "latipun7/agent-skill-collections/skills/skillopt",
        [CatalogSource(), QuietSource()],
    )

    assert bundle is None
    assert meta is not None
    assert meta.source == "skills.sh"
    assert meta.identifier.endswith("latipun7/agent-skill-collections/skills/skillopt")
    assert matched is not None
    assert matched.__class__ is CatalogSource




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






# ── _existing_categories ────────────────────────────────────────────────────






# ---------------------------------------------------------------------------
# browse_skills — dedup by identifier, not name
# ---------------------------------------------------------------------------




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



# ---------------------------------------------------------------------------
# Local-edit protection in do_update (ported from paperclipai/paperclip#10978)
# ---------------------------------------------------------------------------


def _update_env(monkeypatch, tmp_path, *, edit_after_install: bool):
    """Install a fake hub skill on disk, optionally edit it, and wire mocks.

    Returns (console_sink, installs_list).
    """
    import hermes_cli.skills_hub as cli_hub
    import tools.skills_hub as hub
    from tools.skills_guard import content_hash

    skills_dir = tmp_path / "skills"
    skill_dir = skills_dir / "category" / "hub-skill"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# hub-skill\noriginal\n")

    recorded = content_hash(skill_dir)
    if edit_after_install:
        (skill_dir / "SKILL.md").write_text("# hub-skill\nuser edited\n")

    monkeypatch.setattr(hub, "SKILLS_DIR", skills_dir)
    monkeypatch.setattr(hub, "check_for_skill_updates", lambda **_kwargs: [{
        "name": "hub-skill",
        "identifier": "someone/hub-skill",
        "source": "github",
        "status": "update_available",
    }])
    monkeypatch.setattr(hub, "HubLockFile", lambda: type("L", (), {
        "get_installed": lambda self, name: {
            "install_path": "category/hub-skill",
            "content_hash": recorded,
        }
    })())

    installs = []
    monkeypatch.setattr(
        cli_hub, "do_install",
        lambda identifier, category="", force=False, console=None, source_id=None:
            installs.append(identifier),
    )

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    return console, sink, installs


def test_do_update_skips_locally_edited_skill(monkeypatch, tmp_path):
    """A hub skill whose on-disk hash drifted from the lockfile is skipped."""
    console, sink, installs = _update_env(monkeypatch, tmp_path, edit_after_install=True)

    do_update(console=console)

    assert installs == []
    out = sink.getvalue()
    assert "local edits" in out
    assert "--force" in out


def test_do_update_force_overwrites_local_edits(monkeypatch, tmp_path):
    """--force restores the destructive replace for edited skills."""
    console, sink, installs = _update_env(monkeypatch, tmp_path, edit_after_install=True)

    do_update(console=console, force=True)

    assert installs == ["someone/hub-skill"]
    assert "local edits" not in sink.getvalue()


def test_do_update_unmodified_skill_updates_normally(monkeypatch, tmp_path):
    """No local drift -> the update proceeds without --force."""
    console, sink, installs = _update_env(monkeypatch, tmp_path, edit_after_install=False)

    do_update(console=console)

    assert installs == ["someone/hub-skill"]
    assert "Updated 1 skill(s)" in sink.getvalue()


# ---------------------------------------------------------------------------
# do_audit: stale hub-path repair + ambiguity refusal (wiring)
# ---------------------------------------------------------------------------


def _make_skill(dir_path, name, body="body"):
    dir_path.mkdir(parents=True, exist_ok=True)
    (dir_path / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: d\n---\n\n{body}\n", encoding="utf-8"
    )
    return dir_path


def _audit_console():
    sink = StringIO()
    return Console(file=sink, force_terminal=False, color_system=None), sink


def test_do_audit_repairs_stale_hub_path(hub_env, monkeypatch, tmp_path):
    """A moved skill (lock points at old top-level path) is repaired, then scanned."""
    import tools.skills_hub as hub

    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir / "mlops" / "alpha", "alpha")

    lock = hub.HubLockFile()
    lock.record_install(
        name="alpha", source="github", identifier="x",
        trust_level="trusted", scan_verdict="pass",
        skill_hash="h", install_path="alpha", files=["SKILL.md"],
    )

    monkeypatch.setattr("tools.skills_guard.scan_skill", lambda path, source=None: "R")
    monkeypatch.setattr("tools.skills_guard.format_scan_report", lambda result: "scan ok")

    console, sink = _audit_console()
    do_audit(console=console)

    out = sink.getvalue()
    assert "Repaired hub path for alpha" in out
    assert "mlops/alpha" in out
    # Lock was rewritten to the repaired path via the validated updater.
    assert lock.get_installed("alpha")["install_path"] == "mlops/alpha"


def test_do_audit_refuses_ambiguous_stale_path(hub_env, monkeypatch, tmp_path):
    """Two same-named skills => audit reports ambiguity and does not repoint the lock."""
    import tools.skills_hub as hub

    skills_dir = tmp_path / "skills"
    _make_skill(skills_dir / "mlops" / "alpha", "alpha", body="one")
    _make_skill(skills_dir / "research" / "alpha", "alpha", body="two")

    lock = hub.HubLockFile()
    lock.record_install(
        name="alpha", source="github", identifier="x",
        trust_level="trusted", scan_verdict="pass",
        skill_hash="h", install_path="alpha", files=["SKILL.md"],
    )

    monkeypatch.setattr("tools.skills_guard.scan_skill", lambda path, source=None: "R")
    monkeypatch.setattr("tools.skills_guard.format_scan_report", lambda result: "scan ok")

    console, sink = _audit_console()
    do_audit(console=console)

    out = sink.getvalue()
    assert "ambiguous" in out.lower()
    # Lock path left untouched: no silent repoint to either candidate.
    assert lock.get_installed("alpha")["install_path"] == "alpha"
