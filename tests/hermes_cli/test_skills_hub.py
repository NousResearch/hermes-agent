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
    monkeypatch.setattr(cli_hub, "do_install", lambda identifier, category="", force=False, console=None, source_id=None: installs.append((identifier, category, force)))

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


def test_do_update_pins_install_to_locked_source(monkeypatch):
    """do_update must forward the lockfile's `source` to do_install.

    Without the pin, a bare identifier like "reddit" reaches
    _resolve_short_name()'s fuzzy catalog search inside do_install and can
    resolve to a same-named skill in another registry.
    """
    import tools.skills_hub as hub
    import hermes_cli.skills_hub as cli_hub

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    calls = []

    monkeypatch.setattr(hub, "check_for_skill_updates", lambda **_kwargs: [
        {"name": "reddit", "identifier": "reddit", "source": "clawhub",
         "status": "update_available"},
    ])
    monkeypatch.setattr(hub, "HubLockFile", lambda: type("L", (), {
        "get_installed": lambda self, name: {"install_path": "category/" + name}
    })())
    monkeypatch.setattr(
        cli_hub, "do_install",
        lambda identifier, category="", force=False, console=None, source_id=None:
            calls.append({"identifier": identifier, "source_id": source_id}),
    )

    do_update(console=console)

    assert calls == [{"identifier": "reddit", "source_id": "clawhub"}]


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

