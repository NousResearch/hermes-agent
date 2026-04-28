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
    assert "1 hub-installed, 1 builtin, 1 local" in output


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

    assert "hub-skill" not in output
    assert "builtin-skill" in output
    assert "local-skill" in output
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
