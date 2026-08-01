from io import StringIO
from unittest.mock import patch

import pytest
from rich.console import Console

from cli import ChatConsole
from hermes_cli.skills_hub import do_audit, do_check, do_install, do_list, do_update, handle_skills_slash


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


def test_do_audit_replays_official_lock_entries_with_source_provenance(
    monkeypatch, tmp_path, hub_env
):
    import tools.skills_guard as guard
    import tools.skills_hub as hub

    skill_dir = hub.SKILLS_DIR / "autonomous-ai-agents" / "honcho"
    skill_dir.mkdir(parents=True)
    skill_text = "# Honcho\nRun `hermes honcho setup`.\n"
    (skill_dir / "SKILL.md").write_text(skill_text)

    entry = {
        "name": "honcho",
        "source": "official",
        "identifier": "official/autonomous-ai-agents/honcho",
        "trust_level": "builtin",
        "install_path": "autonomous-ai-agents/honcho",
    }
    official_bundle = hub.SkillBundle(
        name="honcho",
        files={"SKILL.md": skill_text},
        source="official",
        identifier=entry["identifier"],
        trust_level="builtin",
    )
    scanned = {}

    def _scan_skill(skill_path, source="community"):
        scanned["source"] = source
        return guard.ScanResult(
            skill_name="honcho",
            source=source,
            trust_level="builtin" if source == "official" else "community",
            verdict="dangerous",
        )

    monkeypatch.setattr(hub, "HubLockFile", lambda: _DummyLockFile([entry]))
    monkeypatch.setattr(hub.OptionalSkillSource, "fetch", lambda _self, _identifier: official_bundle)
    monkeypatch.setattr(guard, "scan_skill", _scan_skill)
    monkeypatch.setattr(
        guard,
        "format_scan_report",
        lambda result: f"source={result.source} trust={result.trust_level}",
    )

    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    do_audit("honcho", console=console)

    assert scanned["source"] == "official"
    assert "trust=builtin" in sink.getvalue()


def test_do_audit_scans_verified_official_snapshot(monkeypatch, tmp_path, hub_env):
    import tools.skills_guard as guard
    import tools.skills_hub as hub

    skill_dir = hub.SKILLS_DIR / "devops" / "demo"
    skill_dir.mkdir(parents=True)
    official_text = "# Official\n"
    installed_file = skill_dir / "SKILL.md"
    installed_file.write_text(official_text)
    entry = {
        "name": "demo",
        "source": "official",
        "identifier": "official/devops/demo",
        "trust_level": "builtin",
        "install_path": "devops/demo",
    }
    official_bundle = hub.SkillBundle(
        name="demo",
        files={"SKILL.md": official_text},
        source="official",
        identifier=entry["identifier"],
        trust_level="builtin",
    )
    scanned = {}

    def _scan_skill(skill_path, source="community"):
        installed_file.write_text("# Replaced after verification\n")
        scanned["path"] = skill_path
        scanned["source"] = source
        scanned["content"] = (skill_path / "SKILL.md").read_text()
        return guard.ScanResult(
            skill_name="demo",
            source=source,
            trust_level="builtin" if source == "official" else "community",
            verdict="safe",
        )

    monkeypatch.setattr(hub, "HubLockFile", lambda: _DummyLockFile([entry]))
    monkeypatch.setattr(
        hub.OptionalSkillSource,
        "fetch",
        lambda _self, _identifier: official_bundle,
    )
    monkeypatch.setattr(guard, "scan_skill", _scan_skill)
    monkeypatch.setattr(guard, "format_scan_report", lambda _result: "scan complete")

    do_audit("demo", console=Console(file=StringIO(), force_terminal=False))

    assert scanned["path"] != skill_dir
    assert scanned["source"] == "official"
    assert scanned["content"] == official_text
    assert installed_file.read_text() == "# Replaced after verification\n"


def test_audit_path_is_redirect_detects_windows_reparse_points():
    import stat
    from types import SimpleNamespace

    from hermes_cli.skills_hub import _audit_path_is_redirect

    class JunctionLikePath:
        def is_symlink(self):
            return False

        def lstat(self):
            return SimpleNamespace(st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)

    assert _audit_path_is_redirect(JunctionLikePath()) is True  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("entry", "expected"),
    [
        pytest.param(
            {
                "source": "github",
                "identifier": "github/example/community-skill",
                "trust_level": "community",
                "scan_source": "official",
            },
            "github/example/community-skill",
            id="ignore-stored-community-source",
        ),
        pytest.param(
            {
                "source": "skills.sh",
                "identifier": "skills-sh/anthropics/skills/frontend-design",
                "trust_level": "trusted",
                "scan_source": "anthropics/skills/frontend-design",
            },
            "skills-sh/anthropics/skills/frontend-design",
            id="derive-trusted-source-from-identifier",
        ),
        pytest.param(
            {
                "source": "github",
                "identifier": "github/example/community-skill",
                "trust_level": "builtin",
                "scan_source": "official",
            },
            "github/example/community-skill",
            id="reject-forged-builtin-trust",
        ),
        pytest.param(
            {
                "source": "official",
                "identifier": "github/example/community-skill",
                "trust_level": "builtin",
            },
            "community",
            id="reject-forged-official-source",
        ),
        pytest.param(
            {"source": "official", "trust_level": "builtin"},
            "community",
            id="reject-missing-official-identifier",
        ),
        pytest.param(
            {
                "source": "github",
                "identifier": "official",
                "trust_level": "community",
            },
            "community",
            id="reject-bare-official-identifier",
        ),
    ],
)
def test_audit_scan_source_rejects_unverified_lock_provenance(entry, expected):
    from hermes_cli.skills_hub import _audit_scan_source_for_lock_entry

    assert _audit_scan_source_for_lock_entry(entry) == expected


def test_audit_scan_source_requires_package_bound_optional_skills_root(monkeypatch, tmp_path):
    import tools.skills_hub as hub
    from hermes_cli.skills_hub import _audit_scan_source_for_lock_entry

    hostile_root = tmp_path / "hostile-optional-skills"
    hostile_skill = hostile_root / "devops" / "demo"
    hostile_skill.mkdir(parents=True)
    hostile_text = "# Hostile replacement\n"
    (hostile_skill / "SKILL.md").write_text(hostile_text)

    installed_skill = tmp_path / "installed" / "demo"
    installed_skill.mkdir(parents=True)
    (installed_skill / "SKILL.md").write_text(hostile_text)
    monkeypatch.setenv("HERMES_OPTIONAL_SKILLS", str(hostile_root))

    entry = {
        "source": "official",
        "identifier": "official/devops/demo",
        "trust_level": "builtin",
    }

    assert hub.OptionalSkillSource()._optional_dir == hostile_root
    assert _audit_scan_source_for_lock_entry(entry, installed_skill) == "community"


@pytest.mark.parametrize("installed_kind", ["modified", "symlink", "hash-collision"])
def test_audit_scan_source_downgrades_noncanonical_official_content(
    installed_kind, monkeypatch, tmp_path
):
    import tools.skills_hub as hub
    from hermes_cli.skills_hub import _audit_scan_source_for_lock_entry

    official_text = "# Official\n"
    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    bundle_files: dict[str, str | bytes]
    if installed_kind == "modified":
        (skill_path / "SKILL.md").write_text("# Modified\n")
        bundle_files = {"SKILL.md": official_text}
    elif installed_kind == "symlink":
        target = tmp_path / "target.md"
        target.write_text(official_text)
        try:
            (skill_path / "SKILL.md").symlink_to(target)
        except OSError:
            pytest.skip("symlinks unavailable")
        bundle_files = {"SKILL.md": official_text}
    else:
        payload = b"payload\n"
        (skill_path / "SKILL.md").write_bytes(
            official_text.encode() + b"references/payload.md\x00" + payload
        )
        bundle_files = {
            "SKILL.md": official_text,
            "references/payload.md": payload,
        }

    entry = {
        "source": "official",
        "identifier": "official/devops/demo",
        "trust_level": "builtin",
    }
    official_bundle = hub.SkillBundle(
        name="demo",
        files=bundle_files,
        source="official",
        identifier=entry["identifier"],
        trust_level="builtin",
    )
    monkeypatch.setattr(
        hub.OptionalSkillSource,
        "fetch",
        lambda _self, _identifier: official_bundle,
    )

    assert _audit_scan_source_for_lock_entry(entry, skill_path) == "community"


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

