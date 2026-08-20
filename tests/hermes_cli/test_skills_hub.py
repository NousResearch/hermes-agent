from io import StringIO
from pathlib import Path
from typing import cast
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


def test_do_audit_replays_matching_install_attestation_without_refetch(
    monkeypatch, tmp_path, hub_env
):
    import tools.skills_guard as guard
    import tools.skills_hub as hub

    skill_dir = hub.SKILLS_DIR / "autonomous-ai-agents" / "honcho"
    skill_dir.mkdir(parents=True)
    skill_text = "# Honcho\nRun `hermes honcho setup`.\n"
    (skill_dir / "SKILL.md").write_text(skill_text)
    identifier = "official/autonomous-ai-agents/honcho"

    entry = {
        "name": "honcho",
        "source": "official",
        "identifier": identifier,
        "trust_level": "builtin",
        "install_path": "autonomous-ai-agents/honcho",
        "install_attestation": guard.build_install_attestation(
            skill_dir,
            source="official",
            identifier=identifier,
            trust_level="builtin",
            origin_identity="git:NousResearch/hermes-agent@" + "a" * 40,
        ),
    }
    scanned = {}

    def _scan_skill(skill_path, source="community", **kwargs):
        scanned["source"] = source
        scanned["allow_origin_markers"] = kwargs.get("allow_origin_markers")
        return guard.ScanResult(
            skill_name="honcho",
            source=source,
            trust_level="builtin" if source == "official" else "community",
            verdict="dangerous",
        )

    monkeypatch.setattr(hub, "HubLockFile", lambda: _DummyLockFile([entry]))
    monkeypatch.setattr(hub.OptionalSkillSource, "fetch", lambda *_args: None)
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
    assert scanned["allow_origin_markers"] is True
    assert "trust=builtin" in sink.getvalue()


@pytest.mark.parametrize(
    "origin_identity",
    [
        "asserted-by-lock",
        "nix-store:" + "0" * 32 + "-../../forged",
    ],
)
def test_audit_rejects_unknown_origin_identity(tmp_path, origin_identity):
    import tools.skills_guard as guard
    from hermes_cli.skills_hub import _audit_scan_identity_for_lock_entry

    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    (skill_path / "SKILL.md").write_text("# Demo\n")
    identifier = "official/devops/demo"
    entry = {
        "source": "official",
        "identifier": identifier,
        "trust_level": "builtin",
        "install_attestation": guard.build_install_attestation(
            skill_path,
            source="official",
            identifier=identifier,
            trust_level="builtin",
            origin_identity=origin_identity,
        ),
    }

    assert _audit_scan_identity_for_lock_entry(entry, skill_path) == (
        "community",
        False,
    )


def test_audit_rejects_legacy_scan_provenance_as_official_proof(tmp_path):
    import tools.skills_guard as guard
    from hermes_cli.skills_hub import _audit_scan_identity_for_lock_entry

    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    (skill_path / "SKILL.md").write_text("# Demo\n")
    identifier = "official/devops/demo"
    entry = {
        "source": "official",
        "identifier": identifier,
        "trust_level": "builtin",
        "scan_provenance": {
            "source": "official",
            "trust_level": "builtin",
            "bundle_hash": guard.full_content_hash(skill_path),
            "scanner_version": guard.SCANNER_VERSION,
        },
    }

    assert _audit_scan_identity_for_lock_entry(entry, skill_path) == (
        "community",
        False,
    )


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
        "install_attestation": guard.build_install_attestation(
            skill_dir,
            source="official",
            identifier="official/devops/demo",
            trust_level="builtin",
            origin_identity="git:NousResearch/hermes-agent@" + "b" * 40,
        ),
    }
    scanned = {}

    def _scan_skill(skill_path, source="community", **_kwargs):
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
    monkeypatch.setattr(guard, "scan_skill", _scan_skill)
    monkeypatch.setattr(guard, "format_scan_report", lambda _result: "scan complete")

    do_audit("demo", console=Console(file=StringIO(), force_terminal=False))

    assert scanned["path"] != skill_dir
    assert scanned["source"] == "official"
    assert scanned["content"] == official_text
    assert installed_file.read_text() == "# Replaced after verification\n"


def test_do_audit_deep_scan_uses_the_same_private_snapshot(monkeypatch, hub_env):
    import tools.skills_ast_audit as ast_audit
    import tools.skills_guard as guard
    import tools.skills_hub as hub

    skill_dir = hub.SKILLS_DIR / "devops" / "demo"
    skill_dir.mkdir(parents=True)
    installed_file = skill_dir / "SKILL.md"
    installed_file.write_text("# Original\n")
    identifier = "official/devops/demo"
    entry = {
        "name": "demo",
        "source": "official",
        "identifier": identifier,
        "trust_level": "builtin",
        "install_path": "devops/demo",
        "install_attestation": guard.build_install_attestation(
            skill_dir,
            source="official",
            identifier=identifier,
            trust_level="builtin",
        ),
    }
    observed = {}

    def _scan_skill(path, source="community", **_kwargs):
        observed["guard_path"] = path
        observed["guard_content"] = (path / "SKILL.md").read_text()
        installed_file.write_text("# Changed after guard scan\n")
        return guard.ScanResult(
            skill_name="demo",
            source=source,
            trust_level="builtin",
            verdict="safe",
        )

    def _ast_scan(path):
        observed["ast_path"] = path
        observed["ast_content"] = (path / "SKILL.md").read_text()
        return object()

    monkeypatch.setattr(hub, "HubLockFile", lambda: _DummyLockFile([entry]))
    monkeypatch.setattr(guard, "scan_skill", _scan_skill)
    monkeypatch.setattr(guard, "format_scan_report", lambda _result: "guard")
    monkeypatch.setattr(ast_audit, "ast_scan_path", _ast_scan)
    monkeypatch.setattr(ast_audit, "format_ast_report", lambda *_args, **_kwargs: "ast")

    do_audit(
        "demo",
        deep=True,
        console=Console(file=StringIO(), force_terminal=False),
    )

    assert observed["guard_path"] == observed["ast_path"]
    assert observed["guard_path"] != skill_dir
    assert observed["guard_content"] == "# Original\n"
    assert observed["ast_content"] == "# Original\n"


def test_scan_skill_for_audit_rejects_redirected_source_root(tmp_path):
    from hermes_cli.skills_hub import _scan_skill_for_audit

    target = tmp_path / "target"
    target.mkdir()
    (target / "SKILL.md").write_text("# Demo\n")
    source = tmp_path / "demo"
    try:
        source.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks unavailable")

    with pytest.raises(OSError, match="redirect"):
        _scan_skill_for_audit(
            {
                "source": "official",
                "identifier": "official/demo",
                "trust_level": "builtin",
            },
            source,
            lambda *_args, **_kwargs: object(),
        )


def test_scan_skill_for_audit_rejects_source_change_during_snapshot(
    monkeypatch, tmp_path
):
    import shutil
    from hermes_cli.skills_hub import _scan_skill_for_audit

    source = tmp_path / "demo"
    source.mkdir()
    skill_md = source / "SKILL.md"
    skill_md.write_text("# Original\n")
    real_copytree = shutil.copytree

    def _copy_then_change(src, dst, **kwargs):
        result = real_copytree(src, dst, **kwargs)
        skill_md.write_text("# Changed\n")
        return result

    monkeypatch.setattr("hermes_cli.skills_hub.shutil.copytree", _copy_then_change)
    with pytest.raises(OSError, match="changed"):
        _scan_skill_for_audit(
            {
                "source": "github",
                "identifier": "github/example/demo",
                "trust_level": "community",
            },
            source,
            lambda *_args, **_kwargs: object(),
        )


@pytest.mark.parametrize(
    "snapshot_error",
    [OSError("copy failed"), ValueError("invalid snapshot content")],
    ids=["os-error", "value-error"],
)
def test_do_audit_skips_when_private_snapshot_cannot_be_created(
    monkeypatch, hub_env, snapshot_error
):
    import tools.skills_ast_audit as ast_audit
    import tools.skills_guard as guard
    import tools.skills_hub as hub

    skill_dir = hub.SKILLS_DIR / "devops" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Demo\n")
    entry = {
        "name": "demo",
        "source": "github",
        "identifier": "github/example/demo",
        "trust_level": "community",
        "install_path": "devops/demo",
    }
    calls = {"guard": 0, "ast": 0}

    def _guard(*_args, **_kwargs):
        calls["guard"] += 1
        raise AssertionError("live tree must not be scanned after snapshot failure")

    def _ast(*_args, **_kwargs):
        calls["ast"] += 1
        raise AssertionError("live tree must not be deep-scanned after snapshot failure")

    monkeypatch.setattr(hub, "HubLockFile", lambda: _DummyLockFile([entry]))
    monkeypatch.setattr(guard, "scan_skill", _guard)
    monkeypatch.setattr(ast_audit, "ast_scan_path", _ast)
    monkeypatch.setattr(
        "hermes_cli.skills_hub.shutil.copytree",
        lambda *_a, **_k: (_ for _ in ()).throw(snapshot_error),
    )

    sink = StringIO()
    do_audit(
        "demo",
        deep=True,
        console=Console(file=sink, force_terminal=False, color_system=None),
    )

    assert calls == {"guard": 0, "ast": 0}
    assert "snapshot" in sink.getvalue().lower()
    assert "skipped" in sink.getvalue().lower()


def test_do_audit_rejects_lock_path_outside_skills_root(monkeypatch, tmp_path, hub_env):
    import tools.skills_guard as guard
    import tools.skills_hub as hub

    outside = tmp_path / "outside" / "demo"
    outside.mkdir(parents=True)
    (outside / "SKILL.md").write_text("# Outside\n")
    entry = {
        "name": "demo",
        "source": "github",
        "identifier": "github/example/demo",
        "trust_level": "community",
        "install_path": str(outside),
    }
    calls = {"scan": 0}

    def _scan(*_args, **_kwargs):
        calls["scan"] += 1
        raise AssertionError("an out-of-root lock path must not be scanned")

    monkeypatch.setattr(hub, "HubLockFile", lambda: _DummyLockFile([entry]))
    monkeypatch.setattr(guard, "scan_skill", _scan)

    sink = StringIO()
    do_audit(
        "demo",
        console=Console(file=sink, force_terminal=False, color_system=None),
    )

    assert calls == {"scan": 0}
    assert "invalid" in sink.getvalue().lower()
    assert "skipped" in sink.getvalue().lower()


def test_do_audit_rejects_symlinked_parent_category(monkeypatch, tmp_path, hub_env):
    import tools.skills_guard as guard
    import tools.skills_hub as hub

    outside_category = tmp_path / "outside-devops"
    target = outside_category / "demo"
    target.mkdir(parents=True)
    (target / "SKILL.md").write_text("# Demo\n")
    hub.SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        (hub.SKILLS_DIR / "devops").symlink_to(
            outside_category,
            target_is_directory=True,
        )
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    entry = {
        "name": "demo",
        "source": "official",
        "identifier": "official/devops/demo",
        "trust_level": "builtin",
        "install_path": "devops/demo",
        "install_attestation": guard.build_install_attestation(
            target,
            source="official",
            identifier="official/devops/demo",
            trust_level="builtin",
        ),
    }
    calls = {"scan": 0}

    def _scan(*_args, **_kwargs):
        calls["scan"] += 1
        raise AssertionError("a redirected category must not be scanned")

    monkeypatch.setattr(hub, "HubLockFile", lambda: _DummyLockFile([entry]))
    monkeypatch.setattr(guard, "scan_skill", _scan)
    sink = StringIO()
    do_audit(
        "demo",
        console=Console(file=sink, force_terminal=False, color_system=None),
    )

    assert calls == {"scan": 0}
    assert "invalid" in sink.getvalue().lower()
    assert "skipped" in sink.getvalue().lower()


def test_audit_path_is_redirect_detects_windows_reparse_points():
    import stat
    from types import SimpleNamespace

    from hermes_cli.skills_hub import _audit_path_is_redirect

    class JunctionLikePath:
        def is_symlink(self):
            return False

        def lstat(self):
            return SimpleNamespace(st_file_attributes=stat.FILE_ATTRIBUTE_REPARSE_POINT)

    assert _audit_path_is_redirect(cast(Path, JunctionLikePath())) is True


def test_scan_provenance_requires_verified_official_origin():
    from hermes_cli.skills_hub import _scan_provenance_for_source

    assert _scan_provenance_for_source(
        "official",
        "official/devops/demo",
        origin_verified=False,
    ) == ("community", False)
    assert _scan_provenance_for_source(
        "official",
        "official/devops/demo",
        origin_verified=True,
        origin_identity="git:NousResearch/hermes-agent@" + "a" * 40,
    ) == ("official", True)
    assert _scan_provenance_for_source(
        "official",
        "official/devops/demo",
        origin_verified=True,
    ) == ("community", False)


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
            ("github/example/community-skill", False),
            id="ignore-stored-community-source",
        ),
        pytest.param(
            {
                "source": "skills.sh",
                "identifier": "skills-sh/anthropics/skills/frontend-design",
                "trust_level": "trusted",
                "scan_source": "anthropics/skills/frontend-design",
            },
            ("skills-sh/anthropics/skills/frontend-design", False),
            id="derive-trusted-source-from-identifier",
        ),
        pytest.param(
            {
                "source": "github",
                "identifier": "github/example/community-skill",
                "trust_level": "builtin",
                "scan_source": "official",
            },
            ("github/example/community-skill", False),
            id="reject-forged-builtin-trust",
        ),
        pytest.param(
            {
                "source": "official",
                "identifier": "github/example/community-skill",
                "trust_level": "builtin",
            },
            ("community", False),
            id="reject-forged-official-source",
        ),
        pytest.param(
            {"source": "official", "trust_level": "builtin"},
            ("community", False),
            id="reject-missing-official-identifier",
        ),
        pytest.param(
            {
                "source": "github",
                "identifier": "official",
                "trust_level": "community",
            },
            ("community", False),
            id="reject-bare-official-identifier",
        ),
        pytest.param(
            {
                "source": "agent-created",
                "identifier": "agent-created",
                "trust_level": "agent-created",
            },
            ("agent-created", True),
            id="preserve-internal-agent-created-origin",
        ),
    ],
)
def test_audit_scan_identity_rejects_unattested_lock_provenance(
    entry, expected, tmp_path
):
    from hermes_cli.skills_hub import _audit_scan_identity_for_lock_entry

    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    (skill_path / "SKILL.md").write_text("# Demo\n")

    assert _audit_scan_identity_for_lock_entry(entry, skill_path) == expected


@pytest.mark.parametrize("installed_kind", ["modified", "symlink"])
def test_audit_install_attestation_downgrades_changed_or_redirected_content(
    installed_kind, tmp_path
):
    import tools.skills_guard as guard
    from hermes_cli.skills_hub import _audit_scan_identity_for_lock_entry

    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    installed_file = skill_path / "SKILL.md"
    installed_file.write_text("# Official\n")
    identifier = "official/devops/demo"
    attested_hash = guard.full_content_hash(skill_path)
    if installed_kind == "modified":
        installed_file.write_text("# Modified\n")
    else:
        target = tmp_path / "target.md"
        target.write_text("# Official\n")
        installed_file.unlink()
        try:
            installed_file.symlink_to(target)
        except OSError:
            pytest.skip("symlinks unavailable")

    entry = {
        "source": "official",
        "identifier": identifier,
        "trust_level": "builtin",
        "install_attestation": {
            "version": 2,
            "hash_scheme": guard.TREE_HASH_SCHEME,
            "source": "official",
            "identifier": identifier,
            "trust_level": "builtin",
            "bundle_hash": attested_hash,
        },
    }

    assert _audit_scan_identity_for_lock_entry(entry, skill_path) == (
        "community",
        False,
    )


def test_audit_install_attestation_does_not_reject_executable_mode(tmp_path):
    import tools.skills_guard as guard
    from hermes_cli.skills_hub import _audit_scan_identity_for_lock_entry

    skill_path = tmp_path / "demo"
    skill_path.mkdir()
    script = skill_path / "run.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)
    (skill_path / "SKILL.md").write_text("# Demo\n")
    identifier = "official/devops/demo"
    entry = {
        "source": "official",
        "identifier": identifier,
        "trust_level": "builtin",
        "install_attestation": guard.build_install_attestation(
            skill_path,
            source="official",
            identifier=identifier,
            trust_level="builtin",
            origin_identity="git:NousResearch/hermes-agent@" + "c" * 40,
        ),
    }

    assert _audit_scan_identity_for_lock_entry(entry, skill_path) == (
        "official",
        True,
    )


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
        lambda skill_path, source="community", **_kwargs: guard.ScanResult(
            skill_name="pending", source=source, trust_level="community", verdict="safe",
        ),
    )
    monkeypatch.setattr(guard, "format_scan_report", lambda result: "scan ok")
    monkeypatch.setattr(guard, "should_allow_install", lambda result, force=False: (True, "ok"))
    return install_calls


@pytest.mark.parametrize(
    ("bundle_source", "bundle_identifier", "install_identifier", "expected_source"),
    [
        ("clawhub", "official", "clawhub/official", "official"),
        (
            "official",
            "official/devops/demo",
            "official/devops/demo",
            "community",
        ),
    ],
    ids=["external-reserved-marker", "unverified-optional-root"],
)
def test_do_install_denies_unverified_official_origin(
    monkeypatch,
    tmp_path,
    bundle_source,
    bundle_identifier,
    install_identifier,
    expected_source,
):
    import tools.skills_guard as guard
    import tools.skills_hub as hub

    bundle = hub.SkillBundle(
        name="demo",
        files={"SKILL.md": "# Demo\n"},
        source=bundle_source,
        identifier=bundle_identifier,
        trust_level="community",
        origin_verified=False,
    )

    class Source:
        def inspect(self, _identifier):
            return hub.SkillMeta(
                name="demo",
                description="demo",
                source=bundle_source,
                identifier=bundle_identifier,
                trust_level="community",
            )

        def fetch(self, _identifier):
            return bundle

    q_path = tmp_path / "skills" / ".hub" / "quarantine" / "demo"
    q_path.mkdir(parents=True)
    (q_path / "SKILL.md").write_text("# Demo\n")
    observed = {}

    def _scan_cached(path, source="community", **kwargs):
        observed["source"] = source
        observed["allow_origin_markers"] = kwargs.get("allow_origin_markers")
        result = guard.ScanResult(
            skill_name="demo",
            source=source,
            trust_level="community",
            verdict="safe",
        )
        provenance = {
            "fresh": True,
            "scanner_version": "test",
            "bundle_hash": guard.full_content_hash(path),
            "rules": [],
            "source_url": "",
            "scanned_at": "now",
        }
        return result, provenance

    monkeypatch.setattr(hub, "SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(hub, "ensure_hub_dirs", lambda: None)
    monkeypatch.setattr(hub, "create_source_router", lambda _auth: [Source()])
    monkeypatch.setattr(hub, "quarantine_bundle", lambda _bundle: q_path)
    monkeypatch.setattr(
        hub,
        "HubLockFile",
        lambda: type("Lock", (), {"get_installed": lambda self, _name: None})(),
    )
    monkeypatch.setattr(
        hub,
        "install_from_quarantine",
        lambda *_args, **_kwargs: tmp_path / "skills" / "demo",
    )
    monkeypatch.setattr(guard, "scan_skill_cached", _scan_cached)
    monkeypatch.setattr(guard, "format_scan_report", lambda _result: "scan")
    monkeypatch.setattr(guard, "should_allow_install", lambda *_args, **_kwargs: (True, "ok"))

    do_install(
        install_identifier,
        skip_confirm=True,
        console=Console(file=StringIO(), force_terminal=False),
    )

    assert observed == {
        "source": expected_source,
        "allow_origin_markers": False,
    }






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
