"""Official skill installs must not die on a user-created category symlink.

Desktop Install cannot pass ``--category``. It derives ``research`` from
``official/research/searxng-search``. When ``skills/research`` is a symlink,
the path check used to refuse the install. The symlink guard stays — uninstall
must not ``rmtree`` through it — but an auto-derived category falls back to a
real directory under ``skills/``.
"""

from contextlib import contextmanager
from io import StringIO

import pytest
from rich.console import Console

import hermes_cli.skills_hub as cli_hub
import tools.skills_hub as hub
import tools.skills_hub_install as hub_install
from tools.skills_guard import ScanResult


def _bundle():
    return type("Bundle", (), {
        "name": "searxng-search",
        "files": {"SKILL.md": "---\nname: searxng-search\n---\n"},
        "source": "official",
        "identifier": "official/research/searxng-search",
        "trust_level": "builtin",
        "metadata": {},
    })()


@contextmanager
def _session():
    yield


def _symlink_layout(tmp_path):
    skills = tmp_path / "skills"
    skills.mkdir()
    external = tmp_path / "external-research"
    external.mkdir()
    (skills / "research").symlink_to(external, target_is_directory=True)
    return skills, external


@pytest.mark.require_symlinks
def test_path_refusal_names_the_symlink(tmp_path, monkeypatch):
    skills, external = _symlink_layout(tmp_path)
    monkeypatch.setattr(hub, "SKILLS_DIR", skills)
    with pytest.raises(ValueError, match=r"research is a symlink to") as exc:
        hub_install._resolve_lock_install_path("research/searxng-search", "searxng-search")
    message = str(exc.value)
    assert message.startswith("Unsafe install path: research/searxng-search")
    assert "external-research" in message or "symlink to" in message
    assert not (external / "searxng-search").exists()


@pytest.mark.require_symlinks
def test_explicit_category_is_not_rewritten(tmp_path, monkeypatch):
    skills, _external = _symlink_layout(tmp_path)
    monkeypatch.setattr(hub, "SKILLS_DIR", skills)
    category, reason = hub_install.avoid_symlinked_category("research", explicit=True)
    assert category == "research"
    assert reason is None


@pytest.mark.require_symlinks
def test_auto_category_symlink_installs_flat(monkeypatch, tmp_path):
    skills, external = _symlink_layout(tmp_path)
    seen = {}

    def _install(q, name, category, bundle, result):
        seen["category"] = category
        seen["name"] = name
        dest = skills / name
        dest.mkdir()
        return dest

    monkeypatch.setattr(hub, "SKILLS_DIR", skills)
    monkeypatch.setattr(hub, "ensure_hub_dirs", lambda: None)
    monkeypatch.setattr(hub, "skills_hub_http_session", _session)
    monkeypatch.setattr(hub, "HubLockFile", lambda: type("Lock", (), {
        "get_installed": lambda self, name: None,
    })())
    monkeypatch.setattr(cli_hub, "_sources", list)
    monkeypatch.setattr(cli_hub, "_pinned_sources", lambda *a, **k: [])
    monkeypatch.setattr(
        cli_hub, "_resolve_source_meta_and_bundle",
        lambda identifier, sources: (None, _bundle(), None),
    )
    monkeypatch.setattr(hub_install, "quarantine_bundle", lambda bundle: tmp_path / "q")
    monkeypatch.setattr(cli_hub, "_scan_quarantined", lambda *a, **k: ScanResult(
        skill_name="searxng-search", source="official", trust_level="builtin", verdict="safe",
    ))
    monkeypatch.setattr(cli_hub, "_print_tier1_advisory", lambda *a, **k: None)
    monkeypatch.setattr(cli_hub, "_announce_blueprint", lambda *a, **k: None)
    monkeypatch.setattr(cli_hub, "_finish_change", lambda *a, **k: None)
    monkeypatch.setattr(hub_install, "install_from_quarantine", _install)

    buf = StringIO()
    bundle, outcome, installed = cli_hub._install_skill(
        "official/research/searxng-search", "", False,
        Console(file=buf, force_terminal=False, highlight=False), True,
        False, "", None,
    )
    text = buf.getvalue()
    assert installed is True
    assert outcome == "success"
    assert bundle.name == "searxng-search"
    assert seen == {"category": "", "name": "searxng-search"}
    assert "research is a symlink" in text
    assert "Installing at 'searxng-search'" in text
    assert not (external / "searxng-search").exists()
