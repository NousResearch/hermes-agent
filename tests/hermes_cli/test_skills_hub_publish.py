"""Local skill publication must bind the scan to independently owned bytes."""

from __future__ import annotations

import copy
import os
from io import StringIO

import pytest
from rich.console import Console


def _write_skill(root, *, support: str = "reviewed support bytes\n"):
    skill = root / "demo-skill"
    (skill / "references").mkdir(parents=True)
    (skill / "SKILL.md").write_text(
        "---\nname: demo-skill\ndescription: Demonstrate publication binding.\n---\n# Demo\n",
        encoding="utf-8",
    )
    (skill / "references" / "notes.txt").write_text(support, encoding="utf-8")
    return skill


class _Authenticated:
    def is_authenticated(self):
        return True

    def get_headers(self):
        return {}


def _console():
    sink = StringIO()
    return Console(file=sink, force_terminal=False, color_system=None), sink


def test_publish_uses_owned_reviewed_bytes_and_refuses_external_hardlinks(tmp_path, monkeypatch):
    """Publication owns the reviewed bytes and rejects known outside inode aliases."""
    import hermes_cli.skills_hub as cli_hub
    import tools.skills_guard as guard
    import tools.skills_hub_github as github

    skill = _write_skill(tmp_path)
    linked = skill / "references" / "notes.txt"
    linked.unlink()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside-controlled bytes\n", encoding="utf-8")
    try:
        os.link(outside, linked)
    except OSError as exc:
        pytest.skip(f"filesystem does not support hard links: {exc}")

    monkeypatch.setattr(github, "GitHubAuth", _Authenticated)
    monkeypatch.setattr(
        cli_hub,
        "_github_publish",
        lambda *_args, **_kwargs: pytest.fail("hard-linked source reached GitHub publication"),
    )
    console, sink = _console()

    cli_hub.do_publish(str(skill), target="github", repo="owner/repo", console=console)

    output = sink.getvalue().lower()
    assert "hard link" in output
    assert "references/notes.txt" in output
    assert outside.read_text(encoding="utf-8") == "outside-controlled bytes\n"

    # Some filesystems do not report a useful link count. Unknown is not evidence
    # of a hard link; the independently owned publication copy remains the guard.
    skill = _write_skill(tmp_path / "unknown-link-metadata")
    source_file = skill / "references" / "notes.txt"
    real_scan = guard.scan_skill
    observed = {}
    monkeypatch.setattr(cli_hub, "_publication_link_count", lambda _st: None, raising=False)

    def scan_owned_copy(snapshot, source="community"):
        snapshot_file = snapshot / "references" / "notes.txt"
        assert snapshot.resolve() != skill.resolve()
        assert snapshot_file.read_text(encoding="utf-8") == "reviewed support bytes\n"
        source_identity = (source_file.stat().st_dev, source_file.stat().st_ino)
        snapshot_identity = (snapshot_file.stat().st_dev, snapshot_file.stat().st_ino)
        if all(source_identity) and all(snapshot_identity):
            assert snapshot_identity != source_identity
        result = real_scan(snapshot, source=source)
        source_file.write_text("changed after scan\n", encoding="utf-8")
        return result

    def publish_owned_copy(snapshot, skill_name, target_repo, auth):
        observed["path"] = snapshot
        observed["bytes"] = (snapshot / "references" / "notes.txt").read_text(encoding="utf-8")
        return True, "published owned copy"

    monkeypatch.setattr(guard, "scan_skill", scan_owned_copy)
    monkeypatch.setattr(github, "GitHubAuth", _Authenticated)
    monkeypatch.setattr(cli_hub, "_github_publish", publish_owned_copy)
    console, sink = _console()

    cli_hub.do_publish(str(skill), target="github", repo="owner/repo", console=console)

    assert observed["path"] != skill
    assert observed["bytes"] == "reviewed support bytes\n"
    assert "published owned copy" in sink.getvalue()


def test_install_and_uninstall_clear_stale_disabled_state(tmp_path, monkeypatch):
    """A reinstall repairs old residue; uninstall removes newly configured residue."""
    import hermes_cli.skills_config as skills_config
    import tools.skills_hub as hub
    from tools.skills_guard import ScanResult
    from tools.skills_hub_install import install_from_quarantine, uninstall_skill
    from tools.skills_hub_models import SkillBundle

    skills_root = tmp_path / "skills"
    hub_root = skills_root / ".hub"
    quarantine_root = hub_root / "quarantine"
    monkeypatch.setattr(hub, "SKILLS_DIR", skills_root)
    monkeypatch.setattr(hub, "HUB_DIR", hub_root)
    monkeypatch.setattr(hub, "LOCK_FILE", hub_root / "lock.json")
    monkeypatch.setattr(hub, "QUARANTINE_DIR", quarantine_root)
    monkeypatch.setattr(hub, "AUDIT_LOG", hub_root / "audit.log")

    config = {
        "skills": {
            "disabled": ["demo-skill", "keep-global"],
            "platform_disabled": {
                "telegram": ["demo-skill", "keep-telegram"],
                "discord": "demo-skill",
            },
        }
    }
    saves = []

    monkeypatch.setattr(skills_config, "load_config", lambda: copy.deepcopy(config))

    def save_config(updated):
        config.clear()
        config.update(copy.deepcopy(updated))
        saves.append(copy.deepcopy(updated))

    monkeypatch.setattr(skills_config, "save_config", save_config)
    monkeypatch.setattr("tools.skill_usage.record_installed", lambda _name: None)
    monkeypatch.setattr(
        "hermes_cli.observability.shared_metrics_disabled.record_skill_removed",
        lambda _name: None,
    )

    staged = quarantine_root / "demo-skill"
    staged.mkdir(parents=True)
    skill_md = "---\nname: demo-skill\ndescription: lifecycle test\n---\n# Demo\n"
    (staged / "SKILL.md").write_text(skill_md, encoding="utf-8")
    bundle = SkillBundle(
        name="demo-skill",
        files={"SKILL.md": skill_md},
        source="community",
        identifier="owner/repo/demo-skill",
        trust_level="community",
    )
    scan = ScanResult(
        skill_name="demo-skill",
        source="community",
        trust_level="community",
        verdict="safe",
    )

    install_from_quarantine(staged, bundle.name, "", bundle, scan)

    assert config["skills"]["disabled"] == ["keep-global"]
    assert config["skills"]["platform_disabled"] == {
        "telegram": ["keep-telegram"],
        "discord": [],
    }

    # A user can disable the installed skill again. Updating the existing lock entry
    # must preserve that intentional choice; only a fresh install repairs stale residue.
    config["skills"]["disabled"].append("demo-skill")
    config["skills"]["platform_disabled"]["telegram"].append("demo-skill")
    config["skills"]["platform_disabled"]["discord"] = ["demo-skill"]
    hub.HubLockFile().record_install(
        name=bundle.name,
        source=bundle.source,
        identifier=bundle.identifier,
        trust_level=bundle.trust_level,
        scan_verdict=scan.verdict,
        skill_hash="sha256:unchanged",
        install_path=bundle.name,
        files=["SKILL.md"],
    )
    assert "demo-skill" in config["skills"]["disabled"]
    assert "demo-skill" in config["skills"]["platform_disabled"]["telegram"]

    # Uninstall removes those entries so a future reinstall starts enabled everywhere.
    ok, _message = uninstall_skill("demo-skill")

    assert ok is True
    assert config["skills"]["disabled"] == ["keep-global"]
    assert config["skills"]["platform_disabled"] == {
        "telegram": ["keep-telegram"],
        "discord": [],
    }
    assert len(saves) == 2
