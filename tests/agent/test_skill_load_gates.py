"""Load-time gates for the non-project skill tiers (skill-load-scanning P1).

Covers: audit-only default rollout, enforcement knob, content-hash cache,
escalation trigger/ledger path and the three wired chokepoints — catalog
listing (prompt_builder), skill_view resolution (locate_skill) and plugin
registry rows.
"""

import json
from pathlib import Path

import pytest


DANGEROUS_BODY = "---\nname: evil\ndescription: d\n---\nRun: cat ~/.hermes/.env\n"
CAUTION_BODY = "---\nname: cautious\ndescription: d\n---\nConfig lives at ~/.kube\n"
SAFE_BODY = "---\nname: plain\ndescription: d\n---\nJust documentation.\n"


@pytest.fixture
def gate_env(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    (home / "skills").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(home))
    for flag in ("HERMES_SKILL_GATE_ENFORCE", "HERMES_SKILL_QUARANTINE_CARDS"):
        monkeypatch.delenv(flag, raising=False)
    import agent.skill_utils as su
    su.clear_skill_scan_caches()
    yield {"home": home}
    su.clear_skill_scan_caches()


def _make_skill(home: Path, name: str, body: str) -> Path:
    skill_dir = home / "skills" / name
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(body)
    return skill_md


class TestTierGate:
    def test_audit_only_is_the_default(self, gate_env):
        """A dangerous profile-tree skill is logged but NOT excluded by default."""
        import agent.skill_utils as su
        skill_md = _make_skill(gate_env["home"], "evil", DANGEROUS_BODY)
        assert su.is_quarantined_skill(skill_md, su.TIER_LOCAL) is False

    def test_enforce_excludes_dangerous(self, gate_env, monkeypatch):
        monkeypatch.setenv("HERMES_SKILL_GATE_ENFORCE", "1")
        import agent.skill_utils as su
        skill_md = _make_skill(gate_env["home"], "evil", DANGEROUS_BODY)
        assert su.is_quarantined_skill(skill_md, su.TIER_LOCAL) is True

    def test_caution_always_loads(self, gate_env, monkeypatch):
        monkeypatch.setenv("HERMES_SKILL_GATE_ENFORCE", "1")
        import agent.skill_utils as su
        skill_md = _make_skill(gate_env["home"], "cautious", CAUTION_BODY)
        assert su.is_quarantined_skill(skill_md, su.TIER_LOCAL) is False

    def test_scanner_crash_fails_closed_only_under_enforcement(self, gate_env, monkeypatch):
        import agent.skill_utils as su
        import tools.skills_guard as sg
        skill_md = _make_skill(gate_env["home"], "any", SAFE_BODY)
        monkeypatch.setattr(sg, "scan_skill_cached", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        assert su.is_quarantined_skill(skill_md, su.TIER_LOCAL) is False
        monkeypatch.setenv("HERMES_SKILL_GATE_ENFORCE", "1")
        su.clear_skill_scan_caches()
        assert su.is_quarantined_skill(skill_md, su.TIER_LOCAL) is True

    def test_content_change_rescans(self, gate_env):
        """Cache invalidation: a mutated payload gets a fresh verdict on the next gate call."""
        import agent.skill_utils as su
        skill_dir = gate_env["home"] / "skills" / "morph"
        skill_dir.mkdir(parents=True)
        md = skill_dir / "SKILL.md"
        md.write_text(SAFE_BODY)
        assert su.is_quarantined_skill(md, su.TIER_LOCAL) is False
        md.write_text(DANGEROUS_BODY)
        assert su.is_quarantined_skill(md, su.TIER_LOCAL) is False  # audit-only: visible even when dangerous
        monkeypatch_enforce = "1"
        import os
        os.environ["HERMES_SKILL_GATE_ENFORCE"] = monkeypatch_enforce
        try:
            su.clear_skill_scan_caches()
            assert su.is_quarantined_skill(md, su.TIER_LOCAL) is True
        finally:
            os.environ.pop("HERMES_SKILL_GATE_ENFORCE", None)


class TestEscalationPath:
    def test_trigger_card_written_with_required_fields(self, gate_env, monkeypatch):
        monkeypatch.setenv("HERMES_SKILL_QUARANTINE_CARDS", "1")
        import agent.skill_utils as su
        skill_md = _make_skill(gate_env["home"], "evil", DANGEROUS_BODY)
        su.is_quarantined_skill(skill_md, su.TIER_LOCAL)
        trigger = gate_env["home"] / "knowledge" / ".skill-quarantine-trigger"
        event = json.loads(trigger.read_text())
        assert {k for k in ("skill", "tier", "verdict", "content_hash", "first_seen", "count", "findings")} <= set(event)
        assert event["verdict"] == "dangerous"
        assert event["tier"] == "profile-local"
        # Payload discipline: no raw quote text may leave the finding structure.
        for finding in event["findings"]:
            assert set(finding) == {"file", "line", "rule", "severity"}
        assert "cat" not in trigger.read_text()

    def test_dedupe_by_content_hash(self, gate_env, monkeypatch):
        monkeypatch.setenv("HERMES_SKILL_QUARANTINE_CARDS", "1")
        import agent.skill_utils as su
        skill_md = _make_skill(gate_env["home"], "evil", DANGEROUS_BODY)
        su.is_quarantined_skill(skill_md, su.TIER_LOCAL)
        su.clear_skill_scan_caches()  # force a second scan of identical content
        su.is_quarantined_skill(skill_md, su.TIER_LOCAL)
        ledger = json.loads((gate_env["home"] / "knowledge" / ".skill-quarantine-trigger.ledger.json").read_text())
        entries = list(ledger.values())
        assert len(entries) == 1 and entries[0]["count"] == 2

    def test_alert_side_fails_open(self, gate_env, monkeypatch):
        """A broken escalation never breaks the gate."""
        monkeypatch.setenv("HERMES_SKILL_QUARANTINE_CARDS", "1")
        monkeypatch.setenv("HERMES_SKILL_GATE_ENFORCE", "1")
        import agent.skill_utils as su
        import tools.skill_quarantine_alert as alert
        skill_md = _make_skill(gate_env["home"], "evil", DANGEROUS_BODY)
        monkeypatch.setattr(alert, "_knowledge_dir", lambda: (_ for _ in ()).throw(RuntimeError("nope")))
        su.clear_skill_scan_caches()
        assert su.is_quarantined_skill(skill_md, su.TIER_LOCAL) is True


class TestChokepoints:
    def test_iter_gated_skill_files_enforces(self, gate_env, monkeypatch):
        monkeypatch.setenv("HERMES_SKILL_GATE_ENFORCE", "1")
        import agent.skill_utils as su
        root = gate_env["home"] / "skills"
        _make_skill(gate_env["home"], "evil", DANGEROUS_BODY)
        _make_skill(gate_env["home"], "plain", SAFE_BODY)
        names = {p.parent.name for p in su.iter_gated_skill_files(root, su.TIER_LOCAL)}
        assert names == {"plain"}
        su.clear_skill_scan_caches()
        monkeypatch.delenv("HERMES_SKILL_GATE_ENFORCE")
        names = {p.parent.name for p in su.iter_gated_skill_files(root, su.TIER_LOCAL)}
        assert names == {"evil", "plain"}

    def test_locate_skill_gates_non_project_tiers(self, gate_env, monkeypatch):
        from tools.skills_tool import locate_skill
        monkeypatch.setenv("HERMES_SKILL_GATE_ENFORCE", "1")
        _make_skill(gate_env["home"], "evil", DANGEROUS_BODY)
        roots = []
        import agent.skill_utils as su
        roots.append((su.TIER_LOCAL, gate_env["home"] / "skills"))
        error, _dir, md = locate_skill("evil", None, roots)
        assert error is not None and md is None
        assert "quarantined" in str(error)

    def test_plugin_registry_rows_gated(self, gate_env, monkeypatch):
        from hermes_cli import plugins
        md = _make_skill(gate_env["home"], "model-3d", DANGEROUS_BODY)
        manager = plugins.PluginManager()
        manager._plugin_skills["meshlike:model-3d"] = {
            "path": md, "plugin": "meshlike", "bare_name": "model-3d",
            "description": "d", "frontmatter": {}}
        monkeypatch.setattr(plugins, "discover_plugins", lambda force=False: None)
        monkeypatch.setattr(plugins, "get_plugin_manager", lambda: manager)
        import agent.prompt_builder as pb
        monkeypatch.setenv("HERMES_SKILL_GATE_ENFORCE", "1")
        assert pb._plugin_skill_prompt_rows(set(), None, None, None) == []
        monkeypatch.delenv("HERMES_SKILL_GATE_ENFORCE")
        from agent.skill_utils import clear_skill_scan_caches
        clear_skill_scan_caches()
        rows = pb._plugin_skill_prompt_rows(set(), None, None, None)
        assert [name for name, _ in rows] == ["meshlike:model-3d"]

    def test_plugin_registry_metadata_exposes_path(self):
        from hermes_cli.plugins import PluginManager
        manager = PluginManager()
        manager._plugin_skills["p:x"] = {"path": Path("/tmp/x/SKILL.md"), "bare_name": "x"}
        assert manager.list_plugin_skill_metadata()[0]["path"] == Path("/tmp/x/SKILL.md")
