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
    yield {"home": home}


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
        assert su.is_quarantined_skill(skill_md, su.TIER_LOCAL) is True

    def test_content_change_rescans(self, gate_env, monkeypatch):
        """Cache invalidation: a mutated payload gets a fresh verdict on the next gate call."""
        import agent.skill_utils as su
        skill_dir = gate_env["home"] / "skills" / "morph"
        skill_dir.mkdir(parents=True)
        md = skill_dir / "SKILL.md"
        md.write_text(SAFE_BODY)
        assert su.is_quarantined_skill(md, su.TIER_LOCAL) is False
        md.write_text(DANGEROUS_BODY)
        assert su.is_quarantined_skill(md, su.TIER_LOCAL) is False  # audit-only: visible even when dangerous
        monkeypatch.setenv("HERMES_SKILL_GATE_ENFORCE", "1")
        assert su.is_quarantined_skill(md, su.TIER_LOCAL) is True


class TestEscalationPath:
    def test_trigger_card_written_with_required_fields(self, gate_env, monkeypatch):
        monkeypatch.setenv("HERMES_SKILL_QUARANTINE_CARDS", "1")
        import agent.skill_utils as su
        skill_md = _make_skill(gate_env["home"], "evil", DANGEROUS_BODY)
        su.is_quarantined_skill(skill_md, su.TIER_LOCAL)
        trigger = gate_env["home"] / "knowledge" / ".skill-quarantine-trigger"
        events = json.loads(trigger.read_text())
        assert isinstance(events, list) and len(events) == 1
        event = events[0]
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
        su.is_quarantined_skill(skill_md, su.TIER_LOCAL)  # second scan of identical content
        trigger = gate_env["home"] / "knowledge" / ".skill-quarantine-trigger"
        assert len(json.loads(trigger.read_text())) == 1  # no duplicate pending event
        ledger = json.loads((gate_env["home"] / "knowledge" / ".skill-quarantine-trigger.ledger.json").read_text())
        entries = list(ledger.values())
        assert len(entries) == 1 and entries[0]["count"] == 2

    def test_trigger_file_accumulates_unconsumed_events(self, gate_env, monkeypatch):
        """A second, different content hash must not clobber an event the kernel
        has not consumed yet (no last-write-wins loss during batch scans)."""
        monkeypatch.setenv("HERMES_SKILL_QUARANTINE_CARDS", "1")
        import agent.skill_utils as su
        su.is_quarantined_skill(_make_skill(gate_env["home"], "evil", DANGEROUS_BODY), su.TIER_LOCAL)
        su.is_quarantined_skill(_make_skill(gate_env["home"], "cautious", CAUTION_BODY), su.TIER_LOCAL)
        trigger = gate_env["home"] / "knowledge" / ".skill-quarantine-trigger"
        events = json.loads(trigger.read_text())
        assert len(events) == 2
        assert len({e["content_hash"] for e in events}) == 2
        assert {e["skill"] for e in events} == {"evil", "cautious"}

    def test_writer_merges_with_kernel_consumption_leftovers(self, gate_env, monkeypatch):
        """The kernel consumes by rewriting the file to its unconsumed events; the
        next writer call merges into those leftovers instead of overwriting."""
        monkeypatch.setenv("HERMES_SKILL_QUARANTINE_CARDS", "1")
        import tools.skill_quarantine_alert as alert
        trigger = gate_env["home"] / "knowledge" / ".skill-quarantine-trigger"
        (gate_env["home"] / "knowledge").mkdir(parents=True, exist_ok=True)
        trigger.write_text(json.dumps([{"skill": "keepme", "content_hash": "bb"}]))  # kernel left one pending
        alert.notify_skill_finding("profile-local", gate_env["home"] / "skills" / "new" / "SKILL.md",
                                   {"verdict": "dangerous", "bundle_hash": "cc", "findings": []}, False)
        events = json.loads(trigger.read_text())
        assert [e["skill"] for e in events] == ["keepme", "new"]

    def test_alert_side_fails_open(self, gate_env, monkeypatch):
        """A broken escalation never breaks the gate."""
        monkeypatch.setenv("HERMES_SKILL_QUARANTINE_CARDS", "1")
        monkeypatch.setenv("HERMES_SKILL_GATE_ENFORCE", "1")
        import agent.skill_utils as su
        import tools.skill_quarantine_alert as alert
        skill_md = _make_skill(gate_env["home"], "evil", DANGEROUS_BODY)
        monkeypatch.setattr(alert, "_knowledge_dir", lambda: (_ for _ in ()).throw(RuntimeError("nope")))
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
        monkeypatch.delenv("HERMES_SKILL_GATE_ENFORCE")
        names = {p.parent.name for p in su.iter_gated_skill_files(root, su.TIER_LOCAL)}
        assert names == {"evil", "plain"}

    def test_catalog_excludes_dangerous_non_project_tier(self, gate_env, monkeypatch):
        """§6.1 reuse semantics: skills_list / not-found fallback must not surface a
        dangerous non-project skill (incl. its frontmatter description) under
        enforcement — the catalog iterates non-project tiers through the gate."""
        from tools.skills_tool import _skill_catalog
        _make_skill(gate_env["home"], "evil", DANGEROUS_BODY)
        _make_skill(gate_env["home"], "plain", SAFE_BODY)
        rows = {s["name"] for s in _skill_catalog()}
        assert "evil" in rows and "plain" in rows  # audit-only: listed, nothing excluded
        monkeypatch.setenv("HERMES_SKILL_GATE_ENFORCE", "1")
        rows = {s["name"] for s in _skill_catalog()}
        assert "evil" not in rows and "plain" in rows  # enforcement: dangerous excluded

    def test_catalog_cache_survives_gate_scan_cache_writes(self, gate_env, monkeypatch):
        """The gate's scan cache (<home>/cache/skill_scans) must not invalidate the
        catalog dir signature — otherwise every scan thrashes through a full re-parse."""
        import tools.skills_tool as st
        from tools.skills_tool import _skill_catalog
        _make_skill(gate_env["home"], "plain", SAFE_BODY)
        calls = []
        real_parse = st._parse_frontmatter

        def counting_parse(content):
            calls.append(1)
            return real_parse(content)

        monkeypatch.setattr(st, "_parse_frontmatter", counting_parse)
        _skill_catalog()
        first_scan_calls = len(calls)
        assert first_scan_calls >= 1
        _skill_catalog()
        assert len(calls) == first_scan_calls  # second call served from cache

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
        rows = pb._plugin_skill_prompt_rows(set(), None, None, None)
        assert [name for name, _ in rows] == ["meshlike:model-3d"]

    def test_plugin_registry_metadata_exposes_path(self):
        from hermes_cli.plugins import PluginManager
        manager = PluginManager()
        manager._plugin_skills["p:x"] = {"path": Path("/tmp/x/SKILL.md"), "bare_name": "x"}
        assert manager.list_plugin_skill_metadata()[0]["path"] == Path("/tmp/x/SKILL.md")
