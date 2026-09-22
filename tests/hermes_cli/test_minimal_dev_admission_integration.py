from io import StringIO
import sys
from types import SimpleNamespace

import pytest
from rich.console import Console


def _install_env(monkeypatch, tmp_path, events):
    import hermes_cli.skills_hub as cli_hub
    import tools.skills_hub as hub
    import tools.skills_hub_install as hub_install
    import tools.skills_guard as skills_guard

    q_path = tmp_path / "quarantine" / "demo"
    q_path.mkdir(parents=True)
    (q_path / "SKILL.md").write_text(
        "---\nname: demo\n---\nbody\n", encoding="utf-8"
    )

    bundle = SimpleNamespace(
        name="demo",
        source="github",
        trust_level="community",
        identifier="owner/repo/demo",
        metadata={},
        files={"SKILL.md": "body"},
    )
    monkeypatch.setattr(hub, "ensure_hub_dirs", lambda: None)
    monkeypatch.setattr(hub, "SKILLS_DIR", tmp_path / "skills")
    monkeypatch.setattr(
        hub,
        "HubLockFile",
        lambda: SimpleNamespace(get_installed=lambda _name: None),
    )
    monkeypatch.setattr(
        cli_hub,
        "_sources",
        lambda: [SimpleNamespace(source_id=lambda: "github")],
    )
    monkeypatch.setattr(
        cli_hub,
        "_pinned_sources",
        lambda c, s, source_id, identifier: s,
    )
    monkeypatch.setattr(
        cli_hub,
        "_full_identifier",
        lambda identifier, sources, c: identifier,
    )
    monkeypatch.setattr(
        cli_hub,
        "_resolve_source_meta_and_bundle",
        lambda identifier, sources: (None, bundle, sources[0]),
    )
    monkeypatch.setattr(
        cli_hub,
        "_resolve_url_bundle_name",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(hub_install, "quarantine_bundle", lambda _bundle: q_path)
    monkeypatch.setattr(
        cli_hub,
        "_scan_quarantined",
        lambda *args: SimpleNamespace(verdict="safe", findings=[]),
    )
    monkeypatch.setattr(
        skills_guard,
        "should_allow_install",
        lambda result, force=False: (True, ""),
    )
    monkeypatch.setattr(
        cli_hub, "_print_tier1_advisory", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        cli_hub, "_format_extra_metadata_lines", lambda metadata: []
    )
    monkeypatch.setattr(
        cli_hub, "_announce_blueprint", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        cli_hub, "_finish_change", lambda *args, **kwargs: None
    )

    def _install(*args, **kwargs):
        events.append("install")
        dest = tmp_path / "skills" / "demo"
        dest.mkdir(parents=True, exist_ok=True)
        return dest

    monkeypatch.setattr(hub_install, "install_from_quarantine", _install)
    sink = StringIO()
    console = Console(file=sink, force_terminal=False, color_system=None)
    return cli_hub, bundle, q_path, console, sink


def test_default_off_preserves_active_main_install_path(monkeypatch, tmp_path):
    events = []
    cli_hub, bundle, _q_path, console, _sink = _install_env(
        monkeypatch, tmp_path, events
    )

    def _external(candidate, source_id="", source_version=None):
        events.append("external_off")
        return SimpleNamespace(
            mode="off",
            executed=False,
            decision="OFF",
            allow_continue=True,
            candidate_sha256="",
            evidence_dir="",
            approval_binding=None,
            error="",
        )

    monkeypatch.setitem(
        sys.modules,
        "tools.external_skill_admission",
        SimpleNamespace(run_external_admission=_external),
    )

    cli_hub.do_install(
        bundle.identifier, console=console, force=True, skip_confirm=True
    )

    assert events == ["external_off", "install"]


@pytest.mark.parametrize("decision", ["BLOCK", "ERROR", "ALLOW_TO_LAB"])
@pytest.mark.parametrize("force,skip_confirm", [(False, False), (True, False), (False, True), (True, True)])
def test_configured_non_quarantine_never_installs_or_prompts(
    monkeypatch, tmp_path, decision, force, skip_confirm
):
    events = []
    cli_hub, bundle, q_path, console, sink = _install_env(
        monkeypatch, tmp_path, events
    )

    def _external(candidate, source_id="", source_version=None):
        events.append("external")
        return SimpleNamespace(
            mode="enforce",
            executed=True,
            decision=decision,
            allow_continue=(decision == "ALLOW_TO_LAB"),
            candidate_sha256="a" * 64,
            evidence_dir="C:/local-fixture",
            approval_binding=None,
            error="boom" if decision == "ERROR" else "",
        )

    monkeypatch.setitem(
        sys.modules,
        "tools.external_skill_admission",
        SimpleNamespace(run_external_admission=_external),
    )

    def _must_not_confirm(*args, **kwargs):
        raise AssertionError("configured non-off gate must not reach install confirmation")

    monkeypatch.setattr(cli_hub, "_confirm_install", _must_not_confirm)

    cli_hub.do_install(
        bundle.identifier,
        console=console,
        force=force,
        skip_confirm=skip_confirm,
    )

    assert events == ["external"]
    assert q_path.exists()
    assert "not installed" in sink.getvalue().lower()


def test_quarantine_without_dev_dependency_fails_closed(monkeypatch, tmp_path):
    events = []
    cli_hub, bundle, q_path, console, sink = _install_env(
        monkeypatch, tmp_path, events
    )

    def _external(candidate, source_id="", source_version=None):
        events.append("external")
        return SimpleNamespace(
            mode="enforce",
            executed=True,
            decision="QUARANTINE",
            allow_continue=False,
            candidate_sha256="b" * 64,
            evidence_dir="C:/local-fixture",
            approval_binding=object(),
            error="",
        )

    monkeypatch.setitem(
        sys.modules,
        "tools.external_skill_admission",
        SimpleNamespace(run_external_admission=_external),
    )

    cli_hub.do_install(
        bundle.identifier,
        console=console,
        force=True,
        skip_confirm=True,
    )

    assert events == ["external"]
    assert q_path.exists()
    assert "foreground_review_required" in sink.getvalue().lower()


@pytest.mark.parametrize("allowed", [True, False])
def test_quarantine_injected_dependency_is_terminal(
    monkeypatch, tmp_path, allowed
):
    events = []
    cli_hub, bundle, q_path, console, sink = _install_env(
        monkeypatch, tmp_path, events
    )
    bundle.metadata["version"] = "v1"

    external = SimpleNamespace(
        mode="enforce",
        executed=True,
        decision="QUARANTINE",
        allow_continue=False,
        candidate_sha256="c" * 64,
        evidence_dir="C:/local-fixture",
        approval_binding=object(),
        error="",
    )

    def _external(candidate, source_id="", source_version=None):
        events.append("external")
        assert candidate == q_path
        assert source_id == bundle.identifier
        assert source_version == "v1"
        return external

    def _simulate(result, candidate, *, source_id="", source_version=""):
        events.append("lab")
        assert result is external
        assert candidate == q_path
        assert source_id == bundle.identifier
        assert source_version == "v1"
        return SimpleNamespace(
            handled=True,
            allowed=allowed,
            outcome=(
                "ALLOW_TO_LAB_SIMULATION"
                if allowed
                else "LAB_SIMULATION_REJECTED"
            ),
            reason=(
                "ALLOW_TO_LAB_SIMULATION"
                if allowed
                else "DEV_APPROVAL_REJECTED"
            ),
        )

    monkeypatch.setitem(
        sys.modules,
        "tools.external_skill_admission",
        SimpleNamespace(run_external_admission=_external),
    )

    cli_hub.do_install(
        bundle.identifier,
        console=console,
        force=True,
        skip_confirm=True,
        dev_lab_integration=SimpleNamespace(simulate=_simulate),
    )

    assert events == ["external", "lab"]
    assert q_path.exists()
    text = sink.getvalue().lower()
    assert "lab simulation allowed" in text if allowed else "lab simulation rejected" in text
    assert "not installed" in text if allowed else "quarantine preserved" in text


@pytest.mark.parametrize("decision", ["BLOCK", "ERROR", "ALLOW_TO_LAB"])
def test_non_quarantine_never_calls_injected_dependency(
    monkeypatch, tmp_path, decision
):
    events = []
    cli_hub, bundle, q_path, console, _sink = _install_env(
        monkeypatch, tmp_path, events
    )

    def _external(candidate, source_id="", source_version=None):
        events.append("external")
        return SimpleNamespace(
            mode="enforce",
            executed=True,
            decision=decision,
            allow_continue=False,
            candidate_sha256="d" * 64,
            evidence_dir="C:/local-fixture",
            approval_binding=None,
            error="",
        )

    def _must_not_simulate(*args, **kwargs):
        raise AssertionError("non-QUARANTINE must never reach dev approval")

    monkeypatch.setitem(
        sys.modules,
        "tools.external_skill_admission",
        SimpleNamespace(run_external_admission=_external),
    )

    cli_hub.do_install(
        bundle.identifier,
        console=console,
        force=True,
        skip_confirm=True,
        dev_lab_integration=SimpleNamespace(simulate=_must_not_simulate),
    )

    assert events == ["external"]
    assert q_path.exists()


def test_real_dev_lab_adapter_inside_current_main_do_install(monkeypatch, tmp_path):
    from tools.external_skill_admission import _candidate_hash
    from tools.quarantine_approval_models import (
        APPROVAL_BINDING_SCHEMA,
        ApprovalBinding,
        evidence_bundle_digest,
    )
    from tools.quarantine_dev_context import DevHumanApprovalContext
    from tools.quarantine_dev_human_approval import derive_confirmation_code
    from tools.quarantine_lab_integration import DevLabQuarantineIntegration

    events = []
    cli_hub, bundle, q_path, console, sink = _install_env(
        monkeypatch, tmp_path, events
    )
    bundle.metadata["version"] = "v1"

    sentinel = tmp_path / "EXECUTED.txt"
    (q_path / "payload.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(sentinel)!r}).write_text('executed', encoding='utf-8')\n",
        encoding="utf-8",
    )

    candidate_sha = _candidate_hash(q_path)
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "decision.json").write_text(
        '{"decision":"QUARANTINE"}\n', encoding="utf-8"
    )
    digest = evidence_bundle_digest(evidence)

    binding = ApprovalBinding(
        schema_version=APPROVAL_BINDING_SCHEMA,
        candidate_sha256=candidate_sha,
        source_id=bundle.identifier,
        source_version="v1",
        gate_release_sha256="e" * 64,
        policy_version="minimal-merge-candidate-v1",
        hermes_guard_version="skills-guard-v5",
        cisco_version="local-fixture-v1",
        nvidia_version="local-fixture-v1",
        scanner_completeness=(
            ("hermes", True),
            ("cisco", True),
            ("nvidia", True),
        ),
        scanner_reason_codes=(
            ("hermes", ()),
            ("cisco", ("local_fixture_review",)),
            ("nvidia", ()),
        ),
        evidence_digest=digest,
    )
    external = SimpleNamespace(
        mode="enforce",
        executed=True,
        decision="QUARANTINE",
        allow_continue=False,
        candidate_sha256=candidate_sha,
        evidence_dir=str(evidence),
        approval_binding=binding,
        error="",
    )

    def _external(candidate, source_id="", source_version=None):
        events.append("external")
        assert candidate == q_path
        assert source_id == bundle.identifier
        assert source_version == "v1"
        return external

    attempt = "5" * 32
    code = derive_confirmation_code(binding, attempt)

    class ReviewConsole:
        def is_interactive(self):
            return True

        def show_review(self, text):
            assert candidate_sha in text
            assert digest in text

        def read_confirmation(self, prompt):
            events.append("hash_confirm")
            return f"APPROVE LAB {code}"

    context = DevHumanApprovalContext(
        environment="LAB",
        simulation_only=True,
        background=False,
        production_enforce=False,
        provider_execution_enabled=False,
        browser_execution_enabled=False,
        financial_execution_enabled=False,
        destructive_host_actions_enabled=False,
        active_production_config=False,
        active_skill_store_target=False,
    )
    integration = DevLabQuarantineIntegration(
        context=context,
        review_console=ReviewConsole(),
        final_confirm=lambda _summary: events.append("final_confirm") or True,
        install_attempt_id=attempt,
    )

    before = {
        p.relative_to(q_path).as_posix(): p.read_bytes()
        for p in q_path.rglob("*")
        if p.is_file()
    }

    monkeypatch.setitem(
        sys.modules,
        "tools.external_skill_admission",
        SimpleNamespace(run_external_admission=_external),
    )

    cli_hub.do_install(
        bundle.identifier,
        console=console,
        force=True,
        skip_confirm=True,
        dev_lab_integration=integration,
    )

    after = {
        p.relative_to(q_path).as_posix(): p.read_bytes()
        for p in q_path.rglob("*")
        if p.is_file()
    }

    assert events == ["external", "hash_confirm", "final_confirm"]
    assert before == after
    assert sentinel.exists() is False
    assert q_path.exists()
    assert "lab simulation allowed" in sink.getvalue().lower()
    assert "not installed" in sink.getvalue().lower()
