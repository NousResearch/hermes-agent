"""Why installs and updates fail: failure_class on hermes.extension.install.count and
hermes.update.run, registry on hub skill installs (closed sets, never error text)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hermes_cli import observability
from hermes_cli.observability import shared_metrics_contract as contract
from hermes_cli.observability import shared_metrics_fields as fields
from hermes_cli.observability import shared_metrics_update as update_metrics

_SCHEMA = Path(observability.__file__).parent / "schemas/hermes.shared_metrics.v4.schema.json"


def _schema_dimensions(metric: str) -> list[dict]:
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8-sig"))
    (definition,) = (d for d in schema["$defs"].values() if d.get("properties", {}).get("name", {}).get("const") == metric)
    dims = definition["properties"]["dimensions"]
    return dims["oneOf"] if "oneOf" in dims else [dims]


@pytest.mark.parametrize("metric", [contract.EXTENSION_INSTALL_METRIC, contract.UPDATE_RUN_METRIC])
def test_schema_current_shape_is_exactly_the_contract_and_the_pre_split_shape_still_drains(metric):
    current, legacy = _schema_dimensions(metric)
    values = contract._COUNTER_DIMENSION_VALUES[metric]
    assert set(current["required"]) == set(values) == contract._METRIC_FIELDS[metric] - set(
        contract._IDENTIFIER_FIELDS.get(metric, ()))
    for field, spec in current["properties"].items():
        if "enum" in spec:
            assert set(spec["enum"]) == set(values[field]), field
    assert set(legacy["required"]) in {frozenset(s) for s in contract._LEGACY_METRIC_FIELDS[metric]}


def test_registry_ids_are_exactly_the_hub_router_adapters():
    from tools.skills_hub_search import create_source_router

    assert {src.source_id() for src in create_source_router()} == contract.EXTENSION_REGISTRY_IDS
    assert contract.EXTENSION_REGISTRIES == contract.EXTENSION_REGISTRY_IDS | {"none", "other", "unresolved"}
    assert contract.EXTENSION_FAILURE_CLASSES == frozenset().union(*contract.EXTENSION_KIND_FAILURE_CLASSES.values())


def test_a_raised_install_error_is_classified_by_type_and_never_carries_its_text():
    """Invariant: no free-form string reaches failure_class; an untagged exception keeps only its type's
    class, a user path in its message never leaves."""
    from hermes_cli.plugins_cmd import PluginOperationError

    leaky = "/home/alice/secret-repo: Permission denied"
    cases = [
        ("plugin", PluginOperationError(leaky, failure_class="clone_failed"), "clone_failed"),
        ("plugin", PluginOperationError(leaky), "other"),
        ("mcp_server", PermissionError(13, leaky), "permission"),
        ("mcp_server", ConnectionRefusedError(leaky), "network"),
        ("skill", FileNotFoundError(leaky), "filesystem_error"),
        ("skill", type("AcmeInternalError", (Exception,), {})(leaky), "exception"),
        ("skill", type("Tagged", (Exception,), {"failure_class": leaky})(), "other"),
    ]
    for kind, error, expected in cases:
        dims = fields.extension_install_fields(kind=kind, source="url", name=leaky, outcome="failed", error=error)
        assert dims["failure_class"] == expected, (kind, error)
        assert leaky not in json.dumps(dims)
        assert contract.counter_dimensions_are_valid(contract.EXTENSION_INSTALL_METRIC, dims)
    ok = fields.extension_install_fields(kind="skill", source="hub", name="x", outcome="success", registry="skills.sh")
    assert (ok["failure_class"], ok["registry"]) == ("none", "skills-sh")
    assert fields.extension_install_fields(kind="plugin", source="url", name=None, outcome="ok")["registry"] == "none"


def _receipt(outcome: str, stages: list[tuple[str, str]], **extra) -> dict:
    marks = [{"name": name, "outcome": result, "at": "2026-10-06T10:00:01+00:00",
              **({"mode": "git"} if name == "apply" else {})} for name, result in stages]
    return {"schema": 1, "update_id": "a" * 32, "started_at": "2026-10-06T10:00:00+00:00",
            "finished_at": "2026-10-06T10:00:02+00:00", "outcome": outcome, "pre_update": {}, "stages": marks,
            "steps": [], "fleet": [], **extra}


_ALL_PASSED = [("plan", "success"), ("snapshot", "success"), ("apply", "success"), ("deps", "success"),
               ("build", "success"), ("restart", "success")]


def test_a_failed_update_whose_every_stage_passed_names_where_and_why_it_failed():
    """Invariant: a failed git run never reads failed_stage=other. Both receipt shapes that did on main:
    the post-restart verification's ``partial`` with a clean fleet, and a run ending on a skipped
    restart that left the fleet owing one (update_completion._complete_selected exit 1)."""
    partial = _receipt("partial", _ALL_PASSED, fleet=[{"state": "current"}],
                       runtime_outcomes=[{"outcome": "unaccounted"}])
    skipped = _receipt("failed", [*_ALL_PASSED[:5], ("restart", "skipped")], exit_code=1,
                       stop_reason="completion exited 1")
    windows = _receipt("partial", _ALL_PASSED, fleet=[{"state": "current"}],
                       gateway_restart={"incomplete": True, "phase_error": "x"})
    stale = _receipt("partial", _ALL_PASSED, fleet=[{"state": "stale"}])
    got = {}
    for label, receipt in {"partial": partial, "skipped": skipped, "windows": windows, "stale": stale}.items():
        run, _ = update_metrics.update_receipt_fields(receipt)
        assert run["apply_mode"] == "git" and run["outcome"] == "failed"
        assert run["failed_stage"] != "other", label
        assert contract.counter_dimensions_are_valid(contract.UPDATE_RUN_METRIC, run)
        got[label] = (run["failed_stage"], run.get("failure_class"))
    assert got == {
        "partial": ("verify", "fleet_unverified"), "skipped": ("restart", "restart_failed"),
        "windows": ("verify", "restart_failed"), "stale": ("verify", "fleet_stale"),
    }


@pytest.mark.parametrize(("failed", "failure_class"), [("build", "build_failed"), ("restart", "restart_failed")])
def test_a_partial_run_with_a_failed_stage_keeps_that_stage_and_its_verify_row_passes(failed, failure_class):
    """Invariant: the verification writes ``partial`` for a failed build or restart too, so a
    failed stage mark (not the synthetic verify stage) is where the run failed."""
    stages = [(name, "failed" if name == failed else result) for name, result in _ALL_PASSED]
    extra = {"gateway_restart": {"incomplete": True}} if failed == "restart" else {}
    run, stage_rows = update_metrics.update_receipt_fields(
        _receipt("partial", stages, exit_code=1, fleet=[{"state": "current"}], **extra))
    assert (run["failed_stage"], run["failure_class"]) == (failed, failure_class)
    assert [(s["stage"], s["outcome"]) for s in stage_rows if s["outcome"] == "failed"] == [(failed, "failed")]


def test_a_run_that_died_before_a_stage_mark_has_a_failed_row_for_that_stage():
    """Invariant: stage rows agree with the run row on where a FAILED run stopped, and a committed run
    that owes follow-ups (C3) is a run-level success whose failed stage still shows at stage level."""
    fetch_failed = _receipt("failed", [("plan", "success"), ("snapshot", "skipped")], exit_code=1,
                            stop_reason="sys.exit(1)")
    run, stage_rows = update_metrics.update_receipt_fields(fetch_failed)
    failed_rows = [s for s in stage_rows if s["outcome"] == "failed"]
    assert run["failed_stage"] == "apply" and [s["stage"] for s in failed_rows] == ["apply"]
    assert all(contract.counter_dimensions_are_valid(contract.UPDATE_STAGE_METRIC, s) for s in stage_rows)

    deps_owed = _receipt("success", [*_ALL_PASSED[:3], ("deps", "failed")],
                         followups=[{"step": "dependencies", "reason": "uv sync exited 2"}])
    run, stage_rows = update_metrics.update_receipt_fields(deps_owed)
    assert (run["outcome"], run["failed_stage"]) == ("success", "none")
    assert [s["stage"] for s in stage_rows if s["outcome"] == "failed"] == ["deps"]


@pytest.mark.parametrize(("receipt", "expected"), [
    (_receipt("success", _ALL_PASSED), "none"),
    (_receipt("refused", [], steps=[{"name": "admission", "ok": False}], stop_reason="docker"), "managed_install"),
    (_receipt("refused", [("plan", "success"), ("snapshot", "success")], exit_code=2,
              stop_reason="historical takeover completion"), "lock_held"),
    (_receipt("failed", [("plan", "success"), ("snapshot", "success")], exit_code=1, stop_reason="sys.exit(1)"),
     "aborted_before_apply"),
    (_receipt("failed", [("plan", "success"), ("snapshot", "success")]), "git_failed"),
    (_receipt("failed", [("plan", "success")], exit_code=1,
              stop_reason="KeyboardInterrupt: /home/alice/x"), "interrupted"),
    (_receipt("failed", [("plan", "success")], exit_code=1, stop_reason="PermissionError: [Errno 13] /x"),
     "permission_denied"),
    (_receipt("failed", [("plan", "success")], exit_code=1, stop_reason="OSError: [Errno 28] No space left: /x"),
     "disk_full"),
    (_receipt("failed", [("plan", "success")], exit_code=1, stop_reason="FileExistsError: [Errno 17] /x"),
     "os_error"),
    (_receipt("failed", [("plan", "success")], exit_code=1, stop_reason="InstallError: venv: uv"), "deps_failed"),
    (_receipt("failed", [("plan", "success"), ("snapshot", "success"), ("apply", "success")], exit_code=1,
              stop_reason="completion exited 1"), "deps_failed"),
    (_receipt("failed", _ALL_PASSED[:4] + [("build", "failed"), ("restart", "success")], exit_code=1), "build_failed"),
    (_receipt("failed", [("plan", "success")], exit_code=1, stop_reason="AcmeError: secret"), "exception"),
    (_receipt("failed", _ALL_PASSED, exit_code=1, stop_reason="Windows gateway recovery failed: PermissionError: x"),
     "restart_failed"),
])
def test_update_failure_class_reads_only_receipt_fields(receipt, expected):
    run, _ = update_metrics.update_receipt_fields(receipt)
    assert run["failure_class"] == expected
    assert contract.counter_dimensions_are_valid(contract.UPDATE_RUN_METRIC, run)


class _HostileError(Exception):
    """A third-party exception whose attribute hooks raise (a PM worker, an SDK under -W error)."""

    def __getattr__(self, name):
        raise RuntimeError("hostile getattr")


def test_failure_classifiers_never_raise_on_an_exception_whose_attribute_hooks_raise():
    """Invariant: classification is inert. The raise-site classifiers run outside the metrics guard
    (collection on or off), so a raising hook must not replace the user's error, and the metric row
    still records (as ``exception``) instead of being dropped."""
    from hermes_cli.mcp_config import probe_failure_class
    from hermes_cli.plugins_cmd_install import _publish_failure_class

    assert _publish_failure_class(_HostileError("x")) == "deps_failed"
    assert probe_failure_class(_HostileError("x")) == "connect_failed"
    for kind in contract.EXTENSION_KINDS:
        dims = fields.extension_install_fields(kind=kind, source="url", name=None, outcome="failed", error=_HostileError("x"))
        assert dims["failure_class"] == "exception", kind
        assert contract.counter_dimensions_are_valid(contract.EXTENSION_INSTALL_METRIC, dims)


def test_contract_rejects_contradictory_extension_install_rows():
    """Invariant: the validator enforces what the per-field enums cannot: a kind's own failure set,
    ``none`` exactly on success, a registry only on skill rows. Pre-split rows still validate."""
    ok = {"kind": "skill", "name": "custom", "outcome": "failed", "source": "hub", "failure_class": "ambiguous",
          "registry": "github"}
    assert contract.counter_dimensions_are_valid(contract.EXTENSION_INSTALL_METRIC, ok)
    for bad in ({"kind": "plugin", "registry": "none"}, {"outcome": "success"}, {"failure_class": "none"},
                {"kind": "mcp_server", "failure_class": "other"}):
        assert not contract.counter_dimensions_are_valid(contract.EXTENSION_INSTALL_METRIC, {**ok, **bad}), bad
    legacy = {key: ok[key] for key in ("kind", "name", "outcome", "source")}
    assert contract.counter_dimensions_are_valid(contract.EXTENSION_INSTALL_METRIC, legacy)


def test_a_declined_dependency_consent_carries_its_class_not_its_copy(monkeypatch):
    """The refusal reason names its closed class where it is produced, so rewording the copy cannot
    turn a decline into ``manifest_invalid``; the user still reads the same text."""
    from types import SimpleNamespace

    from hermes_cli import plugins_cmd_install

    console = SimpleNamespace(print=lambda *a, **k: None)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    consented, reason = plugins_cmd_install._consent_python_deps("p", ("dep",), console)
    assert (consented, reason) == (False, "dependency install skipped (non-interactive)")
    assert fields.tagged_failure_class(reason) == "non_interactive"
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    monkeypatch.setattr(plugins_cmd_install, "_ask_yes_no", lambda *a: False)
    assert fields.tagged_failure_class(plugins_cmd_install._consent_python_deps("p", ("dep",), console)[1]) == "deps_declined"


@pytest.mark.parametrize(("identifier", "registry"), [
    ("skills.sh/acme/x", "skills-sh"), ("skils-sh/acme/x", "skills-sh"), ("OFFICIAL/x/y", "official"),
    ("lobehub/x", "lobehub"), ("well-known:https://h/x", "well-known"), ("https://h/.well-known/skills/x", "well-known"),
    ("https://h/a/SKILL.md", "url"), ("acme-corp/skills/x", "unresolved"),
])
def test_an_unserved_identifier_names_the_registry_its_adapter_accepts(identifier, registry):
    from hermes_cli.skills_hub import _registry_from_identifier

    assert _registry_from_identifier(identifier) == registry


@pytest.mark.parametrize(("url", "expected"), [
    ("n8n.example.com/mcp-server/http", "config_invalid"),     # a pasted URL with no scheme
    ("${HERMES_TEST_UNSET_MCP_URL}", "missing_credentials"),   # the URL's setup value never arrived
])
def test_an_oauth_card_install_that_fails_before_authorization_keeps_its_class(monkeypatch, url, expected):
    """Invariant: the card/agent OAuth install (``mcp_oauth.start``) rebuilds the worker's error from
    its text, so it must carry the worker's closed class; the row is never the bare ``exception``."""
    from tools.connectors import mcp_oauth

    monkeypatch.delenv("HERMES_TEST_UNSET_MCP_URL", raising=False)
    with pytest.raises(Exception) as raised:
        mcp_oauth.start("n8n-probe", cfg={"url": url, "auth": "oauth"}, url_timeout=60)
    dims = fields.extension_install_fields(kind="mcp_server", source="catalog", name=None, outcome="failed",
                                           error=raised.value)
    assert dims["failure_class"] == expected
