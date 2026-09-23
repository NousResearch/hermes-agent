"""Operator CLI and locale regression tests for engineering workflow."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

from agent.i18n import SUPPORTED_LANGUAGES, _load_catalog, t
from hermes_cli.engineering_cmd import _pick_assignments, _read_checks, build_parser, run_cli


def test_picker_assigns_only_models_from_shared_catalogue(monkeypatch):
    rows = {
        "providers": [
            {
                "slug": "provider-a",
                "name": "A",
                "authenticated": True,
                "models": ["one", "two"],
            },
            {
                "slug": "provider-b",
                "name": "B",
                "authenticated": False,
                "models": ["three"],
            },
        ]
    }
    monkeypatch.setattr(
        "hermes_cli.engineering_cmd.sys.stdin", SimpleNamespace(isatty=lambda: True)
    )
    monkeypatch.setattr("hermes_cli.engineering_cmd._catalogue", lambda: rows)
    picks = iter([0, 1, 0, 0, 0, 1])
    monkeypatch.setattr(
        "hermes_cli.main_provider_setup._prompt_provider_choice",
        lambda *a, **kw: next(picks),
    )
    result = _pick_assignments()
    assert result == {
        "planner": {"provider": "provider-a", "model": "two"},
        "worker": {"provider": "provider-a", "model": "one"},
        "reviewer": {"provider": "provider-a", "model": "two"},
    }


def test_noninteractive_picker_never_chooses_default_silently(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.engineering_cmd.sys.stdin", SimpleNamespace(isatty=lambda: False)
    )
    assert _pick_assignments() is None


def test_checks_file_is_operator_owned_argv_and_bounded(tmp_path):
    path = tmp_path / "checks.json"
    path.write_text(
        json.dumps([{"id": "unit", "argv": ["python", "-m", "pytest"], "timeout": 30}]),
        encoding="utf-8",
    )
    assert _read_checks(path)[0].argv == ("python", "-m", "pytest")
    path.write_text(
        json.dumps([{"id": "unit", "argv": "python -m pytest", "timeout": 30}]),
        encoding="utf-8",
    )
    with pytest.raises(ValueError):
        _read_checks(path)


def test_cli_reports_status_without_objective_or_model_data(
    monkeypatch, tmp_path, capsys
):
    checks = tmp_path / "checks.json"
    checks.write_text(
        json.dumps([{"id": "unit", "argv": ["true"], "timeout": 30}]), encoding="utf-8"
    )
    monkeypatch.setattr(
        "hermes_cli.engineering_cmd._pick_assignments",
        lambda: {
            stage: {"provider": "provider-a", "model": "model-a"}
            for stage in ("planner", "worker", "reviewer")
        },
    )
    monkeypatch.setattr(
        "hermes_cli.engineering_cmd._run_workflow",
        lambda **kw: SimpleNamespace(
            status="DONE",
            reason="verified",
            run_id="run",
            workspace_id="workspace",
            attempts=1,
            revision=1,
            stage_calls=2,
            decision_required="",
        ),
    )
    args = argparse.Namespace(
        objective="private objective",
        workspace=str(tmp_path),
        checks_file=str(checks),
        backend="native",
        image="",
    )
    assert run_cli(args) == 0
    output = capsys.readouterr().out
    assert json.loads(output)["status"] == "DONE"
    assert "private objective" not in output
    assert "model-a" not in output


def test_cli_reasoning_flags_bind_each_operator_stage(tmp_path, monkeypatch):
    checks = tmp_path / "checks.json"
    checks.write_text(
        json.dumps([{"id": "unit", "argv": ["true"], "timeout": 30}]), encoding="utf-8"
    )
    parser = argparse.ArgumentParser()
    build_parser(parser.add_subparsers(dest="command"))
    args = parser.parse_args([
        "engineering", "--objective", "Repair module",
        "--workspace", str(tmp_path), "--checks-file", str(checks),
        "--planner-reasoning", "medium", "--worker-reasoning", "medium",
        "--reviewer-reasoning", "high",
    ])
    monkeypatch.setattr(
        "hermes_cli.engineering_cmd._pick_assignments",
        lambda: {stage: {"provider": "provider-a", "model": stage}
                 for stage in ("planner", "worker", "reviewer")},
    )
    seen = {}

    def fake_run(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(
            status="DONE", reason="verified", run_id="run", workspace_id="workspace",
            attempts=1, revision=1, stage_calls=3, decision_required="",
        )

    monkeypatch.setattr("hermes_cli.engineering_cmd._run_workflow", fake_run)
    assert run_cli(args) == 0
    assert [seen["assignments"][stage]["reasoning_effort"] for stage
            in ("planner", "worker", "reviewer")] == ["medium", "medium", "high"]

def test_cli_planner_failure_reports_safe_code_without_provider_exception(
    monkeypatch, tmp_path, capsys
):
    secret = "synthetic-API-KEY-123456"
    checks = tmp_path / "checks.json"
    checks.write_text(
        json.dumps([{"id": "unit", "argv": ["true"], "timeout": 30}]), encoding="utf-8"
    )
    monkeypatch.setattr(
        "hermes_cli.engineering_cmd._pick_assignments",
        lambda: {stage: {"provider": "provider-a", "model": "model-a"}
                 for stage in ("planner", "worker", "reviewer")},
    )
    monkeypatch.setattr("hermes_cli.inventory.load_picker_context", lambda: object())
    monkeypatch.setattr(
        "hermes_cli.inventory.build_model_options_payload",
        lambda *args, **kwargs: {
            "providers": [{
                "slug": "provider-a", "models": ["model-a"], "authenticated": True,
            }]
        },
    )

    def failed_request(**kwargs):
        raise RuntimeError(f"Authorization: Bearer {secret}")

    monkeypatch.setattr("agent.engineering_runner.call_llm", failed_request)
    args = argparse.Namespace(
        objective="Private task", workspace=str(tmp_path), checks_file=str(checks),
        backend="native", image="",
    )
    assert run_cli(args) == 2
    output = capsys.readouterr().out
    assert json.loads(output)["reason"] == "provider_request_failed"
    assert secret not in output
    assert "Authorization" not in output

@pytest.mark.parametrize("lang", SUPPORTED_LANGUAGES)
def test_engineering_strings_cover_authoritative_locale_registry(lang):
    for key in (
        "planner",
        "worker",
        "reviewer",
        "select_provider",
        "select_model",
        "command_help",
        "objective_help",
        "workspace_help",
        "checks_help",
        "backend_help",
        "image_help",
        "reasoning_help",
    ):
        full_key = f"engineering.{key}"
        assert full_key in _load_catalog(lang)
        result = t(full_key, lang=lang, stage="Planner")
        assert result and result != full_key


def test_arabic_picker_labels_are_translated_and_keep_stage_placeholder():
    text = t("engineering.select_provider", lang="ar", stage="المنفذ")
    assert "المنفذ" in text
    assert text != t("engineering.select_provider", lang="en", stage="Worker")
