"""Tests for the non-blocking locale source-equality audit."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.locale_source_equality import (
    filter_by_namespace,
    find_source_equal_keys,
    flatten_catalog,
    format_report,
    group_by_namespace,
    main,
)


def _write_catalog(path: Path, payload: dict) -> None:
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")


def test_flatten_catalog_keeps_only_string_leaves():
    assert flatten_catalog(
        {
            "approval": {"header": "Header", "count": 2},
            "gateway": {"status": {"model": "Model"}},
            "ignored": ["list"],
        }
    ) == {
        "approval.header": "Header",
        "gateway.status.model": "Model",
    }
    assert flatten_catalog([]) == {}


def test_find_source_equal_keys_is_exact_and_ignores_missing_keys():
    source = {"same": "Text", "case": "Text", "missing": "Text"}
    target = {"same": "Text", "case": "text"}
    assert find_source_equal_keys(source, target) == {"same"}


def test_group_by_parent_namespace_is_deterministic():
    groups = group_by_namespace(
        {
            "gateway.model.switched",
            "approval.header",
            "gateway.status.model",
            "gateway.status.context",
        }
    )
    assert groups == {
        "approval": ["approval.header"],
        "gateway.model": ["gateway.model.switched"],
        "gateway.status": ["gateway.status.context", "gateway.status.model"],
    }


def test_namespace_filter_includes_descendants():
    keys = {"approval.header", "gateway.status.model", "gateway.context.header"}
    assert filter_by_namespace(keys, "gateway.status") == {"gateway.status.model"}
    assert filter_by_namespace(keys, "gateway") == {
        "gateway.status.model",
        "gateway.context.header",
    }
    assert filter_by_namespace(keys, None) == keys


def test_report_states_that_matches_need_review():
    report = format_report(
        {"gateway.status": ["gateway.status.model"]}, "en", "fr"
    )
    assert "Source-equal review: en -> fr" in report
    assert "[gateway.status] (1)" in report
    assert "Total: 1" in report
    assert "not an automatic defect" in report


def test_cli_is_non_blocking_by_default(tmp_path, capsys):
    source = tmp_path / "en.yaml"
    target = tmp_path / "fr.yaml"
    _write_catalog(source, {"gateway": {"status": {"model": "Model"}}})
    _write_catalog(target, {"gateway": {"status": {"model": "Model"}}})

    assert main(["--source", str(source), "--target", str(target)]) == 0
    assert "gateway.status.model" in capsys.readouterr().out


def test_cli_fail_mode_and_namespace_filter(tmp_path, capsys):
    source = tmp_path / "en.yaml"
    target = tmp_path / "fr.yaml"
    payload = {
        "approval": {"header": "Same"},
        "gateway": {"status": {"model": "Same"}},
    }
    _write_catalog(source, payload)
    _write_catalog(target, payload)

    assert main(
        [
            "--source",
            str(source),
            "--target",
            str(target),
            "--namespace",
            "gateway.status",
            "--fail-on-equal",
        ]
    ) == 1
    output = capsys.readouterr().out
    assert "gateway.status.model" in output
    assert "approval.header" not in output


def test_cli_fail_mode_stays_zero_when_filtered_scope_is_clean(tmp_path):
    source = tmp_path / "en.yaml"
    target = tmp_path / "fr.yaml"
    _write_catalog(source, {"gateway": {"status": {"model": "Model"}}})
    _write_catalog(target, {"gateway": {"status": {"model": "Modèle"}}})

    assert main(
        [
            "--source",
            str(source),
            "--target",
            str(target),
            "--namespace",
            "gateway.status",
            "--fail-on-equal",
        ]
    ) == 0


def test_cli_reports_missing_catalog(tmp_path, capsys):
    target = tmp_path / "fr.yaml"
    _write_catalog(target, {})
    assert main(
        [
            "--source",
            str(tmp_path / "missing.yaml"),
            "--target",
            str(target),
        ]
    ) == 2
    assert "Locale audit error" in capsys.readouterr().err
