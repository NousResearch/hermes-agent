"""Tests: ``hermes repos`` read commands — filters, JSON shape, drift report."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from hermes_cli import repos as repos_cmd
from hermes_cli import repos_provenance as prov
from hermes_cli import repos_registry as reg


def _row(
    name,
    *,
    surface="plugin",
    status=prov.STATUS_ENABLED,
    sha="a" * 40,
    repo="https://github.com/o/r",
):
    return prov.RepoInstall(
        surface=surface, name=name, status=status, sha=sha, repo=repo
    ).to_row()


@pytest.fixture
def rows(monkeypatch):
    data = [
        _row("on-plugin"),
        _row("off-plugin", status=prov.STATUS_DISABLED),
        _row("a-skill", surface="skill", status=prov.STATUS_ENABLED),
    ]
    monkeypatch.setattr(
        prov,
        "collect_installs",
        lambda *a, **k: [
            prov.RepoInstall(
                surface=row["surface"],
                name=row["name"],
                status=row["status"],
                sha=row["sha"],
                repo=row["repo"],
                provenance=row["provenance"],
                path=row["path"],
                manage=row.get("manage", {}),
            )
            for row in data
        ],
    )
    return data


def test_filter_by_surface(rows):
    args = SimpleNamespace(surface="skill", enabled=False, disabled=False)
    assert [row["name"] for row in repos_cmd._filter_rows(rows, args)] == ["a-skill"]


def test_filter_enabled_and_disabled(rows):
    enabled = SimpleNamespace(surface=None, enabled=True, disabled=False)
    disabled = SimpleNamespace(surface=None, enabled=False, disabled=True)
    assert [row["name"] for row in repos_cmd._filter_rows(rows, enabled)] == [
        "on-plugin",
        "a-skill",
    ]
    assert [row["name"] for row in repos_cmd._filter_rows(rows, disabled)] == [
        "off-plugin"
    ]


def test_list_json_shape(rows, capsys):
    repos_cmd.cmd_list(
        SimpleNamespace(
            json=True, plain=False, surface=None, enabled=False, disabled=False
        )
    )
    out = json.loads(capsys.readouterr().out)
    assert {row["surface"] for row in out} == {"plugin", "skill"}
    assert set(out[0]) >= {
        "id",
        "surface",
        "name",
        "repo",
        "sha",
        "status",
        "provenance",
    }


def test_list_plain(rows, capsys):
    repos_cmd.cmd_list(
        SimpleNamespace(
            json=False, plain=True, surface=None, enabled=False, disabled=False
        )
    )
    out = capsys.readouterr().out
    assert "on-plugin" in out and "a-skill" in out


def test_show_unknown_name(rows, capsys):
    repos_cmd.cmd_show("nope")
    assert "No installed repo matching" in capsys.readouterr().out


def test_show_matches_name_and_id(rows, capsys):
    repos_cmd.cmd_show("a-skill")
    out = capsys.readouterr().out
    assert "skill:a-skill" in out
    repos_cmd.cmd_show("skill:a-skill")
    assert "skill:a-skill" in capsys.readouterr().out


def test_audit_reports_drift(rows, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(reg, "get_hermes_home", lambda: tmp_path)
    reg.write_registry([_row("stale", sha="b" * 40)])
    repos_cmd.cmd_audit()
    out = capsys.readouterr().out
    assert "Drift detected" in out
    assert "stale" in out
    assert "on-plugin" in out


def test_audit_clean(rows, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(reg, "get_hermes_home", lambda: tmp_path)
    reg.write_registry(
        [
            _row("on-plugin"),
            _row("off-plugin", status=prov.STATUS_DISABLED),
            _row("a-skill", surface="skill"),
        ]
    )
    repos_cmd.cmd_audit()
    assert "No drift" in capsys.readouterr().out


def test_significant_ignores_timestamps():
    row = _row("x")
    assert repos_cmd._significant(row) == repos_cmd._significant(
        {**row, "installed_at": "then", "updated_at": "now", "path": "/elsewhere"}
    )
