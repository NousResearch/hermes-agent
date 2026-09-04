"""Regression coverage for the bounded Knowledge Sync CLI surface."""

from __future__ import annotations

import json
import sys

import pytest


def test_knowledge_cli_is_registered_and_fails_closed_for_missing_manifest(
    tmp_path, monkeypatch, capsys
):
    from hermes_cli import main as main_module

    monkeypatch.setattr(main_module, "_plugin_cli_discovery_needed", lambda: False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes",
            "knowledge",
            "sync",
            "--manifest",
            str(tmp_path / "missing.json"),
            "--dry-run",
            "--json",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main_module.main()

    captured = capsys.readouterr()
    assert exc_info.value.code == 2
    assert captured.out, captured.err
    payload = json.loads(captured.out)
    assert payload["ok"] is False
    assert "manifest" in payload["error"].lower()
    assert "invalid choice" not in captured.err.lower()
