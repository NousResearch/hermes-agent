"""``hermes desktop --build-needed`` exposes the content-hash verdict."""

from __future__ import annotations

import argparse
import json

import hermes_cli.main as main
import hermes_cli.main_desktop as main_desktop


def _args(**overrides):
    return argparse.Namespace(build_needed=True, source=False, **overrides)


def test_reports_the_content_hash_verdict_as_json(monkeypatch, capsys, tmp_path):
    desktop = tmp_path / "apps" / "desktop"
    desktop.mkdir(parents=True)
    (desktop / "package.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    seen = {}

    def fake_needed(desktop_dir, project_root, *, source_mode):
        seen["args"] = (desktop_dir, project_root, source_mode)
        return True

    monkeypatch.setattr(main_desktop, "_desktop_build_needed", fake_needed)

    main_desktop.cmd_gui(_args())

    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1]) == {
        "build_needed": True,
        "source_mode": False,
    }
    assert seen["args"] == (desktop, tmp_path, False)


def test_a_probe_failure_answers_null_instead_of_crashing(monkeypatch, capsys, tmp_path):
    desktop = tmp_path / "apps" / "desktop"
    desktop.mkdir(parents=True)
    (desktop / "package.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)

    def broken(*_args, **_kwargs):
        raise RuntimeError("stamp unreadable")

    monkeypatch.setattr(main_desktop, "_desktop_build_needed", broken)
    main_desktop.cmd_gui(_args())

    output = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert output["build_needed"] is None
    assert "stamp unreadable" in output["error"]
