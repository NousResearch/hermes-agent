"""``hermes kanban cap`` — show or set the live concurrency cap (#148006).

Reads print the configured value, the effective cap and its provenance; ``cap N``
writes ``kanban.max_in_progress`` through the normal config writer (comments and
unrelated keys preserved) and touches no other key.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from hermes_cli import kanban_db_dispatch as kbd


def _cmd_cap():
    from hermes_cli.kanban import _cmd_cap as handler
    return handler


def _ns(cap=None, json_=False) -> argparse.Namespace:
    return argparse.Namespace(kanban_action="cap", cap=cap, json=json_)


def _make_home(tmp_path, monkeypatch, config_text: str = "") -> Path:
    home = tmp_path / ".hermes"
    home.mkdir(exist_ok=True)
    (home / "config.yaml").write_text(config_text)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    return home


def _plain(obj):
    """Recursively convert ruamel mappings/sequences to plain dict/list."""
    if isinstance(obj, dict):
        return {k: _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def test_cap_prints_configured_effective_and_source(tmp_path, monkeypatch, capsys):
    _make_home(tmp_path, monkeypatch, "kanban:\n  max_in_progress: 20\n")
    rc = _cmd_cap()(_ns(cap=None))
    out = capsys.readouterr().out
    assert rc == 0
    assert "kanban.max_in_progress: 20" in out
    assert "effective cap:          20" in out
    assert "no restart" in out

    # Unset: the memory-derived default, with the provenance stated.
    monkeypatch.setattr(kbd, "_system_memory_sample", lambda: {"mem_total_kib": 16 * 1024 * 1024})
    _make_home(tmp_path, monkeypatch, "gateway:\n  loop_watchdog: true\n")
    rc = _cmd_cap()(_ns(cap=None))
    out = capsys.readouterr().out
    assert rc == 0
    assert "unset" in out
    assert "effective cap:          8" in out  # clamp(16 GiB / 512 MiB, 2..8)
    assert "memory-derived" in out


def test_cap_write_preserves_foreign_keys_and_comments(tmp_path, monkeypatch, capsys):
    before_text = (
        "# keep me\n"
        "\n"
        "gateway:\n"
        "  loop_watchdog: true\n"
        "\n"
        "kanban:\n"
        "  dispatch_interval_seconds: 60\n"
    )
    home = _make_home(tmp_path, monkeypatch, before_text)
    from hermes_cli.config import load_config_readonly

    before = _plain(load_config_readonly())
    rc = _cmd_cap()(_ns(cap=20))
    out = capsys.readouterr().out
    assert rc == 0

    text = (home / "config.yaml").read_text()
    assert "# keep me" in text, "YAML comments must survive the cap write"
    assert "loop_watchdog" in text

    after = _plain(load_config_readonly())
    assert after["kanban"]["max_in_progress"] == 20
    # No key outside kanban.max_in_progress changed.
    assert {k: v for k, v in before.items() if k != "kanban"} == {
        k: v for k, v in after.items() if k != "kanban"
    }
    assert after["kanban"]["dispatch_interval_seconds"] == 60
    assert "kanban.max_in_progress: 20" in out


def test_cap_rejects_non_positive(tmp_path, monkeypatch, capsys):
    before_text = "gateway:\n  loop_watchdog: true\n"
    home = _make_home(tmp_path, monkeypatch, before_text)
    rc = _cmd_cap()(_ns(cap=0))
    err = capsys.readouterr().err
    assert rc == 2
    assert "must be >= 1" in err
    assert (home / "config.yaml").read_text() == before_text, "config must be untouched"


def test_cap_json_shape(tmp_path, monkeypatch, capsys):
    from hermes_cli.config import get_config_path

    _make_home(tmp_path, monkeypatch, "kanban:\n  max_in_progress: 20\n")
    rc = _cmd_cap()(_ns(cap=None, json_=True))
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert set(payload) == {
        "configured", "effective", "source", "config_path", "dispatch_interval_seconds",
    }
    assert payload["configured"] == 20
    assert payload["effective"] == 20
    assert payload["source"] == "config"
    assert payload["config_path"] == str(get_config_path())
    assert payload["dispatch_interval_seconds"] == 60

    # Unset + memory unreadable -> unlimited provenance.
    monkeypatch.setattr(kbd, "_system_memory_sample", dict)
    _make_home(tmp_path, monkeypatch, "gateway:\n  loop_watchdog: true\n")
    rc = _cmd_cap()(_ns(cap=None, json_=True))
    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["configured"] is None
    assert payload["effective"] is None
    assert payload["source"] == "unlimited"
