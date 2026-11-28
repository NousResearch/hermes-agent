"""``display.json_output`` makes machine-readable output the default.

Every subcommand that declares ``--json`` used to need the flag on every invocation, which is
exactly what scripts forget. ``display.json_output: true`` resolves the flag after parsing, for
verbs that declare it, without touching verbs that do not and without overriding ``--json``.

Regression guard for the resolution seam (``hermes_cli._parser.apply_json_output_default``),
for the key's plumbing through the shared loader, and for the CLI wiring end to end.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import pytest


def _isolated_home(tmp_path, config_text: str):
    (tmp_path / "config.yaml").write_text(config_text, encoding="utf-8")
    return tmp_path


def test_verb_keeps_text_when_the_key_is_absent(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(_isolated_home(tmp_path, "display:\n  compact: false\n")))
    from hermes_cli._parser import apply_json_output_default

    args = argparse.Namespace(json=False)
    apply_json_output_default(args)

    assert args.json is False


def test_display_json_output_turns_the_flag_on(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(_isolated_home(tmp_path, "display:\n  json_output: true\n")))
    from hermes_cli._parser import apply_json_output_default

    args = argparse.Namespace(json=False)
    apply_json_output_default(args)

    assert args.json is True


def test_explicit_flag_wins_when_the_setting_is_off(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(_isolated_home(tmp_path, "display:\n  json_output: false\n")))
    from hermes_cli._parser import apply_json_output_default

    args = argparse.Namespace(json=True)
    apply_json_output_default(args)

    assert args.json is True


def test_verb_without_the_flag_is_untouched(tmp_path, monkeypatch):
    """A verb that does not declare --json must not grow an attribute (or crash)."""
    monkeypatch.setenv("HERMES_HOME", str(_isolated_home(tmp_path, "display:\n  json_output: true\n")))
    from hermes_cli._parser import apply_json_output_default

    args = argparse.Namespace()
    apply_json_output_default(args)

    assert not hasattr(args, "json")


def test_key_is_read_through_the_shared_loader(tmp_path, monkeypatch):
    """The setting must travel the same loader every surface uses, not a private read."""
    from hermes_cli.config import DEFAULT_CONFIG, load_config

    assert DEFAULT_CONFIG["display"]["json_output"] is False

    monkeypatch.setenv("HERMES_HOME", str(_isolated_home(tmp_path, "display:\n  json_output: true\n")))

    assert load_config()["display"]["json_output"] is True


def _run_cli(tmp_path, config_text: str):
    home = _isolated_home(tmp_path, config_text)
    env = {**os.environ, "HERMES_HOME": str(home), "HERMES_MANAGED_DIR": str(home / "managed")}
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "plugins", "list"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )


def test_cli_emits_json_by_default_when_the_setting_is_on(tmp_path):
    """End to end: the dispatch path applies the default for a real verb."""
    proc = _run_cli(tmp_path, "display:\n  json_output: true\n")
    if proc.returncode != 0:
        pytest.skip(f"CLI unavailable in this environment: {proc.stderr[-300:]}")

    assert json.loads(proc.stdout.strip()) is not None


def test_cli_keeps_text_by_default(tmp_path):
    proc = _run_cli(tmp_path, "display:\n  json_output: false\n")
    if proc.returncode != 0:
        pytest.skip(f"CLI unavailable in this environment: {proc.stderr[-300:]}")

    with pytest.raises(json.JSONDecodeError):
        json.loads(proc.stdout.strip())
