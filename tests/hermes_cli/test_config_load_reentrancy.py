"""A config load must not re-enter itself through a log handler.

A log record emitted from INSIDE the config-load critical section is rendered by handlers whose
formatter reads config (``agent/redact.py::RedactingFormatter`` → ``_redact_enabled`` →
``load_config_readonly``). ``_CONFIG_LOCK`` is an ``RLock``, so the same thread recursed instead of
blocking — one full config parse per emitted record, ~14 stack frames each.

Reported symptom: ``errors.log`` carried "Your settings file (…/config.yaml) has a formatting
error … Details: maximum recursion depth exceeded while calling a Python object" shortly after a
gateway start, and both quarantined copies parsed fine. The trigger was an unwritable
``backups/config`` (root-owned 0600 files written by another process), but the trap needs no second
process: any backup fault re-arms it, and the recursion tip lands in the load's own
``except Exception`` — which read it as broken YAML.
"""

from __future__ import annotations

import io
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli import config as cfg_mod
from hermes_cli import config_effective as cfg_eff_mod
from hermes_cli.config_backups import backup_config, list_config_backups, load_newest_good_backup

REPO = Path(__file__).resolve().parents[2]

VALID = (
    "security:\n"
    "  redact_secrets: true\n"
    "model:\n"
    "  default: test/guard\n"
    "approvals:\n"
    "  deny:\n"
    "    - 'curl*evil*'\n"
)
BROKEN = "approvals:\n  deny: [unclosed\n"

# A gateway start sits dozens of frames deep; the cycle only tipped into RecursionError at depth in
# production, so reproduce it at depth rather than assert about it.
LOAD_DEPTH = 300


def _load_at_depth(depth: int):
    """Call ``load_config()`` from *depth* frames down."""
    if depth <= 0:
        return cfg_mod.load_config()
    return _load_at_depth(depth - 1)


@pytest.fixture
def redacting_handler(tmp_path):
    """A WARNING-level root handler whose formatter reads config, like the real gateway wiring.

    ``_redact_enabled()`` only consults config under a HERMES_HOME override, so the fixture sets one
    for the duration (that is the multiplex/profile case the gateway runs in)."""
    from agent import redact as redact_mod
    from hermes_constants import reset_hermes_home_override, set_hermes_home_override

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(redact_mod.RedactingFormatter("%(levelname)s %(message)s"))
    root = logging.getLogger()
    previous_level = root.level
    redact_mod._REDACT_ENABLED_BY_HOME.clear()
    root.addHandler(handler)
    root.setLevel(logging.DEBUG)
    token = set_hermes_home_override(tmp_path)
    try:
        yield stream
    finally:
        reset_hermes_home_override(token)
        root.removeHandler(handler)
        root.setLevel(previous_level)
        redact_mod._REDACT_ENABLED_BY_HOME.clear()


@pytest.fixture
def config_parses(monkeypatch):
    """Count actual ``config.yaml`` parses — "one parse per load on this thread" is the invariant."""
    counter = {"count": 0}
    original = cfg_mod.fast_safe_load

    def counting(stream):
        counter["count"] += 1
        return original(stream)

    monkeypatch.setattr(cfg_mod, "fast_safe_load", counting)
    return counter


def test_a_failed_backup_does_not_re_enter_the_config_load(
        tmp_path, monkeypatch, redacting_handler, config_parses):
    """Unwritable ``backups/config`` + a config-reading log formatter: one parse, no quarantine."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(VALID, encoding="utf-8")
    backups = tmp_path / "backups" / "config"
    backups.mkdir(parents=True)
    backups.chmod(0o555)  # every backup attempt now raises EACCES from inside the load

    try:
        first = _load_at_depth(LOAD_DEPTH)
        assert config_parses["count"] == 1, "a log record from inside the load re-entered it"

        # A new signature forces a real re-parse, and therefore a real backup attempt.
        config_path.write_text(VALID + "# edited\n", encoding="utf-8")
        second = _load_at_depth(LOAD_DEPTH)
        assert config_parses["count"] == 2, "a log record from inside the load re-entered it"

        # The other loader holds the same critical section and can emit the same way.
        config_path.write_text(VALID + "# edited again\n", encoding="utf-8")
        effective = cfg_eff_mod.load_user_config_effective()
    finally:
        backups.chmod(0o755)

    assert config_parses["count"] == 2, "load_user_config_effective re-entered the merged load"
    assert effective["model"]["default"] == "test/guard"
    for loaded in (first, second):
        assert loaded["model"]["default"] == "test/guard"
        assert loaded["approvals"]["deny"] == ["curl*evil*"]
    # The failure warnings were emitted from inside the critical section: a handler that reads
    # config must never be handed them (that is exactly what recursed).
    log = redacting_handler.getvalue()
    assert "Could not back up" not in log
    assert "formatting error" not in log and "RecursionError" not in log
    assert not list(tmp_path.rglob("config.yaml.corrupt.*"))
    assert cfg_mod.get_active_config_parse_failure() is None


@pytest.mark.skipif(os.geteuid() == 0, reason="root reads a mode-000 file regardless of its mode")
def test_backups_that_cannot_be_read_do_not_disable_recovery(tmp_path, monkeypatch):
    """An unreadable newest ``good`` copy is skipped rather than ending the search, and a failed
    comparison is not turned into "cannot back up"."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    root = tmp_path / "backups" / "config"
    root.mkdir(parents=True)
    older = root / "config.yaml.good.20260101-000000"
    newest = root / "config.yaml.good.20260102-000000"
    older.write_text("model:\n  default: test/older\n", encoding="utf-8")
    # Same byte length as the content below: filecmp compares sizes first, so only an unreadable
    # candidate reaches the read that used to raise.
    newest.write_text("model:\n  default: test/oldes\n", encoding="utf-8")
    config_path.write_text("model:\n  default: test/newes\n", encoding="utf-8")

    newest.chmod(0o000)
    try:
        assert load_newest_good_backup(config_path) == {"model": {"default": "test/older"}}

        # A fresh process whose config.yaml is broken must still find the readable copy.
        config_path.write_text(BROKEN, encoding="utf-8")
        recovered, _stderr = _fresh_load(tmp_path)

        fresh_copy = backup_config(config_path, "good")
    finally:
        newest.chmod(0o644)

    assert recovered["model"]["default"] == "test/older"
    assert fresh_copy is not None and fresh_copy != newest
    assert fresh_copy.read_text(encoding="utf-8") == BROKEN
    assert len(list_config_backups(config_path, "good")) == 3


def _fresh_load(home: Path) -> tuple[dict, str]:
    env = {**os.environ, "HERMES_HOME": str(home), "PYTHONPATH": str(REPO)}
    proc = subprocess.run(
        [sys.executable, "-c",
         "import json; from hermes_cli.config import load_config; print(json.dumps(load_config()))"],
        cwd=REPO, env=env, text=True, capture_output=True, check=True, stdin=subprocess.DEVNULL)
    return json.loads(proc.stdout), proc.stderr


def test_malformed_yaml_is_still_loud_and_its_warning_does_not_re_parse(
        tmp_path, monkeypatch, caplog, redacting_handler, config_parses):
    """The parse-failure contract must not be weakened by the classification split — and the loud
    warning it emits, rendered by a config-reading handler from inside the load, must not reparse."""
    import time

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(VALID, encoding="utf-8")
    assert cfg_mod.load_config()["model"]["default"] == "test/guard"

    time.sleep(0.05)
    config_path.write_text(BROKEN, encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
        recovered = cfg_mod.load_config()

    # First load + this one; a re-entrant read is served in flight instead of parsing again.
    assert config_parses["count"] == 2
    assert recovered["model"]["default"] == "test/guard"  # last-known-good, not defaults
    assert any("formatting error" in record.getMessage() for record in caplog.records)
    rendered = redacting_handler.getvalue()
    assert "formatting error" in rendered and "RecursionError" not in rendered
    quarantined = list((tmp_path / "backups" / "config").glob("config.yaml.corrupt.*"))
    assert len(quarantined) == 1 and quarantined[0].read_text(encoding="utf-8") == BROKEN


def test_a_non_parse_load_fault_is_not_reported_as_broken_yaml(tmp_path, monkeypatch, caplog):
    """An unreadable file says nothing about its contents: report the real error, quarantine nothing."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(VALID, encoding="utf-8")
    real_open = open

    def deny_config_reads(file, mode="r", *args, **kwargs):
        if Path(file) == config_path and "r" in mode:
            raise PermissionError("denied")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", deny_config_reads)
    with caplog.at_level(logging.WARNING, logger="hermes_cli.config"):
        loaded = cfg_mod.load_config()

    messages = [record.getMessage() for record in caplog.records]
    assert any("PermissionError" in message for message in messages)
    assert not any("formatting error" in message for message in messages)
    assert loaded  # the load itself still succeeds (defaults here: nothing loaded yet)
    assert list(tmp_path.rglob("config.yaml.corrupt.*")) == []
    assert cfg_mod.get_active_config_parse_failure() is None
