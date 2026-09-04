"""Tests for `hermes curator audit` and the status registry-health block."""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout
from types import SimpleNamespace

import hermes_cli.curator as curator_cli
from tools.skills_duplicate_audit import DuplicateCandidate

_CANDIDATES = [
    DuplicateCandidate(
        name_a="ai-voice-cloning",
        name_b="cosyvoice2-voice-cloning",
        confidence="high",
        signals=("identical normalized body hash",),
        ownership_a="agent",
        ownership_b="agent",
    ),
    DuplicateCandidate(
        name_a="git-commit-helper",
        name_b="commit-message-generator",
        confidence="medium",
        signals=("similar descriptions (0.91)",),
        ownership_a="agent",
        ownership_b="hub",
    ),
]


def _run(handler, **kwargs) -> str:
    buf = io.StringIO()
    with redirect_stdout(buf):
        rc = handler(SimpleNamespace(**kwargs))
    assert rc == 0
    return buf.getvalue()


def test_audit_subcommand_is_registered():
    """A handler nobody can dispatch to is dead code."""
    parser = argparse.ArgumentParser(prog="hermes curator")
    curator_cli.register_cli(parser)

    args = parser.parse_args(["audit"])
    assert args.func is curator_cli._cmd_audit
    assert args.json is False

    assert parser.parse_args(["audit", "--json"]).json is True


def test_audit_prints_pairs_with_evidence(monkeypatch):
    """Every pair carries its signals and both origins — without the evidence the
    reader cannot tell a real duplicate from a shared template."""
    from tools import skills_duplicate_audit

    monkeypatch.setattr(skills_duplicate_audit, "scan_duplicates", lambda: _CANDIDATES)

    out = _run(curator_cli._cmd_audit, json=False)

    assert "ai-voice-cloning <-> cosyvoice2-voice-cloning" in out
    assert "identical normalized body hash" in out
    assert "High confidence" in out
    assert "Medium confidence" in out
    assert "hub" in out
    assert "not merge decisions" in out


def test_audit_json_round_trips(monkeypatch):
    from tools import skills_duplicate_audit

    monkeypatch.setattr(skills_duplicate_audit, "scan_duplicates", lambda: _CANDIDATES)

    payload = json.loads(_run(curator_cli._cmd_audit, json=True))

    assert payload["summary"] == {
        "possible_duplicate_pairs": 2,
        "high_confidence_pairs": 1,
        "medium_confidence_pairs": 1,
    }
    assert payload["candidates"][0]["name_a"] == "ai-voice-cloning"
    assert payload["candidates"][0]["signals"] == ["identical normalized body hash"]


def test_audit_reports_clean_registry(monkeypatch):
    from tools import skills_duplicate_audit

    monkeypatch.setattr(skills_duplicate_audit, "scan_duplicates", lambda: [])

    assert "None found" in _run(curator_cli._cmd_audit, json=False)


def test_registry_health_block_lists_counts(monkeypatch):
    from tools import skills_duplicate_audit

    monkeypatch.setattr(skills_duplicate_audit, "scan_duplicates", lambda: _CANDIDATES)

    buf = io.StringIO()
    with redirect_stdout(buf):
        curator_cli._print_registry_health()
    out = buf.getvalue()

    assert "registry health" in out
    assert "possible duplicate pairs  2" in out
    assert "curator audit" in out


def test_registry_health_stays_silent_when_clean(monkeypatch):
    """Status is already dense. A registry with nothing to report should add no
    lines at all."""
    from tools import skills_duplicate_audit

    monkeypatch.setattr(skills_duplicate_audit, "scan_duplicates", lambda: [])

    buf = io.StringIO()
    with redirect_stdout(buf):
        curator_cli._print_registry_health()

    assert buf.getvalue() == ""


def test_registry_health_survives_a_broken_scan(monkeypatch):
    """`status` reports curator state; a duplicate-scan failure is not a reason to
    deny the user the rest of it."""
    from tools import skills_duplicate_audit

    def _boom():
        raise OSError("skills directory vanished")

    monkeypatch.setattr(skills_duplicate_audit, "scan_duplicates", _boom)

    buf = io.StringIO()
    with redirect_stdout(buf):
        curator_cli._print_registry_health()

    assert buf.getvalue() == ""
