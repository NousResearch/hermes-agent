"""Damaged opt-out state must not grant permission to reinstall default tools."""

import json

import pytest

from pm.defaults import declined, declined_path, default_packages, record_declined
from pm.package import InstallError


@pytest.mark.parametrize("payload", [
    None,
    [],
    {"schema": 1},
    {"schema": 1, "declined": None},
    {"schema": 1, "declined": "agent-browser"},
    {"schema": 1, "declined": {"agent-browser": True}},
    {"schema": 1, "declined": ["agent-browser", 1]},
    {"schema": 1, "declined": [""]},
    {"schema": 2, "declined": ["agent-browser"]},
])
def test_damaged_opt_out_refuses_selection_and_preserves_record(tmp_path, monkeypatch, payload):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    path = declined_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    original = json.dumps(payload).encode("utf-8")
    path.write_bytes(original)

    with pytest.raises(InstallError, match="cannot read"):
        default_packages(["agent-browser"])
    with pytest.raises(InstallError, match="cannot read"):
        record_declined(add=["cua-driver"])
    assert path.read_bytes() == original


def test_valid_opt_out_keeps_unknown_names_and_missing_record_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    assert declined() == frozenset()
    assert record_declined(add=["agent-browser", "future-default"]) == frozenset({"agent-browser", "future-default"})
    assert default_packages(["agent-browser"]) == []
    assert record_declined(remove=["agent-browser"]) == frozenset({"future-default"})
    assert declined() == frozenset({"future-default"})
