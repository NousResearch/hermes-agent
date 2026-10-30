"""``hermes sessions outcome add|list|remove`` (#138653) end to end against a real temp-home state.db.

``list --json`` is the stable contract external consumers (dreaming/consolidation plugins) parse, so its
envelope and fields are pinned here. Provenance (``recorder``) is set by core, never by a flag.
"""

import json
import sys

import pytest

import hermes_state

_SID = "20261030_101500_abc123"
_FIELDS = {"id", "session_id", "signal", "verdict", "source", "recorder", "ref", "note", "observed_at", "recorded_at"}


@pytest.fixture()
def home(tmp_path, monkeypatch):
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    db = hermes_state.SessionDB(db_path=tmp_path / "state.db")
    db.create_session(_SID, source="cli")
    db.set_session_title(_SID, "Fix the flaky upload test")
    db.close()
    return tmp_path


def _run(monkeypatch, capsys, *argv):
    import hermes_cli.main as main_mod
    monkeypatch.setattr(sys, "argv", ["hermes", "sessions", "outcome", *argv])
    try:
        main_mod.main()
        code = 0
    except SystemExit as e:
        code = e.code or 0
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _list(monkeypatch, capsys, *argv):
    code, out, _ = _run(monkeypatch, capsys, "list", "--json", *argv)
    assert code == 0
    payload = json.loads(out)
    assert set(payload) == {"version", "outcomes"} and payload["version"] == 1
    return payload["outcomes"]


def test_add_resolves_prefix_and_title_then_list_json_is_the_documented_envelope(home, monkeypatch, capsys):
    code, out, _ = _run(monkeypatch, capsys, "add", "20261030_1015", "--signal", "merged", "--source", "github",
                        "--url", "https://github.com/o/r/pull/3", "--observed-at", "2026-10-29T12:00:00Z")
    assert code == 0 and f"merged (good) from github for session '{_SID}'." in out
    code, _, _ = _run(monkeypatch, capsys, "add", "Fix the flaky upload test", "--signal", "custom",
                      "--verdict", "neutral", "--source", "ci", "--note", "shipped behind a flag")
    assert code == 0

    custom, merged = _list(monkeypatch, capsys)
    assert set(custom) == _FIELDS == set(merged)
    assert {k: custom[k] for k in ("session_id", "signal", "verdict", "source", "recorder", "note")} == {
        "session_id": _SID, "signal": "custom", "verdict": "neutral", "source": "ci", "recorder": "cli",
        "note": "shipped behind a flag"}
    assert (merged["ref"], merged["observed_at"]) == ("https://github.com/o/r/pull/3", 1_793_275_200.0)
    assert isinstance(merged["id"], int) and isinstance(merged["recorded_at"], float)


def test_a_command_run_by_an_agent_records_as_agent_and_removes_only_agent_rows(home, monkeypatch, capsys):
    _run(monkeypatch, capsys, "add", _SID, "--signal", "merged", "--source", "github")
    monkeypatch.setenv("HERMES_SESSION_ID", _SID)
    code, out, _ = _run(monkeypatch, capsys, "add", _SID, "--signal", "ci_passed", "--source", "github", "--json")
    agent_row = json.loads(out)
    assert code == 0 and agent_row["recorder"] == "agent"
    user_row = next(r for r in _list(monkeypatch, capsys) if r["recorder"] == "cli")

    code, _, err = _run(monkeypatch, capsys, "remove", str(user_row["id"]))
    assert code == 3 and "may remove" in err
    assert _run(monkeypatch, capsys, "remove", str(agent_row["id"]))[0] == 0
    monkeypatch.delenv("HERMES_SESSION_ID")
    assert _run(monkeypatch, capsys, "remove", str(user_row["id"]))[0] == 0
    assert _list(monkeypatch, capsys) == []


@pytest.mark.parametrize("argv,code,needle", [
    (["add", "nope", "--signal", "merged", "--source", "github"], 3, "No session 'nope'"),
    (["list", "nope"], 3, "No session 'nope'"),
    (["remove", "999"], 3, "No outcome 999"),
    (["add", _SID, "--signal", "custom", "--source", "github"], 2, "no default"),
    (["add", _SID, "--signal", "merged", "--source", "self:cli"], 2, "plain label"),
    (["add", _SID, "--signal", "merged", "--source", "github", "--observed-at", "soon"], 2, "--observed-at"),
    (["list", "--since", "last tuesday"], 2, "Invalid value for --since"),
])
def test_bad_input_exits_with_the_documented_code_and_cause_on_stderr(home, monkeypatch, capsys, argv, code, needle):
    got, out, err = _run(monkeypatch, capsys, *argv)
    assert got == code and needle in err and out == ""


@pytest.mark.parametrize("argv", [
    ["add", _SID, "--signal", "shipped", "--source", "github"],  # outside the fixed vocabulary
    ["add", _SID, "--signal", "merged"],                         # --source is required
    ["add", _SID, "--signal", "merged", "--source", "g", "--current"],  # no --current: name the session
    ["add", _SID, "--signal", "merged", "--source", "g", "--recorder", "plugin:x"],  # provenance is core's
    [],
])
def test_usage_errors_exit_2(home, monkeypatch, capsys, argv):
    assert _run(monkeypatch, capsys, *argv)[0] == 2


def test_list_filters_and_human_table(home, monkeypatch, capsys):
    for signal in ("ci_passed", "reverted"):
        _run(monkeypatch, capsys, "add", _SID, "--signal", signal, "--source", "github")
    assert [r["signal"] for r in _list(monkeypatch, capsys, "--verdict", "bad", "--since", "1h")] == ["reverted"]
    assert [r["signal"] for r in _list(monkeypatch, capsys, _SID, "--signal", "ci_passed")] == ["ci_passed"]
    code, out, _ = _run(monkeypatch, capsys, "list", "--limit", "1")
    assert code == 0 and "reverted" in out and "ci_passed" not in out


def test_list_on_a_profile_without_a_store_is_an_empty_envelope(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("hermes_constants.get_hermes_home", lambda: tmp_path)
    assert _list(monkeypatch, capsys) == [] and not (tmp_path / "state.db").exists()
