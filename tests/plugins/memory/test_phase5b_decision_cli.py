"""Phase 5b — integrate accepted ADRs into context() + `hermes memory decision` CLI.

Small milestone bridging Phase 5 (ADR storage) into real use:
- context() must surface accepted decisions (and never proposed drafts).
- The CLI verb exposes list/get/search/project/accept/draft through MemoryAPI.
"""

import json

import pytest

from hermes_cli.memory_api import MemoryAPI


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "hermes"
    (h / "memory" / "adr").mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(h))
    return h


@pytest.fixture
def api(home):
    return MemoryAPI()


def _ns(**kw):
    from types import SimpleNamespace

    return SimpleNamespace(**kw)


# --------------------------------------------------------------------------- #
# context() integration
# --------------------------------------------------------------------------- #
def test_context_includes_accepted_adrs(api):
    draft = api.draft_decision("Use Protocol interfaces", context="c", decision="use Protocol", project="hermes-aios")
    # proposed only -> not in context
    assert api.context("anything").decision == []
    # accept -> now surfaces in context
    api.accept_decision(draft.id, approved_by="joe")
    bundle = api.context("anything")
    ids = [d.extra.get("id") for d in bundle.decision]
    assert draft.id in ids


def test_context_excludes_proposed_from_decisions(api):
    api.draft_decision("Still a draft", context="c", decision="d", project="hermes-aios")
    assert api.context("x").decision == []


def test_context_decision_provenance_shape(api):
    draft = api.draft_decision("P", context="c", decision="d", project="hermes-aios")
    api.accept_decision(draft.id, approved_by="joe")
    d = api.context("x").decision[0]
    assert d.provider == "adr"
    assert d.layer == "L4"
    assert d.retrieval_method == "adr"
    assert d.extra.get("status") == "accepted"


# --------------------------------------------------------------------------- #
# CLI handler (hermes memory decision ...)
# --------------------------------------------------------------------------- #
def test_cli_decision_list_shows_accepted(api, capsys):
    from hermes_cli.main import _cmd_memory_decision

    draft = api.draft_decision("Accepted one", context="c", decision="yes", project="hermes-aios")
    api.accept_decision(draft.id, approved_by="joe")

    _cmd_memory_decision(_ns(decision_command="list", project=None, limit=20, json_output=False))
    out = capsys.readouterr().out
    assert "hermes-aios/001" in out
    assert "[accepted]" in out


def test_cli_decision_get_accepted(api, capsys):
    from hermes_cli.main import _cmd_memory_decision

    draft = api.draft_decision("Gettable", context="c", decision="d", project="hermes-aios")
    api.accept_decision(draft.id, approved_by="joe")

    _cmd_memory_decision(_ns(decision_command="get", decision_id=draft.id, json_output=False))
    out = capsys.readouterr().out
    assert draft.id in out


def test_cli_decision_get_proposed_is_invisible(api, capsys):
    from hermes_cli.main import _cmd_memory_decision

    draft = api.draft_decision("Not accepted", context="c", decision="d", project="hermes-aios")
    # The CLI get path only queries accepted; a proposed id yields nothing.
    _cmd_memory_decision(_ns(decision_command="get", decision_id=draft.id, json_output=False))
    out = capsys.readouterr().out
    assert "No accepted decision" in out


def test_cli_decision_accept_gates_authority(api, capsys):
    from hermes_cli.main import _cmd_memory_decision

    draft = api.draft_decision("To accept", context="c", decision="d", project="hermes-aios")
    _cmd_memory_decision(_ns(decision_command="accept", decision_id=draft.id, by="joe",
                              supersedes=[], status="accepted"))
    out = capsys.readouterr().out
    assert "Accepted" in out
    assert api.decision(id=draft.id)[0].status == "accepted"


def test_cli_decision_json_emits_records(api, capsys):
    from hermes_cli.main import _cmd_memory_decision

    draft = api.draft_decision("JSON one", context="c", decision="d", project="hermes-aios")
    api.accept_decision(draft.id, approved_by="joe")
    _cmd_memory_decision(_ns(decision_command="list", project=None, limit=20, json_output=True))
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list) and payload
    assert payload[0]["status"] == "accepted"
