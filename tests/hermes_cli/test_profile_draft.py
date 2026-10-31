"""Drafting a bot from a description: catalog-checked capabilities, partial regenerate, no writes.

The aux model is replaced by a canned reply at ``agent.auxiliary_client.call_llm`` (where
``draft_profile`` imports it); everything else — catalog read from a real profile tree, the
``profiles.draft`` RPC through the gateway dispatcher, the CLI create path — runs for real
against a temp ``HERMES_HOME``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import profile_draft

REPLY = {
    "name": "Inbox Triage!",
    "title": "Inbox Triage",
    "description": "Sorts new mail and flags what needs a human.",
    "sections": {
        "role": "You triage the user's inbox.",
        "voice": "Short and plain.",
        "does": "- Label mail\n- Flag urgent threads",
        "doesnt": "- Send replies",
        "silence": "Say nothing when the inbox is empty.",
    },
    "skills": ["himalaya", "made-up-skill"],
    "toolsets": ["web", "teleport"],
    "mcp_servers": ["gmail", "ghost-server"],
}


class _FakeLLM:
    def __init__(self, reply: str):
        self.reply, self.calls = reply, []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply))])


@pytest.fixture
def llm(monkeypatch):
    fake = _FakeLLM(json.dumps(REPLY))
    monkeypatch.setattr("agent.auxiliary_client.call_llm", fake)
    return fake


def _catalog() -> profile_draft.Catalog:
    return profile_draft.Catalog(skills=["email/himalaya", "github/github-pr"], toolsets=["web", "terminal"],
                                 mcp_servers=["gmail"])


def test_suggestions_outside_the_target_catalog_are_dropped_and_reported(llm):
    draft = profile_draft.draft_profile("triage my inbox", catalog=_catalog())

    assert draft.ok and draft.name == "inbox-triage"
    assert (draft.skills, draft.toolsets, draft.mcp_servers) == (["email/himalaya"], ["web"], ["gmail"])
    assert {(d["kind"], d["name"]) for d in draft.dropped} == {
        ("skills", "made-up-skill"), ("toolsets", "teleport"), ("mcp_servers", "ghost-server")}
    # The model was only offered what is installed.
    prompt = llm.calls[0]["messages"][1]["content"]
    assert "email/himalaya" in prompt and "ghost-server" not in prompt
    assert profile_draft.split_soul(draft.soul) == REPLY["sections"]


def test_regenerating_one_section_keeps_every_other_part_verbatim(llm):
    current = {"name": "keeper", "title": "Keeper", "description": "Old blurb.",
               "sections": {"role": "Old role.", "voice": "Old voice."}, "toolsets": ["terminal"]}

    draft = profile_draft.draft_profile("triage my inbox", catalog=_catalog(), current=current,
                                        note="friendlier", only="voice")

    assert draft.ok
    assert (draft.name, draft.title, draft.description) == ("keeper", "Keeper", "Old blurb.")
    assert draft.sections == {"role": "Old role.", "voice": "Short and plain.", "does": "", "doesnt": "",
                              "silence": ""}
    assert draft.toolsets == ["terminal"] and draft.dropped == []


@pytest.mark.parametrize("reply", ['{"name": "x", "sections": {"role": "cut', ""])
def test_unusable_reply_is_a_reason_not_an_exception(monkeypatch, reply):
    monkeypatch.setattr("agent.auxiliary_client.call_llm", _FakeLLM(reply))

    draft = profile_draft.draft_profile("anything", catalog=_catalog())

    assert not draft.ok and draft.reason and draft.soul.strip() == ""


# ── RPC + CLI against a temp HERMES_HOME ───────────────────────────────────────────────────


@pytest.fixture
def home(tmp_path, monkeypatch):
    from tui_gateway import server
    launch = tmp_path / ".hermes"
    skill = launch / "skills" / "email" / "himalaya"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: himalaya\n---\n", encoding="utf-8")
    (launch / "config.yaml").write_text("mcp_servers:\n  gmail:\n    command: gmail-mcp\n", encoding="utf-8")
    bot = launch / "profiles" / "scout"
    bot.mkdir(parents=True)
    (bot / "SOUL.md").write_text("# Scout\n\n## Role\n\nOld role.\n\n## Messaging other agents\n\nKeep me.\n",
                                 encoding="utf-8")
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(launch))
    monkeypatch.setattr(server, "_hermes_home", launch)
    return launch


def _tree(root: Path) -> dict:
    # Empty ``*.lock`` files are the read-side flock of the auth store, not profile content.
    return {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file() and p.suffix != ".lock"}


def test_rpc_drafts_from_the_target_catalog_and_writes_nothing(home, llm):
    from tui_gateway import server
    # Loading config seeds the launch home's defaults; the draft itself must not touch any profile.
    before = _tree(home / "profiles")

    resp = server._methods["profiles.draft"](1, {"description": "triage my inbox"})

    result = resp["result"]
    assert result["ok"] and result["skills"] == ["email/himalaya"] and result["mcp_servers"] == ["gmail"]
    assert [s["key"] for s in result["sections"]] == ["role", "voice", "does", "doesnt", "silence"]
    assert "previous" not in result
    assert _tree(home / "profiles") == before

    edit = server._methods["profiles.draft"](2, {"name": "scout", "description": "be terse"})["result"]
    assert edit["previous"]["sections"][0] == {"key": "role", "title": "Role", "text": "Old role."}
    assert "Keep me." in edit["previous"]["soul"]
    assert _tree(home / "profiles") == before


def _create_args(**over):
    base = dict(profile_name="mailbot", clone=False, clone_all=False, no_alias=True, no_skills=True,
                clone_from=None, clone_channels=False, sync_imports=False, description=None,
                from_prompt="triage my inbox", yes=False)
    return SimpleNamespace(**(base | over))


def test_cli_from_prompt_refuses_without_a_tty_unless_yes(home, llm, monkeypatch, capsys):
    from hermes_cli import profile_cmd
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    with pytest.raises(SystemExit):
        profile_cmd._profile_create(_create_args())
    assert not (home / "profiles" / "mailbot").exists() and llm.calls == []
    assert "--yes" in capsys.readouterr().out

    profile_cmd._profile_create(_create_args(yes=True))
    created = home / "profiles" / "mailbot"
    assert profile_draft.split_soul((created / "SOUL.md").read_text(encoding="utf-8")) == REPLY["sections"]
    assert "Sorts new mail" in (created / "profile.yaml").read_text(encoding="utf-8")


# ── review round: round trips, saved-profile edits, publish-time identity, target-scoped catalog ──


def test_rpc_result_sent_back_as_current_keeps_every_other_section(home, llm):
    from tui_gateway import server
    first = server._methods["profiles.draft"](1, {"description": "triage my inbox"})["result"]
    llm.reply = json.dumps(REPLY | {"title": "Changed", "sections": {k: "NEW" for k in REPLY["sections"]}})

    voice = server._methods["profiles.draft"](2, {"description": "triage my inbox", "current": first,
                                                  "note": "friendlier", "only": "voice"})["result"]
    title = server._methods["profiles.draft"](3, {"description": "triage my inbox", "current": first,
                                                  "note": "shorter", "only": "title"})["result"]

    texts = {s["key"]: s["text"] for s in voice["sections"]}
    assert texts == {k: ("NEW" if k == "voice" else v) for k, v in REPLY["sections"].items()}
    assert title["ok"] and title["title"] == "Changed" and title["sections"] == first["sections"]


def test_edit_mode_keeps_saved_text_and_capabilities_the_draft_cannot_see(home, llm):
    from tui_gateway import server
    skill = home / "profiles" / "scout" / "skills" / "email" / "himalaya"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: himalaya\n---\n", encoding="utf-8")

    edit = server._methods["profiles.draft"](1, {"name": "scout", "description": "x", "note": "terse",
                                                 "only": "voice"})["result"]

    # The model saw the whole saved file, and the proposed SOUL keeps the custom section.
    assert "Keep me." in llm.calls[0]["messages"][1]["content"]
    assert "## Messaging other agents\n\nKeep me." in edit["soul"]
    assert "## Voice\n\nShort and plain." in edit["soul"] and "Old role." in edit["soul"]
    # Only the voice was asked for: the enabled capabilities come back unchanged.
    assert edit["skills"] == edit["previous"]["skills"] == ["himalaya"]
    assert edit["toolsets"] == edit["previous"]["toolsets"]


def test_cli_from_prompt_publishes_the_profile_with_the_draft_identity(home, llm, monkeypatch):
    from hermes_cli import profile_cmd, profiles
    seen = {}

    def snapshot(canon):
        d = home / "profiles" / canon
        seen["soul"] = (d / "SOUL.md").read_text(encoding="utf-8-sig")
        seen["meta"] = (d / "profile.yaml").read_text(encoding="utf-8-sig")
    monkeypatch.setattr(profiles, "_notify_multiplexer", snapshot)

    profile_cmd._profile_create(_create_args(yes=True))

    assert profile_draft.split_soul(seen["soul"]) == REPLY["sections"]
    assert "Sorts new mail" in seen["meta"] and "Inbox Triage" in seen["meta"]


def test_catalog_lists_managed_mcp_servers_and_the_targets_toolsets(home, llm, tmp_path, monkeypatch):
    from hermes_cli import managed_scope
    from hermes_constants import get_hermes_home
    from tui_gateway import server
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / "config.yaml").write_text("mcp_servers:\n  corp:\n    command: corp-mcp\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    managed_scope.invalidate_managed_cache()
    homes = []
    monkeypatch.setattr(profile_draft, "_toolset_names", lambda: homes.append(get_hermes_home()) or ["web"])

    llm.reply = json.dumps(REPLY | {"mcp_servers": ["corp"]})
    result = server._methods["profiles.draft"](1, {"description": "triage", "source": "scout"})["result"]

    assert result["mcp_servers"] == ["corp"]  # scout has no config.yaml; the managed server still counts
    assert homes == [home / "profiles" / "scout"]


def test_edit_mode_section_sent_back_empty_is_removed_from_the_saved_soul(home, llm):
    from tui_gateway import server
    # The change request removes Role: the reply empties it and leaves every other part out.
    llm.reply = json.dumps({"sections": {"role": "", "voice": "Terse."}})

    edit = server._methods["profiles.draft"](1, {"name": "scout", "description": "x",
                                                 "note": "drop the role, be terse"})["result"]

    assert edit["ok"]
    assert "## Role" not in edit["soul"] and "Old role." not in edit["soul"]
    assert "## Voice\n\nTerse." in edit["soul"] and "## Messaging other agents\n\nKeep me." in edit["soul"]
    # Omitted parts are untouched, not cleared.
    assert edit["description"] == edit["previous"]["description"]
