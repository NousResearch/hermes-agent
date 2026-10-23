"""Behaviour contracts for the terminal skills screen (``hermes skills``).

Two things have to hold for that screen to be safe to open: the remote skills the
profile's MCP servers publish must be listed so they can be switched off from the
same place as local ones, and saving must never drop a disable entry the screen
could not see. The second is the dangerous one — a server that is simply not
connected would otherwise have its skills silently switched back on.
"""

import pytest

from hermes_cli import skills_config, skills_remote


LOCAL = [{"name": "deploy", "description": "local deploy", "category": "general"}]
REMOTE = [{"name": "acme::team/jira", "description": "remote jira", "category": "mcp/acme",
           "server": "acme", "uri": "skill://team/jira/SKILL.md",
           "source": {"backend": "mcp", "remote": True}}]


@pytest.fixture
def stub_remote(monkeypatch):
    """Drive the remote half of ``_list_all_skills`` without a live server."""
    from tools import skills_tool_mcp

    state = {"rows": list(REMOTE), "servers": {"acme"}, "discovered": []}
    monkeypatch.setattr(skills_tool_mcp, "known_server_labels", lambda: set(state["servers"]))
    monkeypatch.setattr(skills_tool_mcp, "list_rows",
                        lambda *a, **k: list(state["rows"]) if state["rows"] else [])
    return state


def _patch_local(monkeypatch, rows=LOCAL):
    from tools import skills_tool
    monkeypatch.setattr(skills_tool, "find_all_skills", lambda **k: [dict(r) for r in rows])


# ── the listing itself ──────────────────────────────────────────────────────

def test_remote_skills_are_listed_alongside_local_ones(monkeypatch, stub_remote):
    _patch_local(monkeypatch)
    names = [s["name"] for s in skills_config._list_all_skills()]
    assert names == ["deploy", "acme::team/jira"]


def test_the_row_name_is_the_key_the_disable_list_stores(monkeypatch, stub_remote):
    """Whatever this screen toggles has to be what `skills.disabled` is matched on."""
    _patch_local(monkeypatch)
    row = next(s for s in skills_config._list_all_skills() if s["name"].startswith("acme::"))
    assert row["category"] == "mcp/acme"


def test_a_server_that_is_not_connected_leaves_the_local_listing_intact(monkeypatch, stub_remote):
    _patch_local(monkeypatch)
    stub_remote["rows"] = []
    names = [s["name"] for s in skills_config._list_all_skills()]
    assert names == ["deploy"]


def test_no_opted_in_server_means_no_connection_attempt(monkeypatch, stub_remote):
    _patch_local(monkeypatch)
    stub_remote["servers"] = set()
    stub_remote["rows"] = []
    called = []
    monkeypatch.setattr(skills_remote, "_connect_servers", lambda s: called.append(s))
    assert [s["name"] for s in skills_config._list_all_skills()] == ["deploy"]
    assert called == []


def test_an_empty_catalog_connects_once_to_the_opted_in_servers(monkeypatch, stub_remote):
    _patch_local(monkeypatch)
    stub_remote["rows"] = []
    called = []
    monkeypatch.setattr(skills_remote, "_connect_servers", lambda s: called.append(s))
    skills_config._list_all_skills()
    assert called == [["acme"]], "only the servers that publish skills are contacted"


def test_a_broken_remote_path_leaves_the_screen_working(monkeypatch, stub_remote):
    """Whatever the MCP side does — no SDK, an unreachable server, a bad catalog —
    the operator still gets their on-disk skills and can still toggle those."""
    from tools import skills_tool_mcp
    _patch_local(monkeypatch)

    def _boom(*_a, **_k):
        raise ImportError("mcp extra not installed")

    monkeypatch.setattr(skills_tool_mcp, "list_rows", _boom)
    assert [s["name"] for s in skills_config._list_all_skills()] == ["deploy"]


# ── saving must not lose what the screen could not show ─────────────────────

def test_an_entry_for_an_unlisted_skill_survives_a_save():
    """A remote skill whose server is down is not in the list — its entry must remain."""
    skills = [dict(LOCAL[0])]
    disabled = {"acme::team/jira", "plugin:thing", "deploy"}
    assert skills_config._disabled_after_selection(skills, disabled, chosen=set()) == disabled


def test_an_unlisted_entry_survives_even_when_every_listed_skill_is_enabled():
    skills = [dict(LOCAL[0])]
    new = skills_config._disabled_after_selection(skills, {"acme::team/jira", "deploy"}, chosen={0})
    assert new == {"acme::team/jira"}


def test_the_listed_rows_are_recomputed_from_the_selection():
    skills = [dict(LOCAL[0]), dict(REMOTE[0])]
    new = skills_config._disabled_after_selection(skills, set(), chosen={1})
    assert new == {"deploy"}


def test_switching_everything_off_disables_exactly_the_listed_rows():
    skills = [dict(LOCAL[0]), dict(REMOTE[0])]
    new = skills_config._disabled_after_selection(skills, {"stale::gone"}, chosen=set())
    assert new == {"stale::gone", "deploy", "acme::team/jira"}


# ── `hermes skills list` is a different surface and must show them too ──────

def test_do_list_shows_remote_skills_with_their_source(monkeypatch, stub_remote):
    """The plain listing is where most people look first; it gets the same rows."""
    import io

    from rich.console import Console

    from hermes_cli import skills_hub

    console = Console(file=io.StringIO(), width=200)
    skills_hub.do_list(console=console)
    out = console.file.getvalue()
    assert "acme::team/jira" in out, out
    assert "mcp/acme" in out, out
    assert "1 remote" in out, out


def test_do_list_omits_the_remote_count_when_there_are_none(monkeypatch, stub_remote):
    import io

    from rich.console import Console

    from hermes_cli import skills_hub

    stub_remote["servers"] = set()
    stub_remote["rows"] = []
    console = Console(file=io.StringIO(), width=200)
    skills_hub.do_list(console=console)
    assert "remote" not in console.file.getvalue()


def test_source_mcp_selects_only_the_remote_rows(monkeypatch, stub_remote):
    import io

    from rich.console import Console

    from hermes_cli import skills_hub

    _patch_local(monkeypatch)
    console = Console(file=io.StringIO(), width=200)
    skills_hub.do_list(source_filter="mcp", console=console)
    out = console.file.getvalue()
    assert "acme::team/jira" in out, out
    assert "deploy" not in out, out
