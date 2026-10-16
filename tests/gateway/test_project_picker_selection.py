"""Picker listing/labels/eligibility (#138777): stable ids, archived exclusion, distinguishable labels."""

import pytest

from gateway.project_picker import (
    eligible_projects,
    list_projects_for_profile,
    new_request_for_event,
    parse_new_request,
    picker_page,
)


def _row(pid, name, slug, folders=("/w",), archived=False):
    return {"id": pid, "name": name, "slug": slug, "archived": archived,
            "folders": [{"path": p, "label": ""} for p in folders]}


def test_archived_and_folderless_projects_are_never_listed():
    rows = [_row("a", "A", "a"), _row("b", "B", "b", archived=True), _row("c", "C", "c", folders=())]
    assert [p["id"] for p in eligible_projects(rows)] == ["a"]


def test_long_duplicate_names_keep_their_distinguishing_slug_within_telegram_limit():
    long_name = "A very long Desktop project name " * 3
    state = {"projects": [_row("1", long_name, "alpha-one"), _row("2", long_name, "beta-two")]}
    _, choices, _ = picker_page(state, 0)
    labels = [c["label"] for c in choices]
    assert len(set(labels)) == 2
    assert all(len(label) <= 60 for label in labels)
    assert labels[0].endswith("alpha-one") and labels[1].endswith("beta-two")


def test_duplicate_folder_basenames_and_parents_stay_distinguishable():
    state = {"folders": [
        {"path": "/a/team/app", "label": ""}, {"path": "/b/team/app", "label": ""},
        {"path": "/c/other/app", "label": ""}, {"path": "/c/other/app/", "label": ""},
    ]}
    _, choices, _ = picker_page(state, 0, folder=True)
    labels = [c["label"] for c in choices]
    assert len(set(labels)) == len(labels)
    assert all(len(label) <= 60 for label in labels)


def test_page_actions_index_the_stable_snapshot_and_cover_every_project():
    state = {"projects": [_row(str(i), f"P{i:02d}", f"p{i}") for i in range(19)]}
    seen = []
    page = 0
    while True:
        _, choices, controls = picker_page(state, page)
        seen += [int(c["action"].split("_")[1]) for c in choices]
        nxt = [c for c in controls if c["label"] == "▶"]
        if not nxt:
            break
        page = int(nxt[0]["action"].split("_")[1])
    assert seen == list(range(19))


@pytest.mark.parametrize("text", ["/new it's done", "/new say \"hi", "/new --project=x", "/new a -project"])
def test_other_platforms_keep_legacy_new_titles_verbatim(text):
    from gateway.config import Platform
    from gateway.platforms.event import MessageEvent
    from gateway.session import SessionSource

    source = SessionSource(platform=Platform.DISCORD, chat_id="c", user_id="u", chat_type="dm")
    request = new_request_for_event(MessageEvent(text=text, source=source))
    assert request.kind == "plain" and request.title == text.split(" ", 1)[1]


def test_apostrophe_titles_stay_plain_on_telegram():
    assert parse_new_request("it's done", strict_flags=True).kind == "plain"
    assert parse_new_request("release -notes", strict_flags=True).kind == "plain"  # dash after word 0 is text


def test_registry_listing_reports_archived_so_revalidation_can_drop_it(tmp_path):
    from hermes_cli import projects_db

    home = tmp_path / "p"
    home.mkdir()
    ws = tmp_path / "w"
    ws.mkdir()
    with projects_db.connect_closing(home / "projects.db") as conn:
        pid = projects_db.create_project(conn, name="Gone", folders=[str(ws)])
        assert eligible_projects(list_projects_for_profile(home))
        projects_db.archive_project(conn, pid)
    rows = list_projects_for_profile(home)
    assert rows[0]["archived"] and eligible_projects(rows) == []
