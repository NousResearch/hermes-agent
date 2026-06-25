from hermes_cli import agents_os
from hermes_cli.agents_os import AgentsOSService, connect, resolve_paths, utc_now
from hermes_cli.agents_os_seo import seo_mission_control_payload
from hermes_cli.agents_os_web import mission_control_html


import pytest


@pytest.fixture()
def agents_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes-home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("AGENTS_OS_HOME", str(home / "agents_os"))
    monkeypatch.setenv("AGENTS_OS_VAULT_ROOT", str(tmp_path / "vault"))
    agents_os.main(["init", "--no-vault"])
    return home


def test_seo_payload_empty_state_is_safe_local(agents_home):
    payload = seo_mission_control_payload(resolve_paths(None))

    assert payload["local_only"] is True
    assert payload["publish_enabled"] is False
    assert payload["outreach_enabled"] is False
    assert payload["credentials_required_for_live_metrics"] is True
    assert payload["goals"] == []
    assert payload["keyword_queue"] == []
    assert payload["draft_queue"] == []
    assert payload["review_gates"] == []
    assert {"publish", "outreach", "credentials", "analytics"}.issubset(set(payload["approval_gates"]))


def test_seo_payload_groups_local_tasks_and_artifacts(agents_home, tmp_path):
    paths = resolve_paths(None)
    goal = tmp_path / "goal.md"
    keyword = tmp_path / "keywords.md"
    draft = tmp_path / "draft.md"
    review = tmp_path / "review.md"
    for path in [goal, keyword, draft, review]:
        path.write_text("# draft only\n\nNE OBJAVLJUJ\n", encoding="utf-8")

    now = utc_now()
    with connect(paths) as conn:
        conn.execute(
            "INSERT INTO tasks(id,title,status,workflow,priority,created_at,updated_at,notes,route,approval_required) VALUES(?,?,?,?,?,?,?,?,?,?)",
            ("seo-task", "Example SEO Goal", "pending", "seo-goal", 2, now, now, "draft only", "local:direct", 0),
        )
        for artifact_id, kind, title, path, workflow in [
            ("seo-goal-art", "seo_goal", "SEO Goal", goal, "seo-goal"),
            ("keyword-art", "keyword_research", "Keyword Research", keyword, "keyword-research"),
            ("draft-art", "seo_draft", "SEO Draft", draft, "seo-draft"),
            ("review-art", "publish_gate", "Review Gate", review, "publish-gate"),
        ]:
            conn.execute(
                "INSERT INTO artifacts(id,kind,title,path,task_id,workflow,created_at) VALUES(?,?,?,?,?,?,?)",
                (artifact_id, kind, title, str(path), "seo-task", workflow, now),
            )
        conn.commit()

    payload = seo_mission_control_payload(paths)

    assert [item["id"] for item in payload["goals"]] == ["seo-task"]
    assert [item["id"] for item in payload["keyword_queue"]] == ["keyword-art"]
    assert [item["id"] for item in payload["draft_queue"]] == ["draft-art"]
    assert [item["id"] for item in payload["review_gates"]] == ["review-art"]
    assert payload["publish_enabled"] is False


def test_mission_control_html_contains_seo_panel(agents_home):
    html = mission_control_html(AgentsOSService(resolve_paths(None)))

    assert "SEO Mission Control" in html
    assert "/api/seo" in html
    assert "publish disabled" in html.lower()
