"""Replacing a cloud agent's data with a move bundle: what the incoming copy wins, what the hosting
platform keeps, and that undo restores every part.

Real files, a real ``state.db`` and the real offline session writer; nothing is stubbed.
"""
import json
import zipfile

import hermes_yaml as yaml

from hermes_cli import move_receive
from hermes_cli.move_bundle import build_bundle
from hermes_cli.move_sessions import OfflineSessions, visible_session_ids


def _skill(home, rel, body):
    d = home / "skills" / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\nname: {rel.split('/')[-1]}\ndescription: x\n---\n{body}\n")


def _seed(home, sessions):
    from hermes_state import SessionDB
    db = SessionDB(db_path=home / "state.db")
    try:
        db.import_sessions([{"id": sid, "source": "desktop", "title": title,
                             "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}
                            for sid, title in sessions])
    finally:
        db.close()


def _laptop(tmp_path):
    home = tmp_path / "laptop"
    (home / "memories").mkdir(parents=True)
    (home / "memories" / "MEMORY.md").write_text("laptop fact\n")
    (home / "SOUL.md").write_text("I am the laptop persona.\n")
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "local/llama", "provider": "custom"}, "display": {"skin": "midnight"},
        "web": {"backend": "tavily"}}))
    _skill(home, "custom/shared", "laptop copy")
    (home / "cron").mkdir()
    (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": [
        {"id": "laptop-job", "name": "brief", "prompt": "x", "schedule": {"kind": "cron", "expr": "0 9 * * *"}}]}))
    _seed(home, [("lap1", "Laptop chat")])
    return home


def _cloud(tmp_path):
    home = tmp_path / "cloud"
    (home / "memories").mkdir(parents=True)
    (home / "memories" / "MEMORY.md").write_text("cloud fact\n")
    (home / "SOUL.md").write_text("I am the cloud persona.\n")
    # What Hermes Cloud seeds: Nous model, the managed tool gateway, the Chronos scheduler.
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "nous/hermes", "provider": "nous"}, "display": {"skin": "default"},
        "web": {"backend": "firecrawl", "use_gateway": True}, "cron": {"provider": "chronos"}}))
    _skill(home, "custom/shared", "cloud copy")
    _skill(home, "custom/cloud-only", "stays")
    (home / "cron").mkdir()
    (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": [
        {"id": "cloud-job", "name": "cloud", "prompt": "y", "schedule": {"kind": "cron", "expr": "0 8 * * *"},
         "enabled": True}]}))
    (home / ".env").write_text("TOOL_GATEWAY_USER_TOKEN=cloud-owned\n")
    _seed(home, [("cl1", "Cloud chat")])
    return home


def _received(tmp_path, cloud):
    out = tmp_path / "move.zip"
    if not out.exists():
        build_bundle(_laptop(tmp_path), "default", out, include_history=True)
    data = out.read_bytes()
    import base64, hashlib
    state = move_receive.begin(cloud, size=len(data), sha256=hashlib.sha256(data).hexdigest())
    move_receive.chunk(cloud, state["id"], offset=0, data_b64=base64.b64encode(data).decode())
    return move_receive.finish(cloud, state["id"])


def test_replace_overwrites_the_agent_but_keeps_what_the_cloud_set_up_and_undo_restores_it(tmp_path):
    cloud = _cloud(tmp_path)
    before = {rel: (cloud / rel).read_bytes() for rel in
              ("config.yaml", "SOUL.md", "memories/MEMORY.md", "cron/jobs.json", ".env")}
    state = _received(tmp_path, cloud)
    assert state["preview"]["target"]["conversations"] == 1
    state = move_receive.apply(cloud, state["id"], sessions=OfflineSessions(cloud), replace=set(move_receive.STEPS))
    assert state["status"] == "applied", state

    assert (cloud / "memories" / "MEMORY.md").read_text() == "laptop fact"
    assert (cloud / "SOUL.md").read_text() == "I am the laptop persona.\n"
    assert "laptop copy" in (cloud / "skills" / "custom" / "shared" / "SKILL.md").read_text()
    assert (cloud / "skills" / "custom" / "cloud-only" / "SKILL.md").is_file()
    jobs = json.loads((cloud / "cron" / "jobs.json").read_text())["jobs"]
    assert [j["id"] for j in jobs] == ["laptop-job"] and jobs[0]["enabled"] is False
    assert visible_session_ids(cloud) == ["lap1"]
    config = yaml.safe_load((cloud / "config.yaml").read_text())
    assert config["display"]["skin"] == "midnight"
    # Hosting stays: model, managed tools, scheduler, credentials.
    assert config["model"] == {"default": "nous/hermes", "provider": "nous"}
    assert config["web"] == {"backend": "firecrawl", "use_gateway": True}
    assert config["cron"] == {"provider": "chronos"}
    assert (cloud / ".env").read_bytes() == before[".env"]

    state = move_receive.undo(cloud, state["id"], sessions=OfflineSessions(cloud))
    assert state["status"] == "undone" and not state["result"]["undo_problems"], state
    for rel, data in before.items():
        if rel == "cron/jobs.json":  # rewritten by the scheduler's own writer: compare the jobs
            assert json.loads((cloud / rel).read_text())["jobs"] == json.loads(data)["jobs"]
        else:
            assert (cloud / rel).read_bytes() == data, rel
    assert "cloud copy" in (cloud / "skills" / "custom" / "shared" / "SKILL.md").read_text()
    assert visible_session_ids(cloud) == ["cl1"]

    # Changing your mind after an undo: the same bundle again still brings its conversation, though
    # the undo deleted that id here and the store never restores a deleted id.
    again = _received(tmp_path, cloud)
    again = move_receive.apply(cloud, again["id"], sessions=OfflineSessions(cloud), replace={"sessions"})
    assert again["steps"]["sessions"]["imported"] == 1, again["steps"]["sessions"]
    assert [sid.startswith("lap1-m") for sid in visible_session_ids(cloud)] == [True]


def test_merge_by_default_and_replace_needs_the_part_named(tmp_path):
    cloud = _cloud(tmp_path)
    state = _received(tmp_path, cloud)
    # Only memory is replaced; everything else merges, so the cloud's own copies survive.
    state = move_receive.apply(cloud, state["id"], sessions=OfflineSessions(cloud), replace={"memory"})
    assert state["replace"] == ["memory"]
    assert (cloud / "memories" / "MEMORY.md").read_text() == "laptop fact"
    assert (cloud / "SOUL.md").read_text() == "I am the cloud persona.\n"
    assert "cloud copy" in (cloud / "skills" / "custom" / "shared" / "SKILL.md").read_text()
    assert {j["id"] for j in json.loads((cloud / "cron" / "jobs.json").read_text())["jobs"]} == {"cloud-job", "laptop-job"}
    assert sorted(visible_session_ids(cloud)) == ["cl1", "lap1"]
    with zipfile.ZipFile(tmp_path / "move.zip") as zf:
        assert ".env" not in zf.namelist()
