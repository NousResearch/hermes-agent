"""Move to Hermes Cloud: the bundle a source writes and the merge a running target applies.

Real files, real zip, real routes on the dashboard app. The one seam replaced is the session
import callback, which a live target routes through its gateway's revision-fenced mutation.
"""
import base64
import hashlib
import json
import zipfile
from pathlib import Path

import pytest
import hermes_yaml as yaml

from hermes_cli import move_receive
from hermes_cli.move_bundle import MANIFEST_NAME, build_bundle, keys_to_reenter, portable_config


def _skill(root: Path, rel: str, body: str = "do the thing") -> Path:
    d = root / "skills" / rel
    d.mkdir(parents=True, exist_ok=True)
    (d / "SKILL.md").write_text(f"---\nname: {rel.split('/')[-1]}\ndescription: x\n---\n{body}\n")
    return d


def _source_home(root: Path) -> Path:
    home = root / "source"
    (home / "memories").mkdir(parents=True)
    (home / "memories" / "MEMORY.md").write_text("prefers tabs\n§\nworks on hermes\n")
    (home / "memories" / "USER.md").write_text("name is Sam\n")
    (home / "SOUL.md").write_text("You are terse and kind.\n")
    (home / ".env").write_text("OPENROUTER_API_KEY=sk-or-real\nTELEGRAM_BOT_TOKEN=123:abc\n"
                               "FIRECRAWL_API_KEY=fc\nSUDO_PASSWORD=hunter2\nEMPTY_KEY=\n")
    (home / "auth.json").write_text('{"credential_pool": {"x": [{"access_token": "secret"}]}}')
    (home / "config.yaml").write_text(yaml.safe_dump({
        "model": {"default": "anthropic/claude-opus", "provider": "openrouter"},
        "display": {"skin": "midnight"},
        "terminal": {"backend": "local", "cwd": "/Users/sam/code"},
        "mcp_servers": {"local": {"command": "/usr/local/bin/thing"}, "remote": {"url": "https://mcp.x"},
                        "npx": {"command": "npx", "args": ["-y", "pkg"], "env": {"API_TOKEN": "t0k"}}},
        "delegation": {"api_key": "sk-leak", "model": "m"},
    }))
    _skill(home, "custom/mine")
    big = _skill(home, "data/huge")
    (big / "blob.bin").write_bytes(b"\0" * (26 * 1024 * 1024))
    pristine = _skill(home, "bundled/stock")
    from tools.skills_sync import dir_hash
    (home / "skills" / ".bundled_manifest").write_text(f"stock:{dir_hash(pristine)}\n")
    (home / "cron").mkdir()
    (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": [
        {"id": "job1", "name": "daily", "prompt": "summarize", "schedule": {"kind": "cron", "expr": "0 9 * * *"},
         "enabled": True, "state": "scheduled", "next_run_at": "2026-01-01T09:00:00", "last_status": "ok"},
        {"id": "job2", "name": "script", "prompt": "", "script": "/Users/sam/x.sh",
         "schedule": {"kind": "interval", "minutes": 30}, "enabled": True},
    ]}))
    return home


def _bundle(tmp_path: Path, **kw) -> tuple[Path, dict]:
    home = _source_home(tmp_path)
    out = tmp_path / "move.zip"
    report = build_bundle(home, "default", out, **kw)
    return out, report.as_dict()


def test_bundle_never_carries_credentials_or_machine_state(tmp_path):
    out, inv = _bundle(tmp_path, include_history=False)
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        blob = b"".join(zf.read(n) for n in names)
        config = yaml.safe_load(zf.read("config.yaml"))
        manifest = json.loads(zf.read(MANIFEST_NAME))
    assert not names & {".env", "auth.json", "state.db"}
    for secret in (b"sk-or-real", b"123:abc", b"hunter2", b"sk-leak", b"t0k", b"\"secret\""):
        assert secret not in blob, secret
    assert "terminal" not in config and "terminal" in inv["config_dropped"]
    assert config["display"] == {"skin": "midnight"}
    assert config["mcp_servers"]["npx"]["env"] == {}
    assert manifest["schema_version"] == 1 and manifest["kind"] == "hermes-move-bundle"


def test_bundle_inventory_names_what_stays_behind(tmp_path):
    _out, inv = _bundle(tmp_path, include_history=False)
    assert inv["skills"] == ["custom/mine"]
    assert [s["skill"] for s in inv["skills_skipped"]] == ["data/huge"]
    assert {m["name"]: m["portable"] for m in inv["mcp_servers"]} == {"local": False, "npx": True, "remote": True}
    keys = {k["key"]: k["category"] for k in inv["keys_to_reenter"]}
    assert keys == {"OPENROUTER_API_KEY": "provider", "TELEGRAM_BOT_TOKEN": "messaging"}
    jobs = {j["id"]: j["needs_this_machine"] for j in inv["cron_jobs"]}
    assert jobs == {"job1": [], "job2": ["script"]}


def test_portable_config_scrubs_nested_secrets():
    config, dropped = portable_config({"a": {"token": "x", "keep": 1, "list": [{"password": "p", "n": 2}]},
                                       "_config_version": 9, "browser": {}})
    assert config == {"a": {"keep": 1, "list": [{"n": 2}]}}
    assert dropped == ["_config_version", "browser"]


def test_keys_never_include_values(tmp_path):
    home = _source_home(tmp_path)
    assert all(set(k) == {"key", "category", "label"} for k in keys_to_reenter(home))


# --------------------------------------------------------------------------- receive


def _target_home(root: Path) -> Path:
    home = root / "target"
    (home / "memories").mkdir(parents=True)
    (home / "memories" / "MEMORY.md").write_text("cloud note\n§\nprefers tabs\n")
    (home / "config.yaml").write_text(yaml.safe_dump({"model": {"default": "nous/hermes", "provider": "nous"},
                                                      "display": {"skin": "default"}}))
    _skill(home, "custom/mine", "the cloud copy")
    (home / "cron").mkdir()
    (home / "cron" / "jobs.json").write_text(json.dumps({"jobs": []}))
    return home


def _upload(home: Path, data: bytes, chunk: int = 1024) -> dict:
    state = move_receive.begin(home, size=len(data), sha256=hashlib.sha256(data).hexdigest())
    for offset in range(0, len(data), chunk):
        state = move_receive.chunk(home, state["id"], offset=offset,
                                   data_b64=base64.b64encode(data[offset:offset + chunk]).decode())
    return move_receive.finish(home, state["id"])


class _Sessions:
    def __init__(self):
        self.seen = []

    def __call__(self, session):
        self.seen.append(session["id"])
        return {"imported": 1, "skipped": 0, "imported_ids": [session["id"]]}


def test_receive_merges_without_overwriting_and_undo_restores(tmp_path):
    out, _inv = _bundle(tmp_path, include_history=False)
    home = _target_home(tmp_path)
    before_config = (home / "config.yaml").read_text()
    state = _upload(home, out.read_bytes())
    assert state["status"] == "ready"
    preview = state["preview"]
    assert preview["skills"] == {"add": [], "keep_existing": ["custom/mine"],
                                 "left_behind": preview["skills"]["left_behind"]}
    assert [j["id"] for j in preview["cron"]["add"]] == ["job1", "job2"]

    state = move_receive.apply(home, state["id"], import_session=_Sessions())
    assert state["status"] == "applied", state
    config = yaml.safe_load((home / "config.yaml").read_text())
    assert config["model"] == {"default": "nous/hermes", "provider": "nous"}
    assert config["display"]["skin"] == "default"
    assert "mcp_servers" in config
    memory = (home / "memories" / "MEMORY.md").read_text().split("\n§\n")
    assert memory == ["cloud note", "prefers tabs", "works on hermes"]
    assert (home / "memories" / "USER.md").read_text() == "name is Sam"
    assert "the cloud copy" in (home / "skills" / "custom" / "mine" / "SKILL.md").read_text()
    assert (home / "SOUL.md").read_text() == "You are terse and kind.\n"
    jobs = {j["id"]: j for j in json.loads((home / "cron" / "jobs.json").read_text())["jobs"]}
    assert set(jobs) == {"job1", "job2"}
    assert all(j["enabled"] is False and j["state"] == "paused" for j in jobs.values())
    assert "script and folder paths" in jobs["job2"]["paused_reason"]
    assert "last_status" not in jobs["job1"]

    state = move_receive.undo(home, state["id"])
    assert state["status"] == "undone"
    assert (home / "config.yaml").read_text() == before_config
    assert (home / "memories" / "MEMORY.md").read_text() == "cloud note\n§\nprefers tabs\n"
    assert not (home / "SOUL.md").exists() and not (home / "memories" / "USER.md").exists()
    assert json.loads((home / "cron" / "jobs.json").read_text())["jobs"] == []


def test_customized_soul_on_target_is_kept(tmp_path):
    out, _ = _bundle(tmp_path, include_history=False)
    home = _target_home(tmp_path)
    (home / "SOUL.md").write_text("I am the cloud persona.\n")
    state = _upload(home, out.read_bytes())
    assert state["preview"]["soul"]["action"] == "keep_existing"
    move_receive.apply(home, state["id"], import_session=_Sessions())
    assert (home / "SOUL.md").read_text() == "I am the cloud persona.\n"


def test_new_skill_lands_and_undo_removes_it(tmp_path):
    out, _ = _bundle(tmp_path, include_history=False)
    home = _target_home(tmp_path)
    import shutil
    shutil.rmtree(home / "skills" / "custom")
    state = _upload(home, out.read_bytes())
    assert state["preview"]["skills"]["add"] == ["custom/mine"]
    state = move_receive.apply(home, state["id"], import_session=_Sessions())
    assert (home / "skills" / "custom" / "mine" / "SKILL.md").is_file()
    assert not list((home / "skills").glob(".move-*"))
    move_receive.undo(home, state["id"])
    assert not (home / "skills" / "custom" / "mine").exists()


def test_upload_resumes_replays_and_refuses_gaps(tmp_path):
    out, _ = _bundle(tmp_path, include_history=False)
    data = out.read_bytes()
    home = _target_home(tmp_path)
    digest = hashlib.sha256(data).hexdigest()
    state = move_receive.begin(home, size=len(data), sha256=digest)
    first = base64.b64encode(data[:100]).decode()
    move_receive.chunk(home, state["id"], offset=0, data_b64=first)
    assert move_receive.chunk(home, state["id"], offset=0, data_b64=first)["received"] == 100
    with pytest.raises(move_receive.MoveError) as gap:
        move_receive.chunk(home, state["id"], offset=500, data_b64=first)
    assert gap.value.code == "offset_gap"
    resumed = move_receive.begin(home, size=len(data), sha256=digest)
    assert resumed["id"] == state["id"] and resumed["received"] == 100
    move_receive.chunk(home, state["id"], offset=100, data_b64=base64.b64encode(data[100:]).decode())
    assert move_receive.finish(home, state["id"])["status"] == "ready"


def test_damaged_upload_is_refused_and_reset(tmp_path):
    out, _ = _bundle(tmp_path, include_history=False)
    data = bytearray(out.read_bytes())
    home = _target_home(tmp_path)
    state = move_receive.begin(home, size=len(data), sha256=hashlib.sha256(data).hexdigest())
    data[50] ^= 0xFF
    move_receive.chunk(home, state["id"], offset=0, data_b64=base64.b64encode(bytes(data)).decode())
    with pytest.raises(move_receive.MoveError) as err:
        move_receive.finish(home, state["id"])
    assert err.value.code == "digest_mismatch"
    assert move_receive.get_move(home, state["id"])["received"] == 0


def test_newer_bundle_format_is_refused_before_anything_changes(tmp_path):
    out = tmp_path / "future.zip"
    with zipfile.ZipFile(out, "w") as zf:
        zf.writestr(MANIFEST_NAME, json.dumps({"kind": "hermes-move-bundle", "schema_version": 99}))
    home = _target_home(tmp_path)
    with pytest.raises(move_receive.MoveError) as err:
        _upload(home, out.read_bytes())
    assert "newer Hermes" in str(err.value)


def test_a_failed_step_is_reported_and_the_rest_still_apply(tmp_path):
    out, _ = _bundle(tmp_path, include_history=False)
    home = _target_home(tmp_path)
    state = _upload(home, out.read_bytes())
    (home / "memories" / "MEMORY.md").write_bytes(b"\xff\xfe not utf8")
    state = move_receive.apply(home, state["id"], import_session=_Sessions())
    assert state["status"] == "partial"
    assert state["steps"]["memory"]["status"] == "failed"
    assert state["steps"]["cron"]["status"] == "done"
    again = move_receive.apply(home, state["id"], import_session=_Sessions())
    assert again["steps"]["cron"]["status"] == "done"
    (home / "memories" / "MEMORY.md").write_text("fixed\n")
    healed = move_receive.apply(home, state["id"], import_session=_Sessions())
    assert healed["status"] == "applied"
    assert "works on hermes" in (home / "memories" / "MEMORY.md").read_text()
    jobs = json.loads((home / "cron" / "jobs.json").read_text())["jobs"]
    assert len(jobs) == 2, "a resumed apply must not re-add the jobs the first pass already added"


def test_history_is_exported_and_each_session_is_imported_once(tmp_path):
    home = _source_home(tmp_path)
    from hermes_state import SessionDB
    db = SessionDB(db_path=home / "state.db")
    try:
        db.import_sessions([{"id": f"s{i}", "source": "desktop", "title": f"chat {i}",
                             "messages": [{"role": "user", "content": f"hello {i}"},
                                          {"role": "assistant", "content": "hi"}]} for i in range(3)]
                           + [{"id": "sub", "source": "subagent", "messages": [{"role": "user", "content": "x"}]}])
    finally:
        db.close()
    out = tmp_path / "move.zip"
    report = build_bundle(home, "default", out, include_history=True)
    assert report.sessions == 3
    target = _target_home(tmp_path)
    state = _upload(target, out.read_bytes())
    importer = _Sessions()
    state = move_receive.apply(target, state["id"], import_session=importer)
    assert sorted(importer.seen) == ["s0", "s1", "s2"]
    assert state["steps"]["sessions"]["imported"] == 3


def test_routes_round_trip_on_the_dashboard_app(tmp_path, monkeypatch, _isolate_hermes_home):
    from starlette.testclient import TestClient
    from hermes_constants import get_hermes_home
    from hermes_cli.web_server import _SESSION_HEADER_NAME, _SESSION_TOKEN, app

    home = get_hermes_home()
    src = _source_home(tmp_path)
    for name in ("config.yaml", "SOUL.md", ".env"):
        (home / name).write_text((src / name).read_text())
    import shutil
    shutil.copytree(src / "memories", home / "memories", dirs_exist_ok=True)
    shutil.copytree(src / "skills", home / "skills", dirs_exist_ok=True)
    client = TestClient(app)
    client.headers[_SESSION_HEADER_NAME] = _SESSION_TOKEN

    built = client.post("/api/move/bundle", json={"include_history": False})
    assert built.status_code == 200, built.text
    name = built.json()["name"]
    chunks, offset = [], 0
    while True:
        part = client.get(f"/api/move/bundle/{name}", params={"offset": offset, "length": 4096}).json()
        chunks.append(base64.b64decode(part["data"]))
        offset += len(chunks[-1])
        if part["done"]:
            break
    data = b"".join(chunks)
    assert data[:2] == b"PK"

    # Receive into the same home: everything already exists, so nothing may be overwritten.
    (home / "SOUL.md").write_text("cloud persona\n")
    begun = client.post("/api/move/uploads", json={"size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    assert begun.status_code == 200, begun.text
    move_id = begun.json()["id"]
    for off in range(0, len(data), 8192):
        r = client.post(f"/api/move/uploads/{move_id}/chunk",
                        json={"offset": off, "data": base64.b64encode(data[off:off + 8192]).decode()})
        assert r.status_code == 200, r.text
    finished = client.post(f"/api/move/uploads/{move_id}/finish")
    assert finished.status_code == 200 and finished.json()["preview"]["soul"]["action"] == "keep_existing"
    applied = client.post(f"/api/move/uploads/{move_id}/apply", json={"include": ["config", "soul", "memory"]})
    assert applied.status_code == 200 and applied.json()["status"] == "applied", applied.text
    assert (home / "SOUL.md").read_text() == "cloud persona\n"
    assert client.get("/api/move/uploads").json()["moves"][0]["id"] == move_id
    bad = client.post("/api/move/uploads/" + "0" * 32 + "/apply", json={})
    assert bad.status_code == 404 and bad.json()["detail"]["code"] == "not_found"


def test_a_title_already_used_on_the_target_gets_a_suffix(tmp_path):
    from hermes_state import SessionDB
    home = _source_home(tmp_path)
    db = SessionDB(db_path=home / "state.db")
    db.import_sessions([{"id": "src1", "source": "desktop", "title": "Taxes",
                         "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}])
    db.close()
    out = tmp_path / "move.zip"
    build_bundle(home, "default", out, include_history=True)
    target = _target_home(tmp_path)
    tdb = SessionDB(db_path=target / "state.db")
    tdb.import_sessions([{"id": "other", "source": "cli", "title": "Taxes", "messages": [{"role": "user", "content": "x"}]}])
    tdb.close()

    def real_import(session):
        db = SessionDB(db_path=target / "state.db")
        try:
            return db.import_sessions([session])
        finally:
            db.close()

    state = _upload(target, out.read_bytes())
    state = move_receive.apply(target, state["id"], import_session=real_import)
    assert state["steps"]["sessions"] == {"status": "done", "imported": 1, "already_here": 0, "failed": 0, "errors": []}
    db = SessionDB(db_path=target / "state.db", read_only=True)
    try:
        assert db.get_session("src1")["title"] == "Taxes #2"
    finally:
        db.close()


def test_secret_values_quoted_in_text_are_redacted(tmp_path):
    home = _source_home(tmp_path)
    skill = home / "skills" / "custom" / "mine" / "notes.md"
    (home / ".env").write_text("OPENROUTER_API_KEY=sk-or-real-long\nDISCORD_BOT_TOKEN=123:abc-long-enough\n")
    skill.write_text("my key is sk-or-real-long and the bot is 123:abc-long-enough\n")
    out = tmp_path / "move.zip"
    report = build_bundle(home, "default", out, include_history=False)
    with zipfile.ZipFile(out) as zf:
        text = zf.read("skills/custom/mine/notes.md").decode()
    assert "sk-or-real-long" not in text and "123:abc-long-enough" not in text
    assert "[redacted OPENROUTER_API_KEY]" in text
    assert report.redacted == {"DISCORD_BOT_TOKEN": 1, "OPENROUTER_API_KEY": 1}
