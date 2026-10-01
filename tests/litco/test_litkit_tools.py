"""The ``litkit`` toolset against a fake LitKit: cursor loops, NDJSON export to files, gated
deliverables, permission surfacing, turn identity, registration, and the working-dir boundary."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from litco.litkit import tools as T
from litco.litkit.client import LitKitClient, LitKitConfig, set_default_client
from litco.litkit.context import TurnIdentity, bind_turn, current_turn, reset_turn
from tests.litco._litkit_fake import HOST_SECRET, MATTER_ID, TOKEN, USER_ID, FakeLitKit, FakeReview, ndjson

M = MATTER_ID


def _id(n: int) -> str:
    return f"00000000-0000-4000-8000-{n:012d}"


@pytest.fixture
def fake():
    server = FakeLitKit().start()
    yield server
    server.stop()


@pytest.fixture
def env(fake, tmp_path, monkeypatch):
    """A matter home under tmp_path, a pinned client, and a turn acting for USER_ID in shared/."""
    home = tmp_path / "matter"
    cwd = home / "shared"
    (cwd / "deliverables").mkdir(parents=True)
    monkeypatch.setenv("LITCO_MATTER_HOME", str(home))
    client = LitKitClient(LitKitConfig(instance_url=fake.url, token=TOKEN, host_secret=HOST_SECRET, matter_id=M),
                          sleep=lambda s: None)
    set_default_client(client)
    token = bind_turn(TurnIdentity(turn_id="turn_1", matter_id=M, acting_user=USER_ID, cwd=cwd))
    yield {"home": home, "cwd": cwd, "tmp": tmp_path, "client": client}
    reset_turn(token)
    set_default_client(None)
    client.close()


def call(name: str, /, **args):
    out = T.HANDLERS[name](args)
    try:
        return json.loads(out)
    except ValueError:
        return out


def _files_outside(root: Path, allowed: Path) -> list:
    return sorted(str(p) for p in root.rglob("*") if p.is_file() and not str(p).startswith(str(allowed) + os.sep))


# ---------------------------------------------------------------------------


def test_every_call_asserts_the_turn_user(fake, env):
    fake.route("GET", rf"/api/matters/{M}", {"id": M, "name": "Test Matter", "openaiKeyAvailable": True})
    fake.route("GET", rf"/api/matters/{M}/docs", {"total": 42, "docs": []})
    fake.route("GET", rf"/api/matters/{M}/search/facets",
               {"custodians": ["Doe, Jane", "Roe, Rick"], "productions": [{"id": "p1", "name": "Vol 1", "fileCount": 42}],
                "batesPrefixes": ["ABC"]})
    out = call("litkit_matter")
    assert out["documentCount"] == 42 and out["custodians"] == ["Doe, Jane", "Roe, Rick"]
    assert out["matter"]["name"] == "Test Matter" and "openaiKeyAvailable" not in out["matter"]
    assert len(fake.requests) == 3
    for req in fake.requests:
        assert req.headers["authorization"] == f"Bearer {TOKEN}"
        assert req.headers["x-litkit-acting-user"] == USER_ID
        assert req.headers["x-litkit-user-assertion"].split(".")[1:3] == [USER_ID, M]


def test_cron_work_sends_no_assertion(fake, env):
    token = bind_turn(TurnIdentity(acting_user=None, cwd=env["cwd"]))
    try:
        fake.route("GET", rf"/api/matters/{M}/search", {"hits": []})
        call("litkit_search", query='"price"')
    finally:
        reset_turn(token)
    assert "x-litkit-user-assertion" not in fake.requests[-1].headers


def test_search_limits_and_timeout_hint(fake, env):
    fake.route("GET", rf"/api/matters/{M}/search",
               lambda r: (200, {"hits": [{"docId": _id(i % 3), "bates": f"ABC{i:05d}", "page": 1, "snippet": "x" * 400}
                                         for i in range(int(r.query["limit"][0]))], "searchRanked": True}))
    out = call("litkit_search", query='"average selling price"', limit=5, custodian="Doe, Jane")
    assert out["hits"] == 5 and out["documents"] == 3 and len(out["notes"]) == 2
    assert len(out["results"][0]["snippet"]) <= 280
    assert fake.requests[-1].query["custodian"] == ["Doe, Jane"]
    fake.route("GET", rf"/api/matters/{M}/search", (504, {"error": "search timed out; narrow the query"}))
    out = call("litkit_search", query="the")
    assert out["status"] == 504 and "quoted phrase" in out["error"] and "not evidence" in out["error"]


def test_docs_cursor_loop_saves_complete_census(fake, env):
    pages = {"": ("c1", [1, 2]), "c1": ("c2", [3, 4]), "c2": (None, [5])}

    def docs(req):
        cur = req.query["cursor"][0]
        nxt, ids = pages[cur]
        return 200, {"total": 5 if cur == "" else None, "hasMore": nxt is not None, "nextCursor": nxt,
                     "docs": [{"id": _id(i), "batesStart": f"ABC{i:05d}", "custodian": "Doe, Jane",
                               "documentDate": "2021-01-0%d" % i} for i in ids]}

    fake.route("GET", rf"/api/matters/{M}/docs", docs)
    out = call("litkit_docs", custodian="Doe, Jane", saveAs="doe", limit=2)
    assert out["rows"] == 5 and out["total"] == 5 and out["pages"] == 3 and out["complete"] is True
    assert [r.query["cursor"][0] for r in fake.requests] == ["", "c1", "c2"]
    assert all(r.query["custodian"] == ["Doe, Jane"] and r.query["limit"] == ["2"] for r in fake.requests)
    lines = (env["cwd"] / "census" / "doe.jsonl").read_text().splitlines()
    assert [json.loads(line)["id"] for line in lines] == [_id(i) for i in range(1, 6)]
    # single page mode returns the page and the cursor for the next one
    page = call("litkit_docs", cursor="c1", limit=2)
    assert page["nextCursor"] == "c2" and page["returned"] == 2 and page["hasMore"] is True


def test_docs_overbroad_cursor_is_explained(fake, env):
    fake.route("GET", rf"/api/matters/{M}/docs", (422, {"error": "too broad", "errorKind": "cursor_overbroad"}))
    out = call("litkit_docs", q="the", saveAs="x")
    assert "too broad to page" in out["error"]


def test_export_text_batches_ndjson_into_texts_and_resumes(fake, env):
    ids = [_id(i) for i in range(1, 1203)]
    missing = {_id(7)}

    def export(req):
        batch = req.json()["docIds"]
        assert 1 <= len(batch) <= 500
        rows = []
        for d in batch:
            if d in missing:
                rows.append({"docId": d, "error": "not_found"})
            else:
                n = int(d[-12:])
                row = {"docId": d, "batesStart": f"ABC{n:07d}", "batesEnd": f"ABC{n:07d}", "custodian": "Doe, Jane",
                       "date": "2021-02-03T00:00:00.000Z", "text": f"body of {n}"}
                if n == 9:
                    row.update(truncated=True, fullLength=250000)
                rows.append(row)
        return ndjson(rows)

    fake.route("POST", rf"/api/matters/{M}/export/text", export)
    census = env["cwd"] / "census" / "doe.jsonl"
    census.parent.mkdir(parents=True)
    census.write_text("\n".join(json.dumps({"id": i}) for i in ids[:1000]) + "\n")
    out = call("litkit_export_text", fromCensus="census/doe.jsonl", documentIds=ids[1000:])
    assert out["batches"] == 3 and out["written"] == 1201 and out["notFound"] == 1 and out["truncated"] == 1
    assert [len(r.json()["docIds"]) for r in fake.calls("POST", rf"/api/matters/{M}/export/text")] == [500, 500, 202]
    text = (env["cwd"] / "texts" / "ABC0000001.txt").read_text()
    head, body = text.split("=" * 60 + "\n")
    assert "Bates: ABC0000001" in head and f"docId: {_id(1)}" in head and "Custodian: Doe, Jane" in head
    assert body == "body of 1"
    assert "Truncated: served" in (env["cwd"] / "texts" / "ABC0000009.txt").read_text()
    index = json.loads((env["cwd"] / "texts" / "index.json").read_text())
    assert index[_id(1)]["file"] == "ABC0000001.txt" and _id(7) not in index
    # second run: everything already on disk is skipped; only the missing one is asked for again
    out2 = call("litkit_export_text", documentIds=ids)
    assert out2["skippedExisting"] == 1201 and out2["batches"] == 1
    assert fake.calls("POST", rf"/api/matters/{M}/export/text")[-1].json()["docIds"] == [_id(7)]


def test_text_saves_self_citing_file(fake, env):
    d = _id(5)
    fake.route("GET", rf"/api/documents/{d}", {"doc": {"id": d, "batesStart": "ABC0005", "batesEnd": "ABC0007",
                                                       "custodian": "Roe, Rick", "documentDate": "2020-05-05",
                                                       "subject": "Q3 pricing", "extractedText": "SHOULD NOT LEAK"}})
    fake.route("GET", rf"/api/documents/{d}/text",
               {"chunks": [{"ordinal": 0, "pageStart": 1, "text": "Page one."},
                           {"ordinal": 1, "pageStart": 2, "text": "Page two."}]})
    out = call("litkit_text", documentId=d)
    assert out["saved"] == "texts/ABC0005.txt" and out["chars"] > 0
    saved = (env["cwd"] / "texts" / "ABC0005.txt").read_text()
    assert saved.startswith("Bates: ABC0005 - ABC0007\n") and "Subject: Q3 pricing" in saved
    assert "[page 2]\n\nPage two." in saved


def test_deliver_multipart_blocked_gate_is_reported_not_retried(fake, env):
    draft = env["cwd"] / "deliverables" / "letter.docx"
    draft.write_bytes(b"PK fake docx")
    gate = {"blockedBy": "quote", "blockedReason": "a quotation failed live re-resolve",
            "quote": {"checked": 4, "verifiedTokens": 3, "unverified": ["\"we never priced\""], "hardFail": []},
            "citationFlags": [{"cite": "123 F.4th 1", "flag": "unverified"}], "proseLintFlags": []}
    fake.route("POST", rf"/api/matters/{M}/deliverables", (422, {"blocked": True, "gate": gate, "validity": {"ok": True}}))
    out = call("litkit_deliver", path="deliverables/letter.docx", deliverableClass="pleading",
               provenance={"sources": ["ABC0005"]})
    assert out["blocked"] is True
    assert out["gate"]["blockedBy"] == "quote" and out["gate"]["quotesUnverified"] == ["\"we never priced\""]
    assert "Report these gate findings to the user" in out["instruction"]
    posts = fake.calls("POST", rf"/api/matters/{M}/deliverables")
    assert len(posts) == 1
    form = posts[0].form()
    assert form["deliverableClass"] == "pleading" and form["file"] == ("letter.docx", b"PK fake docx")
    assert json.loads(form["provenance"]) == {"sources": ["ABC0005"]}
    assert posts[0].headers["x-litkit-acting-user"] == USER_ID
    saved = env["cwd"] / out["fullFindings"]
    assert saved.is_file() and json.loads(saved.read_text())["status"] == 422
    assert not str(out["fullFindings"]).startswith("deliverables")  # findings never go back to the thread


def test_deliver_success_and_new_version(fake, env):
    draft = env["cwd"] / "deliverables" / "memo.pdf"
    draft.write_bytes(b"%PDF-1.7")
    doc = _id(77)
    fake.route("POST", rf"/api/matters/{M}/deliverables",
               (201, {"blocked": False, "documentId": doc, "versionId": _id(78), "versionNumber": 2,
                      "path": "Work Product/Reports/memo.pdf", "store": "litspace", "sha256": "ab", "sizeBytes": 8,
                      "gate": {"quote": {"checked": 0}}}))
    out = call("litkit_deliver", path=str(draft), deliverableClass="memo", documentId=doc, note="v2")
    assert out["blocked"] is False and out["versionNumber"] == 2
    form = fake.requests[-1].form()
    assert form["documentId"] == doc and form["note"] == "v2"


def test_validity_failure(fake, env):
    (env["cwd"] / "bad.docx").write_bytes(b"not a zip")
    fake.route("POST", rf"/api/matters/{M}/deliverables", (422, {"error": "validity_gate_failed", "detail": "bad zip"}))
    out = call("litkit_deliver", path="bad.docx", deliverableClass="draft")
    assert out["blocked"] is True and out["reason"] == "validity_gate_failed"


@pytest.mark.parametrize("tool,args,route", [
    ("litkit_review", {"action": "resume", "jobId": "job1"}, rf"/api/matters/{M}/review-jobs/job1/resume"),
    ("litkit_ingest", {"action": "retry", "productionId": "p1"}, rf"/api/matters/{M}/productions/p1/exceptions/retry"),
    ("litkit_ingest", {"action": "reingest", "productionId": "p1"}, rf"/api/matters/{M}/productions/p1/reingest"),
])
def test_admin_passthrough_returns_permission_error_plainly(fake, env, tool, args, route):
    fake.route("POST", route, (403, {"error": "forbidden"}))
    out = call(tool, **args)
    assert out["permission_denied"] is True and out["status"] == 403
    assert out["error"].startswith("not permitted for this user on this matter")
    assert len(fake.calls("POST", route)) == 1


def test_notify_defaults_to_the_acting_user(fake, env):
    fake.route("POST", r"/api/notifications/emit", {"ok": True, "id": "n1"})
    call("litkit_notify", title="Binder ready", link="/matters/x")
    assert fake.requests[-1].json() == {"kind": "agent_notify", "title": "Binder ready", "matterId": M,
                                        "userId": USER_ID, "link": "/matters/x"}
    call("litkit_notify", title="Heads up", matterWide=True)
    assert "userId" not in fake.requests[-1].json()


def test_private_memory_needs_a_lawyer(fake, env):
    fake.route("POST", r"/api/agent/actions", lambda r: (200, {"ok": True, "echo": r.json()}))
    out = call("litkit_remember", content="prefers short memos", scope="user", kind="strategy")
    assert out["echo"] == {"action": "remember", "matterId": M,
                           "args": {"kind": "strategy", "content": "prefers short memos", "acl": {"scope": "user"}}}
    token = bind_turn(TurnIdentity(acting_user=None, cwd=env["cwd"]))
    try:
        out = call("litkit_remember", content="x", scope="user")
    finally:
        reset_turn(token)
    assert "needs a lawyer" in out["error"]


def test_actions_passthrough_and_rejection(fake, env):
    fake.route("POST", r"/api/agent/actions",
               lambda r: (400, {"ok": False, "error": "query required"}) if not r.json()["args"] else
               (200, {"ok": True, "summary": "3 terms"}))
    assert call("litkit_actions", action="term_frequency", args={"query": "x"})["summary"] == "3 terms"
    assert call("litkit_actions", action="term_frequency")["error"] == "query required"
    assert "must be one of" in call("litkit_actions", action="repair")["error"]


def test_large_results_spill_to_a_file(fake, env):
    fake.route("GET", rf"/api/matters/{M}/review-jobs", {"jobs": [{"id": f"j{i}", "note": "y" * 200} for i in range(200)]})
    out = T.HANDLERS["litkit_review"]({"action": "list"})
    assert out.startswith("<persisted-output>") and "Full output saved to: " in out
    path = Path(out.split("Full output saved to: ")[1].splitlines()[0])
    assert path.is_file() and str(path).startswith(str(env["cwd"]))


def test_no_tool_writes_outside_the_working_dir(fake, env):
    evil = "../../../evil"
    d = _id(3)
    fake.route("GET", rf"/api/documents/{d}", {"doc": {"batesStart": evil, "fileName": "../../x.xlsx"}})
    fake.route("GET", rf"/api/documents/{d}/text", {"extractedText": "t"})
    fake.route("GET", rf"/api/documents/{d}/pdf", (200, b"%PDF-1.4"))
    fake.route("GET", rf"/api/documents/{d}/native", (200, b"PK"))
    fake.route("POST", rf"/api/matters/{M}/export/text",
               lambda r: ndjson([{"docId": x, "batesStart": "../../../../etc/passwd", "text": "t"} for x in r.json()["docIds"]]))
    fake.route("GET", rf"/api/matters/{M}/chat/attachments/.*",
               (200, b"bytes", {"Content-Disposition": 'attachment; filename="../../../../boom.txt"'}))
    fake.route("GET", r"/api/litspace/files/.*/content", (200, b"file"))
    fake.route("GET", r"/api/litspace/files/[^/]+", {"filename": "../../../../x.docx"})
    fake.route("GET", r"/api/litlex/opinions/.*", {"opinion": {"id": "o"}})
    fake.route("GET", rf"/api/matters/{M}/docs", {"total": 1, "hasMore": False, "nextCursor": None, "docs": []})

    call("litkit_text", documentId=d, dir="../../outside")
    call("litkit_pdf", documentId=d, dir="/etc")
    call("litkit_pdf", documentId=d, native=True)
    call("litkit_export_text", documentIds=[d], dir="../..")
    call("litkit_attachment", fileId=_id(4))
    call("litkit_files", action="read", fileId=_id(5), dir="../../..")
    call("litkit_litlex", action="opinion", opinionId="../../o")
    call("litkit_docs", saveAs="../../census")
    assert _files_outside(env["tmp"], env["cwd"]) == []
    # input paths are fenced too: nothing outside the matter's directories can be uploaded
    outside = env["tmp"] / "secret.txt"
    outside.write_text("private")
    for path in (str(outside), "../../secret.txt", "/etc/hosts"):
        out = call("litkit_deliver", path=path, deliverableClass="memo")
        assert "outside" in out["error"]
    assert not fake.calls("POST", rf"/api/matters/{M}/deliverables")


def test_runner_binds_the_verified_acting_user_for_tools(tmp_path, monkeypatch):
    from litco import hermes_runner
    from litco.hermes_runner import HermesTurnRunner
    from litco.turn_server import TurnContext, TurnRequest
    from tools.thread_context import propagate_context_to_thread
    import threading

    runner = HermesTurnRunner()
    runner._session_map = hermes_runner._SessionMap(tmp_path / "sessions.json")
    monkeypatch.setattr(runner, "_session_db", lambda: None)
    seen = {}

    class Agent:
        session_id = "x"
        model = "fake"

        def interrupt(self, **kw):
            pass

        def run_conversation(self, user_message, conversation_history, task_id):
            seen["main"] = current_turn()
            t = threading.Thread(target=propagate_context_to_thread(lambda: seen.setdefault("worker", current_turn())))
            t.start()
            t.join()
            return {"final_response": "ok"}

    monkeypatch.setattr(runner, "_build_agent", lambda ctx, sid, mapper: Agent())
    cwd = tmp_path / "users" / "u1"
    (cwd / "deliverables").mkdir(parents=True)
    req = TurnRequest(matter_id=M, user_id=USER_ID, session_id="s", text="hi", attachments=[], channel="slack",
                      kind="dm", acting_user=USER_ID)
    ctx = TurnContext(turn_id="turn_9", request=req, home=tmp_path, cwd=cwd, emit=lambda t, f: None)
    assert runner.run(ctx).text == "ok"
    for key in ("main", "worker"):
        assert seen[key].acting_user == USER_ID and seen[key].cwd == cwd and seen[key].turn_id == "turn_9"
    assert current_turn() is None
    # an unverified userId is not asserted
    req2 = TurnRequest(matter_id=M, user_id=USER_ID, session_id="s2", text="hi", attachments=[], channel="slack",
                       kind="channel")
    ctx2 = TurnContext(turn_id="turn_10", request=req2, home=tmp_path, cwd=cwd, emit=lambda t, f: None)
    runner.run(ctx2)
    assert seen["main"].acting_user is None


def test_toolset_registers_through_plugin_discovery(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    monkeypatch.setenv("LITCO_INSTANCE_URL", "http://127.0.0.1:9")
    monkeypatch.setenv("LITCO_AGENT_TOKEN", TOKEN)
    from hermes_cli.plugins import discover_plugins
    discover_plugins(force=True)
    from tools.registry import registry
    for name in T.SCHEMAS:
        entry = registry.get_entry(name)
        assert entry is not None and entry.toolset == "litkit", name
    assert T.check_available() is True
    monkeypatch.delenv("LITCO_AGENT_TOKEN")
    assert T.check_available() is False


def test_schemas_are_well_formed():
    from hermes_yaml import safe_load
    manifest = safe_load((Path(__file__).parents[2] / "plugins" / "litkit" / "plugin.yaml").read_text())
    assert sorted(name for name, _s, _h in T.TOOLS) == sorted(manifest["provides_tools"])
    for name, schema, _handler in T.TOOLS:
        assert schema["name"] == name and schema["parameters"]["type"] == "object"
        assert set(schema["parameters"]["required"]) <= set(schema["parameters"]["properties"])
        assert len(schema["description"]) < 400


def test_deliver_before_the_file_exists(env, fake):
    before = len(fake.requests)
    out = call("litkit_deliver", path="deliverables/memo.docx", deliverableClass="memo")
    assert out["file_missing"] is True
    assert "does not exist" in out["error"] and "Write the file first" in out["error"]
    assert "litkit_deliver again" in out["error"]
    assert len(fake.requests) == before  # nothing reached LitKit
    folder = call("litkit_quote_check", path="deliverables")
    assert "is a folder" in folder["error"]


def test_native_download_keeps_the_extension_dot(fake, env):
    d = _id(7)
    fake.route("GET", rf"/api/documents/{d}", {"doc": {"batesStart": "ABC0001", "fileName": "budget.XLSX"}})
    fake.route("GET", rf"/api/documents/{d}/native", (200, b"PK"))
    out = call("litkit_pdf", documentId=d, native=True)
    assert out["saved"].endswith("natives/ABC0001.XLSX"), out


def test_tag_apply_rejects_non_uuid_document_ids_before_any_request(fake, env):
    for docs in (["../../api/admin"], [_id(1), "not-an-id"]):
        out = call("litkit_tags", action="apply", tagId=_id(2), documentIds=docs)
        assert "uuid" in out["error"], out
    assert not [r for r in fake.requests if "/tags" in r.path or "bulk-tag" in r.path]


def _history(n: int = 3):
    jane, raj = _id(101), _id(102)
    msgs = [{"id": _id(200 + i), "threadId": _id(300 + i % 2), "threadTitle": "t", "seq": i,
             "role": "user", "authorUserId": jane if i % 2 else raj, "text": f"message {i}",
             "origin": "web", "externalRef": None, "mentionsAgent": False, "mentionUserIds": [],
             "createdAt": f"2026-09-29T10:0{i}:00.000Z"} for i in range(n)]
    msgs.append({"id": _id(299), "threadId": _id(300), "seq": 9, "role": "assistant", "authorUserId": None,
                 "text": "Here is the summary.", "createdAt": "2026-09-29T09:00:00.000Z"})
    msgs.append({"id": _id(298), "threadId": _id(301), "seq": 1, "role": "user", "authorUserId": None,
                 "externalRef": {"name": "Pat Slack"}, "text": "from slack", "createdAt": "2026-09-29T08:00:00Z"})
    return {"channel": {"id": "c1", "slug": "depo-prep", "name": "depo prep", "topic": "Smith depo"},
            "messages": msgs, "nextBefore": "2026-09-29T08:00:00Z",
            "people": [{"id": jane, "name": "Jane Doe", "email": "jane@firm.test"},
                       {"id": raj, "name": None, "email": "raj@firm.test"}]}


def test_channel_history_defaults_to_the_turns_channel_and_asserts_the_user(fake, env):
    fake.route("GET", rf"/api/matters/{M}/channels/depo-prep/history", _history())
    token = bind_turn(TurnIdentity(turn_id="turn_1", matter_id=M, acting_user=USER_ID, cwd=env["cwd"],
                                   litkit_channel="depo-prep"))
    try:
        out = call("litkit_channel_history")
    finally:
        reset_turn(token)
    req = fake.requests[-1]
    assert req.path == f"/api/matters/{M}/channels/depo-prep/history"
    assert req.query == {"limit": ["50"]}  # no before: not sent
    assert req.headers["authorization"] == f"Bearer {TOKEN}"
    assert req.headers["x-litkit-acting-user"] == USER_ID
    assert req.headers["x-litkit-user-assertion"].split(".")[1:3] == [USER_ID, M]
    assert out["channel"] == {"slug": "depo-prep", "name": "depo prep", "topic": "Smith depo"}
    assert out["nextBefore"] == "2026-09-29T08:00:00Z" and out["messages"] == 5
    assert out["rows"][0] == {"author": "raj@firm.test", "at": "2026-09-29T10:00:00.000Z",
                              "threadId": _id(300), "text": "message 0"}
    assert out["rows"][1]["author"] == "Jane Doe"
    assert [r["author"] for r in out["rows"][3:]] == ["Ana", "Pat Slack (Slack)"]
    assert all(set(r) == {"author", "at", "threadId", "text"} for r in out["rows"])


def test_channel_history_caps_limit_and_passes_before(fake, env):
    fake.route("GET", rf"/api/matters/{M}/channels/[^/]+/history", _history(1))
    call("litkit_channel_history", channel="#depo-prep", limit=500, before="2026-09-29T08:00:00Z")
    req = fake.requests[-1]
    assert req.path == f"/api/matters/{M}/channels/depo-prep/history"
    assert req.query == {"limit": ["100"], "before": ["2026-09-29T08:00:00Z"]}
    call("litkit_channel_history", channel="a/b", limit=-5)
    assert fake.requests[-1].path == f"/api/matters/{M}/channels/a%2Fb/history"
    assert fake.requests[-1].query["limit"] == ["1"]


def test_channel_history_needs_a_channel_and_a_valid_before(fake, env):
    before = len(fake.requests)
    assert "not arrive in a LitKit channel" in call("litkit_channel_history")["error"]
    assert "ISO time" in call("litkit_channel_history", channel="depo-prep", before="yesterday")["error"]
    assert len(fake.requests) == before  # nothing reached LitKit
    fake.route("GET", rf"/api/matters/{M}/channels/nope/history", (404, {"error": "channel_not_found"}))
    out = call("litkit_channel_history", channel="nope")
    assert out["status"] == 404 and "error" in out


# ---------------------------------------------------------------------------
# review & tag: Ana registers the structure and proposes the run herself


CRITERIA = [{"title": "Pricing communications", "description": "Any discussion of list or net price.",
             "tagName": "Pricing"},
            "Board materials"]


def test_review_structure_is_registered_then_proposed(fake, env):
    review = FakeReview(fake)
    made = call("litkit_review", action="criteria", criteriaAction="create", name="Part 11 requests",
                criteria=CRITERIA, description="RFP set 2")
    set_id = made["set"]["id"]
    sent = fake.calls("POST", rf"/api/matters/{M}/criteria-sets")[-1].json()
    assert sent == {"name": "Part 11 requests", "description": "RFP set 2",
                    "criteria": [CRITERIA[0], {"title": "Board materials"}]}
    assert call("litkit_review", action="criteria")["mine"][0]["id"] == set_id
    assert call("litkit_review", action="criteria", criteriaAction="get", criteriaSetId=set_id)["set"]["id"] == set_id
    work_sets = call("litkit_review", action="work_sets")["workSets"]
    out = call("litkit_review", action="create", criteriaSetId=set_id, scope={"workSetId": work_sets[0]["id"]},
               tags=["Pricing", "Board", "Pricing"], createMissingTags=True, name="Part 11 first pass")
    assert out["proposalId"] and out["estimatedCount"] == 1234 and out["criteriaSetVersion"] == 1
    assert "card in this thread" in out["next"] and "Review screen" in out["next"]
    assert review.proposals[0]["request"] == {"scope": {"workSetId": work_sets[0]["id"]}, "criteriaSetId": set_id,
                                              "tags": ["Pricing", "Board"], "createMissingTags": True,
                                              "name": "Part 11 first pass"}
    req = fake.calls("POST", rf"/api/matters/{M}/review-jobs/propose")[-1]
    assert req.headers["x-litkit-acting-user"] == USER_ID


def test_review_criteria_change_is_a_new_version_then_reproposed(fake, env):
    review = FakeReview(fake)
    set_id = call("litkit_review", action="criteria", criteriaAction="create", name="Part 11",
                  criteria=CRITERIA)["set"]["id"]
    added = CRITERIA + [{"title": "REV-NR", "tagName": "REV-NR", "description": "Part 11 item 6"}]
    out = call("litkit_review", action="criteria", criteriaAction="update", criteriaSetId=set_id, criteria=added,
               changeNote="add REV-NR as Part 11 item 6", baseVersion=1)
    assert out["version"] == 2
    patch = fake.calls("PATCH", rf"/api/matters/{M}/criteria-sets/{set_id}")[-1].json()
    assert patch["changeNote"] == "add REV-NR as Part 11 item 6" and patch["baseVersion"] == 1
    assert patch["criteria"][-1] == {"title": "REV-NR", "tagName": "REV-NR", "description": "Part 11 item 6"}
    assert call("litkit_review", action="criteria", criteriaAction="publish", criteriaSetId=set_id)["ok"] is True
    assert fake.calls("POST", rf"/api/matters/{M}/criteria-sets/{set_id}/publish")[-1].json() == {}
    versions = call("litkit_review", action="criteria", criteriaAction="versions", criteriaSetId=set_id)
    assert [v["version"] for v in versions["versions"]] == [1, 2]
    again = call("litkit_review", action="create", criteriaSetId=set_id, scope={"batesRange": {
        "start": " ABC0000001", "end": "ABC0004000"}}, tags=["Pricing", "REV-NR"])
    assert again["criteriaSetVersion"] == 2
    assert review.proposals[-1]["request"]["scope"] == {"batesRange": {"start": "ABC0000001", "end": "ABC0004000"}}


def test_review_create_billing_quote_then_confirm(fake, env):
    review = FakeReview(fake, price_usd=41.5)
    base = {"action": "create", "criteria": CRITERIA, "tags": ["Pricing"],
            "scope": {"filter": {"custodian": "Doe, Jane", "dateFrom": "2021-01-01"}}}
    quoted = call("litkit_review", **base)
    assert quoted["requiresApproval"] is True and quoted["quote"]["amountEstUsd"] == 41.5
    assert "Nothing is proposed yet" in quoted["next"] and "quoteId=q1" in quoted["next"]
    assert review.proposals == []
    first = fake.calls("POST", rf"/api/matters/{M}/review-jobs/propose")[-1].json()
    assert "quoteId" not in first and "userConfirmed" not in first
    assert first["criteria"] == [CRITERIA[0], {"title": "Board materials"}]
    # userConfirmed without the quote id never reaches LitKit
    n = len(fake.requests)
    assert "quoteId" in call("litkit_review", **base, userConfirmed=True)["error"]
    assert len(fake.requests) == n
    done = call("litkit_review", **base, quoteId="q1", userConfirmed=True)
    assert done["proposalId"] and "card in this thread" in done["next"]
    second = fake.calls("POST", rf"/api/matters/{M}/review-jobs/propose")[-1].json()
    assert second["quoteId"] == "q1" and second["userConfirmed"] is True
    assert review.quotes == {"q1": "consumed"} and len(review.proposals) == 1


def test_review_create_sends_optimizations_considered_only_when_given(fake, env):
    FakeReview(fake)
    base = {"action": "create", "criteriaSetId": _id(1), "tags": ["P"], "scope": {"workSetId": _id(2)}}
    call("litkit_review", **base)
    assert "optimizationsConsidered" not in fake.calls("POST", rf"/api/matters/{M}/review-jobs/propose")[-1].json()
    notes = [" Jev first pass on: 4,000 topical docs ", "scope 2021-2023: 4,000 of 9,100", ""]
    call("litkit_review", **base, optimizations_considered=notes)
    sent = fake.calls("POST", rf"/api/matters/{M}/review-jobs/propose")[-1].json()
    assert sent["optimizationsConsidered"] == ["Jev first pass on: 4,000 topical docs", "scope 2021-2023: 4,000 of 9,100"]
    n = len(fake.requests)
    assert "at most 5" in call("litkit_review", **base, optimizations_considered=[f"o{i}" for i in range(6)])["error"]
    assert "200 characters" in call("litkit_review", **base, optimizations_considered=["x" * 201])["error"]
    assert len(fake.requests) == n


def test_review_create_sends_the_turns_thread_as_thread_id(fake, env):
    """ana-review-contract: threadId is the turn's sessionId, so the Launch card posts in that thread."""
    FakeReview(fake)
    base = {"action": "create", "criteriaSetId": _id(1), "tags": ["P"], "scope": {"workSetId": _id(2)}}
    thread = "7d0c6f2e-3a1b-4c5d-8e9f-0a1b2c3d4e5f"

    def proposed_under(identity):
        token = bind_turn(identity)
        try:
            call("litkit_review", **base)
        finally:
            reset_turn(token)
        return fake.calls("POST", rf"/api/matters/{M}/review-jobs/propose")[-1].json()

    turn = TurnIdentity(turn_id="turn_2", matter_id=M, acting_user=USER_ID, cwd=env["cwd"], thread_id=thread)
    assert proposed_under(turn)["threadId"] == thread
    # A session id LitKit could not resolve as a thread (it takes a uuid) is left out, not sent to fail.
    slack = TurnIdentity(turn_id="turn_3", matter_id=M, acting_user=USER_ID, cwd=env["cwd"], thread_id="C0123:1712.5")
    assert "threadId" not in proposed_under(slack)
    # Outside a turn (cron, unattended work) there is no thread.
    token = bind_turn(None)
    try:
        call("litkit_review", **base)
    finally:
        reset_turn(token)
    assert "threadId" not in fake.calls("POST", rf"/api/matters/{M}/review-jobs/propose")[-1].json()


def test_review_create_second_approver_is_explained(fake, env):
    fake.route("POST", rf"/api/matters/{M}/review-jobs/propose",
               {"proposed": False, "requiresApproval": True, "needsSecondApprover": True,
                "quote": {"id": "q9", "amountEstUsd": 9000, "docCount": 90000}})
    out = call("litkit_review", action="create", criteriaSetId="builtin:privilege", tags=["Privileged"],
               scope={"documentIds": [_id(1)]}, quoteId="q9", userConfirmed=True)
    assert "different matter admin" in out["next"] and "q9" in out["next"]


@pytest.mark.parametrize("args,message", [
    ({"tags": ["P"], "criteriaSetId": _id(1)}, "'scope' must be an object"),
    ({"tags": ["P"], "criteriaSetId": _id(1), "scope": {}}, "exactly one of"),
    ({"tags": ["P"], "criteriaSetId": _id(1), "scope": {"workSetId": _id(2), "documentIds": [_id(3)]}},
     "exactly one of"),
    ({"tags": ["P"], "criteriaSetId": _id(1), "scope": {"workSetId": "../../admin"}}, "uuid"),
    ({"tags": ["P"], "criteriaSetId": _id(1), "scope": {"documentIds": [_id(1), "ABC0001"]}}, "uuid"),
    ({"tags": ["P"], "criteriaSetId": _id(1), "scope": {"batesRange": {"start": "ABC1"}}}, "{start, end}"),
    ({"tags": ["P"], "criteriaSetId": _id(1), "scope": {"filter": "custodian=Doe"}}, "scope.filter"),
    ({"tags": ["P"], "scope": {"workSetId": _id(2)}}, "exactly one of criteriaSetId"),
    ({"tags": ["P"], "criteriaSetId": _id(1), "criteria": CRITERIA, "scope": {"workSetId": _id(2)}},
     "exactly one of criteriaSetId"),
    ({"tags": ["P"], "criteriaSetId": "../x", "scope": {"workSetId": _id(2)}}, "criteria set id"),
    ({"tags": ["P"], "criteria": [], "scope": {"workSetId": _id(2)}}, "non-empty list"),
    ({"tags": ["P"], "criteria": [{"description": "no title"}], "scope": {"workSetId": _id(2)}}, "needs a title"),
    ({"tags": ["P"], "criteria": [{"title": "t", "disposition": "maybe"}], "scope": {"workSetId": _id(2)}},
     "disposition"),
    ({"criteriaSetId": _id(1), "scope": {"workSetId": _id(2)}}, "'tags' must be a non-empty list"),
    ({"tags": [" "], "criteriaSetId": _id(1), "scope": {"workSetId": _id(2)}}, "at least one tag"),
    ({"tags": [f"t{i}" for i in range(26)], "criteriaSetId": _id(1), "scope": {"workSetId": _id(2)}}, "at most 25"),
])
def test_review_create_rejects_bad_arguments_before_any_request(fake, env, args, message):
    out = call("litkit_review", action="create", **args)
    assert message in out["error"], out
    assert fake.requests == []


@pytest.mark.parametrize("args,message", [
    ({"criteriaAction": "create", "criteria": CRITERIA}, "'name' is required"),
    ({"criteriaAction": "create", "name": "x"}, "non-empty list"),
    ({"criteriaAction": "get"}, "'criteriaSetId' is required"),
    ({"criteriaAction": "update", "criteriaSetId": _id(1)}, "non-empty list"),
    ({"criteriaAction": "publish", "criteriaSetId": "a/b"}, "criteria set id"),
    ({"criteriaAction": "delete", "criteriaSetId": _id(1)}, "criteriaAction must be one of"),
])
def test_review_criteria_rejects_bad_arguments_before_any_request(fake, env, args, message):
    out = call("litkit_review", action="criteria", **args)
    assert message in out["error"], out
    assert fake.requests == []


def test_review_accept_tags_and_unknown_actions(fake, env):
    FakeReview(fake)
    out = call("litkit_review", action="accept_tags", jobId="job1", includeRationaleNotes=True)
    assert out == {"action": "accept_tags", "jobId": "job1", "result": {"ok": True, "accepted": 17, "jobId": "job1"}}
    assert fake.calls("POST", rf"/api/matters/{M}/review-jobs/job1/accept-all-tags")[-1].json() == {
        "includeRationaleNotes": True}
    call("litkit_review", action="accept_tags", jobId="job2")
    assert fake.requests[-1].json() == {}
    n = len(fake.requests)
    assert "'jobId' is required" in call("litkit_review", action="accept_tags")["error"]
    assert "action must be one of" in call("litkit_review", action="launch", jobId="job1")["error"]
    assert len(fake.requests) == n


def test_review_create_permission_refusal_is_plain(fake, env):
    fake.route("POST", rf"/api/matters/{M}/review-jobs/propose", (403, {"error": "forbidden"}))
    out = call("litkit_review", action="create", criteriaSetId=_id(1), tags=["P"], scope={"workSetId": _id(2)})
    assert out["permission_denied"] is True and out["error"].startswith("not permitted")
    assert len(fake.calls("POST", rf"/api/matters/{M}/review-jobs/propose")) == 1


# ---------------------------------------------------------------------------
# memory: the "memory failed" pill. LitKit's remember accepts five kinds; the tool used to
# default to "note", which LitKit answers 400 `remember: invalid kind "note"`.


def test_remember_defaults_to_a_kind_litkit_accepts(fake, env):
    fake.route("POST", r"/api/agent/actions", lambda r: (200, {"ok": True, "echo": r.json()}))
    out = call("litkit_remember", content="Jane Doe left Acme in March 2022")
    assert out["echo"]["args"] == {"kind": "fact", "content": "Jane Doe left Acme in March 2022"}
    # kind has no enum: matter kinds and firm/person kinds differ, so the description names each set
    # and the tool validates at call time.
    desc = T.SCHEMAS["litkit_remember"]["parameters"]["properties"]["kind"]["description"]
    assert all(k in desc for k in T.REMEMBER_KINDS)


def test_remember_rejects_a_kind_litkit_would_refuse_before_any_request(fake, env):
    for kind in ("note", "style"):
        out = call("litkit_remember", content="x", kind=kind)
        assert "kind must be one of fact, strategy" in out["error"], out
    assert fake.requests == []

# FIRM_AGENT_HOST 4: firm and person memory
# ---------------------------------------------------------------------------

SHARED = r"/api/agent/shared-memory"


def test_remember_firm_and_person_go_to_the_shared_route(fake, env):
    fake.route("POST", SHARED, lambda r: (200, {"id": "sm1", "status": "proposed" if r.json()["scope"] == "firm"
                                                 else "active"}))
    out = call("litkit_remember", content="Cite exhibits as  Ex. N at p.", scope="firm")
    assert out == {"saved": True, "scope": "firm", "id": "sm1", "status": "proposed",
                   "notes": ["a Firm admin must confirm a firm convention before other matters see it"]}
    req = fake.requests[-1]
    assert req.json() == {"scope": "firm", "kind": "convention", "content": "Cite exhibits as Ex. N at p."}
    assert req.headers["x-litkit-acting-user"] == USER_ID and "x-litkit-turn-grant" not in req.headers

    out = call("litkit_remember", content="Short memos.", scope="person")
    assert out["saved"] is True and out["status"] == "active" and "notes" not in out
    assert fake.requests[-1].json() == {"scope": "person", "kind": "preference", "content": "Short memos."}
    assert not fake.calls("POST", r"/api/agent/actions")  # never a matter write


def test_remember_shared_checks_before_calling(fake, env):
    assert "must be one of convention" in call("litkit_remember", content="x", scope="firm", kind="fact")["error"]
    assert "at most 1000" in call("litkit_remember", content="x" * 1001, scope="person")["error"]
    token = bind_turn(TurnIdentity(acting_user=None, cwd=env["cwd"]))
    try:
        assert "needs a lawyer" in call("litkit_remember", content="x", scope="firm")["error"]
    finally:
        reset_turn(token)
    assert fake.requests == []


def test_remember_shared_reports_refusal_and_missing_route(fake, env):
    out = call("litkit_remember", content="Short memos.", scope="person")  # today's app: no route
    assert out["saved"] is False and out["unavailable"] is True and "nothing was saved" in out["message"]
    assert not fake.calls("POST", r"/api/agent/actions")

    fake.route("POST", SHARED, (422, {"error": "matter facts stay in matter memory", "code": "matter_fact"}))
    out = call("litkit_remember", content="We settled at $4.2M.", scope="firm")
    assert out["saved"] is False and out["status"] == 422
    assert out["error"] == "matter facts stay in matter memory: matter_fact"
    assert "scope=matter" in out["notes"][0]


def test_recall_merges_matter_firm_and_person(fake, env):
    fake.route("POST", r"/api/agent/actions", lambda r: (200, {"ok": True, "memories": [{"id": "m1"}],
                                                              "args": r.json()["args"]}))
    fake.route("GET", SHARED, lambda r: (200, {"items": [
        {"id": f"{r.query['scope'][0]}1", "kind": "convention", "content": f"{r.query['scope'][0]} rule",
         "status": "active"}, "junk"]}))
    out = call("litkit_recall", query="memo", limit=99)
    assert out["matter"] == {"ok": True, "memories": [{"id": "m1"}], "args": {"query": "memo", "limit": 50}}
    assert out["firm"] == [{"id": "firm1", "kind": "convention", "content": "firm rule"}]
    assert out["person"] == [{"id": "person1", "kind": "convention", "content": "person rule"}]
    gets = fake.calls("GET", SHARED)
    assert [g.query["scope"] for g in gets] == [["firm"], ["person"]]
    assert all(g.headers["x-litkit-acting-user"] == USER_ID for g in gets)

    fake.requests.clear()
    out = call("litkit_recall", scope="firm")
    assert set(out) == {"firm", "notes"} and len(fake.requests) == 1


def test_recall_is_fail_soft_without_the_shared_route(fake, env):
    fake.route("POST", r"/api/agent/actions", (200, {"ok": True, "memories": []}))
    out = call("litkit_recall")
    assert out["matter"] == {"ok": True, "memories": []}
    assert out["firm"] == [] and out["person"] == []
    assert out["notes"].count("firm and personal memory are not available on this LitKit yet") == 1

    fake.route("GET", SHARED, (500, {"error": "boom"}))
    out = call("litkit_recall", scope="firm")
    assert out["firm"] == [] and "could not be read" in out["notes"][0]


def test_recall_without_a_lawyer_skips_person_notes(fake, env):
    fake.route("POST", r"/api/agent/actions", (200, {"ok": True}))
    fake.route("GET", SHARED, {"items": []})
    token = bind_turn(TurnIdentity(acting_user=None, cwd=env["cwd"]))
    try:
        out = call("litkit_recall")
    finally:
        reset_turn(token)
    assert "person" not in out and [g.query["scope"] for g in fake.calls("GET", SHARED)] == [["firm"]]


# ---------------------------------------------------------------------------
# FIRM_AGENT_HOST 6: cross-matter search
# ---------------------------------------------------------------------------

CROSS = r"/api/agent/cross-matter-search"


def _with_grant(env, grant="grant.v1.mac", acting=USER_ID):
    return bind_turn(TurnIdentity(turn_id="turn_1", matter_id=M, acting_user=acting, cwd=env["cwd"],
                                  turn_grant=grant))


def test_cross_matter_search_without_a_grant_points_to_a_private_thread(fake, env):
    out = call("litkit_cross_matter_search", query="ZEBRA-7731")  # the env turn carries no grant
    assert out["available"] is False and "private thread" in out["message"]
    token = _with_grant(env, acting=None)
    try:
        assert call("litkit_cross_matter_search", query="ZEBRA-7731")["available"] is False
    finally:
        reset_turn(token)
    assert fake.requests == []


def test_cross_matter_search_sends_the_grant_and_returns_labelled_snippets(fake, env):
    other = _id(77)
    fake.route("POST", CROSS, (200, {"hits": [
        {"matterId": other, "matterName": "Matter B", "documentId": _id(i), "bates": f"B{i:05d}",
         "title": "Email", "snippet": "ZEBRA-7731 " + "x" * 400, "text": "full text must not pass"}
        for i in range(12)]}))
    token = _with_grant(env)
    try:
        out = call("litkit_cross_matter_search", query="ZEBRA-7731", matterIds=[other])
    finally:
        reset_turn(token)
    req = fake.requests[-1]
    assert req.json() == {"query": "ZEBRA-7731", "matterIds": [other]}
    assert req.headers["x-litkit-turn-grant"] == "grant.v1.mac"
    assert req.headers["x-litkit-acting-user"] == USER_ID and req.headers["authorization"] == f"Bearer {TOKEN}"
    assert out["available"] is True and out["hits"] == 10
    first = out["results"][0]
    assert first["matterName"] == "Matter B" and first["bates"] == "B00000" and "text" not in first
    assert len(first["snippet"]) <= 300
    assert any("matter name" in n for n in out["notes"]) and any("never save" in n for n in out["notes"])
    assert _files_outside(env["tmp"], env["tmp"] / "nothing") == []  # no snippet is written into this matter


def test_cross_matter_search_other_answers(fake, env):
    token = _with_grant(env)
    try:
        out = call("litkit_cross_matter_search", query="x")  # today's app: no route
        assert out["available"] is False and "not available on this LitKit" in out["message"]
        fake.route("POST", CROSS, (403, {"error": "turn_grant_invalid"}))
        out = call("litkit_cross_matter_search", query="x")
        assert out["permission_denied"] is True and "private thread" in out["message"]
        assert "must be a list" in call("litkit_cross_matter_search", query="x", matterIds=["nope"])["error"]
    finally:
        reset_turn(token)
    assert len(fake.calls("POST", CROSS)) == 2


# -- canonical links (clickable references) --------------------------------------
# Tools hand Ana a ready-made app-relative link for each document, LitSpace file and folder,
# built only from ids LitKit returned. A bad id drops the link and keeps the row.

LS = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"
LS_OTHER = "bbbbbbbb-cccc-4ddd-8eee-ffffffffffff"


def _links(value):
    """Every ``link`` / ``folderLink`` value anywhere in a tool result."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in ("link", "folderLink"):
                yield item
            else:
                yield from _links(item)
    elif isinstance(value, list):
        for item in value:
            yield from _links(item)


def _target(link):
    return link.rsplit("](", 1)[1][:-1]


def _no_origin(fake, *results):
    links = [link for result in results for link in _links(result)]
    assert links
    for link in links:
        assert "http" not in link and fake.url not in link and TOKEN not in link and len(link) <= 400


def test_files_list_links_each_row_to_its_own_litspace_matter(fake, env):
    fake.route("GET", rf"/api/litspace/matters/{M}/files", {"files": [
        {"litspaceMatterId": LS, "litspaceFileId": _id(1), "filename": "Smith (final) [v2].pdf", "bytes": 9},
        {"litspaceMatterId": LS_OTHER, "litspaceFileId": _id(2), "filename": "b.docx"},
        {"litspaceMatterId": LS, "litspaceFileId": "not-a-uuid", "filename": "c.pdf", "mime": "application/pdf"}]})
    out = call("litkit_files", action="list")
    first, second, bad = out["results"]
    assert _target(first["link"]) == f"/litspace/matters/{LS}/documents/{_id(1)}"
    assert first["link"].startswith("[Smith (final) \\[v2\\].pdf](")
    assert _target(second["link"]) == f"/litspace/matters/{LS_OTHER}/documents/{_id(2)}"
    assert "link" not in bad and bad == {"fileId": "not-a-uuid", "filename": "c.pdf", "mime": "application/pdf",
                                         "bytes": None, "updatedAt": None}
    _no_origin(fake, out)


def test_files_search_links_the_file_and_its_folder(fake, env):
    fake.route("GET", rf"/api/litspace/matters/{M}/files", {"files": [{"litspaceMatterId": LS}]})
    fake.route("GET", rf"/api/litspace/matters/{LS}/search", {"rows": [
        {"documentId": _id(1), "filename": "2026-09-15 Exhibit A - Privilege Log.pdf",
         "folderDisplay": "Productions/Volume 1", "snippet": "privilege"},
        {"documentId": _id(2), "filename": "root.pdf", "folderDisplay": "", "snippet": "x"},
        {"documentId": "bad", "filename": "odd.pdf", "folderDisplay": "Productions", "snippet": "y"}]})
    out = call("litkit_files", action="search", query="privilege")
    nested, root, bad = out["results"]
    assert _target(nested["link"]) == f"/litspace/matters/{LS}/documents/{_id(1)}"
    assert _target(nested["folderLink"]) == f"/litspace/matters/{LS}/files?path=Productions/Volume%201"
    assert "link" in root and "folderLink" not in root
    assert "link" not in bad and _target(bad["folderLink"]) == f"/litspace/matters/{LS}/files?path=Productions"
    assert bad["fileId"] == "bad" and bad["filename"] == "odd.pdf" and bad["snippet"] == "y"
    _no_origin(fake, out)


def test_files_read_links_only_when_litkit_names_the_litspace_matter(fake, env):
    fake.route("GET", r"/api/litspace/files/.*/content", (200, b"file"))
    fake.route("GET", rf"/api/litspace/files/{_id(5)}", {"filename": "log.pdf", "litspaceMatterId": LS})
    fake.route("GET", rf"/api/litspace/files/{_id(6)}", {"filename": "other.pdf"})
    linked = call("litkit_files", action="read", fileId=_id(5))
    assert _target(linked["link"]) == f"/litspace/matters/{LS}/documents/{_id(5)}"
    assert linked["link"].startswith("[log.pdf](")
    assert "link" not in call("litkit_files", action="read", fileId=_id(6))
    _no_origin(fake, linked)


def test_docs_page_links_only_rows_without_bates(fake, env):
    fake.route("GET", rf"/api/matters/{M}/docs", {"total": 3, "hasMore": False, "nextCursor": None, "docs": [
        {"id": _id(1), "batesStart": "ABC00001", "fileName": "a.pdf"},
        {"id": _id(2), "fileName": "Smith (final) [v2].pdf"},
        {"id": _id(3)}, {"id": "nope", "fileName": "x.pdf"}]})
    out = call("litkit_docs")
    bates, named, bare, bad = out["docs"]
    assert "link" not in bates
    assert _target(named["link"]) == f"/matters/{M}?doc={_id(2)}" and named["link"].startswith("[Smith (final)")
    assert bare["link"].startswith("[document](")
    assert bad == {"id": "nope", "fileName": "x.pdf"}
    _no_origin(fake, out)
    # census files written with saveAs carry no links
    call("litkit_docs", saveAs="all")
    assert not any("link" in json.loads(line) for line in (env["cwd"] / "census" / "all.jsonl").read_text().splitlines())


def test_search_links_only_hits_without_bates(fake, env):
    fake.route("GET", rf"/api/matters/{M}/search", {"hits": [
        {"docId": _id(1), "bates": "ABC00001", "page": 2, "snippet": "a"},
        {"docId": _id(2), "page": 4, "snippet": "b"},
        {"docId": _id(3), "snippet": "c"}, {"docId": "nope", "page": 1, "snippet": "d"}]})
    out = call("litkit_search", query='"price"')
    bates, paged, unpaged, bad = out["results"]
    assert "link" not in bates
    assert _target(paged["link"]) == f"/matters/{M}?doc={_id(2)}&page=4" and paged["link"].startswith("[document](")
    assert _target(unpaged["link"]) == f"/matters/{M}?doc={_id(3)}"
    assert "link" not in bad and bad["snippet"] == "d" and bad["page"] == 1
    _no_origin(fake, out)


def test_single_document_tools_label_the_link_by_bates_then_file_name(fake, env):
    with_bates, without = _id(7), _id(8)
    fake.route("GET", rf"/api/matters/{M}/bates-resolve", {"documentId": with_bates})
    fake.route("GET", rf"/api/documents/{with_bates}", {"doc": {"batesStart": "ABC00007", "fileName": "a.msg"}})
    fake.route("GET", rf"/api/documents/{without}", {"doc": {"fileName": "Smith (final) [v2].pdf"}})
    fake.route("GET", r"/api/documents/[^/]+/text", {"extractedText": "t"})
    fake.route("GET", r"/api/documents/[^/]+/pdf", (200, b"%PDF-1.4"))
    results = []
    for tool in ("litkit_document", "litkit_text", "litkit_pdf"):
        by_bates = call(tool, bates="ABC00007")
        named = call(tool, documentId=without)
        assert by_bates["link"] == f"[ABC00007](/matters/{M}?doc={with_bates})"
        assert named["link"] == f"[Smith (final) \\[v2\\].pdf](/matters/{M}?doc={without})"
        results += [by_bates, named]
    _no_origin(fake, *results)


def test_pdf_without_metadata_still_links_as_document(fake, env):
    d = _id(9)
    fake.route("GET", rf"/api/documents/{d}", (404, {"error": "not_found"}))
    fake.route("GET", rf"/api/documents/{d}/pdf", (200, b"%PDF-1.4"))
    assert call("litkit_pdf", documentId=d)["link"] == f"[document](/matters/{M}?doc={d})"


def test_a_refused_lookup_stays_the_plain_permission_result(fake, env):
    fake.route("GET", rf"/api/litspace/matters/{M}/files", (403, {"error": "forbidden"}))
    out = call("litkit_files", action="list")
    assert "not permitted" in out["error"] and not list(_links(out))
