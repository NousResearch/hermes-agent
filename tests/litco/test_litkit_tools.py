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
from tests.litco._litkit_fake import HOST_SECRET, MATTER_ID, TOKEN, USER_ID, FakeLitKit, ndjson

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


def call(name: str, **args):
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
    out = call("litkit_remember", content="prefers short memos", scope="user", kind="style")
    assert out["echo"] == {"action": "remember", "matterId": M,
                           "args": {"kind": "style", "content": "prefers short memos", "acl": {"scope": "user"}}}
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
    assert len(T.TOOLS) == 24
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
