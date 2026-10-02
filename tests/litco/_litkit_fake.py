"""A fake LitKit instance for the toolset tests: a real HTTP server on localhost.

Routes are registered as ``fake.route(method, path_regex, handler)``; a handler receives the
recorded request and returns ``(status, body)`` or ``(status, body, headers)``. ``body`` may be
a dict/list (sent as JSON), ``bytes``, or a ``str``. Every request is recorded with its method,
path, query, headers and raw body so tests can assert what the client sent.
"""

from __future__ import annotations

import json
import re
import socketserver
import threading
from dataclasses import dataclass, field
from email.parser import BytesParser
from email.policy import default as email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Tuple
from urllib.parse import parse_qs, urlsplit

TOKEN = "lkm_" + "A" * 43
HOST_SECRET = "host-secret-for-tests"
MATTER_ID = "11111111-2222-3333-4444-555555555555"
USER_ID = "99999999-8888-7777-6666-555555555555"


@dataclass
class Recorded:
    method: str
    path: str
    query: Dict[str, List[str]]
    headers: Dict[str, str]
    body: bytes
    raw_headers: List[Tuple[str, str]] = field(default_factory=list)
    status: int = 0  # the status the fake answered

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8") or "null")

    def form(self) -> Dict[str, Any]:
        """Parse a multipart body into ``{field: str | (filename, bytes)}``."""
        ctype = self.headers.get("content-type", "")
        msg = BytesParser(policy=email_policy).parsebytes(
            f"Content-Type: {ctype}\r\n\r\n".encode("utf-8") + self.body)
        out: Dict[str, Any] = {}
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            filename = part.get_filename()
            payload = part.get_payload(decode=True) or b""
            out[name] = (filename, payload) if filename else payload.decode("utf-8")
        return out


class _QuickServer(ThreadingHTTPServer):
    """``HTTPServer.server_bind`` does a reverse DNS lookup (``getfqdn``) that can stall for
    tens of seconds on some hosts; skip it."""

    daemon_threads = True

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name, self.server_port = host, port


class FakeLitKit:
    def __init__(self) -> None:
        self.routes: List[Tuple[str, re.Pattern, Callable[[Recorded], Any]]] = []
        self.requests: List[Recorded] = []
        self._lock = threading.Lock()
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a: Any) -> None:  # quiet
                pass

            def _handle(self) -> None:
                length = int(self.headers.get("content-length") or 0)
                body = self.rfile.read(length) if length else b""
                parts = urlsplit(self.path)
                rec = Recorded(method=self.command, path=parts.path, query=parse_qs(parts.query, keep_blank_values=True),
                               headers={k.lower(): v for k, v in self.headers.items()}, body=body,
                               raw_headers=list(self.headers.items()))
                with fake._lock:
                    fake.requests.append(rec)
                result = fake._dispatch(rec)
                if not isinstance(result, tuple):
                    result = (200, result)
                status, payload, headers = result if len(result) == 3 else (result[0], result[1], {})
                rec.status = status
                if isinstance(payload, (dict, list)):
                    data = json.dumps(payload).encode("utf-8")
                    headers = {"Content-Type": "application/json", **headers}
                elif isinstance(payload, str):
                    data = payload.encode("utf-8")
                    headers = {"Content-Type": "text/plain", **headers}
                else:
                    data = payload or b""
                    headers = {"Content-Type": "application/octet-stream", **headers}
                self.send_response(status)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            do_GET = do_POST = do_PATCH = do_DELETE = do_PUT = _handle

        self.server = _QuickServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05},
                                       daemon=True)

    @property
    def url(self) -> str:
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "FakeLitKit":
        self.thread.start()
        return self

    def stop(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def route(self, method: str, pattern: str, handler: Any) -> None:
        fn = handler if callable(handler) else (lambda _req, _h=handler: _h)
        self.routes.insert(0, (method.upper(), re.compile("^" + pattern + "$"), fn))

    def _dispatch(self, rec: Recorded) -> Tuple:
        for method, pattern, fn in self.routes:
            if method == rec.method and pattern.match(rec.path):
                return fn(rec)
        return (404, {"error": "not_found"})

    def calls(self, method: str, pattern: str) -> List[Recorded]:
        rx = re.compile("^" + pattern + "$")
        return [r for r in self.requests if r.method == method and rx.match(r.path)]


def ndjson(rows: List[Dict[str, Any]]) -> Tuple[int, bytes, Dict[str, str]]:
    return 200, ("\n".join(json.dumps(r) for r in rows) + "\n").encode("utf-8"), {
        "Content-Type": "application/x-ndjson"}


class FakeReview:
    """LitKit's review-structure routes for one matter, with state, shaped as litkit-app answers them
    (docs/plans/ana-review-contract.md and the route code): criteria sets and their versions (an edit is
    a publish of the whole list from a base version; there is no PATCH), the matter's work sets,
    ``review-jobs/propose`` with the review_run billing protocol, a proposal's launch, withdraw and read,
    the launched job's status and records, and accept-all-tags.

    Launch follows the app's thread rule as a test sees it: LitKit, not the host, decides whether the
    person answered after the card. ``reply(proposal_id)`` records that they did; until then launch
    answers ``409 no_reply_after_card``. Launch needs the turn's grant (``X-LitKit-Turn-Grant``), as the app
    requires. ``launch_route = False`` makes the launch and withdraw routes
    404 as an app that predates them does.

    Billing follows litkit-app's ``review_run``: with ``price_usd > 0``, a propose without a quoteId
    returns ``requiresApproval`` and a draft quote and proposes nothing; a propose with that quoteId
    and ``userConfirmed: true`` approves the quote and queues the proposal. A quoteId the person has
    not approved is a 400 with a sentence in ``error``."""

    def __init__(self, fake: FakeLitKit, matter_id: str = MATTER_ID, *, price_usd: float = 0.0,
                 doc_count: int = 1234) -> None:
        self.price_usd = price_usd
        self.doc_count = doc_count
        self.sets: Dict[str, Dict[str, Any]] = {}
        self.quotes: Dict[str, str] = {}  # quote id -> draft | approved | consumed
        self.proposals: List[Dict[str, Any]] = []
        self.replied: set = set()  # proposal ids the person has answered "launch" to in the proposal's thread
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.launch_route = True
        self.sets["builtin:privilege"] = {"id": "builtin:privilege", "scope": "firm", "name": "Privilege",
                                          "builtIn": True, "currentVersion": 1, "versions": [{
                                              "version": 1, "changeNote": "", "criteria": [{
                                                  "title": "Privileged communications", "tagName": "Privileged"}]}]}
        self.work_sets = [
            {"id": "00000000-0000-4000-8000-00000000a001", "name": "Crowder emails", "status": "open",
             "docCount": 812, "creator": {"id": USER_ID, "name": "Lawyer"},
             "assignee": {"id": "00000000-0000-4000-8000-00000000b002", "name": "Reviewer"}},
            {"id": "00000000-0000-4000-8000-00000000a002", "name": "Hot docs QC", "status": "open",
             "docCount": 40, "creator": {"id": "00000000-0000-4000-8000-00000000b002", "name": "Reviewer"},
             "assignee": {"id": USER_ID, "name": "Lawyer"}},
            {"id": "00000000-0000-4000-8000-00000000a003", "name": "Unassigned", "status": "open",
             "docCount": 5, "creator": {"id": "00000000-0000-4000-8000-00000000b003", "name": "Other"},
             "assignee": None},
        ]
        base = f"/api/matters/{matter_id}"
        sid = r"(?:[0-9a-f-]{36}|builtin:[a-z0-9_-]+)"
        fake.route("GET", rf"{base}/criteria-sets", lambda r: {
            "matter": None, "mine": [self._summary(s) for s in self.sets.values() if not s.get("builtIn")],
            "firm": [], "builtIn": [self._summary(s) for s in self.sets.values() if s.get("builtIn")]})
        fake.route("POST", rf"{base}/criteria-sets", self._create)
        fake.route("GET", rf"{base}/criteria-sets/{sid}", lambda r: self._get(r.path.split("/")[-1]))
        fake.route("POST", rf"{base}/criteria-sets/{sid}/publish", self._publish)
        fake.route("GET", rf"{base}/criteria-sets/{sid}/versions",
                   lambda r: self._with_set(r.path.split("/")[-2], lambda s: {"versions": [
                       {k: v[k] for k in ("version", "changeNote")} for v in s["versions"]]}))
        fake.route("GET", rf"{base}/work-sets", lambda r: {"sets": self.work_sets})
        fake.route("POST", rf"{base}/review-jobs/propose", self._propose)
        pid = r"[0-9a-f-]{36}"
        fake.route("GET", rf"{base}/proposals/{pid}", lambda r: self._with_proposal(r, self._read))
        fake.route("POST", rf"{base}/proposals/{pid}/launch", lambda r: self._unrouted(r, self._launch))
        fake.route("POST", rf"{base}/proposals/{pid}/withdraw", lambda r: self._unrouted(r, self._withdraw))
        fake.route("GET", rf"{base}/review-jobs/{pid}",
                   lambda r: self._with_job(r.path.split("/")[-1], lambda j: {"job": j}))
        fake.route("GET", rf"{base}/review-jobs/{pid}/records",
                   lambda r: self._with_job(r.path.split("/")[-2], lambda j: {"records": [], "total": 0}))
        fake.route("POST", rf"{base}/review-jobs/[^/]+/accept-all-tags",
                   lambda r: {"ok": True, "accepted": 17, "jobId": r.path.split("/")[-2]})

    @staticmethod
    def _summary(s: Dict[str, Any]) -> Dict[str, Any]:
        return {"id": s["id"], "scope": s["scope"], "name": s["name"], "currentVersion": s["currentVersion"],
                "criteriaCount": len(s["versions"][-1]["criteria"])}

    def _with_set(self, set_id: str, fn: Callable[[Dict[str, Any]], Any]) -> Any:
        found = self.sets.get(set_id)
        return fn(found) if found else (404, {"error": "not_found"})

    def _get(self, set_id: str) -> Any:
        return self._with_set(set_id, lambda s: {"set": self._summary(s), "version": s["currentVersion"],
                                                 "criteria": s["versions"][-1]["criteria"],
                                                 "unresolvedTagNames": []})

    def _create(self, req: Recorded) -> Any:
        body = req.json()
        if body.get("scope") not in ("user", "firm") or not body.get("name") or not body.get("criteria"):
            return 400, {"error": "invalid input", "details": {"scope": "Required"}}
        set_id = f"00000000-0000-4000-8000-{len(self.sets) + 1:012d}"
        self.sets[set_id] = {"id": set_id, "scope": body["scope"], "name": body["name"], "currentVersion": 1,
                             "versions": [{"version": 1, "criteria": body["criteria"], "changeNote": ""}]}
        return 201, {"set": {"id": set_id, "scope": body["scope"], "name": body["name"], "currentVersion": 1},
                     "version": 1, "criteria": body["criteria"]}

    def _publish(self, req: Recorded) -> Any:
        body = req.json()
        if not isinstance(body.get("criteria"), list) or not body["criteria"] \
                or not isinstance(body.get("baseVersion"), int):
            return 400, {"error": "invalid input", "details": {"criteria": "Required", "baseVersion": "Required"}}

        def publish(s: Dict[str, Any]) -> Any:
            current = s["versions"][-1]
            if body["baseVersion"] != s["currentVersion"]:
                return 409, {"error": "stale_version", "currentVersion": s["currentVersion"],
                             "criteria": current["criteria"]}
            if body["criteria"] == current["criteria"]:
                return {"ok": True, "set": self._summary(s), "version": s["currentVersion"], "written": False,
                        "criteria": current["criteria"]}
            s["currentVersion"] += 1
            s["versions"].append({"version": s["currentVersion"], "criteria": body["criteria"],
                                  "changeNote": body.get("changeNote", "")})
            return {"ok": True, "set": self._summary(s), "version": s["currentVersion"], "written": True,
                    "criteria": body["criteria"]}
        return self._with_set(req.path.split("/")[-2], publish)

    def _propose(self, req: Recorded) -> Any:
        body = req.json()
        set_id = body.get("criteriaSetId")
        if set_id and set_id not in self.sets:
            return 400, {"error": "criteriaSetId names no criteria set this person can see",
                         "code": "criteria_set_not_found"}
        rows = self.sets[set_id]["versions"][-1]["criteria"] if set_id else body.get("criteria") or []
        row_tags = list(dict.fromkeys(r["tagName"] for r in rows if r.get("tagName")))
        if body.get("tags") is not None and sorted(body["tags"]) != sorted(row_tags):
            return 400, {"error": "tags must name exactly the criteria rows' tags", "code": "tags_mismatch"}
        quote_id = body.get("quoteId")
        if self.price_usd > 0:
            if not quote_id:
                quote_id = f"q{len(self.quotes) + 1}"
                self.quotes[quote_id] = "draft"
                return {"proposed": False, "requiresApproval": True,
                        "quote": {"id": quote_id, "amountEstUsd": self.price_usd, "docCount": self.doc_count,
                                  "sku": "review_fast", "expiresAt": "2026-10-01T00:00:00Z"}}
            if self.quotes.get(quote_id) == "draft" and body.get("userConfirmed") is True:
                self.quotes[quote_id] = "approved"
            if self.quotes.get(quote_id) != "approved":
                return 400, {"error": "billing: the quote is not approved; ask the person and send userConfirmed"}
            self.quotes[quote_id] = "consumed"
        proposal_id = f"00000000-0000-4000-8000-{len(self.proposals) + 501:012d}"
        version = self.sets[set_id]["currentVersion"] if set_id else None
        proposal = {"id": proposal_id, "kind": "review_run", "status": "pending",
                    "name": body.get("name") or "Review & tag", "tags": body.get("tags") or row_tags, "newTags": [],
                    "criteriaCount": len(rows),
                    "criteriaSet": {"id": set_id, "name": self.sets[set_id]["name"],
                                    "version": body.get("criteriaSetVersion") or version} if set_id else None,
                    "tier": body.get("tier") or "fast", "autoApply": body.get("applyTags") is not False,
                    "threadId": body.get("threadId"), "firstPass": None}
        self.proposals.append({**proposal, "request": body})
        return 201, {"proposed": True, "proposal": proposal, "estimatedDocCount": self.doc_count,
                     "estimate": {"lane": "cloud", "totalUsd": self.price_usd}, "parallel": [],
                     "postCardId": "00000000-0000-4000-8000-00000000c001" if body.get("threadId") else None}

    # -- proposals: launch, withdraw, read -------------------------------------------------------
    def reply(self, proposal_id: str) -> None:
        """The person answered "launch it" in the proposal's thread, after the card."""
        self.replied.add(proposal_id)

    def _proposal(self, proposal_id: str) -> Any:
        return next((p for p in self.proposals if p["id"] == proposal_id), None)

    def _with_proposal(self, req: Recorded, fn: Callable[[Dict[str, Any], Recorded], Any]) -> Any:
        if not req.headers.get("x-litkit-acting-user") and req.method == "POST":
            return 403, {"error": "acting_user_required"}
        found = self._proposal(req.path.split("/")[5])
        return fn(found, req) if found else (404, {"error": "not_found"})

    def _unrouted(self, req: Recorded, fn: Callable[[Dict[str, Any], Recorded], Any]) -> Any:
        if not self.launch_route:
            return 404, "<!DOCTYPE html><html><body>404: This page could not be found.</body></html>", {
                "Content-Type": "text/html; charset=utf-8"}
        return self._with_proposal(req, fn)

    def _with_job(self, job_id: str, fn: Callable[[Dict[str, Any]], Any]) -> Any:
        found = self.jobs.get(job_id)
        return fn(found) if found else (404, {"error": "not_found"})

    @staticmethod
    def _launched(p: Dict[str, Any]) -> Any:
        return {"reviewJobId": p["launchedReviewJobId"], "scopeDocCount": p["launchedDocCount"]} \
            if p.get("launchedReviewJobId") else None

    def _read(self, p: Dict[str, Any], _req: Recorded) -> Any:
        return {"proposal": {"id": p["id"], "kind": "review_run", "status": p["status"],
                             "payload": {"name": p["name"], "tags": p["tags"]}, "threadId": p["threadId"],
                             "createdAt": "2026-10-01T12:00:00Z", "reviewedAt": None,
                             "launched": self._launched(p)}}

    def _launch(self, p: Dict[str, Any], req: Recorded) -> Any:
        # The app reads the post the turn answers from the turn grant it minted; it never takes the host's word.
        if not req.headers.get("x-litkit-turn-grant"):
            return 403, {"error": "turn_grant_required",
                         "message": "A launch needs the turn grant LitKit sent with this turn."}
        body = req.json()
        if not isinstance(body, dict) or set(body) != {"threadId"}:
            return 400, {"error": "invalid input"}
        if p["status"] == "accepted" and p.get("launchedReviewJobId"):
            return {"ok": True, "alreadyLaunched": True, "launched": self._launched(p)}
        if p["status"] != "pending":
            return 409, {"error": "already reviewed", "status": p["status"], "launched": self._launched(p),
                         "message": f"This proposal is {p['status']}; it can no longer be launched."}
        if not p["threadId"] or p["threadId"] != body["threadId"]:
            return 409, {"error": "not_this_thread",
                         "message": "This run was proposed in another thread; launch it from there."}
        if p["id"] not in self.replied:
            return 409, {"error": "no_reply_after_card",
                         "message": "The person has not answered in this thread since the card was posted. Ask "
                                    "whether to launch, and launch only after they say yes."}
        job_id = f"00000000-0000-4000-8000-{len(self.jobs) + 901:012d}"
        self.jobs[job_id] = {"id": job_id, "status": "running", "name": p["name"], "totalDocs": self.doc_count,
                             "processedDocs": 0, "createdBy": req.headers["x-litkit-acting-user"]}
        p.update(status="accepted", launchedReviewJobId=job_id, launchedDocCount=self.doc_count)
        return {"ok": True, "status": "accepted", "launched": self._launched(p), "applied": None}

    def _withdraw(self, p: Dict[str, Any], _req: Recorded) -> Any:
        if p["status"] != "pending":
            return 409, {"error": "already reviewed", "status": p["status"],
                         "message": f"This proposal is {p['status']}; only a pending proposal can be withdrawn."}
        p["status"] = "rejected"
        return {"ok": True, "status": "rejected"}
