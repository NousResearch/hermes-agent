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
    """LitKit's review-structure routes for one matter, with state: criteria sets and their versions,
    work sets, ``review-jobs/propose`` with the review_run billing protocol, and accept-all-tags.

    Billing follows litkit-app's ``review_run``: with ``price_usd > 0``, a propose without a quoteId
    returns ``requiresApproval`` and a draft quote and proposes nothing; a propose with that quoteId
    and ``userConfirmed: true`` approves the quote and queues the proposal. A quoteId the person has
    not approved answers 402."""

    def __init__(self, fake: FakeLitKit, matter_id: str = MATTER_ID, *, price_usd: float = 0.0,
                 doc_count: int = 1234) -> None:
        self.price_usd = price_usd
        self.doc_count = doc_count
        self.sets: Dict[str, Dict[str, Any]] = {}
        self.quotes: Dict[str, str] = {}  # quote id -> draft | approved | consumed
        self.proposals: List[Dict[str, Any]] = []
        self.work_sets = [{"id": "00000000-0000-4000-8000-00000000a001", "name": "Crowder emails", "docCount": 812}]
        base = f"/api/matters/{matter_id}"
        sid = r"(?:[0-9a-f-]{36}|builtin:[a-z0-9_-]+)"
        fake.route("GET", rf"{base}/criteria-sets", lambda r: {"matter": None, "mine": list(self.sets.values()),
                                                                 "firm": [], "builtIn": []})
        fake.route("POST", rf"{base}/criteria-sets", self._create)
        fake.route("GET", rf"{base}/criteria-sets/{sid}", lambda r: self._get(r.path.split("/")[-1]))
        fake.route("PATCH", rf"{base}/criteria-sets/{sid}", self._update)
        fake.route("POST", rf"{base}/criteria-sets/{sid}/publish", self._publish)
        fake.route("GET", rf"{base}/criteria-sets/{sid}/versions",
                   lambda r: self._with_set(r.path.split("/")[-2], lambda s: {"versions": s["versions"]}))
        fake.route("GET", rf"{base}/work-sets", lambda r: {"workSets": self.work_sets})
        fake.route("POST", rf"{base}/review-jobs/propose", self._propose)
        fake.route("POST", rf"{base}/review-jobs/[^/]+/accept-all-tags",
                   lambda r: {"ok": True, "accepted": 17, "jobId": r.path.split("/")[-2]})

    def _with_set(self, set_id: str, fn: Callable[[Dict[str, Any]], Any]) -> Any:
        found = self.sets.get(set_id)
        return fn(found) if found else (404, {"error": "not_found"})

    def _get(self, set_id: str) -> Any:
        return self._with_set(set_id, lambda s: {"set": {k: s[k] for k in ("id", "name", "currentVersion",
                                                                            "publishedVersion")},
                                                 "criteria": s["versions"][-1]["criteria"]})

    def _create(self, req: Recorded) -> Any:
        body = req.json()
        set_id = f"00000000-0000-4000-8000-{len(self.sets) + 1:012d}"
        self.sets[set_id] = {"id": set_id, "name": body["name"], "currentVersion": 1, "publishedVersion": None,
                             "versions": [{"version": 1, "criteria": body["criteria"], "changeNote": ""}]}
        return 201, {"set": {"id": set_id, "name": body["name"], "currentVersion": 1}, "version": 1,
                     "criteria": body["criteria"]}

    def _update(self, req: Recorded) -> Any:
        body = req.json()

        def bump(s: Dict[str, Any]) -> Any:
            s["currentVersion"] += 1
            s["versions"].append({"version": s["currentVersion"], "criteria": body["criteria"],
                                  "changeNote": body.get("changeNote", "")})
            return {"set": {"id": s["id"], "currentVersion": s["currentVersion"]}, "version": s["currentVersion"]}
        return self._with_set(req.path.split("/")[-1], bump)

    def _publish(self, req: Recorded) -> Any:
        def publish(s: Dict[str, Any]) -> Any:
            s["publishedVersion"] = s["currentVersion"]
            return {"ok": True, "set": {"id": s["id"]}, "version": s["currentVersion"]}
        return self._with_set(req.path.split("/")[-2], publish)

    def _propose(self, req: Recorded) -> Any:
        body = req.json()
        set_id = body.get("criteriaSetId")
        if set_id and set_id not in self.sets:
            return 404, {"error": "criteria_set_not_found"}
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
                return 402, {"error": "quote_not_approved", "quoteId": quote_id}
            self.quotes[quote_id] = "consumed"
        proposal = {"proposalId": f"00000000-0000-4000-8000-{len(self.proposals) + 501:012d}",
                    "estimatedCount": self.doc_count, "status": "pending",
                    "estimate": {"amountEstUsd": self.price_usd, "docCount": self.doc_count},
                    "criteriaSetVersion": self.sets[set_id]["currentVersion"] if set_id else None,
                    "tags": body["tags"]}
        self.proposals.append({**proposal, "request": body})
        return proposal
