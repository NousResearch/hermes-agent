#!/usr/bin/env python3
"""qa_gateway.py — the QA machine every reconnect test card drives.

Runs against the TEST Hermes (:10119) with zero privileged access: HTTP via the
standard library, one WebSocket library, and the test profile's own dashboard
credentials. There is no SSH, no systemctl, no production access anywhere in
this file, and it never writes to the server.

Commands
--------
  ticket   Sign in, mint a ws ticket, report elapsed ms. Drives both mint
           endpoints the app uses: POST /api/webapp/socket-ticket (the Webapp
           page's one-use ticket) and POST /api/auth/ws-ticket (the session's).
           The first exists only on the gateway's Webapp listener surface; the
           dashboard here answers 404 there, and open/phases use the one that
           answers — named in every report.
  open     Sign in, mint a ticket, open /api/ws on it, report elapsed ms for
           the mint and for the upgrade.
  phases   Run the reconnect sequence N times and print the phase table with
           p50/p95 (names per C1's schema in
           apps/desktop/src/store/desktop-metrics.ts):
              detect_ms pre_ipc_ms dial_ms ready_ms replay_barrier_ms usable_ms
           A gateway-side driver can only observe part of that schema: it
           closes the socket on purpose (detect_ms = 0.0 by construction) and
           it has no client replay to wait behind. Those columns print n/a and
           are filled in by C1's client-side instrumentation; the measurable
           ones are real measurements of the test instance.
           With --pre-probe PATH, pre_ipc_ms additionally models the Desktop
           revalidation leg the reconnect path pays before the mint (#C3): one
           GET of PATH folded in front of the ticket round trip. Same
           instance, same driver, two paths = an honest before/after of which
           endpoint the client probes (/api/status 0.18-1.9 s vs /api/health
           single-digit ms).
  handshake
           Count the post-upgrade handshake's APPLICATION ROUND TRIPS per
           upgrade (--cycles, --watermarks): a backend advertising
           `resumed_handshake` spends ONE `client.hello` (answered with
           capabilities + skin etag + every session's replay + the open
           requests), anything else spends the legacy sequence — one
           `client.capabilities` plus one `session.events.since` per held
           cursor. Same driver, same instance, before/after a deploy = the wire
           cost a reconnect actually charges, which is what the program states
           ("1 ticket POST + 1 upgrade + 1 hello/resumed exchange").

Credentials
-----------
Read from the test profile's .env (default
~/.hermes/profiles/hermes-test/.env), never from the command line, the repo or
a card:
    HERMES_DASHBOARD_BASIC_AUTH_USERNAME
    HERMES_DASHBOARD_BASIC_AUTH_PASSWORD
Environment variables of the same name win when set.

Examples
--------
    scripts/run-in-hermes-env python deploy/qa_gateway.py ticket
    scripts/run-in-hermes-env python deploy/qa_gateway.py open
    scripts/run-in-hermes-env python deploy/qa_gateway.py phases --cycles 20
    # over Tailscale, from any QA machine on the tailnet:
    ... deploy/qa_gateway.py --base-url https://llmbox.tail291e4.ts.net:10119 phases --cycles 20
"""

from __future__ import annotations

import argparse
import asyncio
import http.cookiejar
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_BASE_URL = "http://127.0.0.1:10119"
DEFAULT_ENV_FILE = os.path.expanduser("~/.hermes/profiles/hermes-test/.env")
AUTH_PROVIDER = "basic"
PHASES = ("detect_ms", "pre_ipc_ms", "dial_ms", "ready_ms", "replay_barrier_ms", "usable_ms")
#: The two ticket endpoints, in the order the app uses them. Only a surface whose
#: SurfacePolicy sets ``gateway_ticket`` (the gateway's Webapp listener) serves the
#: first; the dashboard answers 404 there and its own ticket is the second.
TICKET_ENDPOINTS = (
    ("POST /api/webapp/socket-ticket",
     "/api/webapp/socket-ticket?" + urllib.parse.urlencode({"socket": "gateway"})),
    ("POST /api/auth/ws-ticket", "/api/auth/ws-ticket"),
)
#: How long the socket waits for anything else the server wants to say.
READY_BUDGET_S = 2.0
RPC_BUDGET_S = 5.0


# --------------------------------------------------------------------------
# credentials
# --------------------------------------------------------------------------
def read_env_file(path: str) -> dict[str, str]:
    """Minimal dotenv reader (KEY=VALUE, optional quotes, # comments)."""
    values: dict[str, str] = {}
    try:
        with open(path, encoding="utf-8-sig") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, raw = line.partition("=")
                key = key.strip()
                # strip an optional `export ` prefix
                if key.startswith("export "):
                    key = key[len("export "):].strip()
                value = raw.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                values[key] = value
    except FileNotFoundError:
        raise SystemExit(f"qa_gateway: env file not found: {path}")
    return values


def credentials(env_file: str) -> tuple[str, str]:
    file_values = read_env_file(env_file)
    username = os.environ.get("HERMES_DASHBOARD_BASIC_AUTH_USERNAME") or file_values.get(
        "HERMES_DASHBOARD_BASIC_AUTH_USERNAME", "")
    password = os.environ.get("HERMES_DASHBOARD_BASIC_AUTH_PASSWORD") or file_values.get(
        "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD", "")
    if not username or not password:
        raise SystemExit(
            "qa_gateway: no dashboard credentials. Set HERMES_DASHBOARD_BASIC_AUTH_USERNAME and "
            "HERMES_DASHBOARD_BASIC_AUTH_PASSWORD, or point --env-file at the test profile's .env "
            f"(default {DEFAULT_ENV_FILE}). Credentials are never passed on the command line."
        )
    return username, password


# --------------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------------
class Http:
    """Cookie-carrying JSON client for the gated test dashboard."""

    def __init__(self, base_url: str, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        scheme = urllib.parse.urlparse(self.base_url).scheme
        self._jar = http.cookiejar.CookieJar()
        handlers: list[urllib.request.BaseHandler] = [urllib.request.HTTPCookieProcessor(self._jar)]
        if scheme == "https":
            # The tailnet URL is a real cert; the loopback default is plain http.
            handlers.append(urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        self._opener = urllib.request.build_opener(*handlers)

    def send(self, method: str, path: str, body: dict | None = None) -> tuple[int, str]:
        """Raw HTTP: (status, body text). A non-2xx status is data, not an exception —
        surface-specific endpoints answer 404 where another surface mints."""
        data = None
        headers = {"Accept": "application/json"}
        if body is not None:
            data = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with self._opener.open(req, timeout=self.timeout) as resp:
                return resp.status, resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read().decode("utf-8", "replace")
        except urllib.error.URLError as exc:
            raise SystemExit(f"qa_gateway: {method} {path} -> {exc.reason}") from exc

    def request(self, method: str, path: str, body: dict | None = None) -> dict:
        status, text = self.send(method, path, body)
        if status != 200:
            raise SystemExit(f"qa_gateway: {method} {path} -> HTTP {status}: {text[:300]}")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {"raw": text}

    def login(self, username: str, password: str) -> None:
        self.request("POST", "/auth/password-login",
                     {"provider": AUTH_PROVIDER, "username": username, "password": password})

    def _mint(self, path: str) -> tuple[str, float, int]:
        """One ticket mint: (ticket, elapsed_ms, status); ticket is '' when refused."""
        t0 = time.perf_counter()
        status, text = self.send("POST", path)
        elapsed = (time.perf_counter() - t0) * 1000.0
        if status != 200:
            return "", elapsed, status
        try:
            return str(json.loads(text).get("ticket", "")), elapsed, status
        except json.JSONDecodeError:
            return "", elapsed, status

    def mint_ticket(self) -> tuple[str, float, str]:
        """The ticket for ONE ``/api/ws`` dial on this server: (ticket, ms, source).

        ``POST /api/webapp/socket-ticket`` is the Webapp page's one-use ticket and
        exists only on a surface whose SurfacePolicy sets ``gateway_ticket`` — the
        gateway's Webapp listener (``hermes webapp``), not this dashboard. The
        dashboard surface answers 404 there; its own ticket is
        ``POST /api/auth/ws-ticket`` (session cookie, one per socket). Whichever
        answers is returned, named, so every measurement says where it came from.
        """
        webapp_path = "/api/webapp/socket-ticket?" + urllib.parse.urlencode({"socket": "gateway"})
        ticket, elapsed, _status = self._mint(webapp_path)
        if ticket:
            return ticket, elapsed, "webapp"
        ticket, elapsed, _status = self._mint("/api/auth/ws-ticket")
        if ticket:
            return ticket, elapsed, "auth"
        raise SystemExit(
            "qa_gateway: neither ticket endpoint minted a ticket. Is the login cookie valid "
            "(credentials) and the listener up?")

    def probe_ms(self, path: str) -> float:
        """One reachability probe: elapsed ms of GET <path>. Any HTTP answer
        counts (a 404 still proves the listener is up); a transport failure is
        the outage the caller's reconnect exists to survive, so send() surfaces
        it as its own fatal."""
        t0 = time.perf_counter()
        self.send("GET", path)
        return (time.perf_counter() - t0) * 1000.0

    def ws_url(self, ticket: str) -> str:
        parsed = urllib.parse.urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        query = urllib.parse.urlencode({"ticket": ticket})
        return urllib.parse.urlunparse((scheme, parsed.netloc, "/api/ws", "", query, ""))


# --------------------------------------------------------------------------
# WebSocket
# --------------------------------------------------------------------------
def _load_websockets():
    try:
        import websockets  # import here so --help works without it
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "qa_gateway: the 'websockets' package is required (run inside the checkout's "
            "environment: scripts/run-in-hermes-env python deploy/qa_gateway.py ...)"
        ) from exc
    return websockets


def phase_values(rows: list[dict[str, float | None]], name: str) -> list[float]:
    """One phase's measured values across cycles (n/a rows are skipped)."""
    out: list[float] = []
    for row in rows:
        value = row[name]
        if value is not None:
            out.append(value)
    return out


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    if len(ordered) == 1:
        return ordered[0]
    index = q * (len(ordered) - 1)
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1f}"


def print_table(rows: list[dict[str, float | None]], cycles: int, sources: set[str],
                pre_probe: str | None = None) -> None:
    header = "cycle".ljust(6) + "".join(name.rjust(19) for name in PHASES)
    print(header)
    print("-" * len(header))
    for i, row in enumerate(rows, start=1):
        print(str(i).ljust(6) + "".join(fmt(row[name]).rjust(19) for name in PHASES))
    print("-" * len(header))
    for label, q in (("p50", 0.50), ("p95", 0.95)):
        cells = []
        for name in PHASES:
            values = phase_values(rows, name)
            cells.append(fmt(percentile(values, q) if values else None).rjust(19))
        print(label.ljust(6) + "".join(cells))
    print(f"\nn={len(rows)} cycles on {cycles} requested; all values are milliseconds.")
    print(f"ticket source: {', '.join(sorted(sources)) or 'n/a'}")
    if pre_probe:
        print(f"pre_ipc_ms here = GET {pre_probe} + ticket mint (the Desktop revalidation leg, #C3).")
    print("detect_ms is 0.0 by construction here (this driver closes the socket itself);")
    print("replay_barrier_ms is n/a because a gateway-side driver has no replay to wait behind.")
    print("Both are filled by the client-side instrumentation in C1.")


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------
def cmd_ticket(http: Http, args) -> int:
    """Both mint endpoints, timed: what this surface serves and what it refuses."""
    rows = []
    for label, path in TICKET_ENDPOINTS:
        ticket, ms, status = http._mint(path)
        rows.append({"endpoint": label, "status": status, "ms": round(ms, 3),
                     "ticket_chars": len(ticket)})
    used = next((row for row in rows if row["ticket_chars"]), None)
    if args.json:
        print(json.dumps({"endpoints": rows,
                          "used": used["endpoint"] if used else None}, indent=2))
    else:
        for row in rows:
            note = (f"ticket {row['ticket_chars']} chars" if row["ticket_chars"]
                    else f"HTTP {row['status']} — not served on this surface")
            print(f"{row['endpoint']:<34}{row['ms']:9.1f} ms  ({note})")
        print(f"\nticket source for open/phases: {used['endpoint'] if used else 'NONE'}")
    return 0 if used else 1


def cmd_open(http: Http, args) -> int:
    websockets = _load_websockets()
    ticket, mint_ms, source = http.mint_ticket()
    if not ticket:
        print("qa_gateway: the server returned no ticket", file=sys.stderr)
        return 1
    url = http.ws_url(ticket)
    t0 = time.perf_counter()

    async def dial() -> float:
        async with websockets.connect(url, open_timeout=args.timeout,
                                      close_timeout=2, ping_interval=None) as ws:
            elapsed = (time.perf_counter() - t0) * 1000.0
            await ws.close()
            return elapsed

    open_ms = asyncio.run(dial())
    result = {"ticket_ms": round(mint_ms, 3), "ws_open_ms": round(open_ms, 3),
              "ticket_source": source}
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"{source} ticket{'':<9}{mint_ms:9.1f} ms")
        print(f"ws  /api/ws (upgrade){open_ms:7.1f} ms")
    return 0


def cmd_phases(http: Http, args) -> int:
    _load_websockets()
    rows: list[dict[str, float | None]] = []
    sources: set[str] = set()
    previous = None
    for _ in range(args.cycles):
        # Each cycle reuses the same view of the server; the close that starts
        # the next cycle is the previous socket's, exactly as a browser's would be.
        current = ProbeSocket(http, args.timeout, args.pre_probe)
        asyncio.run(current.round(previous))
        rows.append(current.phase)
        sources.add(current.source)
        previous = current
    if args.json:
        summary = {}
        for name in PHASES:
            values = phase_values(rows, name)
            summary[name] = {
                "p50": round(percentile(values, 0.50), 3) if values else None,
                "p95": round(percentile(values, 0.95), 3) if values else None,
                "n": len(values),
            }
        print(json.dumps({"cycles": args.cycles, "rows": rows, "summary": summary,
                          "ticket_sources": sorted(sources),
                          "pre_probe": args.pre_probe or None}, indent=2))
    else:
        print_table(rows, args.cycles, sources, args.pre_probe)
    return 0


async def _await_response(ws, request_id: int, budget: float) -> dict | None:
    """The response frame for *request_id* within *budget* seconds, or None (events interleave)."""
    deadline = time.perf_counter() + budget
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return None
        try:
            raw = await asyncio.wait_for(ws.recv(), remaining)
        except TimeoutError:
            return None
        try:
            message = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(message, dict) and message.get("id") == request_id:
            return message


async def _handshake_cycle(http: Http, args, ticket: str, source: str) -> dict:
    """One upgrade's post-upgrade handshake, counted: what the backend asks the client to spend.

    ``--watermarks`` stands in for the sessions a reconnecting client holds: the fold carries every
    one of them in its single request, while the legacy sequence needs one ``session.events.since``
    per session — so the column that matters is ``round_trips``, not the milliseconds alone.
    """
    websockets = _load_websockets()
    url = http.ws_url(ticket)
    watermarks = {f"w{i}": 1 for i in range(args.watermarks)}
    row: dict = {"source": source, "fold": False, "round_trips": None, "handshake_ms": None,
                 "ready_ms": None, "sessions_replayed": None, "skin_etag": None, "error": None}
    async with websockets.connect(url, open_timeout=args.timeout, close_timeout=2,
                                  ping_interval=None) as ws:
        t0 = time.perf_counter()
        try:
            ready = json.loads(await asyncio.wait_for(ws.recv(), READY_BUDGET_S))
        except TimeoutError:
            row["error"] = "no gateway.ready frame"
            return row
        payload = (ready.get("params") or {}).get("payload") or {}
        row["ready_ms"] = (time.perf_counter() - t0) * 1000.0
        row["fold"] = payload.get("resumed_handshake") is True
        t_rpc = time.perf_counter()
        if row["fold"]:
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "client.hello",
                                      "params": {"capabilities": {"server_requests": True},
                                                 "watermarks": watermarks}}))
            answer = await _await_response(ws, 1, RPC_BUDGET_S)
            row["handshake_ms"] = (time.perf_counter() - t_rpc) * 1000.0
            row["round_trips"] = 1
            row["error"] = ((answer or {}).get("error") or {}).get("message")
            result = (answer or {}).get("result") or {}
            row["sessions_replayed"] = len(result.get("per_session_replay") or [])
            row["skin_etag"] = bool(result.get("skin_etag"))
        else:
            # No advertisement: the sequence the fold replaces — the capabilities probe, then one
            # replay request per watermarked session (a real client sends both, from its cursors).
            await ws.send(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "client.capabilities",
                                      "params": {"server_requests": True}}))
            answer = await _await_response(ws, 2, RPC_BUDGET_S)
            row["handshake_ms"] = (time.perf_counter() - t_rpc) * 1000.0
            row["round_trips"] = 1 + len(watermarks)
            row["error"] = ((answer or {}).get("error") or {}).get("message")
    return row


def cmd_handshake(http: Http, args) -> int:
    """The wire cost of a reconnect's post-upgrade handshake, per cycle, with the round-trip count.

    The program target is that after the fold a reconnect costs ONE ticket batch POST + one upgrade +
    one hello/resumed exchange. This command measures what the deployed build actually charges: a
    backend advertising ``resumed_handshake`` spends ONE round trip (``client.hello``, answered with
    capabilities + skin etag + every session's replay + the open requests); anything else spends the
    legacy sequence — ``client.capabilities`` then one ``session.events.since`` per watermark.
    """
    _load_websockets()
    rows = []
    for _ in range(args.cycles):
        ticket, _mint_ms, source = http.mint_ticket()
        if not ticket:
            raise SystemExit("qa_gateway: the server returned no ticket")
        rows.append(asyncio.run(_handshake_cycle(http, args, ticket, source)))
    folds = [row for row in rows if row["fold"]]
    round_trips = [float(row["round_trips"]) for row in rows if row["round_trips"] is not None]
    handshake_ms = [row["handshake_ms"] for row in rows if row["handshake_ms"] is not None]
    if args.json:
        print(json.dumps({"cycles": args.cycles, "watermarks": args.watermarks, "rows": rows,
                          "fold": len(folds) == len(rows) and bool(rows),
                          "round_trips_p50": percentile(round_trips, 0.50) if round_trips else None,
                          "handshake_ms_p50": percentile(handshake_ms, 0.50) if handshake_ms else None,
                          "handshake_ms_p95": percentile(handshake_ms, 0.95) if handshake_ms else None},
                         indent=2))
        return 0
    header = "cycle".ljust(6) + "fold".rjust(7) + "round_trips".rjust(13) + "ready_ms".rjust(11) \
        + "handshake_ms".rjust(15) + "replayed".rjust(10) + "skin_etag".rjust(11)
    print(header)
    print("-" * len(header))
    for index, row in enumerate(rows, start=1):
        print(str(index).ljust(6) + ("yes" if row["fold"] else "no").rjust(7)
              + fmt(row["round_trips"]).rjust(13) + fmt(row["ready_ms"]).rjust(11)
              + fmt(row["handshake_ms"]).rjust(15) + fmt(row["sessions_replayed"]).rjust(10)
              + ("yes" if row["skin_etag"] else "no").rjust(11))
    print("-" * len(header))
    if handshake_ms:
        print(f"{'p50'.ljust(6)}{'':7}{'':13}{'':11}{percentile(handshake_ms, 0.50):.1f}".ljust(len(header)))
        print(f"{'p95'.ljust(6)}{'':7}{'':13}{'':11}{percentile(handshake_ms, 0.95):.1f}".ljust(len(header)))
    named = {row["error"] for row in rows if row["error"]}
    for error in sorted(named):
        print(f"answer error: {error}")
    print(f"\nn={len(rows)} cycles on {args.cycles} requested; --watermarks models {args.watermarks} "
          f"held session cursor(s); all values are milliseconds.")
    if folds and len(folds) == len(rows):
        print("wire cost per reconnect: 1 ticket POST + 1 upgrade + 1 hello/resumed round trip.")
    elif rows and not folds:
        print("wire cost per reconnect: 1 ticket POST + 1 upgrade + 1 client.capabilities round trip "
              f"+ 1 session.events.since per held cursor ({args.watermarks} here).")
    else:
        print("mixed advertisement: some sockets advertise the fold, some do not — read the table.")
    return 0


class ProbeSocket:
    """One reconnect cycle, holding the live socket so the next cycle closes a real one."""

    def __init__(self, http: Http, timeout: float, pre_probe_path: str | None = None):
        self._http = http
        self._timeout = timeout
        self._pre_probe_path = pre_probe_path
        self.phase: dict[str, float | None] = {name: None for name in PHASES}
        self.source = ""
        self._ws = None

    async def round(self, previous) -> None:
        websockets = _load_websockets()
        # The cycle starts at the close instant. A gateway-side driver closes
        # deliberately, so it knows it instantly: detect_ms is 0.0 here, and only
        # the client (C1) can measure a real detection delay.
        self.phase["detect_ms"] = 0.0
        if previous is not None:
            await previous.close()
        # --pre-probe models the Desktop revalidation leg (#C3): the client's
        # reconnect GETs a reachability endpoint BEFORE minting, and pre_ipc_ms
        # is that whole chain. Without the flag the column stays what it was:
        # the mint round trip alone (the spine's gateway-side contract).
        pre_probe_ms = (self._http.probe_ms(self._pre_probe_path)
                        if self._pre_probe_path else 0.0)
        ticket, pre_ipc_ms, self.source = self._http.mint_ticket()
        self.phase["pre_ipc_ms"] = pre_ipc_ms + pre_probe_ms
        if not ticket:
            raise SystemExit("qa_gateway: the server returned no ticket")
        url = self._http.ws_url(ticket)
        t0 = time.perf_counter()
        ws = await asyncio.wait_for(websockets.connect(url, open_timeout=self._timeout,
                                                      close_timeout=2, ping_interval=None),
                                    self._timeout)
        self._ws = ws
        self.phase["dial_ms"] = (time.perf_counter() - t0) * 1000.0
        # ready_ms: the first frame the server offers after the upgrade.
        try:
            await asyncio.wait_for(ws.recv(), READY_BUDGET_S)
            self.phase["ready_ms"] = (time.perf_counter() - t0) * 1000.0
        except TimeoutError:
            self.phase["ready_ms"] = None  # nothing unsolicited; the RPC below still measures
        # usable_ms: one real application round trip on the open socket.
        request_id = int(time.time() * 1000) % 1_000_000
        t_rpc = time.perf_counter()
        await ws.send(json.dumps({"jsonrpc": "2.0", "id": request_id,
                                  "method": "client.capabilities",
                                  "params": {"capabilities": {}}}))
        deadline = t_rpc + RPC_BUDGET_S
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                break
            try:
                raw = await asyncio.wait_for(ws.recv(), remaining)
            except TimeoutError:
                break
            try:
                message = json.loads(raw)
            except (TypeError, ValueError):
                continue
            if isinstance(message, dict) and message.get("id") == request_id:
                self.phase["usable_ms"] = (time.perf_counter() - t_rpc) * 1000.0
                break
        # replay_barrier_ms needs a client with a replay watermark; a gateway-side
        # ticket carries no replay state, so C1's client instrumentation fills it.
        self.phase["replay_barrier_ms"] = None

    async def close(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass


# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qa_gateway.py",
        description="Drive the TEST Hermes (:10119) through the reconnect sequence.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("QA_BASE_URL", DEFAULT_BASE_URL),
                        help=f"test instance base URL (default {DEFAULT_BASE_URL})")
    parser.add_argument("--env-file", default=os.environ.get("QA_ENV_FILE", DEFAULT_ENV_FILE),
                        help="env file holding the dashboard credentials")
    parser.add_argument("--timeout", type=float, default=10.0, help="per-operation timeout, seconds")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    sub = parser.add_subparsers(dest="command", required=True)

    def sub_json(p: argparse.ArgumentParser) -> None:
        # The documented position is AFTER the subcommand (RUNBOOK.md and the C1
        # contract: 'phases --cycles 20 --json'). default=SUPPRESS: when the flag
        # is absent the subparser writes nothing to the namespace, so a top-level
        # '--json phases' still works; either position sets args.json.
        p.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                       help="machine-readable output")

    ticket = sub.add_parser("ticket", help="mint a ws ticket and report elapsed ms")
    sub_json(ticket)
    open_ = sub.add_parser("open", help="open /api/ws on a fresh ticket and report elapsed ms")
    sub_json(open_)
    phases = sub.add_parser("phases", help="run the reconnect sequence and print the phase table")
    phases.add_argument("--cycles", type=int, default=20, help="reconnect cycles (default 20)")
    phases.add_argument("--pre-probe", dest="pre_probe", metavar="PATH", default=None,
                        help="GET PATH before each mint and fold it into pre_ipc_ms "
                             "(models the Desktop revalidation leg, #C3)")
    sub_json(phases)
    handshake = sub.add_parser(
        "handshake",
        help="count the post-upgrade handshake's application round trips (the fold's wire cost)")
    handshake.add_argument("--cycles", type=int, default=20, help="upgrades to count (default 20)")
    handshake.add_argument("--watermarks", type=int, default=4,
                           help="session cursors the client holds (the legacy sequence spends one "
                                "replay round trip each; the fold carries them all in one)")
    sub_json(handshake)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    username, password = credentials(args.env_file)
    http = Http(args.base_url, timeout=args.timeout)
    http.login(username, password)
    if args.command == "ticket":
        return cmd_ticket(http, args)
    if args.command == "open":
        return cmd_open(http, args)
    if args.command == "phases":
        return cmd_phases(http, args)
    if args.command == "handshake":
        return cmd_handshake(http, args)
    raise SystemExit(f"qa_gateway: unknown command {args.command}")


if __name__ == "__main__":
    sys.exit(main())
