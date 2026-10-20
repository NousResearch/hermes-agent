#!/usr/bin/env python3
"""Dead-localhost mock media stack for the talaria test suite.

Binds 127.0.0.1 on the 7 service ports and serves synthetic canned
fixtures. NEVER contacts real services: tests run under a dead-proxy
egress trap (proxy vars -> closed loopback port 9, no_proxy covers
127.0.0.1/localhost), so any request that escapes loopback dies instantly.

Auth model (mirrors talaria.core):
  - X-Api-Key header (sonarr/radarr/lidarr/prowlarr/overseerr; bazarr
    accepts the capital X-API-Key spelling too, API parity)
  - jellyfin: `Authorization: MediaBrowser ..., Token="<key>"`
    (scheme case-insensitive). Expected key = JELLYFIN_KEY env
    at server start (shared with tests by construction).
  - unauthenticated routes (key=False): arr /initialize.json probes,
    jellyfin /System/Info/Public and /System/Ping.

Jellyfin negative traps: any request carrying an
api_key=/ApiKey= query param or an X-Emby-Token header is answered 400
and logged as a violation — the skill must never emit those legacy channels.
The /emby/… route prefix matches nothing and 404s (removed in 12.0).

Routing: per-port ordered list of (path regex, fixture, needs_key,
method); first match wins. Matched + wrong key -> 401 *arr-shaped error.
No match -> 404. Every request appends one JSONL log line with header
NAMES only (never values — redaction by construction) plus the decoded
body for write-path assertions.

Usage:  python3 mock_server.py [--fixtures-dir DIR] [--log PATH]
Emits {"event": "ready"} as the first log line once every port is bound.
Terminate with SIGTERM/SIGINT.
"""

import argparse
import json
import os
import re
import signal
import threading
import os
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DEFAULTS = {
    "fixtures-dir": Path(__file__).resolve().parent / "fixtures",
    # default log lives OUTSIDE the repo (tempdir); the harness always
    # passes --log explicitly with its per-run path
    "log": Path(os.environ.get(
        "TALARIA_MOCK_LOG",
        Path(tempfile.gettempdir()) / "talaria-mock-request-log.jsonl")),
}

# the 7 in-scope services on their default ports
PORTS = {
    8989: "sonarr", 7878: "radarr", 8686: "lidarr", 9696: "prowlarr",
    6767: "bazarr", 5055: "overseerr", 8096: "jellyfin",
}

EXPECTED_KEY_ENV = {
    "sonarr": "SONARR_KEY", "radarr": "RADARR_KEY", "lidarr": "LIDARR_KEY",
    "prowlarr": "PROWLARR_KEY", "bazarr": "BAZARR_KEY",
    "overseerr": "OVERSEERR_KEY", "jellyfin": "JELLYFIN_KEY",
}


def R(pattern, fixture, key=True, method=None):
    # method: optional HTTP verb constraint; None = any.
    return (re.compile(pattern), fixture, key, method)


ROUTES = {
    "sonarr": [
        R(r"^/initialize\.json$", "sonarr__initialize", key=False),
        R(r"^/api/v3/system/status$", "sonarr__system_status"),
        R(r"^/api/v3/health$", "sonarr__health"),
        R(r"^/api/v3/wanted/missing", "sonarr__wanted_missing"),
        R(r"^/api/v3/calendar", "sonarr__calendar"),
        R(r"^/api/v3/history", "sonarr__history"),
        R(r"^/api/v3/queue$", "sonarr__queue"),
        R(r"^/api/v3/rootfolder$", "sonarr__rootfolder"),
        R(r"^/api/v3/qualityprofile", "sonarr__qualityprofile"),
        R(r"^/api/v3/command$", "sonarr__command"),
        R(r"^/api/v3/series/lookup", "sonarr__series_lookup"),
        R(r"^/api/v3/series/\d+$", "sonarr__deleted", method="DELETE"),
        R(r"^/api/v3/series/\d+$", "sonarr__series_item"),
        R(r"^/api/v3/series$", "sonarr__series_item", method="POST"),
        R(r"^/api/v3/series", "sonarr__series"),
        R(r"^/api/v3/episode", "sonarr__episodes"),
    ],
    "radarr": [
        R(r"^/initialize\.json$", "radarr__initialize", key=False),
        R(r"^/api/v3/system/status$", "radarr__system_status"),
        R(r"^/api/v3/health$", "radarr__health"),
        R(r"^/api/v3/wanted/missing", "radarr__wanted_missing"),
        R(r"^/api/v3/calendar", "radarr__calendar"),
        R(r"^/api/v3/history", "radarr__history"),
        R(r"^/api/v3/queue$", "radarr__queue"),
        R(r"^/api/v3/rootfolder$", "radarr__rootfolder"),
        R(r"^/api/v3/qualityprofile", "radarr__qualityprofile"),
        R(r"^/api/v3/command$", "radarr__command"),
        R(r"^/api/v3/movie/lookup", "radarr__movie_lookup"),
        R(r"^/api/v3/movie/\d+$", "radarr__deleted", method="DELETE"),
        R(r"^/api/v3/movie/\d+", "radarr__movie_item"),
        R(r"^/api/v3/movie(\?|$)", "radarr__movie_item", method="POST"),
        R(r"^/api/v3/movie(\?|$)", "radarr__movie_list", method="GET"),
    ],
    "lidarr": [
        R(r"^/initialize\.json$", "lidarr__initialize", key=False),
        R(r"^/api/v1/system/status$", "lidarr__system_status"),
        R(r"^/api/v1/health$", "lidarr__health"),
        R(r"^/api/v1/rootfolder$", "lidarr__rootfolder"),
        R(r"^/api/v1/queue$", "lidarr__queue"),
        R(r"^/api/v1/search\?", "lidarr__search"),
        R(r"^/api/v1/artist", "lidarr__artists"),
    ],
    "prowlarr": [
        R(r"^/initialize\.json$", "prowlarr__initialize", key=False),
        R(r"^/api/v1/system/status$", "prowlarr__system_status"),
        R(r"^/api/v1/health$", "prowlarr__health"),
        R(r"^/api/v1/indexerstats", "prowlarr__indexer_stats"),
        R(r"^/api/v1/indexer/testall", "prowlarr__indexer_test",
          method="POST"),
        R(r"^/api/v1/indexer/test", "prowlarr__indexer_test", method="POST"),
        R(r"^/api/v1/indexer/\d+", "prowlarr__indexer_action"),
        R(r"^/api/v1/indexer", "prowlarr__indexers"),
        R(r"^/api/v1/search", "prowlarr__search"),
        R(r"^/api/v1/log", "prowlarr__log"),
        R(r"^/api/v1/applications/\d+", "prowlarr__app_action"),
        R(r"^/api/v1/applications$", "prowlarr__app_created", method="POST"),
        R(r"^/api/v1/applications", "prowlarr__applications"),
        R(r"^/api/v1/command$", "prowlarr__command"),
    ],
    "bazarr": [
        R(r"^/api/system/status", "bazarr__system_status"),
        R(r"^/api/system/settings", "bazarr__settings"),
        R(r"^/api/system/health", "bazarr__health"),
        R(r"^/api/history", "bazarr__history"),
        R(r"^/api/episodes/wanted", "bazarr__episodes_wanted"),
        R(r"^/api/movies/wanted", "bazarr__movies_wanted"),
        R(r"^/api/episodes/search", "bazarr__search_result", method="POST"),
        R(r"^/api/movies/search", "bazarr__search_result", method="POST"),
    ],
    "overseerr": [
        R(r"^/api/v1/status$", "overseerr__status"),
        R(r"^/api/v1/request/count$", "overseerr__request_count"),
        R(r"^/api/v1/request/\d+/approve$", "overseerr__request_item",
          method="POST"),
        R(r"^/api/v1/request/\d+/decline$", "overseerr__request_item",
          method="POST"),
        R(r"^/api/v1/request/\d+$", "overseerr__request_item"),
        R(r"^/api/v1/request", "overseerr__requests"),
        R(r"^/api/v1/user", "overseerr__users"),
    ],
    "jellyfin": [
        R(r"^/System/Info/Public$", "jellyfin__system_info_public",
          key=False),
        R(r"^/System/Ping$", "jellyfin__ping", key=False),
        R(r"^/System/Info$", "jellyfin__system_info"),
        R(r"^/System/ActivityLog/Entries", "jellyfin__activitylog"),
        R(r"^/Sessions$", "jellyfin__sessions"),
        R(r"^/Users$", "jellyfin__users"),
        R(r"^/Library/VirtualFolders$", "jellyfin__virtual_folders"),
        R(r"^/Library/Refresh$", "jellyfin__refresh", method="POST"),
        R(r"^/Items/Counts$", "jellyfin__counts"),
        R(r"^/Items/Latest", "jellyfin__latest"),
        R(r"^/Items/\d+/Refresh$", "jellyfin__refresh", method="POST"),
        R(r"^/UserItems/Resume", "jellyfin__resume"),
        R(r"^/Shows/NextUp", "jellyfin__nextup"),
        R(r"^/Items", "jellyfin__items"),
        # NOTE: /emby/… and /mediabrowser/… prefixes intentionally have NO
        # route — they 404, matching their removal in Jellyfin 12.0.
    ],
}

ARR_401 = {"error": "Unauthorized", "message": "Invalid API key"}
ARR_404 = {"error": "NotFound", "message": "No such endpoint in mock"}
LEGACY_TRAP = {"error": "LegacyAuthChannel",
               "message": "api_key query / X-Emby-Token are forbidden here"}

MEDIABROWSER_SCHEME = re.compile(r"(?i)^mediabrowser\s+(.*)$")
TOKEN_PART = re.compile(r'Token="([^"]*)"')
LEGACY_QUERY = re.compile(r"(?i)(^|&)(api_?key)=")

# Real-service request contracts the fixture routes alone don't model.
# Learned from the 17/10 live bug: the suite was green on bare
# GET /api/v3/episode because the mock served the fixture happily, while
# real Sonarr 400s — "seriesId or episodeIds must be provided". Entries:
# (service, method, path regex, predicate on parsed query -> veto?,
#  400 payload shaped like the real service's error).
#
# Jellyfin entries pinned by the 17/10 live-12.2.0 probes (second 400
# class, same drift lesson): /Items/Latest answers 400 unless userId
# rides the query (any other param set), and bare /Items answers 400
# when user-scoped params appear without userId — userdata sorts
# (sortBy=DatePlayed/PlayCount) or recursive+filters=IsPlayed. With
# userId the exact same shapes return 200. Non-user params
# (sortBy=SortName, recursive alone, non-recursive filters) stay 200
# with or without userId, so the predicate must NOT veto those.
CONTRACT_400S = [
    ("sonarr", "GET", re.compile(r"^/api/v3/episode$"),
     lambda q: not (q.get("seriesId") or q.get("episodeIds")),
     {"error": "BadRequest",
      "message": "seriesId or episodeIds must be provided"}),
    ("jellyfin", "GET", re.compile(r"^/Items/Latest$"),
     lambda q: not q.get("userId"),
     {"error": "Error processing request.",
      "message": "userId is required on /Items/Latest (live 12.2.0)"}),
    ("jellyfin", "GET", re.compile(r"^/Items$"),
     lambda q: not q.get("userId") and (
         q.get("sortBy", [""])[0] in ("DatePlayed", "PlayCount")
         or (q.get("recursive", [""])[0].lower() in ("true", "1")
             and "IsPlayed" in q.get("filters", [""]))),
     {"error": "Error processing request.",
      "message": "user-scoped /Items query requires userId (live 12.2.0)"}),
]


def contract_violation(service, method, parsed):
    """400 payload if the request breaks a real-service contract."""
    for svc, verb, path_re, broken, payload in CONTRACT_400S:
        if (service, method) == (svc, verb) and path_re.match(parsed.path) \
                and broken(parse_qs(parsed.query or "")):
            return payload
    return None


class MockState:
    def __init__(self, fixtures_dir, log_path):
        self.fixtures_dir = Path(fixtures_dir)
        self.log_path = Path(log_path)
        self.log_lock = threading.Lock()
        self.expected = {
            svc: os.environ.get(env, "")
            for svc, env in EXPECTED_KEY_ENV.items()}

    def fixture(self, name):
        p = self.fixtures_dir / (name + ".json")
        if not p.is_file():
            return None
        return json.loads(p.read_text())

    def log(self, record):
        with self.log_lock:
            with open(self.log_path, "a") as fh:
                fh.write(json.dumps(record) + "\n")


def pick_fixture(service, method, parsed):
    """Return (fixture_name, needs_key) or (None, False) if no route."""
    target = parsed.path + ("?" + parsed.query if parsed.query else "")
    for regex, fixture, needs_key, mmethod in ROUTES.get(service, []):
        if mmethod and method != mmethod:
            continue
        if regex.search(target):
            return fixture, needs_key
    return None, False


def request_auth(handler, parsed):
    """(scheme, key) the client presented — values never logged."""
    auth_header = handler.headers.get("Authorization") or ""
    m = MEDIABROWSER_SCHEME.match(auth_header)
    if m:
        t = TOKEN_PART.search(m.group(1))
        return "mediabrowser", (t.group(1) if t else "")
    hdr = handler.headers.get("X-Api-Key") or handler.headers.get("X-API-Key")
    if hdr:
        return "x-api-key", hdr
    return None, ""


def legacy_violation(handler, parsed):
    """jellyfin negative traps: api_key/ApiKey query or X-Emby-Token."""
    if LEGACY_QUERY.search(parsed.query or ""):
        return "api_key-in-query"
    if handler.headers.get("X-Emby-Token"):
        return "x-emby-token-header"
    return None


def make_handler(service, port, state):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):   # silence stderr noise
            pass

        def _serve(self):
            parsed = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            body_text = ""
            if length:
                body_text = self.rfile.read(length).decode(
                    "utf-8", "replace")[:2000]
            scheme, key = request_auth(self, parsed)
            expected = state.expected.get(service, "")
            violation = legacy_violation(self, parsed) \
                if service == "jellyfin" else None
            fixture_name, needs_key = pick_fixture(service, self.command,
                                                    parsed)
            key_ok = (not needs_key) or (expected != "" and key == expected)
            contract = None

            if violation:
                status, payload = 400, LEGACY_TRAP
            elif fixture_name is None:
                status, payload = 404, ARR_404
            elif not key_ok:
                status, payload = 401, ARR_401
            else:
                contract = contract_violation(service, self.command, parsed)
                if contract:
                    status, payload = 400, contract
                else:
                    fx = state.fixture(fixture_name)
                    if fx is None:
                        status, payload = 500, {"error": "fixture-missing",
                                                "fixture": fixture_name}
                    else:
                        status, payload = 200, fx

            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            try:
                self.wfile.write(data)
            except BrokenPipeError:
                pass

            record = {
                "ts": time.time(), "port": port, "service": service,
                "method": self.command, "path": parsed.path,
                "query": parsed.query,
                "headers": sorted(k.lower() for k in self.headers.keys()),
                "auth": scheme, "key_ok": key_ok, "status": status,
                "fixture": fixture_name,
            }
            if body_text:
                record["body"] = body_text
            if violation:
                record["violation"] = violation
            if contract:
                record["contract_400"] = True
            state.log(record)

        def do_GET(self):
            self._serve()

        def do_POST(self):
            self._serve()

        do_PUT = do_POST
        do_DELETE = do_POST          # reads body too, for write asserts

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixtures-dir", default=str(DEFAULTS["fixtures-dir"]))
    ap.add_argument("--log", default=str(DEFAULTS["log"]))
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()

    state = MockState(args.fixtures_dir, args.log)
    state.log_path.parent.mkdir(parents=True, exist_ok=True)
    state.log_path.write_text("")            # truncate per run

    servers = []
    for port, service in PORTS.items():
        srv = ThreadingHTTPServer((args.host, port),
                                  make_handler(service, port, state))
        srv.daemon_threads = True
        servers.append(srv)

    for srv in servers:
        threading.Thread(target=srv.serve_forever, daemon=True).start()

    state.log({"event": "ready", "ports": sorted(PORTS),
               "expected_keys": {s: bool(v)
                                 for s, v in state.expected.items()}})

    def shutdown(*_):
        for srv in servers:
            srv.shutdown()
        os._exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
