#!/usr/bin/env python3
"""No-auth HTTP client for a local asset-studio server.

asset-studio is a single-operator, local-only FastAPI app (binds to
127.0.0.1, no auth layer by design — see agents/PRODUCT.md in the
asset-studio repo). Multiple checkouts (main/agents/coder/tester/uat) can
each run their own instance on their own port, so this client takes
--host/--port explicitly with no baked-in default port.

Named subcommands cover the ~14 reads an agent reaches for most: v2 flows
(the current carousel-authoring pipeline), accounts/templates, character
generation jobs and cutouts, and the issue-#7 batch queue system. Everything
else is reachable through the generic `call` subcommand, backed by
references/endpoints.md (or the live `spec` subcommand, which dumps
asset-studio's own /openapi.json).

Safety: every non-GET request is refused unless --confirm is passed. Paths
on the HIGH_RISK_MARKERS list print a loud banner even with --confirm —
these are the calls that spend real money via the external Magnific API
(character generation, cutout background removal, reel start/retry, batch
queue execution) or a capped-but-real LLM call (caption generation). This
script never decides "is it OK to spend this" on its own — the agent must
have gotten explicit, specific confirmation from the user in chat first;
--confirm only records that the agent did so. No implicit retries.

Binary responses (slide preview PNGs, flow/batch export ZIPs) are detected
by Content-Type and not dumped to stdout as garbled text. Pass --out <path>
to save the bytes; without it, only size/content-type are reported.

There are no SSE/streaming routes in asset-studio's API (verified against
the live OpenAPI schema and a repo-wide grep for text/event-stream —  the
only hit is asset-studio's own outbound client talking to the external
Magnific API, not anything this server exposes) so, unlike the commander-api
skill this one is modeled on, there is no `stream()` helper here.
"""
import argparse
import json
import socket
import sys
import urllib.error
import urllib.parse
import urllib.request

MAX_LIST_ITEMS = 15
MAX_BODY_CHARS = 6000

# (method, path substring) — calls that trigger real spend (Magnific credits
# or a capped LLM call) get a loud banner even with --confirm. Substring
# matching, not full pattern matching: false positives (warn when unneeded)
# are fine, false negatives are not. Method is checked so read-alike/adjacent
# routes sharing a substring (e.g. PUT .../caption saving an edited caption,
# which is free) don't get flagged just because POST .../caption (which
# generates one via a $0.05-capped `claude -p` call) does.
HIGH_RISK_MARKERS = [
    ("POST", "/characters/generate"),   # Magnific character-pose generation
    ("POST", "/characters/cutout"),     # Magnific background removal
    ("POST", "/reel/start"),            # Magnific image->video (feature-flagged)
    ("POST", "/reel/retry"),            # re-submits the same paid job
    ("POST", "/caption"),               # claude -p, --max-budget-usd 0.05
    ("POST", "/start"),                 # POST /api/batch/{id}/start — runs a
                                         # headless `claude -p ... mcp__magnific`
                                         # subprocess per queued job
    ("POST", "/more-options"),          # spawns K more paid Magnific variations
]

# Export and compose were checked against the source (services/v2flows.py,
# compositer/compositor.py, services/render/html_renderer.py): both render
# already-picked slide content with Pillow/Chromium in-process. Neither
# calls Magnific or an LLM, so neither is on the high-risk list — they still
# require --confirm structurally (any non-GET does) but don't warrant the
# loud banner.


def _confirm_gate(method, path, confirmed):
    method = method.upper()
    if method in ("GET", "HEAD"):
        return
    if not confirmed:
        print(
            f"REFUSED: {method} {path} is a mutating call. Pass --confirm "
            "only after the user has explicitly approved this exact action "
            "in chat.",
            file=sys.stderr,
        )
        sys.exit(2)
    is_high_risk = method == "DELETE" or any(
        method == m and marker in path for m, marker in HIGH_RISK_MARKERS
    )
    if is_high_risk:
        print(
            f"!!! HIGH-RISK CALL: {method} {path} !!!\n"
            "This spends real money — Magnific credits or a capped LLM call "
            "(or, for DELETE, destroys stored data). Proceeding because "
            "--confirm was passed — this assumes the user approved *this "
            "specific action*, not a generic 'go ahead'.",
            file=sys.stderr,
        )


def _shrink(obj):
    """Cap list length and overall size so one call can't blow the context."""
    if isinstance(obj, list):
        shrunk = [_shrink(item) for item in obj[:MAX_LIST_ITEMS]]
        if len(obj) > MAX_LIST_ITEMS:
            shrunk.append(f"... ({len(obj) - MAX_LIST_ITEMS} more items truncated)")
        return shrunk
    if isinstance(obj, dict):
        return {k: _shrink(v) for k, v in obj.items()}
    return obj


def _fetch_raw(base_url, method, path, body=None, query=None, timeout=30):
    """Issue the HTTP call. Returns (status, raw_bytes, content_type) or
    (None, error_dict, None) on connection failure."""
    url = base_url.rstrip("/") + "/" + path.lstrip("/")
    if query:
        url += "?" + urllib.parse.urlencode({k: v for k, v in query.items() if v is not None})
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, data=data, headers=headers, method=method.upper())
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
            status = resp.status
            content_type = resp.headers.get("Content-Type", "")
    except urllib.error.HTTPError as e:
        raw = e.read()
        status = e.code
        content_type = e.headers.get("Content-Type", "") if e.headers else ""
    except (socket.timeout, TimeoutError) as e:
        # Some renders (slide/flow preview, export) go through headless
        # Chromium via Playwright and can legitimately take a while — but a
        # live check against this skill's reference instance found
        # GET .../slides/{i}/preview hanging past 60s with no response at
        # all (not just slow), which looks like a server-side issue rather
        # than normal render latency. Report cleanly instead of a raw
        # traceback so the agent can decide whether to retry with a bigger
        # --timeout or give up and tell the user the render is stuck.
        return None, {
            "error": f"request timed out after {timeout}s: {e}",
            "url": url,
            "hint": "raise --timeout for slow Chromium-backed renders (slide "
                     "preview / export), or the server may be stuck — see "
                     "SKILL.md pitfalls",
        }, None
    except urllib.error.URLError as e:
        return None, {"error": f"connection failed: {e.reason}", "url": url}, None

    return status, raw, content_type


def _parse_body(raw, content_type, out_path=None):
    """Decode a response body for display, or describe it if it's binary."""
    content_type = content_type or ""
    if "application/json" in content_type or not content_type:
        try:
            return json.loads(raw.decode()) if raw else {}
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    if content_type.startswith("text/"):
        return {"raw": raw.decode(errors="replace")[:MAX_BODY_CHARS]}

    # Binary (image/png, application/zip, video/mp4, ...) — never dump raw
    # bytes into a JSON response. Save to --out if given, else just describe.
    result = {"binary": True, "content_type": content_type, "byte_size": len(raw)}
    if out_path:
        with open(out_path, "wb") as f:
            f.write(raw)
        result["saved_to"] = out_path
    else:
        result["note"] = "pass --out <path> to save these bytes"
    return result


def request(base_url, method, path, body=None, query=None, timeout=30, out_path=None):
    status, raw, content_type = _fetch_raw(base_url, method, path, body, query, timeout)
    if status is None:
        return raw  # connection error dict
    parsed = _parse_body(raw, content_type, out_path)
    result = {"status": status, "body": _shrink(parsed)}
    if len(json.dumps(result, default=str)) > MAX_BODY_CHARS:
        result["body"] = {"truncated": True, "preview": str(parsed)[:MAX_BODY_CHARS]}
    return result


# name -> (path_template, path_params, query_params[(name, required)], help)
SHORTCUTS = {
    "accounts": (
        "/api/accounts", [], [],
        "List registered accounts (handle, brand colors, fonts, action presets)",
    ),
    "account_templates": (
        "/api/accounts/{handle}/templates", ["handle"], [],
        "List an account's slide layouts (cover/content/cta/quote/stat) with slot manifests",
    ),
    "flows": (
        "/api/v2/flows", [], [],
        "List all v2 carousel flows (queue view: status, title, slide count, timestamps)",
    ),
    "flow": (
        "/api/v2/flows/{flow_id}", ["flow_id"], [],
        "Get full v2 flow state (slides, layouts, texts, caption, status)",
    ),
    "slide_preview": (
        "/api/v2/flows/{flow_id}/slides/{slide_index}/preview",
        ["flow_id", "slide_index"], [],
        "Render a v2 slide preview PNG (binary — pass --out to save it)",
    ),
    "templates": (
        "/api/templates", [], [],
        "List v1 compose slide templates (default/cinematic/bold) — the Pillow "
        "compose pipeline's templates, NOT the v2 account layout packs",
    ),
    "template": (
        "/api/templates/{name}", ["name"], [],
        "Get one v1 compose template's full JSON",
    ),
    "character_job": (
        "/api/accounts/{handle}/characters/jobs/{job_id}", ["handle", "job_id"], [],
        "Poll a character-generation job (variations, status)",
    ),
    "cutouts": (
        "/api/accounts/{handle}/characters/cutouts", ["handle"], [("action", False)],
        "List background-removed character cutouts (optional action filter)",
    ),
    "reel_job": (
        "/api/reel/job/{job_id}", ["job_id"], [],
        "Poll a Make Reel (image->video) job — 403 if FEATURE_MAKE_REEL is off",
    ),
    "batch_status": (
        "/api/batch/{batch_id}/status", ["batch_id"], [],
        "Batch-queue progress (issue #7 queue system: index/percent complete)",
    ),
    "batch_queue": (
        "/api/batch/{batch_id}/queue", ["batch_id"], [],
        "List jobs in a batch queue, in order",
    ),
    "batch_job": (
        "/api/batch/jobs/{job_id}", ["job_id"], [],
        "Single batch-queue job detail (status, output_url, cost)",
    ),
    "batch_summary": (
        "/api/batch/{batch_id}/summary", ["batch_id"], [],
        "Aggregate Magnific credits + agent cost for a batch",
    ),
}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="asset-studio binds to 127.0.0.1 only")
    p.add_argument(
        "--port", type=int, required=True,
        help="No safe default — asset-studio runs as several parallel worktree "
             "instances (main/agents/coder/tester/uat), each on its own port "
             "(that worktree's .env PORT, or 8000). Ask the user which "
             "instance they mean if it's not obvious.",
    )
    p.add_argument("--out", default=None, help="Save a binary response (PNG/ZIP/MP4) to this path")
    p.add_argument(
        "--timeout", type=int, default=30,
        help="Seconds before giving up (default 30). Slide preview and flow/"
             "batch export render through headless Chromium and can be slow "
             "or, per a live check against this skill's reference instance, "
             "hang outright — raise this if you hit a timeout error and want "
             "to retry once with more patience.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, (path_template, path_params, query_params, help_text) in SHORTCUTS.items():
        sp = sub.add_parser(name, help=help_text)
        for param in path_params:
            sp.add_argument(param)
        for qname, required in query_params:
            sp.add_argument(f"--{qname}", required=required, default=None)

    sp = sub.add_parser("spec", help="Dump asset-studio's live OpenAPI schema")
    sp.add_argument("--path", default="", help="Optional path filter substring")

    sp = sub.add_parser("call", help="Generic request to any asset-studio route")
    sp.add_argument("method")
    sp.add_argument("path")
    sp.add_argument("--json", dest="json_body", default=None, help="JSON request body")
    sp.add_argument(
        "--confirm",
        action="store_true",
        help="Required for any non-GET call; only pass after explicit user approval",
    )

    args = p.parse_args()
    base_url = f"http://{args.host}:{args.port}"

    if args.cmd == "spec":
        status, raw, content_type = _fetch_raw(base_url, "GET", "/openapi.json")
        if status is None:
            print(json.dumps(raw, indent=2))
            return
        parsed = _parse_body(raw, content_type)
        if isinstance(parsed, dict) and "paths" in parsed:
            # Drop $ref schema definitions — bulky and rarely needed for a
            # basic method/path/param lookup, which is what this is for.
            parsed = {k: v for k, v in parsed.items() if k != "components"}
            if args.path:
                parsed["paths"] = {
                    k: v for k, v in parsed.get("paths", {}).items() if args.path in k
                }
        result = {"status": status, "body": _shrink(parsed)}
        if len(json.dumps(result, default=str)) > MAX_BODY_CHARS:
            result["body"] = {"truncated": True, "preview": str(parsed)[:MAX_BODY_CHARS]}
        print(json.dumps(result, indent=2))
        return

    if args.cmd == "call":
        body = json.loads(args.json_body) if args.json_body else None
        _confirm_gate(args.method, args.path, args.confirm)
        result = request(base_url, args.method, args.path, body, timeout=args.timeout, out_path=args.out)
        print(json.dumps(result, indent=2))
        return

    # Named GET shortcuts
    path_template, path_params, query_params, _ = SHORTCUTS[args.cmd]
    path = path_template
    for param in path_params:
        path = path.replace(f"{{{param}}}", getattr(args, param))
    query = {qname: getattr(args, qname) for qname, _ in query_params}
    result = request(base_url, "GET", path, query=query, timeout=args.timeout, out_path=args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
