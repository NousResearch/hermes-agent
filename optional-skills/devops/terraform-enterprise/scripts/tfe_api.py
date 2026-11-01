#!/usr/bin/env python3
"""HCP Terraform / Terraform Enterprise API helper (stdlib only).

Reads the standard TFE environment variables: TFE_TOKEN (required), TFE_ADDRESS (default
https://app.terraform.io; TFE_HOSTNAME is accepted as the tfe provider's spelling) and
TFE_ORGANIZATION (default organization for workspace commands). Prints JSON on stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_ADDRESS = "https://app.terraform.io"
JSONAPI = "application/vnd.api+json"
TIMEOUT = 30


class TfeError(RuntimeError):
    pass


def base_url(env=None) -> str:
    """API root for TFE_ADDRESS (a URL or a bare hostname), falling back to TFE_HOSTNAME."""
    env = os.environ if env is None else env
    address = (env.get("TFE_ADDRESS") or env.get("TFE_HOSTNAME") or DEFAULT_ADDRESS).strip()
    if "://" not in address:
        address = "https://" + address
    return address.rstrip("/") + "/api/v2"


def token(env=None) -> str:
    env = os.environ if env is None else env
    value = (env.get("TFE_TOKEN") or "").strip()
    if not value:
        raise TfeError("TFE_TOKEN is not set (user or team API token from the TFE settings page)")
    return value


def organization(explicit, env=None) -> str:
    env = os.environ if env is None else env
    org = explicit or (env.get("TFE_ORGANIZATION") or "").strip()
    if not org:
        raise TfeError("no organization: pass --org or set TFE_ORGANIZATION")
    return org


def request(method: str, path: str, body=None, params=None):
    url = base_url() + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {token()}", "Content-Type": JSONAPI, "Accept": JSONAPI})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            raw = resp.read()
    except urllib.error.HTTPError as e:
        raise TfeError(f"{method} {path}: HTTP {e.code} {_error_detail(e.read())}") from None
    except urllib.error.URLError as e:
        raise TfeError(f"{method} {path}: {e.reason}") from None
    return json.loads(raw) if raw else {}


def _error_detail(raw: bytes) -> str:
    try:
        errors = json.loads(raw).get("errors") or []
    except (ValueError, AttributeError):
        return raw.decode(errors="replace")[:300]
    return "; ".join(e.get("detail") or e.get("title") or str(e) if isinstance(e, dict) else str(e)
                     for e in errors)


def paginate(path: str, params=None, limit: int = 100):
    """Collect ``data`` across JSON:API pages until ``limit`` items or the last page."""
    items, page = [], 1
    while len(items) < limit:
        doc = request("GET", path, params={**(params or {}), "page[size]": 100, "page[number]": page})
        items.extend(doc.get("data") or [])
        page = ((doc.get("meta") or {}).get("pagination") or {}).get("next-page")
        if not page:
            break
    return items[:limit]


def _attrs(item: dict, *keys) -> dict:
    a = item.get("attributes") or {}
    return {"id": item.get("id"), **{k: a.get(k) for k in keys}}


def summarize_workspace(ws: dict) -> dict:
    out = _attrs(ws, "name", "terraform-version", "execution-mode", "auto-apply", "locked",
                 "resource-count", "working-directory", "updated-at")
    out["current-run"] = (((ws.get("relationships") or {}).get("current-run") or {})
                          .get("data") or {}).get("id")
    return out


def summarize_run(run: dict) -> dict:
    return _attrs(run, "status", "message", "source", "is-destroy", "plan-only", "has-changes",
                  "created-at")


def workspace_id(ref: str, org) -> str:
    """Accept a ws-... id as is; resolve anything else as a name in the organization."""
    if ref.startswith("ws-"):
        return ref
    path = f"/organizations/{urllib.parse.quote(organization(org))}/workspaces/{urllib.parse.quote(ref)}"
    return request("GET", path)["data"]["id"]


def cmd_whoami(a):
    d = request("GET", "/account/details")["data"]
    return {"address": base_url(), **_attrs(d, "username", "email", "is-service-account")}


def cmd_workspaces(a):
    params = {"search[name]": a.search} if a.search else None
    org = urllib.parse.quote(organization(a.org))
    return [summarize_workspace(w) for w in paginate(f"/organizations/{org}/workspaces", params, a.limit)]


def cmd_workspace(a):
    return summarize_workspace(request("GET", f"/workspaces/{workspace_id(a.workspace, a.org)}")["data"])


def cmd_runs(a):
    return [summarize_run(r) for r in paginate(f"/workspaces/{workspace_id(a.workspace, a.org)}/runs",
                                               limit=a.limit)]


def cmd_run(a):
    run = request("GET", f"/runs/{a.run_id}", params={"include": "plan"})
    out = summarize_run(run["data"])
    for inc in run.get("included") or []:
        if inc.get("type") == "plans":
            out["plan"] = _attrs(inc, "status", "resource-additions", "resource-changes",
                                 "resource-destructions", "log-read-url")
    return out


def cmd_plan_log(a):
    plan_id = request("GET", f"/runs/{a.run_id}")["data"]["relationships"]["plan"]["data"]["id"]
    log_url = request("GET", f"/plans/{plan_id}")["data"]["attributes"]["log-read-url"]
    # The log URL is pre-signed: sending the API token to it is unnecessary.
    with urllib.request.urlopen(log_url, timeout=TIMEOUT) as resp:
        text = resp.read().decode(errors="replace")
    return {"run": a.run_id, "plan": plan_id, "log": text[-a.tail:] if a.tail else text}


def cmd_outputs(a):
    ws = workspace_id(a.workspace, a.org)
    items = paginate(f"/workspaces/{ws}/current-state-version-outputs", limit=1000)
    return [_attrs(o, "name", "type", "sensitive", "value") for o in items]


def cmd_start_run(a):
    body = {"data": {"type": "runs",
                     "attributes": {"message": a.message, "is-destroy": a.destroy,
                                    "plan-only": a.plan_only},
                     "relationships": {"workspace": {"data": {
                         "type": "workspaces", "id": workspace_id(a.workspace, a.org)}}}}}
    return summarize_run(request("POST", "/runs", body=body)["data"])


def _run_action(action):
    def handler(a):
        request("POST", f"/runs/{a.run_id}/actions/{action}", body={"comment": a.comment})
        return {"run": a.run_id, "action": action, "queued": True}
    return handler


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    def add(name, handler, *, ws=False, run=False, limit=False):
        sp = sub.add_parser(name)
        sp.set_defaults(handler=handler)
        if ws:
            sp.add_argument("workspace", help="workspace name or ws-... id")
        if run:
            sp.add_argument("run_id", help="run-... id")
        if ws or name == "workspaces":
            sp.add_argument("--org", help="organization (default: TFE_ORGANIZATION)")
        if limit:
            sp.add_argument("--limit", type=int, default=20)
        return sp

    add("whoami", cmd_whoami)
    add("workspaces", cmd_workspaces, limit=True).add_argument("--search", help="name substring")
    add("workspace", cmd_workspace, ws=True)
    add("runs", cmd_runs, ws=True, limit=True)
    add("run", cmd_run, run=True)
    add("plan-log", cmd_plan_log, run=True).add_argument(
        "--tail", type=int, default=20000, help="keep the last N characters (0 = all)")
    add("outputs", cmd_outputs, ws=True)
    start = add("start-run", cmd_start_run, ws=True)
    start.add_argument("--message", default="Queued by Hermes")
    start.add_argument("--plan-only", action="store_true", help="speculative plan, cannot be applied")
    start.add_argument("--destroy", action="store_true")
    for action in ("apply", "discard", "cancel"):
        add(action, _run_action(action), run=True).add_argument("--comment", default="via Hermes")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = args.handler(args)
    except TfeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    json.dump(result, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
