#!/usr/bin/env python3
"""Seerr API client for the seerr skill.

Talks to a Seerr server (https://github.com/seerr-team/seerr — the project that
Overseerr and Jellyseerr merged into). Seerr proxies Radarr and Sonarr, so root
folders, quality profiles and download progress all come from Seerr.

Radarr/Sonarr credentials are OPTIONAL and only needed for the two things Seerr
cannot report: `diskspace` and per-item `queue` detail.

Stdlib only. Configuration comes from the environment; when a variable is absent
we fall back to sourcing ~/.hermes/.env, because Hermes does not always export a
skill's variables into the terminal tool's environment.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

DEFAULT_TIMEOUT = 20
SEERR_API_PREFIX = "/api/v1"
ARR_API_PREFIX = "/api/v3"


class SeerrError(RuntimeError):
    """A user-facing error: bad config, HTTP failure, or unusable response."""


# ── Configuration ─────────────────────────────────────────────────────────

_ENV_LOADED = False


def _load_dotenv() -> None:
    """Populate os.environ from ~/.hermes/.env for variables Hermes did not export.

    Existing environment variables always win; only missing keys are filled in.
    """
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True

    try:
        raw = (pathlib.Path.home() / ".hermes" / ".env").read_text(encoding="utf-8")
    except OSError:
        return

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")


def _require(name: str, explicit: str | None = None, *, hint: str = "") -> str:
    _load_dotenv()
    value = (explicit or os.environ.get(name) or "").strip()
    if not value:
        raise SeerrError(f"{name} is not set.{' ' + hint if hint else ''}")
    return value


def _base_url(name: str, explicit: str | None = None, *, hint: str = "") -> str:
    url = _require(name, explicit, hint=hint)
    if not url.startswith(("http://", "https://")):
        raise SeerrError(f"{name} must start with http:// or https:// (got {url!r})")
    return url.rstrip("/")


# ── HTTP ──────────────────────────────────────────────────────────────────


def _encode_params(params: dict) -> str:
    """Percent-encode a query string.

    quote_via=quote (not the default quote_plus) so a space becomes %20 rather
    than '+', and every reserved character is escaped: a title like
    'Tom & Jerry' must not smuggle a bare '&' into the query string.
    """
    clean = {k: v for k, v in params.items() if v is not None and v != ""}
    return urllib.parse.urlencode(clean, quote_via=urllib.parse.quote)


def _call(
    endpoint: str,
    api_key: str,
    *,
    method: str = "GET",
    params: dict | None = None,
    body: dict | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> Any:
    if params:
        endpoint = f"{endpoint}?{_encode_params(params)}"

    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(endpoint, data=data, method=method)
    request.add_header("X-Api-Key", api_key)
    request.add_header("Accept", "application/json")
    if data is not None:
        request.add_header("Content-Type", "application/json")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", "replace")[:300]
        except Exception:  # noqa: BLE001 - the error body is best-effort
            pass
        if exc.code in (401, 403):
            raise SeerrError(f"API key rejected (HTTP {exc.code}).") from exc
        if exc.code == 409:
            raise SeerrError("Already requested — this title is in Seerr already.") from exc
        raise SeerrError(f"HTTP {exc.code}. {detail}".strip()) from exc
    except urllib.error.URLError as exc:
        raise SeerrError(f"Cannot reach {endpoint}: {exc.reason}") from exc

    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SeerrError("Server returned a non-JSON response.") from exc


def seerr(path: str, args, **kwargs) -> Any:
    url = _base_url("SEERR_URL", args.url, hint="Point it at your Seerr server.")
    key = _require("SEERR_API_KEY", args.api_key, hint="Seerr -> Settings -> General -> API Key.")
    return _call(f"{url}{SEERR_API_PREFIX}/{path.lstrip('/')}", key, **kwargs)


def arr(service: str, path: str, **kwargs) -> Any:
    """Call Radarr/Sonarr directly. Only used by `diskspace` and `queue`."""
    prefix = service.upper()
    hint = f"Needed only for `{prefix.lower()}` disk-space and queue detail."
    url = _base_url(f"{prefix}_URL", None, hint=hint)
    key = _require(f"{prefix}_API_KEY", None, hint=hint)
    return _call(f"{url}{ARR_API_PREFIX}/{path.lstrip('/')}", key, **kwargs)


# ── Formatting ────────────────────────────────────────────────────────────


def _year(item: dict) -> str:
    date = item.get("releaseDate") or item.get("firstAirDate") or ""
    return date[:4] if date else "----"


def _title(item: dict) -> str:
    return item.get("title") or item.get("name") or "Untitled"


def _gb(value: float) -> str:
    return f"{value / 1e9:.1f} GB"


def _progress(entry: dict) -> str:
    size = entry.get("size") or 0
    left = entry.get("sizeLeft")
    if not size or left is None:
        return entry.get("status") or "downloading"
    return f"{100.0 * (size - left) / size:.0f}% ({entry.get('status') or 'downloading'})"


# ── Commands ──────────────────────────────────────────────────────────────


def cmd_search(args) -> int:
    payload = seerr("search", args, params={"query": args.query, "page": args.page})
    results = [
        item
        for item in payload.get("results", [])
        if item.get("mediaType") in ("movie", "tv")
        and (args.type is None or item.get("mediaType") == args.type)
    ][: args.limit]

    if args.json:
        print(json.dumps(results, indent=2))
        return 0
    if not results:
        print(f"No movie or TV results for {args.query!r}.")
        return 0

    for index, item in enumerate(results, start=1):
        kind = item.get("mediaType")
        status = (item.get("mediaInfo") or {}).get("status")
        flag = " [already in library]" if status in (3, 4, 5) else ""
        print(f"{index}. {_title(item)} ({_year(item)}) - {kind} - tmdb:{item.get('id')}{flag}")
    return 0


def cmd_servers(args) -> int:
    """List servers with their root folders and quality profiles.

    This is the supported way to discover a concrete rootFolder value: Seerr
    reads it from the Servarr instance, so nothing has to be hand-configured.
    """
    servers = seerr(f"service/{args.service}", args)
    if isinstance(servers, dict):
        servers = servers.get("servers", []) or []

    detailed = [(s, seerr(f"service/{args.service}/{s.get('id')}", args)) for s in servers]

    if args.json:
        print(json.dumps([{"server": s, "details": d} for s, d in detailed], indent=2))
        return 0
    if not detailed:
        print(f"No {args.service} servers configured in Seerr.")
        return 0

    for server, info in detailed:
        default = " (default)" if server.get("isDefault") else ""
        fourk = " [4K]" if server.get("is4k") else ""
        print(f"server {server.get('id')}: {server.get('name')}{default}{fourk}")
        for folder in info.get("rootFolders") or []:
            print(f"  root folder: {folder.get('path')}")
        for profile in info.get("profiles") or []:
            print(f"  profile {profile.get('id')}: {profile.get('name')}")
    return 0


def cmd_seasons(args) -> int:
    payload = seerr(f"tv/{args.tmdb_id}", args)
    seasons = [s for s in payload.get("seasons", []) if s.get("seasonNumber", 0) > 0]

    if args.json:
        print(json.dumps(seasons, indent=2))
        return 0

    print(f"{payload.get('name') or 'Unknown'} ({_year(payload)})")
    for season in seasons:
        print(f"  season {season.get('seasonNumber')}: {season.get('episodeCount') or 0} episodes")
    return 0


def cmd_request(args) -> int:
    body: dict = {"mediaType": args.type, "mediaId": args.tmdb_id}

    if args.type == "tv":
        if not args.seasons:
            raise SeerrError("A TV request needs --seasons (e.g. --seasons 1,2 or --seasons all).")
        if args.seasons.strip().lower() == "all":
            body["seasons"] = "all"
        else:
            try:
                body["seasons"] = [int(s) for s in args.seasons.split(",") if s.strip()]
            except ValueError as exc:
                raise SeerrError(f"--seasons must be numbers or 'all' (got {args.seasons!r}).") from exc

    # Seerr's request-body field is `rootFolder` -- NOT `rootFolderPath`.
    # An unknown key is silently ignored and the request lands in the default
    # folder, which looks exactly like success.
    if args.root_folder:
        body["rootFolder"] = args.root_folder
    if args.server_id is not None:
        body["serverId"] = args.server_id
    if args.profile_id is not None:
        body["profileId"] = args.profile_id

    payload = seerr("request", args, method="POST", body=body)
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    media = payload.get("media") or {}
    print(
        f"Requested: {args.type} tmdb:{args.tmdb_id} "
        f"(request id {payload.get('id')}, status {media.get('status', 'pending')})"
    )
    return 0


def cmd_status(args) -> int:
    payload = seerr("request", args, params={"take": args.take, "sort": "added"})
    results = payload.get("results", [])

    if args.json:
        print(json.dumps(results, indent=2))
        return 0
    if not results:
        print("No requests found.")
        return 0

    for entry in results:
        media = entry.get("media") or {}
        downloads = media.get("downloadStatus") or []
        detail = _progress(downloads[0]) if downloads else f"status {media.get('status')}"
        label = f"request {entry.get('id')}: tmdb:{media.get('tmdbId')} ({media.get('mediaType')})"
        print(f"{label} - {detail}")
    return 0


def cmd_diskspace(args) -> int:
    """Free space. Seerr does not expose this — ask Radarr/Sonarr directly."""
    mounts = arr(args.service, "diskspace")
    if args.json:
        print(json.dumps(mounts, indent=2))
        return 0
    for mount in mounts:
        free, total = mount.get("freeSpace", 0), mount.get("totalSpace", 0)
        print(f"{mount.get('path'):<30} {_gb(free)} free / {_gb(total)} total")
    return 0


def cmd_queue(args) -> int:
    """Per-item download queue detail, which Seerr summarises but does not itemise."""
    payload = arr(args.service, "queue", params={"pageSize": args.take})
    records = payload.get("records", []) if isinstance(payload, dict) else payload

    if args.json:
        print(json.dumps(records, indent=2))
        return 0
    if not records:
        print("Queue is empty.")
        return 0

    for item in records:
        size, left = item.get("size") or 0, item.get("sizeleft")
        done = f"{100.0 * (size - left) / size:.0f}%" if size and left is not None else "?"
        print(f"{item.get('title', 'Unknown')[:50]:<52} {done:>5}  {item.get('status', '')}")
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="seerr.py", description=__doc__)
    parser.add_argument("--url", help="Seerr base URL (default: $SEERR_URL)")
    parser.add_argument("--api-key", help="Seerr API key (default: $SEERR_API_KEY)")
    parser.add_argument("--json", action="store_true", help="Emit raw JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    search = sub.add_parser("search", help="Search movies and TV shows")
    search.add_argument("query")
    search.add_argument("--type", choices=["movie", "tv"])
    search.add_argument("--limit", type=int, default=8)
    search.add_argument("--page", type=int, default=1)
    search.set_defaults(func=cmd_search)

    servers = sub.add_parser("servers", help="Servers, root folders and quality profiles")
    servers.add_argument("service", choices=["radarr", "sonarr"])
    servers.set_defaults(func=cmd_servers)

    seasons = sub.add_parser("seasons", help="List seasons for a TV show")
    seasons.add_argument("tmdb_id", type=int)
    seasons.set_defaults(func=cmd_seasons)

    request = sub.add_parser("request", help="Submit a request")
    request.add_argument("--type", choices=["movie", "tv"], required=True)
    request.add_argument("--tmdb-id", type=int, required=True)
    request.add_argument("--seasons", help="TV only: '1,2' or 'all'")
    request.add_argument("--root-folder", help="Path from `servers` output")
    request.add_argument("--server-id", type=int)
    request.add_argument("--profile-id", type=int)
    request.set_defaults(func=cmd_request)

    status = sub.add_parser("status", help="Recent requests and download progress")
    status.add_argument("--take", type=int, default=10)
    status.set_defaults(func=cmd_status)

    diskspace = sub.add_parser("diskspace", help="Free space (needs Radarr/Sonarr creds)")
    diskspace.add_argument("--service", choices=["radarr", "sonarr"], default="radarr")
    diskspace.set_defaults(func=cmd_diskspace)

    queue = sub.add_parser("queue", help="Per-item queue detail (needs Radarr/Sonarr creds)")
    queue.add_argument("--service", choices=["radarr", "sonarr"], default="radarr")
    queue.add_argument("--take", type=int, default=20)
    queue.set_defaults(func=cmd_queue)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except SeerrError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
