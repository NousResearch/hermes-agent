#!/usr/bin/env python3
"""Seerr API client for the seerr skill.

Talks to a Seerr server (https://github.com/seerr-team/seerr — the project that
Overseerr and Jellyseerr merged into). Seerr is the single integration point:
it proxies Radarr/Sonarr, so root folders, quality profiles and download
progress all come from Seerr and no Radarr/Sonarr credentials are needed.

Stdlib only. Reads SEERR_URL and SEERR_API_KEY from the environment; both can
be overridden with --url / --api-key.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_TIMEOUT = 20
API_PREFIX = "/api/v1"


class SeerrError(RuntimeError):
    """A user-facing error: bad config, HTTP failure, or unusable response."""


# ── HTTP ──────────────────────────────────────────────────────────────────


def _base_url(explicit: str | None = None) -> str:
    url = (explicit or os.environ.get("SEERR_URL") or "").strip()
    if not url:
        raise SeerrError(
            "SEERR_URL is not set. Point it at your Seerr server, "
            "e.g. http://localhost:5055"
        )
    if not url.startswith(("http://", "https://")):
        raise SeerrError(f"SEERR_URL must start with http:// or https:// (got {url!r})")
    return url.rstrip("/")


def _api_key(explicit: str | None = None) -> str:
    key = (explicit or os.environ.get("SEERR_API_KEY") or "").strip()
    if not key:
        raise SeerrError(
            "SEERR_API_KEY is not set. Find it in Seerr under "
            "Settings -> General -> API Key."
        )
    return key


def _encode_params(params: dict) -> str:
    """Percent-encode a query string.

    quote_via=quote (not the default quote_plus) so a space becomes %20 rather
    than '+', and every reserved character is escaped: a title like
    'Tom & Jerry' must not smuggle a bare '&' into the query string.
    """
    clean = {k: v for k, v in params.items() if v is not None and v != ""}
    return urllib.parse.urlencode(clean, quote_via=urllib.parse.quote)


def api(
    path: str,
    *,
    method: str = "GET",
    params: dict | None = None,
    body: dict | None = None,
    url: str | None = None,
    api_key: str | None = None,
    timeout: int = DEFAULT_TIMEOUT,
) -> dict:
    endpoint = f"{_base_url(url)}{API_PREFIX}/{path.lstrip('/')}"
    if params:
        endpoint = f"{endpoint}?{_encode_params(params)}"

    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(endpoint, data=data, method=method)
    request.add_header("X-Api-Key", _api_key(api_key))
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
        except Exception:  # noqa: BLE001 - error body is best-effort
            pass
        if exc.code in (401, 403):
            raise SeerrError(
                "Seerr rejected the API key (HTTP "
                f"{exc.code}). Check SEERR_API_KEY."
            ) from exc
        raise SeerrError(f"Seerr returned HTTP {exc.code}. {detail}".strip()) from exc
    except urllib.error.URLError as exc:
        raise SeerrError(f"Cannot reach Seerr at {_base_url(url)}: {exc.reason}") from exc

    if not raw.strip():
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SeerrError("Seerr returned a non-JSON response.") from exc


# ── Formatting ────────────────────────────────────────────────────────────


def _year(item: dict) -> str:
    date = item.get("releaseDate") or item.get("firstAirDate") or ""
    return date[:4] if date else "----"


def _title(item: dict) -> str:
    return item.get("title") or item.get("name") or "Untitled"


def _progress(entry: dict) -> str:
    size = entry.get("size") or 0
    left = entry.get("sizeLeft")
    if not size or left is None:
        return entry.get("status") or "downloading"
    done = 100.0 * (size - left) / size
    return f"{done:.0f}% ({entry.get('status') or 'downloading'})"


# ── Commands ──────────────────────────────────────────────────────────────


def cmd_search(args) -> int:
    payload = api(
        "search",
        params={"query": args.query, "page": args.page},
        url=args.url,
        api_key=args.api_key,
    )
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
        kind = "movie" if item.get("mediaType") == "movie" else "tv"
        status = (item.get("mediaInfo") or {}).get("status")
        flag = " [already in library]" if status in (3, 4, 5) else ""
        print(f"{index}. {_title(item)} ({_year(item)}) - {kind} - tmdb:{item.get('id')}{flag}")
    return 0


def cmd_servers(args) -> int:
    """List Radarr/Sonarr servers with their root folders and quality profiles.

    This is the supported way to discover a concrete rootFolder value: Seerr
    reads it from the Servarr instance, so nothing has to be hand-configured.
    """
    servers = api(f"service/{args.service}", url=args.url, api_key=args.api_key)
    if isinstance(servers, dict):
        servers = servers.get("servers", servers) or []

    detailed = []
    for server in servers:
        info = api(
            f"service/{args.service}/{server.get('id')}",
            url=args.url,
            api_key=args.api_key,
        )
        detailed.append({"server": server, "details": info})

    if args.json:
        print(json.dumps(detailed, indent=2))
        return 0
    if not detailed:
        print(f"No {args.service} servers configured in Seerr.")
        return 0

    for entry in detailed:
        server, info = entry["server"], entry["details"]
        default = " (default)" if server.get("isDefault") else ""
        fourk = " [4K]" if server.get("is4k") else ""
        print(f"server {server.get('id')}: {server.get('name')}{default}{fourk}")
        for folder in info.get("rootFolders") or []:
            print(f"  root folder: {folder.get('path')}")
        for profile in info.get("profiles") or []:
            print(f"  profile {profile.get('id')}: {profile.get('name')}")
    return 0


def cmd_seasons(args) -> int:
    payload = api(f"tv/{args.tmdb_id}", url=args.url, api_key=args.api_key)
    seasons = [
        season
        for season in payload.get("seasons", [])
        if season.get("seasonNumber", 0) > 0  # season 0 is Specials
    ]
    if args.json:
        print(json.dumps(seasons, indent=2))
        return 0

    print(f"{payload.get('name') or 'Unknown'} ({_year(payload)})")
    for season in seasons:
        episodes = season.get("episodeCount") or 0
        print(f"  season {season.get('seasonNumber')}: {episodes} episodes")
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

    # Seerr's request body field is `rootFolder` -- NOT `rootFolderPath`.
    # A wrong key is silently ignored and the request lands in the default folder.
    if args.root_folder:
        body["rootFolder"] = args.root_folder
    if args.server_id is not None:
        body["serverId"] = args.server_id
    if args.profile_id is not None:
        body["profileId"] = args.profile_id

    payload = api("request", method="POST", body=body, url=args.url, api_key=args.api_key)
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
    payload = api(
        "request",
        params={"take": args.take, "sort": "added"},
        url=args.url,
        api_key=args.api_key,
    )
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
        label = f"request {entry.get('id')}: tmdb:{media.get('tmdbId')} ({media.get('mediaType')})"
        if downloads:
            print(f"{label} - {_progress(downloads[0])}")
        else:
            print(f"{label} - status {media.get('status')}")
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

    servers = sub.add_parser("servers", help="List servers, root folders and profiles")
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
