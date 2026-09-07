"""Native Spotify tools for Hermes (registered via plugins/spotify).

Each tool routes ``args["action"]`` through a dict dispatch table; every entry
has the signature ``(client, args, action) -> str`` and issues its Spotify Web
API call through ``client.request`` (auth refresh + error mapping live there).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from hermes_cli.auth import get_auth_status
from plugins.spotify.client import (
    SpotifyClient, SpotifyError, normalize_spotify_id, normalize_spotify_uri, normalize_spotify_uris)
from tools.registry import tool_error, tool_result

_Handler = Callable[[SpotifyClient, dict, str], str]


def _check_spotify_available() -> bool:
    try:
        return bool(get_auth_status("spotify").get("logged_in"))
    except Exception:
        return False


def _spotify_tool_error(exc: Exception) -> str:
    if isinstance(exc, SpotifyError):  # includes SpotifyAPIError / SpotifyAuthRequiredError
        return tool_error(str(exc))
    return tool_error(f"Spotify tool failed: {type(exc).__name__}: {exc}")


# Inside a ``_dispatcher`` boundary ``raise SpotifyError(msg)`` renders exactly like ``return tool_error(msg)``.
def _required(value: Any, message: str) -> Any:
    if value is None:
        raise SpotifyError(message)
    return value


def _nonblank(raw: Any, message: str) -> str:
    value = str(raw or "").strip()
    if not value:
        raise SpotifyError(message)
    return value


def _coerce_bool(raw: Any, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        cleaned = raw.strip().lower()
        if cleaned in {"1", "true", "yes", "on"}:
            return True
        if cleaned in {"0", "false", "no", "off"}:
            return False
    return default


def _as_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    items = raw if isinstance(raw, list) else [raw]
    return [str(item).strip() for item in items if str(item).strip()]


_offset = lambda args: max(0, int(args.get("offset") or 0))  # noqa: E731


def _limit(args: dict, default: int = 20) -> int:
    """Clamp ``limit`` to Spotify's 1..50 window; non-numeric input falls back to *default*."""
    raw: Any = args.get("limit")
    try:
        value = int(raw)
    except Exception:
        value = default
    return max(1, min(50, value))


_ok = lambda action, result, **extra: tool_result({"success": True, "action": action, **extra, "result": result})  # noqa: E731


def _dispatcher(tool_name: str, default: str, table: Dict[str, _Handler], prepare: Optional[Callable[[dict], dict]] = None):
    """Build a tool handler that routes ``args['action']`` through *table*.

    The client is constructed outside the error boundary (auth failures propagate
    as exceptions, as they always have). *prepare* runs inside the boundary before
    the action lookup, so its validation errors surface even for unknown actions.
    """
    def handle(args: dict, **kw) -> str:
        action = str(args.get("action") or default).strip().lower()
        client = SpotifyClient()
        try:
            if prepare is not None:
                args = prepare(args)
            handler = table.get(action)
            if handler is None:
                return tool_error(f"Unknown {tool_name} action: {action}")
            return handler(client, args, action)
        except Exception as exc:
            return _spotify_tool_error(exc)
    handle.__name__ = handle.__qualname__ = f"_handle_{tool_name}"
    return handle


# -- spotify_playback ---------------------------------------------------------

# action -> (flag key reported False, fallback message) when Spotify returns 204/empty.
_EMPTY_PLAYBACK = {
    "get_currently_playing": ("is_playing", "Spotify is not currently playing anything."),
    "get_state": ("has_active_device", "No active Spotify playback session was found.")}


def _pb_read(fetch: Callable[..., Any], args: dict, action: str) -> str:
    payload = fetch(market=args.get("market"))
    if isinstance(payload, dict) and payload.get("empty"):
        flag, fallback = _EMPTY_PLAYBACK[action]
        payload = {"success": True, "action": action, flag: False, "status_code": payload.get("status_code", 204),
                   "message": payload.get("message") or fallback}
    return tool_result(payload)


_CONTEXT_TYPES = (("album", "spotify:album:", "/album/"), ("playlist", "spotify:playlist:", "/playlist/"), ("artist", "spotify:artist:", "/artist/"))


def _pb_play(client: SpotifyClient, args: dict, action: str) -> str:
    offset = args.get("offset")
    payload_offset = {k: v for k, v in offset.items() if v is not None} if isinstance(offset, dict) else None
    uris = normalize_spotify_uris(_as_list(args.get("uris")), "track") if args.get("uris") else None
    context_uri = None
    if args.get("context_uri"):
        raw = str(args.get("context_uri"))
        # Infer the context type so mismatches raise; unknown kinds pass through unchecked.
        context_type = next((t for t, prefix, frag in _CONTEXT_TYPES if raw.startswith(prefix) or frag in raw), None)
        context_uri = normalize_spotify_uri(raw, context_type)
    body = {"context_uri": context_uri, "uris": uris, "offset": payload_offset, "position_ms": args.get("position_ms")}
    return _ok(action, client.request("PUT", "/me/player/play", params={"device_id": args.get("device_id")}, json_body=body))


def _pb_cmd(client: SpotifyClient, args: dict, action: str, method: str, path: str, **extra: Any) -> str:
    """Player command whose only free argument is ``device_id`` (*extra* = fixed params)."""
    return _ok(action, client.request(method, path, params={**extra, "device_id": args.get("device_id")}))


_pb_device_cmd = lambda method, path: (lambda c, a, act: _pb_cmd(c, a, act, method, path))  # noqa: E731


def _pb_required_param(path: str, param: str, convert: Callable[[Any], Any]) -> _Handler:
    """PUT *path* with ``param`` (required, converted) plus device_id."""
    return lambda c, a, act: _pb_cmd(c, a, act, "PUT", path, **{param: convert(_required(a.get(param), f"{param} is required for action='{act}'"))})


def _pb_repeat_state(args: dict) -> str:
    state = str(args.get("state") or "").strip().lower()
    if state not in {"track", "context", "off"}:
        raise SpotifyError("state must be one of: track, context, off")
    return state


def _pb_recently_played(client: SpotifyClient, args: dict, action: str) -> str:
    after, before = args.get("after"), args.get("before")
    if after and before:
        return tool_error("Provide only one of 'after' or 'before'")
    params = {"limit": _limit(args), "after": int(after) if after is not None else None, "before": int(before) if before is not None else None}
    return tool_result(client.request("GET", "/me/player/recently-played", params=params))


_handle_spotify_playback = _dispatcher("spotify_playback", "get_state", {
    "get_state": lambda c, a, act: _pb_read(c.get_playback_state, a, act),
    "get_currently_playing": lambda c, a, act: _pb_read(c.get_currently_playing, a, act),
    "play": _pb_play,
    "pause": _pb_device_cmd("PUT", "/me/player/pause"),
    "next": _pb_device_cmd("POST", "/me/player/next"),
    "previous": _pb_device_cmd("POST", "/me/player/previous"),
    "seek": _pb_required_param("/me/player/seek", "position_ms", int),
    "set_repeat": lambda c, a, act: _pb_cmd(c, a, act, "PUT", "/me/player/repeat", state=_pb_repeat_state(a)),
    "set_shuffle": lambda c, a, act: _pb_cmd(c, a, act, "PUT", "/me/player/shuffle", state=str(_coerce_bool(a.get("state"))).lower()),
    "set_volume": _pb_required_param("/me/player/volume", "volume_percent", lambda v: max(0, min(100, int(v)))),
    "recently_played": _pb_recently_played,
})


# -- spotify_devices / spotify_queue / spotify_search ---------------------------

_handle_spotify_devices = _dispatcher("spotify_devices", "list", {
    "list": lambda c, a, act: tool_result(c.request("GET", "/me/player/devices")),
    "transfer": lambda c, a, act: _ok(act, c.request("PUT", "/me/player", json_body={
        "device_ids": [_nonblank(a.get("device_id"), "device_id is required for action='transfer'")], "play": _coerce_bool(a.get("play")),
    })),
})


def _queue_add(client: SpotifyClient, args: dict, action: str) -> str:
    uri = normalize_spotify_uri(str(args.get("uri") or ""), None)
    return _ok(action, client.request("POST", "/me/player/queue", params={"uri": uri, "device_id": args.get("device_id")}), uri=uri)


_handle_spotify_queue = _dispatcher("spotify_queue", "get", {
    "get": lambda c, a, act: tool_result(c.request("GET", "/me/player/queue")), "add": _queue_add})

_SEARCH_TYPES = {"album", "artist", "playlist", "track", "show", "episode", "audiobook"}


def _handle_spotify_search(args: dict, **kw) -> str:
    client = SpotifyClient()
    query = str(args.get("query") or "").strip()
    if not query:
        return tool_error("query is required")
    raw_types = _as_list(args.get("types") or args.get("type") or ["track"])
    search_types = [value.lower() for value in raw_types if value.lower() in _SEARCH_TYPES]
    if not search_types:
        return tool_error("types must contain one or more of: album, artist, playlist, track, show, episode, audiobook")
    params = {"q": query, "type": ",".join(search_types), "limit": _limit(args, 10), "offset": _offset(args),
              "market": args.get("market"), "include_external": args.get("include_external")}
    try:
        return tool_result(client.request("GET", "/search", params=params))
    except Exception as exc:
        return _spotify_tool_error(exc)


# -- spotify_playlists ---------------------------------------------------------

_playlist_path = lambda args, suffix="": f"/playlists/{normalize_spotify_id(str(args.get('playlist_id') or ''), 'playlist')}{suffix}"  # noqa: E731


_handle_spotify_playlists = _dispatcher("spotify_playlists", "list", {
    "list": lambda c, a, act: tool_result(c.request("GET", "/me/playlists", params={"limit": _limit(a), "offset": _offset(a)})),
    "get": lambda c, a, act: tool_result(c.request("GET", _playlist_path(a), params={"market": a.get("market")})),
    "create": lambda c, a, act: tool_result(c.request("POST", "/me/playlists", json_body={
        "name": _nonblank(a.get("name"), "name is required for action='create'"), "public": _coerce_bool(a.get("public")),
        "collaborative": _coerce_bool(a.get("collaborative")), "description": a.get("description")})),
    "add_items": lambda c, a, act: tool_result(c.request("POST", _playlist_path(a, "/items"), json_body={
        "uris": normalize_spotify_uris(_as_list(a.get("uris"))), "position": a.get("position")})),
    "remove_items": lambda c, a, act: tool_result(c.request("DELETE", _playlist_path(a, "/items"), json_body={
        "items": [{"uri": u} for u in normalize_spotify_uris(_as_list(a.get("uris")))], "snapshot_id": a.get("snapshot_id")})),
    "update_details": lambda c, a, act: tool_result(c.request("PUT", _playlist_path(a), json_body={
        "name": a.get("name"), "public": a.get("public"), "collaborative": a.get("collaborative"), "description": a.get("description")})),
})


# -- spotify_albums ------------------------------------------------------------

_page_params = lambda args: {"limit": _limit(args), "offset": _offset(args), "market": args.get("market")}  # noqa: E731
_prepare_album = lambda args: {**args, "_path": f"/albums/{normalize_spotify_id(str(args.get('album_id') or args.get('id') or ''), 'album')}"}  # noqa: E731


_handle_spotify_albums = _dispatcher("spotify_albums", "get", {
    "get": lambda c, a, act: tool_result(c.request("GET", a["_path"], params={"market": a.get("market")})),
    "tracks": lambda c, a, act: tool_result(c.request("GET", a["_path"] + "/tracks", params=_page_params(a))),
}, prepare=_prepare_album)


# -- spotify_library — saved tracks + saved albums, selected by `kind` ---------

def _lib_remove_uris(args: dict) -> str:
    item_type = args["_item_type"]
    ids = [normalize_spotify_id(item, item_type) for item in _as_list(args.get("ids") or args.get("items"))]
    return ",".join(f"spotify:{item_type}:{i}" for i in _required(ids or None, "ids/items is required for action='remove'"))


_dispatch_library = _dispatcher("spotify_library", "list", {
    "list": lambda c, a, act: tool_result(c.request("GET", f"/me/{a['kind']}", params=_page_params(a))),
    "save": lambda c, a, act: tool_result(c.request("PUT", "/me/library", params={
        "uris": ",".join(normalize_spotify_uris(_as_list(a.get("uris") or a.get("items")), a["_item_type"]))})),
    "remove": lambda c, a, act: tool_result(c.request("DELETE", "/me/library", params={"uris": _lib_remove_uris(a)})),
})


def _handle_spotify_library(args: dict, **kw) -> str:
    kind = str(args.get("kind") or "").strip().lower()
    if kind not in {"tracks", "albums"}:
        return tool_error("kind must be one of: tracks, albums")
    return _dispatch_library({**args, "kind": kind, "_item_type": kind[:-1]})  # tracks->track, albums->album


# -- Schemas (sent to the model — byte-stable; property order matters) ----------

COMMON_STRING = {"type": "string"}
_INT = {"type": "integer"}
_BOOL = {"type": "boolean"}
_STR_ARRAY = {"type": "array", "items": COMMON_STRING}


_strs = lambda *names: dict.fromkeys(names, COMMON_STRING)  # noqa: E731
_enum = lambda *values: {"type": "string", "enum": list(values)}  # noqa: E731
_idesc = lambda text: {"type": "integer", "description": text}  # noqa: E731


def _schema(name: str, description: str, properties: dict, required: tuple = ("action",)) -> dict:
    return {"name": name, "description": description, "parameters": {"type": "object", "properties": properties, "required": list(required)}}


SPOTIFY_PLAYBACK_SCHEMA = _schema("spotify_playback", "Control Spotify playback, inspect the active playback state, or fetch recently played tracks.", {
    "action": _enum("get_state", "get_currently_playing", "play", "pause", "next", "previous", "seek", "set_repeat", "set_shuffle", "set_volume", "recently_played"),
    **_strs("device_id", "market", "context_uri"), "uris": _STR_ARRAY, "offset": {"type": "object"}, "position_ms": _INT,
    "state": {"description": "For set_repeat use track/context/off. For set_shuffle use boolean-like true/false.", "oneOf": [{"type": "string"}, {"type": "boolean"}]},
    "volume_percent": _INT, "limit": _idesc("For recently_played: number of tracks (max 50)"),
    "after": _idesc("For recently_played: Unix ms cursor (after this timestamp)"), "before": _idesc("For recently_played: Unix ms cursor (before this timestamp)"),
})
SPOTIFY_DEVICES_SCHEMA = _schema("spotify_devices", "List Spotify Connect devices or transfer playback to a different device.",
                                 {"action": _enum("list", "transfer"), "device_id": COMMON_STRING, "play": _BOOL})
SPOTIFY_QUEUE_SCHEMA = _schema("spotify_queue", "Inspect the user's Spotify queue or add an item to it.", {"action": _enum("get", "add"), **_strs("uri", "device_id")})
SPOTIFY_SEARCH_SCHEMA = _schema(
    "spotify_search", "Search the Spotify catalog for tracks, albums, artists, playlists, shows, or episodes.",
    {"query": COMMON_STRING, "types": _STR_ARRAY, "type": COMMON_STRING, "limit": _INT, "offset": _INT, **_strs("market", "include_external")}, ("query",),
)
SPOTIFY_PLAYLISTS_SCHEMA = _schema("spotify_playlists", "List, inspect, create, update, and modify Spotify playlists.", {
    "action": _enum("list", "get", "create", "add_items", "remove_items", "update_details"),
    **_strs("playlist_id", "market"), "limit": _INT, "offset": _INT, **_strs("name", "description"),
    "public": _BOOL, "collaborative": _BOOL, "uris": _STR_ARRAY, "position": _INT, "snapshot_id": COMMON_STRING})
SPOTIFY_ALBUMS_SCHEMA = _schema("spotify_albums", "Fetch Spotify album metadata or album tracks.",
                                {"action": _enum("get", "tracks"), **_strs("album_id", "id", "market"), "limit": _INT, "offset": _INT})
SPOTIFY_LIBRARY_SCHEMA = _schema("spotify_library", "List, save, or remove the user's saved Spotify tracks or albums. Use `kind` to select which.", {
    "kind": {"type": "string", "enum": ["tracks", "albums"], "description": "Which library to operate on"},
    "action": _enum("list", "save", "remove"),
    "limit": _INT, "offset": _INT, "market": COMMON_STRING, "uris": _STR_ARRAY, "ids": _STR_ARRAY, "items": _STR_ARRAY,
}, ("kind", "action"))


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'SpotifyAPIError': ('plugins.spotify.client', 'SpotifyAPIError'),
    'SpotifyAuthRequiredError': ('plugins.spotify.client', 'SpotifyAuthRequiredError'),
}


def __getattr__(name):  # PEP 562 — lazy so no import cycles
    target = _PLUGIN_COMPAT_LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    from hermes_cli.plugin_compat import warn_once
    warn_once(__name__, name, *target)
    return getattr(importlib.import_module(target[0]), target[1])
# ---- END PLUGIN-COMPAT ----
