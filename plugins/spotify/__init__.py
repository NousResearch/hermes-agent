"""Spotify integration plugin — bundled, auto-loaded.

Registers 7 tools (playback, devices, queue, search, playlists, albums,
library) into the ``spotify`` toolset. Each tool's handler is gated by
``_check_spotify_available()`` — when the user has not run ``hermes auth
spotify``, the tools remain registered (so they appear in ``hermes
tools``) but the runtime check prevents dispatch.

Why a plugin instead of a top-level ``tools/`` file?

- ``plugins/`` is where third-party service integrations live (see
  ``plugins/image_gen/`` for the backend-provider pattern, ``plugins/
  disk-cleanup/`` for the standalone pattern). ``tools/`` is reserved
  for foundational capabilities (terminal, read_file, web_search, etc.).
- Mirroring the image_gen plugin layout (``plugins/<category>/<backend>/``
  for categories, flat ``plugins/<name>/`` for standalones) makes new
  service integrations a pattern contributors can copy.
- Bundled + ``kind: backend`` auto-loads on startup just like image_gen
  backends — no user opt-in needed, no ``plugins.enabled`` config.

The Spotify auth flow (``hermes auth spotify``), CLI plumbing, and docs
are unchanged. This move is purely structural.
"""

from __future__ import annotations

from plugins.spotify.tools import (
    SPOTIFY_ALBUMS_SCHEMA,
    SPOTIFY_DEVICES_SCHEMA,
    SPOTIFY_LIBRARY_SCHEMA,
    SPOTIFY_PLAYBACK_SCHEMA,
    SPOTIFY_PLAYLISTS_SCHEMA,
    SPOTIFY_QUEUE_SCHEMA,
    SPOTIFY_SEARCH_SCHEMA,
    _check_spotify_available,
    _handle_spotify_albums,
    _handle_spotify_devices,
    _handle_spotify_library,
    _handle_spotify_playback,
    _handle_spotify_playlists,
    _handle_spotify_queue,
    _handle_spotify_search,
)

_TOOLS = (
    ("spotify_playback",  SPOTIFY_PLAYBACK_SCHEMA,  _handle_spotify_playback,  "🎵"),
    ("spotify_devices",   SPOTIFY_DEVICES_SCHEMA,   _handle_spotify_devices,   "🔈"),
    ("spotify_queue",     SPOTIFY_QUEUE_SCHEMA,     _handle_spotify_queue,     "📻"),
    ("spotify_search",    SPOTIFY_SEARCH_SCHEMA,    _handle_spotify_search,    "🔎"),
    ("spotify_playlists", SPOTIFY_PLAYLISTS_SCHEMA, _handle_spotify_playlists, "📚"),
    ("spotify_albums",    SPOTIFY_ALBUMS_SCHEMA,    _handle_spotify_albums,    "💿"),
    ("spotify_library",   SPOTIFY_LIBRARY_SCHEMA,   _handle_spotify_library,   "❤️"),
)


def register(ctx) -> None:
    """Register all Spotify tools. Called once by the plugin loader."""
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(
            name=name,
            toolset="spotify",
            schema=schema,
            handler=handler,
            check_fn=_check_spotify_available,
            emoji=emoji,
        )
