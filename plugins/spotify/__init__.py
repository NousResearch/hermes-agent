"""Spotify integration plugin — bundled, auto-loaded.

Registers 7 tools into the ``spotify`` toolset. Tools stay registered (so they
appear in ``hermes tools``) but ``_check_spotify_available()`` gates dispatch
until the user has run ``hermes auth spotify``.

Why a plugin rather than a ``tools/`` module: ``tools/`` is reserved for
foundational capabilities; third-party service integrations live under
``plugins/`` (flat ``plugins/<name>/`` for standalones, like image_gen backends),
and ``kind: backend`` bundled plugins auto-load with no ``plugins.enabled`` opt-in.
"""

from __future__ import annotations

from plugins.spotify import tools as _t

_TOOLS = (
    ("spotify_playback",  _t.SPOTIFY_PLAYBACK_SCHEMA,  _t._handle_spotify_playback,  "🎵"),
    ("spotify_devices",   _t.SPOTIFY_DEVICES_SCHEMA,   _t._handle_spotify_devices,   "🔈"),
    ("spotify_queue",     _t.SPOTIFY_QUEUE_SCHEMA,     _t._handle_spotify_queue,     "📻"),
    ("spotify_search",    _t.SPOTIFY_SEARCH_SCHEMA,    _t._handle_spotify_search,    "🔎"),
    ("spotify_playlists", _t.SPOTIFY_PLAYLISTS_SCHEMA, _t._handle_spotify_playlists, "📚"),
    ("spotify_albums",    _t.SPOTIFY_ALBUMS_SCHEMA,    _t._handle_spotify_albums,    "💿"),
    ("spotify_library",   _t.SPOTIFY_LIBRARY_SCHEMA,   _t._handle_spotify_library,   "❤️"),
)


def register(ctx) -> None:
    """Register all Spotify tools. Called once by the plugin loader."""
    for name, schema, handler, emoji in _TOOLS:
        ctx.register_tool(name=name, toolset="spotify", schema=schema, handler=handler, check_fn=_t._check_spotify_available, emoji=emoji)


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.


_PLUGIN_COMPAT_LAZY = {
    'SPOTIFY_ALBUMS_SCHEMA': ('plugins.spotify.tools', 'SPOTIFY_ALBUMS_SCHEMA'),
    'SPOTIFY_DEVICES_SCHEMA': ('plugins.spotify.tools', 'SPOTIFY_DEVICES_SCHEMA'),
    'SPOTIFY_LIBRARY_SCHEMA': ('plugins.spotify.tools', 'SPOTIFY_LIBRARY_SCHEMA'),
    'SPOTIFY_PLAYBACK_SCHEMA': ('plugins.spotify.tools', 'SPOTIFY_PLAYBACK_SCHEMA'),
    'SPOTIFY_PLAYLISTS_SCHEMA': ('plugins.spotify.tools', 'SPOTIFY_PLAYLISTS_SCHEMA'),
    'SPOTIFY_QUEUE_SCHEMA': ('plugins.spotify.tools', 'SPOTIFY_QUEUE_SCHEMA'),
    'SPOTIFY_SEARCH_SCHEMA': ('plugins.spotify.tools', 'SPOTIFY_SEARCH_SCHEMA'),
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
