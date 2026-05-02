# lastfm plugin

Last.fm music discovery and metadata for Hermes — 5 tools covering
similarity-based discovery, artist/track/tag metadata, and trending charts.
**No OAuth required** — only a free API key.

Get a free key at: https://www.last.fm/api/account/create

## Setup

```bash
export LASTFM_API_KEY="your_key_here"
# or add to ~/.hermes/.env:
# LASTFM_API_KEY=your_key_here
```

The plugin gates all tools on `LASTFM_API_KEY` being set. Tools appear in
`hermes tools` but return a helpful error if the key is missing.

## Tools

| Tool | Description |
|---|---|
| `lastfm_discover` | Multi-seed similarity discovery — find music similar to one or more artists/tracks |
| `lastfm_artist` | Artist biography, similar artists, top tracks/albums, genre tags, artist search |
| `lastfm_track` | Track info (playcount, album, wiki), similar tracks, track search |
| `lastfm_tag` | Genre/mood exploration — top artists, tracks, albums for a tag; related tags |
| `lastfm_charts` | Trending charts globally or by country |

## Usage examples

### Discovery

```
lastfm_discover(artists=["Boards of Canada", "Aphex Twin"], count=10)
lastfm_discover(artists=["Boards of Canada"], tracks=["Autechre:Gantz Graf"], scoring="boost")
lastfm_discover(tracks=["Radiohead:Exit Music (For a Film)"], count=15)
```

**Scoring modes** (when multiple seeds are provided):

| Mode | Formula | Best for |
|---|---|---|
| `avg` | `sum(matches) / total_seeds` | General use — cross-seed artists naturally rank higher |
| `max` | `best single-seed match` | Pure similarity, ignores seed count |
| `boost` | `avg × (1 + 0.3 × extra_seeds)` | Explicitly rewards artists that appear across many seeds |

### Artist metadata

```
lastfm_artist(action="info",       artist="Aphex Twin")
lastfm_artist(action="similar",    artist="Boards of Canada", limit=15)
lastfm_artist(action="top_tracks", artist="Autechre", limit=10)
lastfm_artist(action="top_albums", artist="Radiohead")
lastfm_artist(action="top_tags",   artist="Four Tet")
lastfm_artist(action="search",     artist="Burial")
```

### Track metadata

```
lastfm_track(action="info",    artist="Boards of Canada", track="Roygbiv")
lastfm_track(action="similar", artist="Aphex Twin",       track="Windowlicker", limit=20)
lastfm_track(action="search",  track="Avril 14th")
```

### Genre exploration

```
lastfm_tag(action="top_artists", tag="ambient",    limit=20)
lastfm_tag(action="top_tracks",  tag="post-rock")
lastfm_tag(action="top_albums",  tag="shoegaze")
lastfm_tag(action="similar",     tag="drone")
lastfm_tag(action="info",        tag="IDM")
```

### Charts

```
lastfm_charts(action="top_tracks", limit=20)
lastfm_charts(action="top_artists", country="Germany", limit=10)
lastfm_charts(action="top_tracks",  country="Japan")
```

## Architecture

```
plugins/lastfm/
  plugin.yaml   — manifest (name, version, requires_env: [LASTFM_API_KEY])
  client.py     — LastFmClient: thin urllib wrapper, no dependencies
  tools.py      — 5 handlers + JSON schemas + _check_lastfm_available()
  __init__.py   — register() wires tools into the "lastfm" toolset
```

`client.py` uses only the Python standard library (`urllib`, `json`) — no
`httpx`, `requests`, or other third-party packages needed.

## API coverage

| Last.fm method | Used by |
|---|---|
| `artist.getSimilar` | `lastfm_discover`, `lastfm_artist similar` |
| `artist.getTopTracks` | `lastfm_discover`, `lastfm_artist top_tracks` |
| `artist.getTopAlbums` | `lastfm_artist top_albums` |
| `artist.getTopTags` | `lastfm_discover`, `lastfm_artist top_tags` |
| `artist.getInfo` | `lastfm_discover`, `lastfm_artist info` |
| `artist.search` | `lastfm_artist search` |
| `track.getSimilar` | `lastfm_discover`, `lastfm_track similar` |
| `track.getInfo` | `lastfm_track info` |
| `track.search` | `lastfm_track search` |
| `tag.getTopArtists` | `lastfm_tag top_artists` |
| `tag.getTopTracks` | `lastfm_tag top_tracks` |
| `tag.getTopAlbums` | `lastfm_tag top_albums` |
| `tag.getSimilar` | `lastfm_tag similar` |
| `tag.getInfo` | `lastfm_tag info` |
| `chart.getTopArtists` | `lastfm_charts top_artists` (global) |
| `chart.getTopTracks` | `lastfm_charts top_tracks` (global) |
| `geo.getTopArtists` | `lastfm_charts top_artists` (country) |
| `geo.getTopTracks` | `lastfm_charts top_tracks` (country) |

## Last.fm API Terms of Service

- **Attribution** — The [Last.fm API ToS](https://www.last.fm/api/tos) requires displaying "Powered by Last.fm" attribution wherever their data appears. All tool responses include a `_source` field (`"Powered by Last.fm"`) for this purpose. Applications built on top of this plugin should surface that attribution to end users.
- **Commercial use** — The free API key is for non-commercial use only. If you're building a commercial product, contact Last.fm at partners@last.fm before deploying.
- **Rate limit** — ~5 requests per second. `lastfm_discover` with multiple seeds makes several sequential calls; stay within this limit.

## Notes

- All API calls are read-only — no write/scrobble endpoints are used.
- Last.fm free API keys have a rate limit of ~5 req/s. `lastfm_discover`
  with many seeds makes multiple calls; for large seed lists the `max`
  scoring mode makes fewer requests (one call per seed, no enrichment pass).
- The `lastfm_discover` response includes `seeds_matched` and `matched_seeds`
  per result so the agent can explain *why* an artist was recommended.
- User scrobble endpoints (`user.getTopArtists` etc.) are available in
  `client.py` but not exposed as tools — they require a Last.fm username.
  A `lastfm_user` tool can be added as a follow-up if there's demand.
