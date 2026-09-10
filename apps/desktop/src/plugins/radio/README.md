# Radio

Bundled Hermes Desktop plugin, **off by default**. Enable **Radio** in
Settings → Plugins. Playback starts only when you press Play.

## Controls

The status-bar widget has a compact braille waveform, station picker,
play/pause, and double-arrow Next. The picker includes pinned stations,
search, a larger waveform, available song metadata, volume, and a broadcaster
website link. Next changes stations; live broadcasts do not support skipping
songs. Pause releases the stream; playing again returns to live.

Presets include Nightride FM, Radio Paradise, and EVE Radio (GamingNow).
Search combines local stations with [Radio Browser](https://api.radio-browser.info/)
results, accepts HTTPS non-HLS streams, and deduplicates by ID and URL.
Pins, station, and volume use plugin-scoped local storage.

## Plugin boundary

`plugin.js` imports only `@hermes/plugin-sdk`, `react`, and
`react/jsx-runtime`. Existing bundled discovery finds it automatically;
`defaultEnabled: false` uses the ordinary live enable toggle. No shell,
SDK, backend, dependency, or registry changes are needed.

The same plain-ESM file can be loaded through the runtime plugin door at
`$HERMES_HOME/desktop-plugins/radio/plugin.js` for development. Do not install
that duplicate alongside the bundled version. Layout styles use Hermes theme
tokens and are removed on disable. Playback, timers, and audio nodes are also
released. Closing the window stops playback; there is no background audio
service. Other windows pause when a window starts playing.

## Audio and privacy

Both waveform sizes use real audio samples, with two dim previous traces.
Silence stays flat; reduced motion freezes the display. Streams that cannot
be analysed retry without CORS and show an activity indicator, not invented
levels. Nightride and EVE Radio song metadata is fetched while the picker is
open and playback is active. Artist names link directly to their profile, never
search results or a song. Apple's public artist lookup resolves unambiguous exact
names to Apple Music profiles. MusicBrainz is a fallback for Spotify, Bandcamp,
or official artist homepages. Missing or ambiguous matches stay plain text, with
no dead link or reserved icon gap. Artist names are sent to these services only
while the picker shows live metadata; results are cached for a day. MusicBrainz
requests are paced across windows. No account or API key is needed.

Audio connects directly to broadcasters. Searches send the typed station query
to Radio Browser. There are no accounts, keys, analytics, recording, or
rebroadcasting. Disabled plugins perform no work; enabling does not autoplay.
Stream availability, regional restrictions, and metadata vary. Hermes is not
endorsed by these broadcasters.

## References

- [Nightride streams](https://stream.nightride.fm/)
- [Radio Paradise stream links](https://radioparadise.com/listen/stream-links)
- [EVE Radio / GamingNow](https://gamingnow.net/eve-radio/)
- [Original EVE jukebox](https://ashy.vargur.dev/the-eve-online-jukebox-project/): compact player/list hierarchy.
- [node-drawille](https://github.com/madbence/node-drawille) (MIT): braille pixel packing prior art.
- [Phosphor](https://github.com/hubertlim/oscilloscope_playground) (MIT): previous-frame persistence inspiration.

No upstream code, assets, shaders, or dependencies are copied.
