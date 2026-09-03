"""Discord music queue and playback-control behavior."""

import asyncio
import threading
import time
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from plugins.platforms.discord.music import (
    BufferedAudioSource,
    DiscordMusicManager,
    MusicControlView,
    MusicResolutionError,
    MusicSession,
    MusicTrack,
    TrackResolver,
    parse_natural_music_control,
    parse_natural_play_request,
)
from gateway.config import PlatformConfig
from plugins.platforms.discord.adapter import DiscordAdapter


def _track(title: str, requester_id: int = 42) -> MusicTrack:
    return MusicTrack(
        title=title,
        webpage_url=f"https://example.com/{title}",
        lookup=f"ytsearch1:{title}",
        requester_id=requester_id,
        requester_name=f"user-{requester_id}",
    )


def test_natural_play_request_extracts_song_title_and_artist():
    assert (
        parse_natural_play_request("Hey Jarvis, play Paranoid by Rich Amiri")
        == "Paranoid by Rich Amiri"
    )


def test_natural_play_request_preserves_supported_link_queries():
    url = "https://www.youtube.com/watch?v=example"
    assert parse_natural_play_request(f"Jarvis play {url}") == url


def test_natural_play_request_ignores_unaddressed_play_chatter():
    assert parse_natural_play_request("play Paranoid by Rich Amiri") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hey Jarvis pause the song", "pause"),
        ("Jarvis resume the music", "resume"),
        ("Jarvis play", "resume"),
        ("Jarvis play the current song", "resume"),
        ("Jarvis play the next song", "skip"),
        ("Jarvis skip this song", "skip"),
        ("Jarvis go back to the previous song", "previous"),
        ("Jarvis repeat this song", "repeat"),
        ("Jarvis show the music queue", "queue"),
        ("Jarvis clear the music queue", "clear"),
    ],
)
def test_natural_music_controls_are_distinguished_from_track_requests(text, expected):
    assert parse_natural_music_control(text) == expected
    assert parse_natural_play_request(text) is None


@pytest.mark.asyncio
async def test_slash_music_errors_do_not_expose_exception_details():
    adapter = object.__new__(DiscordAdapter)
    interaction = SimpleNamespace(
        response=SimpleNamespace(
            is_done=MagicMock(return_value=False),
            send_message=AsyncMock(),
        )
    )

    await adapter._send_music_error(
        interaction, RuntimeError("@everyone signed-url backend detail")
    )

    interaction.response.send_message.assert_awaited_once_with(
        "I couldn't complete that music request. Please try again.", ephemeral=True
    )


def test_music_session_exposes_now_playing_and_fifo_queue():
    session = MusicSession(guild_id=7)
    first = _track("first")
    second = _track("second", requester_id=99)

    session.enqueue([first, second])
    session.current = session.queue.popleft()

    assert session.current is first
    assert list(session.queue) == [second]
    assert "first" in session.render_queue()
    assert "second" in session.render_queue()
    assert "user-99" in session.render_queue()


def test_only_current_requester_or_administrator_can_use_playback_controls():
    session = MusicSession(guild_id=7, current=_track("current", requester_id=42))

    assert session.can_control(user_id=42, administrator=False)
    assert not session.can_control(user_id=99, administrator=False)
    assert session.can_control(user_id=99, administrator=True)


def test_spotify_track_is_translated_to_a_searchable_audio_track():
    resolver = TrackResolver(
        spotify_metadata=lambda _url: [
            {
                "title": "Cut To The Feeling",
                "artist": "Carly Rae Jepsen",
                "webpage_url": "https://open.spotify.com/track/abc",
                "thumbnail": "https://i.scdn.co/image/cover",
            }
        ]
    )

    tracks = resolver.resolve(
        "https://open.spotify.com/track/abc",
        requester_id=42,
        requester_name="nahv",
    )

    assert len(tracks) == 1
    assert tracks[0].title == "Cut To The Feeling — Carly Rae Jepsen"
    assert tracks[0].lookup == "ytsearch5:Cut To The Feeling Carly Rae Jepsen audio"
    assert tracks[0].requester_id == 42


def test_spotify_public_metadata_fallback_extracts_artist_title_and_cover():
    class Response:
        def __init__(
            self, *, text="", payload=None, url="https://open.spotify.com/track/abc"
        ):
            self.text = text
            self._payload = payload
            self.url = url

        def raise_for_status(self):
            return None

        def json(self):
            return self._payload

    def http_get(url, **kwargs):
        if "/oembed" in url:
            return Response(
                payload={"title": "A Song", "thumbnail_url": "https://i.scdn.co/cover"}
            )
        return Response(
            text='<meta property="og:description" content="Artist Name · A Song · Song · 2026">'
        )

    resolver = TrackResolver(http_get=http_get)
    tracks = resolver.resolve(
        "https://open.spotify.com/track/abc",
        requester_id=42,
        requester_name="nahv",
    )

    assert tracks[0].title == "A Song — Artist Name"
    assert tracks[0].thumbnail == "https://i.scdn.co/cover"


def test_spotify_album_and_playlist_links_are_rejected_explicitly():
    resolver = TrackResolver(http_get=MagicMock(), url_guard=lambda _url: True)

    with pytest.raises(MusicResolutionError, match="Spotify track links"):
        resolver.resolve(
            "https://open.spotify.com/playlist/abc",
            requester_id=42,
            requester_name="nahv",
        )


def test_spotify_short_link_cannot_redirect_to_a_collection():
    class Response:
        text = ""
        url = "https://open.spotify.com/playlist/abc"

        def raise_for_status(self):
            return None

        def json(self):
            return {"title": "Playlist", "thumbnail_url": None}

    resolver = TrackResolver(
        http_get=lambda *_args, **_kwargs: Response(), url_guard=lambda _url: True
    )
    with pytest.raises(MusicResolutionError, match="Spotify track links"):
        resolver.resolve(
            "https://spotify.link/short",
            requester_id=42,
            requester_name="nahv",
        )


def test_youtube_link_metadata_and_fresh_audio_stream_are_resolved_lazily():
    calls = []

    def extract(query, *, flat):
        calls.append((query, flat))
        if flat:
            return {
                "title": "A Song",
                "webpage_url": "https://www.youtube.com/watch?v=abc",
                "duration": 123,
                "thumbnail": "https://i.ytimg.com/abc.jpg",
            }
        return {
            "title": "A Song",
            "webpage_url": "https://www.youtube.com/watch?v=abc",
            "url": "https://rr.example.googlevideo.com/audio",
        }

    resolver = TrackResolver(media_extractor=extract, url_guard=lambda _url: True)
    tracks = resolver.resolve(
        "https://www.youtube.com/watch?v=abc",
        requester_id=42,
        requester_name="nahv",
    )

    assert tracks[0].title == "A Song"
    assert calls == [("https://www.youtube.com/watch?v=abc", True)]
    assert (
        resolver.resolve_stream(tracks[0]) == "https://rr.example.googlevideo.com/audio"
    )
    assert calls[-1] == ("https://www.youtube.com/watch?v=abc", False)


def test_text_search_queues_one_result_but_keeps_playback_fallback_candidates():
    def extract(query, *, flat):
        assert query == "ytsearch5:requested song"
        assert flat is True
        return {
            "entries": [
                {
                    "title": "First result",
                    "webpage_url": "https://www.youtube.com/watch?v=first",
                },
                {
                    "title": "Second result",
                    "webpage_url": "https://www.youtube.com/watch?v=second",
                },
            ]
        }

    tracks = TrackResolver(media_extractor=extract).resolve(
        "requested song",
        requester_id=42,
        requester_name="nahv",
    )

    assert [track.title for track in tracks] == ["First result"]
    assert tracks[0].lookup == "ytsearch5:requested song"


def test_ytdlp_prefers_audio_or_bounded_quality_hls_and_enables_packaged_js_runtime(
    monkeypatch,
):
    captured = {}

    class FakeYoutubeDL:
        def __init__(self, options):
            captured.update(options)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def extract_info(self, query, download):
            assert query == "https://www.youtube.com/watch?v=abc"
            assert download is False
            return {"url": "https://manifest.googlevideo.com/audio.m3u8"}

    monkeypatch.setattr("yt_dlp.YoutubeDL", FakeYoutubeDL)

    TrackResolver._extract_media(
        "https://www.youtube.com/watch?v=abc",
        flat=False,
    )

    assert captured["format"] == (
        "bestaudio[protocol^=m3u8]/"
        "best[protocol^=m3u8][acodec!=none][height<=480]/"
        "worst[protocol^=m3u8][acodec!=none]/"
        "bestaudio/best"
    )
    assert captured["ignoreerrors"] is True
    assert captured["extractor_args"] == {
        "youtube": {"player_client": ["android"]}
    }
    assert "node" in captured["js_runtimes"]


def test_non_http_or_untrusted_extractor_streams_are_rejected():
    track = _track("unsafe")
    file_resolver = TrackResolver(
        media_extractor=lambda *_args, **_kwargs: {"url": "file:///etc/passwd"},
        url_guard=lambda _url: True,
    )
    with pytest.raises(MusicResolutionError, match="HTTP or HTTPS"):
        file_resolver.resolve_stream(track)

    untrusted_resolver = TrackResolver(
        media_extractor=lambda *_args, **_kwargs: {
            "url": "https://attacker.example/audio"
        },
        url_guard=lambda _url: True,
    )
    with pytest.raises(MusicResolutionError, match="trusted media host"):
        untrusted_resolver.resolve_stream(track)


def test_youtube_hls_is_limited_to_provider_controlled_manifest_host():
    track = MusicTrack(
        title="youtube",
        webpage_url="https://www.youtube.com/watch?v=abc",
        lookup="https://www.youtube.com/watch?v=abc",
        requester_id=42,
        requester_name="nahv",
    )
    resolver = TrackResolver(
        media_extractor=lambda *_args, **_kwargs: {
            "url": "https://attacker.akamaized.net/playlist.m3u8",
            "protocol": "m3u8_native",
        },
        url_guard=lambda _url: True,
    )

    with pytest.raises(MusicResolutionError, match="trusted manifest host"):
        resolver.resolve_stream(track)


def test_non_youtube_multitenant_cdn_stream_is_rejected():
    track = _track("unsafe")
    resolver = TrackResolver(
        media_extractor=lambda *_args, **_kwargs: {
            "url": "https://attacker.akamaized.net/audio.mp4"
        },
        url_guard=lambda _url: True,
    )

    with pytest.raises(MusicResolutionError, match="trusted media host"):
        resolver.resolve_stream(track)


def test_ffmpeg_source_restricts_nested_network_protocols(monkeypatch):
    upstream = MagicMock()
    ffmpeg = MagicMock(return_value=upstream)
    monkeypatch.setattr("discord.FFmpegPCMAudio", ffmpeg)

    source = DiscordMusicManager._default_audio_source(
        "https://manifest.googlevideo.com/audio.m3u8"
    )

    assert isinstance(source, BufferedAudioSource)
    assert source.source is upstream
    before_options = ffmpeg.call_args.kwargs["before_options"]
    options = ffmpeg.call_args.kwargs["options"]
    assert ffmpeg.call_args.kwargs["executable"]
    assert "-protocol_whitelist http,https,tcp,tls,crypto" in before_options
    assert "file" not in before_options
    assert "aresample=48000" in options
    assert "resampler=soxr" not in options
    source.cleanup()


def test_buffered_audio_source_returns_silence_instead_of_blocking_on_underflow():
    first = b"a" * BufferedAudioSource.FRAME_SIZE
    second = b"b" * BufferedAudioSource.FRAME_SIZE
    release_second = threading.Event()

    class StallingSource:
        def __init__(self):
            self.reads = 0
            self.cleaned = False

        def read(self):
            self.reads += 1
            if self.reads == 1:
                return first
            if self.reads == 2:
                release_second.wait(timeout=2)
                return second
            return b""

        def cleanup(self):
            self.cleaned = True

        def is_opus(self):
            return False

    upstream = StallingSource()
    source = BufferedAudioSource(
        upstream,
        prebuffer_frames=1,
        max_buffer_frames=4,
    )
    assert source.wait_until_ready(timeout=1)
    assert source.read() == first

    started = time.perf_counter()
    assert source.read() == BufferedAudioSource.SILENCE_FRAME
    assert time.perf_counter() - started < 0.05

    release_second.set()
    deadline = time.monotonic() + 1
    frame = BufferedAudioSource.SILENCE_FRAME
    while frame == BufferedAudioSource.SILENCE_FRAME and time.monotonic() < deadline:
        time.sleep(0.01)
        frame = source.read()
    assert frame == second

    source.cleanup()
    assert upstream.cleaned is True


def test_buffered_audio_source_ends_a_permanently_stalled_stream():
    release_read = threading.Event()

    class StuckSource:
        def read(self):
            release_read.wait(timeout=2)
            return b""

        def cleanup(self):
            release_read.set()

    source = BufferedAudioSource(
        StuckSource(),
        prebuffer_frames=1,
        max_buffer_frames=2,
        stall_timeout=0.1,
    )
    try:
        assert source.read() == BufferedAudioSource.SILENCE_FRAME
        time.sleep(0.11)
        assert source.read() == b""
        assert "stalled" in str(source._current_error).lower()
    finally:
        source.cleanup()


@pytest.mark.asyncio
async def test_startup_cancellation_cleans_buffered_source():
    release_read = threading.Event()

    class BlockingSource:
        cleaned = False

        def read(self):
            release_read.wait(timeout=2)
            return b""

        def cleanup(self):
            self.cleaned = True
            release_read.set()

    upstream = BlockingSource()
    source = BufferedAudioSource(upstream, prebuffer_frames=1, max_buffer_frames=2)
    vc = MagicMock()
    vc.is_connected.return_value = True
    adapter = SimpleNamespace(_voice_clients={7: vc})
    resolver = MagicMock()
    resolver.resolve_stream.return_value = "https://cdn.example/audio"
    manager = DiscordMusicManager(
        adapter,
        resolver=resolver,
        audio_source_factory=lambda _url: source,
    )
    session = MusicSession(guild_id=7, queue=deque([_track("cancelled")]))

    task = asyncio.create_task(manager._start_next(session))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    try:
        assert upstream.cleaned is True
    finally:
        source.cleanup()


@pytest.mark.asyncio
async def test_startup_decoder_error_is_reported_instead_of_buffer_timeout():
    class BrokenSource:
        def read(self):
            raise RuntimeError("decoder exploded")

        def cleanup(self):
            return None

    source = BufferedAudioSource(BrokenSource(), prebuffer_frames=1)
    vc = MagicMock()
    vc.is_connected.return_value = True
    adapter = SimpleNamespace(
        _voice_clients={7: vc},
        _voice_receivers={},
        _voice_mixers={},
    )
    resolver = MagicMock()
    resolver.resolve_stream.return_value = "https://cdn.example/audio"
    manager = DiscordMusicManager(
        adapter,
        resolver=resolver,
        audio_source_factory=lambda _url: source,
    )
    session = MusicSession(guild_id=7, queue=deque([_track("broken")]))
    session.panel_message = SimpleNamespace(edit=AsyncMock())
    session.text_channel = SimpleNamespace()

    await manager._start_next(session)

    assert session.last_error is not None
    assert "decoder exploded" in session.last_error


@pytest.mark.asyncio
async def test_missing_voice_connection_is_rejected_before_audio_source_creation():
    factory = MagicMock()
    adapter = SimpleNamespace(
        _voice_clients={},
        _voice_receivers={},
        _voice_mixers={},
    )
    resolver = MagicMock()
    resolver.resolve_stream.return_value = "https://cdn.example/audio"
    manager = DiscordMusicManager(adapter, resolver=resolver, audio_source_factory=factory)
    session = MusicSession(guild_id=7, queue=deque([_track("disconnected")]))
    session.panel_message = SimpleNamespace(edit=AsyncMock())
    session.text_channel = SimpleNamespace()

    await manager._start_next(session)

    factory.assert_not_called()
    assert session.last_error is not None
    assert "voice connection was lost" in session.last_error.lower()


def test_spotify_http_uses_connect_time_ssrf_safe_client(monkeypatch):
    client = MagicMock()
    client.get.return_value = "response"
    context = MagicMock()
    context.__enter__.return_value = client
    monkeypatch.setattr(
        "tools.url_safety.create_ssrf_safe_client",
        MagicMock(return_value=context),
    )

    result = TrackResolver._default_http_get(
        "https://open.spotify.com/track/abc",
        follow_redirects=True,
        timeout=15.0,
    )

    assert result == "response"
    client.get.assert_called_once_with("https://open.spotify.com/track/abc")


def test_private_or_local_media_urls_are_rejected_before_extraction():
    extractor = MagicMock()
    resolver = TrackResolver(
        media_extractor=extractor,
        url_guard=lambda url: not url.startswith("http://127.0.0.1"),
    )

    with pytest.raises(MusicResolutionError, match="private or local"):
        resolver.resolve(
            "http://127.0.0.1:8080/secrets",
            requester_id=42,
            requester_name="nahv",
        )

    extractor.assert_not_called()


def test_generic_web_urls_without_a_supported_extractor_are_rejected():
    extractor = MagicMock()
    resolver = TrackResolver(
        media_extractor=extractor,
        url_guard=lambda _url: True,
        url_support_checker=lambda _url: False,
    )

    with pytest.raises(MusicResolutionError, match="supported streaming platform"):
        resolver.resolve(
            "https://attacker.example/redirect",
            requester_id=42,
            requester_name="nahv",
        )

    extractor.assert_not_called()


def test_only_curated_streaming_hosts_are_allowed_for_ytdlp():
    assert TrackResolver._has_supported_extractor("https://www.youtube.com/watch?v=abc")
    assert TrackResolver._has_supported_extractor("https://youtu.be/abc")
    assert not TrackResolver._has_supported_extractor(
        "https://soundcloud.com/artist/track"
    )
    assert not TrackResolver._has_supported_extractor("https://vimeo.com/123")
    assert not TrackResolver._has_supported_extractor(
        "https://www.facebook.com/watch/abc"
    )


def test_playlist_entry_lookup_is_revalidated_before_playback():
    extractor = MagicMock(
        return_value={
            "entries": [
                {
                    "title": "unsafe",
                    "webpage_url": "http://127.0.0.1/private",
                }
            ]
        }
    )
    resolver = TrackResolver(
        media_extractor=extractor,
        url_guard=lambda url: "127.0.0.1" not in url,
        url_support_checker=lambda _url: True,
    )

    with pytest.raises(MusicResolutionError, match="playlist entry"):
        resolver.resolve(
            "https://www.youtube.com/playlist?list=abc",
            requester_id=42,
            requester_name="nahv",
        )

    extractor.assert_called_once()


@pytest.mark.asyncio
async def test_panel_disables_mentions_with_compatible_allowed_mentions(monkeypatch):
    import discord

    class CompatibleAllowedMentions:
        def __init__(self, *, everyone=True, roles=True, users=True, replied_user=True):
            self.everyone = everyone
            self.roles = roles
            self.users = users
            self.replied_user = replied_user

    monkeypatch.setattr(discord, "AllowedMentions", CompatibleAllowedMentions)
    manager = DiscordMusicManager(SimpleNamespace(_voice_clients={}))
    session = MusicSession(guild_id=7, current=_track("@everyone"))
    session.panel_message = SimpleNamespace(edit=AsyncMock())
    session.text_channel = SimpleNamespace()

    await manager._update_panel(session)

    mentions = session.panel_message.edit.await_args.kwargs["allowed_mentions"]
    assert mentions.everyone is False
    assert mentions.roles is False
    assert mentions.users is False
    assert mentions.replied_user is False


@pytest.mark.asyncio
async def test_add_joins_requesters_vc_starts_fifo_playback_and_updates_public_panel():
    track = _track("first")
    resolver = MagicMock()
    resolver.resolve.return_value = [track]
    resolver.resolve_stream.return_value = "https://cdn.example/audio"

    panel = SimpleNamespace(edit=AsyncMock())
    text_channel = SimpleNamespace(send=AsyncMock(return_value=panel))
    voice_channel = SimpleNamespace(id=12, guild=SimpleNamespace(id=7))
    voice_client = MagicMock()
    voice_client.is_connected.return_value = True
    voice_client.is_playing.return_value = False
    adapter = SimpleNamespace(
        _voice_clients={7: voice_client},
        _voice_receivers={},
        _voice_mixers={},
        join_voice_channel=AsyncMock(return_value=True),
        _cancel_voice_timeout=MagicMock(),
    )
    interaction = SimpleNamespace(
        guild=SimpleNamespace(id=7),
        channel=text_channel,
        user=SimpleNamespace(
            id=42,
            display_name="nahv",
            voice=SimpleNamespace(channel=voice_channel),
        ),
        response=SimpleNamespace(defer=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )
    source = object()
    manager = DiscordMusicManager(
        adapter,
        resolver=resolver,
        audio_source_factory=lambda _url: source,
        view_factory=lambda _manager, _guild_id: "controls",
    )

    await manager.add(interaction, "first")

    adapter.join_voice_channel.assert_awaited_once_with(
        voice_channel,
        text_channel_id=None,
    )
    voice_client.play.assert_called_once()
    assert voice_client.play.call_args.args[0] is source
    assert voice_client.play.call_args.kwargs["bitrate"] == 192
    assert voice_client.play.call_args.kwargs["signal_type"] == "music"
    assert manager.sessions[7].current is track
    text_channel.send.assert_awaited_once()
    panel.edit.assert_awaited()
    assert "first" in panel.edit.await_args.kwargs["content"]


@pytest.mark.asyncio
async def test_add_message_accepts_a_song_name_without_an_interaction_or_link(monkeypatch):
    import discord

    class CompatibleAllowedMentions:
        def __init__(self, *, everyone=True, roles=True, users=True, replied_user=True):
            self.everyone = everyone
            self.roles = roles
            self.users = users
            self.replied_user = replied_user

    monkeypatch.setattr(discord, "AllowedMentions", CompatibleAllowedMentions)
    track = _track("Paranoid — Rich Amiri")
    resolver = MagicMock()
    resolver.resolve.return_value = [track]
    resolver.resolve_stream.return_value = "https://cdn.example/audio"
    panel = SimpleNamespace(edit=AsyncMock())
    text_channel = SimpleNamespace(id=99, send=AsyncMock(return_value=panel))
    voice_channel = SimpleNamespace(id=12, guild=SimpleNamespace(id=7))
    voice_client = MagicMock()
    voice_client.is_connected.return_value = True
    voice_client.is_playing.return_value = False
    adapter = SimpleNamespace(
        _voice_clients={7: voice_client},
        _voice_receivers={},
        _voice_mixers={},
        join_voice_channel=AsyncMock(return_value=True),
        _cancel_voice_timeout=MagicMock(),
    )
    message = SimpleNamespace(
        guild=SimpleNamespace(id=7),
        channel=text_channel,
        author=SimpleNamespace(
            id=42,
            display_name="nahv",
            voice=SimpleNamespace(channel=voice_channel),
        ),
    )
    manager = DiscordMusicManager(
        adapter,
        resolver=resolver,
        audio_source_factory=lambda _url: object(),
        view_factory=lambda *_args: "controls",
    )

    await manager.add_message(message, "Paranoid by Rich Amiri")

    resolver.resolve.assert_called_once_with(
        "Paranoid by Rich Amiri",
        requester_id=42,
        requester_name="nahv",
    )
    adapter.join_voice_channel.assert_awaited_once_with(
        voice_channel,
        text_channel_id=99,
    )
    assert manager.sessions[7].current is track
    assert text_channel.send.await_count == 2
    ack_mentions = text_channel.send.await_args_list[1].kwargs["allowed_mentions"]
    assert ack_mentions.everyone is False
    assert ack_mentions.roles is False
    assert ack_mentions.users is False


@pytest.mark.asyncio
async def test_add_request_waits_for_guild_lock_before_joining_voice():
    resolver = MagicMock()
    resolver.resolve.return_value = [_track("queued")]
    adapter = SimpleNamespace(join_voice_channel=AsyncMock(return_value=True))
    manager = DiscordMusicManager(adapter, resolver=resolver)
    manager._update_panel = AsyncMock()
    manager._start_next = AsyncMock()
    guild = SimpleNamespace(id=7)
    user = SimpleNamespace(id=42, display_name="nahv")
    text_channel = SimpleNamespace(id=99)
    voice_channel = SimpleNamespace(id=12)
    lock = manager._locks.setdefault(7, asyncio.Lock())
    await lock.acquire()

    request = asyncio.create_task(
        manager._add_request(
            guild=guild,
            user=user,
            text_channel=text_channel,
            voice_channel=voice_channel,
            query="Paranoid by Rich Amiri",
        )
    )
    await asyncio.sleep(0.05)

    adapter.join_voice_channel.assert_not_awaited()

    lock.release()
    await request
    adapter.join_voice_channel.assert_awaited_once()


@pytest.mark.asyncio
async def test_adding_music_preserves_an_existing_paused_queue():
    resolver = MagicMock()
    resolver.resolve.return_value = [_track("next")]
    vc = MagicMock()
    vc.is_paused.return_value = True
    adapter = SimpleNamespace(
        _voice_clients={7: vc},
        join_voice_channel=AsyncMock(return_value=True),
    )
    manager = DiscordMusicManager(adapter, resolver=resolver)
    manager._update_panel = AsyncMock()
    current = _track("current")
    manager.sessions[7] = MusicSession(guild_id=7, current=current)

    await manager._add_request(
        guild=SimpleNamespace(id=7),
        user=SimpleNamespace(id=42, display_name="nahv"),
        text_channel=SimpleNamespace(id=99),
        voice_channel=SimpleNamespace(id=12),
        query="another song",
    )

    vc.resume.assert_not_called()
    assert manager.sessions[7].current is current
    assert [track.title for track in manager.sessions[7].queue] == ["next"]


@pytest.mark.asyncio
async def test_adding_music_recovers_if_a_completion_callback_was_lost():
    resolver = MagicMock()
    resolver.resolve.return_value = [_track("next")]
    resolver.resolve_stream.return_value = "https://cdn.example/next"
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = False
    vc.is_paused.return_value = False
    adapter = SimpleNamespace(
        _voice_clients={7: vc},
        _voice_receivers={},
        _voice_mixers={},
        join_voice_channel=AsyncMock(return_value=True),
        _cancel_voice_timeout=MagicMock(),
    )
    manager = DiscordMusicManager(
        adapter,
        resolver=resolver,
        audio_source_factory=lambda _url: object(),
        view_factory=lambda *_args: "controls",
    )
    stale = _track("stale")
    manager.sessions[7] = MusicSession(guild_id=7, current=stale)

    await manager._add_request(
        guild=SimpleNamespace(id=7),
        user=SimpleNamespace(id=42, display_name="nahv"),
        text_channel=SimpleNamespace(id=99, send=AsyncMock()),
        voice_channel=SimpleNamespace(id=12),
        query="next song",
    )

    session = manager.sessions[7]
    assert list(session.history) == [stale]
    assert session.current.title == "next"
    vc.play.assert_called_once()


@pytest.mark.asyncio
async def test_lost_skip_callback_still_bypasses_repeat_when_new_music_is_added():
    current = _track("current", 42)
    resolver = MagicMock()
    resolver.resolve.return_value = [_track("next", 42)]
    resolver.resolve_stream.return_value = "https://cdn.example/next"
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = True
    vc.is_paused.return_value = False
    adapter = SimpleNamespace(
        _voice_clients={7: vc},
        _voice_receivers={},
        _voice_mixers={},
        join_voice_channel=AsyncMock(return_value=True),
        _cancel_voice_timeout=MagicMock(),
    )
    manager = DiscordMusicManager(
        adapter,
        resolver=resolver,
        audio_source_factory=lambda _url: object(),
        view_factory=lambda *_args: "controls",
    )
    session = MusicSession(
        guild_id=7,
        current=current,
        repeat_current=True,
        playback_generation=1,
    )
    session.text_channel = SimpleNamespace(send=AsyncMock())
    manager.sessions[7] = session
    interaction = SimpleNamespace(
        user=SimpleNamespace(
            id=42,
            guild_permissions=SimpleNamespace(administrator=False),
        ),
        response=SimpleNamespace(send_message=AsyncMock()),
    )

    await manager.control(interaction, 7, "skip")
    vc.is_playing.return_value = False
    await manager._add_request(
        guild=SimpleNamespace(id=7),
        user=SimpleNamespace(id=42, display_name="nahv"),
        text_channel=session.text_channel,
        voice_channel=SimpleNamespace(id=12),
        query="next song",
    )

    assert list(session.history) == [current]
    assert session.current is not None
    assert session.current.title == "next"


@pytest.mark.asyncio
async def test_unplayable_track_is_skipped_and_next_track_starts():
    bad = _track("bad")
    good = _track("good")
    resolver = MagicMock()
    resolver.resolve_stream.side_effect = [
        MusicResolutionError("unplayable"),
        "https://cdn.example/good",
    ]
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = False
    adapter = SimpleNamespace(
        _voice_clients={7: vc},
        _voice_receivers={},
        _voice_mixers={},
        _cancel_voice_timeout=MagicMock(),
    )
    manager = DiscordMusicManager(
        adapter, resolver=resolver, view_factory=lambda *_args: "controls"
    )
    session = MusicSession(guild_id=7)
    session.enqueue([bad, good])
    session.panel_message = SimpleNamespace(edit=AsyncMock())
    session.text_channel = SimpleNamespace()

    await manager._start_next(session)

    assert session.current is good
    vc.play.assert_called_once()


@pytest.mark.asyncio
async def test_panel_edit_failure_cannot_discard_a_track_that_started_playing():
    first = _track("first")
    second = _track("second")
    resolver = MagicMock()
    resolver.resolve_stream.return_value = "https://cdn.example/audio"
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = False
    vc.is_paused.return_value = False
    adapter = SimpleNamespace(
        _voice_clients={7: vc},
        _voice_receivers={},
        _voice_mixers={},
        _cancel_voice_timeout=MagicMock(),
    )
    manager = DiscordMusicManager(
        adapter,
        resolver=resolver,
        audio_source_factory=lambda _url: object(),
        view_factory=lambda *_args: "controls",
    )
    session = MusicSession(guild_id=7, queue=deque([first, second]))
    session.panel_message = SimpleNamespace(
        edit=AsyncMock(side_effect=RuntimeError("deleted panel"))
    )
    session.text_channel = SimpleNamespace(send=AsyncMock(side_effect=RuntimeError("forbidden")))

    await manager._start_next(session)

    assert session.current is first
    assert list(session.queue) == [second]
    vc.play.assert_called_once()


@pytest.mark.asyncio
async def test_panel_render_failure_cannot_discard_a_track_that_started_playing():
    first = _track("first")
    second = _track("second")
    resolver = MagicMock()
    resolver.resolve_stream.return_value = "https://cdn.example/audio"
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = False
    vc.is_paused.return_value = False
    adapter = SimpleNamespace(
        _voice_clients={7: vc},
        _voice_receivers={},
        _voice_mixers={},
        _cancel_voice_timeout=MagicMock(),
    )
    manager = DiscordMusicManager(
        adapter,
        resolver=resolver,
        audio_source_factory=lambda _url: object(),
        view_factory=MagicMock(side_effect=RuntimeError("broken controls")),
    )
    session = MusicSession(guild_id=7, queue=deque([first, second]))
    session.text_channel = SimpleNamespace(send=AsyncMock())

    await manager._start_next(session)

    assert session.current is first
    assert list(session.queue) == [second]
    vc.play.assert_called_once()


@pytest.mark.asyncio
async def test_track_completion_advances_to_the_next_queued_song():
    first = _track("first")
    second = _track("second")
    resolver = MagicMock()
    resolver.resolve_stream.return_value = "https://cdn.example/second"
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = False
    vc.is_paused.return_value = False
    adapter = SimpleNamespace(
        _voice_clients={7: vc},
        _voice_receivers={},
        _voice_mixers={},
        _cancel_voice_timeout=MagicMock(),
    )
    manager = DiscordMusicManager(
        adapter,
        resolver=resolver,
        audio_source_factory=lambda _url: object(),
        view_factory=lambda *_args: "controls",
    )
    session = MusicSession(guild_id=7, current=first, queue=deque([second]))
    session.playback_generation = 1
    session.panel_message = SimpleNamespace(edit=AsyncMock())
    session.text_channel = SimpleNamespace(send=AsyncMock())
    manager.sessions[7] = session

    await manager._on_track_end(7, generation=1)

    assert session.current is second
    assert list(session.history) == [first]
    assert not session.queue
    vc.play.assert_called_once()


@pytest.mark.asyncio
async def test_lost_natural_completion_callback_still_advances_existing_queue():
    first = _track("first")
    second = _track("second")
    resolver = MagicMock()
    resolver.resolve_stream.side_effect = [
        "https://cdn.example/first",
        "https://cdn.example/second",
    ]
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = False
    vc.is_paused.return_value = False

    def _play(*_args, **_kwargs):
        vc.is_playing.return_value = True

    vc.play.side_effect = _play
    adapter = SimpleNamespace(
        _voice_clients={7: vc},
        _voice_receivers={},
        _voice_mixers={},
        _cancel_voice_timeout=MagicMock(),
    )
    manager = DiscordMusicManager(
        adapter,
        resolver=resolver,
        audio_source_factory=lambda _url: object(),
        view_factory=lambda *_args: "controls",
        stop_reconcile_delay=0.01,
    )
    session = MusicSession(guild_id=7, queue=deque([first, second]))
    session.text_channel = SimpleNamespace(send=AsyncMock())
    manager.sessions[7] = session

    await manager._start_next(session)
    vc.is_playing.return_value = False
    await asyncio.sleep(0.05)

    assert session.current is second
    assert list(session.history) == [first]
    assert vc.play.call_count == 2


@pytest.mark.asyncio
async def test_playback_failure_is_reported_and_not_added_to_history():
    failed = _track("failed")
    next_track = _track("next")
    resolver = MagicMock()
    resolver.resolve_stream.return_value = "https://cdn.example/next"
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = False
    vc.is_paused.return_value = False
    adapter = SimpleNamespace(
        _voice_clients={7: vc},
        _voice_receivers={},
        _voice_mixers={},
        _cancel_voice_timeout=MagicMock(),
    )
    manager = DiscordMusicManager(
        adapter,
        resolver=resolver,
        audio_source_factory=lambda _url: object(),
        view_factory=lambda *_args: "controls",
    )
    session = MusicSession(
        guild_id=7,
        current=failed,
        queue=deque([next_track]),
        playback_generation=1,
    )
    session.text_channel = SimpleNamespace(send=AsyncMock())
    manager.sessions[7] = session

    await manager._on_track_end(
        7,
        error=RuntimeError("decoder stalled"),
        generation=1,
    )

    assert session.current is next_track
    assert failed not in session.history
    assert session.last_error == "Playback failed for **failed**."
    assert "decoder stalled" not in session.last_error


@pytest.mark.asyncio
async def test_source_terminal_error_reaches_the_completion_handler():
    source = SimpleNamespace(
        _current_error=RuntimeError("stream stalled"),
        cleanup=MagicMock(),
    )
    resolver = MagicMock()
    resolver.resolve_stream.return_value = "https://cdn.example/current"
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = False
    vc.is_paused.return_value = False
    adapter = SimpleNamespace(
        _voice_clients={7: vc},
        _voice_receivers={},
        _voice_mixers={},
        _cancel_voice_timeout=MagicMock(),
    )
    manager = DiscordMusicManager(
        adapter,
        resolver=resolver,
        audio_source_factory=lambda _url: source,
        view_factory=lambda *_args: "controls",
    )
    track = _track("current")
    session = MusicSession(guild_id=7, queue=deque([track]))
    session.text_channel = SimpleNamespace(send=AsyncMock())
    manager.sessions[7] = session

    await manager._start_next(session)
    after = vc.play.call_args.kwargs["after"]
    after(None)
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert session.current is None
    assert track not in session.history
    assert session.last_error == "Playback failed for **current**."
    assert "stream stalled" not in session.last_error


@pytest.mark.asyncio
async def test_music_playback_keeps_voice_receiver_live_for_spoken_controls():
    receiver = SimpleNamespace(pause=MagicMock())
    resolver = MagicMock()
    resolver.resolve_stream.return_value = "https://cdn.example/audio"
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = False
    vc.is_paused.return_value = False
    adapter = SimpleNamespace(
        _voice_clients={7: vc},
        _voice_receivers={7: receiver},
        _voice_mixers={},
        _cancel_voice_timeout=MagicMock(),
    )
    manager = DiscordMusicManager(
        adapter,
        resolver=resolver,
        audio_source_factory=lambda _url: object(),
        view_factory=lambda *_args: "controls",
    )
    session = MusicSession(guild_id=7, queue=deque([_track("current")]))
    session.text_channel = SimpleNamespace(send=AsyncMock())

    await manager._start_next(session)

    receiver.pause.assert_not_called()
    vc.play.assert_called_once()


@pytest.mark.asyncio
async def test_skip_button_rejects_non_requester_and_allows_current_requester():
    vc = MagicMock()
    adapter = SimpleNamespace(_voice_clients={7: vc})
    manager = DiscordMusicManager(
        adapter,
        view_factory=lambda *_args: "controls",
    )
    manager.sessions[7] = MusicSession(guild_id=7, current=_track("current", 42))

    denied = SimpleNamespace(
        user=SimpleNamespace(
            id=99, guild_permissions=SimpleNamespace(administrator=False)
        ),
        response=SimpleNamespace(send_message=AsyncMock()),
    )
    allowed = SimpleNamespace(
        user=SimpleNamespace(
            id=42, guild_permissions=SimpleNamespace(administrator=False)
        ),
        response=SimpleNamespace(send_message=AsyncMock()),
    )

    await manager.control(denied, 7, "skip")
    vc.stop.assert_not_called()
    assert denied.response.send_message.await_args.kwargs["ephemeral"] is True

    await manager.control(allowed, 7, "skip")
    vc.stop.assert_called_once_with()


@pytest.mark.asyncio
async def test_panel_control_rejects_user_outside_hermes_allowlist():
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = True
    adapter = SimpleNamespace(
        _voice_clients={7: vc},
        _is_allowed_user=MagicMock(return_value=False),
    )
    manager = DiscordMusicManager(
        adapter,
        view_factory=lambda *_args: "controls",
    )
    manager.sessions[7] = MusicSession(guild_id=7, current=_track("current", 42))
    interaction = SimpleNamespace(
        guild=SimpleNamespace(id=7),
        user=SimpleNamespace(
            id=99, guild_permissions=SimpleNamespace(administrator=True)
        ),
        response=SimpleNamespace(send_message=AsyncMock()),
    )

    await manager.control(interaction, 7, "skip")

    adapter._is_allowed_user.assert_called_once_with(
        "99",
        author=interaction.user,
        guild=interaction.guild,
        is_dm=False,
        channel_ids=None,
    )
    vc.stop.assert_not_called()
    assert "not authorized" in interaction.response.send_message.await_args.args[0]


@pytest.mark.asyncio
async def test_panel_control_passes_validated_channel_scope_to_authorization():
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = True
    channel = SimpleNamespace(id=99)
    user = SimpleNamespace(
        id=42, guild_permissions=SimpleNamespace(administrator=False)
    )
    allow_user = MagicMock(
        side_effect=lambda _uid, **kwargs: kwargs.get("channel_ids") == {"99"}
    )
    adapter = SimpleNamespace(
        _voice_clients={7: vc},
        _is_allowed_user=allow_user,
        _discord_channel_keys_from_channel=MagicMock(return_value={"99"}),
        _get_parent_channel_id=MagicMock(return_value=None),
    )
    manager = DiscordMusicManager(adapter, view_factory=lambda *_args: "controls")
    manager.sessions[7] = MusicSession(guild_id=7, current=_track("current", 42))
    interaction = SimpleNamespace(
        guild=SimpleNamespace(id=7),
        channel=channel,
        user=user,
        response=SimpleNamespace(send_message=AsyncMock()),
    )

    await manager.control(interaction, 7, "skip")

    vc.stop.assert_called_once()
    allow_user.assert_called_once_with(
        "42",
        author=user,
        guild=interaction.guild,
        is_dm=False,
        channel_ids={"99"},
    )


@pytest.mark.asyncio
async def test_skip_recovers_and_advances_when_the_audio_player_is_already_stopped():
    current = _track("current", 42)
    next_track = _track("next", 42)
    resolver = MagicMock()
    resolver.resolve_stream.return_value = "https://cdn.example/next"
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = False
    vc.is_paused.return_value = False
    adapter = SimpleNamespace(
        _voice_clients={7: vc},
        _voice_receivers={},
        _voice_mixers={},
        _cancel_voice_timeout=MagicMock(),
    )
    manager = DiscordMusicManager(
        adapter,
        resolver=resolver,
        audio_source_factory=lambda _url: object(),
        view_factory=lambda *_args: "controls",
    )
    session = MusicSession(guild_id=7, current=current, queue=deque([next_track]))
    session.text_channel = SimpleNamespace(send=AsyncMock())
    manager.sessions[7] = session
    interaction = SimpleNamespace(
        user=SimpleNamespace(
            id=42,
            guild_permissions=SimpleNamespace(administrator=False),
        ),
        response=SimpleNamespace(send_message=AsyncMock()),
    )

    await manager.control(interaction, 7, "skip")

    assert session.current is next_track
    assert list(session.history) == [current]
    vc.play.assert_called_once()


@pytest.mark.asyncio
async def test_skip_advances_when_voice_stop_loses_its_completion_callback():
    current = _track("current", 42)
    next_track = _track("next", 42)
    resolver = MagicMock()
    resolver.resolve_stream.return_value = "https://cdn.example/next"
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.side_effect = [True, False, False]
    vc.is_paused.return_value = False
    adapter = SimpleNamespace(
        _voice_clients={7: vc},
        _voice_receivers={},
        _voice_mixers={},
        _cancel_voice_timeout=MagicMock(),
    )
    manager = DiscordMusicManager(
        adapter,
        resolver=resolver,
        audio_source_factory=lambda _url: object(),
        view_factory=lambda *_args: "controls",
        stop_reconcile_delay=0.01,
    )
    session = MusicSession(
        guild_id=7,
        current=current,
        queue=deque([next_track]),
        playback_generation=1,
    )
    session.text_channel = SimpleNamespace(send=AsyncMock())
    manager.sessions[7] = session
    interaction = SimpleNamespace(
        user=SimpleNamespace(
            id=42,
            guild_permissions=SimpleNamespace(administrator=False),
        ),
        response=SimpleNamespace(send_message=AsyncMock()),
    )

    await manager.control(interaction, 7, "skip")
    await asyncio.sleep(0.05)

    assert session.current is next_track
    assert list(session.history) == [current]
    vc.play.assert_called_once()


@pytest.mark.asyncio
async def test_skip_advances_instead_of_repeating_when_repeat_is_enabled():
    current = _track("current", 42)
    next_track = _track("next", 42)
    resolver = MagicMock()
    resolver.resolve_stream.return_value = "https://cdn.example/next"
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = True
    vc.is_paused.return_value = False
    adapter = SimpleNamespace(
        _voice_clients={7: vc},
        _voice_receivers={},
        _voice_mixers={},
        _cancel_voice_timeout=MagicMock(),
    )
    manager = DiscordMusicManager(
        adapter,
        resolver=resolver,
        audio_source_factory=lambda _url: object(),
        view_factory=lambda *_args: "controls",
    )
    session = MusicSession(
        guild_id=7,
        current=current,
        queue=deque([next_track]),
        repeat_current=True,
        playback_generation=1,
    )
    session.text_channel = SimpleNamespace(send=AsyncMock())
    manager.sessions[7] = session
    interaction = SimpleNamespace(
        user=SimpleNamespace(
            id=42,
            guild_permissions=SimpleNamespace(administrator=False),
        ),
        response=SimpleNamespace(send_message=AsyncMock()),
    )

    await manager.control(interaction, 7, "skip")
    await manager._on_track_end(7, generation=1)

    assert session.current is next_track
    assert list(session.history) == [current]


@pytest.mark.asyncio
async def test_play_pause_button_toggles_voice_client_state():
    vc = MagicMock()
    vc.is_paused.side_effect = [False, True]
    adapter = SimpleNamespace(_voice_clients={7: vc})
    manager = DiscordMusicManager(adapter, view_factory=lambda *_args: "controls")
    manager.sessions[7] = MusicSession(guild_id=7, current=_track("current", 42))
    interaction = SimpleNamespace(
        user=SimpleNamespace(
            id=42, guild_permissions=SimpleNamespace(administrator=False)
        ),
        response=SimpleNamespace(send_message=AsyncMock()),
    )

    await manager.control(interaction, 7, "play_pause")
    vc.pause.assert_called_once_with()

    await manager.control(interaction, 7, "play_pause")
    vc.resume.assert_called_once_with()


@pytest.mark.asyncio
async def test_play_pause_button_does_not_claim_it_paused_a_stopped_player():
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_paused.return_value = False
    vc.is_playing.return_value = False
    adapter = SimpleNamespace(_voice_clients={7: vc})
    manager = DiscordMusicManager(adapter, view_factory=lambda *_args: "controls")
    manager.sessions[7] = MusicSession(guild_id=7, current=_track("stale", 42))
    interaction = SimpleNamespace(
        user=SimpleNamespace(
            id=42,
            guild_permissions=SimpleNamespace(administrator=False),
        ),
        response=SimpleNamespace(send_message=AsyncMock()),
    )

    await manager.control(interaction, 7, "play_pause")

    vc.pause.assert_not_called()
    assert "nothing is playing" in interaction.response.send_message.await_args.args[0].lower()


@pytest.mark.asyncio
async def test_natural_pause_and_play_commands_are_idempotent_not_a_toggle():
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = True
    vc.is_paused.return_value = False
    channel = SimpleNamespace(send=AsyncMock())
    member = SimpleNamespace(
        id=42,
        guild_permissions=SimpleNamespace(administrator=False),
    )
    message = SimpleNamespace(
        guild=SimpleNamespace(id=7),
        channel=channel,
        author=member,
    )
    manager = DiscordMusicManager(
        SimpleNamespace(_voice_clients={7: vc}),
        view_factory=lambda *_args: "controls",
    )
    manager.sessions[7] = MusicSession(guild_id=7, current=_track("current", 42))

    await manager.control_message(message, "pause")
    vc.pause.assert_called_once_with()

    vc.is_playing.return_value = False
    vc.is_paused.return_value = True
    await manager.control_message(message, "resume")
    vc.resume.assert_called_once_with()

    vc.is_playing.return_value = True
    vc.is_paused.return_value = False
    await manager.control_message(message, "resume")
    vc.pause.assert_called_once_with()
    assert "already playing" in channel.send.await_args.args[0].lower()


@pytest.mark.asyncio
async def test_repeat_button_toggles_current_track_repeat_mode():
    adapter = SimpleNamespace(_voice_clients={7: MagicMock()})
    manager = DiscordMusicManager(adapter, view_factory=lambda *_args: "controls")
    session = MusicSession(guild_id=7, current=_track("current", 42))
    session.panel_message = SimpleNamespace(edit=AsyncMock())
    session.text_channel = SimpleNamespace()
    manager.sessions[7] = session
    interaction = SimpleNamespace(
        user=SimpleNamespace(
            id=42, guild_permissions=SimpleNamespace(administrator=False)
        ),
        response=SimpleNamespace(send_message=AsyncMock()),
    )

    await manager.control(interaction, 7, "repeat")
    assert session.repeat_current is True

    await manager.control(interaction, 7, "repeat")
    assert session.repeat_current is False


@pytest.mark.asyncio
async def test_previous_button_replays_history_then_returns_to_interrupted_song():
    previous = _track("previous", 42)
    current = _track("current", 42)
    resolver = MagicMock()
    resolver.resolve_stream.return_value = "https://cdn.example/audio"
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = False
    vc.is_paused.return_value = False
    adapter = SimpleNamespace(
        _voice_clients={7: vc},
        _voice_receivers={},
        _voice_mixers={},
        _cancel_voice_timeout=MagicMock(),
    )
    manager = DiscordMusicManager(
        adapter,
        resolver=resolver,
        audio_source_factory=lambda _url: object(),
        view_factory=lambda *_args: "controls",
    )
    session = MusicSession(guild_id=7, current=current)
    session.history.append(previous)
    session.panel_message = SimpleNamespace(edit=AsyncMock())
    session.text_channel = SimpleNamespace()
    manager.sessions[7] = session
    interaction = SimpleNamespace(
        user=SimpleNamespace(
            id=42, guild_permissions=SimpleNamespace(administrator=False)
        ),
        response=SimpleNamespace(send_message=AsyncMock()),
    )

    await manager.control(interaction, 7, "previous")

    assert session.current is previous
    assert list(session.queue)[0] is current
    vc.stop.assert_called_once_with()
    vc.play.assert_called_once()


@pytest.mark.asyncio
async def test_previous_replays_finished_history_for_its_requester():
    previous = _track("previous", 42)
    resolver = MagicMock()
    resolver.resolve_stream.return_value = "https://cdn.example/audio"
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = False
    vc.is_paused.return_value = False
    adapter = SimpleNamespace(
        _voice_clients={7: vc},
        _voice_receivers={},
        _voice_mixers={},
        _cancel_voice_timeout=MagicMock(),
    )
    manager = DiscordMusicManager(
        adapter,
        resolver=resolver,
        audio_source_factory=lambda _url: object(),
        view_factory=lambda *_args: "controls",
    )
    session = MusicSession(guild_id=7)
    session.history.append(previous)
    session.text_channel = SimpleNamespace(send=AsyncMock())
    manager.sessions[7] = session
    interaction = SimpleNamespace(
        user=SimpleNamespace(
            id=42, guild_permissions=SimpleNamespace(administrator=False)
        ),
        response=SimpleNamespace(send_message=AsyncMock()),
    )

    await manager.control(interaction, 7, "previous")

    assert session.current is previous
    vc.play.assert_called_once()


@pytest.mark.asyncio
async def test_previous_preserves_queue_state_when_voice_is_disconnected():
    previous = _track("previous", 42)
    current = _track("current", 42)
    vc = MagicMock()
    vc.is_connected.return_value = False
    manager = DiscordMusicManager(
        SimpleNamespace(_voice_clients={7: vc}),
        view_factory=lambda *_args: "controls",
    )
    session = MusicSession(guild_id=7, current=current)
    session.history.append(previous)
    manager.sessions[7] = session
    interaction = SimpleNamespace(
        user=SimpleNamespace(
            id=42,
            guild_permissions=SimpleNamespace(administrator=False),
        ),
        response=SimpleNamespace(send_message=AsyncMock()),
    )

    await manager.control(interaction, 7, "previous")

    assert session.current is current
    assert list(session.history) == [previous]
    assert not session.queue
    vc.stop.assert_not_called()


@pytest.mark.asyncio
async def test_previous_defers_before_resolving_stream():
    previous = _track("previous", 42)
    current = _track("current", 42)
    resolver = MagicMock()
    resolver.resolve_stream.return_value = "https://cdn.example/previous"
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = False
    adapter = SimpleNamespace(
        _voice_clients={7: vc},
        _voice_receivers={},
        _voice_mixers={},
        _cancel_voice_timeout=MagicMock(),
    )
    manager = DiscordMusicManager(
        adapter, resolver=resolver, view_factory=lambda *_args: "controls"
    )
    session = MusicSession(guild_id=7, current=current)
    session.history.append(previous)
    session.panel_message = SimpleNamespace(edit=AsyncMock())
    session.text_channel = SimpleNamespace()
    manager.sessions[7] = session
    interaction = SimpleNamespace(
        user=SimpleNamespace(
            id=42, guild_permissions=SimpleNamespace(administrator=False)
        ),
        response=SimpleNamespace(defer=AsyncMock(), is_done=lambda: True),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    await manager.control(interaction, 7, "previous")

    interaction.response.defer.assert_awaited_once_with(ephemeral=True)
    interaction.followup.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_voice_client_does_not_claim_skip_succeeded():
    adapter = SimpleNamespace(_voice_clients={})
    manager = DiscordMusicManager(adapter)
    session = MusicSession(guild_id=7, current=_track("current", 42))
    manager.sessions[7] = session
    interaction = SimpleNamespace(
        user=SimpleNamespace(
            id=42, guild_permissions=SimpleNamespace(administrator=False)
        ),
        response=SimpleNamespace(send_message=AsyncMock()),
    )

    await manager.control(interaction, 7, "skip")

    assert session.current is not None
    assert "lost" in interaction.response.send_message.await_args.args[0]


@pytest.mark.asyncio
async def test_music_panel_has_previous_play_pause_repeat_and_skip_buttons():
    manager = SimpleNamespace(control=AsyncMock())
    view = MusicControlView(manager, 7)

    labels = {item.label for item in view.children}
    assert labels == {"Previous", "Play/Pause", "Repeat", "Skip"}
    assert view.timeout is None


@pytest.mark.asyncio
async def test_finishing_an_empty_queue_restores_voice_listening_and_inactivity_timer():
    receiver = SimpleNamespace(resume=MagicMock())
    adapter = SimpleNamespace(
        _voice_clients={7: MagicMock()},
        _voice_receivers={7: receiver},
        _voice_mixers={},
        _voice_fx_cfg={"enabled": True},
        _install_voice_mixer=AsyncMock(),
        _reset_voice_timeout=MagicMock(),
    )
    manager = DiscordMusicManager(adapter, view_factory=lambda *_args: "controls")
    session = MusicSession(guild_id=7, current=_track("last", 42))
    session.panel_message = SimpleNamespace(edit=AsyncMock())
    session.text_channel = SimpleNamespace()
    manager.sessions[7] = session

    await manager._on_track_end(7)

    assert session.current is None
    receiver.resume.assert_called_once_with()
    adapter._install_voice_mixer.assert_awaited_once_with(7, adapter._voice_clients[7])
    adapter._reset_voice_timeout.assert_called_once_with(7)


@pytest.mark.asyncio
async def test_voice_disconnect_discards_stale_music_state_and_updates_panel():
    adapter = SimpleNamespace(_voice_clients={})
    manager = DiscordMusicManager(adapter, view_factory=lambda *_args: "controls")
    session = MusicSession(guild_id=7, current=_track("current", 42))
    session.enqueue([_track("next", 99)])
    session.panel_message = SimpleNamespace(edit=AsyncMock())
    session.text_channel = SimpleNamespace()
    manager.sessions[7] = session

    await manager.on_voice_disconnected(7)

    assert 7 not in manager.sessions
    session.panel_message.edit.assert_awaited_once()
    assert (
        "Nothing is playing" in session.panel_message.edit.await_args.kwargs["content"]
    )


@pytest.mark.asyncio
async def test_voice_disconnect_serializes_on_persistent_guild_lock():
    import asyncio

    adapter = SimpleNamespace(_voice_clients={})
    manager = DiscordMusicManager(adapter, view_factory=lambda *_args: "controls")
    session = MusicSession(guild_id=7, current=_track("current", 42))
    session.panel_message = SimpleNamespace(edit=AsyncMock())
    session.text_channel = SimpleNamespace()
    manager.sessions[7] = session
    lock = manager._locks.setdefault(7, asyncio.Lock())

    await lock.acquire()
    disconnect = asyncio.create_task(manager.on_voice_disconnected(7))
    await asyncio.sleep(0)

    assert 7 in manager.sessions

    lock.release()
    await disconnect

    assert 7 not in manager.sessions
    assert manager._locks[7] is lock


@pytest.mark.asyncio
async def test_show_queue_defers_before_waiting_for_guild_lock():
    import asyncio

    adapter = SimpleNamespace(_voice_clients={})
    manager = DiscordMusicManager(adapter, view_factory=lambda *_args: "controls")
    session = MusicSession(guild_id=7, current=_track("current", 42))
    session.panel_message = SimpleNamespace(edit=AsyncMock())
    manager.sessions[7] = session
    interaction = SimpleNamespace(
        channel=SimpleNamespace(),
        response=SimpleNamespace(defer=AsyncMock()),
        followup=SimpleNamespace(send=AsyncMock()),
    )
    lock = manager._locks.setdefault(7, asyncio.Lock())

    await lock.acquire()
    show = asyncio.create_task(manager.show_queue(interaction, 7))
    await asyncio.sleep(0)

    interaction.response.defer.assert_awaited_once_with(ephemeral=True)
    session.panel_message.edit.assert_not_awaited()

    lock.release()
    await show

    session.panel_message.edit.assert_awaited_once()
    interaction.followup.send.assert_awaited_once_with(
        "The public music queue is up to date.", ephemeral=True
    )


@pytest.mark.asyncio
async def test_adapter_voice_leave_notifies_music_manager():
    adapter = object.__new__(DiscordAdapter)
    adapter._voice_locks = {}
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._client = None
    adapter._voice_mixers = {}
    adapter._voice_clients = {7: SimpleNamespace(is_connected=lambda: False)}
    adapter._voice_timeout_tasks = {}
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    lock_was_held = None

    async def _on_voice_disconnected(_guild_id):
        nonlocal lock_was_held
        lock_was_held = adapter._voice_locks[7].locked()

    adapter._music_manager = SimpleNamespace(
        on_voice_disconnected=AsyncMock(side_effect=_on_voice_disconnected)
    )

    await adapter.leave_voice_channel(7)

    adapter._music_manager.on_voice_disconnected.assert_awaited_once_with(7)
    assert lock_was_held is False


@pytest.mark.asyncio
async def test_join_is_rejected_while_voice_leave_cleanup_is_in_progress():
    adapter = object.__new__(DiscordAdapter)
    adapter._client = object()
    adapter._voice_locks = {}
    adapter._voice_leaving = {7}
    channel = SimpleNamespace(guild=SimpleNamespace(id=7))

    joined = await adapter.join_voice_channel(channel)

    assert joined is False


@pytest.mark.asyncio
async def test_add_racing_with_leave_cannot_orphan_a_new_voice_player():
    adapter = object.__new__(DiscordAdapter)
    adapter._client = SimpleNamespace(get_guild=MagicMock(return_value=None))
    adapter._voice_locks = {}
    adapter._voice_leaving = set()
    adapter._voice_receivers = {}
    adapter._voice_listen_tasks = {}
    adapter._voice_mixers = {}
    adapter._voice_timeout_tasks = {}
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    adapter._is_allowed_user = MagicMock(return_value=True)

    disconnect_started = asyncio.Event()
    allow_disconnect = asyncio.Event()
    old_vc = MagicMock()
    old_vc.is_connected.return_value = True
    old_vc.is_playing.return_value = False

    async def _disconnect():
        disconnect_started.set()
        await allow_disconnect.wait()

    old_vc.disconnect = _disconnect
    adapter._voice_clients = {7: old_vc}

    resolver = MagicMock()
    resolver.resolve.return_value = [_track("new", 42)]
    manager = DiscordMusicManager(
        adapter,
        resolver=resolver,
        audio_source_factory=lambda _url: object(),
        view_factory=lambda *_args: "controls",
    )
    adapter._music_manager = manager
    new_channel = SimpleNamespace(
        id=12,
        guild=SimpleNamespace(id=7),
        connect=AsyncMock(),
    )

    leave_task = asyncio.create_task(adapter.leave_voice_channel(7))
    await asyncio.wait_for(disconnect_started.wait(), timeout=1)
    add_task = asyncio.create_task(
        manager._add_request(
            guild=SimpleNamespace(id=7),
            user=SimpleNamespace(id=42, display_name="user-42"),
            text_channel=SimpleNamespace(id=99),
            voice_channel=new_channel,
            query="new",
        )
    )
    await asyncio.sleep(0)
    allow_disconnect.set()

    with pytest.raises(MusicResolutionError, match="could not join"):
        await asyncio.wait_for(add_task, timeout=1)
    await asyncio.wait_for(leave_task, timeout=1)

    new_channel.connect.assert_not_awaited()
    assert 7 not in adapter._voice_clients
    assert 7 not in manager.sessions
    assert 7 not in adapter._voice_leaving


@pytest.mark.asyncio
async def test_joining_an_existing_voice_connection_refreshes_its_text_binding():
    adapter = object.__new__(DiscordAdapter)
    adapter._client = object()
    adapter._voice_locks = {}
    adapter._voice_text_channels = {}
    adapter._voice_sources = {}
    adapter._reset_voice_timeout = MagicMock()
    existing = SimpleNamespace(
        channel=SimpleNamespace(id=12),
        is_connected=lambda: True,
    )
    adapter._voice_clients = {7: existing}
    channel = SimpleNamespace(id=12, guild=SimpleNamespace(id=7))

    joined = await adapter.join_voice_channel(
        channel,
        text_channel_id=99,
        source={"chat_id": "99"},
    )

    assert joined is True
    assert adapter._voice_text_channels[7] == 99
    assert adapter._voice_sources[7] == {"chat_id": "99"}


@pytest.mark.asyncio
async def test_clear_queue_is_administrator_only_and_stops_current_song():
    vc = MagicMock()
    receiver = SimpleNamespace(resume=MagicMock())
    adapter = SimpleNamespace(
        _voice_clients={7: vc},
        _voice_receivers={7: receiver},
        _reset_voice_timeout=MagicMock(),
    )
    manager = DiscordMusicManager(adapter, view_factory=lambda *_args: "controls")
    session = MusicSession(guild_id=7, current=_track("current", 42))
    session.enqueue([_track("next", 99)])
    manager.sessions[7] = session

    member = SimpleNamespace(
        user=SimpleNamespace(
            id=42, guild_permissions=SimpleNamespace(administrator=False)
        ),
        response=SimpleNamespace(send_message=AsyncMock()),
    )
    admin = SimpleNamespace(
        user=SimpleNamespace(
            id=1, guild_permissions=SimpleNamespace(administrator=True)
        ),
        response=SimpleNamespace(send_message=AsyncMock()),
    )

    await manager.admin_action(member, 7, "clear")
    assert list(session.queue)
    vc.stop.assert_not_called()

    await manager.admin_action(admin, 7, "clear")
    assert not session.queue
    assert session.current is None
    vc.stop.assert_called_once_with()
    receiver.resume.assert_called_once_with()
    adapter._reset_voice_timeout.assert_called_once_with(7)


@pytest.mark.asyncio
async def test_clear_suppresses_delayed_stop_callback_from_old_track():
    vc = MagicMock()
    adapter = SimpleNamespace(
        _voice_clients={7: vc},
        _voice_receivers={},
        _reset_voice_timeout=MagicMock(),
    )
    manager = DiscordMusicManager(adapter)
    session = MusicSession(guild_id=7, current=_track("old", 42))
    old_generation = session.playback_generation
    manager.sessions[7] = session
    interaction = SimpleNamespace(
        user=SimpleNamespace(
            id=1, guild_permissions=SimpleNamespace(administrator=True)
        ),
        response=SimpleNamespace(send_message=AsyncMock()),
    )

    await manager.admin_action(interaction, 7, "clear")
    replacement = _track("replacement", 99)
    session.current = replacement
    await manager._on_track_end(7, generation=old_generation)

    assert session.current is replacement


@pytest.mark.asyncio
async def test_stale_callback_cannot_consume_replacement_track_if_callbacks_reorder():
    adapter = SimpleNamespace(
        _voice_clients={},
        _voice_receivers={},
        _voice_mixers={},
        _reset_voice_timeout=MagicMock(),
    )
    manager = DiscordMusicManager(adapter)
    replacement = _track("replacement", 99)
    session = MusicSession(guild_id=7, current=replacement)
    session.playback_generation = 2
    session.panel_message = SimpleNamespace(edit=AsyncMock())
    session.text_channel = SimpleNamespace()
    manager.sessions[7] = session

    await manager._on_track_end(7, generation=1)

    assert session.current is replacement
    assert not session.history

    await manager._on_track_end(7, generation=2)

    assert session.current is None
    assert list(session.history) == [replacement]


@pytest.mark.asyncio
async def test_admin_action_defers_before_waiting_for_guild_lock():
    vc = MagicMock()
    adapter = SimpleNamespace(_voice_clients={7: vc})
    manager = DiscordMusicManager(adapter)
    manager.sessions[7] = MusicSession(guild_id=7, current=_track("current", 42))
    interaction = SimpleNamespace(
        user=SimpleNamespace(
            id=1, guild_permissions=SimpleNamespace(administrator=True)
        ),
        response=SimpleNamespace(defer=AsyncMock(), is_done=lambda: True),
        followup=SimpleNamespace(send=AsyncMock()),
    )

    await manager.admin_action(interaction, 7, "forceskip")

    interaction.response.defer.assert_awaited_once_with(ephemeral=True)
    interaction.followup.send.assert_awaited_once()


@pytest.mark.asyncio
async def test_force_skip_bypasses_repeat_and_advances_the_queue():
    current = _track("current", 42)
    next_track = _track("next", 42)
    resolver = MagicMock()
    resolver.resolve_stream.return_value = "https://cdn.example/next"
    vc = MagicMock()
    vc.is_connected.return_value = True
    vc.is_playing.return_value = True
    vc.is_paused.return_value = False
    adapter = SimpleNamespace(
        _voice_clients={7: vc},
        _voice_receivers={},
        _voice_mixers={},
        _cancel_voice_timeout=MagicMock(),
    )
    manager = DiscordMusicManager(
        adapter,
        resolver=resolver,
        audio_source_factory=lambda _url: object(),
        view_factory=lambda *_args: "controls",
    )
    session = MusicSession(
        guild_id=7,
        current=current,
        queue=deque([next_track]),
        repeat_current=True,
        playback_generation=1,
    )
    session.text_channel = SimpleNamespace(send=AsyncMock())
    manager.sessions[7] = session
    interaction = SimpleNamespace(
        user=SimpleNamespace(
            id=1,
            guild_permissions=SimpleNamespace(administrator=True),
        ),
        response=SimpleNamespace(send_message=AsyncMock()),
    )

    await manager.admin_action(interaction, 7, "forceskip")
    await manager._on_track_end(7, generation=1)

    assert session.current is next_track
    assert list(session.history) == [current]


class _FakeTree:
    def __init__(self):
        self.commands = {}

    def command(self, *, name, description):
        def decorator(callback):
            self.commands[name] = callback
            return callback

        return decorator

    def add_command(self, command):
        self.commands[command.name] = command

    def get_commands(self):
        return [SimpleNamespace(name=name) for name in self.commands]


def test_discord_registers_music_queue_and_admin_slash_commands():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(tree=_FakeTree())

    adapter._register_slash_commands()

    assert {"play", "musicqueue", "forceskip", "clearqueue"}.issubset(
        adapter._client.tree.commands
    )


@pytest.mark.asyncio
async def test_natural_text_play_command_routes_directly_to_music_manager():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    manager = SimpleNamespace(add_message=AsyncMock())
    adapter._music_manager = manager
    message = SimpleNamespace(content="Hey Jarvis play Paranoid by Rich Amiri")

    handled = await adapter._maybe_handle_natural_music_message(message)

    assert handled is True
    manager.add_message.assert_awaited_once_with(
        message, "Paranoid by Rich Amiri"
    )


@pytest.mark.asyncio
async def test_natural_text_pause_routes_directly_to_music_controls():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    manager = SimpleNamespace(
        add_message=AsyncMock(),
        control_message=AsyncMock(),
    )
    adapter._music_manager = manager
    message = SimpleNamespace(content="Hey Jarvis pause the song")

    handled = await adapter._maybe_handle_natural_music_message(message)

    assert handled is True
    manager.control_message.assert_awaited_once_with(message, "pause")
    manager.add_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_natural_text_music_error_disables_discord_mentions(monkeypatch):
    import discord

    class CompatibleAllowedMentions:
        def __init__(self, *, everyone=True, roles=True, users=True, replied_user=True):
            self.everyone = everyone
            self.roles = roles
            self.users = users
            self.replied_user = replied_user

    monkeypatch.setattr(discord, "AllowedMentions", CompatibleAllowedMentions)
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._music_manager = SimpleNamespace(
        add_message=AsyncMock(side_effect=RuntimeError("@everyone"))
    )
    channel = SimpleNamespace(send=AsyncMock())
    message = SimpleNamespace(
        content="Hey Jarvis play Paranoid by Rich Amiri",
        channel=channel,
    )

    handled = await adapter._maybe_handle_natural_music_message(message)

    assert handled is True
    mentions = channel.send.await_args.kwargs["allowed_mentions"]
    assert "@everyone" not in channel.send.await_args.args[0]
    assert mentions.everyone is False
    assert mentions.roles is False
    assert mentions.users is False


@pytest.mark.asyncio
async def test_natural_voice_play_command_routes_to_linked_music_queue():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    guild = SimpleNamespace(id=7)
    member = SimpleNamespace(
        id=42,
        display_name="nahv",
        voice=SimpleNamespace(channel=SimpleNamespace(id=12)),
    )
    guild.get_member = MagicMock(return_value=member)
    channel = SimpleNamespace(id=99, send=AsyncMock())
    adapter._client = SimpleNamespace(
        get_guild=MagicMock(return_value=guild),
        get_channel=MagicMock(return_value=channel),
    )
    adapter._voice_text_channels = {7: 99}
    manager = SimpleNamespace(add_message=AsyncMock())
    adapter._music_manager = manager

    handled = await adapter._maybe_handle_natural_music_voice(
        7, 42, "Hey Jarvis play Paranoid by Rich Amiri"
    )

    assert handled is True
    request = manager.add_message.await_args.args[0]
    assert request.guild is guild
    assert request.author is member
    assert request.channel is channel
    assert manager.add_message.await_args.args[1] == "Paranoid by Rich Amiri"


@pytest.mark.asyncio
async def test_natural_voice_resume_routes_directly_to_music_controls():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    guild = SimpleNamespace(id=7)
    member = SimpleNamespace(
        id=42,
        display_name="nahv",
        voice=SimpleNamespace(channel=SimpleNamespace(id=12)),
    )
    guild.get_member = MagicMock(return_value=member)
    channel = SimpleNamespace(id=99, send=AsyncMock())
    adapter._client = SimpleNamespace(
        get_guild=MagicMock(return_value=guild),
        get_channel=MagicMock(return_value=channel),
    )
    adapter._voice_text_channels = {7: 99}
    manager = SimpleNamespace(
        add_message=AsyncMock(),
        control_message=AsyncMock(),
    )
    adapter._music_manager = manager

    handled = await adapter._maybe_handle_natural_music_voice(
        7, 42, "Hey Jarvis play"
    )

    assert handled is True
    request = manager.control_message.await_args.args[0]
    assert request.guild is guild
    assert request.author is member
    assert request.channel is channel
    assert manager.control_message.await_args.args[1] == "resume"
    manager.add_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_natural_voice_music_respects_the_linked_channels_ignore_gate():
    adapter = DiscordAdapter(
        PlatformConfig(
            enabled=True,
            token="***",
            extra={"ignored_channels": ["99"]},
        )
    )
    guild = SimpleNamespace(id=7)
    member = SimpleNamespace(id=42)
    guild.get_member = MagicMock(return_value=member)
    channel = SimpleNamespace(id=99, name="music", send=AsyncMock())
    adapter._client = SimpleNamespace(
        get_guild=MagicMock(return_value=guild),
        get_channel=MagicMock(return_value=channel),
    )
    adapter._voice_text_channels = {7: 99}
    manager = SimpleNamespace(
        add_message=AsyncMock(),
        control_message=AsyncMock(),
    )
    adapter._music_manager = manager

    handled = await adapter._maybe_handle_natural_music_voice(
        7, 42, "Hey Jarvis skip this song"
    )

    assert handled is False
    manager.control_message.assert_not_awaited()
    manager.add_message.assert_not_awaited()


@pytest.mark.asyncio
async def test_natural_voice_play_falls_back_when_discord_context_is_missing():
    adapter = DiscordAdapter(PlatformConfig(enabled=True, token="***"))
    adapter._client = SimpleNamespace(
        get_guild=MagicMock(return_value=None),
        get_channel=MagicMock(return_value=None),
    )
    adapter._voice_text_channels = {7: 99}

    handled = await adapter._maybe_handle_natural_music_voice(
        7, 42, "Hey Jarvis play Paranoid by Rich Amiri"
    )

    assert handled is False
