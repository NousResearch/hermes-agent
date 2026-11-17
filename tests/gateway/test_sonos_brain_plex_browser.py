import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call

import pytest


SONOS_BRAIN_PATH = Path.home() / ".hermes" / "sonos-brain" / "sonos_brain.py"
spec = importlib.util.spec_from_file_location("sonos_brain_plex_browser", SONOS_BRAIN_PATH)
assert spec is not None and spec.loader is not None
sonos_brain = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sonos_brain
spec.loader.exec_module(sonos_brain)


TRACKS = [
    {"ratingKey": "11", "title": "Quiet Light", "grandparentTitle": "North Star", "parentTitle": "Night <Sky>", "duration": 183000},
    {"ratingKey": "12", "title": "Other Song", "grandparentTitle": "Another", "parentTitle": "Day", "duration": 95000},
    {"ratingKey": "13", "title": "A Third Song", "grandparentTitle": "Elsewhere", "parentTitle": "Quiet Light", "duration": 120000},
]


def test_search_filters_title_artist_and_album_case_insensitively_and_bounds_results():
    found = sonos_brain.search_catalog_metadata(TRACKS, "quiet light", limit=2)

    assert [item["ratingKey"] for item in found] == ["11", "13"]


def test_plex_search_emits_only_safe_fields_and_does_not_echo_query(monkeypatch, capsys):
    client = SimpleNamespace(search_tracks=lambda query, limit: TRACKS[:limit])
    monkeypatch.setattr(sonos_brain, "load_plex_client", lambda: client)
    query = "<script>alert(1)</script>"

    result = sonos_brain._cmd_plex_search(Namespace(query=query, limit=1))
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    assert payload == {
        "success": True,
        "tracks": [{
            "id": "11",
            "title": "Quiet Light",
            "artist": "North Star",
            "album": "Night <Sky>",
            "duration_ms": 183000,
        }],
    }
    assert query not in json.dumps(payload)
    assert set(payload["tracks"][0]) == {"id", "title", "artist", "album", "duration_ms"}


def test_plex_play_plays_exact_selected_track_via_safe_stream_path(monkeypatch, capsys):
    chosen = TRACKS[1]
    client = MagicMock()
    client._search_metadata_window.return_value = TRACKS
    client.playable_track.return_value = {
        "title": "Other Song", "artist": "Quiet Lights", "album": "Day",
        "duration": 95000, "uri": "http://plex.invalid/stream/selected", "ratingKey": "12",
    }
    sonos = MagicMock()
    clear_manual_stop = MagicMock()
    lock = MagicMock()
    lock.__enter__.return_value = None
    acquire_lock = MagicMock(return_value=lock)
    monkeypatch.setattr(sonos_brain, "load_plex_client", lambda: client)
    monkeypatch.setattr(sonos_brain, "SonosClient", lambda: sonos)
    monkeypatch.setattr(sonos_brain, "playback_blocked", lambda: False)
    monkeypatch.setattr(sonos_brain, "set_uri_with_retry", MagicMock())
    monkeypatch.setattr(sonos_brain, "play_with_retry", MagicMock())
    monkeypatch.setattr(sonos_brain, "_exclusive_operation_lock", acquire_lock)
    monkeypatch.setattr(sonos_brain, "_clear_manual_stop_for_start", clear_manual_stop)

    result = sonos_brain._cmd_plex_play(Namespace(rating_key="12", no_wait=True))
    payload = json.loads(capsys.readouterr().out)

    assert result == 0
    client.playable_track.assert_called_once_with(chosen)
    sonos_brain.set_uri_with_retry.assert_called_once()
    assert sonos_brain.set_uri_with_retry.call_args.args[0] is sonos
    assert sonos_brain.set_uri_with_retry.call_args.args[1] == "http://plex.invalid/stream/selected"
    sonos_brain.play_with_retry.assert_called_once_with(sonos)
    acquire_lock.assert_called_once_with(sonos_brain.SPEAKER_OPERATION_LOCK)
    clear_manual_stop.assert_called_once()
    assert payload == {
        "success": True,
        "track": {"title": "Other Song", "artist": "Quiet Lights", "album": "Day"},
        "started": True,
    }
    assert "uri" not in json.dumps(payload).lower()


def test_plex_search_rejects_overlong_query_without_fetching(monkeypatch, capsys):
    monkeypatch.setattr(sonos_brain, "load_plex_client", lambda: pytest.fail("must not load Plex"))

    result = sonos_brain._cmd_plex_search(
        Namespace(query="x" * (sonos_brain.PLEX_SEARCH_MAX_QUERY_CHARS + 1), limit=20)
    )

    assert result == 2
    assert json.loads(capsys.readouterr().out) == {
        "success": False, "error": "Invalid Plex search query"
    }


def test_plex_search_sanitizes_backend_failure(monkeypatch, capsys):
    client = SimpleNamespace(search_tracks=lambda query, limit: (_ for _ in ()).throw(sonos_brain.SonosError("private request details")))
    monkeypatch.setattr(sonos_brain, "load_plex_client", lambda: client)

    result = sonos_brain._cmd_plex_search(Namespace(query="jazz", limit=5))
    output = capsys.readouterr().out

    assert result == 2
    assert json.loads(output) == {"success": False, "error": "Plex search failed"}
    assert "private request details" not in output


def test_plex_play_rejects_invalid_rating_key_without_fetching(monkeypatch, capsys):
    monkeypatch.setattr(sonos_brain, "load_plex_client", lambda: pytest.fail("must not load Plex"))

    result = sonos_brain._cmd_plex_play(Namespace(rating_key="1" * 33, no_wait=True))

    assert result == 2
    assert json.loads(capsys.readouterr().out) == {
        "success": False, "error": "Invalid Plex track id"
    }


def test_plex_cli_commands_are_registered_with_bounded_arguments():
    parser = sonos_brain.build_parser()

    search = parser.parse_args(["plex-search", "--query", "jazz", "--limit", "7"])
    play = parser.parse_args(["plex-play", "12", "--no-wait"])

    assert search.command == "plex-search" and search.query == "jazz" and search.limit == 7
    assert play.command == "plex-play" and play.rating_key == "12" and play.no_wait is True
