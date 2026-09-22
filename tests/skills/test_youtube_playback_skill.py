"""Tests for skills/media/youtube-playback/scripts/play_youtube.py."""

import io
import json
import sys
from pathlib import Path
from unittest import mock

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "skills" / "media" / "youtube-playback" / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import play_youtube


class TestExtractVideoId:
    def test_standard_watch_url(self):
        assert play_youtube.extract_video_id("https://www.youtube.com/watch?v=IyR25B-IGyg") == "IyR25B-IGyg"

    def test_short_url(self):
        assert play_youtube.extract_video_id("https://youtu.be/IyR25B-IGyg") == "IyR25B-IGyg"

    def test_shorts_url(self):
        assert play_youtube.extract_video_id("https://www.youtube.com/shorts/IyR25B-IGyg") == "IyR25B-IGyg"

    def test_embed_url(self):
        assert play_youtube.extract_video_id("https://www.youtube.com/embed/IyR25B-IGyg") == "IyR25B-IGyg"

    def test_raw_id(self):
        assert play_youtube.extract_video_id("IyR25B-IGyg") == "IyR25B-IGyg"

    def test_query_is_not_id(self):
        assert play_youtube.extract_video_id("Iron Man music") is None


class TestSearchYouTube:
    def test_direct_id_query(self):
        results = play_youtube.search_youtube("IyR25B-IGyg")
        assert len(results) == 1
        assert results[0]["id"] == "IyR25B-IGyg"
        assert "watch?v=IyR25B-IGyg" in results[0]["url"]

    @mock.patch("urllib.request.urlopen")
    def test_parsed_yt_initial_data(self, mock_urlopen):
        mock_html = """
        <html>
        <script>
        var ytInitialData = {
          "contents": {
            "twoColumnSearchResultsRenderer": {
              "primaryContents": {
                "sectionListRenderer": {
                  "contents": [
                    {
                      "itemSectionRenderer": {
                        "contents": [
                          {
                            "videoRenderer": {
                              "videoId": "IyR25B-IGyg",
                              "title": { "runs": [{ "text": "Back In Black Iron Man" }] },
                              "ownerText": { "runs": [{ "text": "AC/DC Official" }] },
                              "lengthText": { "simpleText": "4:15" },
                              "thumbnail": { "thumbnails": [{ "url": "https://img.youtube.com/1.jpg" }] }
                            }
                          }
                        ]
                      }
                    }
                  ]
                }
              }
            }
          }
        };</script>
        </html>
        """
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = mock_html.encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        results = play_youtube.search_youtube("Iron Man music", max_results=1)
        assert len(results) == 1
        assert results[0]["id"] == "IyR25B-IGyg"
        assert results[0]["title"] == "Back In Black Iron Man"
        assert results[0]["channel"] == "AC/DC Official"
        assert results[0]["duration"] == "4:15"

    @mock.patch("urllib.request.urlopen")
    def test_regex_fallback(self, mock_urlopen):
        mock_html = """
        <html>
        <div>Some random text /watch?v=ZNf6faavxDw more content /watch?v=IyR25B-IGyg</div>
        </html>
        """
        mock_resp = mock.MagicMock()
        mock_resp.read.return_value = mock_html.encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_urlopen.return_value = mock_resp

        results = play_youtube.search_youtube("Fallback test", max_results=2)
        assert len(results) == 2
        assert results[0]["id"] == "ZNf6faavxDw"
        assert results[1]["id"] == "IyR25B-IGyg"


class TestOpenInBrowser:
    @mock.patch("webbrowser.open")
    def test_open_with_autoplay(self, mock_open):
        mock_open.return_value = True
        res = play_youtube.open_in_browser("https://www.youtube.com/watch?v=IyR25B-IGyg", autoplay=True)
        assert res is True
        mock_open.assert_called_once_with("https://www.youtube.com/watch?v=IyR25B-IGyg&autoplay=1")


class TestMainCli:
    @mock.patch("play_youtube.search_youtube")
    def test_main_json_no_open(self, mock_search, capsys):
        mock_search.return_value = [{
            "id": "IyR25B-IGyg",
            "title": "Back In Black",
            "url": "https://www.youtube.com/watch?v=IyR25B-IGyg",
            "duration": "4:15",
            "channel": "AC/DC",
            "thumbnail": "https://example.com/thumb.jpg",
        }]

        test_args = ["play_youtube.py", "Iron Man music", "--no-open", "--json"]
        with mock.patch.object(sys, "argv", test_args):
            ret = play_youtube.main()
            assert ret == 0

        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["success"] is True
        assert data["opened"] is False
        assert data["top"]["id"] == "IyR25B-IGyg"
