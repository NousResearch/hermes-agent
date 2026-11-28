import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).parents[2] / "skills/media/minimax-music/scripts/generate_music.py"
)
SPEC = importlib.util.spec_from_file_location("minimax_music", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class Response:
    def __init__(self, data):
        self.data = data

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return self.data


def args(tmp_path, *extra):
    return MODULE.parser().parse_args([
        "--prompt",
        "piano",
        "--output-format",
        "hex",
        "--output",
        str(tmp_path / "song.mp3"),
        *extra,
    ])


def success_opener(requests):
    def opener(request, timeout):
        requests.append(request)
        return Response(
            json.dumps({
                "base_resp": {"status_code": 0},
                "data": {"status": 2, "audio": "494433"},
            }).encode()
        )

    return opener


def test_generate_music_uses_global_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "global-key")
    requests = []

    output = MODULE.generate(
        args(tmp_path, "--instrumental"), opener=success_opener(requests)
    )

    assert requests[0].full_url == MODULE.ENDPOINTS["global"]
    assert requests[0].headers["Authorization"] == "Bearer global-key"
    assert output.read_bytes() == b"ID3"


def test_generate_music_uses_cn_api_key_and_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "global-key")
    monkeypatch.setenv("MINIMAX_CN_API_KEY", "cn-key")
    request_args = args(
        tmp_path, "--region", "cn", "--aigc-watermark", "--instrumental"
    )
    requests = []

    MODULE.generate(request_args, opener=success_opener(requests))
    payload = json.loads(requests[0].data)

    assert requests[0].full_url == MODULE.ENDPOINTS["cn"]
    assert requests[0].headers["Authorization"] == "Bearer cn-key"
    assert payload["aigc_watermark"] is True


def test_generate_music_sends_lyrics_optimizer(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "global-key")
    request_args = args(tmp_path, "--lyrics-optimizer")
    requests = []

    MODULE.generate(request_args, opener=success_opener(requests))
    payload = json.loads(requests[0].data)

    assert payload["lyrics_optimizer"] is True


def test_generate_music_allows_lyrics_without_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "global-key")
    request_args = MODULE.parser().parse_args([
        "--lyrics",
        "[Verse]\nOnly lyrics are needed",
        "--output-format",
        "hex",
        "--output",
        str(tmp_path / "song.mp3"),
    ])
    requests = []

    MODULE.generate(request_args, opener=success_opener(requests))
    payload = json.loads(requests[0].data)

    assert "prompt" not in payload
    assert payload["lyrics"] == "[Verse]\nOnly lyrics are needed"


def test_generate_music_downloads_url_output(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "global-key")
    requests = []

    def opener(request, timeout):
        requests.append(request)
        if len(requests) == 1:
            return Response(
                json.dumps({
                    "base_resp": {"status_code": 0},
                    "data": {
                        "status": 2,
                        "audio": "https://example.com/song.mp3",
                    },
                }).encode()
            )
        return Response(b"downloaded-audio")

    output = MODULE.generate(
        args(tmp_path, "--output-format", "url", "--instrumental"), opener=opener
    )

    assert requests[1] == "https://example.com/song.mp3"
    assert output.read_bytes() == b"downloaded-audio"


def test_cn_region_requires_cn_api_key(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "global-key")
    monkeypatch.delenv("MINIMAX_CN_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="MINIMAX_CN_API_KEY is required"):
        MODULE.generate(args(tmp_path, "--region", "cn", "--instrumental"))


def test_vocal_generation_requires_lyrics_or_optimizer(tmp_path):
    with pytest.raises(SystemExit, match="--lyrics or --lyrics-optimizer"):
        MODULE.generate(args(tmp_path))


@pytest.mark.parametrize(
    "cover_args",
    [
        [],
        ["--audio-url", "https://example.com/a.mp3", "--audio-base64", "AAAA"],
    ],
)
def test_cover_model_requires_exactly_one_direct_audio_source(tmp_path, cover_args):
    with pytest.raises(SystemExit, match="exactly one"):
        MODULE.generate(args(tmp_path, "--model", "music-cover", *cover_args))


def test_cover_feature_id_rejects_direct_audio(tmp_path):
    with pytest.raises(SystemExit, match="cannot be combined"):
        MODULE.generate(
            args(
                tmp_path,
                "--model",
                "music-cover",
                "--cover-feature-id",
                "feature",
                "--audio-url",
                "https://example.com/a.mp3",
                "--lyrics",
                "new lyrics",
            )
        )


@pytest.mark.parametrize("flag", ["--lyrics-optimizer", "--instrumental"])
def test_cover_model_rejects_generation_only_options(tmp_path, flag):
    with pytest.raises(SystemExit, match="generation-only options"):
        MODULE.generate(
            args(
                tmp_path,
                "--model",
                "music-cover",
                "--audio-url",
                "https://example.com/a.mp3",
                flag,
            )
        )


def test_cover_model_requires_prompt(tmp_path):
    request_args = MODULE.parser().parse_args([
        "--model",
        "music-cover",
        "--audio-url",
        "https://example.com/a.mp3",
        "--output",
        str(tmp_path / "cover.mp3"),
    ])

    with pytest.raises(SystemExit, match="--prompt is required for cover models"):
        MODULE.generate(request_args)


def test_cover_feature_id_requires_lyrics(tmp_path):
    with pytest.raises(SystemExit, match="--lyrics is required"):
        MODULE.generate(
            args(
                tmp_path,
                "--model",
                "music-cover",
                "--cover-feature-id",
                "feature",
            )
        )


def test_cover_feature_id_is_sent_without_direct_audio(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "global-key")
    request_args = args(
        tmp_path,
        "--model",
        "music-cover",
        "--cover-feature-id",
        "feature",
        "--lyrics",
        "new lyrics",
    )
    requests = []

    MODULE.generate(request_args, opener=success_opener(requests))
    payload = json.loads(requests[0].data)

    assert payload["cover_feature_id"] == "feature"
    assert "audio_url" not in payload
    assert "audio_base64" not in payload


def test_generation_model_rejects_cover_inputs(tmp_path):
    with pytest.raises(SystemExit, match="cover inputs require"):
        MODULE.generate(args(tmp_path, "--audio-url", "https://example.com/a.mp3"))
