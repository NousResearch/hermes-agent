import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "skills/media/minimax-music/scripts/generate_music.py"
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


def test_generate_music_uses_cn_fields_and_decodes_hex(tmp_path, monkeypatch):
    monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
    args = MODULE.parser().parse_args([
        "--prompt", "piano", "--region", "cn", "--aigc-watermark",
        "--output-format", "hex", "--output", str(tmp_path / "song.mp3"),
    ])
    requests = []

    def opener(request, timeout):
        requests.append(request)
        return Response(json.dumps({
            "base_resp": {"status_code": 0},
            "data": {"status": 2, "audio": "494433"},
        }).encode())

    output = MODULE.generate(args, opener=opener)
    payload = json.loads(requests[0].data)
    assert requests[0].full_url == MODULE.ENDPOINTS["cn"]
    assert requests[0].headers["Authorization"] == "Bearer test-key"
    assert payload["model"] == "music-3.0"
    assert payload["aigc_watermark"] is True
    assert output.read_bytes() == b"ID3"
