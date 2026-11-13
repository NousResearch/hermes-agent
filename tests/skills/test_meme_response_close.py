"""Template downloads release both supported HTTP backends."""

import importlib.util
import io
import urllib.request
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests


@pytest.mark.parametrize("backend", ["requests", "urllib"])
@pytest.mark.parametrize("fails", [False, True])
def test_template_download_closes_response(monkeypatch, backend, fails):
    path = Path(__file__).resolve().parents[2] / "optional-skills/creative/meme-generation/scripts/generate_meme.py"
    spec = importlib.util.spec_from_file_location("meme_response_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if backend == "requests":
        response = requests.Response()
        response.status_code = 503 if fails else 200
        response._content = b"image bytes"
        response.raw = io.BytesIO(response._content)
        response.close = Mock(wraps=response.close)
        monkeypatch.setattr(module._requests, "get", Mock(return_value=response))
        error = requests.HTTPError
    else:
        monkeypatch.setattr(module, "_requests", None)
        response = io.BytesIO(b"image bytes")
        if fails:
            response.read = Mock(side_effect=OSError("read interrupted"))
        monkeypatch.setattr(urllib.request, "urlopen", Mock(return_value=response))
        error = OSError

    if fails:
        with pytest.raises(error):
            module._fetch_url("https://image.example/template.png")
    else:
        assert module._fetch_url("https://image.example/template.png") == b"image bytes"
    if backend == "requests":
        response.close.assert_called_once_with()
    else:
        assert response.closed
