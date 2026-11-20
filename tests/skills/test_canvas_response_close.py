"""Canvas pagination must close every response, including failing pages."""

import importlib.util
import io
from pathlib import Path
from unittest.mock import Mock

import pytest
import requests


@pytest.fixture
def canvas():
    path = Path(__file__).resolve().parents[2] / "optional-skills/productivity/canvas/scripts/canvas_api.py"
    spec = importlib.util.spec_from_file_location("canvas_response_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CANVAS_API_TOKEN = "test-token"
    return module


def _response(body, status=200, next_url=None):
    response = requests.Response()
    response.status_code = status
    response._content = body
    response.raw = io.BytesIO(body)
    response.close = Mock(wraps=response.close)
    if next_url:
        response.headers["Link"] = f'<{next_url}>; rel="next"'
    return response


@pytest.mark.parametrize("limit,expected", [(10, [1, 2, 3]), (2, [1, 2])])
def test_pagination_closes_pages(canvas, monkeypatch, limit, expected):
    url = "https://canvas.example/api/v1/courses"
    second_url = url + "?page=2"
    pages = [_response(b"[1]", next_url=second_url), _response(b"[2, 3]")]
    get = Mock(side_effect=pages)
    monkeypatch.setattr(canvas.requests, "get", get)

    assert canvas._paginated_get(url, params={"per_page": 1}, max_items=limit) == expected
    for page in pages:
        page.close.assert_called_once_with()
    assert get.call_args_list[0].kwargs["params"] == {"per_page": 1}
    assert get.call_args_list[1].args == (second_url,)
    assert get.call_args_list[1].kwargs["params"] is None


@pytest.mark.parametrize("status,body,error", [
    (503, b"[]", requests.HTTPError),
    (200, b"invalid json", requests.exceptions.JSONDecodeError),
])
def test_failed_page_closes_response(canvas, monkeypatch, status, body, error):
    response = _response(body, status)
    monkeypatch.setattr(canvas.requests, "get", lambda *args, **kwargs: response)

    with pytest.raises(error):
        canvas._paginated_get("https://canvas.example/api/v1/courses")
    response.close.assert_called_once_with()
