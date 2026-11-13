"""A memoized ``/models`` catalog answers only lookups made with the credential that fetched it (#154244).

The endpoint here answers 401 without ``Authorization`` and a 777,777-token window with one (or a
different window when ``keyless`` lists one), so the window a lookup gets shows whose catalog answered.
"""

from unittest.mock import MagicMock, patch

import pytest

import hermes_agent.agent.model_metadata as mm

URL = "https://keyed.example/v1"


@pytest.fixture(autouse=True)
def _fresh_memos(tmp_path, monkeypatch):
    mm._endpoint_model_metadata_cache.clear()
    mm._endpoint_model_metadata_cache_time.clear()
    monkeypatch.setattr(mm, "_get_endpoint_metadata_cache_path", lambda: tmp_path / "endpoint_model_metadata.json")
    yield
    mm._endpoint_model_metadata_cache.clear()
    mm._endpoint_model_metadata_cache_time.clear()


class _Endpoint:
    def __init__(self, keyless=None):
        self.keyless = keyless  # None: a keyless caller gets 401
        self.calls = []

    def stream(self, url, headers=None, **kwargs):
        keyed = bool((headers or {}).get("Authorization"))
        self.calls.append("keyed" if keyed else "keyless")
        response = MagicMock()
        window = 777_777 if keyed else self.keyless
        response.status_code = 200 if window else 401
        response.json.return_value = {"data": [{"id": "m-x", "context_length": window}]}
        ctx = MagicMock()
        ctx.__enter__.return_value = response
        return ctx


def _window(api_key):
    return mm.fetch_endpoint_model_metadata(URL, api_key=api_key).get("m-x", {}).get("context_length")


def _new_process():
    mm._endpoint_model_metadata_cache.clear()
    mm._endpoint_model_metadata_cache_time.clear()


def test_a_keyless_401_does_not_answer_a_later_minted_key_lookup():
    endpoint = _Endpoint()
    with patch.object(mm.model_metadata_http, "stream", side_effect=endpoint.stream):
        assert _window("") is None
        assert _window(lambda: "minted-token") == 777_777
    assert endpoint.calls == ["keyless", "keyed"]


def test_a_keyless_catalog_on_disk_does_not_answer_a_keyed_lookup():
    endpoint = _Endpoint(keyless=111_111)
    with patch.object(mm.model_metadata_http, "stream", side_effect=endpoint.stream):
        assert _window("") == 111_111
        _new_process()
        assert _window("sk-real") == 777_777
        _new_process()
        assert _window(lambda: "minted-token") == 777_777
    assert endpoint.calls == ["keyless", "keyed", "keyed"]


def test_each_credential_still_reuses_its_own_disk_memo_across_processes():
    endpoint = _Endpoint(keyless=111_111)
    with patch.object(mm.model_metadata_http, "stream", side_effect=endpoint.stream):
        for api_key in ("", "sk-real", lambda: "minted-token"):
            first = _window(api_key)
            _new_process()
            assert _window(api_key) == first
            _new_process()
    assert endpoint.calls == ["keyless", "keyed", "keyed"]
