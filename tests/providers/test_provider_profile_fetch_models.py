"""Provider catalog probes decode HTTP content encodings."""

from __future__ import annotations

import gzip
import json
from contextlib import contextmanager
from unittest.mock import patch

from providers.base import ProviderProfile


class _Response:
    def __init__(self, body: bytes, encoding: str):
        self._body = body
        self.headers = {"Content-Encoding": encoding}

    def read(self) -> bytes:
        return self._body


@contextmanager
def _response_context(response: _Response):
    yield response


def test_fetch_models_decodes_gzip_response():
    profile = ProviderProfile(name="test", base_url="https://provider.test/v1")
    payload = {"data": [{"id": "model-a"}, {"id": "model-b"}]}
    response = _Response(gzip.compress(json.dumps(payload).encode()), "gzip")

    with patch(
        "hermes_cli.urllib_security.open_credentialed_url",
        return_value=_response_context(response),
    ):
        assert profile.fetch_models() == ["model-a", "model-b"]
