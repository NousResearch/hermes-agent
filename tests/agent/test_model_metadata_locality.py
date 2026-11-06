"""Local endpoint classification tests."""

import pytest

from agent.model_metadata import is_local_endpoint


@pytest.mark.parametrize(
    ("base_url", "expected"),
    [
        ("http://gateway.namespace.svc.cluster.local:8080/v1", False),
        ("http://my-laptop.local:11434", True),
        ("http://localhost:11434", True),
        ("http://127.0.0.1:11434", True),
        ("http://192.168.1.20:11434", True),
        ("https://8.8.8.8/v1", False),
    ],
)
def test_is_local_endpoint_classifies_hosts(base_url, expected):
    assert is_local_endpoint(base_url) is expected
