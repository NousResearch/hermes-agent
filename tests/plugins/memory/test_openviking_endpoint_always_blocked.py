"""OpenViking endpoint always-blocked floor."""

from plugins.memory.openviking import _DEFAULT_ENDPOINT, _normalize_openviking_url


def test_openviking_blocks_metadata_endpoint():
    assert _normalize_openviking_url("http://169.254.169.254/") == _DEFAULT_ENDPOINT


def test_openviking_keeps_default_loopback():
    assert _normalize_openviking_url("http://127.0.0.1:1933") == "http://127.0.0.1:1933"


def test_openviking_blocks_ecs_metadata_hostname():
    assert (
        _normalize_openviking_url("http://metadata.google.internal/computeMetadata/v1/")
        == _DEFAULT_ENDPOINT
    )
