"""Security tests for the RSS watcher."""

from __future__ import annotations

import importlib.util
from io import BytesIO
from pathlib import Path

import pytest


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "optional-skills/devops/watchers/scripts/watch_rss.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("watch_rss_security_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_read_feed_rejects_response_over_limit():
    mod = _load_module()
    response = BytesIO(b"x" * (mod.MAX_FEED_BYTES + 1))

    with pytest.raises(ValueError, match="exceeds"):
        mod._read_limited_response(response)


def test_feed_url_rejects_local_file_scheme():
    mod = _load_module()

    with pytest.raises(ValueError, match="http or https"):
        mod._validate_feed_url("file:///etc/passwd")


def test_parse_feed_rejects_entity_expansion():
    mod = _load_module()
    payload = b"""<?xml version='1.0'?>
    <!DOCTYPE rss [<!ENTITY x 'boom'>]>
    <rss><channel><item><guid>&x;</guid></item></channel></rss>"""

    with pytest.raises(SystemExit):
        mod._parse_feed(payload)
