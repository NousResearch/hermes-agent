"""Regression (#78359): a response ending in a bare URL is not truncated.

``_has_natural_response_ending()`` only treated punctuation and emoji as
natural endings. A URL such as ``http://host/index.html`` ends with ``l``,
which is not in the punctuation set, so on Ollama-hosted GLM models
``_should_treat_stop_as_truncated()`` fired spuriously, triggering a fake
continuation whose text got glued to the URL without a separator.

The fix: if the last whitespace-delimited token of the stripped content
starts with ``http://``, ``https://``, or ``ftp://``, treat it as a
natural ending. Any URL is a natural ending — no extension matching.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest


@pytest.fixture(scope="module")
def has_natural_response_ending():
    with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}):
        from run_agent import AIAgent

        return AIAgent._has_natural_response_ending


@pytest.mark.parametrize(
    "content",
    [
        "http://host/index.html",
        "https://example.com/path/to/page",
        "ftp://server/file.txt",
        "http://host:8731/path/index.html",
        "Check this: http://host/page",
    ],
)
def test_url_ending_is_natural(has_natural_response_ending, content):
    assert has_natural_response_ending(content) is True


def test_period_ending_is_natural(has_natural_response_ending):
    assert has_natural_response_ending("Some text.") is True


def test_no_punctuation_is_not_natural(has_natural_response_ending):
    assert has_natural_response_ending("Some text") is False


def test_text_after_url_is_not_natural(has_natural_response_ending):
    # URL is not the last token, so it does not end the response.
    assert has_natural_response_ending("http://host/page and more text") is False


def test_empty_is_not_natural(has_natural_response_ending):
    assert has_natural_response_ending("") is False


def test_non_url_scheme_is_not_natural(has_natural_response_ending):
    assert has_natural_response_ending("MEDIA:/tmp/file.png") is False