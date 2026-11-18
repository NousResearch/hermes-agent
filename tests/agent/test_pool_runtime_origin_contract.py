"""Typed pool endpoint declarations reject controls and malformed opt-in sets atomically."""

import pytest

from agent.credential_pool_plugin import _allowed_origins, exact_https_origin
from types import SimpleNamespace

ORIGIN = "https://api.business.example"


@pytest.mark.parametrize("control", ["\t", "\r", "\n", "\x00", "\x1f", "\x7f"])
def test_origin_rejects_controls_before_url_parser_normalizes(control):
    assert exact_https_origin("https://api.busi" + control + "ness.example") is None


@pytest.mark.parametrize("declaration", [
    42, False, None, ORIGIN, {ORIGIN: True}, {ORIGIN},
    (ORIGIN, 42), (ORIGIN, "https://invalid.example/path"),
    (ORIGIN, "https://api.busi\tness.example"),
])
def test_malformed_declaration_authorizes_no_origins(declaration):
    assert _allowed_origins(SimpleNamespace(credential_pool_base_urls=declaration)) == set()


@pytest.mark.parametrize("declaration", [(ORIGIN,), [ORIGIN, ORIGIN + ":443"], ()])
def test_valid_origin_sequence_is_normalized(declaration):
    assert _allowed_origins(SimpleNamespace(credential_pool_base_urls=declaration)) == ({ORIGIN} if declaration else set())
