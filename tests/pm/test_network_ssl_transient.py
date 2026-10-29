"""TLS handshake failures that are network faults, not protocol errors.

A proxy that accepts the connection and then stalls mid-handshake makes
``urllib`` raise a bare ``ssl.SSLError`` ("_ssl.c:993: The handshake operation
timed out"). Classifying every bare ``SSLError`` as permanent meant a package
download died on that transient fault instead of retrying.
"""

from __future__ import annotations

import ssl

import pytest

from pm.network import is_transient


@pytest.mark.parametrize(
    "error",
    [
        ssl.SSLError("_ssl.c:993: The handshake operation timed out"),
        ssl.SSLError("_ssl.c:988: The handshake operation timed out"),
        ssl.SSLError("timed out"),
    ],
)
def test_handshake_timeout_is_transient(error):
    assert is_transient(error) is True


@pytest.mark.parametrize(
    "error",
    [
        ssl.SSLError("certificate verify failed"),
        ssl.SSLError("wrong version number"),
        ssl.SSLError("tlsv1 alert protocol version"),
    ],
)
def test_protocol_errors_stay_permanent(error):
    assert is_transient(error) is False


def test_eof_errors_stay_transient():
    assert is_transient(ssl.SSLEOFError("eof")) is True
    assert is_transient(ssl.SSLZeroReturnError("zero return")) is True
