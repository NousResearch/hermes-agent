"""Regression tests: OlmMachine's crypto logger must tolerate a plain stdlib
``mau.crypto`` logger.

OlmMachine types its logger as mautrix's ``TraceLogger`` and calls
``trace()``/``silly()`` while handling encrypted to-device events. Nothing in
this process guarantees ``logging.getLogger("mau.crypto")`` returns that
custom class — when it is a plain stdlib ``Logger``, the missing ``trace()``
raises ``AttributeError`` inside the crypto handler and the room-key event is
dropped before processing, silently breaking E2EE decryption. The adapter
therefore passes an explicit compatibility shim to ``OlmMachine``.
"""

import logging


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: list = []

    def emit(self, record) -> None:
        self.records.append(record)


def _plain_logger() -> logging.Logger:
    # Direct construction bypasses any custom logger class installed via
    # logging.setLoggerClass, so this is guaranteed to be a plain Logger
    # regardless of test import order.
    plain = logging.Logger("test.mau.crypto.plain", level=1)
    assert not hasattr(plain, "trace")
    assert not hasattr(plain, "silly")
    return plain


def test_crypto_logger_shim_preserves_trace_and_silly_levels():
    from plugins.platforms.matrix.adapter import _MautrixCryptoLogger

    plain = _plain_logger()
    handler = _ListHandler()
    plain.addHandler(handler)

    shim = _MautrixCryptoLogger(plain, {})
    shim.trace("encrypted to-device event")
    shim.silly("crypto detail")

    assert [r.levelno for r in handler.records] == [5, 1]
    assert handler.records[0].getMessage() == "encrypted to-device event"
    assert handler.records[1].getMessage() == "crypto detail"


def test_crypto_logger_shim_does_not_promote_trace_or_silly_to_debug():
    from plugins.platforms.matrix.adapter import _MautrixCryptoLogger

    plain = logging.Logger("test.mau.crypto.debug", level=logging.DEBUG)
    handler = _ListHandler()
    plain.addHandler(handler)

    shim = _MautrixCryptoLogger(plain, {})
    shim.trace("encrypted to-device event")
    shim.silly("crypto detail")

    assert handler.records == []


def test_crypto_logger_factory_wraps_mau_crypto_logger():
    from plugins.platforms.matrix.adapter import _mautrix_crypto_logger

    shim = _mautrix_crypto_logger()

    assert shim.logger is logging.getLogger("mau.crypto")
    # The shimmed logger must offer mautrix's non-standard levels no matter
    # which class the ambient "mau.crypto" logger happens to be.
    assert callable(shim.trace)
    assert callable(shim.silly)
