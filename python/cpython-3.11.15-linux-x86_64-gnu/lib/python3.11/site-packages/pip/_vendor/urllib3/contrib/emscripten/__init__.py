from __future__ import annotations

import pip._vendor.urllib3.connection as urllib3_connection

from ...connectionpool import HTTPConnectionPool, HTTPSConnectionPool
from .connection import EmscriptenHTTPConnection, EmscriptenHTTPSConnection


def inject_into_urllib3() -> None:
    # override connection classes to use emscripten specific classes
    # n.b. mypy complains about the overriding of classes below
    # if it isn't ignored
    HTTPConnectionPool.ConnectionCls = EmscriptenHTTPConnection
    HTTPSConnectionPool.ConnectionCls = EmscriptenHTTPSConnection
    urllib3_connection.HTTPConnection = EmscriptenHTTPConnection  # type: ignore[misc,assignment]
    urllib3_connection.HTTPSConnection = EmscriptenHTTPSConnection  # type: ignore[misc,assignment]
    urllib3_connection.VerifiedHTTPSConnection = EmscriptenHTTPSConnection  # type: ignore[assignment]
