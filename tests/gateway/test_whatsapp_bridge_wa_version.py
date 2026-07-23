"""The WhatsApp bridge must negotiate the version WhatsApp is actually serving.

Baileys exposes two version helpers, and they are not interchangeable:

* ``fetchLatestBaileysVersion()`` reads the *Baileys project's* committed
  ``Defaults/index.ts`` off GitHub master and scrapes ``const version = [...]``
  out of a hardcoded line offset. It reports whatever the library maintainers
  last committed, which lags the live protocol.
* ``fetchLatestWaWebVersion()`` fetches ``https://web.whatsapp.com/sw.js`` and
  parses ``client_revision`` — the protocol version WhatsApp Web is serving
  right now.

When the bridge hands ``makeWASocket`` a stale version, WhatsApp rejects the
handshake and the bridge never connects. So the version passed to the socket
has to originate from ``fetchLatestWaWebVersion``.

``bridge.js`` starts an Express server and a Baileys socket at module load (see
the note atop ``bridge.native.test.mjs``), so it cannot be imported for a unit
test. These are source contracts on the two lines that carry the invariant.
"""

from __future__ import annotations

import re
from pathlib import Path

BRIDGE_JS = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "whatsapp-bridge"
    / "bridge.js"
)

_BAILEYS_IMPORT = re.compile(
    r"import\s*\{(?P<names>[^}]*)\}\s*from\s*['\"]@whiskeysockets/baileys['\"]"
)
# `const { version } = await <fn>()` — whatever the bridge awaits to get it.
_VERSION_SOURCE = re.compile(
    r"const\s*\{\s*version\s*\}\s*=\s*await\s+(?P<fn>\w+)\s*\("
)


def _source() -> str:
    return BRIDGE_JS.read_text(encoding="utf-8")


def _baileys_imports(source: str) -> set[str]:
    """Named specifiers the bridge pulls out of @whiskeysockets/baileys."""
    match = _BAILEYS_IMPORT.search(source)
    assert match, "bridge.js no longer has a named import from @whiskeysockets/baileys"
    return {name.strip() for name in match.group("names").split(",") if name.strip()}


def _make_wa_socket_options(source: str) -> str:
    """The object literal passed to makeWASocket({...})."""
    start = source.index("makeWASocket({") + len("makeWASocket(")
    depth = 0
    for end, char in enumerate(source[start:], start):
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[start : end + 1]
    raise AssertionError("unbalanced makeWASocket({...}) call in bridge.js")


def test_bridge_imports_the_wa_web_version_helper() -> None:
    """The bridge imports the live-WA helper, not the Baileys-pinned one."""
    imports = _baileys_imports(_source())

    assert "fetchLatestWaWebVersion" in imports
    assert "fetchLatestBaileysVersion" not in imports


def test_socket_version_comes_from_the_wa_web_fetch() -> None:
    """The version handed to makeWASocket is awaited from fetchLatestWaWebVersion."""
    source = _source()

    match = _VERSION_SOURCE.search(source)
    assert match, "bridge.js no longer derives `version` from an awaited call"
    assert match.group("fn") == "fetchLatestWaWebVersion"

    # A stale helper left behind anywhere would still be a live regression.
    assert "fetchLatestBaileysVersion(" not in source

    # ...and the fetched version actually reaches the socket.
    assert re.search(r"(?m)^\s*version\s*,", _make_wa_socket_options(source))
