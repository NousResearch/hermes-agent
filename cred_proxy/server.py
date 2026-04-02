"""mitmproxy addon that substitutes hermes-proxy:// credential placeholders.

The addon intercepts every HTTP/HTTPS request flowing through the proxy and
replaces ``hermes-proxy://<name>`` tokens in headers and the request body.
mitmproxy handles all HTTPS MITM, CA management, cert issuance, and
Content-Length recalculation automatically.
"""

import logging

from .store import CredStore
from .substitutor import substitute

logger = logging.getLogger(__name__)


class CredentialProxyAddon:
    def __init__(self, store: CredStore | None = None):
        self._store = store if store is not None else CredStore()

    def request(self, flow) -> None:
        """Substitute credential placeholders in request headers and body."""
        # Headers
        for key in list(flow.request.headers.keys()):
            val = flow.request.headers[key]
            new_val = substitute(val, self._store)
            if new_val != val:
                logger.debug("Substituted credential in request header %r", key)
                flow.request.headers[key] = new_val

        # Body — mitmproxy recalculates Content-Length when content is set
        if flow.request.content:
            text = flow.request.content.decode("utf-8", errors="replace")
            new_text = substitute(text, self._store)
            if new_text != text:
                logger.debug("Substituted credential in request body")
                flow.request.content = new_text.encode("utf-8", errors="replace")


async def run_proxy(port: int, unix_socket=None, on_started=None) -> None:
    """Start mitmproxy on 127.0.0.1:<port> and block until shutdown.

    unix_socket is accepted for API compatibility but ignored — mitmproxy
    uses TCP only.  on_started() is called once the proxy is fully listening.
    """
    from pathlib import Path

    from mitmproxy import options
    from mitmproxy.tools.dump import DumpMaster

    _confdir = str(Path.home() / ".hermes" / "state" / "cred-proxy-ca")
    Path(_confdir).mkdir(parents=True, exist_ok=True)

    store = CredStore()
    addon = CredentialProxyAddon(store)

    class _StartupNotifier:
        def running(self) -> None:
            if on_started is not None:
                on_started()

    opts = options.Options(
        listen_host="127.0.0.1",
        listen_port=port,
        confdir=_confdir,
    )
    master = DumpMaster(opts, with_termlog=False, with_dumper=False)
    master.addons.add(_StartupNotifier())
    master.addons.add(addon)
    await master.run()
