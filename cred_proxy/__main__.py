"""Entry point for ``python -m cred_proxy``.

Used by daemon.start() to spawn the server as a background process.
The PID file and Unix socket are created inside _run_server() once the
server is bound, ensuring callers only see the daemon as ready once it
is live.
"""

from cred_proxy.daemon import _run_server

_run_server()
