"""Listener loopback des décisions approuvées (cible de JB_DECISION_PUSH_URL).

Thread HTTP léger (stdlib) démarré au chargement du plugin. Bind STRICTEMENT loopback (jamais
0.0.0.0) — garde-fou symétrique de `requireLoopbackURL` côté daemon Go. Reçoit les `DecisionItem`
poussées par le daemon et délègue au replay. Best-effort : un échec de bind (port déjà pris =
déjà démarré) est avalé pour rester idempotent.
"""

from __future__ import annotations

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

logger = logging.getLogger(__name__)

_LOOPBACK_HOSTS = {"127.0.0.1", "::1", "localhost"}
_started = False
_lock = threading.Lock()


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:  # noqa: N802 (API imposée par BaseHTTPRequestHandler)
        from . import config, replay

        if self.path.rstrip("/") != config.decision_path().rstrip("/"):
            self.send_response(404)
            self.end_headers()
            return

        length = int(self.headers.get("Content-Length", "0") or "0")
        raw = self.rfile.read(length) if length else b"{}"
        try:
            decision = json.loads(raw or b"{}")
        except Exception:
            self.send_response(400)
            self.end_headers()
            return

        # On acquitte vite, puis on exécute (l'exécution réelle peut prendre un peu de temps).
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

        try:
            replay.handle_decision(decision)
        except Exception as exc:  # ne jamais laisser une exception tuer le serveur
            logger.warning("jb_outbound: traitement de décision échoué : %s", exc)

    def log_message(self, *_args) -> None:  # silence des logs HTTP par requête
        return


def start() -> None:
    """Démarre le listener (idempotent, loopback-only, no-op hors box Jean-Billie)."""
    global _started
    from . import config

    if not config.enabled():
        return

    with _lock:
        if _started:
            return
        host, port = config.listen_addr()
        if host not in _LOOPBACK_HOSTS:
            logger.error(
                "jb_outbound: refus de binder le listener hors loopback (%s) — désactivé.", host
            )
            return
        try:
            server = ThreadingHTTPServer((host, port), _Handler)
        except OSError as exc:
            # Port déjà pris → on suppose qu'une instance écoute déjà (idempotent).
            logger.info("jb_outbound: listener non démarré (%s:%s déjà pris ? %s)", host, port, exc)
            return
        threading.Thread(
            target=server.serve_forever, daemon=True, name="jb-outbound-listener"
        ).start()
        _started = True
        logger.info("jb_outbound: listener de décisions à l'écoute sur %s:%s", host, port)
