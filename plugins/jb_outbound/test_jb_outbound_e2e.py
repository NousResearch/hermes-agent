"""E2e du plugin jb_outbound : boucle complète sur HTTP loopback RÉEL (aucun mock réseau).

On câble le VRAI listener du plugin + un STUB de control daemon (serveur HTTP loopback qui imite
jb-daemon : reçoit /v1/draft et /v1/result). On prouve l'invariant produit : **aucun envoi réel
n'a lieu avant la décision approuvée**, et l'envoi est rejoué (puis le résultat remonté) après.

Seul le « cerveau » de l'envoi réel (registry.dispatch de Hermes) est simulé — tout le transport
loopback (dépôt de proposition, push de décision, remontée de résultat) est exercé pour de vrai.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import types
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_PLUGINS_DIR = Path(__file__).resolve().parents[1]
if str(_PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGINS_DIR))

import jb_outbound.listener as listener  # noqa: E402
import jb_outbound.middleware as middleware  # noqa: E402

_DRAFT_PORT = 18442   # stub daemon (où le plugin dépose drafts/résultats)
_LISTEN_PORT = 18444  # listener du plugin (où le daemon pousse les décisions)


class _StubDaemon:
    """Faux control daemon : enregistre les drafts et résultats reçus sur le loopback."""

    def __init__(self) -> None:
        self.drafts: list = []
        self.results: list = []
        stub = self

        class _H(BaseHTTPRequestHandler):
            def log_message(self, *_a):  # silence
                return

            def do_POST(self):  # noqa: N802
                n = int(self.headers.get("Content-Length", "0") or "0")
                body = json.loads(self.rfile.read(n) or b"{}")
                if self.path == "/v1/draft":
                    stub.drafts.append(body)
                elif self.path == "/v1/result":
                    stub.results.append(body)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')

        self._server = ThreadingHTTPServer(("127.0.0.1", _DRAFT_PORT), _H)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self._server.shutdown()


def _post(url: str, payload: dict) -> int:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        return int(getattr(r, "status", 200))


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("JB_DRAFT_ADDR", f"127.0.0.1:{_DRAFT_PORT}")
    monkeypatch.setenv("JB_DECISION_PUSH_URL", f"http://127.0.0.1:{_LISTEN_PORT}/jb/decision")

    # registry.dispatch simulé : enregistre les envois RÉELS (le replay l'appelle).
    sent: list = []
    fake_registry = types.SimpleNamespace(
        dispatch=lambda name, args: sent.append((name, args)) or "{}"
    )
    tools_mod = types.ModuleType("tools")
    reg_mod = types.ModuleType("tools.registry")
    reg_mod.registry = fake_registry  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tools", tools_mod)
    monkeypatch.setitem(sys.modules, "tools.registry", reg_mod)

    listener._started = False  # autoriser un démarrage propre du listener pour le test
    return sent


def _raise_if_called(_a):
    raise AssertionError("l'envoi NE doit PAS s'exécuter avant validation")


def test_e2e_rien_ne_part_avant_la_decision(env):
    sent = env
    stub = _StubDaemon()
    try:
        listener.start()  # vrai listener du plugin sur 127.0.0.1:18444
        time.sleep(0.15)

        # 1) L'assistant tente un envoi Telegram → intercepté → proposition déposée, RIEN envoyé.
        out = json.loads(
            middleware.make_middleware()(
                tool_name="send_message",
                args={"chat_id": "42", "content": "Bonjour, voici votre devis."},
                next_call=_raise_if_called,
            )
        )
        assert out["status"] == "queued_for_approval"
        assert len(stub.drafts) == 1, "une proposition a été déposée sur le daemon"
        assert stub.drafts[0]["kind"] == "sms"
        jb_id = stub.drafts[0]["payload"]["jb_id"]

        # INVARIANT : aucun envoi réel tant que le client n'a pas validé.
        assert sent == []

        # 2) Le daemon pousse la décision approuvée sur le listener du plugin.
        _post(
            f"http://127.0.0.1:{_LISTEN_PORT}/jb/decision",
            {"id": "prop-42", "kind": "sms", "to": "42", "payload": {"jb_id": jb_id}},
        )

        # 3) Attendre la remontée du résultat (le replay s'exécute côté listener).
        deadline = time.time() + 3.0
        while time.time() < deadline and not stub.results:
            time.sleep(0.02)

        # L'envoi réel a été rejoué à l'identique, APRÈS la décision.
        assert sent == [("send_message", {"chat_id": "42", "content": "Bonjour, voici votre devis."})]
        # Le résultat a été remonté au daemon (id de proposition round-trippé).
        assert stub.results and stub.results[-1] == {"id": "prop-42", "status": "executed", "error": ""}
    finally:
        stub.stop()
