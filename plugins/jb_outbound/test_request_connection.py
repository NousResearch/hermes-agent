"""Tests de « request_tool_connection » (capacité B) après résorption F2.

Couvre : l'enregistrement PAR LE PLUGIN (ctx.register_tool, toolset « messaging » conservé —
l'entrée _HERMES_CORE_TOOLS du cœur est résorbée), le gate check_fn (JB_DECISION_PUSH_URL),
le repli honnête quand le daemon est injoignable, et la résolution du toolset registre
« messaging » via toolsets.get_toolset (le chemin qu'emprunte l'allowlist platform_toolsets).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Rendre le paquet `jb_outbound` importable comme paquet de premier niveau (plugins/ sur le path)
# — même alias que le PluginManager et les autres suites du plugin, pour que les identités de
# fonctions (check_fn) soient comparables (un import sous `plugins.jb_outbound` créerait un
# SECOND objet module).
_PLUGINS_DIR = Path(__file__).resolve().parents[1]
if str(_PLUGINS_DIR) not in sys.path:
    sys.path.insert(0, str(_PLUGINS_DIR))

import jb_outbound.request_connection as request_connection  # noqa: E402


@pytest.fixture()
def on_box(monkeypatch):
    """Simule la box Jean-Billie (même garde que le plugin)."""
    monkeypatch.setenv("JB_DECISION_PUSH_URL", "http://127.0.0.1:8444/v1/decisions")
    yield


def test_check_fn_gate(monkeypatch):
    monkeypatch.delenv("JB_DECISION_PUSH_URL", raising=False)
    assert request_connection.check_request_tool_connection_requirements() is False
    monkeypatch.setenv("JB_DECISION_PUSH_URL", "http://127.0.0.1:8444/v1/decisions")
    assert request_connection.check_request_tool_connection_requirements() is True


def test_intention_vide_refusee():
    out = json.loads(request_connection.request_tool_connection("  "))
    assert out == {"error": "intention requise"}


def test_daemon_injoignable_repli_honnete(monkeypatch):
    """Pas de bluff : daemon absent → statut « unavailable » + message franc, jamais d'exception."""

    def _boom(*a, **k):  # urlopen
        raise OSError("connexion refusée")

    monkeypatch.setattr(request_connection.urllib.request, "urlopen", _boom)
    out = json.loads(request_connection.request_tool_connection("facturer un client"))
    assert out["status"] == "unavailable"
    assert out["message"]


def test_register_enregistre_l_outil(on_box):
    """Le plugin enregistre request_tool_connection via ctx.register_tool (zéro patch du cœur)."""
    import jb_outbound

    calls = {"tools": [], "middleware": [], "hooks": []}

    class FakeCtx:
        def register_tool(self, **kw):
            calls["tools"].append(kw)

        def register_middleware(self, kind, cb):
            calls["middleware"].append(kind)

        def register_hook(self, name, cb):
            calls["hooks"].append(name)

    jb_outbound.register(FakeCtx())
    names = [t["name"] for t in calls["tools"]]
    assert "request_tool_connection" in names
    tool = next(t for t in calls["tools"] if t["name"] == "request_tool_connection")
    # Toolset « messaging » CONSERVÉ : c'est l'entrée explicite de l'allowlist platform_toolsets
    # du bundle (lane S, monorepo) qui expose l'outil — la changer casserait le couplage cross-repo.
    assert tool["toolset"] == "messaging"
    assert tool["check_fn"] is request_connection.check_request_tool_connection_requirements
    assert tool["schema"]["parameters"]["required"] == ["capability"]
    # Handler : round-trip minimal sans réseau (intention vide → refus local).
    out = json.loads(tool["handler"]({"capability": ""}))
    assert out == {"error": "intention requise"}


def test_toolset_messaging_resolu_depuis_le_registre():
    """« messaging » n'a AUCUNE entrée statique dans TOOLSETS : une fois l'outil enregistré au
    registre (comme le fait ctx.register_tool), resolve_toolset("messaging") doit le porter —
    c'est le chemin exact de l'allowlist platform_toolsets émise par le bundle."""
    import toolsets
    from tools.registry import registry

    assert "messaging" not in toolsets.TOOLSETS
    # L'entrée cœur est bien résorbée (F2) : plus de request_tool_connection dans le composite.
    assert "request_tool_connection" not in toolsets._HERMES_CORE_TOOLS

    already = "request_tool_connection" in registry.get_tool_names_for_toolset("messaging")
    if not already:
        registry.register(
            name="request_tool_connection",
            toolset="messaging",
            schema=request_connection.REQUEST_TOOL_CONNECTION_SCHEMA,
            handler=lambda args, **kw: request_connection.request_tool_connection(
                capability=args.get("capability", "")
            ),
            check_fn=request_connection.check_request_tool_connection_requirements,
            emoji="🔌",
        )
    assert "request_tool_connection" in toolsets.resolve_toolset("messaging")
