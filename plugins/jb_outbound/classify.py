"""Classification d'un appel d'outil : exécuter / proposer / bloquer.

« Rien ne part sans accord » couvre TOUS les canaux. Deux familles d'outils émettent vers
l'extérieur et passent toutes deux par le `tool_execution` middleware :
  - `send_message` : envoi gateway (Telegram, etc.).
  - `mcp_composio_*` : email / réseaux sociaux via Composio (appels d'outils MCP).

Décision (confirmée) : tout outil d'envoi non répertorié est **BLOQUÉ** (fail-closed) — jamais
auto-envoyé. Un outil composio clairement en lecture passe ; clairement en écriture devient une
proposition ; ambigu → bloqué (on élargit les listes au besoin).
"""

from __future__ import annotations

PASS = "pass"        # exécuter normalement (lecture / hors périmètre)
PROPOSE = "propose"  # transformer en proposition à valider
BLOCK = "block"      # fail-closed : refuser (envoi non répertorié)

# Envois gateway directs.
SEND_TOOLS = {"send_message"}

_COMPOSIO_PREFIX = "mcp_composio_"

# Marqueurs d'ACTION dans le nom d'outil MCP (ex. mcp_composio_GMAIL_SEND_EMAIL).
_WRITE_MARKERS = ("SEND", "CREATE", "POST", "REPLY", "PUBLISH", "ADD", "UPDATE", "DELETE", "DRAFT")
_READ_MARKERS = ("GET", "LIST", "FETCH", "SEARCH", "READ", "FIND", "RETRIEVE", "EXPORT")


def classify(tool_name: str) -> str:
    name = tool_name or ""
    if name in SEND_TOOLS:
        return PROPOSE
    if name.startswith(_COMPOSIO_PREFIX):
        upper = name.upper()
        if any(m in upper for m in _WRITE_MARKERS):
            return PROPOSE
        if any(m in upper for m in _READ_MARKERS):
            return PASS
        # Action composio inconnue → fail-closed (on ne laisse rien partir par défaut).
        return BLOCK
    # Tout le reste (outils internes, lecture, etc.) : exécution normale.
    return PASS
