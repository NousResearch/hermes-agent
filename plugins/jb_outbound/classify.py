"""Classification d'un appel d'outil : exécuter / proposer / bloquer.

« Rien ne part sans accord » couvre TOUS les canaux. Trois familles d'outils passent par le
`tool_execution` middleware :
  - `send_message` : envoi gateway (Telegram, etc.) → proposition.
  - `mcp_composio_*` : email / réseaux sociaux via Composio. Lecture → passe ; écriture → proposition ;
    action composio AMBIGUË → bloquée (fail-closed, on élargit les listes au besoin).
  - MCP ADDITIONNELS (hors Composio) : déclarés par l'opérateur via la table managée
    (`managed.json`, cf. `managed.py`). Une fonction listée comme ACTION (write/egress) devient une
    proposition (dashboard) ; toute autre fonction d'un MCP additionnel est une lecture → passe.
    **Pas de blocage** pour ces serveurs : c'est l'allowlist `tools.include` + la table managée qui
    bornent ce qui est exposé et ce qui requiert validation.
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
    from . import managed

    name = tool_name or ""
    if name in SEND_TOOLS:
        return PROPOSE
    # Action d'un MCP additionnel managé (hors Composio) → proposition (dashboard). Les lectures de
    # ces serveurs ne sont PAS dans la table → elles tombent en PASS plus bas (aucun blocage).
    if managed.action_for(name) is not None:
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
