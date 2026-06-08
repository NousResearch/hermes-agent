"""jb_outbound — greffe Jean-Billie « rien ne part sans accord » sur Hermes Agent.

Un seul plugin, zéro patch du cœur. Enregistre un `tool_execution` middleware qui intercepte les
envois sortants (Telegram via `send_message`, email/réseaux via Composio MCP) et les transforme en
PROPOSITIONS déposées sur le control daemon (loopback). L'envoi réel n'a lieu qu'après approbation,
rejoué par le listener de décisions. Voir le README du plugin.

Le plugin est opt-in via `plugins.enabled: [jb_outbound]` (config Hermes) et reste passif hors de la
box Jean-Billie (quand `JB_DECISION_PUSH_URL` n'est pas posé).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    """Point d'entrée appelé par le PluginManager de Hermes au chargement."""
    from . import listener
    from .middleware import make_middleware

    ctx.register_middleware("tool_execution", make_middleware())
    listener.start()  # idempotent, loopback-only, no-op si JB_DECISION_PUSH_URL absent
    logger.info(
        "jb_outbound: interception d'envoi active (proposition → validation → exécution)"
    )
