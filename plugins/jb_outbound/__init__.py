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
    from . import listener, delegation_activity, produce
    from .middleware import make_middleware

    ctx.register_middleware("tool_execution", make_middleware())
    # Bus de délégation → fil d'activité (D2) + contributeurs du draft (D3). Hooks natifs, thread parent.
    ctx.register_hook("subagent_start", delegation_activity.on_subagent_start)
    ctx.register_hook("subagent_stop", delegation_activity.on_subagent_stop)
    # Outil « creer_support » (Ma marque, Option A) : l'agent DÉCLENCHE la production d'un support
    # brandé via le daemon loopback (même chemin que les drafts) ; la plateforme rend le document
    # (gabarits fixes + charte) et le range dans les Documents du client. Toolset PLUGIN (activé par
    # défaut sur toutes les plateformes) ; check_fn = même garde que le plugin → invisible hors box.
    ctx.register_tool(
        name="creer_support",
        toolset="jb_studio",
        schema=produce.CREER_SUPPORT_SCHEMA,
        handler=lambda args, **kw: produce.creer_support(
            type_de_support=args.get("type_de_support", ""),
            contenu=args.get("contenu"),
            kit_id=args.get("kit_id"),
        ),
        check_fn=produce.check_creer_support_requirements,
        emoji="🎨",
    )
    listener.start()  # idempotent, loopback-only, no-op si JB_DECISION_PUSH_URL absent
    logger.info(
        "jb_outbound: interception d'envoi active (proposition → validation → exécution)"
    )
