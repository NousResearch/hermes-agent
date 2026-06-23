"""
Self-escalation: dynamic thinking ON/OFF toggle via model self-detection.

When enabled, the model runs with thinking OFF by default. If it encounters
a complex task, it signals [ESCALATE_THINKING: true] in its response, and
Hermes re-runs the iteration with thinking ON.

See ~/self-escalation-implementation-plan.md for full architecture.
Issue: https://github.com/NousResearch/hermes-agent/issues/50240
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict

logger = logging.getLogger(__name__)


@dataclass
class ThinkingState:
    """Tracks the current thinking mode state for a conversation turn."""

    enabled: bool = True
    active: bool = False  # True = thinking ON currently
    escalation_count: int = 0
    max_escalations: int = 1
    marker: str = "[ESCALATE_THINKING"
    default_state: str = "off"
    escalation_reason: str = ""  # Store reason for user notification

    def reset(self) -> None:
        """Reset escalation counter for a new user turn."""
        self.escalation_count = 0
        self.escalation_reason = ""  # Clear reason for next turn
        # active returns to default_state
        self.active = self.default_state == "on"


def load_thinking_config() -> ThinkingState:
    """Read config.yaml and return the initial thinking state.

    Uses the readonly (no-deepcopy) fast path since we only read values.
    Falls back to defaults if config is missing or malformed.
    """
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly()
        thinking_cfg = cfg.get("thinking", {})
    except Exception:
        thinking_cfg = {}

    enabled = thinking_cfg.get("enabled", True)
    default_state = thinking_cfg.get("default_state", "off")
    return ThinkingState(
        enabled=enabled,
        active=(default_state == "on"),
        escalation_count=0,
        max_escalations=thinking_cfg.get("max_escalations_per_turn", 1),
        marker=thinking_cfg.get("detection_marker", "[ESCALATE_THINKING"),
        default_state=default_state,
    )


# ── Provider support ─────────────────────────────────────────────────────

# All providers that support thinking toggle via extra_body
# (The OpenAI SDK rejects unknown top-level kwargs, so ALL providers
# must use extra_body to pass chat_template_kwargs to the backend.)
_SUPPORTED_PROVIDERS = {
    "custom", "llama-cpp", "llama-cpp-gpu-pc",
    "openrouter", "nous",
}


def is_provider_supported(provider: str) -> bool:
    """Return True if the provider supports chat_template_kwargs."""
    if not provider:
        return False
    return (provider or "").strip().lower() in _SUPPORTED_PROVIDERS


def inject_thinking_kwargs(
    api_kwargs: Dict[str, Any],
    state: ThinkingState,
    provider: str,
) -> Dict[str, Any]:
    """Inject chat_template_kwargs with enable_thinking into the API payload.

    All providers use extra_body — the OpenAI SDK rejects unknown top-level
    kwargs, so even llama.cpp servers need it nested in extra_body.

    Returns the modified api_kwargs dict.
    """
    if not state.enabled:
        return api_kwargs

    if not is_provider_supported(provider):
        return api_kwargs

    kwargs = {"enable_thinking": state.active}

    # ALL providers: use extra_body (SDK-compatible)
    api_kwargs.setdefault("extra_body", {})["chat_template_kwargs"] = kwargs

    logger.debug(
        "[self-escalation] Injected chat_template_kwargs: %s (provider=%s)",
        kwargs,
        provider,
    )
    return api_kwargs


def build_thinking_prompt(state: ThinkingState) -> str:
    """Build the ephemeral thinking-mode instruction prompt.

    This is injected into api_messages (the per-call copy), NOT into
    the cached system prompt. The prompt cache remains stable.

    Based on the optimized prompt from v4 tests (8/8 = 100% success rate).
    """
    state_label = "ON" if state.active else "OFF"

    if state.active:
        # Already in ON mode — no need for escalation instructions
        return f"\n\n## MODE ACTUEL\nthinking: {state_label}"

    return f"""

    ## SELF-ESCALATION (Dynamic Thinking Control)

    Tu es en thinking OFF. Si la tâche est complexe, tu DOIS répondre en premier avec :

    [ESCALATE_THINKING: true]
    [RAISON: 3 à 5 mots max, bref]

    Puis tu peux donner une réponse préliminaire.

    ### Quand escalader (thinking ON) :
    - Analyse de code avec bugs à identifier ou failles de sécurité
    - Debug d'erreur avec traceback à diagnostiquer
    - Données contradictoires à arbitrer
    - Planification multi-étapes (architecture, déploiement, stratégie)
    - Synthèse de plusieurs sources d'information
    - Question ouverte demandant réflexion et jugement
    - **Tool call échoué (2× d'affilée)** — si un appel d'outil renvoie une erreur, **escalade immédiatement** pour réfléchir à la correction

    ### Quand NE PAS escalader (répondre directement) :
    - Calcul mathématique, conversion d'unités
    - Traduction, reformulation
    - Question factuelle (date, définition, formule)
    - Lister le contenu d'un répertoire ou d'un fichier
    - Navigation, recherche d'information simple
    - Tool call direct (exécuter une commande, lire un fichier)

    ### Cas borderline — règle d'or :
    - Un simple "pourquoi cette erreur ?" avec une ligne d'erreur claire → RÉPOND DIRECTEMENT
    - "Quels fichiers sont dans ce dossier ?" → RÉPOND DIRECTEMENT
    - Mais "analyse ce code et trouve les problèmes" → ESCALADE
    - Mais "explique le concept X" → RÉPOND DIRECTEMENT

    ### Échecs d'outils → escalation automatique

    Si tes appels d'outils échouent (erreur inconnue, JSON invalide, mauvais arguments), **ne répète pas la même erreur**. Réponds d'abord avec :

    [ESCALATE_THINKING: true]
    [RAISON: outil X a échoué, je dois réfléchir à la correction]

    Le système basculera en thinking ON pour te donner le temps de raisonner.

    ## MODE ACTUEL
    thinking: {state_label}"""


def inject_thinking_prompt(
    effective_system: str,
    state: ThinkingState,
) -> str:
    """Append thinking-mode instructions to the effective system prompt.

    NOTE: This is called on the api_messages copy, not on the cached
    system prompt. The prompt cache remains byte-stable across turns.
    """
    if not state.enabled:
        return effective_system

    return effective_system + build_thinking_prompt(state)


def detect_escalation(content: str, state: ThinkingState) -> bool:
    """Check if the model response contains an escalation request.

    Returns True only if:
    - Thinking state is enabled
    - Model is NOT already in thinking ON mode
    - Escalation count hasn't reached the limit
    - The marker is found in the content
    """
    if not state.enabled:
        return False
    if state.active:
        return False
    if state.escalation_count >= state.max_escalations:
        return False
    if not content:
        return False
    return state.marker in content


def escalate(state: ThinkingState) -> bool:
    """Activate thinking ON and increment the escalation counter.

    Returns True if escalation was performed, False if limit reached.
    """
    if state.escalation_count >= state.max_escalations:
        logger.debug(
            "[self-escalation] Max escalations reached (%d/%d)",
            state.escalation_count,
            state.max_escalations,
        )
        return False

    state.active = True
    state.escalation_count += 1
    logger.debug(
        "[self-escalation] Escalated to thinking ON (count: %d/%d)",
        state.escalation_count,
        state.max_escalations,
    )
    return True


def extract_escalation_reason(content: str, state: ThinkingState) -> str:
    """Extract the reason given by the model for escalation.

    Looks for [RAISON: ...] lines and returns their text (without the
    brackets). Returns empty string if no reason is found.
    
    The reason is truncated to 50 characters to avoid polluting context.
    """
    if not content:
        return ""
    for line in content.split("\n"):
        stripped = line.strip()
        if stripped.startswith("[RAISON:"):
            # [RAISON: bla bla]  or  [RAISON: bla bla
            text = stripped[len("[RAISON:"):].rstrip("]").strip()
            if text:
                return text[:50] if len(text) > 50 else text
        elif stripped.startswith("[RAISON :"):
            text = stripped[len("[RAISON :"):].rstrip("]").strip()
            if text:
                return text[:50] if len(text) > 50 else text
    return ""


def strip_escalation_marker(content: str, state: ThinkingState) -> str:
    """Remove the escalation marker and reason from the model response.

    Strips lines containing [ESCALATE_THINKING:...] and [RAISON:...]
    so the user doesn't see the internal signal.
    """
    if not content:
        return content

    lines = content.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if state.marker in stripped:
            continue
        if stripped.startswith("[RAISON:") or stripped.startswith("[RAISON :"):
            continue
        cleaned.append(line)

    result = "\n".join(cleaned).strip()
    return result if result else content  # Don't return empty if all was markers
