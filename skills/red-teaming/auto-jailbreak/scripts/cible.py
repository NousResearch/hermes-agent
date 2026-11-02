"""Cible PyRIT : le modele attaque, branchable sur n'importe quel endpoint.

Par defaut : le proxy Claude local (claude-proxy, port 8000), qui expose
l'API Anthropic Messages et execute le CLI `claude` derriere. Pas de cle a
payer, c'est l'abonnement Claude qui repond.

Pour tester TON modele local (Ollama, LM Studio, vLLM, un serveur perso),
il suffit de changer 2 variables d'environnement, sans toucher au code :

    LITELLM_ENDPOINT=http://localhost:11434   # ton serveur
    LITELLM_MODEL=ollama/mon-modele           # ton modele

Un orchestrateur externe (ex. un routeur red-team) les positionne automatiquement depuis
`/pyrit cible <endpoint> [modele]`.

Usage :
    from cible import cible_proxy
    target = cible_proxy()
"""
import os

from pyrit.prompt_target import LiteLLMChatTarget

# Proxy Claude local par defaut. Port 8000 = claude-proxy/server.js vivant.
ENDPOINT = os.environ.get("LITELLM_ENDPOINT", "http://localhost:8000/anthropic")

# Suffixe « -notools » : le proxy lance le CLI claude avec `--tools ""`.
# Sans lui, la cible aurait acces aux fichiers et agirait sur la machine.
# On veut une cible qui repond en texte, rien de plus.
# Pour un modele local, ce suffixe n'a pas d'effet et ne gene pas.
MODELE = os.environ.get("LITELLM_MODEL", "anthropic/claude-sonnet-4-5-notools")

# Cle API. Le proxy local n'authentifie pas, mais litellm exige une valeur.
# Pour un vrai fournisseur, positionne LITELLM_API_KEY.
CLE = os.environ.get("LITELLM_API_KEY", "proxy-local")

try:
    MAX_TOKENS = int(os.environ.get("LITELLM_MAX_TOKENS", "1024"))
except ValueError:
    MAX_TOKENS = 1024


def cible_proxy(**kwargs) -> LiteLLMChatTarget:
    """Cible LiteLLM pointant sur l'endpoint configure. kwargs -> constructeur."""
    parametres = {
        "model_name": MODELE,
        "endpoint": ENDPOINT,
        "api_key": CLE,
        "max_tokens": MAX_TOKENS,
    }
    parametres.update(kwargs)
    return LiteLLMChatTarget(**parametres)


# --- Adversaire : le LLM qui REDIGE l'escalade en Crescendo ----------------
# Roles distincts : la cible est TESTEE, l'adversaire ATTAQUE. Un modele
# censure (Claude) refuse de jouer l'attaquant. On peut donc lui donner un
# endpoint separe, non censure, sans changer la cible. Si les variables
# LITELLM_ADVERSE_* ne sont pas posees, l'adversaire retombe sur la cible :
# comportement identique a avant (un seul modele pour les deux roles).
ENDPOINT_ADVERSE = os.environ.get("LITELLM_ADVERSE_ENDPOINT", ENDPOINT)
MODELE_ADVERSE = os.environ.get("LITELLM_ADVERSE_MODEL", MODELE)
CLE_ADVERSE = os.environ.get("LITELLM_ADVERSE_API_KEY", CLE)


def cible_adverse(**kwargs) -> LiteLLMChatTarget:
    """LLM adversaire (redacteur d'escalade Crescendo).

    Endpoint/modele/cle propres via LITELLM_ADVERSE_*, sinon = la cible.
    """
    parametres = {
        "model_name": MODELE_ADVERSE,
        "endpoint": ENDPOINT_ADVERSE,
        "api_key": CLE_ADVERSE,
        "max_tokens": MAX_TOKENS,
    }
    parametres.update(kwargs)
    return LiteLLMChatTarget(**parametres)
