# Contribuer à Hermes Agent

Merci de contribuer à Hermes Agent ! Ce guide couvre tout ce dont vous avez besoin : mettre en place votre environnement de développement, comprendre l'architecture, décider quoi construire et faire fusionner votre PR.

---

## Priorités de contribution

Nous valorisons les contributions dans cet ordre :

1. **Corrections de bugs** — plantages, comportement incorrect, perte de données. Toujours la priorité absolue.
2. **Compatibilité multiplateforme** — macOS, différentes distributions Linux et WSL2 sous Windows. Nous voulons qu'Hermes fonctionne partout.
3. **Durcissement de la sécurité** — injection shell, injection de prompt, traversée de chemins, élévation de privilèges. Voir [Sécurité](#considérations-de-sécurité).
4. **Performance et robustesse** — logique de nouvelle tentative, gestion des erreurs, dégradation contrôlée.
5. **Nouvelles compétences** — mais uniquement celles largement utiles. Voir [Compétence ou outil ?](#compétence-ou-outil-)
6. **Nouveaux outils** — rarement nécessaires. La plupart des capacités devraient être des compétences. Voir plus bas.
7. **Documentation** — corrections, clarifications, nouveaux exemples.

---

## Avant de commencer : cherchez d'abord

Une recherche rapide avant de vous lancer vous fait gagner du temps et garde la file des PR propre — les doublons sont fréquents ici, alors une minute en amont vaut le coup.

- **Cherchez dans les PR et issues ouvertes *et* fusionnées** votre sujet ou le symptôme de votre erreur — la vérification de doublons du template de PR n'intervient qu'au moment de la revue, une fois le travail déjà fait :
  ```bash
  gh search issues --repo NousResearch/hermes-agent "<your terms>"
  gh search prs --repo NousResearch/hermes-agent --state all "<your terms>"
  ```
  Ou passez par l'interface web : [issues](https://github.com/NousResearch/hermes-agent/issues?q=) · [PRs (tous états)](https://github.com/NousResearch/hermes-agent/pulls?q=is%3Apr).
- **Le suivi des issues peut être en retard sur le code.** Beaucoup de fonctionnalités demandées sont déjà implémentées dans l'arborescence ; cherchez donc aussi la capacité dans le code source (`search_files`, ou le grep de votre éditeur) avant de la proposer.
- **Si une PR ouverte traite déjà le sujet**, envisagez de la relire ou de l'améliorer plutôt que d'ouvrir un doublon concurrent.
- **Pour les travaux d'envergure**, commentez l'issue pour signaler que vous vous en occupez, afin que personne d'autre ne démarre la même chose.

En lien : #38284 couvre le pendant côté agent — Hermes lui-même vérifiant les issues et PR existantes avant de se lancer dans un auto-dépannage approfondi. Cette section en est le complément pour les contributeurs humains.

---

## Compétence ou outil ?

C'est la question la plus fréquente chez les nouveaux contributeurs. La réponse est presque toujours **compétence**.

### Faites-en une compétence quand :

- La capacité peut s'exprimer comme des instructions + des commandes shell + des outils existants
- Elle enveloppe une CLI externe ou une API que l'agent peut appeler via `terminal` ou `web_extract`
- Elle n'a pas besoin d'intégration Python sur mesure ni de gestion de clés API embarquée dans l'agent
- Exemples : recherche arXiv, workflows git, gestion de Docker, traitement de PDF, e-mail via des outils CLI

### Faites-en un outil quand :

- Il exige une intégration de bout en bout avec clés API, flux d'authentification ou configuration multi-composants gérée par le harnais de l'agent
- Il nécessite une logique de traitement sur mesure qui doit s'exécuter avec précision à chaque fois (pas du « au mieux » issu de l'interprétation du LLM)
- Il manipule des données binaires, du streaming ou des événements temps réel qui ne peuvent pas passer par le terminal
- Exemples : automatisation de navigateur (gestion de sessions Browserbase), TTS (encodage audio + livraison sur la plateforme), analyse de vision (manipulation d'images en base64)

### La compétence doit-elle être embarquée ?

Les compétences embarquées (dans `skills/`) sont livrées avec chaque installation d'Hermes. Elles doivent être **largement utiles à la majorité des utilisateurs** :

- Traitement de documents, recherche web, workflows de développement courants, administration système
- Utilisées régulièrement par un large éventail de personnes

Si votre compétence est officielle et utile mais pas universellement nécessaire (par exemple l'intégration d'un service payant, une dépendance lourde), placez-la dans **`optional-skills/`** — elle est livrée avec le dépôt mais n'est pas activée par défaut. Les utilisateurs peuvent la découvrir via `hermes skills browse` (étiquetée « official ») et l'installer avec `hermes skills install` (sans avertissement tiers, confiance intégrée).

Si votre compétence est spécialisée, issue de la communauté ou de niche, elle a plus sa place sur un **Skills Hub** — téléversez-la sur un registre de compétences et partagez-la sur le [Discord de Nous Research](https://discord.gg/NousResearch). Les utilisateurs peuvent l'installer avec `hermes skills install`.

---

## Fournisseurs de mémoire : à publier comme plugin autonome

**Nous n'acceptons plus de nouveaux fournisseurs de mémoire dans ce dépôt.** L'ensemble des fournisseurs intégrés sous `plugins/memory/` (honcho, mem0, supermemory, byterover, hindsight, holographic, openviking, retaindb) est clos. Si vous voulez ajouter un nouveau backend de mémoire, publiez-le comme **dépôt de plugin autonome** que les utilisateurs installent dans `~/.hermes/plugins/` (ou via un entry point pip).

Les plugins de mémoire autonomes :

- Implémentent la même ABC `MemoryProvider` (`agent/memory_provider.py`) — `sync_turn`, `prefetch`, `shutdown`, et éventuellement `post_setup(hermes_home, config)` pour l'intégration avec l'assistant de configuration
- Utilisent le même système de découverte — `discover_memory_providers()` les récupère dans les répertoires de plugins utilisateur/projet et les entry points pip
- S'intègrent à `hermes memory setup` via `post_setup()` — sans toucher au code du cœur
- Peuvent enregistrer leurs propres sous-commandes CLI via `register_cli(subparser)` dans un fichier `cli.py`
- Bénéficient des mêmes hooks de cycle de vie et de la même plomberie de configuration que les fournisseurs intégrés

Les PR qui ajoutent un nouveau répertoire sous `plugins/memory/` seront fermées avec un renvoi vers la publication du fournisseur dans son propre dépôt. Les fournisseurs déjà intégrés restent ; les corrections de bugs les concernant sont les bienvenues.

Ce n'est pas une question de niveau de qualité — c'est une décision de couplage et de maintenance. Les fournisseurs de mémoire sont le type de plugin le plus courant et ils n'ont pas tous vocation à vivre dans cette arborescence.

---

## Intégrations de produits tiers : à publier comme plugin autonome

La même règle s'étend à **tout plugin qui intègre le produit ou le projet de quelqu'un d'autre** — backends d'observabilité/métriques, connecteurs SaaS d'éditeurs, tableaux de bord d'analytique, intégrations de services payants et autres intégrations tierces similaires. **Elles n'entrent pas dans ce dépôt.**

La raison est la charge de maintenance, pas la qualité. Chaque produit externe absorbé dans le cœur devient notre responsabilité : le maintenir en état de marche face à une base de code qui évolue vite, pour un backend que nous ne possédons pas et ne contrôlons pas. Hermes publie beaucoup et le cœur avance rapidement ; y coupler des produits tiers crée une charge sans fin pour les mainteneurs.

Publiez-les plutôt comme **dépôt de plugin autonome** :

- Implémentez l'ABC concernée et utilisez le chemin de découverte de plugins existant (`~/.hermes/plugins/`, `.hermes/plugins/` du projet, ou un entry point pip) — voir [Build a Hermes Plugin](https://hermes-agent.nousresearch.com/docs/guides/build-a-hermes-plugin)
- Enregistrez les hooks de cycle de vie (`pre_tool_call`, `post_tool_call`, `pre_llm_call`, `post_llm_call`, `on_session_start`, `on_session_end`), les outils (`ctx.register_tool`) et les sous-commandes CLI (`ctx.register_cli_command`) à travers la surface que nous exposons déjà — aucun changement du cœur nécessaire
- Si votre plugin a besoin d'une capacité que le framework n'expose pas, c'est une demande de fonctionnalité pour **élargir la surface générique des plugins** (un nouveau hook ou une méthode `ctx`) — jamais un cas particulier pour votre plugin dans le cœur
- Faites-en la promotion sur le canal `#plugins-skills-and-skins` du [Discord de Nous Research](https://discord.gg/NousResearch) pour que les utilisateurs puissent le trouver et l'installer

Un plugin de produit tiers bien construit peut passer la revue automatisée et être fermé quand même pour cette raison — c'est une décision de placement, pas un verdict sur le code. Les PR qui ajoutent un tel répertoire sous `plugins/` seront fermées avec un renvoi vers la publication dans un dépôt dédié.

---

## Mise en place de l'environnement de développement

### Prérequis

| Prérequis | Notes |
|-------------|-------|
| **Git** | Avec l'extension `git-lfs` installée |
| **Python 3.11–3.13** | uv l'installera s'il manque |
| **uv** | Gestionnaire de paquets Python rapide ([installation](https://docs.astral.sh/uv/)) |
| **Node.js 20+** | Optionnel — nécessaire pour les outils navigateur et le pont WhatsApp (correspond aux engines du `package.json` racine) |

### Installation avec l'installeur standard

Pour la plupart des contributeurs, le meilleur amorçage de développement est le même
chemin que celui des utilisateurs : lancer l'installeur standard, puis travailler dans
le dépôt qu'il a cloné. L'installeur crée le venv d'Hermes, câble la commande
`hermes`, enregistre la méthode d'installation pour `hermes update` et clone le
projet git complet dans `$HERMES_HOME/hermes-agent` (généralement
`~/.hermes/hermes-agent`). Votre environnement de développement reste ainsi sur la
même disposition que celle attendue par la CLI, l'outil de mise à jour, l'installeur
paresseux de dépendances, le gateway et la documentation.

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"

# Add dev/test extras on top of the standard install.
uv pip install -e ".[all,dev]"

# Optional: browser tools / docs site dependencies.
npm install
```

Ensuite, créez vos branches et lancez les tests depuis ce checkout :

```bash
git checkout -b fix/description
scripts/run_tests.sh
```

### Repli : clonage manuel

À n'utiliser que si vous ne voulez délibérément pas de la disposition d'installation
gérée par Hermes (par exemple un clone jetable dans un conteneur ou un job CI). Si
vous installez de cette manière, veillez à lancer le point d'entrée `hermes` depuis
ce venv ; exécuter le `python3 -m hermes_cli.main` du système peut ramasser des
paquets Python système sans rapport.

Créez le venv **en dehors** de l'arborescence source clonée. Un venv qui vit dans le
répertoire depuis lequel l'agent opère peut être effacé par une commande à chemin
relatif que l'agent exécute contre son propre checkout (`rm -rf venv`,
`uv venv venv`, etc.), ce qui détruit silencieusement le runtime en cours en pleine
session. Le garder hors de l'arborescence garantit qu'aucun chemin relatif depuis
l'espace de travail ne le résout.

```bash
git clone https://github.com/NousResearch/hermes-agent.git
cd hermes-agent

# Create venv with Python 3.11, OUTSIDE the source tree
uv venv ~/.hermes/venvs/hermes-dev --python 3.11
export VIRTUAL_ENV="$HOME/.hermes/venvs/hermes-dev"
export PATH="$VIRTUAL_ENV/bin:$PATH"

# Install with all extras (messaging, cron, CLI menus, dev tools)
uv pip install -e ".[all,dev]"

# Optional: browser tools
npm install
```

### Configurer pour le développement

```bash
mkdir -p ~/.hermes/{cron,sessions,logs,memories,skills}
cp cli-config.yaml.example ~/.hermes/config.yaml
touch ~/.hermes/.env

# Add at minimum an LLM provider key:
echo "OPENROUTER_API_KEY=***" >> ~/.hermes/.env
```

### Lancer

```bash
# The standard installer already put `hermes` on PATH.
hermes doctor
hermes chat -q "Hello"
```

Si vous avez utilisé le repli par clonage manuel, lancez `./hermes` depuis le
checkout ou créez explicitement un lien symbolique vers le venv de ce clone :

```bash
mkdir -p ~/.local/bin
ln -sf "$(pwd)/venv/bin/hermes" ~/.local/bin/hermes
```

### Lancer les tests

```bash
# Preferred — matches CI (hermetic env, 4 xdist workers); see AGENTS.md
scripts/run_tests.sh

# Alternative (activate the venv first). The wrapper is still recommended
# for parity with GitHub Actions before you open a PR:
pytest tests/ -v
```

---

## Structure du projet

```
hermes-agent/
├── run_agent.py              # AIAgent class — core conversation loop, tool dispatch, session persistence
├── cli.py                    # HermesCLI class — interactive TUI, prompt_toolkit integration
├── model_tools.py            # Tool orchestration (thin layer over tools/registry.py)
├── toolsets.py               # Tool groupings and presets (hermes-cli, hermes-telegram, etc.)
├── hermes_state.py           # SQLite session database with FTS5 full-text search, session titles
├── batch_runner.py           # Parallel batch processing for trajectory generation
│
├── agent/                    # Agent internals (extracted modules)
│   ├── prompt_builder.py         # System prompt assembly (identity, skills, context files, memory)
│   ├── context_compressor.py     # Auto-summarization when approaching context limits
│   ├── auxiliary_client.py       # Resolves auxiliary OpenAI clients (summarization, vision)
│   ├── display.py                # KawaiiSpinner, tool progress formatting
│   ├── model_metadata.py         # Model context lengths, token estimation
│   └── trajectory.py             # Trajectory saving helpers
│
├── hermes_cli/               # CLI command implementations
│   ├── main.py                   # Entry point, argument parsing, command dispatch
│   ├── config.py                 # Config management, migration, env var definitions
│   ├── setup.py                  # Interactive setup wizard
│   ├── auth.py                   # Provider resolution, OAuth, Nous Portal
│   ├── models.py                 # OpenRouter model selection lists
│   ├── banner.py                 # Welcome banner, ASCII art
│   ├── commands.py               # Central slash command registry (CommandDef), autocomplete, gateway helpers
│   ├── callbacks.py              # Interactive callbacks (clarify, sudo, approval)
│   ├── doctor.py                 # Diagnostics
│   ├── skills_hub.py             # Skills Hub CLI + /skills slash command
│   └── skin_engine.py            # Skin/theme engine — data-driven CLI visual customization
│
├── tools/                    # Tool implementations (self-registering)
│   ├── registry.py               # Central tool registry (schemas, handlers, dispatch)
│   ├── approval.py               # Dangerous command detection + per-session approval
│   ├── terminal_tool.py          # Terminal orchestration (sudo, env lifecycle, backends)
│   ├── file_operations.py        # read_file, write_file, search, patch, etc.
│   ├── web_tools.py              # web_search, web_extract (Parallel/Firecrawl + Gemini summarization)
│   ├── vision_tools.py           # Image analysis via multimodal models
│   ├── delegate_tool.py          # Subagent spawning and parallel task execution
│   ├── code_execution_tool.py    # Sandboxed Python with RPC tool access
│   ├── session_search_tool.py    # Search past conversations with FTS5 + anchored windows
│   ├── cronjob_tools.py          # Scheduled task management
│   ├── skill_tools.py            # Skill search, load, manage
│   └── environments/             # Terminal execution backends
│       ├── base.py                   # BaseEnvironment ABC
│       ├── local.py, docker.py, ssh.py, singularity.py, modal.py, daytona.py
│
├── gateway/                  # Messaging gateway
│   ├── run.py                    # GatewayRunner — platform lifecycle, message routing, cron
│   ├── config.py                 # Platform configuration resolution
│   ├── session.py                # Session store, context prompts, reset policies
│   └── platforms/                # Platform adapters
│       ├── telegram.py, discord_adapter.py, slack.py, whatsapp.py
│
├── scripts/                  # Installer and bridge scripts
│   ├── install.sh                # Linux/macOS installer
│   ├── install.ps1               # Windows PowerShell installer
│   └── whatsapp-bridge/          # Node.js WhatsApp bridge (Baileys)
│
├── skills/                   # Bundled skills (copied to ~/.hermes/skills/ on install)
├── optional-skills/          # Official optional skills (discoverable via hub, not activated by default)
├── tests/                    # Test suite
├── website/                  # Documentation site (hermes-agent.nousresearch.com)
│
├── cli-config.yaml.example   # Example configuration (copied to ~/.hermes/config.yaml)
└── AGENTS.md                 # Development guide for AI coding assistants
```

### Configuration utilisateur (stockée dans `~/.hermes/`)

| Chemin | Rôle |
|------|---------|
| `~/.hermes/config.yaml` | Réglages (modèle, terminal, toolsets, compression, etc.) |
| `~/.hermes/.env` | Clés API et secrets |
| `~/.hermes/auth.json` | Identifiants OAuth (Nous Portal) |
| `~/.hermes/skills/` | Toutes les compétences actives (embarquées + installées depuis le hub + créées par l'agent) |
| `~/.hermes/memories/` | Mémoire persistante (MEMORY.md, USER.md) |
| `~/.hermes/state.db` | Base de données de sessions SQLite |
| `~/.hermes/sessions/` | Index de routage du gateway (`sessions.json`), traces de request-dump, transcriptions `*.jsonl` du gateway et (en option) instantanés JSON par session quand `sessions.write_json_snapshots: true` est défini. Les instantanés par session sont désactivés par défaut ; state.db fait foi. |
| `~/.hermes/cron/` | Données des tâches planifiées |
| `~/.hermes/whatsapp/session/` | Identifiants du pont WhatsApp |

---

## Vue d'ensemble de l'architecture

### Boucle principale

```
User message → AIAgent._run_agent_loop()
  ├── Build system prompt (prompt_builder.py)
  ├── Build API kwargs (model, messages, tools, reasoning config)
  ├── Call LLM (OpenAI-compatible API)
  ├── If tool_calls in response:
  │     ├── Execute each tool via registry dispatch
  │     ├── Add tool results to conversation
  │     └── Loop back to LLM call
  ├── If text response:
  │     ├── Persist session to DB
  │     └── Return final_response
  └── Context compression if approaching token limit
```

### Patrons de conception clés

- **Outils auto-enregistrés** : chaque fichier d'outil appelle `registry.register()` à l'import. `model_tools.py` déclenche la découverte en important tous les modules d'outils.
- **Regroupement en toolsets** : les outils sont regroupés en toolsets (`web`, `terminal`, `file`, `browser`, etc.) qui peuvent être activés/désactivés par plateforme.
- **Persistance des sessions** : toutes les conversations sont stockées dans SQLite (`hermes_state.py`) avec recherche plein texte et titres de session uniques. Les instantanés JSON par session dans `~/.hermes/sessions/` ont été remplacés par le stockage SQLite et sont désactivés par défaut ; réactivez-les avec `sessions.write_json_snapshots: true` si un outillage externe consomme directement les fichiers JSON.
- **Injection éphémère** : les prompts système et les messages de préremplissage sont injectés au moment de l'appel API, jamais persistés dans la base de données ni dans les logs.
- **Abstraction des fournisseurs** : l'agent fonctionne avec n'importe quelle API compatible OpenAI. La résolution du fournisseur a lieu à l'initialisation (OAuth Nous Portal, clé API OpenRouter ou endpoint personnalisé).
- **Routage des fournisseurs** : avec OpenRouter, `provider_routing` dans config.yaml contrôle la sélection du fournisseur (tri par débit/latence/prix, autorisation/exclusion de fournisseurs spécifiques, politiques de rétention des données). Ces réglages sont injectés dans `extra_body.provider` des requêtes API.

---

## Style de code

- **PEP 8** avec des exceptions pragmatiques (nous n'imposons pas de longueur de ligne stricte)
- **Commentaires** : uniquement pour expliquer une intention non évidente, des compromis ou des bizarreries d'API. Ne racontez pas ce que fait le code — `# increment counter` n'apporte rien
- **Gestion des erreurs** : attrapez des exceptions spécifiques. Loguez avec `logger.warning()`/`logger.error()` — utilisez `exc_info=True` pour les erreurs inattendues afin que les stack traces apparaissent dans les logs
- **Multiplateforme** : ne supposez jamais Unix. Voir [Compatibilité multiplateforme](#compatibilité-multiplateforme)

---

## Ajouter un nouvel outil

Avant d'écrire un outil, demandez-vous : [ne devrait-ce pas être une compétence ?](#compétence-ou-outil-)

Les outils s'enregistrent eux-mêmes auprès du registre central. Chaque fichier d'outil regroupe au même endroit son schéma, son handler et son enregistrement :

```python
"""my_tool — Brief description of what this tool does."""

import json
from tools.registry import registry


def my_tool(param1: str, param2: int = 10, **kwargs) -> str:
    """Handler. Returns a string result (often JSON)."""
    result = do_work(param1, param2)
    return json.dumps(result)


MY_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "What this tool does and when the agent should use it.",
        "parameters": {
            "type": "object",
            "properties": {
                "param1": {"type": "string", "description": "What param1 is"},
                "param2": {"type": "integer", "description": "What param2 is", "default": 10},
            },
            "required": ["param1"],
        },
    },
}


def _check_requirements() -> bool:
    """Return True if this tool's dependencies are available."""
    return True


registry.register(
    name="my_tool",
    toolset="my_toolset",
    schema=MY_TOOL_SCHEMA,
    handler=lambda args, **kw: my_tool(**args, **kw),
    check_fn=_check_requirements,
)
```

**Câblage dans un toolset (obligatoire) :** les outils intégrés sont découverts
automatiquement : tout fichier `tools/*.py` contenant un appel
`registry.register(...)` au niveau supérieur est importé par
`discover_builtin_tools()` dans `tools/registry.py` au chargement de `model_tools`.
Il n'y a **aucune** liste d'imports manuelle à maintenir dans `model_tools.py`.

Vous devez néanmoins ajouter le nom de l'outil à la liste appropriée dans
`toolsets.py` (par exemple `_HERMES_CORE_TOOLS` ou un toolset dédié) ; sinon
l'outil s'enregistre mais n'est jamais exposé à l'agent. Si vous introduisez un
nouveau toolset, ajoutez-le dans `toolsets.py` et câblez-le dans les presets de
plateforme concernés.

Voir `AGENTS.md` (section **Adding New Tools**) pour les chemins sensibles aux
profils et l'arbitrage plugin vs cœur.

---

## Ajouter une compétence

Les compétences embarquées vivent dans `skills/`, organisées par catégorie. Les compétences optionnelles officielles utilisent la même structure dans `optional-skills/` :

```
skills/
├── research/
│   └── arxiv/
│       ├── SKILL.md              # Required: main instructions
│       └── scripts/              # Optional: helper scripts
│           └── search_arxiv.py
├── productivity/
│   └── ocr-and-documents/
│       ├── SKILL.md
│       ├── scripts/
│       └── references/
└── ...
```

### Format de SKILL.md

```markdown
---
name: my-skill
description: Brief description (shown in skill search results)
version: 1.0.0
author: Your Name
license: MIT
platforms: [macos, linux]          # Optional — restrict to specific OS platforms
                                   #   Valid: macos, linux, windows
                                   #   Omit to load on all platforms (default)
required_environment_variables:    # Optional — secure setup-on-load metadata
  - name: MY_API_KEY
    prompt: API key
    help: Where to get it
    required_for: full functionality
prerequisites:                     # Optional legacy runtime requirements
  env_vars: [MY_API_KEY]           #   Backward-compatible alias for required env vars
  commands: [curl, jq]             #   Advisory only; does not hide the skill
metadata:
  hermes:
    tags: [Category, Subcategory, Keywords]
    related_skills: [other-skill-name]
    fallback_for_toolsets: [web]       # Optional — show only when toolset is unavailable
    requires_toolsets: [terminal]      # Optional — show only when toolset is available
---

# Skill Title

Brief intro.

## When to Use
Trigger conditions — when should the agent load this skill?

## Prerequisites
Env vars, install steps, MCP setup, API key sourcing.

## How to Run
Canonical invocation through the `terminal` tool.

## Quick Reference
Table of common commands or API calls.

## Procedure
Step-by-step instructions the agent follows.

## Pitfalls
Known failure modes and how to handle them.

## Verification
How the agent confirms it worked.
```

### Compétences spécifiques à une plateforme

Les compétences peuvent déclarer les plateformes qu'elles prennent en charge via le champ de frontmatter `platforms`. Les compétences portant ce champ sont automatiquement masquées du prompt système, de `skills_list()` et des commandes slash sur les plateformes incompatibles.

```yaml
platforms: [macos]            # macOS only (e.g., iMessage, Apple Reminders)
platforms: [macos, linux]     # macOS and Linux
platforms: [windows]          # Windows only
```

Si le champ est omis ou vide, la compétence se charge sur toutes les plateformes (rétrocompatible). Voir `skills/apple/` pour des exemples de compétences réservées à macOS.

### Activation conditionnelle des compétences

Les compétences peuvent déclarer des conditions qui contrôlent leur apparition dans le prompt système, selon les outils et toolsets disponibles dans la session courante. C'est principalement utilisé pour les **compétences de repli** — des alternatives qui ne doivent apparaître que lorsqu'un outil principal est indisponible.

Quatre champs sont pris en charge sous `metadata.hermes` :

```yaml
metadata:
  hermes:
    fallback_for_toolsets: [web]      # Show ONLY when these toolsets are unavailable
    requires_toolsets: [terminal]     # Show ONLY when these toolsets are available
    fallback_for_tools: [web_search]  # Show ONLY when these specific tools are unavailable
    requires_tools: [terminal]        # Show ONLY when these specific tools are available
```

**Sémantique :**
- `fallback_for_*` : la compétence est une solution de secours. Elle est **masquée** quand les outils/toolsets listés sont disponibles, et **affichée** quand ils ne le sont pas. À utiliser pour des alternatives gratuites à des outils premium.
- `requires_*` : la compétence a besoin de certains outils pour fonctionner. Elle est **masquée** quand les outils/toolsets listés sont indisponibles. À utiliser pour des compétences qui dépendent de capacités précises (par exemple une compétence qui n'a de sens qu'avec un accès au terminal).
- Si les deux sont spécifiés, les deux conditions doivent être satisfaites pour que la compétence apparaisse.
- Si aucun n'est spécifié, la compétence est toujours affichée (rétrocompatible).

**Exemples :**

```yaml
# DuckDuckGo search — shown when Firecrawl (web toolset) is unavailable
metadata:
  hermes:
    fallback_for_toolsets: [web]

# Smart home skill — only useful when terminal is available
metadata:
  hermes:
    requires_toolsets: [terminal]

# Local browser fallback — shown when Browserbase is unavailable
metadata:
  hermes:
    fallback_for_toolsets: [browser]
```

Le filtrage a lieu au moment de la construction du prompt, dans `agent/prompt_builder.py`. La fonction `build_skills_system_prompt()` reçoit l'ensemble des outils et toolsets disponibles depuis l'agent et utilise `_skill_should_show()` pour évaluer les conditions de chaque compétence.

### Métadonnées de configuration d'une compétence

Les compétences peuvent déclarer des métadonnées de configuration sécurisée au chargement via le champ de frontmatter `required_environment_variables`. Des valeurs manquantes ne masquent pas la compétence à la découverte ; elles déclenchent une invite sécurisée (CLI uniquement) au moment où la compétence est réellement chargée.

```yaml
required_environment_variables:
  - name: TENOR_API_KEY
    prompt: Tenor API key
    help: Get a key from https://developers.google.com/tenor
    required_for: full functionality
```

L'utilisateur peut ignorer la configuration et continuer à charger la compétence. Hermes n'expose au modèle que des métadonnées (`stored_as`, `skipped`, `validated`) — jamais la valeur du secret.

L'ancien `prerequisites.env_vars` reste pris en charge et est normalisé vers la nouvelle représentation.

```yaml
prerequisites:
  env_vars: [TENOR_API_KEY]       # Legacy alias for required_environment_variables
  commands: [curl, jq]            # Advisory CLI checks
```

Les sessions gateway et de messagerie ne collectent jamais de secrets dans le flux de conversation ; elles demandent à l'utilisateur de lancer `hermes setup` ou de mettre à jour `~/.hermes/.env` en local.

**Quand déclarer des variables d'environnement requises :**
- La compétence utilise une clé API ou un token qui doit être collecté de manière sécurisée au chargement
- La compétence peut rester utile si l'utilisateur ignore la configuration, quitte à se dégrader proprement

**Quand déclarer des prérequis de commandes :**
- La compétence repose sur un outil CLI qui peut ne pas être installé (par exemple `himalaya`, `openhue`, `ddgs`)
- Traitez les vérifications de commandes comme des indications, pas comme un masquage à la découverte

Voir `skills/gifs/gif-search/` et `skills/email/himalaya/` pour des exemples.

### Normes de rédaction des compétences (NON NÉGOCIABLES)

Toute compétence nouvelle ou modernisée — embarquée, optionnelle ou contribuée — doit respecter ces normes avant la fusion. Les relecteurs rejettent les PR qui les enfreignent.

1. **`description` ≤ 60 caractères, une seule phrase, terminée par un point.** Les descriptions longues alourdissent l'interface de listage des compétences et diluent l'attention du modèle quand de nombreuses compétences sont chargées. Énoncez la capacité, pas l'implémentation. Pas de vocabulaire marketing (« powerful », « comprehensive », « seamless », « advanced »). Ne répétez pas le nom de la compétence. Vérifiez avec :
   ```python
   import re, pathlib
   m = re.search(r'^description: (.*)$',
                 pathlib.Path('skills/<cat>/<name>/SKILL.md').read_text(),
                 re.MULTILINE)
   assert len(m.group(1)) <= 60, len(m.group(1))
   ```

   Bien : `Search arXiv papers by keyword, author, category, or ID.`
   Mal : `A powerful and comprehensive skill that allows the agent to search arXiv for relevant academic papers using various criteria including keywords, authors, and categories.`

2. **Les outils cités dans la prose de SKILL.md doivent être des outils Hermes natifs ou des serveurs MCP que la compétence attend explicitement.** Quand la compétence a besoin d'une capacité, désignez le bon outil par son nom entre backticks : `` `terminal` ``, `` `web_extract` ``, `` `web_search` ``, `` `read_file` ``, `` `write_file` ``, `` `patch` ``, `` `search_files` ``, `` `vision_analyze` ``, `` `browser_navigate` ``, `` `delegate_task` ``, `` `image_generate` ``, `` `text_to_speech` ``, `` `cronjob` ``, `` `memory` ``, `` `skill_view` ``, `` `todo` ``, `` `execute_code` ``.

   Ne nommez PAS des utilitaires shell que l'agent a déjà enveloppés :

   | À ne pas dire | À dire |
   |---|---|
   | `grep`, `rg` | `search_files` |
   | `cat`, `head`, `tail` | `read_file` |
   | `sed`, `awk` | `patch` |
   | `find`, `ls` | `search_files` (avec `target='files'`) |
   | `curl` pour extraire du contenu | `web_extract` |
   | `echo > file`, `cat <<EOF` | `write_file` |

   Si la compétence dépend d'un serveur MCP, nommez-le et documentez son installation dans `## Prerequisites`. Les CLI tierces (par exemple `ffmpeg`, `gh`, un SDK particulier) peuvent être invoquées depuis les fichiers de script, mais la prose doit présenter l'interaction comme « invoquer via l'outil `terminal` », pas comme une session shell manuelle.

3. **Le gating `platforms:` doit être audité contre les imports réels des scripts.** Les compétences qui utilisent des primitives exclusivement POSIX (`fcntl`, `termios`, `os.setsid`, `os.kill(pid, 0)` pour tester la vivacité, `/proc`, chemins `/tmp` codés en dur, `signal.SIGKILL`, heredocs bash, `osascript`, `apt`, `systemctl`) doivent déclarer leurs plateformes prises en charge via le frontmatter `platforms:`. La posture par défaut est de d'abord rendre le code multiplateforme — `tempfile.gettempdir()`, `pathlib.Path`, `psutil.pid_exists()`, filtrage au niveau Python plutôt que `grep`. Ne restreignez à un ensemble plus étroit que si la dépendance est réellement liée à une plateforme (par exemple `osascript` est propre à macOS, `/proc` à Linux).

4. **`author` crédite d'abord le contributeur humain.** Pour les contributions externes, le vrai nom du contributeur + son pseudo GitHub viennent en premier (`Jane Doe (jane-doe)`) ; « Hermes Agent » est le collaborateur secondaire. Si le commit du contributeur affiche « Hermes Agent » comme auteur parce qu'il a utilisé Hermes pour rédiger la compétence, remplacez-le par son nom réel — on crédite l'humain, pas l'outil.

5. **Le corps de SKILL.md suit l'ordre de sections moderne.** Titre `# <Skill> Skill`, introduction de 2-3 phrases indiquant ce que la compétence fait et ne fait pas, puis :
   - `## When to Use` — conditions de déclenchement
   - `## Prerequisites` — variables d'environnement, étapes d'installation, configuration MCP, obtention des clés API
   - `## How to Run` — invocation canonique via l'outil `terminal`
   - `## Quick Reference` — référence brute des commandes/API
   - `## Procedure` — étapes numérotées avec commandes prêtes à copier-coller
   - `## Pitfalls` — limites connues, quotas, choses qui semblent cassées mais ne le sont pas
   - `## Verification` — une seule commande qui prouve que la compétence fonctionne

   Visez ~200 lignes pour une compétence complexe, ~100 pour une simple. Supprimez les introductions redondantes, la prose marketing et les ré-explications de variables d'environnement déjà documentées dans `## Prerequisites`.

6. **Les scripts vont dans `scripts/`, les références dans `references/`, les templates dans `templates/`.** N'attendez pas du modèle qu'il réécrive inline des parseurs, des parcours XML ou de la logique non triviale à chaque appel — livrez un script utilitaire. Référencez les scripts depuis SKILL.md par chemin relatif au répertoire de la compétence.

7. **Les tests vivent dans `tests/skills/test_<skill>_skill.py`** et n'utilisent que la stdlib + pytest + `unittest.mock`. Aucun appel réseau réel. Lancez-les via `scripts/run_tests.sh tests/skills/test_<skill>_skill.py -q`. Ils doivent passer dans l'environnement CI hermétique (aucune clé API qui fuite). Utilisez `monkeypatch` et `tmp_path` pour toute dépendance aux variables d'environnement ou au système de fichiers.

8. **Les ajouts à `.env.example` sont isolés dans un bloc clairement délimité.** Ne touchez pas au reste du fichier — les versions de `.env.example` fournies par les contributeurs sont généralement obsolètes, et les modifications hors du bloc propre à la compétence seront écartées lors de la récupération. Commentez toutes les valeurs avec `#` (c'est de la documentation, pas de la configuration active).

### Recommandations pour les compétences

- **Pas de dépendances externes sauf nécessité absolue.** Préférez la stdlib Python, curl et les outils Hermes existants (`web_extract`, `terminal`, `read_file`).
- **Divulgation progressive.** Placez le workflow le plus courant en premier. Les cas limites et l'usage avancé vont en bas.
- **Fournissez des scripts utilitaires** pour le parsing XML/JSON ou la logique complexe — n'attendez pas du LLM qu'il écrive des parseurs inline à chaque fois.
- **Testez-la.** Lancez `hermes --toolsets skills -q "Use the X skill to do Y"` et vérifiez que l'agent suit correctement les instructions.

---

## Ajouter un skin / thème

Hermes utilise un système de skins piloté par les données — aucun changement de code n'est nécessaire pour ajouter un nouveau skin.

**Option A : skin utilisateur (fichier YAML)**

Créez `~/.hermes/skins/<name>.yaml` :

```yaml
name: mytheme
description: Short description of the theme

colors:
  banner_border: "#HEX"     # Panel border color
  banner_title: "#HEX"      # Panel title color
  banner_accent: "#HEX"     # Section header color
  banner_dim: "#HEX"        # Muted/dim text color
  banner_text: "#HEX"       # Body text color
  response_border: "#HEX"   # Response box border

spinner:
  waiting_faces: ["(⚔)", "(⛨)"]
  thinking_faces: ["(⚔)", "(⌁)"]
  thinking_verbs: ["forging", "plotting"]
  wings:                     # Optional left/right decorations
    - ["⟪⚔", "⚔⟫"]

branding:
  agent_name: "My Agent"
  welcome: "Welcome message"
  response_label: " ⚔ Agent "
  prompt_symbol: "⚔"

tool_prefix: "╎"             # Tool output line prefix
```

Tous les champs sont optionnels — les valeurs manquantes héritent du skin par défaut.

**Option B : skin intégré**

Ajoutez-le au dict `_BUILTIN_SKINS` dans `hermes_cli/skin_engine.py`. Utilisez le même schéma que ci-dessus, mais sous forme de dict Python. Les skins intégrés sont livrés avec le paquet et toujours disponibles.

**Activation :**
- CLI : `/skin mytheme` ou définissez `display.skin: mytheme` dans config.yaml
- Config : `display: { skin: mytheme }`

Voir `hermes_cli/skin_engine.py` pour le schéma complet et les skins existants en exemples.

---

## Compatibilité multiplateforme

Hermes tourne sous Linux, macOS et Windows natif (plus WSL2). Quand vous écrivez du
code qui touche à l'OS, partez du principe que *n'importe quelle* plateforme peut
emprunter votre chemin de code.

> **Avant d'ouvrir votre PR :** lancez `scripts/check-windows-footguns.py` pour
> attraper les pièges Windows courants dans votre diff. C'est du grep, donc peu
> coûteux ; la CI le lance aussi sur chaque PR.

### Règles critiques

1. **N'appelez jamais `os.kill(pid, 0)` pour tester si un processus est vivant.**
   `os.kill(pid, 0)` est un idiome POSIX standard pour vérifier « ce PID est-il
   vivant » — le signal 0 est une simple vérification de permission sans effet.
   **Sous Windows, ce n'est PAS sans effet.** Le `os.kill` de Python sous Windows
   mappe `sig=0` sur `CTRL_C_EVENT` (ils entrent en collision à la valeur entière
   0) et le fait passer par `GenerateConsoleCtrlEvent(0, pid)`, qui diffuse Ctrl+C
   à **tout le groupe de processus de la console** contenant le PID cible.
   « Sonder si vivant » devient silencieusement « tuer la cible et souvent des
   processus sans rapport partageant sa console ». Voir [bpo-14484](https://bugs.python.org/issue14484)
   (ouvert depuis 2012 — ne sera jamais corrigé pour raisons de compatibilité).

   **À privilégier :** utilisez `psutil` (une dépendance de base — toujours disponible) :

   ```python
   import psutil
   if psutil.pid_exists(pid):
       # process is alive — safe on every platform
       ...
   ```

   Si vous avez spécifiquement besoin du wrapper hermes (il a un repli stdlib
   pour les imports en phase d'amorçage, avant que pip install ne se termine),
   utilisez `gateway.status._pid_exists(pid)`. Il appelle d'abord
   `psutil.pid_exists` et se rabat sur une danse maison
   `OpenProcess + WaitForSingleObject` sous Windows, uniquement si psutil est
   introuvable.

   Grep d'audit pour les nouveaux sites d'appel : `rg "os\.kill\([^,]+,\s*0\s*\)"`.
   Tout résultat hors code de test est présumé être un bug de kill silencieux sous
   Windows.

2. **Utilisez `shutil.which()` avant de lancer une commande — ne supposez pas que
   Windows a les outils de Linux.** `wmic` a été retiré à partir de Windows 10
   21H1. `ps`, `kill`, `grep`, `awk`, `fuser`, `lsof`, `pgrep` et la plupart des
   outils CLI POSIX n'existent tout simplement pas sous Windows. Testez la
   disponibilité avec `shutil.which("tool")` et prévoyez un équivalent natif
   Windows — généralement PowerShell via `subprocess.run(["powershell",
   "-NoProfile", "-Command", ...])`.

   Pour l'énumération des processus : `Get-CimInstance Win32_Process` de
   PowerShell est le remplaçant moderne de `wmic process`. Voir
   `hermes_cli/gateway.py::_scan_gateway_pids` pour le motif à suivre.

3. **`termios` et `fcntl` sont réservés à Unix.** Attrapez toujours à la fois
   `ImportError` et `NotImplementedError` :
   ```python
   try:
       from simple_term_menu import TerminalMenu
       menu = TerminalMenu(options)
       idx = menu.show()
   except (ImportError, NotImplementedError):
       # Fallback: numbered menu for Windows
       for i, opt in enumerate(options):
           print(f"  {i+1}. {opt}")
       idx = int(input("Choice: ")) - 1
   ```

4. **Encodage des fichiers.** Windows peut enregistrer les fichiers `.env` en
   `cp1252`. Gérez toujours les erreurs d'encodage :
   ```python
   try:
       load_dotenv(env_path)
   except UnicodeDecodeError:
       load_dotenv(env_path, encoding="latin-1")
   ```
   Les fichiers de configuration (`config.yaml`) peuvent être enregistrés avec un
   BOM UTF-8 par le Bloc-notes et éditeurs similaires — utilisez
   `encoding="utf-8-sig"` pour lire les fichiers qui ont pu être touchés par un
   éditeur graphique Windows.

5. **Gestion des processus.** `os.setsid()`, `os.killpg()`, `os.fork()`,
   `os.getuid()` et la gestion des signaux POSIX diffèrent sous Windows. Protégez
   avec `platform.system()`, `sys.platform` ou `hasattr(os, "setsid")` :
   ```python
   if platform.system() != "Windows":
       kwargs["preexec_fn"] = os.setsid
   else:
       kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
   ```

   **À privilégier :** pour tuer un processus ET ses enfants (ce que fait
   `os.killpg` sous POSIX), utilisez `psutil` — il fonctionne sur toutes les
   plateformes :
   ```python
   import psutil
   try:
       parent = psutil.Process(pid)
       # Kill children first (leaf-up), then the parent.
       for child in parent.children(recursive=True):
           child.kill()
       parent.kill()
   except psutil.NoSuchProcess:
       pass
   ```

6. **Signaux inexistants sous Windows : `SIGALRM`, `SIGCHLD`, `SIGHUP`,
   `SIGUSR1`, `SIGUSR2`, `SIGPIPE`, `SIGQUIT`, `SIGKILL`.** Le module `signal` de
   Python lève `AttributeError` à l'import si vous les référencez sous Windows.
   Utilisez `getattr(signal, "SIGKILL", signal.SIGTERM)` ou placez tout le bloc
   derrière une vérification de plateforme. `loop.add_signal_handler` lève
   `NotImplementedError` sous Windows — attrapez-la toujours.

7. **Séparateurs de chemins.** Utilisez `pathlib.Path` plutôt que la concaténation
   de chaînes avec `/`. Les slashs fonctionnent presque partout sous Windows, mais
   `subprocess.run(["cmd.exe", "/c", ...])` et d'autres contextes shell peuvent
   exiger des antislashs — convertissez avec `str(path)` à la frontière du
   subprocess, pas au cœur de la logique Python.

8. **Les liens symboliques exigent des privilèges élevés sous Windows** (sauf si
   le mode développeur est activé). Les tests qui créent des liens symboliques ont
   besoin de `@pytest.mark.skipif(sys.platform ==
   "win32", reason="Symlinks require elevated privileges on Windows")`.

9. **Les modes de fichiers POSIX (0o600, 0o644, etc.) ne sont PAS appliqués sur
   NTFS** par défaut. Les tests qui font des assertions sur
   `stat().st_mode & 0o777` doivent être ignorés sous Windows — le concept ne se
   transpose pas. Utilisez des ACL (`icacls`, `pywin32`) pour protéger les
   fichiers de secrets sous Windows si nécessaire.

10. **Les démons d'arrière-plan détachés sous Windows exigent `pythonw.exe`, PAS
    `python.exe`.** `python.exe` alloue toujours une console ou s'y attache, ce
    qui le rend vulnérable aux diffusions de `CTRL_C_EVENT` depuis n'importe quel
    processus frère. `pythonw.exe` est la variante sans console. Combinez avec
    `CREATE_NO_WINDOW | DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP |
    CREATE_BREAKAWAY_FROM_JOB` dans `subprocess.Popen(creationflags=...)`.
    Voir `hermes_cli/gateway_windows.py::_spawn_detached` pour l'implémentation
    de référence.

11. **`subprocess.Popen` avec des shims `.cmd` ou `.bat` a besoin de
    `shutil.which` pour la résolution.** Passer `"agent-browser"` à `Popen` sous
    Windows trouve le shim shebang POSIX sans extension dans
    `node_modules/.bin/`, que `CreateProcessW` ne peut pas exécuter — vous
    obtiendrez `WinError 193 "not a valid Win32 application"`. Utilisez
    `shutil.which("agent-browser", path=local_bin)`, qui honore PATHEXT et
    choisit la variante `.CMD` sous Windows.

12. **N'utilisez pas les shebangs shell pour lancer du Python.** `#!/usr/bin/env
    python` ne fonctionne que lorsque le fichier est exécuté à travers un shell
    Unix. `subprocess.run(["./myscript.py"])` échoue sous Windows même si le
    fichier a une ligne shebang. Invoquez toujours Python explicitement :
    `[sys.executable, "myscript.py"]`.

13. **Commandes shell dans les installeurs.** Si vous modifiez
    `scripts/install.sh`, faites la modification équivalente dans
    `scripts/install.ps1`. Ces deux scripts sont l'exemple canonique de
    « fonctionne sous Linux ne veut pas dire fonctionne sous Windows » et ont
    divergé plusieurs fois — gardez-les synchronisés.

14. **Chemins connus redirigés vers OneDrive sous Windows :** Desktop,
    Documents, Pictures, Videos. Le « vrai » chemin quand OneDrive Backup est
    activé est `%USERPROFILE%\OneDrive\Desktop` (etc.), et NON
    `%USERPROFILE%\Desktop` (qui existe en coquille vide). Résolvez
    l'emplacement réel via `ctypes` + `SHGetKnownFolderPath` ou en lisant la clé
    de registre `Shell Folders` — ne supposez jamais `~/Desktop`.

15. **CRLF vs LF dans les scripts générés.** `cmd.exe` et `schtasks` sous Windows
    parsent ligne par ligne ; des fins de ligne mixtes ou uniquement LF peuvent
    casser des fichiers `.cmd` / `.bat` multilignes. Utilisez `open(path, "w",
    encoding="utf-8", newline="\r\n")` — ou `open(path, "wb")` + octets
    explicites — pour générer des scripts que Windows exécutera.

16. **Deux schémas de quoting différents sur une même ligne de commande.**
    `subprocess.run(["schtasks", "/TR", some_cmd])` → schtasks parse lui-même
    `/TR`, ET la chaîne `some_cmd` est re-parsée par `cmd.exe` quand la tâche se
    déclenche. Parseurs différents, règles d'échappement différentes. Utilisez
    deux helpers de quoting distincts et ne les croisez jamais. Voir
    `hermes_cli/gateway_windows.py::_quote_cmd_script_arg` et
    `_quote_schtasks_arg` pour la paire de référence.

### Tester le multiplateforme

Les tests qui utilisent des appels système exclusivement POSIX ont besoin d'un marqueur de skip. Les cas courants :
- Liens symboliques → `@pytest.mark.skipif(sys.platform == "win32", ...)`
- Modes de fichiers `0o600` → `@pytest.mark.skipif(sys.platform.startswith("win"), ...)`
- `signal.SIGALRM` → Unix uniquement (voir `tests/conftest.py::_enforce_test_timeout`)
- `os.setsid` / `os.fork` → Unix uniquement
- Tests de régression Winsock réels / propres à Windows →
  `@pytest.mark.skipif(sys.platform != "win32", reason="Windows-specific regression")`

Si vous monkeypatchez `sys.platform` pour des tests multiplateformes, patchez aussi
`platform.system()` / `platform.release()` / `platform.mac_ver()` — chacun relit
indépendamment le véritable OS, si bien que des tests à moitié patchés empruntent
quand même la mauvaise branche sur un runner Windows.

---

## Considérations de sécurité

Hermes a accès au terminal. La sécurité compte.

### Protections existantes

| Couche | Implémentation |
|-------|---------------|
| **Passage du mot de passe sudo** | Utilise `shlex.quote()` pour empêcher l'injection shell |
| **Détection des commandes dangereuses** | Motifs regex dans `tools/approval.py` avec flux d'approbation utilisateur |
| **Injection de prompt via cron** | Un scanner dans `tools/cronjob_tools.py` bloque les motifs de contournement d'instructions |
| **Liste de refus en écriture** | Chemins protégés (`~/.ssh/authorized_keys`, `/etc/shadow`) résolus via `os.path.realpath()` pour empêcher le contournement par lien symbolique |
| **Garde des compétences** | Scanner de sécurité pour les compétences installées depuis le hub (`tools/skills_guard.py`) |
| **Sandbox d'exécution de code** | Le processus enfant de `execute_code` tourne avec les clés API retirées de l'environnement |
| **Durcissement des conteneurs** | Docker : toutes les capabilities supprimées, pas d'élévation de privilèges, limites de PID, tmpfs à taille limitée |

### Contribuer du code sensible côté sécurité

- **Utilisez toujours `shlex.quote()`** quand vous interpolez une saisie utilisateur dans des commandes shell
- **Résolvez les liens symboliques** avec `os.path.realpath()` avant tout contrôle d'accès fondé sur les chemins
- **Ne loguez pas de secrets.** Les clés API, tokens et mots de passe ne doivent jamais apparaître dans les logs
- **Attrapez des exceptions larges** autour de l'exécution des outils, pour qu'un seul échec ne fasse pas planter la boucle de l'agent
- **Testez sur toutes les plateformes** si votre changement touche aux chemins de fichiers, à la gestion des processus ou aux commandes shell

Si votre PR touche à la sécurité, signalez-le explicitement dans la description.

### Politique d'épinglage des dépendances (durcissement de la chaîne d'approvisionnement)

Après la [compromission de la chaîne d'approvisionnement de litellm](https://github.com/BerriAI/litellm/issues/24512) en mars 2026 et la [campagne du ver Mini Shai-Hulud](https://socket.dev/blog/tanstack-npm-packages-compromised-mini-shai-hulud-supply-chain-attack) en mai 2026, toutes les dépendances doivent suivre ces règles :

| Type de source | Traitement requis | Justification |
|---|---|---|
| **Paquet PyPI** | `>=floor,<next_major` | Les versions PyPI sont immuables une fois publiées, mais de nouvelles versions peuvent entrer dans votre plage. Un plafond `<next_major` empêche une installation en 1.x de passer à un 2.0.0 malveillant. |
| **URL Git** (atroposlib, tinker, yc-bench, Baileys) | SHA de commit complet | Les branches et tags sont des refs mutables ; un SHA est adressé par contenu. |
| **GitHub Actions** | SHA de commit complet + commentaire de version | Les tags d'actions sont des refs mutables (par exemple tj-actions/changed-files, mars 2025). Épinglez avec `uses: owner/action@<sha>  # vX.Y.Z` |
| **Installations pip réservées à la CI** | `==exact` | Builds CI hermétiques ; le churn est acceptable. |

**Toute nouvelle dépendance PyPI dans une PR doit avoir une borne supérieure `<next_major`.** Les PR ajoutant des spécifications `>=X.Y.Z` sans borne seront rejetées par les relecteurs. Le workflow CI `supply-chain-audit.yml` signale aussi les changements de manifestes de dépendances pour revue manuelle.

**Comment déterminer le plafond :**
- Si le paquet est en version `1.x.y`, utilisez `<2`.
- Si le paquet est en version `0.x.y` (pré-1.0), utilisez `<0.(current_minor + 2)` — par exemple si la version courante est `0.29.x`, utilisez `<0.32`. Cela donne ~2 versions mineures de marge tout en gardant la fenêtre assez étroite pour qu'une version issue d'une prise de contrôle hostile ait peu de chances d'y atterrir.
- Exception : les paquets aux API très stables (par exemple `aiohttp-socks`) peuvent utiliser `<1` à la discrétion du relecteur.

**Exemples :**
```toml
# ✅ Correct — post-1.0
"openai>=2.21.0,<3"
"pydantic>=2.12.5,<3"

# ✅ Correct — pre-1.0 (tight minor window)
"asyncpg>=0.29,<0.32"
"aiosqlite>=0.20,<0.23"
"hindsight-client>=0.4.22,<0.5"

# ❌ Rejected — no upper bound
"some-package>=1.2.3"

# ❌ Rejected — too tight (blocks legitimate patches)
"some-package==1.2.3"

# ❌ Rejected — too loose for pre-1.0 (allows 80 minor versions)
"some-package>=0.20,<1"
```

**PR de référence :** #2796 (retrait de litellm), #2810 (passe sur les bornes supérieures), #9801 (épinglage par SHA + CI supply-chain-audit).

---

## Processus de pull request

### Nommage des branches

```
fix/description        # Bug fixes
feat/description       # New features
docs/description       # Documentation
test/description       # Tests
refactor/description   # Code restructuring
```

### Avant de soumettre

1. **Lancez les tests** : `scripts/run_tests.sh` (recommandé ; identique à la CI) ou `pytest tests/ -v` avec le venv du projet activé
2. **Testez manuellement** : lancez `hermes` et exercez le chemin de code que vous avez modifié
3. **Vérifiez l'impact multiplateforme** : si vous touchez aux E/S de fichiers, à la gestion des processus ou au terminal, pensez à macOS, Linux et WSL2
4. **Gardez les PR ciblées** : un seul changement logique par PR. Ne mélangez pas une correction de bug avec un refactoring et une nouvelle fonctionnalité.

### Description de la PR

Incluez :
- **Quoi** a changé et **pourquoi**
- **Comment le tester** (étapes de reproduction pour les bugs, exemples d'usage pour les fonctionnalités)
- **Les plateformes** sur lesquelles vous avez testé
- Les références aux issues liées

### Messages de commit

Nous utilisons les [Conventional Commits](https://www.conventionalcommits.org/) :

```
<type>(<scope>): <description>
```

| Type | À utiliser pour |
|------|---------|
| `fix` | Corrections de bugs |
| `feat` | Nouvelles fonctionnalités |
| `docs` | Documentation |
| `test` | Tests |
| `refactor` | Restructuration du code (sans changement de comportement) |
| `chore` | Build, CI, mises à jour de dépendances |

Scopes : `cli`, `gateway`, `tools`, `skills`, `agent`, `install`, `whatsapp`, `security`, etc.

Exemples :
```
fix(cli): prevent crash in save_config_value when model is a string
feat(gateway): add WhatsApp multi-user session isolation
fix(security): prevent shell injection in sudo password piping
test(tools): add unit tests for file_operations
```

---

## Signaler un problème

- Utilisez les [GitHub Issues](https://github.com/NousResearch/hermes-agent/issues)
- Incluez : OS, version de Python, version d'Hermes (`hermes version`), traceback d'erreur complet
- Incluez les étapes de reproduction
- Vérifiez les issues existantes avant de créer des doublons
- Pour les vulnérabilités de sécurité, merci de les signaler en privé

---

## Communauté

- **Discord** : [discord.gg/NousResearch](https://discord.gg/NousResearch) — pour poser des questions, présenter vos projets et partager des compétences
- **GitHub Discussions** : pour les propositions de conception et les discussions d'architecture
- **Skills Hub** : téléversez des compétences spécialisées sur un registre et partagez-les avec la communauté

---

## Licence

En contribuant, vous acceptez que vos contributions soient placées sous la [licence MIT](LICENSE).
