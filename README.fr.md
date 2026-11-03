<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/">Hermes Agent</a> | <a href="https://hermes-agent.nousresearch.com/">Hermes Desktop</a>
</p>
<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="Licence : MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="Créé par Nous Research"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/Lang-English-lightgrey?style=for-the-badge" alt="English"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-中文-red?style=for-the-badge" alt="中文"></a>
  <a href="README.ur-pk.md"><img src="https://img.shields.io/badge/Lang-اردو-green?style=for-the-badge" alt="اردو"></a>
  <a href="README.es.md"><img src="https://img.shields.io/badge/Lang-Español-orange?style=for-the-badge" alt="Español"></a>
</p>

**L'agent IA qui s'améliore tout seul, créé par [Nous Research](https://nousresearch.com).** C'est le seul agent doté d'une boucle d'apprentissage intégrée : il crée des skills à partir de son expérience, les améliore à l'usage, se rappelle lui-même de conserver ses connaissances, fouille ses propres conversations passées et construit, session après session, un modèle de plus en plus fin de qui vous êtes. Faites-le tourner sur un VPS à 5 $, un cluster GPU ou une infrastructure serverless qui ne coûte presque rien au repos. Il n'est pas lié à votre ordinateur portable : parlez-lui depuis Telegram pendant qu'il travaille sur une MV dans le cloud.

Utilisez le modèle de votre choix : [Nous Portal](https://portal.nousresearch.com), OpenRouter, OpenAI, votre propre endpoint et [bien d'autres](https://hermes-agent.nousresearch.com/docs/integrations/providers). Changez-en avec `hermes model` : aucune modification de code, aucun verrouillage.

<table>
<tr><td><b>Une vraie interface terminal</b></td><td>TUI complète avec édition multiligne, autocomplétion des commandes slash, historique des conversations, interruption et réorientation, et sortie des outils en streaming.</td></tr>
<tr><td><b>Là où vous êtes</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal et CLI, le tout depuis un seul processus gateway. Transcription des messages vocaux, continuité des conversations d'une plateforme à l'autre.</td></tr>
<tr><td><b>Une boucle d'apprentissage fermée</b></td><td>Mémoire gérée par l'agent avec des rappels périodiques. Création autonome de skills après les tâches complexes. Les skills s'améliorent à l'usage. Recherche dans les sessions via FTS5, avec résumés par LLM pour se souvenir d'une session à l'autre. Modélisation dialectique de l'utilisateur avec <a href="https://github.com/plastic-labs/honcho">Honcho</a>. Compatible avec le standard ouvert <a href="https://agentskills.io">agentskills.io</a>.</td></tr>
<tr><td><b>Automatisations planifiées</b></td><td>Planificateur cron intégré, avec livraison sur n'importe quelle plateforme. Rapports quotidiens, sauvegardes nocturnes, audits hebdomadaires : tout en langage naturel, sans surveillance.</td></tr>
<tr><td><b>Délègue et parallélise</b></td><td>Lancez des sous-agents isolés pour mener plusieurs chantiers en parallèle. Dans Desktop, choisissez leur fournisseur, leur modèle et leurs limites dans Settings → Model → Subagents, puis appliquez la route complète avec Apply. Écrivez des scripts Python qui appellent les outils par RPC et réduisent des pipelines en plusieurs étapes à des tours qui ne coûtent rien en contexte.</td></tr>
<tr><td><b>Tourne partout, pas seulement sur votre portable</b></td><td>Six backends de terminal intégrés (local, Docker, SSH, Modal, Daytona et Vercel Sandbox), plus des backends sous forme de plugins comme Singularity/Apptainer. Daytona et Modal offrent une persistance serverless : l'environnement de votre agent hiberne au repos et se réveille à la demande, pour un coût quasi nul entre deux sessions. Faites-le tourner sur un VPS à 5 $ ou un cluster GPU.</td></tr>
<tr><td><b>Prêt pour la recherche</b></td><td>Génération de trajectoires par lots, compression de trajectoires pour entraîner la prochaine génération de modèles capables d'appeler des outils.</td></tr>
</table>

---

## Installation rapide

### Linux, macOS, WSL2

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

### Windows (natif, PowerShell)

> **À savoir :** Windows natif fait tourner Hermes sans WSL ; la CLI, la gateway, la TUI et les outils fonctionnent tous nativement. Si vous préférez WSL2, la commande Linux/macOS ci-dessus y fonctionne aussi. Vous avez trouvé un bug ? Merci d'[ouvrir une issue](https://github.com/NousResearch/hermes-agent/issues).

Lancez ceci dans PowerShell :

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

L'installateur depuis les sources confie à PM l'installation de Python 3.14,
Node.js, npm, ripgrep, FFmpeg et des dépendances Python. Si Git est absent, il
place l'archive vérifiée de Git for Windows dans le magasin d'outils de Hermes,
sans remplacer le Git de votre système. Consultez les
[méthodes d'installation](https://hermes-agent.nousresearch.com/docs/getting-started/installation)
pour le paquet MSIX/App Installer séparé et la façon dont il gère ses mises à jour.

> **Android / Termux :** un dépôt APT signé est disponible pour les appareils aarch64, avec un canal `stable` (versions taguées) et un canal de préversion `canary`. Le paquet inclut Python, Node.js et la TUI. Suivez le [guide Termux](https://hermes-agent.nousresearch.com/docs/getting-started/termux), pas le script d'installation pour ordinateur/serveur.
>
> **Windows :** Windows natif est entièrement pris en charge ; la commande PowerShell ci-dessus installe tout. Si vous préférez WSL2, la commande Linux y fonctionne aussi. L'installation native sous Windows se trouve dans `%LOCALAPPDATA%\hermes` ; sous WSL2, elle se trouve dans `~/.hermes`, comme sous Linux.

Après l'installation :

```bash
source ~/.bashrc    # recharger le shell (ou : source ~/.zshrc)
hermes              # commencez à discuter !
```

### Dépannage

#### Windows Defender ou votre antivirus signale `uv.exe` comme malveillant

Si votre antivirus (Bitdefender, Windows Defender, etc.) met en quarantaine `uv.exe` depuis le dossier `bin` de Hermes (`%LOCALAPPDATA%\hermes\bin\uv.exe`), il s'agit d'un **faux positif**. Ce fichier est `uv`, d'Astral : le gestionnaire de paquets Python écrit en Rust que Hermes embarque pour gérer son environnement Python. Les antivirus basés sur l'apprentissage automatique signalent souvent les binaires Rust non signés qui téléchargent et installent des paquets.

**Pour vérifier que votre copie est authentique :**

```powershell
# Installer GitHub CLI si nécessaire
winget install --id GitHub.cli

# Se connecter à GitHub
gh auth login

# Lancer la vérification
$uv = "$env:LOCALAPPDATA\hermes\bin\uv.exe"
$ver = (& $uv --version).Split(' ')[1]
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$zip = "$env:TEMP\uv.zip"
Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$ver/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip -UseBasicParsing
gh attestation verify $zip --repo astral-sh/uv
Expand-Archive $zip "$env:TEMP\uv_x" -Force
(Get-FileHash "$env:TEMP\uv_x\uv.exe").Hash -eq (Get-FileHash $uv).Hash
```

Si l'attestation affiche « Verification succeeded » et que la dernière ligne affiche `True`, tout va bien.

**Pour ajouter Hermes en liste blanche :**

- **Windows Defender :** lancez PowerShell en administrateur → `Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\hermes\bin"`
- **Bitdefender :** ajoutez une exception dans la console Bitdefender (Protection > Antivirus > Paramètres > Gérer les exceptions)
- Mettez en liste blanche le **dossier**, pas l'empreinte du fichier : Hermes met `uv` à jour et l'empreinte change à chaque version

Pour plus de contexte, consultez les signalements côté Astral : [astral-sh/uv#13553](https://github.com/astral-sh/uv/issues/13553), [astral-sh/uv#15011](https://github.com/astral-sh/uv/issues/15011), [astral-sh/uv#10079](https://github.com/astral-sh/uv/issues/10079).

---

## Premiers pas

```bash
hermes              # CLI interactive : démarrer une conversation
hermes model        # Choisir votre fournisseur de LLM et votre modèle
hermes tools        # Configurer les outils activés
hermes config set   # Définir une valeur de configuration
hermes config get   # Afficher une valeur de configuration
hermes gateway      # Démarrer la gateway de messagerie (Telegram, Discord, etc.)
hermes setup        # Lancer l'assistant de configuration complet (configure tout d'un coup)
hermes update       # Mettre à jour vers la dernière version
hermes doctor       # Diagnostiquer les problèmes
```

📖 **[Documentation complète →](https://hermes-agent.nousresearch.com/docs/)**

---

## Évitez de collectionner les clés d'API : Nous Portal

Hermes fonctionne avec le fournisseur de votre choix, et ça ne changera pas. Mais si vous préférez ne pas gérer cinq clés d'API distinctes pour le modèle, la recherche web, la génération d'images, la synthèse vocale et un navigateur dans le cloud, **[Nous Portal](https://portal.nousresearch.com)** couvre tout cela avec un seul abonnement :

- **Plus de 300 modèles** : choisissez-en un avec `/model <nom>`
- **Tool Gateway** : recherche web, génération d'images (FAL), synthèse vocale (OpenAI), navigateur dans le cloud (Browser Use), le tout passant par votre abonnement. Aucun compte supplémentaire.

Une seule commande après une installation neuve :

```bash
hermes setup --portal
```

Elle vous connecte via OAuth, définit Nous comme fournisseur et active la Tool Gateway. Vérifiez à tout moment ce qui est branché avec `hermes portal info`. Tous les détails sur la [page de documentation de la Tool Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway).

Vous pouvez toujours utiliser vos propres clés outil par outil quand vous le souhaitez : la gateway se règle par backend, ce n'est pas tout ou rien.

---

## Aide-mémoire CLI et messagerie

Hermes a deux points d'entrée : lancez l'interface terminal avec `hermes`, ou démarrez la gateway et parlez-lui depuis Telegram, Discord, Slack, WhatsApp, Signal ou par e-mail. Une fois dans une conversation, beaucoup de commandes slash sont communes aux deux interfaces.

| Action                                | CLI                                           | Plateformes de messagerie                                                              |
| ------------------------------------- | --------------------------------------------- | -------------------------------------------------------------------------------------- |
| Commencer à discuter                  | `hermes`                                      | Lancez `hermes gateway setup` + `hermes gateway start`, puis envoyez un message au bot |
| Démarrer une nouvelle conversation    | `/new` ou `/reset`                            | `/new` ou `/reset`                                                                     |
| Changer de modèle                     | `/model [provider:model]`                     | `/model [provider:model]`                                                              |
| Définir une personnalité              | `/personality [nom]`                          | `/personality [nom]`                                                                   |
| Relancer ou annuler le dernier tour   | `/retry`, `/undo`                             | `/retry`, `/undo`                                                                      |
| Compresser le contexte / voir l'usage | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [jours]`                                             |
| Parcourir les skills                  | `/skills` ou `/<nom-du-skill>`                | `/<nom-du-skill>`                                                                      |
| Interrompre le travail en cours       | `Ctrl+C` ou envoyer un nouveau message        | `/stop` ou envoyer un nouveau message                                                  |
| Statut propre à la plateforme         | `/platforms`                                  | `/status`, `/sethome`                                                                  |

Pour la liste complète des commandes, consultez le [guide de la CLI](https://hermes-agent.nousresearch.com/docs/user-guide/cli) et le [guide de la gateway de messagerie](https://hermes-agent.nousresearch.com/docs/user-guide/messaging).

---

## Documentation

Toute la documentation se trouve sur **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)** (en anglais) :

| Section                                                                                                 | Contenu                                                             |
| ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| [Démarrage rapide](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart)               | Installation → configuration → première conversation en 2 minutes   |
| [Utilisation de la CLI](https://hermes-agent.nousresearch.com/docs/user-guide/cli)                      | Commandes, raccourcis clavier, personnalités, sessions              |
| [Configuration](https://hermes-agent.nousresearch.com/docs/user-guide/configuration)                    | Fichier de configuration, fournisseurs, modèles, toutes les options |
| [Gateway de messagerie](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)                | Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant          |
| [Sécurité](https://hermes-agent.nousresearch.com/docs/user-guide/security)                              | Approbation des commandes, appairage en DM, isolation par conteneur |
| [Outils et toolsets](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools)              | Plus de 40 outils, système de toolsets, backends de terminal        |
| [Système de skills](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills)              | Mémoire procédurale, Skills Hub, création de skills                 |
| [Mémoire](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)                        | Mémoire persistante, profils utilisateur, bonnes pratiques          |
| [Intégration MCP](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp)                   | Connectez n'importe quel serveur MCP pour étendre les capacités     |
| [Planification cron](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron)               | Tâches planifiées avec livraison sur les plateformes                |
| [Fichiers de contexte](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files)    | Le contexte de projet qui oriente chaque conversation               |
| [Architecture](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture)                 | Structure du projet, boucle de l'agent, classes principales         |
| [Contribuer](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing)                   | Environnement de développement, processus de PR, style de code      |
| [Référence de la CLI](https://hermes-agent.nousresearch.com/docs/reference/cli-commands)                | Toutes les commandes et options                                     |
| [Variables d'environnement](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) | Référence complète des variables d'environnement                    |

---

## Migrer depuis OpenClaw ou un autre agent

Si vous venez d'OpenClaw, Claude Code, Codex CLI ou Cursor, Hermes peut importer vos réglages, vos mémoires et vos skills (et, pour OpenClaw, vos clés d'API si vous le demandez).

**Pendant la configuration :** `hermes setup` vous demande si vous voulez importer depuis un autre agent, présélectionne ceux qu'il trouve (`~/.openclaw`, `~/.claude`, `~/.codex`, `~/.cursor`), installe le plugin officiel `agent-importers` et lance l'import en affichant d'abord un aperçu.

**À tout moment après l'installation :**

```bash
hermes plugins install agent-importers          # une fois par profil
hermes import-agent openclaw                    # migration interactive (preset complet)
hermes import-agent openclaw --dry-run          # aperçu de ce qui serait migré
hermes import-agent openclaw --migrate-secrets  # importe aussi les clés d'API autorisées
hermes import-agent claude-code                 # ou codex / cursor
```

Ce qui est importé depuis OpenClaw :

- **SOUL.md** : le fichier de persona
- **Mémoires** : les entrées de MEMORY.md et USER.md
- **Skills** : les skills créés par l'utilisateur → `~/.hermes/skills/openclaw-imports/`
- **Liste blanche de commandes** : les motifs d'approbation
- **Réglages de messagerie** : configuration des plateformes, utilisateurs autorisés, répertoire de travail
- **Clés d'API** : les secrets autorisés (Telegram, OpenRouter, OpenAI, Anthropic, ElevenLabs), uniquement avec `--migrate-secrets`
- **Ressources TTS** : les fichiers audio de l'espace de travail
- **Instructions de l'espace de travail** : AGENTS.md (avec `--workspace-target`)

L'ancienne commande `hermes claw migrate` fonctionne toujours. Consultez le [guide de migration depuis OpenClaw](https://hermes-agent.nousresearch.com/docs/guides/migrate-from-openclaw) et [Importer depuis d'autres agents](https://hermes-agent.nousresearch.com/docs/user-guide/import-from-other-agents).

---

## Contribuer

Les contributions sont les bienvenues ! Consultez le [guide de contribution](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) pour l'environnement de développement, le style de code et le processus de PR.

Commencez par le [workflow de développement PM](website/docs/reference/package-management.md#developer-workflow)
pour l'activation, l'usage quotidien, les changements de dépendances et la sortie de l'environnement.
[Development Setup](CONTRIBUTING.md#development-setup) couvre l'environnement de test séparé et les commandes de vérification.

---

## Communauté

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Issues](https://github.com/NousResearch/hermes-agent/issues)
- 🔌 [computer-use-linux](https://github.com/avifenesh/computer-use-linux) : serveur MCP de contrôle du bureau Linux pour Hermes et d'autres hôtes MCP, avec arbres d'accessibilité AT-SPI, saisie Wayland/X11, captures d'écran et ciblage des fenêtres par le compositeur.
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) : pont WeChat communautaire pour faire tourner Hermes Agent et OpenClaw sur le même compte WeChat.

---

## Licence

MIT : voir [LICENSE](LICENSE).

Créé par [Nous Research](https://nousresearch.com).
