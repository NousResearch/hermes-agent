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
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/Licence-MIT-green?style=for-the-badge" alt="Licence : MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Cr%C3%A9%C3%A9%20par-Nous%20Research-blueviolet?style=for-the-badge" alt="Créé par Nous Research"></a>
  <a href="README.md"><img src="https://img.shields.io/badge/Lang-English-blue?style=for-the-badge" alt="English"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-中文-red?style=for-the-badge" alt="中文"></a>
  <a href="README.ur-pk.md"><img src="https://img.shields.io/badge/Lang-اردو-green?style=for-the-badge" alt="اردو"></a>
  <a href="README.es.md"><img src="https://img.shields.io/badge/Lang-Español-orange?style=for-the-badge" alt="Español"></a>
</p>

**L'agent d'IA qui s'améliore tout seul, conçu par [Nous Research](https://nousresearch.com).** C'est le seul agent doté d'une boucle d'apprentissage intégrée : il crée des compétences à partir de l'expérience, les améliore à l'usage, s'incite lui-même à conserver ce qu'il apprend, fouille dans ses propres conversations passées et construit, de session en session, un modèle de plus en plus fin de qui vous êtes. Faites-le tourner sur un VPS à 5 $, sur un cluster de GPU ou sur une infrastructure serverless qui ne coûte presque rien au repos. Il n'est pas attaché à votre ordinateur portable : parlez-lui depuis Telegram pendant qu'il travaille sur une VM dans le cloud.

Utilisez le modèle de votre choix : [Nous Portal](https://portal.nousresearch.com), OpenRouter, OpenAI, votre propre endpoint, et [bien d'autres](https://hermes-agent.nousresearch.com/docs/integrations/providers). Changez avec `hermes model`, sans modifier une ligne de code et sans dépendance à un fournisseur.

<table>
<tr><td><b>Une vraie interface de terminal</b></td><td>TUI complète avec édition multiligne, autocomplétion des commandes slash, historique des conversations, interruption et redirection en cours de route, et sortie des outils en streaming.</td></tr>
<tr><td><b>Il vit là où vous vivez</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal et CLI, le tout depuis un seul processus de passerelle. Transcription des mémos vocaux, continuité des conversations d'une plateforme à l'autre.</td></tr>
<tr><td><b>Une boucle d'apprentissage fermée</b></td><td>Mémoire entretenue par l'agent, avec des rappels périodiques. Création autonome de compétences après les tâches complexes. Les compétences s'améliorent d'elles-mêmes à l'usage. Recherche FTS5 dans les sessions, avec résumés par LLM, pour se souvenir d'une session à l'autre. Modélisation dialectique de l'utilisateur avec <a href="https://github.com/plastic-labs/honcho">Honcho</a>. Compatible avec le standard ouvert <a href="https://agentskills.io">agentskills.io</a>.</td></tr>
<tr><td><b>Des automatisations planifiées</b></td><td>Planificateur cron intégré, avec livraison sur n'importe quelle plateforme. Rapports quotidiens, sauvegardes nocturnes, audits hebdomadaires : le tout en langage naturel, sans surveillance.</td></tr>
<tr><td><b>Il délègue et parallélise</b></td><td>Lancez des sous-agents isolés pour mener plusieurs chantiers en parallèle. Écrivez des scripts Python qui appellent des outils via RPC et ramènent des pipelines à plusieurs étapes à des tours sans coût de contexte.</td></tr>
<tr><td><b>Il tourne partout, pas seulement sur votre ordinateur portable</b></td><td>Sept backends de terminal : local, Docker, SSH, Singularity, Modal, Daytona et Vercel Sandbox. Daytona et Modal offrent une persistance serverless : l'environnement de votre agent se met en veille quand il est inactif et se réveille à la demande, pour un coût quasi nul entre deux sessions. Faites-le tourner sur un VPS à 5 $ ou sur un cluster de GPU.</td></tr>
<tr><td><b>Prêt pour la recherche</b></td><td>Génération de trajectoires par lots, compression de trajectoires pour entraîner la prochaine génération de modèles capables d'appeler des outils.</td></tr>
</table>

---

## Installation rapide

### Linux, macOS, WSL2

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

### Windows (natif, PowerShell)

> **À savoir :** sous Windows natif, Hermes fonctionne sans WSL : la CLI, la passerelle, la TUI et les outils tournent tous nativement. Si vous préférez WSL2, la commande Linux/macOS ci-dessus y fonctionne aussi. Vous avez trouvé un bogue ? [Ouvrez un ticket](https://github.com/NousResearch/hermes-agent/issues).

Exécutez ceci dans PowerShell :

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

L'installateur depuis les sources confie à PM l'installation de Python 3.14, Node.js, npm, ripgrep, FFmpeg
et des dépendances Python. Si Git est absent, il dépose l'archive vérifiée de
Git for Windows dans l'espace d'outils de Hermes. Il ne remplace pas le Git de votre système.
Consultez les [méthodes d'installation](https://hermes-agent.nousresearch.com/docs/getting-started/installation)
pour le paquet MSIX/App Installer, distinct, et pour savoir qui en gère les mises à jour.

> **Android / Termux :** un dépôt APT signé est disponible pour les appareils aarch64, avec un canal `stable` (versions étiquetées) et un canal `canary` de préversion. Le paquet inclut Python, Node.js et la TUI. Suivez le [guide Termux](https://hermes-agent.nousresearch.com/docs/getting-started/termux) et non le script d'installation pour poste de travail et serveur.
>
> **Windows :** Windows natif est entièrement pris en charge : la commande PowerShell ci-dessus installe tout. Si vous préférez WSL2, la commande Linux y fonctionne aussi. Sous Windows natif, l'installation se trouve dans `%LOCALAPPDATA%\hermes` ; sous WSL2, elle se trouve dans `~/.hermes`, comme sous Linux.

Après l'installation :

```bash
source ~/.bashrc    # recharger le shell (ou : source ~/.zshrc)
hermes              # lancez la conversation !
```

### Dépannage

#### Windows Defender ou votre antivirus signale `uv.exe` comme un logiciel malveillant

Si votre antivirus (Bitdefender, Windows Defender, etc.) met en quarantaine `uv.exe` dans le dossier `bin` de Hermes (`%LOCALAPPDATA%\hermes\bin\uv.exe`), il s'agit d'un **faux positif**. Ce fichier est `uv`, d'Astral : le gestionnaire de paquets Python écrit en Rust que Hermes embarque pour gérer son environnement Python. Les moteurs antivirus à base d'apprentissage automatique signalent couramment les binaires Rust non signés qui téléchargent et installent des paquets.

**Pour vérifier l'authenticité de votre exemplaire :**

```powershell
# Install GitHub CLI if needed
winget install --id GitHub.cli

# Login to GitHub
gh auth login

# Run verification
$uv = "$env:LOCALAPPDATA\hermes\bin\uv.exe"
$ver = (& $uv --version).Split(' ')[1]
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$zip = "$env:TEMP\uv.zip"
Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$ver/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip -UseBasicParsing
gh attestation verify $zip --repo astral-sh/uv
Expand-Archive $zip "$env:TEMP\uv_x" -Force
(Get-FileHash "$env:TEMP\uv_x\uv.exe").Hash -eq (Get-FileHash $uv).Hash
```

Si l'attestation indique « Verification succeeded » et que la dernière ligne affiche `True`, tout est en ordre.

**Pour ajouter Hermes aux exclusions :**
- **Windows Defender :** exécutez PowerShell en tant qu'administrateur → `Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\hermes\bin"`
- **Bitdefender :** ajoutez une exception dans la console Bitdefender (Protection > Antivirus > Paramètres > Gérer les exceptions)
- Excluez le **dossier**, pas l'empreinte du fichier : Hermes met `uv` à jour et l'empreinte change à chaque version

Pour plus de contexte, consultez les signalements d'Astral en amont : [astral-sh/uv#13553](https://github.com/astral-sh/uv/issues/13553), [astral-sh/uv#15011](https://github.com/astral-sh/uv/issues/15011), [astral-sh/uv#10079](https://github.com/astral-sh/uv/issues/10079).

---

## Premiers pas

```bash
hermes              # CLI interactive : démarre une conversation
hermes model        # Choisir votre fournisseur de LLM et votre modèle
hermes tools        # Configurer les outils activés
hermes config set   # Définir une valeur de configuration
hermes config get   # Afficher une valeur de configuration
hermes gateway      # Démarrer la passerelle de messagerie (Telegram, Discord, etc.)
hermes setup        # Lancer l'assistant de configuration complet (configure tout d'un coup)
hermes claw migrate # Migrer depuis OpenClaw (si vous venez d'OpenClaw)
hermes update       # Mettre à jour vers la dernière version
hermes doctor       # Diagnostiquer les problèmes
```

📖 **[Documentation complète →](https://hermes-agent.nousresearch.com/docs/)**

---

## Plus de clés d'API à collectionner : Nous Portal

Hermes fonctionne avec le fournisseur de votre choix, et cela ne changera pas. Mais si vous préférez ne pas rassembler cinq clés d'API distinctes pour le modèle, la recherche web, la génération d'images, la synthèse vocale (TTS) et un navigateur dans le cloud, **[Nous Portal](https://portal.nousresearch.com)** les couvre toutes avec un seul abonnement :

- **Plus de 300 modèles** : choisissez-en un avec `/model <nom>`
- **Tool Gateway** : recherche web, génération d'images (FAL), synthèse vocale (OpenAI), navigateur dans le cloud (Browser Use), le tout acheminé via votre abonnement. Aucun compte supplémentaire.

Une seule commande depuis une installation neuve :

```bash
hermes setup --portal
```

Elle vous connecte via OAuth, définit Nous comme fournisseur et active le Tool Gateway. Vérifiez à tout moment ce qui est branché avec `hermes portal info`. Tous les détails sont sur la [page de documentation du Tool Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway).

Vous pouvez toujours utiliser vos propres clés pour chaque outil quand vous le souhaitez : le Tool Gateway se règle backend par backend, ce n'est pas du tout ou rien.

---

## Aide-mémoire : CLI et messagerie

Hermes a deux points d'entrée : lancez l'interface de terminal avec `hermes`, ou démarrez la passerelle et parlez-lui depuis Telegram, Discord, Slack, WhatsApp, Signal ou Email. Une fois la conversation engagée, de nombreuses commandes slash sont communes aux deux interfaces.

| Action                                  | CLI                                           | Plateformes de messagerie                                                                             |
| --------------------------------------- | --------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| Démarrer la conversation                | `hermes`                                      | Exécutez `hermes gateway setup` + `hermes gateway start`, puis envoyez un message au bot              |
| Repartir d'une conversation neuve       | `/new` ou `/reset`                            | `/new` ou `/reset`                                                                                    |
| Changer de modèle                       | `/model [fournisseur:modèle]`                 | `/model [fournisseur:modèle]`                                                                         |
| Définir une personnalité                | `/personality [nom]`                          | `/personality [nom]`                                                                                  |
| Réessayer ou annuler le dernier tour    | `/retry`, `/undo`                             | `/retry`, `/undo`                                                                                     |
| Compresser le contexte / voir l'usage   | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]`                                                             |
| Parcourir les compétences               | `/skills` ou `/<nom-de-la-compétence>`        | `/<nom-de-la-compétence>`                                                                             |
| Interrompre le travail en cours         | `Ctrl+C` ou envoyer un nouveau message        | `/stop` ou envoyer un nouveau message                                                                 |
| État propre à la plateforme             | `/platforms`                                  | `/status`, `/sethome`                                                                                 |

Pour les listes de commandes complètes, consultez le [guide de la CLI](https://hermes-agent.nousresearch.com/docs/user-guide/cli) et le [guide de la passerelle de messagerie](https://hermes-agent.nousresearch.com/docs/user-guide/messaging).

---

## Documentation

Toute la documentation se trouve sur **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)** :

| Section                                                                                             | Contenu                                                                    |
| --------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| [Démarrage rapide](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart)           | Installation → configuration → première conversation en 2 minutes          |
| [Utilisation de la CLI](https://hermes-agent.nousresearch.com/docs/user-guide/cli)                  | Commandes, raccourcis clavier, personnalités, sessions                     |
| [Configuration](https://hermes-agent.nousresearch.com/docs/user-guide/configuration)                | Fichier de configuration, fournisseurs, modèles, toutes les options        |
| [Passerelle de messagerie](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)         | Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant                 |
| [Sécurité](https://hermes-agent.nousresearch.com/docs/user-guide/security)                          | Approbation des commandes, appairage par message privé, isolation par conteneur |
| [Outils et jeux d'outils](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools)     | Plus de 40 outils, système de jeux d'outils, backends de terminal          |
| [Système de compétences](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills)     | Mémoire procédurale, Skills Hub, création de compétences                   |
| [Mémoire](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)                    | Mémoire persistante, profils utilisateur, bonnes pratiques                 |
| [Intégration MCP](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp)               | Connectez n'importe quel serveur MCP pour étendre les capacités            |
| [Planification cron](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron)           | Tâches planifiées avec livraison sur la plateforme de votre choix          |
| [Fichiers de contexte](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files) | Contexte de projet qui façonne chaque conversation                        |
| [Architecture](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture)             | Structure du projet, boucle de l'agent, classes principales                |
| [Contribuer](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing)               | Environnement de développement, processus de PR, style de code             |
| [Référence de la CLI](https://hermes-agent.nousresearch.com/docs/reference/cli-commands)            | Toutes les commandes et tous les indicateurs                               |
| [Variables d'environnement](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) | Référence complète des variables d'environnement                       |

---

## Migrer depuis OpenClaw

Si vous venez d'OpenClaw, Hermes peut importer automatiquement vos réglages, vos souvenirs, vos compétences et vos clés d'API.

**Lors de la première configuration :** l'assistant de configuration (`hermes setup`) détecte automatiquement `~/.openclaw` et propose de migrer avant le début de la configuration.

**À tout moment après l'installation :**

```bash
hermes claw migrate              # Migration interactive (préréglage complet)
hermes claw migrate --dry-run    # Aperçu de ce qui serait migré
hermes claw migrate --preset user-data   # Migrer sans les secrets
hermes claw migrate --overwrite  # Écraser les conflits existants
```

Ce qui est importé :

- **SOUL.md** : le fichier de personnalité
- **Souvenirs** : les entrées de MEMORY.md et de USER.md
- **Compétences** : les compétences créées par l'utilisateur → `~/.hermes/skills/openclaw-imports/`
- **Liste de commandes autorisées** : les motifs d'approbation
- **Réglages de messagerie** : configurations des plateformes, utilisateurs autorisés, répertoire de travail
- **Clés d'API** : les secrets de la liste d'autorisation (Telegram, OpenRouter, OpenAI, Anthropic, ElevenLabs)
- **Ressources TTS** : les fichiers audio de l'espace de travail
- **Instructions de l'espace de travail** : AGENTS.md (avec `--workspace-target`)

Consultez `hermes claw migrate --help` pour toutes les options, ou utilisez la compétence `openclaw-migration` pour une migration interactive guidée par l'agent, avec aperçus en simulation (dry-run).

---

## Contribuer

Les contributions sont les bienvenues ! Consultez le [guide de contribution](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing) pour l'environnement de développement, le style de code et le processus de PR.

Commencez par le [flux de travail développeur avec PM](website/docs/reference/package-management.md#developer-workflow)
pour l'activation, l'usage au quotidien, les changements de dépendances et la sortie de l'environnement.
[Development Setup](CONTRIBUTING.md#development-setup) décrit l'environnement de test, distinct, et les commandes de vérification.

---

## Communauté

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Issues](https://github.com/NousResearch/hermes-agent/issues)
- 🔌 [computer-use-linux](https://github.com/avifenesh/computer-use-linux) : serveur MCP de contrôle du bureau Linux pour Hermes et d'autres hôtes MCP, avec arbres d'accessibilité AT-SPI, saisie Wayland/X11, captures d'écran et ciblage des fenêtres par le compositeur.
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) : pont WeChat communautaire. Faites tourner Hermes Agent et OpenClaw sur le même compte WeChat.

---

## Licence

MIT : voir [LICENSE](LICENSE).

Conçu par [Nous Research](https://nousresearch.com).
