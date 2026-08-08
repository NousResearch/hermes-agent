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

**L'agent d'IA auto-améliorant conçu par [Nous Research](https://nousresearch.com).** C'est le seul agent doté d'une boucle d'apprentissage intégrée : il crée des compétences à partir de son expérience, les améliore à l'usage, se pousse lui-même à pérenniser ses connaissances, fouille ses propres conversations passées et construit, session après session, un modèle de plus en plus fin de qui vous êtes. Faites-le tourner sur un VPS à 5 $, un cluster GPU ou une infrastructure serverless qui ne coûte presque rien à l'arrêt. Il n'est pas rivé à votre ordinateur portable — parlez-lui depuis Telegram pendant qu'il travaille sur une VM dans le cloud.

Utilisez le modèle de votre choix — [Nous Portal](https://portal.nousresearch.com), OpenRouter, OpenAI, votre propre endpoint et [bien d'autres](https://hermes-agent.nousresearch.com/docs/integrations/providers). Changez de modèle avec `hermes model` — pas de modification de code, pas de verrouillage.

<table>
<tr><td><b>Une vraie interface de terminal</b></td><td>TUI complète avec édition multiligne, autocomplétion des commandes slash, historique des conversations, interruption et redirection à la volée, et sortie des outils en streaming.</td></tr>
<tr><td><b>Il vit là où vous vivez</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal et CLI — le tout depuis un seul processus gateway. Transcription des mémos vocaux, continuité des conversations d'une plateforme à l'autre.</td></tr>
<tr><td><b>Une boucle d'apprentissage fermée</b></td><td>Mémoire organisée par l'agent, avec des rappels périodiques. Création autonome de compétences après les tâches complexes. Les compétences s'améliorent d'elles-mêmes à l'usage. Recherche de sessions FTS5 avec résumé par LLM pour retrouver des informations d'une session à l'autre. Modélisation dialectique de l'utilisateur via <a href="https://github.com/plastic-labs/honcho">Honcho</a>. Compatible avec le standard ouvert <a href="https://agentskills.io">agentskills.io</a>.</td></tr>
<tr><td><b>Automatisations planifiées</b></td><td>Planificateur cron intégré avec livraison sur n'importe quelle plateforme. Rapports quotidiens, sauvegardes nocturnes, audits hebdomadaires — le tout en langage naturel, sans aucune surveillance.</td></tr>
<tr><td><b>Il délègue et parallélise</b></td><td>Lancez des sous-agents isolés pour mener plusieurs chantiers en parallèle. Écrivez des scripts Python qui appellent les outils via RPC, condensant des pipelines à plusieurs étapes en tours sans aucun coût de contexte.</td></tr>
<tr><td><b>Tourne partout, pas seulement sur votre laptop</b></td><td>Six backends de terminal — local, Docker, SSH, Singularity, Modal et Daytona. Daytona et Modal offrent une persistance serverless : l'environnement de votre agent hiberne quand il est inactif et se réveille à la demande, pour un coût quasi nul entre les sessions. Faites-le tourner sur un VPS à 5 $ ou un cluster GPU.</td></tr>
<tr><td><b>Prêt pour la recherche</b></td><td>Génération de trajectoires par lots, compression de trajectoires pour entraîner la prochaine génération de modèles d'appel d'outils.</td></tr>
</table>

---

## Installation rapide

### Linux, macOS, WSL2, Termux

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

### Windows (natif, PowerShell)

> **Attention :** sous Windows natif, Hermes fonctionne sans WSL — la CLI, le gateway, la TUI et les outils tournent tous nativement. Si vous préférez passer par WSL2, la commande Linux/macOS ci-dessus y fonctionne aussi. Vous avez trouvé un bug ? Merci d'[ouvrir un ticket](https://github.com/NousResearch/hermes-agent/issues).

Exécutez ceci dans PowerShell :

```powershell
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

L'installateur s'occupe de tout : uv, Python 3.11, Node.js, ripgrep, ffmpeg, **et un Git Bash portable** (MinGit, décompressé dans `%LOCALAPPDATA%\hermes\git` — aucun droit administrateur requis, complètement isolé de toute installation Git du système). Hermes utilise ce Git Bash embarqué pour exécuter les commandes shell.

Si Git est déjà installé, l'installateur le détecte et l'utilise à la place. Sinon, un téléchargement MinGit d'environ 45 Mo suffit — il ne touchera pas au Git du système et n'interférera pas avec lui.

> **Android / Termux :** le chemin d'installation manuel testé est documenté dans le [guide Termux](https://hermes-agent.nousresearch.com/docs/getting-started/termux). Sur Termux, Hermes installe un extra `.[termux]` restreint, car l'extra complet `.[all]` tire actuellement des dépendances vocales incompatibles avec Android.
>
> **Windows :** Windows natif est entièrement pris en charge — la commande PowerShell ci-dessus installe tout. Si vous préférez WSL2, la commande Linux y fonctionne aussi. L'installation Windows native se trouve dans `%LOCALAPPDATA%\hermes` ; sous WSL2, elle s'installe dans `~/.hermes` comme sous Linux.

Après l'installation :

```bash
source ~/.bashrc    # reload shell (or: source ~/.zshrc)
hermes              # start chatting!
```

### Dépannage

#### Windows Defender ou un antivirus signale `uv.exe` comme malveillant

Si votre antivirus (Bitdefender, Windows Defender, etc.) met en quarantaine `uv.exe` depuis le dossier `bin` de Hermes (`%LOCALAPPDATA%\hermes\bin\uv.exe`), il s'agit d'un **faux positif**. Ce fichier est `uv` d'Astral — le gestionnaire de paquets Python écrit en Rust qu'Hermes embarque pour gérer son environnement Python. Les moteurs antivirus à base de ML signalent couramment les binaires Rust non signés qui téléchargent et installent des paquets.

**Pour vérifier que votre copie est authentique :**

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

Si l'attestation affiche « Verification succeeded » et que la dernière ligne renvoie `True`, tout est en ordre.

**Pour mettre Hermes en liste blanche :**
- **Windows Defender :** lancez PowerShell en administrateur → `Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\hermes\bin"`
- **Bitdefender :** ajoutez une exception dans la console Bitdefender (Protection > Antivirus > Paramètres > Gérer les exceptions)
- Mettez le **dossier** en liste blanche, pas le hash du fichier — Hermes met `uv` à jour et le hash change à chaque version

Pour plus de contexte, consultez les rapports upstream chez Astral : [astral-sh/uv#13553](https://github.com/astral-sh/uv/issues/13553), [astral-sh/uv#15011](https://github.com/astral-sh/uv/issues/15011), [astral-sh/uv#10079](https://github.com/astral-sh/uv/issues/10079).

---

## Premiers pas

```bash
hermes              # Interactive CLI — start a conversation
hermes model        # Choose your LLM provider and model
hermes tools        # Configure which tools are enabled
hermes config set   # Set individual config values
hermes gateway      # Start the messaging gateway (Telegram, Discord, etc.)
hermes setup        # Run the full setup wizard (configures everything at once)
hermes claw migrate # Migrate from OpenClaw (if coming from OpenClaw)
hermes update       # Update to the latest version
hermes doctor       # Diagnose any issues
```

📖 **[Documentation complète →](https://hermes-agent.nousresearch.com/docs/)**

---

## Fini la collection de clés API — Nous Portal

Hermes fonctionne avec le fournisseur de votre choix — ça ne changera pas. Mais si vous préférez ne pas accumuler cinq clés API distinctes pour le modèle, la recherche web, la génération d'images, le TTS et un navigateur cloud, **[Nous Portal](https://portal.nousresearch.com)** couvre tout cela avec un seul abonnement :

- **Plus de 300 modèles** — choisissez n'importe lequel avec `/model <name>`
- **Tool Gateway** — recherche web (Firecrawl), génération d'images (FAL), synthèse vocale (OpenAI), navigateur cloud (Browser Use), le tout routé via votre abonnement. Aucun compte supplémentaire.

Une seule commande depuis une installation vierge :

```bash
hermes setup --portal
```

Elle vous connecte via OAuth, définit Nous comme fournisseur et active le Tool Gateway. Vérifiez à tout moment ce qui est branché avec `hermes portal info`. Tous les détails sur la [page de documentation du Tool Gateway](https://hermes-agent.nousresearch.com/docs/user-guide/features/tool-gateway).

Vous pouvez toujours apporter vos propres clés, outil par outil — le gateway se configure backend par backend, pas en tout ou rien.

---

## Référence rapide : CLI vs messagerie

Hermes a deux points d'entrée : lancez l'interface de terminal avec `hermes`, ou démarrez le gateway et parlez-lui depuis Telegram, Discord, Slack, WhatsApp, Signal ou par e-mail. Une fois la conversation engagée, beaucoup de commandes slash sont communes aux deux interfaces.

| Action                                    | CLI                                           | Plateformes de messagerie                                                        |
| ----------------------------------------- | --------------------------------------------- | -------------------------------------------------------------------------------- |
| Commencer à discuter                      | `hermes`                                      | Lancez `hermes gateway setup` + `hermes gateway start`, puis envoyez un message au bot |
| Repartir d'une conversation vierge        | `/new` ou `/reset`                            | `/new` ou `/reset`                                                               |
| Changer de modèle                         | `/model [provider:model]`                     | `/model [provider:model]`                                                        |
| Définir une personnalité                  | `/personality [name]`                         | `/personality [name]`                                                            |
| Réessayer ou annuler le dernier tour      | `/retry`, `/undo`                             | `/retry`, `/undo`                                                                |
| Compresser le contexte / voir la consommation | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]`                                        |
| Parcourir les compétences                 | `/skills` ou `/<skill-name>`                  | `/<skill-name>`                                                                  |
| Interrompre le travail en cours           | `Ctrl+C` ou envoyer un nouveau message        | `/stop` ou envoyer un nouveau message                                            |
| Statut propre à la plateforme             | `/platforms`                                  | `/status`, `/sethome`                                                            |

Pour la liste complète des commandes, consultez le [guide CLI](https://hermes-agent.nousresearch.com/docs/user-guide/cli) et le [guide du gateway de messagerie](https://hermes-agent.nousresearch.com/docs/user-guide/messaging).

---

## Documentation

Toute la documentation se trouve sur **[hermes-agent.nousresearch.com/docs](https://hermes-agent.nousresearch.com/docs/)** :

| Section                                                                                             | Contenu                                                     |
| --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| [Démarrage rapide](https://hermes-agent.nousresearch.com/docs/getting-started/quickstart)           | Installation → configuration → première conversation en 2 minutes |
| [Utilisation de la CLI](https://hermes-agent.nousresearch.com/docs/user-guide/cli)                  | Commandes, raccourcis clavier, personnalités, sessions      |
| [Configuration](https://hermes-agent.nousresearch.com/docs/user-guide/configuration)                | Fichier de configuration, fournisseurs, modèles, toutes les options |
| [Gateway de messagerie](https://hermes-agent.nousresearch.com/docs/user-guide/messaging)            | Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant  |
| [Sécurité](https://hermes-agent.nousresearch.com/docs/user-guide/security)                          | Approbation des commandes, appairage par DM, isolation en conteneur |
| [Outils et toolsets](https://hermes-agent.nousresearch.com/docs/user-guide/features/tools)          | Plus de 40 outils, système de toolsets, backends de terminal |
| [Système de compétences](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills)     | Mémoire procédurale, Skills Hub, création de compétences    |
| [Mémoire](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)                    | Mémoire persistante, profils utilisateur, bonnes pratiques  |
| [Intégration MCP](https://hermes-agent.nousresearch.com/docs/user-guide/features/mcp)               | Connectez n'importe quel serveur MCP pour étendre les capacités |
| [Planification cron](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron)           | Tâches planifiées avec livraison sur les plateformes        |
| [Fichiers de contexte](https://hermes-agent.nousresearch.com/docs/user-guide/features/context-files) | Un contexte de projet qui façonne chaque conversation       |
| [Architecture](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture)             | Structure du projet, boucle de l'agent, classes principales |
| [Contribuer](https://hermes-agent.nousresearch.com/docs/developer-guide/contributing)               | Environnement de développement, processus de PR, style de code |
| [Référence CLI](https://hermes-agent.nousresearch.com/docs/reference/cli-commands)                  | Toutes les commandes et tous les flags                      |
| [Variables d'environnement](https://hermes-agent.nousresearch.com/docs/reference/environment-variables) | Référence complète des variables d'environnement        |

---

## Migration depuis OpenClaw

Si vous venez d'OpenClaw, Hermes peut importer automatiquement vos réglages, vos mémoires, vos compétences et vos clés API.

**Lors de la première configuration :** l'assistant de configuration (`hermes setup`) détecte automatiquement `~/.openclaw` et propose la migration avant de commencer la configuration.

**À tout moment après l'installation :**

```bash
hermes claw migrate              # Interactive migration (full preset)
hermes claw migrate --dry-run    # Preview what would be migrated
hermes claw migrate --preset user-data   # Migrate without secrets
hermes claw migrate --overwrite  # Overwrite existing conflicts
```

Ce qui est importé :

- **SOUL.md** — fichier de persona
- **Mémoires** — entrées de MEMORY.md et USER.md
- **Compétences** — compétences créées par l'utilisateur → `~/.hermes/skills/openclaw-imports/`
- **Liste blanche de commandes** — motifs d'approbation
- **Réglages de messagerie** — configuration des plateformes, utilisateurs autorisés, répertoire de travail
- **Clés API** — secrets en liste blanche (Telegram, OpenRouter, OpenAI, Anthropic, ElevenLabs)
- **Assets TTS** — fichiers audio de l'espace de travail
- **Instructions d'espace de travail** — AGENTS.md (avec `--workspace-target`)

Voir `hermes claw migrate --help` pour toutes les options, ou utilisez la compétence `openclaw-migration` pour une migration interactive guidée par l'agent, avec prévisualisation en dry-run.

---

## Contribuer

Les contributions sont les bienvenues ! Consultez le [Guide de contribution](CONTRIBUTING.fr.md) pour l'environnement de développement, le style de code et le processus de PR.

Démarrage rapide pour les contributeurs — utilisez l'installateur standard, puis
travaillez depuis le checkout git complet qu'il crée dans `$HERMES_HOME/hermes-agent`
(généralement `~/.hermes/hermes-agent`). C'est la disposition attendue par
`hermes update`, le venv managé, les dépendances chargées à la demande, le gateway
et l'outillage de la documentation.

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
cd "${HERMES_HOME:-$HOME/.hermes}/hermes-agent"
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

Solution de repli avec clone manuel (pour les clones jetables ou la CI, quand vous
ne voulez volontairement pas de la disposition d'installation managée) :

Créez le venv en dehors de l'arborescence clonée — un venv placé dans le répertoire
depuis lequel l'agent opère peut être effacé par une commande en chemin relatif que
l'agent exécute contre son propre checkout, détruisant le runtime en pleine session.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/.hermes/venvs/hermes-dev --python 3.11
source ~/.hermes/venvs/hermes-dev/bin/activate
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

---

## Communauté

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Issues](https://github.com/NousResearch/hermes-agent/issues)
- 🔌 [computer-use-linux](https://github.com/avifenesh/computer-use-linux) — Serveur MCP de contrôle du bureau Linux pour Hermes et d'autres hôtes MCP, avec arbres d'accessibilité AT-SPI, entrée Wayland/X11, captures d'écran et ciblage des fenêtres du compositeur.
- 🔌 [HermesClaw](https://github.com/AaronWong1999/hermesclaw) — Passerelle WeChat communautaire : faites tourner Hermes Agent et OpenClaw sur le même compte WeChat.

---

## Licence

MIT — voir [LICENSE](LICENSE).

Conçu par [Nous Research](https://nousresearch.com).
