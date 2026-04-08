<p align="center">
  <img src="assets/banner.png" alt="Hermes Agent" width="100%">
</p>

# Hermes Agent ☤

<p align="center">
  <a href="https://hermes-agent.nousresearch.com/docs/"><img src="https://img.shields.io/badge/Docs-hermes--agent.nousresearch.com-FFD700?style=for-the-badge" alt="Documentation"></a>
  <a href="https://discord.gg/NousResearch"><img src="https://img.shields.io/badge/Discord-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord"></a>
  <a href="https://github.com/NousResearch/hermes-agent/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
  <a href="https://nousresearch.com"><img src="https://img.shields.io/badge/Built%20by-Nous%20Research-blueviolet?style=for-the-badge" alt="Built by Nous Research"></a>
</p>

**L'agent IA auto-améliorant construit par [Nous Research](https://nousresearch.com).** C'est le seul agent doté d'une boucle d'apprentissage intégrée — il crée des compétences à partir de l'expérience, les améliore en cours d'utilisation, s'incite à conserver ses connaissances, recherche ses propres conversations passées et construit un modèle approfondi de qui vous êtes au fil des sessions. Exécutez-le sur un VPS à 5 $, un cluster GPU ou une infrastructure serverless qui ne coûte presque rien lorsqu'elle est inactive. Il n'est pas lié à votre ordinateur portable — parlez-lui depuis Telegram pendant qu'il travaille sur une VM cloud.

Utilisez le modèle de votre choix — [Nous Portal](https://portal.nousresearch.com), [OpenRouter](https://openrouter.ai) (plus de 200 modèles), [z.ai/GLM](https://z.ai), [Kimi/Moonshot](https://platform.moonshot.ai), [MiniMax](https://www.minimax.io), OpenAI, ou votre propre endpoint. Changez avec `hermes model` — pas de changement de code, pas d'enfermement propriétaire.

<table>
<tr><td><b>Une véritable interface terminal</b></td><td>TUI complet avec édition multiligne, saisie semi-automatique des commandes, historique des conversations, interruption et redirection, et sortie d'outil en streaming.</td></tr>
<tr><td><b>Vit là où vous vivez</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal et CLI — le tout à partir d'un seul processus passerelle. Transcription de mémos vocaux, continuité de conversation multiplateforme.</td></tr>
<tr><td><b>Une boucle d'apprentissage fermée</b></td><td>Mémoire organisée par l'agent avec des rappels périodiques. Création de compétences autonomes après des tâches complexes. Les compétences s'améliorent d'elles-mêmes pendant l'utilisation. Recherche de session FTS5 avec résumé LLM pour le rappel entre sessions. Modélisation de l'utilisateur dialectique <a href="https://github.com/plastic-labs/honcho">Honcho</a>. Compatible avec le standard ouvert <a href="https://agentskills.io">agentskills.io</a>.</td></tr>
<tr><td><b>Automatisations planifiées</b></td><td>Planificateur cron intégré avec livraison sur n'importe quelle plateforme. Rapports quotidiens, sauvegardes nocturnes, audits hebdomadaires — le tout en langage naturel, fonctionnant sans surveillance.</td></tr>
<tr><td><b>Délègue et parallélise</b></td><td>Créez des sous-agents isolés pour des flux de travail parallèles. Écrivez des scripts Python qui appellent des outils via RPC, transformant les pipelines multi-étapes en tours à coût de contexte nul.</td></tr>
<tr><td><b>S'exécute n'importe où</b></td><td>Six backends de terminal — local, Docker, SSH, Daytona, Singularity et Modal. Daytona et Modal offrent une persistance serverless — l'environnement de votre agent hiberne lorsqu'il est inactif et se réveille à la demande, ne coûtant presque rien entre les sessions.</td></tr>
<tr><td><b>Prêt pour la recherche</b></td><td>Génération de trajectoires par lots, environnements Atropos RL, compression de trajectoires pour l'entraînement de la prochaine génération de modèles d'appel d'outils.</td></tr>
</table>

---

## Installation Rapide

```bash
curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh | bash
```

Fonctionne sur Linux, macOS et WSL2. L'installateur s'occupe de tout — Python, Node.js, dépendances et la commande `hermes`. Aucun prérequis sauf git.

> **Windows:** Windows natif n'est pas supporté. Veuillez installer [WSL2](https://learn.microsoft.com/fr-fr/windows/wsl/install) et exécuter la commande ci-dessus.

Après l'installation :

```bash
source ~/.bashrc    # recharger le shell (ou : source ~/.zshrc)
hermes              # commencez à discuter !
```

---

## Mise en Route

### 🖥️ Tableau de Bord Web (Idéal pour les Débutants)
Si vous préférez une interface visuelle plutôt que le terminal, Hermes inclut désormais un tableau de bord Web.

**Pour le démarrer :**
- **Windows :** Double-cliquez sur `start.bat` (Nécessite Python installé sur Windows)
- **Linux/macOS :** Exécutez `./start.sh` ou `hermes dashboard`

Cela ouvrira une interface de chat moderne dans votre navigateur avec un support complet pour :
- 🌗 **Thèmes Sombre/Clair**
- 🇫🇷 **Langues Française & Anglaise**
- 📜 **Historique des Sessions**
- 💡 **Infobulles pour les fonctions**

### ⌨️ Utilisation de la CLI
```bash
hermes              # CLI Interactive — démarrer une conversation
hermes dashboard    # Démarrer le tableau de bord Web
hermes model        # Choisir votre fournisseur LLM et votre modèle
hermes tools        # Configurer les outils activés
hermes config set   # Définir des valeurs de configuration individuelles
hermes gateway      # Démarrer la passerelle de messagerie (Telegram, Discord, etc.)
hermes setup        # Exécuter l'assistant de configuration complet
hermes update       # Mettre à jour vers la dernière version
hermes doctor       # Diagnostiquer les problèmes éventuels
```

📖 **[Documentation complète (EN) →](https://hermes-agent.nousresearch.com/docs/)**

---

## Communauté

- 💬 [Discord](https://discord.gg/NousResearch)
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Issues](https://github.com/NousResearch/hermes-agent/issues)
- 💡 [Discussions](https://github.com/NousResearch/hermes-agent/discussions)

---

## Licence

MIT — voir [LICENSE](LICENSE).

Construit par [Nous Research](https://nousresearch.com).
