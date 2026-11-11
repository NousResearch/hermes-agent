<p align="center">
  <a href="https://unitalk.ai">
    <img src="assets/logo.png" alt="Logo Unitalk" width="120" height="120">
  </a>
</p>

<h1 align="center">Unitalk Desktop ☤</h1>

<p align="center">
  <b>Le Collaborateur IA qui grandit avec votre entreprise. Propulsé par Hermes.</b>
</p>

<p align="center">
  <a href="https://unitalk.ai"><img src="https://img.shields.io/badge/Site%20Web-unitalk.ai-0066FF?style=for-the-badge" alt="Site Web"></a>
  <a href="https://github.com/unitalkai/unitalk-desktop/blob/main/LICENSE"><img src="https://img.shields.io/badge/Licence-MIT-green?style=for-the-badge" alt="Licence: MIT"></a>
  <a href="https://github.com/NousResearch/hermes-agent"><img src="https://img.shields.io/badge/Moteur-Hermes%20Agent-blueviolet?style=for-the-badge" alt="Propulsé par Hermes"></a>
  <a href="https://unitalk.ai/home/marketplace/profils-metier"><img src="https://img.shields.io/badge/Store-Expertises%20M%C3%A9tier-FFD700?style=for-the-badge" alt="Marketplace"></a>
</p>

<p align="center">
  <a href="README.md"><img src="https://img.shields.io/badge/Lang-English-blue?style=for-the-badge" alt="English"></a>
  <a href="README.fr.md"><img src="https://img.shields.io/badge/Lang-Fran%C3%A7ais-0055A5?style=for-the-badge" alt="Français"></a>
  <a href="README.es.md"><img src="https://img.shields.io/badge/Lang-Espa%C3%B1ol-orange?style=for-the-badge" alt="Español"></a>
  <a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-%E4%B8%AD%E6%96%87-red?style=for-the-badge" alt="中文"></a>
  <a href="README.ur-pk.md"><img src="https://img.shields.io/badge/Lang-%D8%A7%D8%B1%D8%AF%D9%88-green?style=for-the-badge" alt="اردو"></a>
</p>

---

## Vue d'ensemble

**Unitalk Desktop** est la distribution professionnelle officielle de [Hermes Agent](https://github.com/NousResearch/hermes-agent) développé par [Nous Research](https://nousresearch.com).

Alors que le moteur Hermes d'origine s'adresse principalement aux développeurs via des lignes de commande en terminal, **Unitalk Desktop** packagera toute cette puissance dans une application installable en 1 clic (`.exe` et `.dmg`), conçue pour les entreprises, les dirigeants et les équipes métiers.

### Proposition de valeur

| Besoin de votre entreprise | Hermes Agent Vanilla | Unitalk Desktop |
| :--- | :--- | :--- |
| **Prise en main** | Nécessite des commandes terminal `curl \| bash` | **Installeurs 1-clic pour Windows (`.exe`) et macOS (`.dmg`)** |
| **Identité** | Prompt anonyme en ligne de commande | **Collaborateur incarné portant le prénom choisi par l'entreprise** |
| **Multimodal** | Sous-agents fragmentés | **Modules natifs intégrés : Chat, Images, Vidéos, Meetings** |
| **Authentification** | Monoutilisateur local non connecté | **Connexion professionnelle simplifiée (Google Workspace, M365, Email)** |
| **Routage IA & Facturation** | Gestion manuelle de clés API multiples | **Unitalk AI Gateway (LiteLLM) avec 10 000 crédits d'essai offerts (5 €)** |
| **Compétences Métier** | Rédaction manuelle de prompts | **Store d'expertises gratuites ([`unitalk.ai/home/marketplace/`](https://unitalk.ai/home/marketplace/profils-metier))** |
| **Propriété des données** | Processus poste de travail éphémère | **Actif durable d'entreprise (mémoire Honcho, bases Qdrant)** |

---

## Capacités clés (Phase 1)

1. **Identité incarnée unique** :
   - Votre Collaborateur IA porte un prénom unique choisi par votre entreprise (*Emma*, *Thomas*, etc.).
   - Pas de démultiplication de personas : les compétences spécialisées sont gérées sous forme de capacités natives (**Images**, **Vidéos**, **Meetings**).

2. **Passerelle IA préconfigurée (AI Gateway)** :
   - Accès aux meilleurs modèles du marché (Claude 3.7 Sonnet, Gemini 2.5 Flash, GPT-4o, modèles open-source) via une passerelle unifiée LiteLLM.
   - Chaque installation comprend **10 000 crédits d'essai offerts (5 €)** pour tester immédiatement sans carte bancaire.

3. **Connexion d'entreprise fluide (Clerk SSO)** :
   - Connexion en 1 clic avec votre compte professionnel (Google Workspace, Microsoft 365 ou email d'entreprise) reliant le Collaborateur à votre organisation.

4. **Store d'expertises métier gratuites** :
   - Enrichissez votre Collaborateur en 1 clic depuis le [Marketplace](https://unitalk.ai/home/marketplace/profils-metier) :
     - **Commercial & Ventes** : SDR, Ingénieur d'affaires, Directeur commercial, Responsable grands comptes.
     - **Organisation & Support** : Assistante de direction, Gestionnaire administratif.
     - **Direction & Stratégie** : Bras droit DG, Veille stratégique et concurrentielle.

---

## Téléchargement & Installation

### Pour les utilisateurs & clients

Le téléchargement en 1 clic est disponible sur le portail officiel :

👉 **[Télécharger Unitalk Desktop pour Windows & macOS](https://unitalk.ai/download)**

- **Windows** : Téléchargez et lancez `Unitalk Setup.exe`.
- **macOS** : Téléchargez `Unitalk.dmg`, glissez l'application dans Applications et lancez.

---

### Pour les développeurs & contributeurs

Pour exécuter et compiler l'application en local depuis les sources :

```bash
# 1. Cloner le dépôt
git clone https://github.com/unitalkai/unitalk-desktop.git
cd unitalk-desktop

# 2. Basculer sur la branche de distribution
git checkout distribution/unitalk

# 3. Installer les dépendances
pnpm install

# 4. Lancer l'application en mode développement
pnpm --filter @hermes/desktop dev
```

---

## Suivi Amont (Upstream) & Licence

Unitalk Desktop est une distribution aval de [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent).

- **Modèle de distribution propre** : Nous ne modifions pas le cœur d'exécution Python de l'agent. Les mises à jour de Nous Research sont fusionnées en continu via `upstream/main`.
- **Licence** : Publié sous licence **MIT**. Consultez le fichier [LICENSE](LICENSE) pour les mentions légales d'origine.

---

*Développé avec ❤️ par [Unitalk](https://unitalk.ai) · Moteur propulsé par [Hermes Agent](https://github.com/NousResearch/hermes-agent) de Nous Research.*
