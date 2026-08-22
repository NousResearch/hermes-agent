# Conversational Persona — Live-Test & Comparison Harness (CLOE / OpenClaw)

Établi pendant CLOE_PERSONA_FIDELITY_AND_GROUNDED_SYNTHESIS_REPAIR_V1 (2026-08-01).
Quand l'opérateur demande un comparatif production-vs-candidat AVEC le vrai provider,
ou un test live d'une session OpenClaw dédiée, suivre ce pattern.

## Harnais comparatif production vs candidat

Un script unique appelle les DEUX côtés avec le même provider (DeepSeek réel) :
- **old** = production HTTP : `POST http://127.0.0.1:4000/chat/completions`, header
  `authorization: Bearer <ADMIN_API_KEY>` (clé lue depuis le conteneur).
- **candidate** = `buildNativeChatCompletions` local avec la stack knowledge
  (registre + gateway unifié + indexer + seed pilot-corpus).

Règles dures du harnais (chacune a coûté un cycle de debug) :
1. **Clé admin en ARGV, jamais en env/source.** `source /tmp/x.env && node ...` ne
   propage PAS la variable au subprocess node, et `$(cat /tmp/token.txt)` dans une
   ligne bash est remplacé par `***` par le terminal Hermes → syntaxe cassée.
   Fix : `node script.mjs "$KEY"` (argv) ou lire la clé via `readFileSync` dans le script.
2. **Reproduire le vrai client** : inclure le system prompt persona
   (`getStableBlock().systemPrompt`) comme premier message système. Sans lui, le LLM
   n'a PAS la persona sur les chemins non-sociaux → réponses en « vous » et génériques.
3. **Prompts sociaux : PAS de tools.** Prompts factuels : tools fournis mais
   `tool_choice: 'none'`. Avec `tool_choice: 'auto'` + tools sur TOUT, DeepSeek appelle
   l'outil exec → `message.content = null` → artefacts `candidate: ""` (le défaut E).
4. **Source of truth runtime** : `seedPilotCorpus` porte un snapshot de déploiement
   STATIQUE et obsolète (ex. commit 3ac856d du 30/07). Surcharger avec
   `indexer.applyEvent('DEPLOYMENT_COMPLETED', { commit: CLOE_LIVE_GIT_COMMIT, ... })`
   où CLOE_LIVE_GIT_COMMIT = le commit réellement déployé. Sinon le candidat annonce
   l'ancien commit comme runtime actif (défaut « commit 302ed75 vs 3ac856d »).
5. **Capturer les réponses COMPLÈTES** (pas de slice à 600 chars) pour le gate humain.

## Pitfall terminal Hermes + secrets (générique)

- `$(cat /tmp/secret.txt)` dans une commande bash → le terminal masque la valeur par
  `***` et la commande devient syntaxiquement invalide.
- `source /tmp/var.env && node -e "..."` → les variables ne sont PAS dans `process.env`
  du node (le shell du terminal ne les hérite pas comme attendu).
- Fix : écrire un script `.sh` qui lit le fichier de secret, ou passer la clé en argv,
  ou `readFileSync` dans le script. Ne JAMAIS afficher la valeur.

## Test live d'un agent OpenClaw dédié (ex. cloe-poc)

- Un agent configuré dans `openclaw.json` (`agents.list`) a son propre workspace avec
  SOUL.md / IDENTITY.md / USER.md — la persona canonique y vit (modifiés par l'opérateur).
- **Le CLI `openclaw terminal --local` cible l'agent `main` par défaut** (workspace
  générique) — PAS l'agent dédié. Le runtime n'expose pas de flag `--agent`.
- Pour cibler un agent précis : session key `agent:<agentId>:<key>`
  (ex. `agent:cloe-poc:cloe-gate-<ts>`). `resolveAgentIdFromSessionKey` dérive l'agent
  du session key (`agent:cloe-poc:...` → cloe-poc). Vérifier dans la sortie TUI :
  « agent cloe-poc (cloe-poc) ».
- **Le TUI non-interactif ne persiste PAS la réponse** et force l'exit
  (« openclaw tui forcing process exit after return ») → le fichier JSONL de session
  n'est pas écrit. Fix : capturer via pty :
  `script -qec "node .../openclaw ... terminal --local --session 'agent:cloe-poc:<k>' --message '<prompt>' --timeout-ms 100000" out.txt`
  La réponse réelle est dans le flux TUI, entrelacée avec l'UI. L'extraire en excluant
  les lignes UI (`agent cloe-poc`, `tokens ?`, `────`, `│`, `⠸`, etc.) et en gardant
  les lignes de contenu français (regex accents).
- **Latence** : poller le fichier pty pour un marqueur UNIQUE de la NOUVELLE réponse.
  Ne pas utiliser un mot présent dans l'historique rejoué (ex. « Cedrick » apparaît
  dans la réponse du prompt 1 → fausse latence sur les prompts suivants).
- `systemPromptReport` (dans `sessions.json`) contient `workspaceDir` → confirmer que
  le workspace corrigé est bien chargé. `abortedLastRun: True` = le run TUI a été
  interrompu par timeout, pas un échec du LLM.
- Le POST `/v1/chat/completions` du gateway OpenClaw (port de contrôle, ex. 18789)
  renvoie 404 en POST (GET → 200). Le vrai canal chat est le CLI local qui route vers
  le conteneur brain (`http://127.0.0.1:4000/chat/completions`).

## Sessions privées — ne JAMAIS lire

L'opérateur a BLOQUÉ (2 fois) la lecture des fichiers de session de l'agent
(`sessions/<uuid>.jsonl`, `sessions/sessions.json`) — même pour les sessions créées
par le test. Les réponses d'un test live doivent venir UNIQUEMENT des captures TUI pty.
La relation Cloe–Cedrick est privée ; ne pas inspecter l'historique conversationnel.

## Pitfall regex JS : `\b` sur mots accentués

`\b` ne matche pas après un caractère accentué (U+00E0 non-word en regex JS).
`/^(t'es toujours là)\b/i` échoue sur « t'es toujours là ? ». Fix : terminer par
`(\b|\s|$)`.

## Pitfall apostrophes dans les fichiers de test JS

Écrire les chaînes avec des guillemets DOUBLES pour éviter le double échappement
(`\\'` → syntaxe cassée). Ne JAMAIS faire un replace global `\\'` → `'` sur un fichier
entier : cela casse les échappements légitimes des autres chaînes.

## Routage : préfixe mission sur une seule ligne → UNRESOLVED

« MISSION: CLOE_PERSONA_... » sur UNE ligne → le router renvoie UNRESOLVED
(`INSUFFICIENT_STRUCTURAL_EVIDENCE`) car le préfixe parsé est « MISSION: CLOE_... »
tout entier (la mission précédente multi-ligne « MISSION:\nCLOE_CONVERSATIONAL... »
avait résolu). Fix : fallback manuel (Step 3a) — mission suite d'une PR précédente
dans le même repo ; vérifier HEAD + présence du code concerné (ex. lah-openclaw-mvp).
