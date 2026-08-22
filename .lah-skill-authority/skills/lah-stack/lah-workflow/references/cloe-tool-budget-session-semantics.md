# CLOE Tool Budget — Session Semantics & "Fresh Session" Trap

Établi lors de HERMES_CLOE_FRESH_SESSION_TOOL_BUDGET_RESET_DIAG_V1 (2026-08-07,
READ_ONLY_AUDIT certifié). Diagnostique pourquoi une "fresh session" Cloé répond
immédiatement `TOOL_BUDGET_EXHAUSTED: YES` sans lancer aucun outil.

## Architecture du budget (conversation-tool-governor.js)

- Feature: CLOE_RETRIEVAL_FIRST_TOOL_BUDGET_AND_PROGRESSIVE_REPLY_V1.
- Le governor est **PUR** : `countToolUsage(messages)` compte les tool calls dans
  l'**historique cumulé rejoué** à chaque tour par le runtime. Il n'y a AUCUN
  compteur persistant, aucun reset par prompt, aucune notion "par tour".
- Budgets (`TOOL_BUDGETS`) :
  - ordinary: `max_total_tool_calls: 4`, `max_repo_searches: 2`,
    `max_empty_or_failed: 1`, `max_repeated_semantic_target: 1`,
    `max_tool_phase_ms: 20000`, `stop_at_ratio: 0.75`
  - diagnostic: `max_total_tool_calls: 16` (+8/4/4/120000)
  - `resolveBudget(missionType)` : `'diagnostic'|'implementation'` → 16, sinon → 4.
- `server.js` route `/chat/completions` (native tool path) hardcode
  `missionType: 'conversational'` → **le budget diagnostic (16) ne s'applique
  JAMAIS aux requêtes dashboard** (vérifié déployé server.js:428, SHA 0908681).
- Quand `budgetCheck.exceeded` → `injectSynthesisDirective` préfixe un message
  système "TOOL POLICY ENFORCEMENT ... Do NOT request any further tool calls.
  Synthesize your answer from the facts already established." (chat-completions-service.js
  lignes ~396-407, 718-722) + `effectiveTools=undefined`, `tool_choice='none'`.
  L'LLM synthétise alors depuis les faits déjà établis.

## Le marqueur `TOOL_BUDGET_EXHAUSTED: YES`

- La chaîne littérale `TOOL_BUDGET_EXHAUSTED` n'existe **NULLE PART** dans le
  code (vérifié : repo canonique, worktree déployé, workspace clone,
  ~/.hermes/openclaw/, npm openclaw/dist, hermes-agent, lah-stack-tools,
  cloe-diagnostic-orchestrator).
- C'est un **rapport LLM** de `governor.budget.exceeded=true`, souvent demandé
  explicitement par le format de sortie de la mission (ex. "Termine par
  TOOL_BUDGET_EXHAUSTED: YES | NO").
- Ne pas chercher un bug de string : vérifier l'état du governor dans la session.

## LE piège : "fresh session" ≠ nouvelle session runtime

Symptôme : l'opérateur envoie un prompt minimal ("Fresh session. Ta seule
mission est de récupérer ...") et Cloé répond immédiatement budget épuisé.

Root cause : le prompt est parti dans la **MÊME session runtime** (même session
key, même fichier .jsonl) qui avait déjà consommé son budget au tour 1 (ex. un
batch parallèle de 4 `memory_search` en un seul tour = 4/4 ordinary).

Timeline réelle observée (session `dashboard:92c7947f` → fichier 9ac8c815) :
1. 20:36:34 — prompt réconciliation complet (7 catégories)
2. 20:36:40 — assistant → **4 appels `memory_search` en parallèle (1 tour)** = 4/4
3. 20:38:56 — 2e prompt (même session) → 20:39:42 "budget d'outils épuisé", 0 appel
4. 20:42:20 — prompt minimal "Fresh session" (même session key !)
5. 20:42:46 — "cette session est déjà à budget d'outils épuisé" → YES

Le texte "Fresh session" dans le prompt ne reset RIEN. Le budget est dérivé de
l'historique du fichier de session. Un reset nécessite un **NOUVEAU session key**
(/new, ou nouveau chat dashboard) qui crée un fichier .jsonl vide.

## Règles de diagnostic

1. Identifier le session key + fichier de la session affectée via
   `~/.openclaw/agents/<agent>/sessions/sessions.json` (le registre = métadonnées,
   lisible sans consentement ; les transcripts .jsonl = privés, consentement opérateur).
2. Lire le .jsonl : compter les `"type":"toolCall"` par message assistant et les
   `"role":"toolResult"` — un seul message assistant peut porter 4+ toolCalls.
3. Vérifier que les prompts successifs partagent le MÊME sessionId (pas de
   nouveau fichier créé entre eux).
4. Vérifier `parentSessionKey` / `forkedFromParent` : un fork `writeForkHeaderOnly`
   produit un transcript enfant SANS `parentSession` dans le header et SANS appels
   hérités — l'épuisement vient alors des propres tool calls de l'enfant, pas de l'héritage.
5. Trajectoire (`*.trajectory.jsonl`) : `session.started`/`model.completed` par run
   pour prouver la chronologie multi-prompt dans la même session.
6. Vérifier que le marqueur est bien LLM-reporté (cf. ci-dessus) avant de chercher un bug.

## Déblocage opérateur (sans code)

- Nouveau chat / `/new` → nouveau session key → budget 0/4.
- Envoyer le prompt minimal EN PREMIER, avant toute mission de réconciliation.

## Fix candidats (mission CODE_CHANGE, décision opérateur)

- Permettre `missionType: 'diagnostic'` (16 calls) pour les prompts
  retrieval/réconciliation (ex. override par header ou détection de préfixe).
- OU plafonner les appels parallèles par tour (max 2) pour qu'un seul batch ne
  puisse pas épuiser la session.
- UX : exposer `governor.budget.usage` restant dans le dashboard.

## Fichiers source (SHA déployé 0908681, canonique b65cc67)

- `lah-openclaw-mvp/src/services/conversation-tool-governor.js`
- `lah-openclaw-mvp/src/services/chat-completions-service.js`
- `lah-openclaw-mvp/src/server.js` (route /chat/completions, missionType hardcodé)
