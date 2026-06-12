# jb_outbound — « rien ne part sans accord » (Jean-Billie)

Greffe Jean-Billie sur **Hermes Agent** (Nous Research, MIT). Un seul plugin + **un point de
greffe optionnel dans le scheduler cron** (`cron/scheduler.py::run_job`, no-op sans le plugin)
→ suivi de l'upstream trivial.

## Ce que ça fait

Tout envoi sortant que l'assistant tente — message **Telegram** (`send_message`) ou **email / réseau
social** via **Composio** (`mcp_composio_*`) — est **intercepté** et transformé en **proposition à
valider**. L'envoi réel n'a lieu qu'**après l'accord du client**.

Les deux familles d'envoi passent par le même `tool_execution` middleware (Hermes) → un seul point
de greffe couvre tous les canaux et tous les déclencheurs (cron, chat, Telegram, email entrant).

## Boucle

```
outil d'envoi appelé
   → middleware : court-circuit (l'outil NE s'exécute PAS)
   → enregistre l'envoi (args complets) dans ~/.hermes/jb_pending/{jb_id}.json   (local, jamais relayé)
   → POST DraftRequest → http://127.0.0.1:8442/v1/draft   (daemon → proposition « pending »)
   → rend au modèle : « préparé, rien ne part tant que ce n'est pas validé »

[client valide dans son espace]

   → daemon pousse la DecisionItem → http://127.0.0.1:8444/jb/decision   (listener du plugin)
   → replay : registry.dispatch(tool_name, args)   (envoi RÉEL — ne repasse pas par le middleware)
   → POST ResultRequest {id, executed|failed} → http://127.0.0.1:8442/v1/result
```

## Attribution (départements) & fil d'activité

Au lancement d'un job cron, le scheduler pose le **contexte d'attribution** du job
(`job_context.py`, ContextVar — les jobs tournent dans des threads du gateway) : casquette lue
dans le front-matter du skill du job (`casquette:` pour les skills gold, `department:` pour les
customs), id du skill, id du job.

- **Stamp des drafts** : tout DraftRequest émis pendant un job porte les champs additifs de
  premier niveau `department`, `skill_id`, `job_id` (omis hors contexte job — chat libre). Le
  daemon ignore les champs inconnus tant que le contrat Go n'est pas étendu (vague 2).
- **Fil d'activité** (`activity.py`) : au début et à la fin de chaque job cron, POST
  fire-and-forget `http://{JB_DRAFT_ADDR}/v1/activity` avec
  `{phase: "started"|"finished", status: "ok"|"error", department?, skill_id?, job_id?, label?}`
  (`label` = nom lisible du job). **Gated par `JB_ACTIVITY_EVENTS=1`** (défaut OFF — la route
  daemon n'existe pas encore). Timeout 2 s, échecs avalés : ne bloque jamais un job.

## Règles

- **Fail-closed** : un outil d'envoi composio non répertorié est **bloqué** (jamais auto-envoyé). On
  élargit les listes dans `classify.py` au besoin.
- **Asynchrone** : l'envoi est rejoué hors du run d'agent (le store survit au redémarrage,
  idempotent sur `jb_id`). Pas de blocage du run en attendant la validation humaine.
- **Minimisation** : les arguments complets (corps, destinataire détaillé) restent **locaux**
  (`jb_pending/`). Le `DraftRequest` ne porte que kind / titre / aperçu / destinataire d'affichage.
- **Loopback only** : le listener bind strictement `127.0.0.1` (garde-fou symétrique du daemon).

## Activation

Opt-in via la config Hermes :

```yaml
plugins:
  enabled: [jb_outbound]
```

Endpoints lus dans l'environnement (posés par le bundle Jean-Billie / le `daemon.env`) :
`JB_DRAFT_ADDR` (défaut `127.0.0.1:8442`), `JB_DECISION_PUSH_URL` (défaut
`http://127.0.0.1:8444/jb/decision`), `JB_ACTIVITY_EVENTS` (`1` pour activer le fil d'activité,
défaut OFF). Sans `JB_DECISION_PUSH_URL`, le plugin reste **passif**.

## Tests

`python -m pytest plugins/jb_outbound/` — autonome (mocke le HTTP loopback et le registre
d'outils, n'a pas besoin d'un environnement Hermes complet). Sous Windows :
`pytest -o addopts=""` (pytest-timeout/SIGALRM indisponible).
