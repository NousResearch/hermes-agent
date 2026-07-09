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

## Outil « creer_support » (Ma marque, Option A)

Le plugin enregistre aussi un **outil** (`ctx.register_tool`, zéro patch du cœur) : `creer_support`.
Quand le client demande un support (« fais-moi un carrousel », un devis, une présentation…), l'agent
émet une INTENTION structurée `{ type, contenu }` — il ne dessine jamais lui-même. Le POST part sur
le **loopback du daemon** (`http://{JB_DRAFT_ADDR}/v1/produce`), qui le relaie au control-plane avec
son identité mTLS (même chemin que les drafts / `request_tool_connection`) ; la plateforme rend le
support de façon **déterministe** (gabarits fixes, charte du client) et le range dans l'Espace
Documents. L'URL signée revient à l'agent, qui la partage **telle quelle** dans la conversation.

- 9 familles (enum fermé) : `presentation`, `devis`, `facture`, `post`, `carrousel`, `story`,
  `prospectus`, `signature`, `lettre` — contenu re-validé/borné côté plateforme.
- **Purement interne** : le document est déposé chez le client, rien ne part vers un tiers.
  L'ENVOI ultérieur du document repasse par la boucle de proposition ci-dessus.
- **Gated** comme le reste du plugin (`JB_DECISION_PUSH_URL`) : hors box Jean-Billie, l'outil est
  invisible (`check_fn`). Relais indisponible → message franc, jamais de bluff.
- Toolset plugin `jb_studio` (activé par défaut sur toutes les plateformes, désactivable via
  `hermes tools`).

## Outil « request_tool_connection » (demander un outil manquant) — greffe F2

Vivait dans `tools/request_tool_connection.py` (fichier additif du cœur) ; depuis F2 (2026-07-09)
le module vit ICI (`request_connection.py`) et s'enregistre par le même seam (`ctx.register_tool`).
Quand il MANQUE un outil pour accomplir une demande, l'agent envoie une INTENTION en langage
naturel au daemon loopback (`/v1/request-connection`) ; le control-plane répond en white-label
(souvent un lien de branchement self-service que le CLIENT clique lui-même — rien ne part vers un
tiers).

- Toolset **« messaging » conservé à l'identique** : c'est l'entrée explicite de l'allowlist
  `platform_toolsets` émise par le bundle (lane S, monorepo) qui expose l'outil — toolset
  REGISTRE (aucune entrée statique dans `toolsets.py`), résolu dynamiquement. Ne pas le renommer
  sans synchroniser le config-generator du monorepo.
- Gated `JB_DECISION_PUSH_URL` (`check_fn`) ; daemon injoignable → réponse franche `unavailable`.

## Aux task « goal_judge » (juge de mission) — greffe F2

Le juge DONE/CONTINUE des missions de fond est **natif** (`hermes_cli/goals.py`) ; seule sa
CONFIG l'était par un bloc `DEFAULT_CONFIG.auxiliary.goal_judge` patché dans le cœur. Depuis F2,
le plugin la déclare via `ctx.register_auxiliary_task("goal_judge", defaults={…})` : le pont natif
fusionne ces defaults SOUS `config.yaml auxiliary.goal_judge` (l'opérateur garde la main), et la
tâche apparaît dans le picker « Configure auxiliary models ». Defaults = valeurs neutres alignées
sur les fallbacks natifs (`provider: auto`, `max_tokens: 4096`, `timeout: 30`) — comportement
identique avec ou sans plugin.

## Rate-limit « burst » (deux verrous, défaut OFF)

Deux garde-fous de débit **additifs**, activés par `JB_RATE_LIMIT_ENABLED=1` (défaut OFF → comportement
strictement identique à avant). Le rate-limit n'**ouvre jamais** un envoi : il n'ajoute qu'un **refus
propre**. `jb_outbound` continue de PROPOSER — rien ne part sans accord.

- **Egress** : un token-bucket par instance, posé dans `middleware.py` JUSTE AVANT le POST réel de la
  proposition au daemon. Au dépassement, l'outil renvoie `status="rate_limited"` (même forme que
  `queued_for_approval`), l'outil réel **ne s'exécute pas** et rien n'est déposé.
- **Tours / re-drive** : même mécanisme (famille `turns`) au seam de re-drive autonome de la flotte
  (`gateway/run.py::_watchdog_sweep`, AVANT `adapter.handle_message`). Au dépassement, le tour bloqué
  n'est **pas** relancé pour ce balayage (jamais de file ni de différé).

L'instance est identifiée par `job_id` / `department` (job cron, via `job_context`), sinon
`HERMES_SESSION_ID`, sinon `"default"` ; côté re-drive, c'est le `conversationId` de la mission.
L'état (jetons restants) persiste en **JSON atomique** sous `<HERMES_HOME>/jb_rate_limit/buckets.json`
pour rester correct **cross-process** (CLI / gateway / cron sont des process distincts). **Horloge
injectable** côté code → tests déterministes.

| Variable | Défaut | Rôle |
|---|---|---|
| `JB_RATE_LIMIT_ENABLED` | _(absent)_ | `1` pour activer les deux verrous (sinon tout passe comme avant). |
| `JB_RATE_LIMIT_EGRESS_RPM` | `30` | Envois sortants autorisés/min par instance. |
| `JB_RATE_LIMIT_TURNS_RPM` | `20` | Tours re-drivés autorisés/min par mission. |
| `JB_RATE_LIMIT_BURST_RATIO` | `1.5` | Capacité du seau = `rpm × ratio` (absorbe une rafale ponctuelle). |

**Défauts généreux par dessein** : un usage sain = quelques egress et une dizaine de tours étalés sur
plusieurs minutes ; ces seuils ne mordent donc qu'un emballement (boucle qui spamme, re-drive en
tempête), jamais le fonctionnement normal.

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
défaut OFF), `JB_RATE_LIMIT_ENABLED` + `JB_RATE_LIMIT_*` (rate-limit « burst », défaut OFF — voir
section dédiée). Sans `JB_DECISION_PUSH_URL`, le plugin reste **passif**.

## Tests

`python -m pytest plugins/jb_outbound/` — autonome (mocke le HTTP loopback et le registre
d'outils, n'a pas besoin d'un environnement Hermes complet). Sous Windows :
`pytest -o addopts=""` (pytest-timeout/SIGALRM indisponible).
