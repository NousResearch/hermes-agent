# Bounded Authenticated Read-Only API Retrieval Pattern

Établi lors de CLOE_EXOCLICK_STATS_IMPORT_AND_FINANCIAL_JOIN_V1 (2026-08-01), mission
BOUNDED_AUTHENTICATED_READ_ONLY_RETRIEVAL avec autorisation opérateur explicite.
S'applique à TOUT fournisseur API (ExoClick, et par extension tout provider) quand :
- une mission READ_ONLY doit récupérer des données qui exigent une authentification ;
- la doctrine/le gate bloque par défaut tous les appels réseau ;
- l'opérateur autorise explicitement UN accès authentifié borné SANS mutation.

## Principe central

> L'authentification n'est jamais la première action. Le PRE-AUTHORIZATION GATE prouve
> d'abord, hors-ligne et sans réseau, que seules les combinaisons méthode+chemin autorisées
> peuvent partir. Ensuite seulement on authentifie.

Séquence obligatoire :
1. Inspecter le client existant → identifier l'endpoint login EXACT et chaque endpoint GET prévu.
2. Construire l'allowlist explicite méthode+chemin (JSON, fichier artefact).
3. Implémenter un garde deny-by-default (`safeFetch`) qui rejette AVANT tout I/O réseau.
4. **Prouver le garde avec un mock fetch (zéro réseau)** : remplacer `globalThis.fetch` par un
   mock qui compte les appels ; vérifier que les requêtes non-allowlistées ne l'atteignent JAMAIS,
   et que seules les combinaisons autorisées le touchent. Compter les appels observés.
5. Vérifier que le gate de mutation (`EXOCLICK_LIVE_ENABLED`) est `false` et inchangé.
6. Seulement si 15/15 PASS : exécuter l'authentification (1 seul POST login) puis les GET bornés.

## Allowlist type (ExoClick)

```json
[
  { "method": "POST", "path": "/v2/login" },                       // SEUL POST autorisé
  { "method": "GET",  "pathPrefix": "/v2/campaigns/" },            // GET /v2/campaigns/{id}
  { "method": "GET",  "path": "/v2/statistics/a/campaign" },
  { "method": "GET",  "path": "/v2/statistics/a/site" },
  { "method": "GET",  "path": "/v2/statistics/a/zone" }
]
```
Tout le reste (GET /campaigns liste, POST /campaigns, PUT/PATCH/DELETE, /sites, /zones,
paths non canoniques) → REJETÉ avant transmission.

## Lifecycle du token (contrat sécurité)

- Lire la clé API depuis `.env` SANS l'imprimer (regex `^EXOCLICK_API_TOKEN=(.+)$`, strip quotes,
  rejeter `replace_me`/vide).
- Échange : `POST /v2/login` body `{ api_token }` → extraire access token
  (`token || access_token || session_token || data.token || data.access_token`).
- Stocker UNIQUEMENT en mémoire processus (`process.env.EXOCLICK_ACCESS_TOKEN`).
- Trace : préfixe 4 chars + longueur seulement (`aa38...(40 chars)`), jamais la valeur complète.
- Redacter le header `Authorization` dans toute trace (`[REDACTED]`).
- Détruire immédiatement après la dernière requête : `delete process.env.EXOCLICK_ACCESS_TOKEN`
  + marqueur `TOKEN_DESTROYED`.
- Ne JAMAIS écrire le token dans les logs, evidence, JSON, commits, receipts.
- Vérification finale : scan regex des artefacts pour tokens/bearer — 0 hit attendu.

## Piège : suppression de fichiers temporaires bloquée

Si l'utilisateur bloque le `rm` des fichiers temporaires (retrieval raw JSONL, harness),
NE PAS retenter ni reformuler. Vérifier passivement qu'aucun secret complet n'y traîne
(seul le préfixe 4-char est présent), documenter l'état, laisser les fichiers en place.

## Faits stats ExoClick (découverts lors de la mission)

- **GET /v2/campaigns/{id} fonctionne sur campagnes archivées** : renvoie nom, statut
  (`calculated_status.status` = "Archived", `status` = -1), `date_created`, ciblage
  (countries/categories/sites), `price` (bid Smart CPM), et **`total_impressions_sent`** (cumul réel).
  `total_budget_spent`, `total_clicks`, `total_impressions` peuvent être `null` sur archivées.
- **GET /v2/statistics/a/{campaign|site|zone}?campaignid=X&date_from=Y&date_to=Z renvoie
  `{"result":[],"request_metadata":[]}` pour des campagnes ARCHIVÉES** sur une fenêtre passée —
  pas une erreur d'auth (HTTP 200), mais absence de lignes quotidiennes. Ne pas conclure
  "échec d'authentification" : c'est une limitation de données (DATA_GAP).
- Query params canoniques : `campaignid`, `date_from`, `date_to` (format `YYYY-MM-DD`).
- Zones dans la réponse campagne : `[{idcampaign, idzone, idsite, price, sub_id_target_type, sub_ids}]`
  — les `idzone`/`idsite` internes ExoClick (ex. 158305) sont DIFFÉRENTS des `zone_id`/`site_id`
  du clickstream LAH (ex. 5870652, issus des macros `{zone_id}` du tracking URL). Le mapping
  exact nécessite les stats par zone ou une table de mapping — ne pas les confondre.
- **Conflit de tracking détectable depuis les métadonnées seules** : une campagne peut avoir
  `sites.blocked: [{url: "bokep.porn"}, {url: "comicspornoxxx.com"}]` alors que le clickstream
  montre du trafic REÇU de ces domaines sur la même campagne. Classification : blocage ajouté
  après la période de trafic OU blocage inefficace → `TRACKING_DEFECT` / `UNRESOLVED`,
  zones contaminées → REVIEW_MANUALLY, jamais ALLOW.

## Verdicts

- **BLOCKED** quand l'authentification est interdite par la mission (POST /login prohibé) :
  documenter le blocage, produire les artefacts avec nulls explicites, toutes zones
  INSUFFICIENT_FINANCIAL_EVIDENCE. Ne pas ouvrir le gate.
- **CERTIFIED_WITH_LIMITATIONS** quand l'auth read-only a réussi ET les métadonnées sont jointes
  mais les stats quotidiennes restent vides : limitations documentées, décisions CAPPED/REVIEW
  sur l'engagement seul, jamais ALLOW_INITIAL/BLOCK_INITIAL sans coût.
- Une zone n'est ALLOW_CAPPED que si engagement fort + pas de conflit de tracking ; échantillon
  < 20 sessions → cap plus strict (ex. $25 au lieu de $50) + `sample_sufficient=false`.

## Validation finale (checks à prouver)

1. Un seul POST login ; 2. tous les autres appels GET ; 3. chaque path allowlisté ;
4. aucun endpoint mutation contacté ; 5. gate LIVE inchangé ; 6. token jamais loggé ;
7. token détruit ; 8. campagnes/IDs bornés ; 9. dates bornées ; 10. aucune conclusion
financière non supportée (spend/CPM/CPC/ROI nulls sans données) ; 11. aucun secret/URL complète ;
12. counts cohérents.
