# Modèle de menace — Hermes Research Protocol PR 0

## Portée et posture

PR 0 ne traite aucune donnée et n’exécute aucun handler. Ce modèle de menace
gèle les contrôles requis avant l’ajout du runtime. Les sources externes, les
instructions du modèle et les chemins d’entrée sont non fiables. La posture
par défaut est fail-closed : une validation absente, ambiguë ou incohérente
refuse l’opération.

## Acteurs, actifs et frontières

Les acteurs sont : utilisateur, planner, approval service, worker, DB,
stockage et source externe. Le planner et le worker sont séparés : le planner
prépare un artefact ; seul un worker peut réclamer une exécution autorisée.

Les actifs sont : plan, hash, approbation, credentials, manifest, evidence et
reports. Le stockage est la frontière de persistance ; la DB est une source de
contexte typé ; la source externe est une entrée non fiable. Les credentials
sont secrets et ne doivent être exposés dans aucun artefact ou journal.

## Menaces et contrôles obligatoires

| Menace | Scénario | Contrôle gelé |
|---|---|---|
| Prompt injection | Une source externe ou un document tente de donner des instructions au planner/worker | Traiter le contenu comme donnée ; aucune source ne devient autorité ; capacités et sorties restent bornées par le plan approuvé |
| Path traversal | Un modèle fournit `../`, un chemin absolu ou un chemin hors root | Ne jamais accepter `path` pour l’écriture ; dériver depuis des identifiants validés, refuser `..` et chemins absolus, vérifier `realpath` |
| Symlink | Un composant du chemin pointe vers une cible inattendue | Opérer relativement à des descripteurs de répertoires de confiance, sans suivre les symlinks, et vérifier la destination réelle |
| TOCTOU | La cible change entre vérification et écriture | Réservation exclusive, temporaire dans le même filesystem, publication atomique sans remplacement (`RENAME_NOREPLACE` ou équivalent), `fsync` du fichier et du répertoire parent, puis relecture et recalcul du hash |
| Approval replay | Un reçu valide est réutilisé ou consommé au-delà de sa portée | `approval_id` imprévisible, expiration, compteur de consommation et réclamation atomique avec refus au-delà du cap |
| Confused deputy | Un worker ou plugin utilise une autorité pour une autre capacité ou un autre run | Portée exacte incluant `capability`, `run_id`, hashes, budgets, durée, exécutions et droits externes ; comparaison stricte |
| SQL injection | Le modèle injecte du SQL ou modifie une requête de contexte | `query_id` allowlisté et paramètres validés ; mapping versionné ; jamais de SQL en entrée |
| SSRF | Une source ou un paramètre force une requête vers un réseau interne | Les sources et destinations sont allowlistées dans une capacité dédiée ; pas de routage réseau implicite ; refus par défaut |
| Exfiltration | Credentials ou données privées fuient vers evidence, reports ou une cible externe | Classification des secrets, filtrage avant persistance/sortie, logs sans DSN ni champs secrets, publication externe jamais auto-routée |
| Partial output | Un artefact ou rapport incomplet est présenté comme valide | Écritures atomiques, reçus avec longueur/hash, relecture de vérification et état explicite ; pas de consommation sans artefact complet |

## Invariants de sécurité à vérifier

Les invariants exacts à préserver sont ceux des sections 3.1 à 3.4 de
l’architecture :

1. Le hash est celui des octets canoniques effectivement persistés, et toute
   réécriture approuvée est refusée.
2. Une approbation porte sur un hash et une portée exacts ; toute divergence,
   expiration, absence ou réutilisation hors cap est refusée atomiquement.
3. La lecture DB passe par un `query_id` allowlisté, une requête paramétrée
   versionnée, un rôle `planner_reader` en lecture seule, une transaction
   `READ ONLY` et des plafonds de temps/lignes/octets.
4. Chaque capacité est isolée par toolset, profil, skill requis et `check_fn` ;
   une capacité indisponible est rejetée explicitement.
5. Toute publication externe exige une approbation humaine explicite pour
   l'opération exacte. Cette approbation n'est jamais accordée automatiquement,
   y compris sous `/yolo` ou avec `approvals.mode: off`.

## Publication externe

La capacité `publish` est manuelle et désactivée initialement. Une publication
externe ne peut **jamais** être auto-routée vers une cible externe ni être
auto-approuvée sous `/yolo` ou `approvals.mode: off`. Le routage automatique ne
peut produire qu’un refus ou un artefact local en attente d’une approbation et
d’une action humaines explicites ; il ne peut pas transformer un rapport en
envoi externe.
