# Architecture et contrats gelés — PR 0

## Statut

Ce document est l’ADR de la baseline. PR 0 ne contient aucun comportement
runtime : pas d’outil, pas de handler, pas de routeur, pas de stockage, pas de
worker et pas d’accès réseau. Toute implémentation ultérieure doit respecter
les contrats ci-dessous avant d’ajouter une capacité.

## Placement et activation

Le plugin Hermes est placé sous `plugins/research_protocol/`. Son identifiant
public de découverte est le nom de manifeste `research-protocol`, cohérent avec
la future entrée `plugins.enabled: [research-protocol]`; l’underscore reste un
détail du nom de package Python.

Le manifeste déclare `kind: standalone`. Sur cette base Hermes, tous les
plugins standalone sont visibles pour l’introspection mais restent désactivés
tant que leur clé n’est pas ajoutée explicitement à `plugins.enabled`. Le
chargeur ne reconnaît pas de champ `default_enabled`; en inventer un donnerait
une fausse assurance. PR 0 fournit seulement un `register()` no-op et
n’enregistre aucun outil ni callback.

## Acteurs et frontières

| Acteur | Responsabilité autorisée | Frontière à préserver |
|---|---|---|
| Utilisateur | Définit l’objectif et accorde une approbation explicite de workflow | L’approbation est humaine, exacte, bornée et consommable |
| Planner | Lit le contexte autorisé et produit un plan canonique | Ne peut pas exécuter, publier ni choisir un chemin arbitraire |
| Approval service | Présente le résumé construit depuis l’artefact et émet un reçu | Ne valide que le hash et la portée exacts |
| Worker | Réclame et exécute une capacité déjà autorisée | Ne peut pas élargir capacité, budget, durée ou entrées |
| DB | Fournit les lectures typées autorisées | Aucun SQL fourni par le modèle ; rôle lecteur sans secrets |
| Stockage | Persiste les artefacts et leur provenance | Root configuré, chemins sûrs, écriture atomique et hash vérifié |
| Source externe | Fournit des candidats ou contenus à collecter | Entrée non fiable ; jamais une autorité d’instruction |

## Actifs

Les actifs à protéger et à relier par provenance sont :

- plan canonique ;
- hash SHA-256 du plan et des artefacts ;
- approbation et reçu de consommation ;
- credentials et secrets ;
- manifest ;
- evidence ;
- reports.

Les credentials ne doivent apparaître ni dans les plans, ni dans les artefacts,
ni dans les rapports, ni dans les logs.

## Contrats de sécurité

### 3.1 Plan canonique et hash

- Le payload validé est converti en JSON canonique UTF-8 avec clés triées, séparateurs compacts et sans valeurs non finies.
- Le fichier est écrit avec une convention d’octets unique et documentée ; le SHA-256 porte sur les octets exacts persistés.
- `plan_artifact_write` ne prend jamais de `path`. Il reçoit `artifact_type`, `artifact_id` et `payload`.
- Les destinations sont dérivées dans un root configuré, résolues par `realpath`, refusent `..`, chemins absolus, composants symlink et écrasement d’un artefact déjà approuvé.
- La résolution, la création du temporaire et la publication sont relatives à
  des descripteurs de répertoires de confiance, sans suivre les symlinks.
- La destination finale est réservée exclusivement et publiée avec une
  primitive atomique sans remplacement (`RENAME_NOREPLACE` ou équivalent) ;
  toute collision est refusée.
- Le temporaire est créé dans le même filesystem ; le fichier puis le
  répertoire parent sont `fsync`, avant relecture et recalcul du hash.
- Le reçu renvoie au minimum `artifact_id`, `schema_version`, `path_relative`, `sha256`, `byte_length` et `created_at`.

### 3.2 Approbation exacte

- `plan_approval_request` reçoit uniquement un `artifact_id` existant, un hash attendu et une portée structurée.
- L’outil relit l’artefact, recalcule le hash, refuse tout écart et construit lui-même le résumé présenté à l’utilisateur.
- La portée contient : `capability`, `run_id`, `input_hashes`, budgets, durée maximale, nombre maximal d’exécutions, date d’expiration et droits externes demandés.
- L’interface utilise l’approbation native Hermes. `/yolo` et `approvals.mode: off` ne transforment pas cette autorisation de workflow en approbation implicite.
- Un reçu d’approbation durable contient un `approval_id` imprévisible, le SHA-256 exact, la portée exacte, le verdict, l’identité/surface disponible, les timestamps et le compteur de consommation.
- Le worker réclame atomiquement une exécution autorisée ; expiration, capacité différente, budget différent, hash différent, réutilisation au-delà du cap ou approbation absente donnent un refus fail-closed.

### 3.3 Lecture PostgreSQL typée

- `plan_context_read` reçoit un `query_id` parmi une allowlist et des paramètres validés ; jamais du SQL.
- Le mapping `query_id -> requête paramétrée -> modèle de sortie` est versionné dans le plugin.
- Le rôle PostgreSQL `planner_reader` est `NOINHERIT`, lecture seule et sans accès aux tables de secrets.
- Une transaction `READ ONLY`, un timeout, un plafond de lignes et un plafond d’octets sont obligatoires.
- Les logs contiennent le `query_id`, le nombre de lignes et la latence, jamais la DSN ni les champs classés secrets.

### 3.4 Capacités isolées

Ces noms sont des capacités du plugin, pas des capacités Hermes natives présumées :

Les états du tableau sont les cibles des phases ultérieures. Dans PR 0, toutes
les capacités et tous les profils ci-dessous sont non implémentés et désactivés.

| Capacité | Skill requis | Toolset plugin | Profil | État initial |
|---|---|---|---|---|
| `planner` | `intake-plan`, `research-plan` | `planner` + natif `clarify` | `planner` | activé pilote |
| `research-collect` | `research-collect` | `research_collect` | `research-collect` | pilote offline/stub |
| `evidence-review` | `evidence-review` | `evidence_review` | `evidence-review` | désactivé jusqu’à phase 5 |
| `build` | `research-build` | `research_build` | `research-build` | désactivé jusqu’à phase 7 |
| `publish` | `research-publish` | `research_publish` | `research-publish` | désactivé ; local export only |

Chaque capacité devra disposer de son propre `check_fn`, de dépendances, de
tests de schéma, de gates, de configuration de profil et de métriques. Si le
skill ou le plugin n’est pas présent, la capacité est indisponible et le
routeur doit la rejeter explicitement.

## Décision de publication

`publish` reste une action manuelle. Une cible externe ne peut **jamais** être
auto-routée, même si un auto-routing général est activé ultérieurement. Toute
publication externe exige une approbation humaine explicite pour l'opération
exacte et une action hors du routage automatique. Cette approbation n'est
jamais automatique, y compris sous `/yolo` ou avec `approvals.mode: off`.
