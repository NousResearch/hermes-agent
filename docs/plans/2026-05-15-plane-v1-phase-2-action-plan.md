# Plane V1 - Plan d'action phase 2

> Suite au document `2026-05-15-plane-v1-review-notes.md`.
> Objectif: transformer les notes de review en plan d'exécution réaliste, en séparant ce qui est déjà partiellement résolu de ce qui manque vraiment.

Date: 2026-05-15 11:36 CEST
Dernière validation terrain: 2026-05-15 16:45 CEST

## Résumé exécutif

Le chantier Plane V1 est déjà dans un état correct pour la lecture et le pont de base vers le kanban Hermes.
La phase 2 prioritaire a bien été implémentée et les cartes kanban associées sont maintenant clôturées.

Validation réelle effectuée:
- `plane_ping` OK
- `plane_add_comment` OK
- `plane_sync_progress` OK
- `plane_import_to_kanban` OK avec réutilisation idempotente de la même carte Hermes `t_87e21cd3`
- création du workdir local OK: `/home/emeric/AI Factory/AIFACTORY-13_validation-live-v1-hermes-plane-updated`

Caveat important découvert en live:
- la stratégie d'idempotence de `plane_create_work_item` n'est pas encore totalement lissée côté retour outil
- en pratique, un retry avec le même couple `external_source` + `external_id` est bien bloqué par Plane, mais remonte encore un `409 Conflict` au lieu d'un retour Hermes propre `already_existed=true`
- autrement dit: la sécurité anti-doublon est là, l'ergonomie de l'idempotence reste à finir

---

## Audit rapide du code actuel vs review notes

### Déjà partiellement ou totalement couvert

#### Point 1 - Idempotence de `plane_import_to_kanban`
Partiellement couvert.

Constat:
- `plane_import_to_kanban` passe déjà un `idempotency_key` à `kb.create_task(...)`
- clé actuelle: `plane:{workspace_slug}:{project_id}:{work_item_id}`
- `kanban_db.create_task()` sait déjà retourner une tâche existante si cette clé existe

Limite:
- le retour n'indique pas explicitement `already_imported=True`
- il n'y a pas de lookup orienté metadata utilisateur, seulement l'idempotency côté création
- la sémantique n'est pas documentée comme garantie produit

Verdict:
- protection technique déjà présente
- contrat outil encore incomplet

#### Point 2 - Sortie compacte par défaut
Plutôt couvert, mais pas proprement spécifié.

Constat:
- `plane_list_work_items` renvoie déjà un résumé compact via `_summarize_item(...)`
- `plane_board_snapshot` renvoie encore trop de données brutes pour `project` et `states`
- `plane_get_work_item` renvoie un item enrichi proche du brut

Verdict:
- l'intention compacte existe déjà
- il faut rendre le contrat cohérent et documenté, avec `verbose=True`

#### Point 3 - PATCH partiel strict pour `plane_update_work_item`
Globalement couvert.

Constat:
- `_build_work_item_payload()` n'ajoute que les champs explicitement passés
- `_handle_update_work_item()` refuse un patch vide
- donc on évite déjà le gros risque d'écrasement total involontaire

Limite:
- la distinction "champ absent" vs "champ volontairement remis à null" n'est pas formalisée
- le code filtre encore `None`, donc il ne permet pas aujourd'hui un reset explicite de certains champs

Verdict:
- risque principal déjà bien réduit
- reste à formaliser et tester le comportement sur `None`

#### Point 9 - `plane_state_id` à l'import
Déjà couvert dans la body Hermes.

Constat:
- `plane_import_to_kanban` écrit déjà `plane_state_id: ...` dans la body de la tâche importée

Limite:
- ce n'est pas stocké dans un vrai champ metadata structuré de la tâche kanban
- c'est donc exploitable, mais moins propre pour les évolutions futures

Verdict:
- acceptable pour V1
- améliorable plus tard

#### Point 11 - le préfixe titre ne doit pas servir de clé
Déjà OK dans l'esprit.

Constat:
- l'idempotence d'import repose sur `work_item_id`, pas sur le titre
- le titre préfixé est un confort de lecture

Verdict:
- simple vérification/rappel doc

---

### Manques réels à traiter

#### Point 1 - Idempotence de `plane_create_work_item`
Non couvert.

Constat:
- `plane_create_work_item` POSTe directement
- `external_source` existe dans le payload
- `external_id` est supporté dans le schéma
- mais aucun lookup préalable n'est fait avant création

Impact:
- retry agent ou réseau = doublons Plane possibles

#### Point 4 - `plane_add_comment`
Absent.

Impact:
- impossible d'utiliser Plane comme tableau de bord vivant pendant qu'Hermes bosse

#### Point 5 - `plane_sync_progress`
Absent.

Impact:
- pas de boucle de retour propre Hermes -> Plane
- usage quotidien vite pénible

#### Point 6 - pipeline markdown -> HTML
Très insuffisant actuellement.

Constat:
- `_markdown_to_html()` fait juste `escape + <br>` dans un `<p>`
- pas de listes, pas de gras, pas de liens, pas de code blocks, pas de nettoyage structuré

Impact:
- rendu Plane pauvre et peu prévisible

#### Point 7 - traçabilité visuelle des writes Nova
Absente.

Impact:
- faible confiance humaine sur l'origine des changements

#### Point 8 - détection de drift Plane -> Hermes
Absente.

Impact:
- risque réel de continuer à travailler dans Hermes sur une carte annulée ou déplacée dans Plane

#### Point 10 - `plane_ping`
Absent.

Impact:
- faible coût, bon gain en diagnostic

#### Point 12 - exemple end-to-end dans la doc
Absent.

Impact:
- pas bloquant techniquement, mais utile pour usage réel et futurs sous-agents

---

## Priorisation recommandée

### P0 réel pour stabiliser la V1

1. `plane_add_comment`
2. `plane_sync_progress`
3. idempotence explicite de `plane_create_work_item`
4. formalisation du contrat compact/verbose des outils de lecture
5. test/clarification du patch partiel strict sur `plane_update_work_item`

### P1 utile juste après

6. traçabilité automatique des writes via commentaire Plane
7. `plane_ping`
8. détection de drift `plane_check_kanban_links`
9. pipeline markdown -> HTML plus propre

### P2 ensuite

10. stockage plus structuré des métadonnées Plane dans le kanban Hermes
11. exemple end-to-end dans la doc
12. polish divers

---

## Ordre d'implémentation conseillé

## Lot A - Fermer la boucle de pilotage humain

### A1. Ajouter `plane_add_comment`
Statut: implémenté en phase 2.

But:
- poster un commentaire simple et fiable sur une carte Plane

À faire:
- `tools/plane_client.py`
  - ajouter méthode cliente pour POST comments
- `tools/plane_tool.py`
  - ajouter outil `plane_add_comment`
  - accepter `work_item_id` ou `sequence_id`
  - `body_markdown` + conversion HTML
  - option `prefix=True` par défaut pour ajouter `[Nova]`
- tests
  - client
  - outil
- doc
  - plan principal
  - `website/docs/user-guide/features/plane.md`

Critère de fin:
- Nova peut poster un commentaire lisible sur une carte Plane sans modifier l'état

### A2. Ajouter `plane_sync_progress`
Statut: implémenté en phase 2.

But:
- faire un seul appel pour refléter dans Plane un jalon Hermes

Contrat retenu:
- outil `plane_sync_progress(hermes_card_id=None, summary, status=None, prefix=True)`
- `hermes_card_id` peut être omis par un worker kanban, fallback sur `HERMES_KANBAN_TASK`
- lookup du lien Plane depuis le body de la tâche Hermes importée, via les champs `plane_work_item_id`, `plane_sequence_id`, `plane_url`
- commentaire Plane posté avec `summary` et préfixe `[Nova]` par défaut
- si `status` est fourni, résolution nom/id d'état puis PATCH de l'état Plane avant le commentaire
- erreur claire si la tâche Hermes n'existe pas ou ne contient pas de lien Plane

À faire:
- `tools/plane_tool.py`
  - nouvel outil `plane_sync_progress`
  - entrées minimales: `hermes_task_id` ou équivalent, `summary`, `status=None`, `prefix=True`
  - récupérer le lien Plane depuis la tâche Hermes importée
  - poster le commentaire
  - si `status` fourni, appeler aussi update state
- selon besoin, helper dans `kanban_db.py` ou lecture directe de la tâche Hermes existante
- tests
  - task liée -> commentaire
  - task liée + status -> commentaire + update state
  - erreur claire si pas de lien Plane
- doc
  - convention d'usage explicite

Critère de fin:
- un worker Hermes peut annoncer son progrès dans Plane sans bricolage multi-appels

Pourquoi ce lot en premier:
- c'est la partie qui change le plus l'expérience réelle
- ça rend Plane utile comme dashboard vivant, sans UI parallèle

---

## Lot B - Sécuriser contre les doublons et comportements flous

### B1. Rendre `plane_create_work_item` idempotent
But:
- éviter les doublons Plane en cas de retry ou concurrence agentique

Statut: implémenté le 2026-05-15.

Contrat retenu:
- priorité à l'`external_id` fourni par l'appelant
- `external_source` vaut `nova-hermes` par défaut
- si `external_id` est absent, Hermes génère `plane-create:<sha256-prefix>` depuis une base stable explicite: workspace Plane, project Plane, external source, nom normalisé
- les champs mutables (`description`, labels, state, dates, assignees) sont exclus du hash pour qu'un retry avec petites différences de payload ne crée pas de doublon
- avant POST, l'outil cherche un item Plane avec le même couple `external_source` + `external_id`
- si trouvé, l'outil retourne l'item existant avec `already_existed=True`, `created=None`, et ne POSTe pas

À faire:
- `tools/plane_client.py`
  - lookup direct par `external_source` + `external_id` tenté via query params
  - fallback via list/search filtré côté client
- `tools/plane_tool.py`
  - lookup avant POST
  - retour enrichi avec `already_existed`, `external_id_generated`, `external_source`, `external_id`
- tests
  - création première fois
  - deuxième appel même external_id
  - génération stable du fallback

Critère de fin:
- un même create rejoué ne duplique plus la carte

### B2. Rendre l'import idempotent visible côté API outil
But:
- formaliser ce qui est déjà en partie vrai

À faire:
- conserver l'`idempotency_key` actuelle
- enrichir la réponse avec `already_imported=True` si la tâche existait déjà
- idéalement retourner aussi `task_id` existant explicitement comme tel
- documenter la garantie

Critère de fin:
- un import répété renvoie clairement la même tâche Hermes

### B3. Verrouiller le comportement patch partiel de `plane_update_work_item`
But:
- rendre le contrat clair et testé

À faire:
- tests ciblés sur:
  - champ absent => non envoyé
  - champ présent avec valeur => envoyé
  - champ présent à `None` => décision explicite à documenter
- si on garde V1 simple:
  - documenter que `None` n'est pas encore un reset explicite supporté

Critère de fin:
- plus d'ambiguïté sur le comportement du PATCH

---

## Lot C - Rendre les lectures plus propres et moins coûteuses

### C1. Définir un vrai mode compact par défaut
Statut: implémenté le 2026-05-15 dans `tools/plane_tool.py`.

Contrat retenu:
- `plane_board_snapshot` compact par défaut: `project` réduit à `{id,name,identifier}`, `states` à `{id,name,group,count}`, items compacts.
- `plane_list_work_items` compact par défaut: items `{id, sequence_id, readable_id, name, state_name, state_id, priority, labels, assignees_names, url}`.
- `plane_get_work_item` compact par défaut avec le même schéma item.
- `verbose=True` ajoute les payloads bruts: `project_payload` / `states_payload` / `items_payload` pour le snapshot, `items_payload` pour la liste, `payload` + `enriched_item` pour le get.

But:
- réduire le bruit contexte pour Nova

À faire:
- `plane_board_snapshot`
  - compacter `project`
  - compacter `states`
  - garder uniquement les champs utiles par défaut
- `plane_list_work_items`
  - déjà proche de l'objectif, juste harmoniser
- `plane_get_work_item`
  - décider un compact riche par défaut
- ajouter `verbose=True` aux outils de lecture
- documenter les schémas de sortie

Critère de fin:
- les outils de lecture ont un contrat stable, compact par défaut, verbeux sur demande

### C2. Option docstring/schema plus précise
But:
- permettre à Nova et aux sous-agents de mieux anticiper les sorties

À faire:
- descriptions de schéma plus précises dans `plane_tool.py`
- doc utilisateur synchronisée

---

## Lot D - Fiabilité d'usage quotidien

### D1. Ajouter `plane_ping`
Statut: implémenté le 2026-05-15.

But:
- health check simple auth + UA + réseau

À faire:
- endpoint `GET /api/v1/users/me/`
- éventuellement enchaîner un `get_project()` léger pour confirmer la config projet
- retourner: `ok`, `latency_ms`, `user`, `workspace`, `project`

Critère de fin:
- diagnostic immédiat de l'intégration

### D2. Ajouter `plane_check_kanban_links`
But:
- détecter le drift Plane -> Hermes avant de bosser pour rien

À faire:
- lister tâches Hermes liées à Plane
- relire leur état Plane
- comparer avec l'état mémorisé à l'import ou avec les attentes Hermes
- retourner la liste des divergences

Critère de fin:
- Nova peut détecter qu'une carte a changé côté Plane avant de continuer le travail

### D3. Traçabilité automatique des writes
But:
- rendre visible dans Plane ce que Nova a modifié

À faire:
- sur create/update, option `trace_comment=True` par défaut
- commentaire style:
  `[Nova] updated: priority=high, target_date=2026-05-20`
- dépend de `plane_add_comment`

Critère de fin:
- un humain voit dans Plane qu'une action vient de Nova

---

## Lot E - Amélioration du rendu markdown -> HTML

But:
- rendre les descriptions/commentaires plus fiables dans Plane

À faire:
- remplacer `_markdown_to_html()` minimaliste par un pipeline défini
- choisir une lib légère et prévisible
- whitelist de tags supportés
- tests sur listes, code, liens, blockquotes, titres simples

Note:
- utile, mais pas le tout premier bloc à faire si l'objectif est de rendre Plane vivant rapidement

---

## Proposition de découpage en exécution

### Sprint 1
- A1 `plane_add_comment`
- A2 `plane_sync_progress`
- B2 réponse explicite `already_imported`
- D1 `plane_ping`

### Sprint 2
- B1 idempotence `plane_create_work_item`
- B3 verrouillage patch partiel
- C1 compact/verbose

### Sprint 3
- D3 traçabilité automatique
- D2 drift detection
- E markdown -> HTML plus propre
- doc end-to-end

---

## Décisions à prendre avant implémentation

1. `plane_sync_progress` prend-il `hermes_task_id` uniquement, ou aussi `plane_work_item_id` pour usage direct hors kanban ?
2. Pour l'idempotence create, veut-on imposer `external_id` à l'appelant ou générer un fallback automatique ?
3. Pour les commentaires automatiques de traçabilité, veut-on `trace_comment=True` par défaut partout, ou seulement sur demande ?
4. Pour `None` dans update, veut-on supporter le reset explicite en phase 2 ou le repousser ?

Recommandation simple:
- `plane_sync_progress`: accepter `hermes_task_id` d'abord, plus propre pour la boucle Hermes -> Plane
- `external_id`: fallback auto si absent
- `trace_comment`: défaut on, désactivable
- reset `None`: pas nécessaire dans le tout prochain lot si ça complique trop

---

## Recommandation finale

Si on veut maximiser la valeur rapidement sans sur-ingénierie:

1. `plane_add_comment`
2. `plane_sync_progress`
3. rendre l'import explicitement idempotent dans sa réponse
4. ajouter `plane_ping`
5. seulement ensuite traiter l'idempotence de création et le compact/verbose

C'est le meilleur ratio valeur / effort / stabilité.

---

## Fichiers probablement touchés en phase 2

Code:
- `tools/plane_client.py`
- `tools/plane_tool.py`
- éventuellement `hermes_cli/kanban_db.py` si helper de lookup utile
- `toolsets.py` si nouveaux outils

Tests:
- `tests/tools/test_plane_client.py`
- `tests/tools/test_plane_tool.py`
- éventuellement tests kanban ciblés si on ajoute un vrai helper côté kanban

Doc:
- `docs/plans/2026-05-14-plane-v1-integration.md`
- `docs/plans/2026-05-15-plane-v1-review-notes.md`
- `website/docs/user-guide/features/plane.md`
