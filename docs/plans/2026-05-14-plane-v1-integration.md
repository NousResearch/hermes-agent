# Plane ↔ Hermes V1 Integration Plan

> Reprise rapide: objectif = donner à Hermes un petit toolset Plane robuste, sans créer de nouvelle UI, sans dupliquer Plane, et sans agir sur Plane autrement qu’à la demande d’Emeric.

Date de cadrage: 2026-05-14 22:56 CEST
Dernière mise à jour d’exécution: 2026-05-15 16:45 CEST

## État d’avancement réel

Implémenté dans le repo:
- `tools/plane_client.py`
- `tools/plane_tool.py`
- `toolsets.py`
- `tests/tools/test_plane_client.py`
- `tests/tools/test_plane_tool.py`

Documenté:
- `docs/plans/2026-05-14-plane-v1-integration.md`
- `website/docs/user-guide/features/plane.md`

Validations faites:
- compilation Python OK
- tests ciblés OK: `27 passed` sur `tests/tools/test_plane_client.py` et `tests/tools/test_plane_tool.py`
- lecture live du projet Plane OK via `plane_board_snapshot` et `plane_list_work_items`
- `plane_ping` validé en live
- `plane_add_comment` validé en live
- `plane_sync_progress` validé en live
- `plane_import_to_kanban` validé en live avec création du workdir local

Points déjà confirmés en live sur le projet `AI_Factory`:
- identifier projet: `AIFACTORY`
- états lus correctement: `Backlog`, `Todo`, `In Progress`, `Waiting`, `Done`, `Cancelled`
- 11 work items actuellement lus par l’outil au moment de la dernière validation
- les URLs Plane générées pointent bien vers le projet configuré

Validation live dédiée V1 / phase 2:
- carte Plane de test créée: `AIFACTORY-13`
- `work_item_id`: `2af7e7d3-145b-4d21-b383-3f4833067cea`
- carte Hermes importée: `t_87e21cd3`
- workdir créé: `/home/emeric/AI Factory/AIFACTORY-13_validation-live-v1-hermes-plane-updated`
- commentaire Plane créé par `plane_add_comment`: `8a4afebe-0896-4f0f-a2f5-45cbd40bcf0f`
- commentaire Plane créé par `plane_sync_progress`: `6a0c130b-e5a1-4189-92bb-64ac80fc3af3`

Point important observé en live:
- le retry de `plane_create_work_item` avec le même couple `external_source` + `external_id` est bien bloqué côté Plane par une unicité serveur, mais le handler Hermes remonte encore actuellement un `409 Conflict` au lieu de retourner proprement `already_existed=true`
- donc la protection anti-doublon est effective, mais l’expérience d’idempotence n’est pas encore complètement lissée côté outil

## Objectif

Construire une V1 minimale utile pour intégrer Plane dans Hermes comme couche opératoire:
- lecture efficace du board Plane
- création / mise à jour ciblée de work items
- import explicite de tâches Plane dans le kanban Hermes
- préparation d’un dossier local `AI Factory` pour les livrables

## Contraintes validées

- Plane reste la source de vérité visuelle et projet.
- Aucun dashboard Hermes dédié à Plane.
- Aucun portail parallèle.
- Pas de sync automatique en V1.
- Les écritures Plane ne doivent se faire qu’à la demande de l’utilisateur.
- Le client Plane doit toujours envoyer un vrai `User-Agent` de navigateur pour éviter les 403 Cloudflare `browser_signature_banned`.

## Pré-requis déjà vérifiés localement

- `~/.hermes/.env` existe.
- `PLANE_API_KEY` présent.
- `PLANE_WORKSPACE=ai_factory` présent.
- `PLANE_PROJECT_ID=8695a8d1-e6fc-44e1-8bd2-9f37158b5124` présent.
- Aucun dossier local canonique `AI Factory` n’a encore été trouvé. Il faudra en créer un explicitement.

## État du repo avant modification

Commande observée:

```bash
git -C /home/emeric/.hermes/hermes-agent status --short --branch
```

Résultat au moment du cadrage:

```text
## main...origin/main [behind 630]
 M agent/context_compressor.py
 M agent/title_generator.py
 M plugins/memory/holographic/__init__.py
 M tools/approval.py
 M tools/browser_camofox.py
```

Implication:
- le repo est déjà sale avant ce chantier
- éviter de mélanger les changements non liés
- si besoin, créer ensuite une branche dédiée avant d’aller plus loin

## Décisions d’architecture V1

### 1. Un client Plane partagé

Créer un module interne unique qui gère:
- lecture de `PLANE_API_KEY`, `PLANE_WORKSPACE`, `PLANE_PROJECT_ID`
- `BASE_URL = https://api.plane.so`
- header `X-API-Key`
- header `User-Agent` navigateur obligatoire
- pagination
- sérialisation / gestion d’erreurs HTTP
- helpers de résolution d’état par nom et de work item par id lisible

### 2. Un petit toolset Hermes `plane`

V1 cible:
- `plane_ping`
- `plane_board_snapshot`
- `plane_list_work_items`
- `plane_get_work_item`
- `plane_create_work_item`
- `plane_update_work_item`
- `plane_add_comment`
- `plane_sync_progress`
- `plane_import_to_kanban`
- `plane_prepare_workdir`

### 3. Plane = pilotage, Hermes = exécution

Convention:
- Plane garde la vérité métier / planning / suivi humain
- Hermes kanban ne sert qu’à l’exécution, la décomposition, la délégation, la revue
- pas de mirroring exhaustif des sous-tâches Hermes dans Plane

### 4. Dossier local canonique des livrables

Créer une racine unique:

```text
/home/emeric/AI Factory/
```

Puis par tâche importée:

```text
/home/emeric/AI Factory/AIFACTORY-12_<slug>/
```

Structure V1 du dossier tâche:

```text
README.md
work/
deliverables/
```

## Mapping Plane ↔ Hermes

Quand une tâche Plane est importée dans le kanban Hermes, conserver au minimum:
- `plane_workspace_slug`
- `plane_project_id`
- `plane_work_item_id`
- `plane_sequence_id`
- `plane_url`
- `plane_state_id` optionnel

Convention de titre Hermes:

```text
[Plane AIFACTORY-12] <titre>
```

Convention recommandée pour la body / commentaire initial Hermes:
- résumé opératoire de la tâche
- URL Plane
- état au moment de l’import
- instruction d’origine si utile

## Fichiers à créer ou modifier

### Code

Créer:
- `tools/plane_client.py`
- `tools/plane_tool.py`

Modifier:
- `toolsets.py`

### Tests

Créer:
- `tests/tools/test_plane_client.py`
- `tests/tools/test_plane_tool.py`
- éventuellement compléter `tests/test_toolsets.py`

### Documentation

Créer ou compléter:
- `docs/plans/2026-05-14-plane-v1-integration.md` (ce fichier)
- `website/docs/user-guide/features/plane.md` si le chantier va assez loin
- éventuellement une référence courte dans le skill local `plane` si des pièges d’implémentation apparaissent

## Détail des outils V1

### `plane_board_snapshot`

But:
- vue synthétique du projet
- états disponibles
- comptes par état
- quelques cartes notables

Entrées V1:
- `project_id` optionnel
- `include_items_per_state` bool optionnel
- `per_state_limit` optionnel

### `plane_list_work_items`

But:
- lister les cartes
- filtrer par état, label, assignee, priorité, texte

Entrées V1:
- `state`
- `label`
- `assignee`
- `priority`
- `query`
- `limit`

### `plane_get_work_item`

But:
- lire une carte précise

Entrées V1:
- `work_item_id` ou `sequence_id`

### `plane_create_work_item`

But:
- créer une carte Plane

Entrées V1:
- `name`
- `description_html` ou `description_markdown` converti en HTML simple
- `priority`
- `state`
- `labels`
- `assignees`
- `start_date`
- `target_date`
- `external_source` défaut `nova-hermes`
- `external_id` optionnel

### `plane_update_work_item`

But:
- mettre à jour une carte existante

Entrées V1:
- `work_item_id`
- mêmes champs modifiables que create

### `plane_add_comment`

But:
- poster un commentaire sur une carte Plane sans changer son état

Entrées V1:
- `work_item_id` ou `sequence_id`
- `body_markdown`, converti en HTML simple
- `prefix` bool optionnel, défaut `true`, ajoute `[Nova]`

Effet attendu:
- `POST /issues/{work_item_id}/comments/` avec `comment_html`
- retourne le commentaire créé, le résumé de la carte, et le HTML envoyé

### `plane_import_to_kanban`

But:
- transformer une ou plusieurs cartes Plane en cartes Hermes

Entrées V1:
- `work_item_ids` ou `sequence_ids`
- `assignee`
- `workspace` défaut `scratch`
- `priority` optionnel
- `create_workdir` bool

Effet attendu:
- crée les cartes Hermes liées à Plane
- ajoute la trace Plane dans le titre + body
- peut enchaîner sur `plane_prepare_workdir`

### `plane_prepare_workdir`

But:
- créer la base locale de travail liée à une carte Plane

Entrées V1:
- `sequence_id`
- `title`
- `base_dir` optionnel, défaut `/home/emeric/AI Factory`

Effet attendu:
- crée le dossier tâche
- crée `README.md`, `work/`, `deliverables/`
- retourne le chemin absolu

## Phasage recommandé

### Phase 1: lecture fiable
- client Plane partagé
- `plane_board_snapshot`
- `plane_list_work_items`
- `plane_get_work_item`
- tests unitaires sur headers, auth, pagination, filtres

### Phase 2: écriture maîtrisée
- `plane_create_work_item`
- `plane_update_work_item`
- `plane_add_comment`
- tests unitaires sur payloads, commentaires et erreurs HTTP
- pas de delete

### Phase 3: pont d’exécution Hermes
- `plane_import_to_kanban`
- `plane_prepare_workdir`
- tests sur mapping et conventions locales

## Détails techniques à respecter

### Headers

Toujours envoyer au minimum:

```text
X-API-Key: <PLANE_API_KEY>
Accept: application/json
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36
```

### Endpoints à couvrir en premier

- `GET /api/v1/workspaces/{workspace}/projects/{project_id}/`
- `GET /api/v1/workspaces/{workspace}/projects/{project_id}/states/?per_page=100`
- `GET /api/v1/workspaces/{workspace}/projects/{project_id}/work-items/?per_page=100&expand=state,assignees,labels`
- `POST /api/v1/workspaces/{workspace}/projects/{project_id}/work-items/`
- `PATCH /api/v1/workspaces/{workspace}/projects/{project_id}/work-items/{work_item_id}/`
- `POST /api/v1/workspaces/{workspace}/projects/{project_id}/issues/{work_item_id}/comments/`

Endpoints utiles mais non bloquants V1:
- `GET /api/v1/users/me/`
- `GET /api/v1/workspaces/{workspace}/work-items/{PROJECTKEY}-{NUMBER}/`
- commentaires, labels, modules, cycles

## Convention de reprise si le chantier s’interrompt

Ordre de reprise conseillé:
1. relire ce fichier
2. relire `tools/plane_client.py` et `tools/plane_tool.py`
3. exécuter les tests Plane ciblés
4. vérifier `toolsets.py`
5. vérifier le kanban Hermes si une carte d’implémentation a été créée

## Checklist de progression

- [x] créer `tools/plane_client.py`
- [x] créer `tools/plane_tool.py`
- [x] enregistrer le toolset `plane` dans `toolsets.py`
- [x] exposer les outils read-only V1
- [x] écrire tests client
- [x] écrire tests outils
- [x] ajouter create/update
- [x] ajouter import kanban
- [x] ajouter préparation workdir
- [x] documenter l’usage utilisateur

## État final V1 après validation live

- [x] p5. Valider en live les écritures Plane de V1 sur une carte de test dédiée
- [x] p6. Valider le pont Plane → Hermes kanban + création du workdir AI Factory sur un import réel

Conclusion courte:
- la V1 est utilisable en réel pour lire Plane, créer et mettre à jour une carte, commenter, refléter un progrès depuis Hermes, importer une carte dans le kanban Hermes et créer le workdir local associé
- il reste un point de finition fonctionnelle sur `plane_create_work_item`: convertir le conflit serveur 409 en réponse idempotente propre quand la carte existe déjà

## Notes CODex / bonnes idées à garder

Les suggestions reçues et jugées utiles:
- commencer par REST, pas par MCP Plane
- ne pas activer DELETE au début
- prioriser lecture d’états, lecture work items, create, update, move, comment
- prévoir `plane_ping` et éventuellement `plane_search_work_items` plus tard

Décision actuelle:
- `plane_ping` est intégré au toolset pour diagnostiquer auth, `User-Agent`, réseau et accès projet
- `plane_move_work_item` peut rester un alias ergonomique ultérieur de `plane_update_work_item` avec seul champ `state`
- `plane_add_comment` est inclus pour refléter le progrès Hermes dans Plane sans changer les champs métier
