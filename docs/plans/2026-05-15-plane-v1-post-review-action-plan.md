---
date: 2026-05-15
audience: emeric, nova
status: plan d'action V1.1, lot statique livré par Claude le 2026-05-15, reste runtime live pour Nova
supersedes_status_in:
  - 2026-05-14-plane-v1-integration.md
  - 2026-05-15-plane-v1-phase-2-action-plan.md
sources:
  - 2026-05-15-plane-v1-review-notes.md
  - rapport Nova du 2026-05-15
  - audit code/docs/tests Claude du 2026-05-15
---

# Plane V1.1 : plan d'action post-review

## 0. Pourquoi ce doc

Nova a livré la V1 + la phase 2 prioritaire de l'intégration Plane le 2026-05-15. Une review croisée (code, docs, tests) a confirmé que la direction est la bonne, mais a aussi identifié plusieurs limites non remontées dans le rapport initial. Ce doc remplace le suivi en cours et devient la référence pour finir la V1.1 proprement avant d'élargir le scope.

Principe directeur inchangé : Plane reste la source de vérité visuelle, Hermes ne fournit que des outils, pas d'UI parallèle.

## 1. Etat actuel honnête

### Ce qui marche réellement
- Architecture client / tool propre (`plane_client.py` transport, `plane_tool.py` orchestration).
- 4 capacités attendues couvertes par des outils dédiés : lire (`plane_board_snapshot`, `plane_list_work_items`, `plane_get_work_item`), modifier (`plane_create_work_item`, `plane_update_work_item`, `plane_add_comment`), importer vers kanban (`plane_import_to_kanban`), livrables (`plane_prepare_workdir` + workdir auto à l'import).
- 27 tests passent (`pytest tests/tools/test_plane_client.py tests/tools/test_plane_tool.py -q`, 0.88s).
- User-Agent navigateur injecté partout, contournement Cloudflare validé.
- Clé API jamais loggée, jamais exposée dans les retours d'outils.
- Workdir live `AIFACTORY-13_validation-live-v1-hermes-plane-updated` créé avec `README.md`, `work/`, `deliverables/validation-live-v1.md`. Cohérent.
- Pont Hermes vers Plane validé en live (commentaire, sync avancement, changement d'état).

### Ce qui est partiel ou cassé
- Bug bloquant `plane_create_work_item` au retry (voir P0/B1).
- Plusieurs limites P1 documentées dans le `review-notes` mais omises du rapport (voir P1).
- Couverture de tests inégale, notamment sur les chemins exécutés en live (voir Tests).
- Doc utilisateur `plane.md` honnête sur le bug 409 mais incomplète sur le reste.

## 2. P0 : bug bloquant à corriger en priorité

### B1. Idempotence 409 sur `plane_create_work_item` [TRAITÉ 2026-05-15 Claude]

Symptôme : un retry avec mêmes `external_source` + `external_id` renvoie `tool_error` (409 Conflict) au lieu d'un `already_existed=true` propre.

Localisation : `tools/plane_tool.py:531` (handler `_handle_create_work_item`). Aucun `except` autour de `client.create_work_item(payload)`. Le `PlaneAPIError(409)` remonte au `except Exception` ligne 542 et devient un `tool_error`.

Cause racine : le lookup préalable `_find_existing_by_external_id` (`plane_client.py:267-285`) fait deux passes mais Plane Cloud ne renvoie pas systématiquement `external_source` / `external_id` dans la liste sans `expand`. Quand le lookup rate alors que la carte existe vraiment côté Plane, le POST déclenche le 409 non rattrapé.

Fix proposé (minimal, dans `_handle_create_work_item`) :

```python
from tools.plane_client import PlaneClient, PlaneAPIError

try:
    item = client.create_work_item(payload)
except PlaneAPIError as exc:
    if exc.status_code == 409:
        existing = _find_existing_by_external_id(client, external_source, external_id)
        if existing:
            enriched = _enrich_item(client, existing, project)
            return _ok(
                item=enriched,
                created=None,
                already_existed=True,
                external_source=external_source,
                external_id=external_id,
                external_id_generated=not external_id_provided,
            )
    raise
```

Test de régression OBLIGATOIRE (sans test, on accepte pas le fix) :
- `test_plane_create_work_item_recovers_from_409_when_lookup_misses`
- Mock `StubClient` qui : retourne `[]` sur `find_work_item_by_external_id`, lève `PlaneAPIError(status_code=409)` sur `create_work_item`, puis retourne la carte existante sur le second `find_work_item_by_external_id`.
- Assert : `already_existed=True`, `created=None`, item enrichi présent.

Bonus optionnel (non bloquant) : passer `?expand=...` dans le premier `list_work_items` pour réduire la fréquence du fallback double-scan.

**Statut 2026-05-15 (Claude) :** fix appliqué dans `tools/plane_tool.py` `_handle_create_work_item` via `try/except PlaneAPIError` + double lookup en cas de 409. Test de régression `test_plane_create_work_item_recovers_from_409_when_lookup_misses` ajouté dans `tests/tools/test_plane_tool.py`. Tous les tests passent.

## 3. P1 : limites non remontées dans le rapport, à clarifier avant de prétendre V1 complète

Ces 6 points sont déjà documentés dans `2026-05-15-plane-v1-review-notes.md` mais n'ont pas été inclus dans le rapport de clôture. Aucun n'est bloquant individuellement, mais leur cumul rend le terme "V1 complète" inexact.

### L1. Rendu Markdown vers HTML insuffisant [REPORTÉ V1.2, décision Nova 2026-05-15]
Pipeline actuel : `escape + <br>`. Listes, gras, italique, code, liens ne fonctionnent pas dans les commentaires Plane envoyés depuis Hermes.
Action : décider si on traite en V1.1 (introduire `markdown` ou `mistune`, lib stable, sans dépendances lourdes) ou si on le reporte explicitement avec une mention claire dans `plane.md`.

**Décision 2026-05-15 (Nova) :** report V1.2.

**Justification :** `markdown` est déjà installé mais non sûr tel quel pour du contenu utilisateur, car il laisse passer HTML brut et liens `javascript:`. `mistune` serait un meilleur candidat pur Python pour une V1.2 sandboxée, avec périmètre minimal recommandé : gras, italique, listes, code inline/blocs, liens http/https uniquement. Pour V1.1, la bonne décision produit est de garder le pipeline minimal déjà documenté plutôt que d'introduire un rendu riche insuffisamment assaini.

**Statut 2026-05-15 (Claude + Nova) :** limite déjà documentée dans `plane.md`. Pas de changement code en V1.1.

### L2. Drift detection absente (`plane_check_kanban_links`) [TRAITÉ 2026-05-15 Nova]
Risque réel : si une carte Plane est annulée / supprimée pendant qu'Hermes bosse dessus, le travail continue dans le vide.
Action V1.1 minimale : un outil `plane_check_kanban_links` qui prend la liste des tâches Hermes liées à des cartes Plane et retourne celles dont l'état Plane est `Cancelled` ou la carte introuvable. Pas d'auto-action, juste du reporting.

**Décision 2026-05-15 (Nova) :** in V1.1.

**Statut 2026-05-15 (Nova) :** implémenté dans `tools/plane_tool.py`. Contrat retenu : entrée `hermes_card_ids: string[]`, sortie `{"items": [...]}` contenant uniquement les anomalies `cancelled` ou `missing`, sans auto-action. Validation live sur carte dédiée `AIFACTORY-15` importée dans le kanban Hermes, puis basculée en `Cancelled` et détectée correctement par l'outil.

### L3. Traçabilité automatique des writes absente [REPORTÉ V1.2, décision Nova 2026-05-15]
Quand Hermes modifie une carte Plane, rien ne distingue visuellement la modification d'une modification humaine.
Action V1.1 : ajouter systématiquement un commentaire `[hermes] <action>` après chaque write (`update_work_item`, `add_comment`, changement d'état). Comportement désactivable via flag d'outil pour les cas où c'est explicitement non voulu.

**Décision 2026-05-15 (Nova) :** report V1.2.

**Justification :** le risque de spam est réel, surtout sur `add_comment` et `plane_sync_progress` qui signent déjà l'origine côté contenu. Le bon comportement produit est plus fin qu'un `always on`: probablement actif par défaut sur `plane_update_work_item` et changements d'état, inactif sur `plane_add_comment`, avec un flag d'override explicite. Ça demande un contrat transversal supplémentaire sur tous les outils d'écriture, donc mieux traité proprement en V1.2.

### L4. `already_imported=True` non exposé sur `plane_import_to_kanban` [TRAITÉ 2026-05-15 Claude]
Le handler ne retourne pas de signal clair quand la carte Plane est déjà importée dans le kanban Hermes. Symétrique au B1.
Action V1.1 : ajouter `already_imported: bool` au retour, basé sur la présence d'un linkage existant.

**Statut 2026-05-15 (Claude) :** implémenté via pré-check SQL sur `idempotency_key` (`plane:<workspace>:<project>:<work_item_id>`) avant l'appel à `kb.create_task`. Aucune modification de `kanban_db.py` requise. Test `test_plane_import_to_kanban_flags_already_imported_when_idempotency_hits` ajouté. Champ `already_imported: bool` ajouté dans chaque entrée `created_tasks`. Doc utilisateur `plane.md` mise à jour.

### L5. PATCH partiel `plane_update_work_item` : comportement sur `None` non formalisé [TRAITÉ 2026-05-15 Claude]
Aujourd'hui, passer `None` à un champ : est-ce ignoré, est-ce que ça écrase ? Pas testé, pas documenté.
Action V1.1 : décider (recommandation : `None` = ignoré, seuls les champs explicitement passés sont envoyés en PATCH), ajouter un test, documenter dans `plane.md`.

**Statut 2026-05-15 (Claude) :** sémantique retenue : `None` (ou absent) = ignoré, non envoyé en PATCH. Liste vide `[]` sur `labels` / `assignees` = signal explicite de clear, transmis tel quel. Comportement déjà conforme dans `_build_work_item_payload` ; commentaire de contrat explicite ajouté. Tests `test_plane_update_work_item_ignores_none_fields_in_patch_payload` et `test_plane_update_work_item_empty_list_clears_labels_and_assignees` ajoutés. Documenté dans `plane.md` section `plane_update_work_item`.

### L6. Repo 630 commits behind `origin/main` + 5 fichiers modifiés non liés [TRAITÉ 2026-05-15 Nova]
Au démarrage du chantier, l'arbre Git était significativement dérivé. Pas de mention dans le rapport.
Action immédiate : `git fetch && git status` partagés, décision explicite (rebase, merge, ou ignorer), nettoyage des fichiers modifiés non liés.

**Décision 2026-05-15 (Nova) :** ne pas rebase ni merge pendant cette passe. Isoler d'abord le lot Plane V1.1 sur branche dédiée, préserver les changements hors scope à part, puis laisser la remise à niveau avec `origin/main` pour une étape Git séparée une fois le lot stabilisé.

**Statut 2026-05-15 (Nova) :** clarifié avec Emeric. Branche de travail `plane-v1.1-nova`, branche de préservation `wip/plane-v1-preserve-20260515`, fichiers hors scope sortis du lot courant dans `stash@{0}`. Aucun changement Claude perdu.

## 4. P2 : code smells secondaires

Pas urgents, mais à traiter quand on touche les fichiers concernés.

### C1. `except Exception: pass` silencieux dans `_check_plane_requirements` [TRAITÉ 2026-05-15 Claude]
`tools/plane_tool.py:33-35`. Si `load_hermes_dotenv` plante autrement que par variable manquante, l'erreur disparaît.
Fix : narrow l'except à la classe d'erreur attendue, logger le reste.

**Statut 2026-05-15 (Claude) :** narrow à `except OSError` (couvre `FileNotFoundError`, `PermissionError`). Commentaire explicite ajouté pour signaler que toute autre exception remontera désormais.

### C2. `prepare_workdir` hardcode `AIFACTORY` [TRAITÉ 2026-05-15 Claude]
`tools/plane_tool.py:424`. Si le projet Plane change un jour, le nom de dossier diverge du `readable_id` réel.
Fix : prendre le `project_key` réel depuis le project Plane récupéré, pas la constante de fallback.

**Statut 2026-05-15 (Claude) :** signature de `prepare_workdir` enrichie d'un kwarg optionnel `project_key`. `_handle_import_to_kanban` passe le `key` calculé via `_project_key(client, project)`. `_handle_prepare_workdir` essaie de résoudre via `get_plane_client().get_project_identifier()` si l'env Plane est configurée, sinon fallback `AIFACTORY`. Le schéma `PLANE_PREPARE_WORKDIR_SCHEMA` expose désormais `project_key` (optionnel). Tests `test_plane_prepare_workdir_honours_custom_project_key` et `test_plane_import_to_kanban_workdir_uses_real_project_key` ajoutés.

### C3. Double scan complet du board à chaque create [TRAITÉ 2026-05-15 Nova]
`tools/plane_client.py:280-285`. Coûteux dès que le board grossit, et appelé systématiquement à chaque `plane_create_work_item`.
Fix : si le premier appel filtré côté serveur retourne 0, ne pas faire le fallback non filtré. Documenter que le lookup externe est best effort.

**Décision 2026-05-15 (Nova) :** in V1.1, mais sous forme de correctif de robustesse plutôt que d'optimisation pure.

**Statut 2026-05-15 (Nova) :** mesuré en live sur le board AI_Factory actuel, charge négligeable. En revanche, le vrai problème observé est plus grave : l'API Plane retourne parfois `404` sur le lookup filtré `external_source` + `external_id`. Le fallback full scan reste donc nécessaire pour la correction fonctionnelle. Implémentation retenue : le filtre serveur reste le premier essai, et les codes `400/404/422` basculent silencieusement vers un scan best effort du board complet. Validation live réussie via création idempotente d'une carte dédiée `AIFACTORY-15`.

### C4. Usage abusif de `Any` contre les guidelines [TRAITÉ 2026-05-15 Claude]
Plusieurs helpers internes prennent `client: Any` alors que `PlaneClient` est connu.
Fix : typer correctement les helpers internes.

**Statut 2026-05-15 (Claude) :** 12 helpers privés re-typés `client: PlaneClient` : `_client_workspace`, `_client_project_id`, `_project_key`, `_summarize_project`, `_plane_url`, `_summarize_item`, `_enrich_item`, `_resolve_item`, `_resolve_state_id`, `_fallback_external_id`, `_prepare_create_idempotency`, `_find_existing_by_external_id`, `_build_work_item_payload`, `_selected_items`. Le `Any` est conservé uniquement pour les dicts JSON Plane (typage runtime via duck-typing tolère le stub client des tests). Tests verts.

### C5. Payload `state_id` + `state` dupliqués [TRAITÉ 2026-05-15 Nova]
`tools/plane_tool.py:401-403` et `:616`. Si c'est un contournement d'un bug Plane connu, le commenter. Sinon, retirer le doublon.

**Décision 2026-05-15 (Nova) :** in V1.1 comme workaround documenté.

**Statut 2026-05-15 (Nova) :** vérifié en live. Le board AI_Factory accepte correctement les transitions quand Hermes envoie à la fois `state_id` et `state`, et ce doublon correspond bien au comportement attendu par l'API dans notre contexte. Le code garde donc les deux champs avec commentaire explicite, et un test de régression vérifie que le PATCH transmis contient bien `state_id` et `state`.

## 5. Trous de tests à combler

Liste minimale pour passer V1.1 :

### T1. Régression 409 (couvre B1) [TRAITÉ 2026-05-15 Claude]
Voir P0/B1.

**Statut 2026-05-15 (Claude) :** `test_plane_create_work_item_recovers_from_409_when_lookup_misses` ajouté dans `tests/tools/test_plane_tool.py`.

### T2. Handler `plane_import_to_kanban` non testé directement [TRAITÉ 2026-05-15 Claude]
Aujourd'hui seul `test_parse_plane_linkage_from_imported_kanban_body` teste un utilitaire. Le handler complet (lookup carte Plane, création tâche kanban, écriture linkage, création workdir) n'a aucun test.
Action : ajouter au minimum 2 tests (chemin nominal + carte déjà importée, lié à L4).

**Statut 2026-05-15 (Claude) :** 3 tests ajoutés dans `tests/tools/test_plane_tool.py` : `test_plane_import_to_kanban_creates_kanban_task_with_plane_linkage` (chemin nominal, body avec linkages + workdir + idempotency_key vérifiés), `test_plane_import_to_kanban_flags_already_imported_when_idempotency_hits` (couvre L4), `test_plane_import_to_kanban_workdir_uses_real_project_key` (couvre C2). `kanban_db` est intégralement mocké, aucun appel réel à SQLite.

### T3. `PlaneClient.create_work_item` et `update_work_item` non testés [TRAITÉ 2026-05-15 Claude]
Pas un seul test direct sur ces deux méthodes du client.
Action : un test par méthode minimum, avec mock HTTP `_FakeResponse`.

**Statut 2026-05-15 (Claude) :** 4 tests ajoutés dans `tests/tools/test_plane_client.py` : `test_create_work_item_posts_payload_to_work_items_endpoint`, `test_create_work_item_rejects_empty_payload`, `test_update_work_item_sends_patch_with_payload`, `test_update_work_item_requires_id_and_payload`. Tous utilisent `_FakeResponse` + `urllib.request.urlopen` mocké.

### T4. Cas d'erreur réseau non couverts [REPORTÉ V1.2, décision Nova 2026-05-15]
Aucun test pour timeout, 5xx, retry behavior. Pas critique en V1 mais à ajouter avant d'aller en prod intensive.

**Décision 2026-05-15 (Nova) :** report V1.2.

**Justification :** le besoin est réel, mais non bloquant pour fermer V1.1. Les chemins critiques de V1.1 ont désormais un filet de sécurité fonctionnel côté create idempotent, et un vrai traitement retry/timeout mérite d'être conçu proprement au niveau client, pas ajouté à moitié dans cette passe.

### T5. `plane_update_work_item` chemin nominal [TRAITÉ 2026-05-15 Claude]
Le seul test actuel valide juste le message d'erreur de validation. Le path qui marche n'est pas couvert.

**Statut 2026-05-15 (Claude) :** `test_plane_update_work_item_nominal_path` ajouté dans `tests/tools/test_plane_tool.py`. Couvre payload construit (name, state_id, labels résolus, priority), `last_update` du StubClient vérifié.

## 6. Doc utilisateur `plane.md` à compléter [TRAITÉ 2026-05-15 Claude]

Manques identifiés :
- Liste exhaustive des champs supportés par `plane_update_work_item` (aujourd'hui : "selected fields" sans énumération).
- Mention explicite de la limite Markdown actuelle (lié à L1).
- Contrat de retour `already_imported` sur `plane_import_to_kanban` (lié à L4).
- Section troubleshooting au-delà du Cloudflare 403 (erreurs idempotence, lookup raté, drift, etc.).
- Un exemple end-to-end du workflow type : import carte Plane vers Hermes, sync progress, fermeture.

**Statut 2026-05-15 (Claude) :** `website/docs/user-guide/features/plane.md` mise à jour sur tous les points :
- Énumération complète des champs supportés par `plane_update_work_item` + section dédiée sur la sémantique PATCH partial (None=ignoré, []=clear explicite).
- Mention explicite de la limite Markdown V1.1 sous `plane_add_comment` et dans la section troubleshooting.
- Contrat `already_imported` documenté sous `plane_import_to_kanban` (return shape complet + comportement idempotent).
- Section troubleshooting enrichie : Cloudflare 403, 409 idempotence, lookup misses, kanban non lié, workdir / project_key.
- Nouvelle section "End-to-end example" avec un workflow JSON complet (import → first sync → comment → close).
- Documentation du nouveau paramètre `project_key` sur `plane_prepare_workdir`.

## 7. Conditions d'acceptation V1.1

V1.1 considérée comme finie si et seulement si :
1. B1 fixé + test de régression T1 vert.
2. T2 et T3 ajoutés et verts.
3. L4 (`already_imported`) exposé dans le retour.
4. L5 (comportement PATCH sur `None`) décidé, testé, documenté.
5. L6 (état Git) clarifié et nettoyé.
6. Doc `plane.md` mise à jour sur les 4 points listés en section 6.
7. C1 (silencieux) et C2 (`AIFACTORY` hardcodé) fixés.
8. Décisions actées pour le lot restant : L1 report V1.2, L2 in V1.1, L3 report V1.2, C3 in V1.1, C5 in V1.1, T4 report V1.2.
9. Les items retenus in V1.1 par Nova sont implémentés, testés en mock et validés en live.

**Etat 2026-05-15 (Nova) :** conditions 1 à 9 satisfaites sur le lot Plane V1.1. Baseline actuelle : `pytest tests/tools/test_plane_client.py tests/tools/test_plane_tool.py tests/test_toolsets.py -q` => `67 passed`.

## 8. Hors scope V1.1 (à acter)

Ne pas glisser dans la V1.1, à reporter explicitement :
- Pipeline Markdown HTML "riche" (au-delà de ce qui est décidé pour L1).
- Traçabilité visuelle avancée (badges, couleurs, etc.).
- Sub-agents / délégation profils Hermes basée sur les tâches Plane.
- UI Hermes dédiée Plane (toujours interdit par le brief initial).

## 9. Notes ouvertes

- Garder Plane comme interface visuelle, pas de dérive.
- À chaque fois qu'on découvre une nouvelle limite non triviale, la consigner ici, pas dans un rapport oral.
- Les fichiers `2026-05-14-plane-v1-integration.md` et `2026-05-15-plane-v1-phase-2-action-plan.md` restent valides comme historique mais ne reflètent plus le vrai statut. Ce doc fait foi.
- **Finding 2026-05-15 (Nova) : assignees Plane attendent des UUID utilisateur, pas un display name.** Aujourd'hui le payload `assignees` n'est pas résolu côté Hermes, ce qui rend l'usage par display name silencieusement non fonctionnel. Embarqué dans le backlog V1.2 (voir section 12).
- **Branche de préservation `wip/plane-v1-preserve-20260515` et `stash@{0}`** : conserver jusqu'à merge V1.1 sur la branche de référence et début effectif de V1.2. À nettoyer ensuite via decision explicite.

## 10. Lot statique livré (Claude, 2026-05-15)

Couvert dans la passe Claude (zéro accès live à Plane, tout mocké) :

| Item | Statut | Tests |
| --- | --- | --- |
| B1 (fix 409 idempotence) | ✅ | +T1 |
| L4 (`already_imported`) | ✅ | +T2 (2 tests) |
| L5 (PATCH `None` = ignoré) | ✅ | +2 tests dédiés |
| C1 (narrow except dotenv) | ✅ | couvert par existants |
| C2 (`project_key` réel) | ✅ | +2 tests (handler + import) |
| C4 (`client: PlaneClient`) | ✅ | tests verts |
| T1 (régression 409) | ✅ | ajouté |
| T2 (handler import_to_kanban) | ✅ | 3 tests ajoutés |
| T3 (client create + update) | ✅ | 4 tests ajoutés |
| T5 (update path nominal) | ✅ | ajouté |
| Doc utilisateur `plane.md` | ✅ | section 6 traitée |

Baseline 27 tests → final **39 tests verts** (`pytest tests/tools/test_plane_client.py tests/tools/test_plane_tool.py -q`).

Fichiers touchés :
- `tools/plane_tool.py` : fix B1, comment PATCH partial, `prepare_workdir(project_key=…)`, narrow except C1, typage `PlaneClient`, schema `plane_prepare_workdir` enrichi, `already_imported` ajouté au retour de `_handle_import_to_kanban`.
- `tests/tools/test_plane_tool.py` : +8 tests.
- `tests/tools/test_plane_client.py` : +4 tests.
- `website/docs/user-guide/features/plane.md` : sections idempotence, PATCH partial, `already_imported`, Markdown limit, end-to-end example, troubleshooting enrichi.

## 11. Reste pour Nova (runtime live requis)

Tout ce qui suit demandait un accès à l'API Plane live ou des décisions produits que Claude n'avait pas vocation à trancher :

| Item | Statut au 2026-05-15 | Note |
| --- | --- | --- |
| **L1 — Markdown riche** | **REPORTÉ V1.2** | `markdown` jugé trop permissif sans sanitization. Candidat recommandé pour V1.2 : `mistune` avec allowlist minimale. |
| **L2 — `plane_check_kanban_links`** | **TRAITÉ 2026-05-15** | Outil implémenté, contrat figé, tests mockés ajoutés, validation live sur `AIFACTORY-15`. |
| **L3 — Traçabilité auto `[hermes]`** | **REPORTÉ V1.2** | Décision produit à affiner pour éviter le spam et définir les overrides par outil. |
| **L6 — État Git** | **TRAITÉ 2026-05-15** | Lot isolé sur `plane-v1.1-nova`, préservation sur `wip/plane-v1-preserve-20260515`, hors scope en `stash@{0}`. |
| **T4 — Cas erreur réseau** | **REPORTÉ V1.2** | Retry/timeout utiles mais non critiques pour fermer V1.1. |
| **C3 — Double scan board** | **TRAITÉ 2026-05-15** | Coût live négligeable, mais fallback full scan conservé pour contourner un `404` réel du filtre Plane. |
| **C5 — `state_id` + `state` dupliqués** | **TRAITÉ 2026-05-15** | Workaround confirmé en live, commenté et couvert par test. |

Synthèse Nova : V1.1 retient **L2 + C3 + C5**. **L1 + L3 + T4** sont explicitement reportés en V1.2.

## 12. Backlog V1.2

Quatre items reportés. Ordre d'attaque recommandé, à valider avant démarrage effectif.

| Ordre | Item | Raison de l'ordre |
| --- | --- | --- |
| 1 | **L1 — Markdown riche** (`mistune` + allowlist minimale) | Brique fondamentale réutilisée par L3 et par tout futur outil qui écrit du contenu sur Plane. À sécuriser en premier. |
| 2 | **Assignees UUID** (résolution display name → UUID) | Bug fonctionnel silencieux remonté par Nova en fin de V1.1. Petit chantier isolé, bénéfice immédiat sur `plane_create_work_item` et `plane_update_work_item`. |
| 3 | **L3 — Traçabilité auto `[hermes]`** | Dépend de L1 (le commentaire de traçabilité doit pouvoir être stylé proprement). Demande une décision produit sur le périmètre par défaut (par outil) et le mécanisme d'override. |
| 4 | **T4 — Erreurs réseau** (timeout, 5xx, retry) | Hygiène prod, indépendant des autres items. Peut être traité en parallèle ou en dernier. À concevoir au niveau client `PlaneClient`, pas au niveau handler. |

Critères d'entrée V1.2 :
- V1.1 mergée sur la branche de référence.
- Branche `wip/plane-v1-preserve-20260515` et `stash@{0}` arbitrés (conservés ou supprimés).
- Nouveau doc `docs/plans/AAAA-MM-JJ-plane-v1.2-action-plan.md` créé pour piloter V1.2 (ce doc-ci reste l'historique V1 / V1.1).
