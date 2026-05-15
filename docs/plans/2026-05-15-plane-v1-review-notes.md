# Plane V1 - Notes de review et points à traiter

> Addendum au plan `2026-05-14-plane-v1-integration.md`.
> Objectif: lister les points pertinents identifiés lors d'une review externe, pour traitement avec Nova.
> Contexte: les outils Plane sont appelés par Nova (agent Hermes) et par ses sous-agents/profils, pas seulement par un humain.

Date de rédaction: 2026-05-15

## Comment utiliser ce document

Les points sont classés par priorité (P0/P1/P2). Pour chaque point:
- **Problème**: ce qui ne va pas ou ce qui manque
- **Proposition**: action concrète
- **Effort estimé**: petit / moyen / gros

À traiter dans l'ordre. Cocher au fur et à mesure. Ne pas tout faire d'un coup si certains points ouvrent un débat.

---

## P0 - À traiter avant de considérer la V1 stable

### [ ] 1. Idempotence stricte de `plane_create_work_item` et `plane_import_to_kanban`

**Problème**
Un retry réseau, un agent qui réessaie, ou deux sous-agents Hermes qui reçoivent la même instruction en parallèle peuvent créer:
- des doublons de cartes Plane (`plane_create_work_item`)
- des doublons de cartes Hermes liées à la même carte Plane (`plane_import_to_kanban`)

Aucun mécanisme de dédup n'est actuellement spécifié.

**Proposition**
- `plane_create_work_item`: utiliser `external_source='nova-hermes'` + `external_id` (fourni par l'appelant ou hash stable du `name`). Avant POST, faire un lookup Plane par `external_id`. Si trouvé, retourner la carte existante avec un flag `already_existed=True` au lieu de re-créer.
- `plane_import_to_kanban`: avant de créer la carte Hermes, chercher dans le kanban Hermes une carte avec metadata `plane_work_item_id` == cible. Si trouvée, retourner cette carte avec `already_imported=True`.

**Statut 2026-05-15**
- `plane_create_work_item` est traité: lookup `external_source` + `external_id`, fallback stable basé sur workspace/project/source/name normalisé, retour `already_existed=True` sur doublon.
- `plane_import_to_kanban` reste à rendre explicitement visible côté retour (`already_imported=True`).

**Effort**: petit/moyen

---

### [ ] 2. Format de sortie compact par défaut pour les outils de lecture

**Problème**
Le plan détaille les entrées de chaque outil mais pas les sorties. Si `plane_list_work_items` ou `plane_board_snapshot` retournent le JSON Plane brut (50+ champs par item), chaque appel pollue le contexte de Nova pour rien. À l'échelle d'une session avec plusieurs appels, ça coûte cher en tokens et dégrade la qualité de raisonnement de l'agent.

**Proposition**
Pour chaque outil de lecture, définir explicitement deux modes:
- Mode par défaut: projection compacte. Pour un work item: `{sequence_id, name, state_name, priority, assignees_names, url}`. Pour un état: `{name, group, count}`.
- Mode `verbose=True`: payload Plane complet.

Documenter le schéma de sortie dans `plane_tool.py` (docstring de l'outil) pour que Nova sache à quoi s'attendre.

**Effort**: moyen

---

### [ ] 3. `plane_update_work_item` doit être un patch partiel strict

**Problème**
Si l'outil renvoie systématiquement tous les champs (même ceux que l'agent n'a pas explicitement passés), il peut écraser silencieusement des modifications faites côté UI Plane entre un read et un write effectués par Nova. C'est typiquement le scénario où je touche moi-même une carte dans Plane et où Nova revient quelques minutes après.

**Proposition**
Le PATCH HTTP envoyé à Plane doit contenir uniquement les champs explicitement passés par l'appelant. Distinguer "champ non passé" de "champ passé à None" (None peut être une valeur de reset volontaire). Tester ça explicitement.

**Effort**: petit (mais à vérifier dans le code actuel, c'est peut-être déjà OK)

---

### [ ] 4. `plane_add_comment` à remonter en V1, pas V1.5

**Problème**
Le plan met `plane_add_comment` en V1.5. Mais c'est l'outil qui ferme la boucle de pilotage humain:
- je garde Plane ouvert sous les yeux (exigence explicite du prompt initial)
- Nova bosse dans Hermes pendant 30 minutes ou 2h
- sans `plane_add_comment`, je dois rouvrir Hermes pour voir où elle en est

Avec, Nova poste sur la carte Plane "[Nova] X fait, Y en cours" et le board Plane reste vivant comme tableau de bord.

**Proposition**
Ajouter en V1:
- `plane_add_comment(work_item_id_or_sequence_id, body_markdown)`
- conversion markdown -> HTML simple comme pour `description` (point #6)
- préfixe automatique `[Nova]` configurable par flag `prefix` (défaut on)

**Effort**: petit

---

## P1 - Important pour l'usage agent quotidien

### [ ] 5. Boucle de retour Hermes -> Plane explicite

**Problème**
Quand Nova finit une sous-tâche dans son kanban interne, par quel mécanisme l'état Plane bouge? Le plan ne décrit pas cette boucle. Sans elle, soit Nova le fait à la main à chaque palier (verbeux, oubliable), soit le board Plane se désynchronise progressivement de la réalité du travail.

**Proposition**
Créer un outil agrégateur:
```
plane_sync_progress(hermes_card_id, status, summary)
```
qui:
- récupère le `plane_work_item_id` lié à la carte Hermes
- poste un commentaire Plane avec le `summary`
- si `status` fourni, change l'état Plane (`In Progress`, `Waiting`, `Done`, etc.)
- retourne l'URL Plane mise à jour

Convention d'usage à documenter: Nova appelle ça à chaque transition d'état significative côté Hermes.

**Effort**: petit/moyen

---

### [ ] 6. Spécifier le pipeline markdown -> HTML

**Problème**
Le plan parle de `description_markdown` converti en HTML simple, sans préciser:
- quel parser
- quels éléments supportés
- quel comportement sur les éléments non supportés

Risque: le rendu Plane diverge de ce que Nova croit avoir envoyé.

**Proposition**
Choisir une lib (`markdown` Python avec extensions minimales, ou `markdown-it-py`). Documenter la liste blanche supportée: `p`, `strong`, `em`, `ul/ol/li`, `code`, `pre`, `a`, `h2/h3`, `blockquote`. Tout le reste est strippé. Tester quelques cas de bord.

**Effort**: petit

---

### [ ] 7. Traçabilité visuelle des actions de Nova côté Plane

**Problème**
`external_source='nova-hermes'` à la création, c'est bien. Mais sur un UPDATE, rien n'indique visuellement dans Plane que c'est Nova qui a touché. Je risque de me demander "c'est moi ou c'est elle qui a changé ce truc?" et de perdre confiance dans le board.

**Proposition**
Après chaque write réussi (`plane_create_work_item`, `plane_update_work_item`), poster automatiquement un mini-commentaire généré:
```
[Nova] updated: priority=high, target_date=2026-05-20
```
Configurable par flag `trace_comment` (défaut on, désactivable au cas par cas).
Dépend du point #4 (`plane_add_comment` doit exister).

**Effort**: petit, dépend de #4

---

### [ ] 8. Détection de drift Plane -> Hermes

**Problème**
Pas de sync auto est une bonne décision. Mais ça veut dire que si je passe AIFACTORY-12 manuellement de `In Progress` à `Cancelled` côté Plane, Nova continue à bosser dessus dans son kanban Hermes pendant des heures. Travail jeté à la poubelle.

**Proposition**
Créer un outil:
```
plane_check_kanban_links(hermes_card_ids=None)
```
qui:
- prend la liste des cartes Hermes liées à Plane (ou un sous-ensemble)
- batch-read leur état actuel côté Plane
- retourne les divergences: `[(hermes_card, plane_state_before, plane_state_now), ...]`

Convention d'usage: Nova l'appelle au démarrage d'une session de travail sur une carte importée, ou au début d'un cycle d'agent.

**Effort**: petit/moyen

---

## P2 - Nice to have, à creuser

### [ ] 9. `plane_state_id` obligatoire à l'import, pas optionnel

**Problème**
Le mapping kanban décrit `plane_state_id` comme optionnel. Si on l'a au moment de l'import, autant le stocker. Sinon, chaque update nécessite un lookup states avant d'envoyer.

**Proposition**
Rendre `plane_state_id` obligatoire dans les metadata stockées par `plane_import_to_kanban`. Si l'API ne le renvoie pas en expand, faire le lookup à l'import (une fois) plutôt qu'à chaque update.

**Effort**: petit

---

### [x] 10. Garder `plane_ping`

**Problème**
Le plan a écarté `plane_ping` du V1. Décision défendable, mais vu le piège Cloudflare déjà rencontré (`browser_signature_banned`), un health-check qui exerce auth + UA + un endpoint simple (`GET /api/v1/users/me/`) coûte 10 lignes et fait gagner du temps au debug futur.

**Proposition**
Ajouter `plane_ping()` qui retourne `{ok: bool, latency_ms: int, user_email: str, project_name: str}`. Outil que Nova peut appeler au début d'une session pour vérifier que l'intégration est saine.

**Effort**: trivial

---

### [ ] 11. Préfixe titre `[Plane AIFACTORY-12]`

**Problème** (mineur)
Le préfixe dans le titre Hermes est fragile si quelqu'un renomme la carte d'un côté ou de l'autre. La metadata reste l'ancre sérieuse.

**Proposition**
Confort de lecture, à garder, mais ne JAMAIS s'en servir comme clé de rapprochement dans le code. Tout matching doit passer par la metadata `plane_work_item_id`. Vérifier que c'est bien le cas dans `plane_import_to_kanban` et autres outils.

**Effort**: vérification, pas de code à écrire si c'est déjà OK

---

### [ ] 12. Documentation utilisateur: ajouter un exemple end-to-end

**Problème**
`website/docs/user-guide/features/plane.md` décrit bien chaque outil mais ne montre pas un flow concret. Un utilisateur (ou un futur sous-agent qui découvre le toolset) ne voit pas comment chaîner les outils.

**Proposition**
Ajouter une section `## Example session` avec un dialogue type:
```
User: Snapshot du board.
Nova: [appelle plane_board_snapshot] -> 5 cartes, dont 2 en Todo.
User: Importe AIFACTORY-3 dans le kanban.
Nova: [appelle plane_get_work_item puis plane_import_to_kanban et plane_prepare_workdir] -> carte Hermes créée, workdir /home/emeric/AI Factory/AIFACTORY-3_<slug>/ prêt.
User: Avance dessus.
[Nova bosse, puis:]
Nova: [appelle plane_sync_progress] -> commentaire posté sur Plane.
```

Documenter aussi le `workspace='scratch'` par défaut de `plane_import_to_kanban` (un utilisateur qui ne connaît pas Hermes ne pige pas ce que ça veut dire).

**Effort**: petit, pure doc

---

## Points écartés après relecture

Pour traçabilité, j'avais initialement levé deux remarques que je retire:

- **Espace dans le chemin `/home/emeric/AI Factory/`**: c'est une exigence nominative explicite du prompt initial ("dossier spécifique nommé AI Factory"). À garder tel quel, le code doit juste gérer les chemins avec espaces (quoting bash, `pathlib`). Non négociable.
- **Repo 630 commits behind origin/main**: vrai signal mais hors scope de ce chantier Plane. À traiter ailleurs.

---

## Suggestion d'ordre de traitement avec Nova demain

1. Audit du code actuel par rapport aux points P0 (1, 2, 3): est-ce que certains sont déjà résolus dans l'implémentation actuelle? Lire `tools/plane_client.py` et `tools/plane_tool.py` avec ces points en tête.
2. Implémenter `plane_add_comment` (point 4) + `plane_sync_progress` (point 5). Ces deux outils ensemble débloquent le workflow "Plane comme dashboard vivant".
3. Traiter les autres P0 dans l'ordre.
4. P1 ensuite.
5. P2 quand le reste est stable.

Pour chaque point traité: mettre à jour le plan principal (`2026-05-14-plane-v1-integration.md`) et la doc utilisateur (`website/docs/user-guide/features/plane.md`) en conséquence.
