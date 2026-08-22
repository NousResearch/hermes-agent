---
name: mission-decomposer
description: "Décompose une mission LAH Stack longue en phases atomiques exécutées par sous-agents isolés. Chaque phase reçoit un contexte minimal et des outils restreints. L'orchestrateur vérifie chaque sortie avant de propager. Zéro framework LLM — pattern Hermes natif avec delegate_task."
category: software-development
---

# Mission Decomposer — Orchestrateur de phases atomiques

## Quand l'utiliser

Après Gate 0 (routing) et avant Gate 1 (CodeGraph). Charge cette skill quand la mission :
- Dépasse ~50 lignes de spécification
- Contient 10+ gates, phases, ou sections
- A déjà causé des oublis ou incohérences lors d'exécutions manuelles
- Combine plusieurs lots (Lot E2, etc.)

Ne pas utiliser pour les missions courtes (< 5 étapes) — le coût de décomposition n'en vaut pas la peine.

## Principe

```
Mission longue (59 gates, 500 lignes)
  → decompose-mission.mjs (déterministe)
  → Plan JSON avec phases atomiques
  → Pour chaque phase :
      - delegate_task(goal, context_minimal, toolsets_limités)
      - Vérification par l'orchestrateur
      - Extraction et propagation des artefacts
  → Pas de contexte ballonné
  → Pas de dépendance LLM superflue
```

L'orchestrateur est **vous** (l'agent Hermes). Le script génère le plan, vous l'exécutez phase par phase.

## Chargement

```bash
skill_view(name='mission-decomposer')
# Charger le script de génération de plan
skill_view(name='mission-decomposer', file_path='scripts/decompose-mission.mjs')
```

## Utilisation

### Étape 1 — Générer le plan

```bash
echo "$MISSION_TEXT" > /tmp/mission.txt
node /home/deploy/.hermes/skills/software-development/mission-decomposer/scripts/decompose-mission.mjs \
  /tmp/mission.txt \
  <REPO_PATH> \
  <MISSION_TYPE>
```

Mission types: `CODE_CHANGE`, `READ_ONLY_AUDIT`, `DESIGN_ONLY`, `MIXED`

Le script produit un JSON structuré avec les phases, leurs dépendances, et les artefacts attendus.

### Étape 2 — Exécuter chaque phase

Pour chaque phase du plan :

```javascript
const result = await delegate_task({
  goal: phase.goal,
  context: buildPhaseContext(phase, state),
  toolsets: phase.toolsets
});

// Vérification par l'orchestrateur (PAS par le sous-agent)
verifyPhase(result, phase.gate_pass);

// Propagation des artefacts
state[phase.id] = extractArtifacts(result);
```

### Étape 3 — Vérifier la cohérence inter-phases

Après toutes les phases, l'orchestrateur exécute les vérifications globales :
- Tests combinés passent
- Aucune régression
- Commit scope propre
- Artefacts durables écrits dans le checkout canonique

## Structure d'une phase

Chaque phase dans le plan est un objet JSON :

```json
{
  "id": "gate0_routing",
  "goal": "Résoudre le repo canonique et créer le worktree",
  "context_fields": ["mission_text", "repo_mappings"],
  "toolsets": ["terminal", "file"],
  "gate_pass": {
    "type": "file_exists|exit_code_zero|test_pass|git_clean",
    "params": {}
  },
  "artifacts_out": ["repos", "worktree", "branch", "head_sha"],
  "depends_on": [],
  "max_retries": 1,
  "critical": true
}
```

## Catégories de gate_pass

| Type | Vérifie | Exemple |
|------|---------|---------|
| `file_exists` | Un fichier a été créé | `{ path: "docs/superpowers/plans/..." }` |
| `exit_code_zero` | Commande retourne 0 | `{ command: "node --test test/x402/**/*.js" }` |
| `git_clean` | Worktree propre | `{ expect: "clean" }` |
| `test_count` | N tests passent | `{ min: 75, zero_fail: true }` |
| `sha_match` | HEAD est un descendant | `{ ancestor: "ba06076" }` |
| `no_credential` | Aucun secret dans les fichiers | `{ patterns: ["process.env", "LAHB_URL"] }` |
| `artifact_propagation` | Les artefacts attendus existent | `{ fields: ["sha", "test_count"] }` |

## Artefacts propagés entre phases

Les artefacts sont le seul canal de communication entre phases. Chaque phase produit un sous-ensemble :

```
Phase 0 (routing)
  → artifacts: { repo, worktree, branch, head_sha }

Phase 1 (codegraph)
  → artifacts: { contracts, modules, patterns }

Phase 2 (design)
  → artifacts: { design_doc_path, architecture }

Phase 3 (implementation)
  → artifacts: { files_changed, test_count }

Phase 4 (verification)
  → artifacts: { combined_pass, no_regression }

Phase 5 (commit)
  → artifacts: { commit_sha, clean_worktree }
```

L'orchestrateur ne porte pas le contenu des fichiers — seulement les chemins et métadonnées.

## Scripts

| Script | Usage |
|--------|-------|
| `scripts/decompose-mission.mjs` | Génère le plan de phases à partir du texte de mission |
| `scripts/phase-executor.mjs` | Exécute une phase unique avec vérification et propagation |

## Références

| Fichier | Usage |
|---------|-------|
| `references/execution-protocol.md` | Protocole d'exécution : buildContext, verifyPhase, propagation d'artefacts |
| `references/lot-e2-closure-execution-example.md` | Exemple concret d'exécution (closure Lot E2) : commandes Git, tests adversaires, classification baseline, règles de commit correctif |

## Pitfalls

| Trap | Symptom | Fix |
|------|---------|-----|
| **Sous-agent trop outillé** | Le sous-agent a terminal+file+web et part en exploration inutile | Restreindre les toolsets à ce que la phase nécessite vraiment |
| **Contexte insuffisant** | Le sous-agent ne sait pas dans quel repo travailler | Toujours inclure `repo_path`, `branch`, `head_sha` dans le contexte |
| **Gate_pass trop vague** | Le vérificateur accepte n'importe quel résultat comme "réussi" | Utiliser des vérifications concrètes : SHA, test count, exit code |
| **Artefacts non propagés** | La phase suivante n'a pas les infos de la phase précédente | `artifact_propagation` doit être un gate_pass explicite |
| **Dépendances circulaires** | Phase A nécessite Phase B qui nécessite Phase A | Le script rejecte les cycles (DAG strict) |
| **Script échoue pour READ_ONLY_AUDIT / DESIGN_ONLY** | `decompose-mission.mjs` retourne `Phase gate6_tests depends on unknown phase: gate5b_implementation` (erreur en sortie, exit 0) — le template de phases est calé sur CODE_CHANGE (gate5b_implementation n'existe pas pour les types sans implémentation) | Ne pas considérer l'erreur comme un blocage : vérifier que le plan JSON est vide/absent, puis exécuter la mission directement sans décomposition scriptée (fallback lah-workflow Gate 0.5). Le script reste fiable pour CODE_CHANGE/MIXED |
| **Sous-agent crée des fichiers hors scope** | Fichiers dans le mauvais repo | Ajouter `SAFETY: write only to <path>` dans le contexte |
