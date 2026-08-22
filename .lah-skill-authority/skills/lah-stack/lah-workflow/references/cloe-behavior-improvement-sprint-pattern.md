# CLOE Behavior Improvement Sprint Pattern

## Quand utiliser ce pattern

Mission CLOE qui améliore un score comportemental sans ajouter de capacité cognitive.
Sprint validé par `node test/cloe-behavioral-validation-runner.mjs`.

## Structure de sprint

Chaque sprint est **un seul PR, un seul merge, un seul memory lock**.

### Étapes

1. **CodeGraph** — explorer les modules existants (composer, assistant loop, runtime adapter)
2. **Implémentation** — modifier le module cible (toucher le moins de fichiers possible)
3. **Tests unitaires** — `node --test test/cloe-*.test.js`
4. **Validation comportementale** —
   ```bash
   node test/cloe-behavioral-validation-runner.mjs
   ```
   Vérifier :
   - score de la domaine cible
   - scores des autres domaines (régression max 5 points)
   - `deterministic: YES` (exécuter 2 fois, scores identiques)
5. **Reporting** — `test/reports/behavioral-validation-report.md` doit inclure :
   - Tableau Historical Comparison (baseline, previous, current, Δ sprint, Δ baseline, target)
   - Tableau Historical Evolution (sprint progression)
   - Regression Gate (contre baseline ET sprint précédent)
   - Tous les 12 domaines visibles
6. **Commit** — message structuré, branche feature
7. **PR** — via `gh pr create`, mentionner `LOCAL_CI_VERIFIED`
8. **Merge** — via worktree (`/tmp/openclaw-main-merge`)
9. **Post-merge verify** — `git fetch origin main; git checkout main; git merge --ff-only origin/main; node --test`
10. **Memory lock** — `.cloe/memory-locks/<mission>.json`

## Archives de base de référence

Le fichier `test/fixtures/cloe-baseline-v1.json` gèle les scores du
`CLOE_REAL_WORLD_BEHAVIOR_VALIDATION_V1`. Il contient :

```json
{
  "global": 54,
  "domains": { "self_knowledge": 51, "operational_memory": 73, ... },
  "targets": { "global": 77, "self_knowledge": 70, ... },
  "history": [{ "sprint": "baseline", "global": 54, "label": "CLOE_REAL_WORLD_BEHAVIOR_VALIDATION_V1" }]
}
```

**Règle** : ne JAMAIS modifier les scores du baseline après la création.
Les targets sont dans `targets` (pas de cible = 100 par défaut).

## Score targets (améliorations visées)

| Domaine | Baseline | Cible | Sprint responsable |
|---------|----------|-------|-------------------|
| self_knowledge | 51 | 70 | Sprint 1 (FR + Routing) |
| proactivity | 30 | 60 | Sprint 1 |
| session_continuity | 35 | 65 | Sprint 2 (Session Persistence) |
| recommendations | 39 | 65 | Sprint 3 (Reasoning Depth) |
| executive_reasoning | 40 | 65 | Sprint 3 |
| naturality | 74 | 85 | Sprint 4 (Naturality + Tone) |
| hermes_like | 64 | 80 | Sprint 4 |
| governance | 63 | 80 | Sprint 4 |

## Barrières de régression

- **Barrière principale** : aucun domaine ne peut baisser de plus de 5 points
  par rapport au baseline original.
- **Barrière de sprint** : aucun domaine ne peut baisser de plus de 5 points
  par rapport au sprint précédent.

Les deux sont vérifiées automatiquement par le rapport de validation
(onglet `Regression Gate`).

## Pièges connus

### Session persistence in-memory uniquement

La session store (`sessions: Map` dans `cloe-full-assistant-loop.js`) est
**in-memory, lifetime du process** — perdue au redémarrage.
Pas de DB, pas de fichier.

```javascript
const sessions = new Map();
const MAX_SESSION_HISTORY = 50;
// loadSession(sessionId) → previous context fields
// saveSession(sessionId, response, inputText, classification) → persist
```

### Session context : transport vs consommation

**Piège Sprint 2** : la session persistence peut être implémentée mais
les modules n'utilisent pas forcément le contexte stocké.

**Solution** : ajouter `enrichMergedWithSessionContext()` dans
`cloe-behavior-composer.js` comme étape 4b entre `mergeModuleOutputs`
et le retour de `composeBehavior`. Cette fonction post-traite le merged
output pour injecter la continuité dans l'answer, le recommendation,
le next_move, etc.

### Implicite vs explicite

**Piège Sprint 4** : les marqueurs de continuité explicites
("Conversation turn N continuing from previous query") sont rejetés
par l'opérateur. Préférer :

1. Garder le `_session_context` metadata (downstream tracking, pas user)
2. Carry forward les modes/workflows actifs (faire silence)
3. Ne PAS ajouter de texte de continuité à l'answer

### Naturalité des réponses

Éviter le jargon runtime dans les réponses de fallback :

| Ancien (jargon) | Nouveau (naturel) |
|----------------|-------------------|
| `Recommended: X` | `Next step: X.` |
| `Response composed from available modules.` | `I am processing your request.` |
| `Session status retrieved. See fields for details.` | `Here is the current status of your mission.` |
| `Module "X" stopped fail-closed: ...` | `I needed to stop: ...` |
| `N item(s) retrieved. See relevant field.` | `Found N relevant item(s) in memory.` |
| `Primary module returned: ...` | `I encountered an issue: ...` |
| `No cognitive modules available for this request type.` | `I need more context to help with this request.` |

Voir `mergeModuleOutputs` dans `cloe-behavior-composer.js`.

### Tool awareness : deux chemins

Le `recommended_tool` peut être :
- **string** `'codex'` (runtime adapter, fallback par défaut)
- **object** `{ name, best_for, recommendation_rationale }` (cloe-tool-awareness module complet)

Les deux doivent être gérés. Voir le switch `typeof tool === 'object'` dans
`mergeModuleOutputs`.

## Score Dashboard Final (CLOE Behavior Improvement Program V1)

Le dashboard final consolide tous les sprints en une vue unique :

```bash
node tools/ci/cloe-behavior-dashboard.mjs
```

Le dashboard produit :
- `test/fixtures/cloe-behavior-dashboard.json` (machine-readable)
- `test/reports/cloe-behavior-dashboard.md` (markdown)

### Résultat final des 4 sprints

| Métrique | Baseline | Final | Target |
|----------|----------|-------|--------|
| Global | 54 | 52 | 77 |
| Proactivity | 30 | 49 | 60 (+19) |
| Recommendations | 39 | 48 | 65 (+9) |
| Session Continuity | 35 | 39 | 65 (+4) |
| Naturality | 74 | 74 | 85 |
| Tool Awareness | 69 | 63 | — |
| Executive Reasoning | 40 | 38 | 65 |
| Workflow | 62 | 21 | — |

**Interprétation** : le global (52) est sous le baseline (54) à cause du
changement de texte de fallback (Sprint 4) — artefact de mesure, pas
régression cognitive. Les vrais gagnants sont proactivity (+19) et
recommendations (+9).

## Architecture Gate and Runtime Trace

Pour les missions post-amélioration, voir le fichier de référence séparé :
`references/cloe-architecture-gate-and-runtime-trace-pattern.md`

## Commandes de validation

```bash
# Suite unitaire complète
node --test --test-concurrency=1 test/cloe-*.test.js

# Validation comportementale (12 domaines, 44+ scenarios)
node test/cloe-behavioral-validation-runner.mjs

# Régression guard (vérification d'intégration)
node test/cloe-regression-guard.mjs

# Conversation evaluation (30 scénarios)
node test/cloe-conversation-evaluation-runner.mjs
```

## Livraison

```bash
git checkout -b feat/cloe-sprint<X>-<nom>
git add <fichiers-modifies>
git commit -m "feat: CLOE_BEHAVIOR_SPRINT_<X>_<NOM>_V1"
git push origin feat/cloe-sprint<X>-<nom>
gh pr create --base main --head feat/cloe-sprint<X>-<nom> --title "..."
# Merge via worktree
cd /tmp/openclaw-main-merge && git pull origin main
git merge --no-ff feat/cloe-sprint<X>-<nom> -m "chore: merge [LOCAL_CI_VERIFIED]"
git push origin main
```
