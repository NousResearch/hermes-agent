# Protocole d'exécution des phases

Un plan de phases JSON a été généré. Voici comment l'exécuter correctement.

## Principe fondamental

**L'orchestrateur ne porte pas le contenu. Il orchestre.**

Chaque phase est exécutée par un `delegate_task` avec :
- Un **contexte minimal** (pas le texte entier de la mission)
- Des **outils restreints** (pas de web dans une phase de commit)
- Une **vérification explicite** (pas de "ça a l'air bon")

## Boucle d'exécution

```javascript
state = {};

for (const phase of plan.phases) {
  // 1. Construire le contexte de cette phase
  const context = buildContext(phase, state);

  // 2. Exécuter la phase
  const result = await delegate_task({
    goal: phase.goal,
    context: context,
    toolsets: phase.toolsets,
  });

  // 3. Vérifier le résultat
  const verified = verifyPhase(result, phase.gate_pass);
  if (!verified && phase.critical) {
    if (phase.max_retries > 0) {
      // Relancer avec contexte enrichi
      // (l'erreur est ajoutée au contexte)
      retryPhase(phase, state, error);
    } else {
      stop; // Arrêt propre
    }
  }

  // 4. Propager les artefacts
  state[phase.id] = extractArtifacts(result, phase.artifacts_out);
}
```

## buildContext(phase, state)

Contexte = uniquement ce dont la phase a besoin :

```javascript
function buildContext(phase, state) {
  const ctx = {
    repoPath: repoPath,
    missionType: missionType,
  };

  // Ajouter les artefacts des phases précédentes
  for (const dep of phase.depends_on) {
    if (state[dep]) {
      ctx[dep] = state[dep];
    }
  }

  return JSON.stringify(ctx, null, 2);
}
```

## Vérifications (gate_pass)

| Type | Code |
|------|------|
| `file_exists` | `fs.existsSync(path)` |
| `exit_code_zero` | `terminal(command).exit_code === 0` |
| `git_clean` | `git status --porcelain` doit être vide (sauf fichiers connus) |
| `test_count` | Parse le output TAP — `pass + fail = total` et `fail === 0` |
| `sha_match` | `git merge-base --is-ancestor base HEAD` |
| `artifact_propagation` | Tous les champs `artifacts_out` sont présents dans le résumé |

## Erreurs connues et reprises

| Erreur | Action |
|--------|--------|
| Sous-agent n'a pas écrit de fichier | Vérifier `artifacts_out` → relancer avec consigne plus stricte |
| Tests échouent | Passer l'erreur de compilation/test dans le contexte de la relance |
| Gate impossible | `critical=true` + `max_retries=0` → arrêt complet, rapport d'échec |
| Artefact manquant | Le sous-agent a peut-être réussi mais n'a pas rapporté l'artefact. Vérifier directement avec `terminal()` |

## Anti-patterns

- ❌ Mettre tout le texte de la mission dans chaque phase
- ❌ Faire confiance au résumé du sous-agent sans vérifier
- ❌ Utiliser `toolsets: ['terminal', 'file', 'web', 'search']` pour chaque phase
- ❌ Relancer une phase sans enrichir le contexte avec l'erreur
- ❌ Ignorer les artefacts non propagés ("c'est bon, je sais ce qu'il a fait")
