# CLOE Architecture Gate and Runtime Trace Pattern

## Quand utiliser ce pattern

- **Architecture Gate**: avant un refactoring majeur, pour analyser l'architecture
  actuelle et documenter la cible. Docs-first, pas de code.
- **Runtime Trace**: quand les réponses de CLOE semblent faibles ou incorrectes —
  pour tracer exactement le chemin emprunté (route → intent → adapter → composer
  → modules → output → LLM).

## Architecture Gate Pattern

### Structure

Une architecture gate est un document qui répond à 10 questions précises :

1. Quels composants génèrent du texte de réponse ?
2. Quels composants devraient seulement générer des faits structurés ?
3. Que doit produire le Behavior Composer ?
4. Que doit produire le Brain Context Builder ?
5. Que doit faire le LLM ?
6. Quels contrôles d'évaluation incitent aux réponses template ?
7. Quels contrôles les remplaceront ?
8. Quels modules existants sont valides et doivent être préservés ?
9. Quelles parties doivent être refactorisées plus tard ?
10. Quelle est la roadmap de migration la plus sûre ?

### Sections du document d'architecture gate

1. Executive summary
2. Current architecture assessment (correct vs drift)
3. Responsibility boundary table (modèle actuel vs cible)
4. Text-generating sites map (keep/convert/remove/fallback)
5. Target architecture diagram/description
6. Reasoning Context contract (ce que le composer doit produire)
7. Exemples de bons contextes structurés vs mauvais templates
8. Principes de l'évaluation V2
9. Roadmap de migration (phases avec effort/dépendances/risques)
10. Non-goals explicites

### Exemple de responsabilité

| Composant | Doit générer | Génère actuellement |
|-----------|-------------|-------------------|
| Modules cognitifs | Faits/signaux structurés | Faits structurés ✅ |
| Behavior Composer | Enveloppe de contexte de raisonnement | Enveloppe structurée + TEXTE TEMPLATE (dérive) |
| Brain Context Builder | Contexte de prompt pour LLM | N'existe pas encore |
| LLM (futur) | Réponse en langage naturel | Pas encore câblé |

### Pipeline cible

```
modules → faits structurés → composer → reasoning context →
context builder → prompt → LLM → réponse naturelle
```

### Pipeline actuel (dérive)

```
modules → faits structurés → composer → texte de réponse template → utilisateur
```

Le point de dérive unique est `mergeModuleOutputs` dans
`cloe-behavior-composer.js` (lignes ~689-732) qui convertit les faits
structurés en templates.

## Runtime Trace Pattern

### Objectif

Tracer le chemin exact d'une question utilisateur à travers :
```
question → Gateway router → classifier → CLOE intent → runtime adapter →
behavior composer → modules sélectionnés → context wiring → Brain Context Builder
→ LLM (si câblé)
```

### Prompts de trace standard

1. `Salut, qui es-tu et quel type d'assistante es-tu censée être ?` — identity
2. `Qu'es-tu capable de faire aujourd'hui, concrètement ?` — capabilities
3. `As-tu une mémoire ? Si oui, quelle mémoire utilises-tu ?` — memory
4. `Où en est-on dans la roadmap CLOE ?` — roadmap
5. `Quel est le prochain move recommandé, et pourquoi ?` — next_move
6. `Qu'est-ce qui bloque actuellement ou mérite attention ?` — blockers
7. `Prépare-moi un prompt Codex pour la prochaine mission` — codex_prompt
8. `Si je te demande de changer ton modèle LLM maintenant, que dois-tu faire ?` — llm_change
9. `Devil advocate: est-ce qu'on est en train de construire Chloé dans la bonne direction ?` — devil_advocate

### Champs de trace par prompt

- prompt_index, prompt_label
- detected_language (fr/en)
- route_selected (l'intent du Gateway router)
- request_type (le type CLOE si applicable)
- cloe_assistant_selected (bool)
- runtime_adapter_called (bool)
- behavior_composer_called (bool)
- selected_modules (moduleId[])
- executed_modules (moduleId[])
- module_output_keys (next_move, blockers, active_modes, etc.)
- context_block_produced (bool)
- context_block_size (bytes)
- context_block_injected_into_brain (bool)
- provider_path_bypassed_local (bool)
- final_answer_excerpt (string)
- answer_length (chars)
- observed_failure_mode (string)
- recommended_fix (string)

### Modes d'exécution

```bash
# Mode local (par défaut) — utilise le router Gateway in-process
node tools/ci/cloe-reasoning-context-runtime-trace.mjs --json

# Générer le rapport markdown
node tools/ci/cloe-reasoning-context-runtime-trace.mjs --write-report docs/architecture/<RAPPORT>.md
```

### Types de failure observés

| Failure | Cause | Recommandation |
|---------|-------|---------------|
| `routed_to_<intent>_instead_of_cloe_assistant` | Gateway classifie comme identity/unknown/self-evolution | Ajouter la prompt au keyword matching de l'intent cloe_assistant |
| `generic_fallback` | Module non atteint — fallback "I am processing" | Vérifier la participation rule du module |
| `template_answer_only` | Module atteint mais mergeModuleOutputs sort un template | Remplacer le template par une extraction de fait structuré |
| `count_only_no_context` | Mémoire retourne un compteur mais pas de contexte | Enrichir l'extraction de la réponse mémoire |
| `generic_status` | Session continuity retourne un statut générique | Enrichir le module session continuity |
| `no_modules_available` | Aucun module cognitive pour ce request type | Ajouter une participation rule |
| `fail_closed` | Bloqué par la gouvernance | Vérifier le payload |

### Résultat clé observé (PR #566)

- Seulement 3/9 prompts atteignent l'intent `cloe_assistant`
- Les 6/9 autres partent vers les intents `identity`, `unknown`, ou `self-evolution`
- Les 3 qui atteignent CLOE produisent des réponses faibles (template, compteur, statut générique)
- Le Gateway router est le premier goulot d'étranglement

## Commandes de validation

```bash
# Lancer la trace
node tools/ci/cloe-reasoning-context-runtime-trace.mjs --json | python3 -m json.tool

# Tester la trace
node --test test/cloe-reasoning-context-runtime-trace.test.js

# Vérifier les chemins alternatifs
node tools/ci/cloe-reasoning-context-runtime-trace.mjs --provider --json   # via provider (si configuré)
```
