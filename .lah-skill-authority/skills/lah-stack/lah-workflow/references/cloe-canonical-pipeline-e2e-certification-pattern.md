# CLOE Canonical Pipeline E2E Certification Pattern

## Quand utiliser ce pattern

Quand tu dois certifier que le pipeline canonique complet de CLOE fonctionne comme un système cohérent — pas des composants isolés — du Guards au Answer Composer en passant par le Policy Resolver, Retrieval Builder, Provider Bridge, et Governance.

**Ne pas confondre avec :**
- `cloe-v3-strategic-benchmark-certification-pattern.md` — V3 strategic decision certification
- `cloe-runtime-integration-and-certification-pattern.md` — VX integration into the runtime adapter
- `cloe-strategic-benchmark-certification-pattern.md` — V5 decision benchmark
- `cloe-operational-validation-pattern.md` — Real operator feedback measurement

## Ce que ce pattern certifie

Que chaque composant canonique :
1. Existe et est importable
2. Reçoit le contrat correct du composant précédent
3. Produit le contrat attendu par le composant suivant
4. S'exécute sur le chemin réel des opérateurs

**Pipeline cible :**
```
Operator Request → Canonical Guards → Intent Classifier → Policy Resolver
→ Router → Retrieval Planner → Evidence Dossier → Provider Bridge
→ Provider → Answer Composer → Governance → Final Response
```

## Structure de la certification

### Scénarios (21 obligatoires)

| Section | Focus | Nombre | Classes testées |
|---------|-------|--------|-----------------|
| A | Deterministic & Security | 6 | A (identity, status, blocked) |
| B | Provider-Enriched Knowledge | 5 | B (diagnostic, analysis, memory) |
| C | Observability & Runtime | 4 | A (status, observability) |
| D | Governance & Execution | 6 | C, D (campaign, mutation, crawl) |
| E | Specialized (V5) | 1 | V5 decision pipeline activation |
| F | Evidence Integrity | — | 9 evidence categories, fidelity checks |

### Trace structurée par scénario

```
operator_input, entry_point, guard_result, canonical_intent, intent_confidence,
response_policy, response_class, retrieval_required, evidence_required,
governance_required, approval_required, retrieval_plan, evidence_categories,
evidence_provenance, provider_called, provider_payload_received,
provider_output_received, answer_composer_called, claim_validation_result,
governance_result, final_response, final_verdict
```

## ⚠️ Piège CRITIQUE: normalizeText supprime les accents (ne les remplace PAS)

`normalizeText()` dans `canonical-intent-classifier.js` :
```javascript
value.toLowerCase().replace(/[^a-z0-9\s]/g, '').trim()
```

Le range `a-z` est ASCII uniquement (U+0061-U+007A). `é` (U+00E9), `è`, `ê`, `à`, `ù`, `ç` NE sont PAS dans ce range. Ils sont **SUPPRIMÉS** du texte, pas remplacés par `e`.

### Ce que ça signifie concrètement

| Mot français | normalizeText() | À NE PAS écrire | À ÉCRIRE |
|-------------|----------------|-----------------|----------|
| exécuter | `excute` | `execute` | `excute` |
| préparer | `prpare` | `prepare` | `prpare` |
| sécurité | `scurit` | `securite` | `scurit` |
| capacité | `capacit` | `capacite` | `capacits?` |
| données | `donnes` | `donnees` | `donn[e]?e?s?` |
| métriques | `mtriques` | `metriques` | `mtriques` |
| problème | `problme` | `probleme` | `problme` |
| règle | `rgle` | `regle` | `rgles?` |
| observabilité | `observabilit` | `observabilite` | `observabilit` |
| mémoire | `mmoire` | `memoire` | `mmoire` |
| précédent | `prcdent` | `precedent` | `prcdent` |
| évaluation | `valuation` | `evaluation` | `valuation` |
| démarre | `dmarre` | `demarre` | `dmarre` |
| déploie | `dploie` | `deploie` | `dploie` |
| étapes | `tapes` | `etapes` | `tapes` |

### Règle absolue

Tous les patterns regex du classifieur doivent être écrits pour la **sortie réelle** de `normalizeText()`.

```javascript
// INCORRECT (ne match JAMAIS):
capability_inquiry: [ /\bcapacités?\b/i ]   // attend é
capability_inquiry: [ /\bcapacites?\b/i ]    // attend 'e' qui n'existe pas

// CORRECT (matche la forme normalisée réelle):
capability_inquiry: [ /\bcapacits?\b/i ]     // match capacité→capacit, capacités→capacits
```

### Piège du trait d'union et des apostrophes

normalizeText() supprime les tirets et apostrophes. `analyse-moi` → `analysemoi`.

Conséquence : les patterns qui attendent `\b` entre les mots (comme `analyse` + `moi`) cassent. Il n'y a pas de word boundary entre `analyse` et `moi` dans `analysemoi`.

**Correctif :** utiliser `analys.{0,15}mtriques` au lieu de `analyse .{0,10}mtriques` (le `.` remplacera le trait d'union supprimé).

### Piège: le routeur et le classifieur utilisent des normalizeText DIFFÉRENTES

C'est un piège TRÈS dangereux qui touche les **intercepteurs de sécurité** :

- **Canonical classifier** : `normalizeText` utilise `/[^a-z0-9\s]/g` → supprime `é` **complètement**
- **Router** (`operator-natural-action-router.js`) : `normalizeText` utilise `stripDiacritics()` puis `/[^a-z0-9]+/g` → convertit `é` **en `e`**

Donc `données` → canonical donne `donnes`, mais le routeur donne `donnees`.

Quand les intercepteurs de sécurité (`isSecretSeeking`, `isUnauthorizedActionRequest`) sont importés par les DEUX chemins, ils reçoivent des textes normalisés DIFFÉREMMENT.

**Correctif :** les patterns de sécurité doivent matcher les DEUX formes. Exemple :
```javascript
// Avant (ne marche que pour une forme):
/(?:donnes|historique)/i

// Après (marche pour les deux formes):
/(?:donn[e]?e?s?|historique)/i
```

## Ordre de priorité des patterns

Ordre validé empiriquement (CLOE_E2E_PATTERN_COVERAGE_IMPROVEMENT_V1) :

1. **Security interceptors** (isSecretSeeking, isUnauthorizedActionRequest) — FIRST
2. **Identity** — très spécifique `qui es[- ]?tu|who are you`
3. **Skill install** — très spécifique
4. **Capability inquiry** — `capacits?|que peux tu faire`
5. **Status** — `sant|statut|etat`
6. **Memory query** — `souviens|historique|prcdent`
7. **Diagnostic** — `bug|erreur|diagnostique|problme`
8. **System analysis** — `analyse les? (mtriques|donn|...)|expliqu.{0,60}architecture`
9. **Tracking** — `mtriques|performance|suivi|clics`
10. **Stack observability** — `stack|docker|container|...|runtime`
11. **Mutating** — `excute|supprime|modifie|dmarre|dploie|active|publie`
12. **Campaign action** — `lance campagne|campagne|launch campaign`
13. **Action preparation** — `prpare .*action|prpare .*(plan|campagne|crawl)`
14. **Governance question** — `gouvernance|scurit|rgles?|puis.je|faut.il`
15. **Business analysis** — `analyse|business|marché|seo`
16. **Morning routine** — `routine du matin`
17. **Crawl request** — `crawl|crawle|scrape`
18. **Unknown** (fallback)

### Pourquoi cet ordre

- `system_analysis` avant `tracking` : `mtriques` ne doit pas capturer `analyse les mtriques` avant que system_analysis ait eu sa chance
- `mutating` avant `campaign_action` : `excute` est plus spécifique que `campagne`
- `campaign_action` avant `action_preparation` : `lance campagne` est une exécution, pas une préparation
- `governance_question` APRÈS `mutating`/`campaign_action`/`action_preparation` : sinon `execute` dans governance_question capture les prompts de mutation

## Procédure de certification

1. **Importer les 4 composants canoniques** (canonical-intent-classifier, canonical-response-policy-resolver, retrieval-context-builder, provider-answer-composer)
2. **Pour chaque scénario** : appeler `classifyCanonicalIntent → resolveResponsePolicy → buildEvidenceDossier → composeAnswer`
3. **Vérifier la chaîne complète** : chaque composant doit être appelé avec le résultat du précédent
4. **Vérifier la gouvernance** : Class C/D doit avoir governance_required=true, Class A/B false
5. **Legacy bypass audit** : vérifier que tous les routers utilisent les composants canoniques
6. **Vérifier le payload provider** : `buildBrainAskResponse` doit injecter `canonical_evidence_dossier`
7. **Vérifier l'Answer Composer** : `composeAnswer()` doit être invoqué

## Réparation (Repair Cycle)

Politique : max 3 cycles. Chaque cycle doit :
1. Identifier la transition exacte qui échoue
2. Produire l'evidence de cause racine (tester la sortie de `normalizeText()`)
3. Appliquer le plus petit correctif borné
4. Ajouter un test de régression
5. Relancer le scénario affecté
6. Relancer la suite complète

**Ne pas faire :** redesigner l'architecture, nouveaux sous-systèmes, élargir les taxonomies sans evidence, affaiblir la gouvernance ou la sécurité.

## Outils réutilisables

Ces scripts vivent dans `lah-openclaw-mvp` :

| Script | Usage |
|--------|-------|
| `test/cloe-end-to-end-certification.mjs` | Certification 21 scénarios complète. Run depuis root. |
| `test/cloe-pattern-audit.mjs` | Audit de tous les patterns contre la sortie réelle de normalizeText(). Vérifie aussi les intercepteurs de sécurité. |

```bash
node test/cloe-end-to-end-certification.mjs
node test/cloe-pattern-audit.mjs
```

## Contexte: CLOE_E2E_PATTERN_COVERAGE_IMPROVEMENT_V1

Cette mission a certifié le pipeline CLOE avec 21/21 scénarios PASS après 3 cycles de réparation. Le verdict initial était PARTIAL (12/21) à cause des gaps de patterns français. Après correction des patterns pour la sortie réelle de normalizeText() et correction de l'ordre de priorité, le verdict est CERTIFIED.

SHA du commit : `5041505` dans `lah-openclaw-mvp`.
