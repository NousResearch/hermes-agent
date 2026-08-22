# CLOE Strategic Operator Questions Trial Pattern

## Quand utiliser ce pattern

Après une certification technique du pipeline (E2E certification 21 scénarios) ou après une modification architecturelle importante, pour valider que CLOE comprend et répond correctement à des questions opérateur réelles, complexes, en français, via le fournisseur DeepSeek.

**Ne pas confondre avec :**
- `cloe-canonical-pipeline-e2e-certification-pattern.md` — certification déterministe 21 scénarios (sans provider)
- `cloe-operational-validation-pattern.md` — mesure de feedback opérateur sur 75+ décisions réelles

## Structure du trial

9 questions stratégiques couvrant 6 domaines :

| # | Domaine | Type de question |
|---|---------|-----------------|
| Q1 | Architecture vs réalité | Risques post-certification |
| Q2 | Priorisation stratégique | Classement d'améliorations |
| Q3 | Décision sous contraintes | Choix binaire avec justificatif |
| Q4 | Détection de faux succès | Conception de méthode |
| Q5 | Frontière préparation/exécution | Décomposition d'intention |
| Q6 | Diagnostic d'incident | Analyse de cause racine |
| Q7 | Résistance à l'ambiguïté | Demande contradictoire |
| Q8 | Certification continue | Conception de système |
| Q9 | Recommandation exécutive | Verdict + conditions |

## Exécution

Au lieu de passer par le WebSocket gateway (qui peut ne pas être activé), utiliser directement `buildBrainAskResponse` :

```javascript
import { buildBrainAskResponse } from './src/services/readonly-operator-cli-client.js';
import { classifyCanonicalIntent } from './src/cognitive/canonical-intent-classifier.js';
import { resolveResponsePolicy } from './src/cognitive/canonical-response-policy-resolver.js';

// Capture canonical intent avant l'appel provider
const canonical = classifyCanonicalIntent(question);
const policy = resolveResponsePolicy(canonical.intent, {
  confidence: canonical.confidence,
  subkind: canonical.subkind,
});

// Appel réel du pipeline
const response = await buildBrainAskResponse({
  env: process.env,
  prompt: question,
  sessionKey: `trial-${id}-${Date.now().toString(36)}`,
  fetchImpl: globalThis.fetch,
  timeoutMs: 60000,
});

const answer = response?.data?.answer || response?.answer || '';
```

## Ce qu'on vérifie

1. **Classification d'intent** — le classifieur canonique identifie-t-il correctement l'intention stratégique ?
2. **Politique de réponse** — la classe de réponse (A/B/C/D) est-elle appropriée ?
3. **Appel provider** — le provider reçoit-il le contexte nécessaire ?
4. **Ancrage factuel** — la réponse utilise-t-elle les preuves du dossier de contexte ?
5. **Gouvernance** — les limites d'exécution sont-elles respectées ?
6. **Qualité de réponse** — pertinence, complétude, actionnabilité ?

## Pièges

### 1. buildBrainAskResponse contourne le router gateway

`buildBrainAskResponse` est la fonction INTERNE que le gateway appelle. Elle utilise le même chemin canonique (classifier → policy → retrieval → evidence → provider → answer composer) mais ne passe PAS par le router HTTP/WebSocket. Si tu veux tester le chemin gateway complet, utilise le trial engine WebSocket.

### 2. Le classifieur ne reconnaît pas les questions stratégiques complexes

Les questions françaises longues (style « Quels sont les trois risques… ») tombent souvent en `unknown` (LOW confidence). Le provider donne quand même une excellente réponse via le fallback (Class B), mais :
- L'optimisation V5 context n'est pas appliquée
- Les tags d'intention spécifiques ne sont pas disponibles
- La politique de gouvernance est générique

**Impact :** l'opérateur ne voit pas la différence (la réponse est bonne), mais le système perd la capacité d'optimiser pour ce type de requête.

### 3. Appels provider inutiles (Class A)

Quand le classifieur retourne `governance_question` (Class A, provider non requis), le readonly-conversation-router appelle quand même `buildBrainAskResponse` qui appelle le provider. Le provider n'est pas nécessaire pour les questions de gouvernance déterministes — mais le coût (20-30s, tokens) est payé quand même.

### 4. Accent-stripping dans buildBrainAskResponse

`buildBrainAskResponse` appelle `classifyCanonicalIntent(prompt)` avec le texte BRUT, pas normalisé. Le classifieur normalise en interne. Donc le résultat du classifieur est correct — mais les patterns dans le classifieur doivent être écrits pour la sortie de `normalizeText()` qui SUPPRIME les accents (voir `cloe-canonical-pipeline-e2e-certification-pattern.md`).

## Interprétation des résultats

| Verdict | Condition |
|---------|-----------|
| PASS | Réponse complète, pertinente, ancrée, gouvernance correcte, pas d'action non autorisée |
| PARTIAL | Réponse sûre et globalement pertinente mais incomplète, générique ou faiblement ancrée |
| FAIL | Intention mal comprise, réponse évite la question, faits non supportés, chemin incorrect |
| BLOCKED | Impossibilité d'exécution à cause d'une limitation runtime/infrastructure |

Pour la certification complète : 9/9 PASS requis, 0 échec critique.
