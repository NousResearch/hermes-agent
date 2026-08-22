# x402 Pure Offline Builder Pattern

## Quand l'utiliser

Quand tu dois transformer un objet certifié (Lot D `ApprovalPreparationPacket`, Lot C intent, etc.) en une requête de soumission locale, déterministe, immuable, **sans réseau, sans identité de transport, sans accréditation**. Ce pattern crée le "Part A" d'un système à deux parties (Part A = builder offline, Part B = adaptateur transport injectable).

Établi pendant `CLOE_X402_LOT_E1_LOCAL_LAHB_SUBMISSION_REQUEST_BUILDER_V1`.

---

## Architecture

```
ApprovalPreparationPacket (Lot D, certifié)
    + SubmissionPreparationContext (caller-supplied, strict)
    │
    ▼
buildX402LahbSubmissionRequest({ packet, context })
    │
    ├── 1. Validate input structure
    ├── 2. Validate caller-supplied context
    ├── 3. Validate packet schema + identities
    ├── 4. Enforce NOT_SUBMITTED
    ├── 5. Reject forbidden authority fields
    ├── 6. Build approval terms from packet
    ├── 7. Build subject summary (seller digest, not raw)
    ├── 8. Build evidence digest (sorted refs → stable)
    ├── 9. Derive approvalTermsDigest
    ├── 10. Derive submissionRequestId (includes preparedAt)
    ├── 11. Derive idempotencyKey (NO preparedAt, NO attemptId)
    ├── 12. Build request object (plain, not frozen)
    └── 13. Return deeply frozen SubmissionRequestBuildResult
```

---

## Identity Doctrine

Deux identités distinctes, deux domaines Canonical encoder séparés.

### submissionRequestId

```
Domain: CLOE_X402_LAHB_SUBMISSION_REQUEST_V1
Bound: schemaVersion, packetId, bindingId, purchaseIntentId,
       submissionPolicyVersion, targetAuthorityId,
       approvalTermsDigest, subjectDigest, evidenceDigest,
       auditReference, preparedAt, transportConstraintsDigest,
       submissionState (NOT_SUBMITTED)
```

- INCLUT `preparedAt` → une préparation différente produit un ID différent
- Utile pour le suivi d'audit local

### idempotencyKey

```
Domain: CLOE_X402_LAHB_IDEMPOTENCY_KEY_V1
Bound: schemaVersion, packetId, bindingId, purchaseIntentId,
       submissionPolicyVersion, targetAuthorityId,
       approvalTermsDigest, subjectDigest, evidenceDigest,
       auditReference, transportConstraintsDigest
```

- N'INCLUT PAS `preparedAt` → même opération logique = même clé
- N'INCLUT PAS `attemptId` → cela appartient à Lot E2/E3
- N'INCLUT PAS `wall-clock` → stable entre sessions

---

## Freeze-While-Building Pattern

Le `deepFreeze` de Lot A ne peut PAS gérer des sous-objets déjà frozen (il fait `value[key] = deepFreeze(value[key])` qui crashe).

### Règle

1. Construis TOUS les sous-objets comme des plain objects (pas d'`Object.freeze` intermédiaire, pas de `Object.freeze([])`)
2. Copie les tableaux callers avec `[...arr]` (pas de `Object.freeze([...arr])`)
3. Utilise `{ ...DEFAULT_CONSTRAINTS }` pour copier des constantes frozen
4. Appelle `deepFreeze` UNE SEULE FOIS au niveau le plus haut

### Exemple

```javascript
// FAUX — pre-freeze intermédiaire → crash deepFreeze
const terms = Object.freeze({ approvalRequired: true, ... });
const request = { approvalTerms: terms, ... };
return deepFreeze(request); // CRASH: ne peut pas reassigner approvalTerms

// CORRECT — plain objects jusqu'au deepFreeze final
const terms = { approvalRequired: true, ... };
const request = { approvalTerms: terms, ... };
return deepFreeze(request); // OK: tout est mutable avant
```

---

## Data Minimisation

Classifie chaque champ du packet source:

| Classification | Comportement |
|----------------|-------------|
| `FULL_VALUE` | Transmis tel quel |
| `CANONICAL_DIGEST_ONLY` | SHA-256 du champ, pas la valeur brute |
| `REDACTED_SUMMARY` | Version réduite |
| `NOT_TRANSMITTED` | Champs de preuve locale, pas envoyés |

### Exemples (Lot D → LAHB)

| Lot D field | Treatment |
|-------------|-----------|
| `packetId` | FULL_VALUE |
| `binding.bindingId` | FULL_VALUE |
| `binding.purchaseIntentId` | FULL_VALUE |
| `sellerIdentity.sellerId` | CANONICAL_DIGEST_ONLY |
| `submissionState` | NOT_TRANSMITTED |
| `executionConstraints` | NOT_TRANSMITTED |

---

## Test Matrix Categories

Un builder offline doit avoir des tests couvrant:

| Catégorie | Exemples |
|-----------|----------|
| Input validation | null, missing packet, wrong schema, array input, unknown fields |
| Packet authority | NOT_SUBMITTED accepted, SUBMITTED/APPROVED rejected |
| Forbidden fields | approvalId, decision, signature, receipt, attemptId, authorizationToken |
| Identity stability | same input → same IDs, different packetId → different IDs |
| Idempotency time-independence | different preparedAt → SAME idempotencyKey |
| Evidence order independence | ['a','b','c'] vs ['c','a','b'] → same digests |
| Immutability | caller not mutated, request frozen, sub-objects frozen, caller arrays not shared |
| Pre-frozen input | parent Object.freeze with mutable child → handled safely |
| Safety | no baseUrl, no apiKey, no fetch, no credentials, no attemptId |
| Domain separation | submissionRequestId ≠ idempotencyKey |

---

## Pitfalls

| Pitfall | Symptom | Fix |
|---------|---------|-----|
| Pre-frozen sub-objets dans deepFreeze | `Cannot assign to read only property 'X' of object` | Construire en plain objects, freeze UNE SEULE fois au top level |
| buildSubjectSummary frozen avant d'arriver au builder | Même crash sur champ `sellerId` | Retourner un plain object, laisser le builder final freezer |
| Tableaux frozen (Object.freeze([])) | `Cannot assign to read only property '0'` | Utiliser `[...arr]` pas `Object.freeze([...arr])` |
| Constante frozen partagée (DEFAULT_TRANSPORT_CONSTRAINTS) | Crash sur le spread du builder | Copier avec `{ ...DEFAULT }` |
| buildApprovalTerms inclut des champs non stables | L'identité change entre runs | Fixer `expiresAt` et `reasonCodes` dans le test, ne pas dépendre de random |
| L'audit reference change entre packets | Mêmes IDs logiques → clés différentes | Forcer `auditReference` dans les tests d'identité |
