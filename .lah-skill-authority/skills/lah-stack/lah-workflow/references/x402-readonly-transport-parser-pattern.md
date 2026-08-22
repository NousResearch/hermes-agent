# x402 Read-Only Transport Parser Pattern

## Quand l'utiliser

Lorsqu'une mission nécessite d'analyser des réponses HTTP 402 (Payment Required) pour extraire des signaux de paiement x402 V2, sans jamais exécuter de paiement, contacter un facilitateur, ou effectuer une requête réseau.

### Commits de référence

| Lot | Description | Commit |
|-----|-------------|--------|
| A | Protocol contracts, validation, normalization, fingerprints | `5daef0c` |
| B | Transport parser (lecture seule) | `36c8aa6` |
| B fix | Proof gaps fermés (fallback, duplicats, UTF-8, temps, caller isolation) | `c9604ac` |
| C | Purchase preparation (intent, policies, option assessment) | `a3eb8e0` |
| C fix | Intent binding, network allowlist, immutability hardening | `c13cc39` |
| D | Approval binding, local packet, execution lock | `97dc020` |

---

## Structure du parseur (Lot B)

```
SyntheticHttpResponse
  ↓ Status classification
  ↓ Header normalization (object or array-of-pairs)
  ↓ x402 signal detection (single, duplicate-identical, conflicting)
  ↓ Source-bound Base64 decoding (one encoding per source, no fallback)
  ↓ Fatal UTF-8 validation (TextDecoder fatal flag)
  ↓ Strict JSON parsing
  ↓ Lot A protocol validation
  ↓ Lot A normalization
  ↓ Immutable ReadOnlyParseResult (no caller-owned references)
```

---

## Structure du prep (Lot C)

```
ReadOnly402ParseResult (VALID_X402_V2 only)
  ↓ Validate Lot B classification
  ↓ Build preparation context (caller-supplied governance)
  ↓ Assess each normalized option against policies
  ↓ Evaluate buyer policies (seller, budget, expected-value, delivery)
  ↓ Handle Lot A option classification (conflict→AMBIGUOUS, duplicate→warning)
  ↓ Select option if exactly one eligible and policy allows
  ↓ Build approval requirement (no approval created)
  ↓ Build execution constraints (all false)
  ↓ Generate deterministic SHA-256 purchase intent identity
  ↓ Build GovernedPurchaseIntent (immutable, non-executable)
  ↓ Return PurchasePreparationResult
```

---

## Structure de l'approval prep (Lot D)

```
GovernedPurchaseIntent (from Lot C only)
  ↓ Validate Lot C intent (schema, purchaseIntentId, required sections)
  ↓ Validate approval preparation context (caller-supplied)
  ↓ Non-executable intent lock: ALL constraints must be false
  ↓ Check if approval is required (from Lot C)
  ↓ Reconcile approval requirements (Lot C cannot be weakened)
  ↓ Check expiry (must be after requestedAt, within max lifetime)
  ↓ Generate deterministic binding ID (Lot A canonical encoder)
  ↓ Build ApprovalBinding (no approval results, no approvalId)
  ↓ Generate deterministic packet ID (separate domain tag)
  ↓ Build ApprovalPreparationPacket (submissionState = NOT_SUBMITTED)
  ↓ Return ApprovalPreparationResult (immutable, non-executable)
```

---

## Structure de l'allowlist (Lot C)

Trois concepts séparés, pas de `startsWith('eip155:')` comme autorisation finale :

```
PROTOCOL_VALID_NETWORK     = syntactically valid CAIP-2 (/^[a-z0-9]+:/)
LOT_A_SUPPORTED_NETWORK    = protocol-wise supported by Lot A
LOT_C_LOCALLY_ALLOWED      = explicitly in allowedNetworks[]
```

Les allowlists sont passées via `policyContext.allowedNetworks`, `policyContext.allowedSchemes`, `policyContext.allowedAssets`. Chaque collection est dé-dupliquée et triée pour déterministe. Si absente, fallback deny-safe.

---

## Canonical binding

Tous les lots utilisent l'encodeur canonique de Lot A :

| Symbole | Usage |
|---------|-------|
| `encodeString(value)` | `s:<byteLength>:<NFC-value>` |
| `encodeUint(value)` | `u:<byteLength>:<value>` |
| `encodeAbsent()` | `a:0:` (intentional absence) |
| `encodeDomain(tag, fields)` | `<tag><f1><f2>...<fN>` |

Domain tags par lot :

| Lot | Tag | Usage |
|-----|-----|-------|
| A | `CLOE_X402_APPROVAL_BINDING_V1` | Approval binding digest |
| A | `CLOE_X402_LOGICAL_OPTION_IDENTITY_V1` | Option identity |
| A | `CLOE_X402_COMPLETE_OPTION_FINGERPRINT_V1` | Complete option |
| C | `CLOE_X402_GOVERNED_PURCHASE_INTENT_V1` | Purchase intent ID |
| D | `CLOE_X402_APPROVAL_PREPARATION_BINDING_V1` | Approval binding ID |
| D | `CLOE_X402_APPROVAL_PREPARATION_PACKET_V1` | Packet identity |

---

## Pitfalls

### 1. Lot A deepFreeze mute les objets — crash sur sous-objets pré-gelés

Lot A's deepFreeze fait `value[i] = deepFreeze(…)` et `value[key] = deepFreeze(…)` — mutation in-place qui crash si la cible est déjà gelée.

**V1 (simple) :** `if (Object.isFrozen(value)) return value;` — saute les gelés mais ne traverse pas leurs enfants.

**V2 (durci) :** Reconstruit par copie (`[...value]` ou `{...value}`) avant de traverser, garantissant que les enfants mutables ne sont pas cachés par un parent gelé.

**Règle :** V1 suffit quand on contrôle le cycle de gel. V2 nécessaire avec caller references ou chaînage de builders.

### 2. Caller references — reconstruire champ par champ

Ne JAMAIS passer `{ ...input }` directement dans deepFreeze. Reconstruire en ne conservant que les primitives. Tests : caller input non muté, non gelé, pas de références partagées.

### 3. Lot A cache les UNSUPPORTED sous INVALID

Inspector `validationResult.optionResults[i].result.status` individuellement.

### 4. Buffer.from('xxx', 'base64') ignore les caractères invalides

Valider l'alphabet avec regex avant décodage.

### 5. Source-bound encoding — zero fallback

Chaque source → un encodage. Header `PAYMENT-REQUIRED` → Base64 standard uniquement.

### 6. Test data : Base64URL = Base64 pour l'ASCII simple

Pour produire des encodages différents : `Buffer.from([0xFB])` → base64 `+w==`, base64url `-w`.

### 7. Repeated headers nécessitent array-of-pairs

### 8. Duplicate classification : 1=OK, N identiques=warning, N différents=AMBIGUOUS

### 9. Fatal UTF-8 : TextDecoder('utf-8', { fatal: true })

### 10. Deterministic time : context.now ou preparationContext.deterministicTime

### 11. Body-only x402 = désactivé (HTTP_402_NON_X402)

### 12. Network policy : trois concepts séparés

`PROTOCOL_VALID` (CAIP-2 syntaxique) ≠ `LOT_A_SUPPORTED` (protocole) ≠ `LOT_C_LOCALLY_ALLOWED` (allowlist explicite). Pas de `eip155:`, pas de prefix, pas de namespace comme règle d'autorisation finale.

### 13. Intent binding : encodeur canonique Lot A

Utiliser `encodeString`/`encodeUint`/`encodeDomain` avec domaine `CLOE_X402_GOVERNED_PURCHASE_INTENT_V1`. Pas de `JSON.stringify` ou BigInt.

### 14. Approval preparation : non-executable intent lock

Toutes les contraintes d'exécution de Lot C doivent être `false`. Sinon `APPROVAL_PREPARATION_BLOCKED`.

### 15. Approval reconciliation : Lot D ne peut pas affaiblir Lot C

Pas de downgrade d'approvalRequired, secondConsent, ou approverRole.

### 16. Approval binding : purchaseIntentId → bindingId

Pas de circularité. `purchaseIntentId` est l'identité autoritaire du commerce.

### 17. Packet : NOT_SUBMITTED hard-coded

Aucun `approvalId`, `approved`, ou `submitted` dans Lot D.

### 18. Test data : purchaseIntentId hex 64 chars

### 19. Docker container file extraction

Quand l'agent Hermes tourne dans un conteneur Docker, les fichiers dans `/tmp/` sont dans le conteneur, pas sur l'hôte. Utiliser `docker cp hermes:/tmp/file /home/deploy/`.

---

## Post-commit

```bash
git status --porcelain=v1
node --test --test-concurrency=1 test/x402/*.js test/x402/transport/*.js test/x402/preparation/*.js test/x402/approval/*.js
```
